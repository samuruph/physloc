"""Progress, ETA and a stage profile for a generation run.

A release run is days of wall clock. Two things live here.

`Progress` is the live view: a tqdm bar when a terminal is attached, plain lines
when it is not, a status line every few minutes, a plain-text log file, and an
ETA that is WEIGHTED rather than naive.

`Profile` is the post-mortem -- where the seconds went, by stage, printed as a
table when the run ends.

--------------------------------------------------------------------------
WHY PROGRESS COUNTS RENDERS, NOT JOBS
--------------------------------------------------------------------------
A release job is one scenario x level x variant, and renders its valid twin
plus every family at every severity -- about forty renders, hours of wall clock
even on its own worker. Counting jobs left the bar at 0 and the ETA at "?" for
the first hours of a run. The worker announces every render as it finishes, so
the bar moves every few seconds on a busy machine and the ETA exists after the
first few renders.

--------------------------------------------------------------------------
WHY THE ETA IS WEIGHTED
--------------------------------------------------------------------------
The obvious ETA is `elapsed / done * remaining`, and on a ladder run it is
wrong by more than a factor of two: an L2 render costs ~2.6x an L0 one because
its HDRI dome encloses the scene. So every job carries a PREDICTED cost -- the
same `SECONDS_PER_CLIP` the `taxonomy` subcommand prices a run from -- shared
evenly across its renders, and the ETA is

    remaining_predicted_work / (completed_predicted_work / elapsed)

The ratio in the denominator is the correction between what the price model
thinks and what this machine delivers, and it absorbs everything the model
leaves out. A render that will not happen -- a family that declines its scene,
a job that dies -- leaves the remaining work instead of sitting in it forever.
"""
from __future__ import annotations

import os
import sys
import threading
import time
from collections import defaultdict

#: How often a running `generate` prints a status line.
HEARTBEAT_SECONDS = 300


def _fmt(seconds: float) -> str:
    """A duration a human can act on: 4d 3h, 3h 12m, 12m 04s, 4.2s."""
    if seconds != seconds or seconds in (float("inf"), float("-inf")):
        return "?"
    seconds = max(0.0, float(seconds))
    if seconds < 10:
        return "%.1fs" % seconds
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    d, h = divmod(h, 24)
    if d:
        return "%dd %02dh" % (d, h)
    if h:
        return "%dh %02dm" % (h, m)
    return "%dm %02ds" % (m, s)


class Profile:
    """Wall-clock totals per named stage, safe to add to from many threads.

    Stages OVERLAP on purpose and the report says so. `worker` is the whole
    container round trip and `render` is the part of it Blender reported, so
    `worker - render` is build plus simulate plus the npz write -- which is the
    number that tells you whether to attack the renderer or everything around
    it.

    The totals are summed across workers, so on a parallel run they add up to
    more than the elapsed time. That is the point: dividing the two gives the
    occupancy actually achieved, which is how you tell an oversubscribed run
    from a well-fed one.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self.totals = defaultdict(float)
        self.counts = defaultdict(int)
        self.t0 = time.perf_counter()

    def add(self, stage: str, seconds: float, n: int = 1) -> None:
        with self._lock:
            self.totals[stage] += float(seconds)
            self.counts[stage] += n

    def timer(self, stage: str):
        """`with prof.timer("annotate"): ...`"""
        return _Timer(self, stage)

    def report(self, out=sys.stdout, workers: int = 1) -> None:
        elapsed = time.perf_counter() - self.t0
        if not self.totals:
            return
        print("\n-- where the time went (totals are summed across workers,"
              "\n   so `worker` and `render` overlap by design)", file=out)
        width = max(len(k) for k in self.totals)
        # Descending, because the first line is the only one most readers need.
        for stage, total in sorted(self.totals.items(), key=lambda kv: -kv[1]):
            n = self.counts[stage]
            print("   %-*s  %9s  over %5d  = %8s each  (%5.1f%% of wall x%d)"
                  % (width, stage, _fmt(total), n,
                     _fmt(total / n) if n else "-",
                     100.0 * total / (elapsed * max(1, workers)), workers),
                  file=out)
        busy = self.totals.get("worker", 0.0) + self.totals.get("annotate", 0.0)
        print("   %-*s  %9s" % (width, "wall clock", _fmt(elapsed)), file=out)
        if workers > 1 and elapsed > 0:
            # OCCUPANCY IS THE PARALLELISM QUESTION. If it is far under the
            # worker count, jobs are queueing -- on memory, or behind a
            # straggler -- rather than on cores, and more workers buy nothing.
            print("   %-*s  %.2f of %d workers busy on average"
                  % (width, "occupancy", busy / elapsed, workers), file=out)


class _Timer:
    def __init__(self, prof, stage):
        self.prof, self.stage = prof, stage

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.prof.add(self.stage, time.perf_counter() - self.t0)
        return False


class Progress:
    """A live bar with a weighted ETA, degrading to plain lines without a tty.

    `weights` is one predicted cost per job, in job order; `None` counts every
    job the same. `renders`, also in job order, is how many renders each job
    makes: given it, the bar counts renders (`render_done`) rather than jobs.
    `log_path` appends every job line and status line, timestamped, to a file
    that `tail -f` can follow whether or not a terminal is attached.

    Every method is thread-safe: jobs run in a pool and the render announcements
    arrive from each worker's output reader.
    """

    def __init__(self, total: int, weights=None, desc="generate",
                 stream=sys.stdout, use_bar=None, renders=None, log_path=None):
        self.total = int(total)
        self.weights = list(weights) if weights else None
        self.total_weight = (float(sum(self.weights)) if self.weights
                             else float(total))
        self.done_weight = 0.0
        self.n = 0
        self.stream = stream
        self.t0 = time.perf_counter()
        self._lock = threading.RLock()
        self._bar = None

        self.renders = list(renders) if renders else None
        self.renders_total = int(sum(self.renders)) if self.renders else 0
        self.renders_done = 0
        self._seen = defaultdict(int)       # renders done or dropped, per job
        self.running = set()
        self.waiting = set()                # queued, e.g. for memory

        self._stop = threading.Event()
        self._heartbeat = None
        self._log = None
        if log_path:
            os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
            self._log = open(log_path, "a", buffering=1)

        # A tqdm bar redraws in place with \r, which turns a log file into one
        # enormous line. Only draw it when someone is actually watching.
        if use_bar is None:
            use_bar = bool(getattr(stream, "isatty", lambda: False)())
        if use_bar:
            try:
                from tqdm import tqdm
                self._bar = tqdm(total=self._bar_total(), desc=desc,
                                 unit="render" if self.renders else "job",
                                 dynamic_ncols=True, file=stream,
                                 bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} "
                                            "[{elapsed}<{postfix}]")
                self._bar.set_postfix_str("eta ?")
            except ImportError:
                self._bar = None
        self._log_line("started: %d job(s)%s" % (
            self.total,
            ", %d render(s)" % self.renders_total if self.renders else ""))

    # ------------------------------------------------------------ internals
    def _bar_total(self) -> int:
        return self.renders_total if self.renders is not None else self.total

    def _weight_for(self, i: int) -> float:
        """The predicted cost of job `i`, with a sane answer past the end.

        Retries are appended after the bar is built. Falling back to 1.0 would
        be catastrophic rather than merely approximate -- the declared weights
        are in HUNDREDS of seconds, so a 1.0 reads as a job that finished
        instantly and drags the ETA toward zero exactly when a run is adding
        work. The mean of what we do know is the honest guess.
        """
        if self.weights and i < len(self.weights):
            return float(self.weights[i])
        if self.weights:
            return float(sum(self.weights)) / len(self.weights)
        return 1.0

    def _renders_for(self, i: int) -> int:
        if self.renders and i < len(self.renders):
            return max(1, int(self.renders[i]))
        if self.renders:
            return max(1, round(sum(self.renders) / len(self.renders)))
        return 1

    def _log_line(self, text: str) -> None:
        if self._log is not None:
            self._log.write("%s  %s\n" % (time.strftime("%Y-%m-%d %H:%M:%S"),
                                          text.strip()))

    def _say(self, line: str) -> None:
        """A line for whoever is watching: above the bar, or on its own."""
        if self._bar is not None:
            self._bar.write(line)
        else:
            print(line, file=self.stream, flush=True)
        self._log_line(line)

    def _refresh_bar(self) -> None:
        if self._bar is not None:
            self._bar.total = self._bar_total()
            self._bar.set_postfix_str("eta %s" % _fmt(self.eta()), refresh=False)
            self._bar.refresh()

    # ----------------------------------------------------------- the API
    def bump(self, extra: int, weights=None, renders=None) -> None:
        """Widen the bar mid-run, for work discovered after it was built.

        Pass the new jobs' `weights` and `renders` when they are known (retries
        are); without them each is priced at the mean of the known jobs.
        """
        if extra <= 0:
            return
        with self._lock:
            if weights is not None and self.weights is not None:
                self.weights.extend(float(w) for w in weights)
                self.total_weight += float(sum(weights))
            else:
                self.total_weight += extra * self._weight_for(len(self.weights or []))
            if self.renders is not None:
                new = ([int(r) for r in renders] if renders is not None
                       else [self._renders_for(len(self.renders))] * extra)
                self.renders.extend(new)
                self.renders_total += sum(new)
            self.total += int(extra)
            self._refresh_bar()

    def job_waiting(self, index: int) -> None:
        """Job `index` is queued -- for memory, or for a core slice."""
        with self._lock:
            self.waiting.add(index)

    def running_count(self) -> int:
        with self._lock:
            return len(self.running)

    def job_started(self, index: int) -> None:
        with self._lock:
            self.waiting.discard(index)
            self.running.add(index)

    def render_done(self, index: int) -> None:
        """One render of job `index` finished: its share of the job's cost is done."""
        with self._lock:
            self.done_weight += self._weight_for(index) / self._renders_for(index)
            self.renders_done += 1
            self._seen[index] += 1
            if self._bar is not None:
                self._bar.set_postfix_str("eta %s" % _fmt(self.eta()), refresh=False)
                self._bar.update(1)

    def render_dropped(self, index: int) -> None:
        """A planned render of job `index` will not happen: it leaves the total."""
        with self._lock:
            self._drop(index, 1)
            self._refresh_bar()

    def _drop(self, index: int, count: int) -> None:
        if count <= 0 or self.renders is None:
            return
        self.total_weight = max(
            self.done_weight,
            self.total_weight - count * self._weight_for(index) / self._renders_for(index))
        self.renders_total = max(self.renders_done, self.renders_total - count)
        self._seen[index] += count

    def eta(self) -> float:
        """Seconds remaining, by predicted work rather than by count."""
        elapsed = time.perf_counter() - self.t0
        if self.done_weight <= 0 or elapsed <= 0:
            return float("nan")
        rate = self.done_weight / elapsed          # predicted-seconds per second
        return max(0.0, self.total_weight - self.done_weight) / rate

    def status_line(self) -> str:
        with self._lock:
            parts = []
            if self.renders is not None:
                parts.append("renders %d/%d" % (self.renders_done, self.renders_total))
            jobs = "jobs %d/%d done, %d running" % (self.n, self.total,
                                                    len(self.running))
            if self.waiting:
                jobs += ", %d waiting for memory" % len(self.waiting)
            parts.append(jobs)
            parts.append("elapsed %s" % _fmt(time.perf_counter() - self.t0))
            parts.append("eta %s" % _fmt(self.eta()))
            return " | ".join(parts)

    def update(self, label: str, weight: float = None, ok: bool = True,
               index: int = None) -> str:
        """Record one finished job. Returns the line it printed, or "".

        `index` is the job's position in the job list. Jobs finish out of order
        in a pool, so a job's weight must come from its index -- counting
        completions would price a cheap job at an expensive one's cost.
        """
        with self._lock:
            self.n += 1
            if index is None:
                index = self.n - 1
            self.running.discard(index)
            self.waiting.discard(index)
            if self.renders is not None:
                # Renders the worker never reported -- it died, or declined
                # without saying -- are not coming.
                self._drop(index, self._renders_for(index) - self._seen[index])
            else:
                self.done_weight += float(self._weight_for(index)
                                          if weight is None else weight)
            eta = self.eta()
            elapsed = time.perf_counter() - self.t0
            line = ("  [%d/%d] %-46s %s  elapsed %s  eta %s"
                    % (self.n, self.total, label, "ok" if ok else "FAILED",
                       _fmt(elapsed), _fmt(eta)))
            if self.renders is not None:
                line += "  renders %d/%d" % (self.renders_done, self.renders_total)
            self._log_line(line)
            if self._bar is not None:
                if self.renders is None:
                    self._bar.update(1)
                self._refresh_bar()
                return ""
            print(line, file=self.stream, flush=True)
            return line

    def skip(self, label: str, weight: float = None, index: int = None) -> None:
        """Record a job that was RESUMED rather than run.

        A skipped job consumed no wall clock, so it must not enter the observed
        rate -- but its predicted cost must leave the REMAINING work, or a
        resume that skips the first thousand jobs would quote an ETA for work
        it is never going to do.
        """
        with self._lock:
            if index is None:
                index = self.n
            if weight is None:
                weight = self._weight_for(index)
            self.n += 1
            self.total_weight = max(0.0, self.total_weight - float(weight))
            if self.renders is not None:
                self.renders_total -= self._renders_for(index)
            line = "  [%d/%d] %-46s resumed" % (self.n, self.total, label)
            self._log_line(line)
            if self._bar is not None:
                if self.renders is None:
                    self._bar.update(1)
                self._refresh_bar()
            else:
                print(line, file=self.stream, flush=True)

    def start_heartbeat(self, interval: float = HEARTBEAT_SECONDS) -> None:
        """Print `status_line()` every `interval` seconds until `close()`."""
        if self._heartbeat is not None:
            return

        def beat():
            while not self._stop.wait(interval):
                self._say("  status: " + self.status_line())

        self._heartbeat = threading.Thread(target=beat, name="progress-heartbeat",
                                           daemon=True)
        self._heartbeat.start()

    def close(self) -> None:
        self._stop.set()
        if self._heartbeat is not None:
            self._heartbeat.join(timeout=2)
            self._heartbeat = None
        if self._bar is not None:
            self._bar.close()
            self._bar = None
        if self._log is not None:
            self._log_line("finished: " + self.status_line())
            self._log.close()
            self._log = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def job_weight(level: str, tier_name: str, seconds_per_clip: dict,
               complexity: dict, n_families: int = 1, n_bins: int = 1) -> float:
    """Predicted seconds for one (level, scenario, variant) job.

    One job renders one valid twin plus `n_families * n_bins` invalid ones, all
    at the level's background rate -- which is the whole reason a ladder needs
    weighting: `hdri` is ~2.6x `solid` at release geometry.

    Falls back to a flat 60 s for a tier the price table has never been
    measured at, so an unmeasured combination gives a poor ETA rather than a
    crash.
    """
    cx = complexity.get(level)
    bg = cx.background if cx is not None else "solid"
    rate = seconds_per_clip.get((tier_name, bg), 60.0)
    return rate * (1 + max(0, int(n_families)) * max(1, int(n_bins)))
