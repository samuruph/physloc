"""Progress, ETA and a stage profile for a generation run.

A release run is hours to days of wall clock and, until now, its only sign of
life was one `[n/total]` line per finished job. That is enough to tell you a run
is alive and not enough to tell you anything else: not when it will finish, and
not which of build / simulate / render / annotate / overlay is eating the time.

Two things live here.

`Progress` is the live view -- a tqdm bar when a terminal is attached, plain
lines when the output is a log file, and an ETA that is WEIGHTED rather than
naive.

`Profile` is the post-mortem -- where the seconds went, by stage, printed as a
table when the run ends.

--------------------------------------------------------------------------
WHY THE ETA IS WEIGHTED
--------------------------------------------------------------------------
The obvious ETA is `elapsed / done * remaining`, and on a ladder run it is
wrong by more than a factor of two. `--complexity all` emits its jobs level by
level -- every L0 job, then every L1, then L2, then L3 -- and an L2 clip costs
~2.6x an L0 one because its HDRI dome encloses the scene. So a naive rate
measured across the L0 block predicts the L2 block at L0 prices and promises an
ending it misses by hours, then walks the estimate back up while you watch.

Instead every job carries a PREDICTED cost -- the same `SECONDS_PER_CLIP` the
`taxonomy` subcommand prices a run from -- and the ETA is

    remaining_predicted_work / (completed_predicted_work / elapsed)

The ratio in the denominator is the correction factor between what the price
model thinks a job costs and what this box is actually delivering, and it
absorbs everything the model leaves out: annotation, overlays, a busy machine,
a worker count the model was not measured at. The estimate is therefore right
in PROPORTION from the first few jobs, rather than only becoming right once the
run is nearly over.
"""
from __future__ import annotations

import sys
import threading
import time
from collections import defaultdict


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
    it. Nesting them rather than making them disjoint means each one is a thing
    you can measure independently, instead of a subtraction you have to
    remember to perform.

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
            # OCCUPANCY IS THE PARALLELISM QUESTION. `--workers 8` on 8 vCPU
            # asks each render to share a core with another; if occupancy is
            # far under the worker count, the workers are queueing on the
            # machine rather than on work, and raising it again buys nothing.
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
    """A live bar with a weighted ETA, degrading to plain lines in a log.

    `weights` is one predicted cost per job, in the same order as the job list.
    Pass `None` and every job counts the same, which is right for a single-level
    run and wrong for a ladder -- see the module docstring.
    """

    def __init__(self, total: int, weights=None, desc="generate",
                 stream=sys.stdout, use_bar=None):
        self.total = int(total)
        self.weights = list(weights) if weights else None
        self.total_weight = float(sum(self.weights)) if self.weights else float(total)
        self.done_weight = 0.0
        self.n = 0
        self.stream = stream
        self.t0 = time.perf_counter()
        self._lock = threading.Lock()
        self._bar = None
        # A tqdm bar redraws in place with \r, which turns a log file into one
        # enormous line. Only draw it when someone is actually watching.
        if use_bar is None:
            use_bar = bool(getattr(stream, "isatty", lambda: False)())
        if use_bar:
            try:
                from tqdm import tqdm
                self._bar = tqdm(total=self.total, desc=desc, unit="job",
                                 dynamic_ncols=True, file=stream,
                                 bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} "
                                            "[{elapsed}<{postfix}]")
            except ImportError:
                self._bar = None

    def _weight_for(self, i: int) -> float:
        """The predicted cost of job `i`, with a sane answer past the end.

        Retries are appended after the bar is built, so they have no declared
        weight. Falling back to 1.0 would be catastrophic rather than merely
        approximate -- the declared weights are in HUNDREDS of seconds, so a
        1.0 reads as a job that finished instantly and drags the ETA toward
        zero exactly when a run is adding work. The mean of what we do know is
        the honest guess.
        """
        if self.weights and i < len(self.weights):
            return float(self.weights[i])
        if self.weights:
            return float(sum(self.weights)) / len(self.weights)
        return 1.0

    def bump(self, extra: int) -> None:
        """Widen the bar mid-run, for work discovered after it was built."""
        if extra <= 0:
            return
        with self._lock:
            self.total += int(extra)
            self.total_weight += extra * self._weight_for(len(self.weights or []))
            if self._bar is not None:
                self._bar.total = self.total
                self._bar.refresh()

    def eta(self) -> float:
        """Seconds remaining, by predicted work rather than by job count."""
        elapsed = time.perf_counter() - self.t0
        if self.done_weight <= 0 or elapsed <= 0:
            return float("nan")
        rate = self.done_weight / elapsed          # predicted-seconds per second
        return (self.total_weight - self.done_weight) / rate

    def update(self, label: str, weight: float = None, ok: bool = True) -> str:
        """Record one finished job. Returns the line it printed, or "".

        Thread-safe: the parallel path calls this from a ThreadPoolExecutor.
        """
        with self._lock:
            self.n += 1
            if weight is None:
                weight = self._weight_for(self.n - 1)
            self.done_weight += float(weight)
            eta = self.eta()
            elapsed = time.perf_counter() - self.t0
            if self._bar is not None:
                self._bar.set_postfix_str("eta %s | %s" % (_fmt(eta), label),
                                          refresh=False)
                self._bar.update(1)
                return ""
            # The plain form carries the same three facts as the bar, because
            # a run watched through `tail -f` on a log is the normal way a long
            # job gets watched.
            line = ("  [%d/%d] %-46s %s  elapsed %s  eta %s"
                    % (self.n, self.total, label,
                       "ok" if ok else "FAILED", _fmt(elapsed), _fmt(eta)))
            print(line, file=self.stream, flush=True)
            return line

    def skip(self, label: str, weight: float = None) -> None:
        """Record a job that was RESUMED rather than run.

        Not the same as a job that finished fast. A skipped job consumed no
        wall clock, so it must not enter the observed rate -- but its predicted
        cost must leave the REMAINING work, or a resume that skips the first
        thousand jobs would spend the rest of the run quoting an ETA for work
        it is never going to do. So the weight comes off the total instead of
        going onto the done pile.
        """
        with self._lock:
            if weight is None:
                weight = self._weight_for(self.n)
            self.n += 1
            self.total_weight = max(0.0, self.total_weight - float(weight))
            if self._bar is not None:
                self._bar.set_postfix_str("eta %s | %s" % (_fmt(self.eta()),
                                                           label),
                                          refresh=False)
                self._bar.update(1)
            else:
                print("  [%d/%d] %-46s resumed" % (self.n, self.total, label),
                      file=self.stream, flush=True)

    def close(self) -> None:
        if self._bar is not None:
            self._bar.close()
            self._bar = None

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
