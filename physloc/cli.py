"""physloc command line.

    conda activate physloc
    python -m physloc.cli taxonomy
    python -m physloc.cli generate --debug -n 2
    python -m physloc.cli annotate out/work/drop/0173
    python -m physloc.cli overlay out/release/samples/.../invalid_solidity_a
    python -m physloc.cli validate out/release

`generate` is the end-to-end path: it shells out to docker/kubric.sh for the
simulate+render half (container) and then runs annotation and overlays here
(host). The two halves meet at the trajectory seam.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import threading
import time
from typing import List, Optional

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------- taxonomy
def cmd_taxonomy(a) -> int:
    from .taxonomy import (DOMAINS, FAMILIES, MEDIA, SCENARIOS,
                           SEVERITY_BINS, build_cells, validate_taxonomy)
    validate_taxonomy()
    print("MEDIA (%d)  -- level 0, the LikePhys-style macro-category" % len(MEDIA))
    for m, why in MEDIA.items():
        staged = sorted(n for n, v in SCENARIOS.items() if v.physics_medium == m)
        print("  %-11s %-58s %s" % (m, why, ", ".join(staged) or "(none staged)"))
    print()
    print("DOMAINS (%d)" % len(DOMAINS))
    for d, q in DOMAINS.items():
        fams = [f for f, v in FAMILIES.items() if v.domain == d]
        print("  %-12s %s" % (d, q))
        print("               families: %s" % ", ".join(fams))
    print("\nSCENARIOS (%d)" % len(SCENARIOS))
    for s, v in SCENARIOS.items():
        print("  %-16s %-46s [%s]" % (s, v.description, v.physics_medium))
    cells = build_cells()
    # THE SAME FILTER `generate` APPLIES, or the estimate is of a different
    # run. `review_severity` names three scenarios and would otherwise have
    # been priced at the full 166-cell matrix -- 1.4 h against its real 8 min.
    def _set(v):
        return {x.strip() for x in str(v or "").split(",") if x.strip()}

    want, fams = _set(getattr(a, "scenario", "")), _set(getattr(a, "family", ""))
    if want:
        cells = [c for c in cells if c[0] in want]
    if fams:
        cells = [c for c in cells if c[1] in fams]
    note = ", ".join(sorted(want | fams))
    print("\nBUILD cells: %d%s"
          % (len(cells),
             " (of %d, filtered to %s)" % (len(build_cells()), note)
             if note else ""))
    if a.verbose:
        for scen, fam in cells:
            print("  %-16s x %s" % (scen, fam))
    _print_release_size(cells, a)
    return 0


#: Throughput of N parallel workers over ONE, by background. Measured on a
#: 32-vCPU box (Xeon 8488C: 16 cores + hyperthreads), release geometry, each
#: worker pinned to its own cores//N slice with Blender capped to match.
#: Seconds per frame PER CONTAINER while all N run:
#:
#:   workers     1      2      4      8      16     32
#:   solid     2.56   3.35   4.89   8.03  14.43  28.06
#:   hdri      5.82     -      -   20.80  39.54     -
#:
#: One render cannot use the box: 17.15 s a frame on one thread, 2.58 on 32 --
#: 6.6x from 32x the cores, because most of a frame is serial. Narrow renders
#: side by side are what buy throughput, and they top out near 3x because the
#: box has 16 PHYSICAL cores: sixteen one-thread renders pinned to those alone
#: run at the isolated 17.3 s a frame, and the hyperthreads add only ~20%.
#: Throughput follows physical cores -- re-measure on a different box.
#:
#: A worker count between two measurements prices at the lower one.
SPEEDUP = {"solid": {1: 1.0, 2: 1.53, 4: 2.09, 8: 2.55, 16: 2.84, 32: 2.92},
           "hdri": {1: 1.0, 8: 2.24, 16: 2.35}}


def speedup_for(workers: int, background: str = "solid") -> float:
    table = SPEEDUP.get(background, SPEEDUP["solid"])
    return table[max(k for k in table if k <= max(1, int(workers)))]


def _workers(value) -> int:
    """`--workers auto` (or `workers: auto` in a config) is one per usable core."""
    if str(value).strip().lower() == "auto":
        try:
            return len(os.sched_getaffinity(0))
        except AttributeError:
            return os.cpu_count() or 1
    return max(1, int(value))

#: Keyed by (tier, background), and each value is the tier's frame count times
#: the MEASURED per-frame render cost for that background. Rendering is what a
#: run spends its time on; build, simulate, annotate and overlay are not in
#: here, so a printed price is a floor and a real job runs somewhat over it.
#:
#: Per 512sq frame, spp 64, all seven passes, one scene, ONE container on an
#: idle 32-vCPU box (Xeon 8488C, the box `SPEEDUP` was measured on):
#:
#:      L0 2.56   |   L2 5.82
#:
#: so 89 frames come to 228 s and 518 s. The step is the DOME, not the
#: environment map: L2 and L3 project their HDRI onto one, and a dome encloses
#: the scene, so rays that miss an object bounce instead of escaping. L1 and L3
#: are priced as L0 and L2; on the previous 8-core box they measured within 3%
#: and 12% of them (7.69/7.93 and 19.31/21.61 s there).
#:
#: The debug pair is 25 frames at the per-frame cost measured at 128sq, and
#: predates the dome -- which is why it still holds: the ground is a cube slab
#: at every level again (`_common.ground`, see docs/roadmap.md), so L0 and L1
#: render the same geometry those numbers were taken on.
#:
#: WHAT WAS WRONG BEFORE: hdri was 2930 s, and was never measured at release
#: geometry at all -- it was the debug tier's 5.5x ratio scaled up. At 128sq
#: the fixed per-frame cost of an environment map dominates and that ratio is
#: real; at 512sq the sampling cost dominates and the true ratio is 2.6x. The
#: guess was 60% high, on the quarter of the dataset that is L2 and L3.
#:
#: `physloc/render/probe_cost.py` reproduces every per-frame number here, and
#: needs an IDLE box to do it -- check `docker ps` first.
SECONDS_PER_CLIP = {("debug", "solid"): 8.0, ("debug", "hdri"): 44.0,
                    ("release", "solid"): 228.0, ("release", "hdri"): 518.0}


#: What a scenario's job costs beyond what `progress.job_weight` models (level,
#: background, families x bins). Measured at release L0, one container, the
#: same valid + one invalid render: `pour` 973 s against `drop` 499 s.
SCENARIO_COST = {"pour": 2.0}


#: A job charged at least this much memory runs after every lighter one.
HEAVY_JOB_GB = 20.0


def _longest_first(jobs, tier, n_bins):
    """The job queue: lighter jobs longest first, then the memory-heavy ones.

    LONGEST FIRST, so the expensive HDRI and scanned-object jobs do not all
    arrive at the end of the queue to run while every other worker idles.

    HEAVY LAST. The first 61-frame release run started both L3 `pour` jobs --
    ~47 GB each -- at once. They held half the machine's memory for hours, 74
    of 96 workers waited for memory behind them, and half the CPU sat idle. A
    job charged HEAVY_JOB_GB or more now runs after every lighter job, when the
    machine is otherwise free, and with the threads to use it (`_threads_for`).

    Stable, so jobs of equal cost keep the order they were emitted in. The
    order changes nothing a job produces -- only which workers are busy when.
    """
    from .progress import job_weight
    from .scenarios.base import COMPLEXITY

    def cost(job):
        _seed, scenario, families, _variant, level, _n = job
        return (job_weight(level, tier, SECONDS_PER_CLIP, COMPLEXITY,
                           n_families=len(families), n_bins=n_bins)
                * SCENARIO_COST.get(scenario, 1.0))

    def key(job):
        heavy = job_memory_gb(job[1], tier, job[4]) >= HEAVY_JOB_GB
        return (heavy, -cost(job))

    return sorted(jobs, key=key)


#: How often `generate` counts the frames of the renders in progress.
FRAME_POLL_SECONDS = 5.0


def _frames_since(exr_dir, since) -> int:
    """Frames the render in progress has written: EXRs modified since `since`.

    Kubric writes one EXR per finished frame into the job's scratch folder, and
    the job's next render overwrites the same names -- so a file older than the
    moment the current render began belongs to a render already counted.
    """
    try:
        entries = os.scandir(exr_dir)
    except OSError:
        return 0
    n = 0
    with entries:
        for entry in entries:
            if not entry.name.endswith(".exr"):
                continue
            try:
                if entry.stat().st_mtime >= since:
                    n += 1
            except OSError:
                continue                    # replaced mid-scan by the next frame
    return n


def _kill_run_containers(run_id) -> int:
    """Kill every render container a run started; they carry its label."""
    try:
        ids = subprocess.run(["docker", "ps", "-q", "--filter",
                              "label=physloc.run=%s" % run_id],
                             capture_output=True, text=True, timeout=30).stdout.split()
        if ids:
            subprocess.run(["docker", "kill"] + ids, capture_output=True, timeout=120)
        return len(ids)
    except (OSError, subprocess.SubprocessError):
        return 0


def _install_stop_handler(stopping, run_id):
    """First SIGINT/SIGTERM: stop starting jobs and kill this run's containers.
    Second: exit immediately. Returns a function restoring the old handlers."""
    import signal

    def handler(signum, frame):
        if stopping.is_set():
            print("\n-- interrupted again: exiting now", file=sys.stderr, flush=True)
            _kill_run_containers(run_id)
            os._exit(130)
        stopping.set()
        print("\n-- stopping: no new jobs will start, killing this run's "
              "containers (Ctrl-C again to exit at once)", file=sys.stderr, flush=True)
        print("-- killed %d render container(s)" % _kill_run_containers(run_id),
              file=sys.stderr, flush=True)

    try:
        previous = {sig: signal.signal(sig, handler)
                    for sig in (signal.SIGINT, signal.SIGTERM)}
    except ValueError:                  # not the main thread: nothing to install
        return lambda: None

    def restore():
        for sig, old in previous.items():
            signal.signal(sig, old)

    return restore


def _changed_dials(tier, resolution, fps, frames, spp):
    """Each geometry dial, or None where it only restates `tier`'s own value."""
    def changed(value, own):
        return None if value is None or int(value) == own else int(value)

    return (changed(resolution, tier.resolution), changed(fps, tier.fps),
            changed(frames, tier.num_frames), changed(spp, tier.samples_per_pixel))


#: Where every generated tree lives: gitignored, and never the repository root.
OUTPUT_ROOT = "out"


def _under_out(path):
    """A relative output path, placed under `out/` unless it already is there.

    `generate --outdir r --workdir w` used to create `r/` and `w/` in the
    repository root, beside the source code. A bare or relative name now lands
    in `out/r` and `out/w`; a path already under `out/` is unchanged. Absolute
    paths, and relative ones that climb out with `..`, are left as given --
    those are an explicit choice of somewhere else.
    """
    if not path or os.path.isabs(path):
        return path
    norm = os.path.normpath(path)
    if norm == os.pardir or norm.startswith(os.pardir + os.sep):
        return norm
    if norm == OUTPUT_ROOT or norm.startswith(OUTPUT_ROOT + os.sep):
        return norm
    return os.path.join(OUTPUT_ROOT, norm)


def _renders_for(families, severity) -> int:
    """How many renders one worker job makes, decided as `render.worker` does.

    Its valid twin, plus one render per family per severity bin -- except a
    family with no magnitude axis, whose bins would be the same clip three
    times, so the worker renders only its strongest.
    """
    from .taxonomy import FAMILIES

    bins = (["weak", "medium", "strong"] if severity == "all"
            else [x.strip() for x in str(severity).split(",") if x.strip()])
    n = 1
    for family in families:
        meta = FAMILIES.get(family)
        graded = meta is None or getattr(meta, "graded", True)
        n += len(bins) if graded else min(1, len(bins))
    return n


#: How far apart each complexity level's seed block sits. Wide enough that no
#: run's variant count can reach the next block, so `--complexity all` produces
#: independent scenes per level rather than one scene laddered four ways, and
#: narrow enough to stay readable in a directory listing.
LEVEL_SEED_STRIDE = 1_000_000


#: What each distractor adds to a clip's render time, as a fraction of the base
#: rate. Measured: 14 families of `drop` at the debug tier took 216 s with six
#: distractors -- 15.5 s a clip against the 8.0 s baseline -- so six bodies
#: nearly double it, and one costs about 16%.
#:
#: Without this, `taxonomy` priced an L3 sweep at 2.0 h when it would have taken
#: 3.9, which defeats the point of pricing a run before committing to it.
DISTRACTOR_COST = 0.157


def _print_release_size(cells, a) -> None:
    """What a given configuration would actually produce, and how long it takes.

    The question "how many clips is this" has a non-obvious answer, because a
    valid twin is shared by every family and severity staged on the same
    scenario and seed. Counting cells times bins times variants overstates the
    render cost by about a third.
    """
    from .taxonomy import SCENARIOS, SEVERITY_BINS
    n_bins = len(SEVERITY_BINS) if a.severity == "all" else 1
    variants = max(1, a.variants)
    scenarios = {s for s, _ in cells}

    # ONE ROW PER LEVEL, because a ladder run is several different prices. The
    # levels differ in background -- an HDRI clip costs ~44 s against a solid
    # background's ~8 s at the debug tier -- and in how many variants they get,
    # so quoting the whole run at L0's rate understated a ladder by more than
    # a factor of two.
    from .scenarios.base import COMPLEXITY

    from .scenarios import TIERS

    levels = _levels_for(a.complexity, variants)
    invalid = valid = renders = 0
    serial = 0.0
    serial_bg = {}
    print("\n-- a release at tier %s / %s / severity %s / %d variant(s)"
          % (a.tier, a.complexity, a.severity, variants))

    # THE CONFIG'S GEOMETRY. Render time is per frame, so a changed frame count
    # scales the price exactly. Resolution and spp change the per-frame cost
    # itself, which was measured only at the tier's own geometry -- say so
    # rather than quote a number that is not one.
    frames_scale = 1.0
    base_tier = TIERS.get(a.tier)
    if base_tier is not None:
        res, _fps, frames, spp = _changed_dials(
            base_tier, getattr(a, "resolution", None), getattr(a, "fps", None),
            getattr(a, "frames", None), getattr(a, "spp", None))
        if frames:
            frames_scale = float(frames) / base_tier.num_frames
            print("   %d frames instead of %d: priced %.2fx"
                  % (frames, base_tier.num_frames, frames_scale))
        unmeasured = []
        if res:
            unmeasured.append("resolution %d" % res)
        if spp:
            unmeasured.append("spp %d" % spp)
        if unmeasured:
            print("   note: %s differ from the %s tier; the price uses the per-frame "
                  "cost measured at %d px / %d spp -- re-measure with "
                  "physloc/render/probe_cost.py"
                  % (" and ".join(unmeasured), a.tier, base_tier.resolution,
                     base_tier.samples_per_pixel))
    for level, n_v in levels:
        cx = COMPLEXITY.get(level)
        bg = cx.background if cx else "solid"
        rate = SECONDS_PER_CLIP.get((a.tier, bg), 60.0) * frames_scale
        # Every extra body is more geometry to shade, every frame -- but only
        # the `distractors` and `multi` conditions carry any, so the average
        # clip pays their combined share of the cost.
        if cx is not None:
            from .scenarios.base import EXTRA_OBJECTS, condition_share

            # Both crowd conditions draw their count from the same range, so
            # the average clip pays the mean count times their combined share.
            mean_extra = 0.5 * (EXTRA_OBJECTS[0] + EXTRA_OBJECTS[1])
            crowded = (condition_share("distractors")
                       + condition_share("multi")
                       + condition_share("camera+multi"))
            rate *= 1.0 + DISTRACTOR_COST * mean_extra * crowded
        inv = len(cells) * n_bins * n_v
        val = len(scenarios) * n_v      # one per scenario+seed, shared
        invalid += inv
        valid += val
        renders += inv + val
        serial += (inv + val) * rate
        serial_bg[bg] = serial_bg.get(bg, 0.0) + (inv + val) * rate
        print("   %-3s x%-3d %5d invalid + %4d valid = %5d renders "
              "@ %6.1f s = %5.1f h" % (level, n_v, inv, val, inv + val, rate,
                                       (inv + val) * rate / 3600.0))
    print("   %d cells x %d bin(s), %d renders total"
          "\n   (valid twins are one per scenario+seed, shared across families"
          "\n    and bins because the prefix is bit-identical)"
          % (len(cells), n_bins, renders))
    workers = _workers(getattr(a, "workers", 0) or 4)
    # Per background, because an HDRI frame scales worse than a solid one.
    parallel = sum(s / speedup_for(workers, bg) for bg, s in serial_bg.items())
    print("   ~%.1f h serial, ~%.1f h at the measured %.2fx on %d worker(s)"
          % (serial / 3600.0, parallel / 3600.0,
             serial / parallel if parallel else 1.0, workers))
    print("   media: %s" % ", ".join(
        "%s %d" % (m, sum(1 for s, _ in cells
                          if SCENARIOS[s].physics_medium == m))
        for m in sorted({SCENARIOS[s].physics_medium for s, _ in cells})))


# ------------------------------------------------------------------ export
def cmd_export(a) -> int:
    from .release.export import export
    out = export(a.root, a.outdir, license_name=a.license)
    if a.push_to:
        from .release.export import upload
        out["url"] = upload(a.outdir, a.push_to, private=a.private)
    print(json.dumps(out, indent=2, default=str))
    return 0


# ----------------------------------------------------------- randomisation
def cmd_stats(a) -> int:
    """Plot what a release actually contains.

    `validate` says it is well formed and `audit` says every cell depicts
    something; neither says what the DISTRIBUTIONS look like, and those are
    what a benchmark is judged on. Reads only `sample.json`, so it runs in
    seconds over a full release and can be re-run after any annotation change.
    """
    from .release.stats import report

    s = report(a.root, a.outdir)
    print("%d samples (%d invalid, %d valid) -> %s"
          % (s["samples"], s["invalid"], s["valid"], s["outdir"]))
    total = max(1, s["invalid"])
    print("   difficulty: " + ", ".join(
        "%s %d (%.0f%%)" % (lv, s["difficulty"].get(lv, 0),
                            100.0 * s["difficulty"].get(lv, 0) / total)
        for lv in ("easy", "moderate", "hard")))
    print("   binding:    " + ", ".join(
        "%s %d" % kv for kv in sorted(s["binding_factors"].items(),
                                      key=lambda kv: -kv[1])))
    return 0


def cmd_relabel(a) -> int:
    """Re-derive every difficulty label under a root, in place.

    A sample carries the label its annotation wrote, under the cuts of that day;
    after a recalibration `stats` would otherwise plot the old ones. This
    measures each clip again -- and each violator, which clips annotated before
    per-object labels have none of -- from the arrays beside its metadata.
    """
    from .annotate import difficulty as D
    from .annotate import layout

    n = 0
    for path in layout.find(a.root):
        if D.relabel(os.path.dirname(path)) is not None:
            n += 1
    print("relabelled %d invalid samples under %s" % (n, a.root))
    return 0


def cmd_randomisation(a) -> int:
    """Count the distinct values the sampler produces on each axis.

    "Is the dataset varied" is otherwise answered by looking at a few clips and
    forming an impression, and an impression cannot tell a working draw from
    one that is being thrown away downstream. Three separate axes turned out to
    be constants that way -- the event frame was clamped to a fixed fraction,
    the floor colour was a literal, and the camera decision was one coin flip
    shared by all thirteen scenarios -- and each of them looked fine.

    Renders nothing. It samples scenes and plans, so it runs in seconds.
    """
    import numpy as np

    from . import scenarios as scen_mod
    from .scenarios import TIERS

    names = ([a.scenario] if a.scenario else sorted(scen_mod.available()))
    tier = TIERS[a.tier]
    n = max(2, int(a.seeds))

    def key(v):
        try:
            return tuple(np.round(np.asarray(v, float).ravel(), 6))
        except Exception:                                      # noqa: BLE001
            return v

    axes = ("camera_pos", "camera_kind", "shape", "material", "mass",
            "size", "aspect", "colour", "floor", "backdrop", "start", "speed",
            "clutter")
    print("distinct values over %d seeds, tier %s / %s" % (n, a.tier, a.complexity))
    print("%-16s %s" % ("scenario", " ".join("%-9s" % x for x in axes)))
    totals = {x: set() for x in axes}
    for name in names:
        sc = scen_mod.get(name)
        seen = {x: set() for x in axes}
        for seed in range(n):
            # THE VARIANT INDEX MOVES WITH THE SEED. Camera motion and
            # distractors are stratified by variant, not drawn per scene, so
            # sampling every seed at variant 0 reported `camera_kind` as a
            # single value on every scenario -- a report saying the camera
            # never varies, from a sweep that never asked it to.
            sp = sc.sample(seed, tier, a.complexity, variant=seed)
            act = next((b for b in sp.bodies
                        if b.role == "actor" and not b.dormant), None)
            fl = next((b for b in sp.bodies if b.role == "floor"), None)
            seen["camera_pos"].add(key(sp.camera_position))
            seen["camera_kind"].add(sp.camera_motion_kind)
            seen["clutter"].add(int(sp.notes.get("n_distractors_placed") or 0))
            seen["backdrop"].add(key(sp.background_color))
            if fl is not None:
                seen["floor"].add(key(fl.color))
            if act is None:
                continue
            seen["shape"].add(act.kind)
            seen["material"].add(act.material)
            seen["mass"].add(round(float(act.mass), 4))
            seen["size"].add(key(act.scale))
            seen["aspect"].add(round(max(act.scale) / max(min(act.scale), 1e-9), 3))
            seen["colour"].add(key(act.color))
            seen["start"].add(key(act.position))
            seen["speed"].add(key(act.velocity))
        for x in axes:
            totals[x] |= {(name, v) for v in seen[x]}
        print("%-16s %s" % (name, " ".join("%-9d" % len(seen[x]) for x in axes)))

    print()
    print("A column of 1s is an axis that is NOT varying. Some are legitimate:")
    print("  shape     forced by scenarios whose family list needs one shape")
    print("  speed     scenarios that start their actor at rest")
    print("  material  one material per scene is deliberate on collision, pour")
    print("            and stack_topple -- their families need matched bodies")
    print("")
    print("`camera_kind` and `clutter` are ORTHOGONAL AXES, stratified by")
    print("variant index, so 2 is the healthy value -- on and off. A 1 here")
    print("means the sweep was too short to reach the axis: they fire at 20%")
    print("and 30%, so below 5 and 4 seeds respectively nothing turns on.")
    return 0


# ---------------------------------------------------------------- generate
#: The worker's word for "this family does not apply to this sample".
NO_PLAN = "injector produced no plan"


def _decline_only(info) -> bool:
    """Whether a non-zero worker result is only a sample-level decline.

    The renderer uses exit code 3 when every requested injector declines its
    sampled scene. That is scheduling information, not a renderer failure: the
    generator must expose the variant records to its fresh-seed retry loop.
    """
    variants = info.get("variants") if isinstance(info, dict) else None
    failed = [item for item in (variants or [])
              if not item.get("ok") and not item.get("skipped")]
    return bool(failed) and all(item.get("error") == NO_PLAN for item in failed)

#: How many fresh seeds a declined cell is offered before it counts as dead.
#: A cell that cannot be built on any of them is a genuine matrix error -- the
#: compatibility table claiming something the code cannot produce.
RETRY_SEEDS = 4


def cmd_generate(a) -> int:
    from .scenarios import TIERS
    from . import scenarios as scen_mod
    from .taxonomy import build_cells

    tier = "debug" if a.debug else a.tier
    if tier not in TIERS:
        from .scenarios.base import LEGACY_TIER_NAMES
        hint = LEGACY_TIER_NAMES.get(tier)
        print("unknown tier %r%s; valid tiers: %s"
              % (tier, (" -- renamed to %r" % hint) if hint else "",
                 ", ".join(TIERS)), file=sys.stderr)
        return 2

    # Individual dials, for sweeping one knob without inventing a tier. Every
    # config spells its tier's geometry out, so a dial that only restates the
    # tier is dropped here: the resume ledger and the worker then see exactly
    # the request they would have seen without it.
    a.resolution, a.fps, a.frames, a.spp = _changed_dials(
        TIERS[tier], a.resolution, a.fps, a.frames, a.spp)
    try:
        tier_obj = TIERS[tier].override(
            resolution=a.resolution, fps=a.fps, num_frames=a.frames,
            samples_per_pixel=a.spp)
    except ValueError as exc:
        print("bad tier override: %s" % exc, file=sys.stderr)
        return 2
    overrides = [("--resolution", a.resolution), ("--fps", a.fps),
                 ("--frames", a.frames), ("--spp", a.spp)]
    overrides = [(k, v) for k, v in overrides if v is not None]
    if overrides:
        print("tier %s with %s -> %s" % (
            tier, " ".join("%s %s" % kv for kv in overrides), tier_obj.name))

    from . import injectors
    have, inj = set(scen_mod.available()), set(injectors.available())
    cells = [c for c in build_cells() if c[0] in have and c[1] in inj]
    # A COMMA LIST, not just one name. `review_severity` needs three scenarios
    # -- the smallest set that reaches all 23 families -- and without this it
    # would have to be three runs into three directories.
    want = {x.strip() for x in str(a.scenario or "").split(",") if x.strip()}
    if want:
        unknown = want - {c[0] for c in cells}
        if unknown and len(want) > 1:
            print("unknown scenario(s): %s" % ", ".join(sorted(unknown)),
                  file=sys.stderr)
            return 2
        cells = [c for c in cells if c[0] in want]
    fams = {x.strip() for x in str(a.family or "").split(",") if x.strip()}
    if fams:
        cells = [c for c in cells if c[1] in fams]
    if len(want) == 1 and len(fams) == 1 and not cells:
        # An off-matrix probe: one named cell the compatibility matrix does not
        # list, for checking whether it ought to be.
        cells = [(next(iter(want)), next(iter(fams)))]
    if a.limit:
        cells = cells[:a.limit]
    if not cells:
        print("no (scenario, family) cells selected", file=sys.stderr)
        return 2

    # One worker run per (scenario, seed) covering every family of that
    # scenario: they share a single scene build, a single HDRI load and a
    # single valid render. At complexity L1 the environment costs ~4.6x an L0
    # render, so re-paying it per family was most of the wall clock.
    by_scenario = {}
    for scenario, family in cells:
        by_scenario.setdefault(scenario, []).append(family)

    work = _under_out(a.workdir) or os.path.join(OUTPUT_ROOT, "work")
    # THE KNOBS, RESOLVED ONCE for the whole run and written where the
    # container can read them. `common.yaml` plus this config's own `params:`
    # block; the worker gets a path, not a parse, because the container has no
    # PyYAML. See `physloc/params.py`.
    from . import config as _cfg
    from . import params as _params

    tunables = _cfg.load_params(getattr(a, "config", None))
    params_path = _params.write(tunables, work)
    _params.apply(tunables)
    print("-- params: %s" % params_path)
    rel = _under_out(a.outdir) or os.path.join(OUTPUT_ROOT, "release")
    for flag, given, used in (("--workdir", a.workdir, work),
                              ("--outdir", a.outdir, rel)):
        if given and os.path.normpath(given) != used:
            print("-- %s %s -> %s  (generated output lives under %s/)"
                  % (flag, given, used, OUTPUT_ROOT))

    done, failed, t0 = [], [], time.perf_counter()
    # One (variant, scenario) is one container run writing to its own
    # directory, so the units are independent and the order they finish in
    # cannot change what any of them produces: the per-clip rng is keyed by
    # (seed, family, severity) through crc32, never by position in a queue.
    # `tests/test_parallel_determinism.py` pins that.
    # The VARIANT INDEX travels with the seed. Some choices are spread across a
    # scenario's variants rather than drawn per scene -- camera motion is one,
    # so that every scenario gets the same share rather than each flipping its
    # own coin -- and the sampler cannot work the index out from the seed.
    levels = _levels_for(a.complexity, a.variants)
    if not levels:
        print("no complexity level selected", file=sys.stderr)
        return 2
    # EVERY LEVEL DRAWS ITS OWN SCENES. Each level gets its own seed block, so
    # an L1 clip is not an L0 clip wearing better materials -- it is a
    # different drop, of a different object, from a different height, under a
    # different camera.
    #
    # The alternative was tried and rejected: reusing one seed block across
    # levels makes every level a re-render of L0, which pairs clip for clip and
    # buys a neat ablation at the cost of the thing the dataset is actually
    # for. A benchmark wants breadth -- more distinct physical events -- and a
    # ladder whose upper levels contain no new scenes contributes none.
    #
    # The stride is far wider than any run's variant count, so blocks cannot
    # overlap and a level's seeds are reproducible from its name alone.
    jobs = [(a.seed + v + LEVEL_SEED_STRIDE * i, scenario, families, v, level,
             n_v)
            for i, (level, n_v) in enumerate(levels)
            for v in range(n_v)
            for scenario, families in sorted(by_scenario.items())]
    if len(levels) > 1:
        print("-- ladder: " + ", ".join("%s x%d" % lv for lv in levels))

    # ----------------------------------------------------------------- resume
    # A RUN THAT DIES AT 80% MUST NOT START OVER. A release run is days, and
    # until now `generate` had no memory at all: every invocation rebuilt every
    # clip, so an interrupted run -- a spot reclaim, a full disk, a Ctrl-C --
    # threw away everything it had already paid for.
    #
    # The ledger is one small json per JOB, written only after the job has
    # fully landed (worker ok, annotation done, overlays built). `--resume`
    # replays a job from its ledger entry instead of running it, and does so
    # ONLY when the recorded request matches the current one exactly. That last
    # part is what makes it safe to leave on: change the family list, the
    # severity ladder, the tier or any render dial and the entry no longer
    # matches, so the job is rebuilt rather than silently reused at the old
    # settings.
    #
    # It is keyed by the job's identity BEFORE it runs -- level, scenario,
    # seed, variant -- because the clip directory is named for the sampled
    # CONDITION, which is not known until the scene has been built.
    release_name = os.path.basename(os.path.normpath(rel)) or "physloc_v0"
    ledger_dir = os.path.join(rel, ".jobs")

    def _ledger_request(job):
        """What the current invocation is asking of this job.

        Every field here changes the OUTPUT. A dial that changes pixels but is
        absent from this dict is a resume bug: the run would keep clips made
        at the old setting and report them as new ones.
        """
        seed, scenario, families, variant, level, n_v = job
        return {"scenario": scenario, "seed": seed, "level": level,
                "variant": variant, "n_variants": n_v,
                "families": sorted(families), "severity": a.severity,
                "tier": tier, "window": a.window, "release": release_name,
                "overlay": not a.no_overlay,
                "dials": {"resolution": a.resolution, "fps": a.fps,
                          "frames": a.frames, "spp": a.spp},
                # The render backend is part of the request: a clip denoised
                # with NLM and one denoised with OIDN are not interchangeable,
                # and resuming across that change would ship a mixed release.
                "backend": {k: os.environ.get(k) for k in
                            ("PHYSLOC_ADAPTIVE", "PHYSLOC_DENOISER",
                             "PHYSLOC_GPU", "PHYSLOC_IMAGE")}}

    def _ledger_path(job):
        seed, scenario, _f, variant, level, _n = job
        return os.path.join(ledger_dir,
                            "%s_%s_%d_v%d.json" % (level, scenario, seed,
                                                   variant))

    def _ledger_load(job):
        """The recorded outcome, or None if absent, stale or unreadable."""
        path = _ledger_path(job)
        try:
            with open(path) as fh:
                entry = json.load(fh)
        except (OSError, ValueError):
            return None
        if entry.get("request") != _ledger_request(job):
            return None
        # A ledger entry is a claim about files. Verify the claim rather than
        # trusting it -- a half-deleted outdir, an rsync that dropped a file,
        # a disk that filled mid-write all leave an entry pointing at nothing,
        # and resuming onto those produces a release with holes in it that
        # `validate` finds days later.
        from .annotate.layout import SAMPLE_METADATA

        for sample in entry.get("samples", []):
            if not os.path.exists(os.path.join(sample, SAMPLE_METADATA)):
                return None
        return entry.get("outcome")

    def _ledger_save(job, outcome):
        if outcome.get("rc") != 0:
            return                      # only completed work is resumable
        samples = [r["samples"]["invalid"] for r in outcome.get("results", [])
                   if isinstance(r.get("samples"), dict)
                   and r["samples"].get("invalid")]
        os.makedirs(ledger_dir, exist_ok=True)
        path = _ledger_path(job)
        tmp = path + ".tmp"
        # Written atomically: a ledger truncated by a kill would be read back
        # as "no entry" at best and as a corrupt one at worst.
        with open(tmp, "w") as fh:
            json.dump({"request": _ledger_request(job), "samples": samples,
                       "outcome": outcome}, fh, default=str)
        os.replace(tmp, path)

    # Predicted cost per job, in job order, so the ETA is right in proportion
    # from the first few jobs rather than only near the end. See
    # physloc/progress.py -- a ladder emits its levels in blocks and they
    # differ by ~2.6x, so an unweighted rate promises an ending it cannot make.
    from .progress import HEARTBEAT_SECONDS, Profile, Progress, job_weight
    from .scenarios.base import COMPLEXITY
    from .taxonomy import SEVERITY_BINS

    n_bins = len(SEVERITY_BINS) if a.severity == "all" else len(
        [x for x in str(a.severity).split(",") if x])
    # LONGEST JOBS FIRST. The ladder emits its levels in blocks, L0 to L3, so
    # the expensive HDRI and scanned-object jobs all land at the end of the
    # queue -- where a few long ones run while every other worker sits idle.
    # Output is unchanged: a job's seed, variant and directory never depend on
    # its position.
    jobs = _longest_first(jobs, tier, n_bins)
    weights = [job_weight(level, tier, SECONDS_PER_CLIP, COMPLEXITY,
                          n_families=len(families), n_bins=n_bins)
               for _seed, _scen, families, _v, level, _n in jobs]
    renders = [_renders_for(families, a.severity)
               for _seed, _scen, families, _v, _level, _n in jobs]
    prof = Profile()

    def run_one(job, index=0):
        seed, scenario, families, variant, level, n_v = job

        # Each render the worker announces moves the bar the moment it lands,
        # and starts the clock for counting the next render's frames. The clock
        # moves BEFORE the bar does, so a frame count taken for the render that
        # just finished can no longer be applied on top of it.
        def on_line(line):
            if line.startswith("PHYSLOC_RENDERED "):
                _mark_render_boundary(index)
                progress.render_done(index)
                # A finished invalid render is a finished clip: annotate it now
                # rather than when the whole job ends.
                tag = line.split(" ", 1)[1].strip()
                if tag != "valid":
                    annotator.submit(os.path.join(job_dir, "variants",
                                                  tag.replace("/", "_")))
            elif line.startswith("PHYSLOC_NOT_RENDERED "):
                _mark_render_boundary(index)
                progress.render_dropped(index)

        # A level of its own in the work tree when the ladder is walked. Not
        # required for correctness any more -- the seed blocks are disjoint, so
        # the scratch paths cannot collide -- but a ladder run's scratch is
        # easier to read, and to delete a level from, when it is grouped.
        here = work if len(levels) == 1 else os.path.join(work, level)
        # Memory first, then a core slice -- always in that order, so two
        # threads can never each hold what the other is waiting for.
        held = memory.acquire(job_memory_gb(scenario, tier, level))
        env = slots.get()
        if stopping.is_set():
            # Admitted after a stop was asked for: start nothing, and pass the
            # memory straight on so the rest of the queue unwinds too.
            slots.put(env)
            memory.release(held)
            return {"scenario": scenario, "seed": seed, "level": level,
                    "variant": variant, "rc": 130,
                    "info": {"stderr": "stopped before it started"},
                    "results": [], "bad": []}
        # RUNNING only from here: a job waiting for memory is not running, and
        # counting it as one made a memory-starved pool look fully busy.
        progress.job_started(index)
        # An unpinned worker's thread count follows how busy the machine is
        # right now. The slot itself goes back to the pool unchanged.
        launch_env = env
        if workers > 1 and "PHYSLOC_CPUSET" not in env:
            launch_env = dict(env, PHYSLOC_THREADS=str(
                _threads_for(progress.running_count(), n_cores)))
        # Where this job's frames appear, for the frame count on the bar.
        job_dir = os.path.join(here, scenario, "%04d" % seed)
        with frame_watch_lock:
            frame_watch[index] = [os.path.join(job_dir, "_scratch", "exr"),
                                  time.time()]
        annotator = _SampleAnnotator(job_dir, rel, overlay=not a.no_overlay,
                                   prof=prof)
        try:
            with prof.timer("worker"):
                rc, info = _run_worker(scenario, seed, tier, ",".join(families),
                                       a.severity, here, complexity=level,
                                       window=a.window, variant=variant,
                                       n_variants=n_v, params_path=params_path,
                                       dials={"resolution": a.resolution,
                                              "fps": a.fps,
                                              "frames": a.frames,
                                              "spp": a.spp},
                                       env=dict(launch_env, **container_cap,
                                                PHYSLOC_MEMORY="%dg" % int(held)),
                                       on_line=on_line)
        finally:
            with frame_watch_lock:
                frame_watch.pop(index, None)
            slots.put(env)
            memory.release(held)
        annotated = annotator.finish()          # clips annotated as they rendered
        # The worker reports what Blender itself spent, per render. Recording
        # it beside the container round trip is what separates "the renderer is
        # slow" from "everything around the renderer is slow" -- and the
        # measured answer decides whether a GPU image is worth building.
        render = info.get("render_seconds") or {}
        if isinstance(render, dict):
            prof.add("render", sum(float(v) for v in render.values()),
                     n=max(1, len(render)))
        if rc != 0 and not _decline_only(info):
            return {"scenario": scenario, "seed": seed, "level": level,
                    "variant": variant,
                    "rc": rc, "info": info, "results": [], "bad": []}
        # A SKIP IS NOT A FAILURE. `colour_shift` cannot act on a scanned
        # asset, so at the GSO level it declines -- and reporting that as a
        # broken cell would train everyone to ignore the failure list.
        bad = [x for x in info.get("variants", [])
               if not x.get("ok") and not x.get("skipped")]
        skipped = [x for x in info.get("variants", []) if x.get("skipped")]
        produced = [x["dir"] for x in info.get("variants", []) if x.get("ok")]
        # Only what was not already annotated as it rendered. Never call
        # `_annotate` with an empty list: `only=[]` means "every variant".
        left = [x for x in info.get("variants", [])
                if x.get("ok") and os.path.normpath(x["dir"]) not in annotated]
        late = []
        if left:
            with prof.timer("annotate+overlay" if not a.no_overlay else "annotate"):
                # ONE CLIP'S ANNOTATION MUST NOT END THE RUN. The inline path
                # logs a failure and defers it to here, where nothing caught
                # it: the second failure went up through the worker pool and
                # took a 130-job sweep down at job 20, with 924 finished
                # samples already on disk and no summary of what went wrong.
                # A clip that cannot be annotated is a bad cell, reported like
                # any other, and the rest of the run still happens. Each is
                # annotated on its own so one failure does not lose the
                # variants queued beside it.
                for x in left:
                    try:
                        late.extend(_annotate(info["outdir"], rel,
                                              overlay=not a.no_overlay,
                                              only=[x["dir"]]))
                    except Exception as exc:                   # noqa: BLE001
                        bad.append(dict(x, ok=False,
                                        error="annotate failed: %r" % exc))
        results = [annotated[os.path.normpath(d)] for d in produced
                   if os.path.normpath(d) in annotated] + late
        return {"scenario": scenario, "seed": seed, "level": level,
                "variant": variant, "rc": 0,
                "info": info, "results": results, "bad": bad,
                "skipped": skipped}

    # A progress line as each job lands. The parallel path used to collect
    # every outcome before printing anything, so a twenty-minute run showed
    # nothing at all until it finished. The ordered results still print
    # afterwards, so the transcript stays identical at any worker count.
    #
    # RETRIES PUSH THE TOTAL UP. A declined cell is rebuilt on a fresh seed, so
    # the job count is not known until the run is over; `bump` widens the bar
    # rather than letting it sit at 100% while work continues.
    # A PLAIN-TEXT LOG BESIDE THE LEDGER. The bar lives in a terminal; the log
    # is what `tail -f` follows from anywhere, with a status line every few
    # minutes so a run is visibly alive long before its first job finishes.
    progress_log = os.path.join(rel, "progress.log")
    progress = Progress(len(jobs), weights=weights, renders=renders,
                        desc="generate %s" % (a.complexity or "L0"),
                        log_path=progress_log,
                        frames_per_render=tier_obj.num_frames)
    print("  progress log: %s  (tail -f it from anywhere)" % progress_log,
          flush=True)
    progress.start_heartbeat(HEARTBEAT_SECONDS)

    # FRAMES ON THE BAR. A render is a whole clip and minutes of wall clock, so
    # a bar counting renders sat still between them. Every few seconds, count
    # the EXRs each running job's current render has written. Host-side and
    # read-only: the container is not asked to report anything new.
    import threading as _threading

    frame_watch = {}                    # job index -> [exr dir, render began]
    frame_watch_lock = _threading.Lock()
    frame_watch_stop = _threading.Event()

    def _mark_render_boundary(index):
        with frame_watch_lock:
            if index in frame_watch:
                frame_watch[index][1] = time.time()

    def _watch_frames():
        while not frame_watch_stop.wait(FRAME_POLL_SECONDS):
            with frame_watch_lock:
                watched = [(i, d, t) for i, (d, t) in frame_watch.items()]
            for index, exr_dir, since in watched:
                frames = _frames_since(exr_dir, since)
                with frame_watch_lock:
                    current = frame_watch.get(index)
                    if current is None or current[1] != since:
                        continue            # that render finished meanwhile
                progress.set_inflight(index, frames)

    _threading.Thread(target=_watch_frames, name="frame-watch",
                      daemon=True).start()

    def run_and_report(job, index):
        label = lambda o: "%-16s seed=%-6d %-3s" % (o["scenario"], o["seed"],
                                                    o.get("level", ""))
        if getattr(a, "resume", False):
            cached = _ledger_load(job)
            if cached is not None:
                _failure_clear(job)
                progress.skip(label(cached), index=index)
                return cached
        progress.job_waiting(index)
        out = run_one(job, index)
        _ledger_save(job, out)
        if out["rc"] != 0:
            out["failure_log"] = _failure_save(job, out)
        else:
            _failure_clear(job)
        progress.update(label(out), ok=out["rc"] == 0, index=index)
        return out

    failure_dir = os.path.join(rel, "failures")

    def _failure_path(job):
        seed, scenario, _families, variant, level, _n = job
        return os.path.join(failure_dir, "%s_%s_%d_v%d.log"
                            % (level, scenario, seed, variant))

    def _failure_clear(job):
        try:
            os.unlink(_failure_path(job))
        except FileNotFoundError:
            pass

    def _failure_save(job, outcome):
        """Write why a job failed next to the release, and return the path.

        The reason used to reach the terminal and nowhere else, capped at 400
        characters in the end-of-run summary. The first v0 release run lost
        every failure that way: its terminal went with the instance, and what
        was left on disk was a progress log saying FAILED 176 times and nothing
        about why. A job that fails now leaves its whole stderr tail -- or the
        worker's own report, when it produced one -- in `failures/`, named like
        its ledger entry, so a rerun of the same job overwrites it.
        """
        seed, scenario, families, variant, level, _n = job
        path = _failure_path(job)
        info = outcome.get("info")
        body = (info.get("stderr") if isinstance(info, dict) and "stderr" in info
                else json.dumps(info, indent=2, default=str))
        try:
            os.makedirs(failure_dir, exist_ok=True)
            with open(path, "w") as fh:
                fh.write("job: level=%s scenario=%s seed=%d variant=%d\n"
                         "families: %s\nexit code: %s\nwhen: %s\n\n%s\n"
                         % (level, scenario, seed, variant, ",".join(families),
                            outcome.get("rc"), time.strftime("%F %T"), body))
        except OSError:
            return None
        return path

    workers = _workers(getattr(a, "workers", 1) or 1)
    slots = _cpu_slots(workers)
    try:
        n_cores = len(os.sched_getaffinity(0))
    except AttributeError:
        n_cores = os.cpu_count() or 1
    # The sum of admitted jobs' hard limits must fit below host RAM. A single
    # limit equal to the whole budget lets many containers exhaust the host.
    host_gb = _host_memory_gb()
    reserve_gb = min(HOST_RESERVE_GB, max(2.0, host_gb * 0.25))
    budget_gb = max(1.0, host_gb - reserve_gb)
    memory = MemoryBudget(budget_gb, backfill_seconds=BACKFILL_SECONDS)
    container_cap = {}

    # STOPPING. A render container does not die with the process that started
    # it, so Ctrl-C used to stop only this front-end while dozens of containers
    # rendered on -- and threads queued for memory went on starting new ones.
    # Every container now carries this run's id as a label; the first Ctrl-C
    # (or SIGTERM) stops new jobs from starting and kills exactly those
    # containers, and a second exits at once.
    import threading

    run_id = "%s-%d" % (time.strftime("%Y%m%d-%H%M%S"), os.getpid())
    container_cap["PHYSLOC_RUN_ID"] = run_id
    stopping = threading.Event()
    restore_signals = _install_stop_handler(stopping, run_id)
    print("  memory budget %.0f GB for %d worker(s); a job waits until its "
          "measured peak fits" % (budget_gb, workers), flush=True)

    def run_all(batch, offset=0):
        """Run `batch`, whose first job is job `offset` of the progress bar."""
        items = list(enumerate(batch, offset))
        if workers > 1 and len(batch) > 1:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=workers) as pool:
                return list(pool.map(lambda ij: run_and_report(ij[1], ij[0]),
                                     items))
        return [run_and_report(job, i) for i, job in items]

    # NO SERIAL WARM-UP JOB. One used to run alone "to populate Kubric's asset
    # cache", but the pinned Kubric has no shared cache: `AssetSource` copies
    # every asset into its own `tempfile.mkdtemp()` inside a `--rm` container,
    # so there is nothing for two containers to race on. At release geometry
    # that one job is ~40 renders, which held a 32-core box at one container
    # for most of half a day.
    outcomes = run_all(jobs)

    # A cell that DECLINED THIS SAMPLE gets another one.
    #
    # A family may legitimately refuse a scene without the cell being wrong.
    # `angular_momentum` declines a sphere, because an untextured sphere's
    # rotation is invisible however fast it spins -- and `toss` draws its shape
    # per seed, so `toss x angular_momentum` exists on the cube seeds and not
    # on the sphere ones. Seed 777 draws a sphere, and the cell silently
    # vanished from the release.
    #
    # `tests/conftest.py::reachable_cell` already walks seeds for exactly this
    # reason; the generator had no equivalent, so the tests said the cell was
    # fine and the run produced nothing. Retry seeds start past the variant
    # block so they can never collide with a seed the main pass already used.
    #
    # A DECLINE IS IDENTIFIED BY THE CLIP IT HAPPENED ON, not by the cell
    # alone: one ladder run offers the same cell a scene at every level, and
    # the same level several variants, so `toss x angular_momentum` can decline
    # at L0/v3 while standing at L2/v0. A retry therefore carries the declining
    # clip's LEVEL and VARIANT INDEX forward and changes only the seed -- the
    # variant index is what picks the condition (`scenarios.base.condition_for`
    # reads it together with the level's variant count), so a retry that
    # renumbered it would quietly hand the rebuilt cell a different camera
    # treatment or a different amount of clutter than the clip it replaces.
    n_at = dict(levels)
    level_i = {lv: i for i, (lv, _) in enumerate(levels)}
    declined = sorted({
        (o["level"], o["variant"], o["scenario"], b.get("family"))
        for o in outcomes
        for b in o["bad"] if b.get("error") == NO_PLAN})
    for attempt in range(RETRY_SEEDS):
        if not declined or stopping.is_set():
            break
        by_clip = {}
        for level, variant, scenario, family in declined:
            by_clip.setdefault((level, variant, scenario), []).append(family)
        retry_jobs = []
        for (level, variant, scenario), fams in sorted(by_clip.items()):
            # A BLOCK PER ATTEMPT, one seed per variant inside it. Past the
            # level's own variant block and still inside its stride, so a retry
            # seed can no more collide with another level than a main-pass one
            # can -- and distinct PER VARIANT, because a clip is written to
            # `<level>/<scenario>/<seed>_<condition>/` and several variants of
            # one level carry the same condition. One seed for the whole level
            # would have had the six `standard` retries of an L0 cell overwrite
            # each other in turn, leaving one clip where six were reported.
            seed = (a.seed + a.variants * (attempt + 1) + variant
                    + LEVEL_SEED_STRIDE * level_i.get(level, 0))
            retry_jobs.append((seed, scenario, sorted(fams), variant, level,
                               n_at.get(level)))
        print("  retrying %d declined cell(s) on %d clip(s), attempt %d"
              % (len(declined), len(retry_jobs), attempt + 1), flush=True)
        # RETRIES WIDEN THE BAR, priced like any other job, so it does not sit
        # at 100% while work continues.
        offset = progress.total
        progress.bump(len(retry_jobs),
                      weights=[job_weight(lv, tier, SECONDS_PER_CLIP, COMPLEXITY,
                                          n_families=len(fams), n_bins=n_bins)
                               for _s, _sc, fams, _v, lv, _n in retry_jobs],
                      renders=[_renders_for(fams, a.severity)
                               for _s, _sc, fams, _v, _lv, _n in retry_jobs])
        # Through the pool like the main pass. A release declines hundreds of
        # cells and each retry is a whole container job, so running them one
        # at a time left every other worker idle for the length of the tail.
        for out in run_all(retry_jobs, offset=offset):
            outcomes.append(out)
            if out["rc"] != 0:
                continue
            clip = (out["level"], out["variant"], out["scenario"])
            still = {clip + (b.get("family"),)
                     for b in out["bad"] if b.get("error") == NO_PLAN}
            declined = [c for c in declined
                        if c[:3] != clip or c in still]

    # The bar owns the terminal until here; the ordered summary below must not
    # be interleaved with a redraw, so close it before anything else prints.
    frame_watch_stop.set()
    progress.close()
    prof.report(workers=workers)
    restore_signals()
    if stopping.is_set():
        # Once more, for a container launched in the instant the stop landed.
        _kill_run_containers(run_id)
        kept = sum(1 for o in outcomes if o.get("rc") == 0)
        print("-- stopped. %d finished job(s) are kept; run the same command "
              "again to resume." % kept, file=sys.stderr, flush=True)
        return 130

    # Reported in job order, not completion order, so two runs at different
    # worker counts produce the same transcript.
    for out in outcomes:
        scenario, seed = out["scenario"], out["seed"]
        if out["rc"] != 0:
            print("worker failed for %s/%d: %s%s"
                  % (scenario, seed, str(out["info"])[:400],
                     ("\n   full log: %s" % out["failure_log"])
                     if out.get("failure_log") else ""), file=sys.stderr)
            failed.append((scenario, seed, "worker"))
            if not a.keep_going:
                return out["rc"]
            continue
        for bad in out["bad"]:
            # A cell that declined this sample and was then built on a later
            # seed is not a failure -- reporting it as one is how a healthy
            # run ends with a scary summary. Only cells still declining after
            # every retry are listed.
            if (bad.get("error") == NO_PLAN
                    and (scenario, bad.get("family")) not in set(declined)):
                continue
            failed.append((scenario, bad.get("family"), bad.get("error")))
            print("  !! %-15s %-17s %-6s %s"
                  % (scenario, bad.get("family"), bad.get("severity"),
                     bad.get("error")), file=sys.stderr)
        for res in out["results"]:
            done.append(res)
            print("  %-15s seed=%-5d %-17s %-6s t_event=%-3d lag=%-2d "
                  "vwin=%-13s sev=%.2f"
                  % (scenario, seed, res["family"], res["severity"],
                     res["t_event"], res["observability_lag"],
                     str(res["violation_windows"])[:13],
                     res.get("peak_severity", res["peak_score"])))
    dt = time.perf_counter() - t0
    print("\n%d pairs in %.1fs (%.1fs/pair)  ->  %s"
          % (len(done), dt, dt / max(len(done), 1), rel))
    # Cells the level cannot express were never owed, so they do not count as
    # missing -- see `Injector.available_at`.
    n_skipped = sum(len(o.get("skipped") or []) for o in outcomes)
    expected = len(cells) * sum(n for _, n in levels) * len(
        ["weak", "medium", "strong"] if a.severity == "all"
        else [x for x in a.severity.split(",") if x.strip()])
    if n_skipped:
        print("   %d cell(s) skipped: the complexity level removed what the "
              "family acts on" % n_skipped)
    expected -= n_skipped
    if len(done) < expected:
        print("\n!! %d of %d expected clips are MISSING -- see the failures below"
              % (expected - len(done), expected), file=sys.stderr)
    if failed:
        print("%d cell(s) produced nothing:" % len(failed), file=sys.stderr)
        for row in failed:
            print("   %s" % (row,), file=sys.stderr)
    # STATS ON EVERY RUN. `physloc stats` had to be remembered, and no script
    # remembered it, so a run's distributions were only ever looked at when
    # someone thought to ask. It reads sample.json only -- seconds, even over
    # a release -- and a failure to plot is reported, never a failed run.
    try:
        from .release.stats import report
        got = report(rel)
        print("stats: %d samples -> %s" % (got["samples"], got["outdir"]))
    except (Exception, SystemExit) as exc:                 # noqa: BLE001
        print("!! stats not written for %s: %r" % (rel, exc), file=sys.stderr)
    # A generated run is already the canonical dataset representation. Finish
    # its global manifest and index in place; `export` only copies this same
    # sample tree to a publication directory.
    try:
        from .release.export import finalize
        manifest = finalize(rel)
        print("dataset manifest: %d samples, %d pairs -> %s"
              % (manifest["samples"], manifest["pairs"], rel))
    except (Exception, SystemExit) as exc:                 # noqa: BLE001
        print("!! dataset manifest not written for %s: %r" % (rel, exc),
              file=sys.stderr)
    return 0


def _levels_for(spec: str, variants: int):
    """[(level, variants at that level)] for a `--complexity` value.

    `all` walks the whole built ladder at the shares declared on `COMPLEXITY`,
    so one run produces a dataset with L0, L1, ... in their intended
    proportions rather than needing a separate run per level and a decision
    about how big each should be.

    A level whose share does not buy a whole variant is DROPPED, which is what
    keeps a short run all-baseline. Naming levels explicitly overrides that:
    `--complexity L2 --variants 1` gives one L2 clip per cell, because asking
    for a level by name is asking for it.
    """
    from .scenarios.base import COMPLEXITY, implemented_complexities, variants_at

    text = str(spec or "").strip()
    if text.lower() in ("all", "ladder"):
        out = [(lv, variants_at(lv, variants))
               for lv in implemented_complexities()]
        return [(lv, n) for lv, n in out if n > 0]
    names = [x.strip() for x in text.split(",") if x.strip()]
    unknown = [x for x in names if x not in COMPLEXITY]
    if unknown:
        raise SystemExit("unknown complexity level(s): %s -- known: %s"
                         % (", ".join(unknown), ", ".join(COMPLEXITY)))
    if len(names) == 1:
        return [(names[0], max(1, int(variants)))]
    out = [(lv, variants_at(lv, variants)) for lv in names]
    return [(lv, n) for lv, n in out if n > 0]


def _cpu_slots(workers: int):
    """A queue of per-worker container environments: a disjoint core slice each.

    Blender sizes its Cycles thread pool from the HOST's core count, so without
    this every container on a 32-core box starts 32 threads and N workers run
    32N threads on 32 cores. A render is mostly NOT sampling -- scene sync,
    pass writing and the denoiser are largely serial -- so one render cannot
    use many cores well, and splitting the box into narrow slices, one render
    each, is what turns cores into throughput.

    One worker gets no slice and keeps Blender's own AUTO.

    NEVER ONE THREAD, AND NEVER ONE CORE. Cycles 2.93 hangs on the worker's
    first frame -- the main thread spins in `sched_yield` inside
    `_cycles.render`, and the render threads never run -- whenever it has fewer
    than two threads OR fewer than two cores. Measured on the worker path
    (`drop x solidity` L2 and a 32-worker L3 `generate`, debug tier):

        threads 1, all cores      hangs        threads 2, all cores   renders
        threads 1, one core       hangs        AUTO,      all cores   renders
        threads 2, one core       hangs

    So a slice narrower than MIN_RENDER_THREADS cores is not pinned at all:
    the worker runs on every core with two threads. Nothing is lost by it --
    sixteen workers measured 14.61 s a frame unpinned and 14.43 pinned; the
    throughput comes from running renders side by side, not from the pinning.
    """
    import queue

    slots = queue.Queue()
    try:
        cores = sorted(os.sched_getaffinity(0))
    except AttributeError:
        cores = list(range(os.cpu_count() or 1))
    if workers <= 1:
        slots.put({})
        return slots
    per = len(cores) // workers
    threads = str(max(MIN_RENDER_THREADS, per))
    for i in range(workers):
        if per >= MIN_RENDER_THREADS:
            chunk = cores[i * per:(i + 1) * per]
            slots.put({"PHYSLOC_THREADS": threads,
                       "PHYSLOC_CPUSET": ",".join(str(c) for c in chunk)})
        else:
            slots.put({"PHYSLOC_THREADS": threads})
    return slots


#: See `_cpu_slots`: Cycles 2.93 hangs below two threads or two cores.
#: `render.worker` applies the same floor to a hand-set `PHYSLOC_THREADS`.
MIN_RENDER_THREADS = 2

#: The most Blender threads one render gets. A single render's scaling flattens
#: past a handful -- 8 threads measured 4.7x one thread, 16 threads 6.1x -- and a
#: render started while few others run keeps its threads when more start later.
MAX_RENDER_THREADS = 8


def _threads_for(running: int, cores: int) -> int:
    """Blender threads for a job starting while `running` jobs, itself included,
    share `cores` cores: the cores per running job, between the floor and cap.

    A fixed two threads assumed every worker would run. When memory admits far
    fewer -- the first 61-frame release run had 22 of 96 rendering -- that left
    half the CPU idle. Thread count does not change the pixels.
    """
    return max(MIN_RENDER_THREADS,
               min(MAX_RENDER_THREADS, int(cores) // max(1, int(running))))


#: Peak memory of ONE worker job, GB, by (scenario, tier, level). Measured with
#: `docker stats` on real worker runs, one container on an otherwise idle box.
#:
#: It is flat in the number of renders: one `drop` job rendering all 43 of its
#: family x severity clips stayed between 0.79 and 0.82 GB throughout. What moves it is WHAT
#: IS IN THE SCENE, and one cell is the outlier: `pour` at L3. From L3 up every
#: moving body becomes a scanned GSO mesh (`scenarios.base._swap_in_gso`), and
#: in `pour` every grain moves -- so 96 grains are 96 textured meshes in
#: Blender and 96 mesh colliders in PyBullet, grinding against each other:
#:
#:      pour, debug:    L0 2.7   L1 2.7   L2 3.5   L3 26.4
#:      pour, release:  L0 5.9
#:      drop, release:  L0 2.6
#:
#: The release-tier L2 and L3 `pour` values are ESTIMATES, deliberately high:
#: the measured debug value scaled by the grain count (212 against 96). A job
#: that overruns its estimate is still capped per container (`PHYSLOC_MEMORY`),
#: so a low guess fails that job without exhausting the host.
#:
#: RELEASE VALUES ARE FROM A LIVE RUN: 96 vCPU / 185 GB, 61 frames, all
#: containers at once, mid-render --
#:
#:      pour     L0 4.5-4.9   L2 3.7-4.0   L3 39-47
#:      others   L2 ~1.1      L3 1.8-3.7   (collision, rolling_ramp)
#:
#: -- charged with a margin. Lower charges once admitted too many workers and
#: the host OOM killer terminated twelve `pour` jobs and the coordinator.
#:
#: A `None` scenario is the charge for every other scenario at that level.
# Full release pour jobs exceeded the old 10 GiB cap (kernel MEMCG OOM).
# Mid-render RSS is not the peak during pass extraction or later variants.
JOB_MEMORY_GB = {
    ("pour", "debug", "L0"): 3.0, ("pour", "debug", "L1"): 3.0,
    ("pour", "debug", "L2"): 4.0, ("pour", "debug", "L3"): 28.0,
    ("pour", "release", "L0"): 20.0, ("pour", "release", "L1"): 20.0,
    ("pour", "release", "L2"): 20.0, ("pour", "release", "L3"): 65.0,
    (None, "release", "L2"): 5.0, (None, "release", "L3"): 8.0,
}

#: How long the head of the memory queue may be passed by smaller jobs that fit.
#: See `MemoryBudget`.
BACKFILL_SECONDS = 1800.0

#: Hard limits require a safe peak allowance, even when average usage is lower.
#: The earlier 1 GB release charge admitted too many jobs at once and caused
#: host OOM kills; a release L0 `drop` alone has been measured at 2.6 GB.
DEFAULT_JOB_MEMORY_GB = {"debug": 3.0, "release": 4.0}

#: Left for the host itself -- the annotator, an editor, the docker daemon.
HOST_RESERVE_GB = 48.0


def job_memory_gb(scenario: str, tier: str, level: str = "L0") -> float:
    for key in ((scenario, tier, level), (None, tier, level)):
        if key in JOB_MEMORY_GB:
            return JOB_MEMORY_GB[key]
    return DEFAULT_JOB_MEMORY_GB.get(tier, 4.0)


def _host_memory_gb() -> float:
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) / (1024.0 * 1024.0)
    except OSError:
        pass
    return 0.0


class MemoryBudget:
    """Admit a job only once its charged memory fits in what is left.

    Workers alone bound CPU, not memory, and memory is what a crowded box runs
    out of first: see `JOB_MEMORY_GB`. A job larger than the whole budget is
    admitted alone rather than never.

    ORDER, WITH BACKFILL. Strict first-in-first-out starved the pool: on the
    first release run the head of the queue was a 6.5 GB job facing 6 GB free,
    and the 77 jobs behind it -- most charged 1 GB -- waited with it, 18 of 96
    workers running and 64% of the CPU idle. So while the head does not fit, a
    later job that DOES fit may start ahead of it -- but only for
    `backfill_seconds` after the head began waiting. After that the queue
    drains in order, so a big job is delayed a bounded time, never starved.
    `backfill_seconds=0` is strict first-in-first-out.
    """

    def __init__(self, total_gb: float, backfill_seconds: float = 0.0):
        import collections
        import threading

        self.total = max(0.0, float(total_gb))
        self.free = self.total
        self.backfill_seconds = max(0.0, float(backfill_seconds))
        self._cv = threading.Condition()
        self._queue = collections.deque()      # [token, time it began waiting]

    def _may_start(self, entry, gb: float) -> bool:
        if self.free < gb:
            return False
        head = self._queue[0]
        if head is entry:
            return True
        return time.monotonic() - head[1] < self.backfill_seconds

    def acquire(self, gb: float) -> float:
        gb = min(max(0.0, float(gb)), self.total)
        entry = [object(), time.monotonic()]
        with self._cv:
            self._queue.append(entry)
            while not self._may_start(entry, gb):
                self._cv.wait()
            self._queue.remove(entry)
            self.free -= gb
            self._cv.notify_all()
        return gb

    def release(self, gb: float) -> None:
        with self._cv:
            self.free += gb
            self._cv.notify_all()


def _run_worker(scenario, seed, tier, family, severity, workdir,
                complexity="L0", window=None, dials=None, variant=0,
                n_variants=None, params_path=None, env=None, on_line=None):
    cmd = ["bash", os.path.join(REPO, "docker", "kubric.sh"),
           "physloc/render/worker.py", "--scenario", scenario,
           "--seed", str(seed), "--tier", tier, "--family", family,
           "--severity", severity, "--complexity", complexity,
           "--variant", str(variant), "--outdir", workdir]
    if n_variants:
        cmd += ["--n-variants", str(int(n_variants))]
    if params_path:
        cmd += ["--params", params_path]
    if window:
        cmd += ["--window", str(window)]
    for flag, value in (dials or {}).items():
        if value is not None:
            cmd += ["--%s" % flag, str(value)]
    # STREAMED, not collected at exit: the worker announces each render as it
    # finishes (`PHYSLOC_RENDERED` / `PHYSLOC_NOT_RENDERED`), and `on_line`
    # hands those to the progress bar while the container is still running.
    # stderr is drained on its own thread so a chatty Blender cannot fill the
    # pipe and stall the worker.
    import collections
    import threading

    proc = subprocess.Popen(cmd, cwd=REPO, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, bufsize=1,
                            env=dict(os.environ, **env) if env else None)
    err = collections.deque(maxlen=2000)
    drain = threading.Thread(target=lambda: err.extend(proc.stderr), daemon=True)
    drain.start()
    info = None
    for line in proc.stdout:
        if line.startswith("PHASE0 "):
            info = json.loads(line[len("PHASE0 "):])
        elif on_line is not None and line.startswith("PHYSLOC_"):
            try:
                on_line(line.strip())
            except Exception:                              # noqa: BLE001
                pass                    # a progress hiccup must not fail a job
    proc.wait()
    drain.join(timeout=5)
    if info is not None:
        return (0, info) if info.get("ok") else (3, info)
    # Keep enough of the tail to contain the actual exception. 600 characters
    # cut the traceback off above the error line, which turned a diagnosable
    # container failure into "something went wrong in a png reader" -- and
    # Blender's own chatter after a crash can push the traceback out of 4000.
    # The whole of it lands in `failures/`, so err on the side of keeping more.
    return (proc.returncode or 4, {"stderr": "".join(err)[-40000:]})


# Each annotation opens two 512px render passes and builds a nine-panel video.
# The render containers have independent memory limits; these host-side jobs do
# not, so their concurrency must be bounded separately.
_ANNOTATION_SLOTS = threading.BoundedSemaphore(2)


def _annotate(workdir, outroot, overlay=True, only=None):
    # A fresh interpreter releases NumPy/HDF5/native allocator arenas at exit.
    # A semaphore bounds concurrent peaks, but cannot reclaim arenas retained
    # by the many long-lived sample-annotator threads in the coordinator.
    from . import params

    with _ANNOTATION_SLOTS:
        request = {"workdir": os.path.abspath(workdir),
                   "outroot": os.path.abspath(outroot), "overlay": overlay,
                   "only": [os.path.abspath(p) for p in only] if only else only,
                   "params": params.CURRENT}
        proc = subprocess.run(
            [sys.executable, "-m", "physloc.annotate.worker"],
            input=json.dumps(request), stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, cwd=REPO)
        if proc.returncode:
            raise RuntimeError("annotation worker exited %d: %s" %
                               (proc.returncode, proc.stderr[-40000:]))
        return json.loads(proc.stdout)


class _SampleAnnotator:
    """Annotate each completed render into a schema-v4 sample immediately.

    Videos used to appear only when a whole job ended: a job renders its valid
    twin and then every family at every severity -- about fifty clips, fifteen
    to twenty hours into a release run on a busy machine -- and only then was
    anything annotated. The worker announces each finished render, and its clip
    is annotated here the moment it lands.

    On a thread of its own, so a slow annotation never stalls reading the
    container's output; one clip at a time, because a job's clips share its
    valid twin's output. A clip that fails here is retried when the job ends.
    """

    def __init__(self, job_dir, outroot, overlay=True, prof=None):
        import queue
        import threading

        self.job_dir, self.outroot = job_dir, outroot
        self.overlay, self.prof = overlay, prof
        self.done = {}                          # normalised variant dir -> result
        self._queue = queue.Queue()
        self._thread = threading.Thread(target=self._run, name="sample-annotator",
                                        daemon=True)
        self._thread.start()

    def submit(self, vdir) -> None:
        self._queue.put(vdir)

    def _run(self) -> None:
        stage = "annotate+overlay" if self.overlay else "annotate"
        while True:
            vdir = self._queue.get()
            if vdir is None:
                return
            t0 = time.perf_counter()
            try:
                for result in _annotate(self.job_dir, self.outroot,
                                        overlay=self.overlay, only=[vdir]):
                    self.done[os.path.normpath(vdir)] = result
            except Exception as exc:                       # noqa: BLE001
                print("  !! annotating %s as it rendered failed (%r); "
                      "retrying when its job ends" % (vdir, exc),
                      file=sys.stderr, flush=True)
            finally:
                if self.prof is not None:
                    self.prof.add(stage, time.perf_counter() - t0)

    def finish(self):
        """Wait for every submitted clip, and return {variant dir: result}."""
        self._queue.put(None)
        self._thread.join()
        return self.done


def cmd_annotate(a) -> int:
    res = _annotate(a.workdir, a.outdir, overlay=not a.no_overlay)
    print(json.dumps(res, indent=2, default=str))
    return 0


def cmd_overlay(a) -> int:
    from .viz.overlay import build
    print(json.dumps(build(a.sample_dir, a.out, upscale=a.upscale), default=str))
    return 0


# ---------------------------------------------------------------- validate
def cmd_grid(a) -> int:
    from .viz.grid import build
    views = [v.strip() for v in a.views.split(",")] if a.views else None
    print(json.dumps(build(a.pair_dir, a.family, a.out, cell=a.cell,
                           views=views), default=str))
    return 0


def cmd_sheet(a) -> int:
    from .viz.grid import sheet
    print(json.dumps(sheet(a.pair_dir, a.out, view=a.view, cell=a.cell,
                           severity=a.severity), default=str))
    return 0


def _condition_in(pair_dir) -> Optional[str]:
    """The condition a sample pair carries, read from `sample.json`.

    Only needed for runs generated before the condition joined the clip path;
    a current one carries it in the directory name.
    """
    import glob

    from .annotate import layout

    for mp in glob.glob(os.path.join(pair_dir, "**", layout.SAMPLE_METADATA),
                        recursive=True):
        try:
            with open(mp) as fh:
                got = json.load(fh)["scene"].get("condition")
        except Exception:                                      # noqa: BLE001
            continue
        if got:
            return str(got).replace("+", "-")
    return None


def cmd_params(a) -> int:
    """Print the knobs in force, and where each one differs from the shipped
    default -- which is the question actually being asked when someone runs
    this: not "what are the settings" but "what has been changed"."""
    import json as _json

    from . import config as _cfg
    from . import params as _params

    got = _cfg.load_params(getattr(a, "config", None))
    if a.json:
        print(_json.dumps(got, indent=2, sort_keys=True))
        return 0
    changed = 0
    for section in sorted(got):
        print("%s:" % section)
        for key in sorted(got[section]):
            now = got[section][key]
            was = _params.DEFAULTS[section][key]
            mark = "" if now == was else "   <- changed from %r" % (was,)
            changed += bool(mark)
            print("  %-20s %s%s" % (key, now, mark))
    print("\n%d value(s) differ from the shipped defaults"
          % changed if changed else "\nall shipped defaults")
    return 0


def cmd_viz(a) -> int:
    """Every grid and sheet for a finished release, in ONE flat directory.

    `grid` and `sheet` each take one pair directory and write beside the samples,
    which is right for a single look and wrong for reviewing a run: a
    ten-variant sweep scatters its videos across twenty folders four levels
    deep, so the files you want to compare are the ones furthest apart.

    This walks the release and collects them under one directory, named so the
    sort order is the reading order:

        <level>_<condition>_<scenario>_<seed>_<family>.mp4
        <level>_<condition>_<scenario>_<seed>_sheet_<bin>.mp4

    THE CONDITION IS IN THE NAME, and early enough to sort on. Comparing a
    family across conditions is the comparison this axis exists for, and with
    the condition buried -- or absent, as it was -- that means opening
    `sample.json` per file or memorising which variant index is which. Sorted,
    every `camera` sample is now adjacent to every other.

    Nothing is re-rendered -- it reads the samples already on disk, so it costs
    seconds and can be run again after any change to the visualisers.
    """
    import glob

    from .viz.grid import build, sheet

    root = a.root
    outdir = a.outdir or os.path.join(root, "viz")
    os.makedirs(outdir, exist_ok=True)
    pairs = sorted(
        d for d in glob.glob(os.path.join(root, "samples", "*", "*", "*", "*"))
        if os.path.isdir(d) and glob.glob(os.path.join(d, "invalid_*")))
    if not pairs:
        print("no clip pairs under %s" % root, file=sys.stderr)
        return 2

    want = {x.strip() for x in (a.severity or "").split(",") if x.strip()}
    made, failed = [], []
    for pair in pairs:
        leaf = os.path.basename(pair)
        scenario = os.path.basename(os.path.dirname(pair))
        level = os.path.basename(os.path.dirname(os.path.dirname(pair)))
        # `0783_distractors` on a current run, a bare `0783` on one generated
        # before the condition joined the path. Read it from the clip's own
        # metadata when the directory does not carry it, so `viz` still names
        # things properly on a run you already have.
        seed, _, cond = leaf.partition("_")
        if not cond:
            cond = _condition_in(pair) or "?"
        stem = "%s_%s_%s_%s" % (level, cond, scenario, seed)
        cells = sorted(os.path.basename(d)
                       for d in glob.glob(os.path.join(pair, "invalid_*")))
        fams, bins = set(), set()
        for c in cells:
            body = c[len("invalid_"):]
            fam, _, sev = body.rpartition("_")
            if fam:
                fams.add(fam)
                bins.add(sev)
        for fam in sorted(fams):
            out = os.path.join(outdir, "%s_%s.mp4" % (stem, fam))
            try:
                build(pair, fam, out)
                made.append(out)
            except Exception as exc:                           # noqa: BLE001
                failed.append((out, repr(exc)))
        for sev in sorted(bins & want if want else bins):
            out = os.path.join(outdir, "%s_sheet_%s.mp4" % (stem, sev))
            try:
                sheet(pair, out, severity=sev)
                made.append(out)
            except Exception as exc:                           # noqa: BLE001
                failed.append((out, repr(exc)))
        print("  %-34s %d famil%s, %d bin(s)"
              % (stem, len(fams), "y" if len(fams) == 1 else "ies", len(bins)),
              flush=True)

    print("\n%d video(s) -> %s" % (len(made), outdir))
    for out, why in failed:
        print("  !! %s: %s" % (os.path.basename(out), why), file=sys.stderr)
    return 0


def cmd_compare(a) -> int:
    """The dataset's structure, side by side -- see `viz/compare.py`.

    One scenario x family with `--scenario` and `--family`; otherwise every
    cell the roots can support, or `--limit` of them per kind spread over the
    scenarios. Reads clips already on disk and renders nothing new.
    """
    from .viz import compare

    kinds = compare.KINDS if a.kind == "all" else (a.kind,)
    outdir = a.outdir or os.path.join(a.roots[0], "compare")
    if a.scenario and a.family:
        recs = compare.index(a.roots)
        made = []
        for kind in kinds:
            path = os.path.join(outdir, kind,
                                "%s__%s.mp4" % (a.scenario, a.family))
            try:
                made.append(compare.one(kind, recs, a.scenario, a.family, path,
                                        severity=a.severity, level=a.level,
                                        n=a.n))
            except ValueError as exc:
                print("  -- %s: %s" % (kind, exc), file=sys.stderr)
        print(json.dumps({"made": made}))
        return 0
    print(json.dumps(compare.batch(a.roots, outdir, kinds, limit=a.limit or None,
                                   severity=a.severity, level=a.level, n=a.n),
                     default=str))
    return 0


def cmd_coverage(a) -> int:
    from .viz.grid import coverage
    print(json.dumps(coverage(a.root, a.out, severity=a.severity), default=str))
    return 0


def cmd_config_path(a) -> int:
    """Where a config would write. Lets a script chain generate -> validate ->
    videos without hardcoding a path the config already knows.

    Reads the config's **generate** block, not its own: `outdir` is a setting of
    the run, and this command exists only to report it.
    """
    from . import config

    outdir = getattr(a, "outdir", None)
    if outdir is None and getattr(a, "config", None):
        _, subs = _build()
        valid = {ac.dest for ac in subs["generate"]._actions} - {"help", "config"}
        try:
            outdir = config.load(a.config, "generate", valid).get("outdir")
        except config.ConfigError as exc:
            print("config error: %s" % exc, file=sys.stderr)
            return 2
    # The same placement `generate` applies, or run.sh would validate and
    # package a different directory than the one it just generated into.
    print(_under_out(outdir) or os.path.join(OUTPUT_ROOT, "release"))
    return 0


def cmd_audit(a) -> int:
    """Report cells whose violation is not visible.

    The tool behind "if a cell does not change anything, do not build it": it
    measures severity, observability and pixel evidence per cell so the decision
    to drop one is a number rather than an opinion.
    """
    from .annotate.audit import (audit, is_invisible, is_unscored,
                                 ladder_failures, weak_failure)

    rows = audit(a.root)
    if not rows:
        print("no invalid clips under %s" % a.root, file=sys.stderr)
        return 2

    # THE WEAKEST BIN IS STILL A VIOLATION. Small, never absent: a weak clip
    # the eye cannot find, or one it can that scores zero, is a label without a
    # picture or a picture without a label. Grouped by cell, because the fix is
    # per family -- a stronger ladder for the first, a residual that sees the
    # violation for the second.
    weak_bad = {}
    for r in rows:
        why = weak_failure(r) if r["severity_bin"] == "weak" else ""
        if why:
            weak_bad.setdefault((r["family"], r["scenario"], why), []).append(r)
    ladders = ladder_failures(rows)
    if ladders:
        print("%d scene(s) whose bins do not climb weak -> medium -> strong:\n"
              % len(ladders))
        for r in ladders:
            print("  %-16s %-18s %-14s %s" % (
                r["scenario"], r["family"], r["why"],
                " / ".join("%.2f" % x for x in r["severities"])))
        print()
    n_weak = sum(1 for r in rows if r["severity_bin"] == "weak")
    print("%d weak clip(s), %d failing the weakest-bin rule (seen AND scored)\n"
          % (n_weak, sum(len(v) for v in weak_bad.values())))
    if weak_bad:
        print("%-18s %-16s %-10s %5s %9s %9s %7s" % (
            "family", "scenario", "why", "clips", "severity", "visible%",
            "frames"))
        for (fam, scn, why), rs in sorted(weak_bad.items()):
            print("%-18s %-16s %-10s %5d %9.3f %9.2f %7d" % (
                fam, scn, why, len(rs),
                min(float(r["peak_severity"]) for r in rs),
                100.0 * min(float(r.get("visible_share", 0.0)) for r in rs),
                min(int(r.get("visible_frames", 0)) for r in rs)))
        print()

    unscored = [r for r in rows if is_unscored(r)]
    if unscored:
        print("%d clip(s) VISIBLY WRONG BUT SCORED ZERO -- a picture with no "
              "label, which is worse than no clip:\n" % len(unscored))
        print("%-16s %-18s %-7s %8s %9s" % ("scenario", "family", "bin",
                                            "severity", "evidence"))
        for r in unscored:
            print("%-16s %-18s %-7s %8.3f %9.4f" % (
                r["scenario"], r["family"], r["severity_bin"],
                r["peak_severity"], r["evidence"]))
        print()

    flagged = [r for r in rows if is_invisible(r)]
    weak = sorted((r for r in rows if not is_invisible(r)),
                  key=lambda r: float(r["peak_severity"]))[:a.show_weak]

    print("%d invalid clips, %d with NO visible violation\n" % (len(rows),
                                                                len(flagged)))
    if flagged:
        print("%-16s %-18s %-7s %8s %6s %9s" % (
            "scenario", "family", "bin", "severity", "obs_f", "evidence"))
        for r in flagged:
            print("%-16s %-18s %-7s %8.3f %6d %9.4f" % (
                r["scenario"], r["family"], r["severity_bin"],
                r["peak_severity"], r["observable_frames"], r["evidence"]))
        print("\nThese depict nothing a viewer can see. Add them to")
        print("taxonomy.NOT_MEANINGFUL with the measured reason, or fix the")
        print("scenario so the family has something to act on.")
    if weak:
        print("\nweakest cells that DO show something:")
        for r in weak:
            print("   %-16s %-18s severity %.3f  evidence %.3f"
                  % (r["scenario"], r["family"], r["peak_severity"],
                     r["evidence"]))
    return 0


def cmd_validate(a) -> int:
    from .schema.validate import validate_release
    rep = validate_release(a.root)
    print(json.dumps(rep, indent=2))
    return 0 if rep["ok"] else 1


# ---------------------------------------------------------------------- #
def _build(suppress: bool = False):
    """The parser, and its subparsers by name.

    Built twice: once normally, and once with `argument_default=SUPPRESS` so
    the second pass reports *only* the options actually typed. That is what
    lets a config fill in the rest without ever overriding a flag -- comparing
    against the defaults instead would make "the user typed the default value"
    indistinguishable from "the user typed nothing".
    """
    kw = {"argument_default": argparse.SUPPRESS} if suppress else {}
    ap = argparse.ArgumentParser(prog="physloc", **kw)
    sub = ap.add_subparsers(dest="cmd", required=True)
    subs = {}

    def add_parser(name, **extra):
        q = sub.add_parser(name, **dict(extra, **kw))
        q.add_argument("--config", metavar="NAME|PATH",
                       help="settings file; a bare name resolves to "
                            "configs/<name>.yaml. Flags override it.")
        subs[name] = q
        return q

    p = add_parser("taxonomy",
                       help="print the taxonomy, and size a release")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="list every (scenario, family) cell")
    p.add_argument("--tier", default="release", help="debug | release")
    p.add_argument("--complexity", default="L0",
                   help="L0..L3, or `all` -- see the README's Complexity ladder section")
    p.add_argument("--scenario",
                   help="price only these scenarios (comma list), so the "
                        "estimate matches a filtered `generate`")
    p.add_argument("--family", help="price only these families (comma list)")
    p.add_argument("--severity", default="all")
    p.add_argument("--variants", type=int, default=5)
    p.add_argument("--workers", type=_workers, default=4,
                   help="price the run at this many parallel workers")
    p.add_argument("--resolution", type=int,
                   help="the run's render size, as generate takes it")
    p.add_argument("--fps", type=int, help="the run's frame rate")
    p.add_argument("--frames", type=int,
                   help="the run's clip length; the price scales with it")
    p.add_argument("--spp", type=int, help="the run's samples per pixel")
    p.set_defaults(fn=cmd_taxonomy)

    p = add_parser("generate", help="simulate+render+annotate end to end")
    p.add_argument("--debug", action="store_true", help="the debug tier (fast, unpublished)")
    p.add_argument("--tier", default="debug",
                   help="debug | release -- see `physloc taxonomy`")
    p.add_argument("--variants", type=int, default=1,
                   help="randomisations per cell: each is a fresh seed, so "
                        "sizes, speeds, colours, camera, HDRI and (where "
                        "physically neutral) the actor's shape all differ")
    p.add_argument("-n", "--limit", type=int, default=0,
                   help="cap the number of cells, for a quick smoke run")
    p.add_argument("--seed", type=int, default=91731)
    p.add_argument("--scenario", help="restrict to one scenario")
    p.add_argument("--family", help="restrict to one family")
    p.add_argument("--workers", type=_workers, default=1,
                   help="container runs in parallel. Each is pinned to its "
                        "own cores//N slice with Blender capped to match, so N "
                        "up to the core count scales. Output is identical at "
                        "any N.")
    p.add_argument("--keep-going", action="store_true",
                   help="carry on past a failing cell and list them at the end")
    p.add_argument("--severity", default="strong",
                   help="weak|medium|strong, or 'all' for the whole ladder in "
                        "one run (default: strong -- one clean variant per "
                        "cell, which is what you want while checking coverage)")
    p.add_argument("--window", type=int, default=None,
                   help="uniform violation duration in frames (sustained "
                        "families only; instant ones stay 1 frame)")
    p.add_argument("--resolution", type=int,
                   help="override the tier's square render size, e.g. 128")
    p.add_argument("--fps", type=int, help="override the tier's frame rate")
    p.add_argument("--frames", type=int,
                   help="override the tier's clip length; must be 4k+1 "
                        "(13, 17, 21, 25, 29, ...) for VAE latent alignment")
    p.add_argument("--spp", type=int,
                   help="override the tier's samples per pixel (render noise "
                        "vs time; frame time is ~1.29 + 0.0074*spp at 256sq)")
    p.add_argument("--complexity", default="L0",
                   help="L0..L3, a comma list, or `all` to walk the whole "
                        "ladder in one run at the declared shares "
                        "(see the README's Complexity ladder section)")
    p.add_argument("--workdir",
                   help="raw passes and trajectories; a relative path is placed "
                        "under out/ (e.g. `w` -> out/w)")
    p.add_argument("--outdir",
                   help="schema-v4 samples; a relative path is placed "
                        "under out/ (e.g. `r` -> out/r)")
    p.add_argument("--no-overlay", action="store_true")
    p.add_argument("--resume", action="store_true",
                   help="skip jobs this outdir has already completed. Safe to "
                        "leave on: a job is reused only when the RECORDED "
                        "request -- families, severity, tier, dials, render "
                        "backend, overlay -- matches the current one exactly "
                        "and every sample it claims still has a sample.json. "
                        "Change any of those and the job is rebuilt. Nothing "
                        "is ever deleted; a resume only declines to redo work.")
    p.set_defaults(fn=cmd_generate)

    p = add_parser("annotate", help="host-side annotation of a worker dir")
    p.add_argument("workdir")
    p.add_argument("--outdir", default="out/release")
    p.add_argument("--no-overlay", action="store_true")
    p.set_defaults(fn=cmd_annotate)

    p = add_parser("overlay", help="annotated mp4 for one invalid sample")
    p.add_argument("sample_dir")
    p.add_argument("--out")
    p.add_argument("--upscale", type=int, default=4)
    p.set_defaults(fn=cmd_overlay)

    p = add_parser("grid", help="one family: valid vs every severity, all views")
    p.add_argument("pair_dir", help=".../samples/<release>/<level>/<scenario>/<seed_condition>/")
    p.add_argument("--family")
    p.add_argument("--out")
    p.add_argument("--views", help="comma list of rgb,mask,sev,causal,div,energy,seg,depth,flow "
                                   "(default: every view the clips have)")
    p.add_argument("--cell", type=int, default=224)
    p.set_defaults(fn=cmd_grid)

    p = add_parser("sheet",
                   help="one scenario+seed: every family x every severity")
    p.add_argument("pair_dir", help=".../samples/<release>/<level>/<scenario>/<seed_condition>/")
    p.add_argument("--view", default="mask",
                   choices=["rgb", "mask", "sev", "causal", "div", "energy",
                            "seg", "depth", "flow"],
                   help="kept for single-view sheets; the default shows every "
                        "view the clips carry, one per row")
    p.add_argument("--severity", default="strong",
                   help="which bin fills the columns")
    p.add_argument("--out")
    p.add_argument("--cell", type=int)
    p.set_defaults(fn=cmd_sheet)

    p = add_parser("coverage",
                       help="one video tiling every invalid clip in a release")
    p.add_argument("root", nargs="?", default="out/release")
    p.add_argument("--out")
    p.add_argument("--severity", default="strong",
                   help="which bin to tile; one lattice per bin")
    p.set_defaults(fn=cmd_coverage)

    p = add_parser("compare",
                   help="dataset structure: levels, variants and conditions "
                        "of one cell side by side")
    p.add_argument("roots", nargs="+",
                   help="release roots to draw from, e.g. out/review_L0 "
                        "out/review_L3 out/review_conditions")
    p.add_argument("--kind", default="all",
                   choices=["all", "levels", "variants", "conditions"])
    p.add_argument("--scenario", help="with --family: just this cell")
    p.add_argument("--family")
    p.add_argument("--severity", default="strong")
    p.add_argument("--level", default="L0",
                   help="the level variants and conditions are drawn at")
    p.add_argument("-n", type=int, default=5, help="variants per video")
    p.add_argument("--limit", type=int, default=0,
                   help="at most this many cells per kind, spread over "
                        "scenarios (0 = every cell)")
    p.add_argument("--outdir", help="default <first root>/compare")
    p.set_defaults(fn=cmd_compare)

    p = add_parser("config-path",
                   help="print the outdir a config resolves to")
    p.add_argument("--outdir")
    p.set_defaults(fn=cmd_config_path)

    p = add_parser("export",
                   help="package an already schema-v4 generated dataset")
    p.add_argument("root", nargs="?", default="out/release",
                   help="a generated release directory (the one with samples/)")
    p.add_argument("--outdir", required=True,
                   help="where to write the packaged dataset")
    p.add_argument("--license", default="CC-BY-4.0")
    p.add_argument("--push-to", metavar="REPO_ID",
                   help="after packaging, upload with the `hf` CLI to this "
                        "Hugging Face dataset repo")
    p.add_argument("--private", action="store_true",
                   help="create the hub repo private")
    p.set_defaults(fn=cmd_export)

    p = add_parser("stats",
                   help="what a release contains: six figures and stats.json")
    p.add_argument("root", help="a release root, e.g. out/physloc_v0")
    p.add_argument("--outdir", help="default <root>/stats")
    p.set_defaults(fn=cmd_stats)

    p = add_parser("relabel",
                   help="re-derive every difficulty label under a root, in place")
    p.add_argument("root", help="a release root, e.g. out/physloc_v0")
    p.set_defaults(fn=cmd_relabel)

    p = add_parser("randomisation",
                   help="how much the sampler actually varies, per axis")
    p.add_argument("--seeds", type=int, default=24,
                   help="how many instances of each scenario to sample")
    p.add_argument("--tier", default="debug")
    p.add_argument("--complexity", default="L0",
                   help="L0..L3 -- see the README's Complexity ladder section. L0-L1 are built.")
    p.add_argument("--scenario", help="restrict to one scenario")
    p.set_defaults(fn=cmd_randomisation)

    p = add_parser("params",
                   help="the generation knobs in force, and what differs")
    p.add_argument("--json", action="store_true",
                   help="machine-readable, the same shape as params.json")
    p.set_defaults(fn=cmd_params)

    p = add_parser("viz",
                   help="every grid and sheet for a release, in one folder")
    p.add_argument("root", nargs="?", default="out/release")
    p.add_argument("--outdir",
                   help="where to collect them (default: <root>/viz)")
    p.add_argument("--severity",
                   help="only these bins for the sheets, e.g. weak,strong "
                        "(default: every bin the release contains)")
    p.set_defaults(fn=cmd_viz)

    p = add_parser("audit",
                   help="cells whose violation is not visible")
    p.add_argument("root", nargs="?", default="out/release")
    p.add_argument("--show-weak", type=int, default=10,
                   help="also list the N weakest cells that DO show something")
    p.set_defaults(fn=cmd_audit)

    p = add_parser("validate", help="schema + cross-checks over a release")
    p.add_argument("root", default="out/release", nargs="?")
    p.set_defaults(fn=cmd_validate)

    if suppress:
        # `argument_default=SUPPRESS` is overridden by any explicit `default=`
        # on an individual argument, which is most of them -- so the first
        # version of this reported `--complexity` as typed on every run and a
        # config could never set it. Force it onto every action instead.
        for parser in [ap] + list(subs.values()):
            parser._defaults = {}
            for action in parser._actions:
                action.default = argparse.SUPPRESS
    return ap, subs


def main(argv: Optional[List[str]] = None) -> int:
    from . import config

    argv = sys.argv[1:] if argv is None else list(argv)
    ap, subs = _build()
    a = ap.parse_args(argv)
    typed = set(vars(_build(suppress=True)[0].parse_args(argv)))

    valid = {ac.dest for ac in subs[a.cmd]._actions} - {"help", "config"}
    try:
        settings = config.load(getattr(a, "config", None), a.cmd, valid)
    except config.ConfigError as exc:
        print("config error: %s" % exc, file=sys.stderr)
        return 2
    for key, value in settings.items():
        if key not in typed:
            setattr(a, key, value)
    # `workers` is a setting of the GENERATE block, so pricing a config would
    # otherwise quote it at taxonomy's default whatever the file says it runs at.
    if a.cmd == "taxonomy" and "workers" not in typed and "workers" not in settings:
        gen_valid = {ac.dest for ac in subs["generate"]._actions} - {"help", "config"}
        try:
            gen = config.load(getattr(a, "config", None), "generate", gen_valid)
        except config.ConfigError:
            gen = {}
        if "workers" in gen:
            a.workers = gen["workers"]
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
