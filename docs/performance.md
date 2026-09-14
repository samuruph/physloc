# Performance and cost — how the numbers were measured

The README says what a run costs and what to expect on other machines. This page is the
evidence: what was measured, on what, and how `generate` uses it.

Unless stated otherwise, every number here was taken on a **32-vCPU box (Intel Xeon Platinum
8488C: 16 physical cores + hyperthreads, 61 GB RAM)**, at release geometry — 512², 64 spp, all
seven render passes — with nothing else running. A different CPU is a different price, so
re-measure on the machine that will run a release (see [Reproducing](#reproducing)).

- [One render and its threads](#one-render-and-its-threads)
- [Renders side by side](#renders-side-by-side)
- [How `generate` schedules workers](#how-generate-schedules-workers)
- [Memory per job](#memory-per-job)
- [Render settings](#render-settings)
- [How a run is priced](#how-a-run-is-priced)
- [Reproducing](#reproducing)

## One render and its threads

Most of a frame is serial — scene sync, pass writing, the NLM denoiser — so one render cannot
use a big machine:

| Blender threads | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| s / frame (L0) | 17.15 | 9.42 | 5.56 | 3.68 | 2.81 | 2.58 |
| speedup | 1.0× | 1.8× | 3.1× | 4.7× | 6.1× | 6.6× |

Thirty-two times the cores buys 6.6×. Throughput has to come from running renders side by side.

## Renders side by side

N containers at once, each pinned to `cores / N` with a matching thread count. Seconds per frame
are per container while all N run; throughput is relative to one container using the whole box:

| workers × threads | 1 × 32 | 2 × 16 | 4 × 8 | 8 × 4 | 16 × 2 | 32 × 1 |
|---|---|---|---|---|---|---|
| L0 s / frame | 2.56 | 3.35 | 4.89 | 8.03 | 14.43 | 28.06 |
| L0 throughput | 1.00× | 1.53× | 2.09× | 2.55× | 2.84× | **2.92×** |
| L2 (HDRI dome) s / frame | 5.82 | – | – | 20.80 | 39.54 | – |
| L2 throughput | 1.00× | – | – | 2.24× | 2.35× | – |
| CPU busy | 32% | 47% | 64% | 79% | 89% | 96% |

- **It flattens near 3× because the box has 16 physical cores.** Sixteen one-thread renders pinned
  to the physical cores alone run at the isolated 17.3 s a frame — no contention — and the
  hyperthreads add only ~20% on top. Throughput follows **physical cores, not vCPUs**.
- **Pinning itself buys nothing measurable** (sixteen workers: 14.61 s a frame unpinned, 14.43
  pinned). It is there to keep each container's thread count honest.
- **HDRI scales worse than a solid background**: an L2 frame costs 2.3× an L0 frame on an idle box
  and 2.74× under full load, which is why `SPEEDUP` in `physloc/cli.py` is kept per background.

*Caveat on the multi-worker rows.* They were taken before `probe_cost.py` gave each run its own
scratch folder, so concurrent containers shared one. Every container rendered the same resolution
and frame count, and the render dominates at 16 frames; what was not isolated is Kubric reading
the passes back. The single-container rows are clean.

## How `generate` schedules workers

`--workers N` (or `workers: auto`, one per core) starts N render containers at once.

**Core slices.** Each worker gets a disjoint slice of `cores / N` cores and the same number of
Blender threads — never fewer than two threads, and never a one-core slice. Cycles 2.93 hangs on
the first frame of a real worker job otherwise: the main thread spins in `sched_yield` inside
`_cycles.render` and the render threads never run. Measured on `drop × solidity`, L2, debug tier:

| Blender threads | cores available | result |
|---|---|---|
| 1 | all | hangs — no frame in 300 s |
| 1 | 1 (pinned) | hangs |
| 2 | 1 (pinned) | hangs |
| 2 | all | renders |
| AUTO | all | renders |

So when `cores / N` is below two, workers run unpinned with two threads each. Thread count does
not change the pixels: all seven passes are byte-identical at 2 and 32 threads.

**No serial warm-up.** Kubric copies each asset into its own temporary folder inside a
throwaway container, so there is no shared cache to race on; 32 containers fetching HDRI and GSO
assets cold at the same moment ran without an error.

**Memory admission.** A job starts only when its charged memory fits in a budget of host RAM
minus 6 GB, and every container is capped at that budget, so an overrun kills one job rather than
letting the host's OOM killer pick a victim. Jobs are admitted in order, **with backfill**: while
the job at the head of the queue does not fit, a later job that does may start ahead of it, for up
to 30 minutes of the head's waiting; after that the queue drains in order. Strict ordering starved
the first release run — the head was a 6.5 GB job facing 6 GB free, the 77 jobs behind it waited
with it, and 18 of 96 workers ran with 64% of the CPU idle. See [Memory per job](#memory-per-job).

**Longest jobs first.** The ladder emits its levels in blocks, which put every expensive HDRI and
scanned-object job at the end of the queue, running while the rest of the pool sat idle. Ordering
the queue by predicted cost, replayed over the real `v0_release` job list without memory limits:

| workers | emitted order | longest first |
|---|---|---|
| 32 | 40.6 h | 30.5 h |
| 96 | 22.7 h | 17.5 h |

(Relative figures from a scheduling simulation that does not model contention — compare rows,
not absolute hours.) A job's seed, variant and output directory never depend on its position, so
the order changes nothing a run produces.

**Retries** of cells that declined their scene run through the same pool.

**Occupancy.** Every `generate` ends with a stage profile whose `occupancy` line reports how many
workers were busy on average. Well below the worker count means jobs were waiting on memory or on
a long straggler, not on cores.

## Memory per job

Peak memory of one worker job (valid plus invalid renders), measured with `docker stats`:

| job | L0 | L1 | L2 | L3 |
|---|---|---|---|---|
| `pour`, debug | 2.7 GB | 2.7 GB | 3.5 GB | **26.4 GB** |
| `pour`, release | 5.9 GB | – | 3.7–4.0 GB (live run) | 39–47 GB (live run, mid-job) |
| `drop`, release | 2.6 GB | – | – | – |

- **Memory is flat in the number of renders**: one `drop` job rendering 43 clips stayed between
  0.79 and 0.82 GB.
- **Every other scenario peaked at 0.3–2.6 GB**, at any level.
- **L3 `pour` is the outlier.** From L3 up every moving body becomes a scanned GSO mesh, and in
  `pour` every grain moves — 96 textured meshes and 96 mesh colliders at the debug tier. Three
  `pour` jobs among 32 workers on the 61 GB box got one OOM-killed. The difficulty condition does
  not change it (25.0–26.4 GB across `standard`, `camera` and `camera+multi`).

**How admission charges a job.** Only the cells in `JOB_MEMORY_GB` (`physloc/cli.py`) are charged
their measured peak; every other job is charged 1 GB, what one typically holds — 32 release
containers rendering together averaged ~0.6 GB each. Charging every job its peak left most of the
pool idle: replaying the release queue on 96 workers and 186 GB, 32.4 h against 22.1 h.

**The release L3 `pour` figure is an estimate** — the debug value scaled by grain count (212
against 96), set high on purpose. It drives most of the remaining queueing, so measure it once on
a new machine (the command is in the README, *Running the full release*) and update
`JOB_MEMORY_GB`.

## Render settings

**The release renders at 64 spp with the NLM denoiser and adaptive sampling off.** Against a
512-spp reference of the same clip (`drop × solidity`, L2, 512², 25 frames, both clips):

| setting | RMSE vs 512 spp | 99th-percentile error | s / frame (L2) |
|---|---|---|---|
| **64 spp, NLM, no adaptive (release)** | **0.58–0.63** | **2.0** | **5.8** |
| 512 spp reference | — | — | 18.0 |

On a 0–255 scale: 99% of pixels are within two levels of a render that costs three times as much.
Adaptive sampling at 64 spp moved the image by RMSE 4.6 from the default — seven times the
default's own error — so it was not adopted.

Earlier experiments, on an 8-core box, L0, against a 512-spp reference:

| backend | s / frame | speedup | RMSE | worst pixel |
|---|---|---|---|---|
| NLM, no adaptive (default) | 7.80 | 1.00× | 0.15 | 7 |
| adaptive + `OPENIMAGEDENOISE`, spp 128 | 5.87 | 1.33× | 0.30 | 10 |
| adaptive + no denoiser, spp 128 | 3.72 | 2.10× | 0.37 | 11 |
| adaptive + no denoiser, spp 64 | 3.57 | 2.18× | 0.45 | 22 |

NLM costs 2.4 s of a 7.8 s frame there, and the dome levels gain less from adaptive sampling
(1.53× at L2) because a dome never fully converges. Prefix identity holds under every setting:
both twins render in one process with one setting.

The settings are environment variables, inherited by `scripts/run.sh` and `generate` and
forwarded into the container by `docker/kubric.sh`. **They change the pixels, so choose before a
run and never mix them inside a release** — every clip records its backend in `plan.json`, and
resuming a run re-renders rather than mixing.

| variable | values | default |
|---|---|---|
| `PHYSLOC_ADAPTIVE` | `1`, or a float noise threshold | unset — off |
| `PHYSLOC_DENOISER` | `off`, `OPENIMAGEDENOISE`, `NLM` | unset — `NLM` |
| `PHYSLOC_GPU` | `1` | unset — CPU |
| `PHYSLOC_THREADS` | Blender render threads (floor 2) | set by `generate` |
| `PHYSLOC_CPUSET` | cores to pin a container to | set by `generate` |
| `PHYSLOC_MEMORY` | container memory cap, e.g. `40g` | set by `generate` |

`scripts/run_fast.sh` pre-sets the adaptive pair for experiments; it is not for the release.

**GPU.** The pinned image's Blender is the CPU-only `bpy` wheel — no OptiX, and CUDA enumerates the
host CPU only — so `PHYSLOC_GPU=1` fails loudly instead of silently rendering on the CPU. Reaching
a card means replacing Blender; `docker/Dockerfile.gpu` is a starting point.

## How a run is priced

`python -m physloc.cli taxonomy --config <name>` prices a run from two constants in
`physloc/cli.py`:

- **`SECONDS_PER_CLIP`** — a tier's frame count times the measured per-frame cost of one
  container, by background:

  | | s / render | frames × s / frame |
  |---|---|---|
  | `release` solid | 228 | 89 × 2.56 (32-vCPU box) |
  | `release` hdri | 518 | 89 × 5.82 (32-vCPU box) |
  | `debug` solid | 8 | 25 × 0.32 (earlier 8-core box) |
  | `debug` hdri | 44 | 25 × 1.76 (earlier 8-core box) |

- **`SPEEDUP`** — the measured throughput at a worker count, per background (the tables above).

`DISTRACTOR_COST` raises the per-render rate for the extra bodies the `distractors` and `multi`
conditions carry, and `SCENARIO_COST` prices `pour` at 2× (973 s against `drop`'s 499 s for the
same two release renders). The price covers rendering only — scene build, simulation, annotation
and overlays come on top — so a real run finishes roughly 10% over it.

**Why HDRI levels cost more.** L2 and L3 project their environment onto a dome that encloses the
scene, so rays that miss an object bounce instead of escaping. The dome is a render-only backdrop
with collisions disabled; the ground is the same slab at every level.

## Reproducing

```bash
# Per-frame cost of one container. Needs an idle box: check `docker ps` first.
bash docker/kubric.sh physloc/render/probe_cost.py --complexity L0 --spp 64 --resolution 512 --frames 4
bash docker/kubric.sh physloc/render/probe_cost.py --complexity L2 --spp 64 --resolution 512 --frames 4

# The same with a render setting changed; refuses to run while a container is up.
bash scripts/probe_backend.sh L2 512 64
```

Each probe renders into its own scratch folder. Kubric's post-processing reads back every EXR in
that folder, so reusing one across runs silently adds the previous run's frames to the timing.
