# Generating data

For whoever builds PhysLoc rather than downloads it: which config to run, what it costs, how to
follow, stop and resume a run. Simulation and rendering run in the pinned Kubric container;
everything else runs on the host. If you only want to use the dataset, the
[README](../README.md) is the document you want.

`python -m physloc.cli <command> --help` documents every command.

- [Configs](#configs) · [Running and checking a config](#running-and-checking-a-config) ·
  [The whole sweep](#the-whole-sweep-in-one-command) · [Overriding a config](#overriding-a-config)
- [Generating one level at a time](#generating-one-level-at-a-time) · [Tiers](#tiers) ·
  [Generation knobs](#generation-knobs)
- [Running the full release](#running-the-full-release) · [Cost and performance](#cost-and-performance)

Simulation and rendering run in the pinned Kubric container; everything else runs on the host.
`python -m physloc.cli <command> --help` documents every command.

## Configs

Each config answers one question. Price any of them first with
`python -m physloc.cli taxonomy --config <name>`.

| config | the question it answers |
|---|---|
| `review` | does **every cell** build? |
| `review_severity` | do **weak / medium / strong** differ, for every family? |
| `review_conditions` | do the **five conditions** do what they claim? |
| `review_L0` … `review_L3` | does **this level** render every scene correctly? |
| `review_ladder` | do the levels come out in their **declared proportions**? |
| `v0_mini` | **the whole dataset in miniature** — every family, level, condition and bin |
| `v0_release` | **the published dataset**: the whole ladder, 10 variants |
| `v0_L0` … `v0_L3` | one level of the release on its own — the four **sum to** `v0_release` |

`v0_mini` has the release's structure made small: ten variants (the fewest at which every level
and condition appears) over three scenarios that between them cover every family.

## Running and checking a config

`bash scripts/run.sh <config>` runs the whole pipeline — generate, validate, **stats**, **audit**,
coverage video, viz, **compare**, export — and passes extra flags through to `generate`. Nothing
has to be remembered afterwards: a finished run holds its own figures, its own audit and its own
structure videos. The steps individually:

```bash
python -m physloc.cli generate --config review_severity
python -m physloc.cli validate  out/review_severity      # schema + cross-checks; must exit 0
python -m physloc.cli audit     out/review_severity      # cells whose violation is not visible
python -m physloc.cli stats     out/review_severity      # the distributions, plotted
python -m physloc.cli viz       out/review_severity      # every grid and sheet, one folder
python -m physloc.cli coverage  out/review_severity      # every invalid clip, one video
python -m physloc.cli compare   out/review_L0 out/review_L3 out/review_conditions  # dataset structure

python test_dataset_loader.py out/review_severity        # load it: structure and shapes
python test_dataset_loader.py out/review_severity --gui  # any clip, any layer or panel
```

## The whole sweep, in one command

`scripts/run_reviews.sh` runs every review config back to back — the overnight job. Each config
gets `run.sh` in full and a log of its own under `out/logs/`, and one config failing never stops
the ones after it. It ends with a `compare` across **every** root the sweep produced, which is the
only place the L0 → L3 ladder can be drawn, since a single review run holds one level.

```bash
tmux new -s reviews                       # a long run belongs in tmux
bash scripts/run_reviews.sh               # every config
bash scripts/run_reviews.sh review_L2 review   # ...or just these
bash scripts/run_reviews.sh --frames 37   # every config at full clip length, into out/*_f37
export PHYSLOC_PUSH_OWNER=<user>          # optional: each run lands on the hub as physloc-<config>
```

Each run leaves everything worth looking at beside its samples:

```
out/<config>/
  coverage_strong.mp4     every invalid clip in one video -- open this first
  audit.txt               the cells whose violation is not visible
  stats/                  the six figures and stats.json
  viz/                    every grid and sheet, one folder
  compare/<level>/        variants and conditions of a cell, side by side
out/compare/              the same across every run of the sweep, including the level ladder
out/logs/<config>.txt     what that run printed
```

`compare` draws at most `PHYSLOC_COMPARE_LIMIT` videos per kind (default 12 per run, 20 across the
sweep), spread over the scenarios, because every eligible cell is hundreds of videos on a full
review. It renders nothing new — it reads finished samples.

`stats` checks the run came out in the shape it declares; it reads only `sample.json`, so it takes
seconds over a full release. **`generate` writes it at the end of every run and `export` ships it
with the release**, where the dataset card shows every figure:

```
out/review_severity/stats/
  1_overview.png              levels, conditions vs their declared shares, severity bins, violator timing
  2_taxonomy.png              domain -> family sunburst; clips per scenario, grouped by medium
  3_coverage.png              the scenario x family lattice under named domain bands, with totals
  4_difficulty.png            easy / moderate / hard; per level; which factor set it; the rule table
  5_difficulty_factors.png    every factor, one dot per clip, against its two cuts
  6_timing_and_severity.png   when events fire, observability lag, measured severity per bin
  stats.json                  the numbers behind all six; `stats.draw(json, dir)` redraws them
```

`4_difficulty.png` prints every cut and whether it was fitted to the review corpus or chosen from what the quantity means; how the label is decided is in the README's [Detection difficulty](../README.md#detection-difficulty).

`viz` re-reads finished samples — nothing is rendered again — and names its videos so the sort
order is the reading order:

```
out/review_severity/viz/
  L0_drop_0777_solidity.mp4        one family: the valid clip beside weak/medium/strong
  L0_drop_0777_sheet_strong.mp4    one scene: every family, at one bin
```

`compare` shows the dataset's structure: one scenario x family, side by side along one axis. It
takes several release roots, because a review run usually holds one level:

```
out/review_L0/compare/           (or --outdir)
  levels/drop__antigravity.mp4     L0 | L1 | L2 | L3 -- each level's OWN scene, not twins
  variants/drop__solidity.mp4      up to 5 variants of the cell at one level (--level, -n)
  conditions/drop__solidity.mp4    standard | camera | distractors | multi | camera+multi
```

Each tile is the invalid sample with its violation mask and timeline; a missing tile says "not
generated". `--scenario drop --family solidity` draws one cell; `--limit N` draws N cells per kind
spread over the scenarios. It reads finished samples only, at a few seconds per video.

## Overriding a config

Anything on the command line overrides the config:

```bash
python -m physloc.cli generate --config review --scenario drop --family solidity
python -m physloc.cli generate --config review --scenario drop,collision -n 6
python -m physloc.cli generate --config review --complexity all --variants 10
PHYSLOC_CAMERA_MOTION=orbit python -m physloc.cli generate --config review --scenario drop
```

`--workers N` runs N render containers at once, and `--workers auto` (used by the `v0_*` configs)
is one per core. Output is byte-identical at any worker count.

Generated output always lives under `out/`: a relative `--outdir` or `--workdir` is placed there
(`--outdir my_run` writes `out/my_run`), so nothing is ever written into the repository root.
Absolute paths are used as given.

## Generating one level at a time

`v0_L0` … `v0_L3` **partition** `v0_release`: each carries the variants that level gets in a full
run, so all four together produce exactly what the release config produces.

```bash
python -m physloc.cli taxonomy --config v0_L0     # price the level
python -m physloc.cli generate --config v0_L0     # ...and generate only that level
```

Use them to spread a release across machines, regenerate one level after a fix, or ship an
L0-only dataset. Nothing collides: the clip path is keyed by level and every level draws from its
own seed block.

## Tiers

A tier is a geometry — how big and how long — and nothing else:

<!-- physloc:tiers -->
| tier | resolution | frames | duration | spp | latent grid |
|---|---|---|---|---|---|
| `debug` | 128² | 25 @ 12 fps | 2.08 s | 16 | 7×8×8 |
| `release` | 512² | 89 @ 30 fps | 2.97 s | 64 | 23×16×16 |
<!-- /physloc:tiers -->

`debug` is for iteration and never published. Frame counts are `4k+1` so they map exactly onto a
video VAE's temporal stride. `v0` / `v1` are what a published dataset is *called*, set by
`--outdir`, not tiers.

**Every config spells out its tier's geometry** in its `defaults:` block, so a run is resized
there rather than by inventing a tier:

```yaml
defaults:
  tier: release
  resolution: 512      # square render size, pixels
  fps: 30              # frames per second
  frames: 89           # clip length; must be 4k+1
  spp: 64              # Cycles samples per pixel
```

As shipped these restate the tier and change nothing. Change one and only that field moves:
every clip records it in its tier name (e.g. `release+f49`), and `taxonomy` scales its price
with `frames`. Resolution and spp change the per-frame cost itself, so re-measure it with
`physloc/render/probe_cost.py` before trusting a price at a new size.

## Generation knobs

Shares, counts and bands live in **`configs/common.yaml`**, and any config may override part of
it in its own `params:` block:

```bash
python -m physloc.cli params                    # what is in force, and what differs from defaults
python -m physloc.cli params --config v0_mini   # ...for one run
```

| section | what it holds |
|---|---|
| `ladder` | each level's share of a full generation |
| `conditions` | the difficulty cycle — its length is the period, its contents the shares |
| `objects` | extra-object counts, violator counts, distractor size / speed / clearance |
| `camera` | motion kinds and weights, travel and dolly ranges |
| `materials` | the mass scale |
| `difficulty` | the detection-difficulty thresholds |

Every layer is validated, so a typo is an error rather than a silently ignored value. The resolved
values are written to `params.json` and recorded in every `sample.json`.


---

## Running the full release

```bash
bash scripts/run.sh v0_release
```

That is the whole command: it generates, validates, visualises and packages the release into
`out/physloc_v0`. Nothing needs editing first — the release config uses one worker per core,
resumes automatically, and renders with the default backend (64 spp, NLM denoiser, adaptive
sampling off).

A release takes days, so start it inside **tmux**, which keeps it running after you close the
terminal or disconnect:

```bash
tmux new -s release              # open a named session
bash scripts/run.sh v0_release   # start the run inside it
                                 # Ctrl-b then d: detach and leave it running
tmux attach -t release           # come back to it later
```

Before the first run on a new machine, check the price:

```bash
python -m physloc.cli taxonomy --config v0_release
```

### Following a run

- **The progress bar** counts frames — every frame of every clip — so it moves steadily, and its
  label keeps the finished clips (renders) and a weighted ETA:

  ```
  generate all:  12%|███▏                  | 74847/623420 [5h 14m<renders 1227/10220 | eta 1d 14h]
  ```

- **A status line every five minutes** appears above the bar, with frames, renders and jobs:

  ```
    status: frames 74847/623420 | renders 1227/10220 | jobs 18/260 done, 84 running, 12 waiting for memory | elapsed 5h 14m | eta 1d 14h
  ```

  A *job* is one container rendering one scenario at one level and variant — its valid clip and
  every family at every severity, one after another; a *render* is one clip; a frame is one of
  its 61 images.

- **`out/physloc_v0/progress.log`** records every finished job and every status line with a
  timestamp. Follow it from any terminal, whether or not the tmux session is attached:

  ```bash
  tail -f out/physloc_v0/progress.log
  ```

- **Samples appear as renders finish.** Each result is consolidated as
  `sample.json`, `rgb.mp4`, and `data.h5` under `out/physloc_v0/samples/`, so the first ones show
  up within the first hour, not when a whole job ends. Until then,
  `out/work_v0/<level>/<scenario>/<seed>/_scratch/images/` holds the frames of each clip in
  progress.
- **At the end**, a stage profile reports where the time went. Its `occupancy` line says how many
  workers were busy on average; far below the worker count means jobs were waiting on memory or
  on a long straggler, not on cores.

### Stopping a run

Each render runs in its own Docker container, and a container does not die with the process that
started it — so stopping a run has to stop the containers too:

- **Ctrl-C once** in the terminal running it: no new jobs start, the run's render containers are
  killed, and it exits saying how many finished jobs are kept. **Ctrl-C twice** exits at once.
- **From any other terminal**, or when the run's terminal is gone:

  ```bash
  bash scripts/stop.sh      # stops every generate / run.sh, then every PhysLoc container
  ```

It reports what is still running afterwards — `0 ... 0` means everything has stopped. Finished jobs
are kept either way, and running the same command again resumes.

### Resuming, splitting and memory

- **Resuming.** Running the same command again continues where it stopped: a finished job is
  skipped when its recorded request — config, dials and render backend — matches, so an
  interruption loses only the jobs in flight and a changed setting re-renders rather than mixing.
  To start over, delete `out/physloc_v0` and `out/work_v0`.
- **Across machines.** Run one level per machine — `bash scripts/run.sh v0_L0` on one,
  `v0_L1` on the next, and so on; see [Generating one level at a time](#generating-one-level-at-a-time).
- **Memory.** A job starts only when its memory fits in host RAM, and every container is capped,
  so a crowded machine cannot OOM-kill its own jobs; smaller jobs may start ahead of a big one
  that does not fit yet. One job type is heavy: release-size L3 `pour`, which measured 39–47 GB
  in a live run and is charged 55 GB in `JOB_MEMORY_GB` (`physloc/cli.py`). On a machine with much
  less memory, check it once — run this and watch `docker stats` in a second terminal:

  ```bash
  python -m physloc.cli generate --config v0_L3 --scenario pour --family continuity \
      --severity strong --variants 1 --workers 1 \
      --outdir out/_pour_l3_probe --workdir out/_pour_l3_probe_work
  ```


---

## Cost and performance

### What each config costs

Priced from constants measured on a 32-vCPU machine, at each config's own worker count.
`taxonomy --config <name>` prints the same figure with a per-level split. The price covers
rendering; a real run finishes roughly 10% over it.

<!-- physloc:costs -->
| config | levels | cells | renders | workers | wall clock |
|---|---|---|---|---|---|
| `review_severity` | L0 | 166 | 511 | 32 | **30 min** |
| `review_conditions` | L0 | 6 | 80 | 32 | **6 min** |
| `review_L0` | L0 | 166 | 179 | 32 | **12 min** |
| `review_L1` | L1 | 166 | 179 | 32 | **12 min** |
| `review_L2` | L2 | 166 | 179 | 32 | **1.2 h** |
| `review_L3` | L3 | 166 | 179 | 32 | **1.2 h** |
| `review_ladder` | L0+L1+L2+L3 | 12 | 320 | 32 | **48 min** |
| `review` | L0+L1+L2+L3 | 166 | 5110 | 32 | **14.0 h** |
| `v0_mini` | L0+L1+L2+L3 | 41 | 2520 | 32 | **6.2 h** |
| `v0_L0` | L0 | 166 | 5110 | 32 | 99 h (**4.1 days**) |
| `v0_L1` | L1 | 166 | 2555 | 32 | 50 h (**2.1 days**) |
| `v0_L2` | L2 | 166 | 1533 | 32 | 84 h (**3.5 days**) |
| `v0_L3` | L3 | 166 | 1022 | 32 | 56 h (**2.3 days**) |
| `v0_release` | L0+L1+L2+L3 | 166 | 10220 | 32 | 289 h (**12.0 days**) |
<!-- /physloc:costs -->

### Where a release's time goes

<!-- physloc:costs_ladder -->
| level | variants | renders | per render | at 32 workers | share of the run |
|---|---|---|---|---|---|
| **L0** | 10 | 5110 | 204 s | 99 h (**4.1 days**) | 34% |
| **L1** | 5 | 2555 | 204 s | 50 h (**2.1 days**) | 17% |
| **L2** | 3 | 1533 | 464 s | 84 h (**3.5 days**) | 29% |
| **L3** | 2 | 1022 | 464 s | 56 h (**2.3 days**) | 19% |
| **all four** | -- | 10220 | -- | 289 h (**12.0 days**) | 100% |
<!-- /physloc:costs_ladder -->

L0 and L1 are three quarters of the renders and about half the time; L2 and L3 cost more per
render because their HDRI dome encloses the scene. `v0_L0` alone is a complete, publishable
dataset.

### Expected time on other machines

Scaled from the 32-vCPU measurements by physical cores, including ~10% for scene build,
simulation and annotation. Rough: the per-core speed of a different CPU is not measured.

| machine | expected | pessimistic (+25%) |
|---|---|---|
| 32 vCPU = 16 cores + HT | 464 h (**19.3 days**) | 24.2 days |
| 64 vCPU = 32 cores + HT | 232 h (**9.7 days**) | 12.1 days |
| 96 vCPU = 48 cores + HT (e.g. c7i.24xlarge) | 155 h (**6.4 days**) | 8.1 days |
| 96 vCPU = 96 cores, no HT (e.g. c7a.24xlarge) | 95 h (**4.0 days**) | 5.0 days |
| 192 vCPU = 96 cores + HT (e.g. c7i.48xlarge) | 77 h (**3.2 days**) | 4.0 days |
| 4 × 96 vCPU (48 cores + HT each) | 39 h (**1.6 days**) | 2.0 days |

- **Prefer physical cores.** One render cannot use many threads, so throughput follows physical
  cores; hyperthreads add only ~20%.
- **Give it RAM**: about 2 GB per vCPU, so memory admission rarely queues anything but L3 `pour`.
- **Several machines scale almost linearly** — split by level, as above.

### Render settings

The release renders at **64 spp with the NLM denoiser and adaptive sampling off**. Against a
512-spp reference, 99% of pixels are within two levels (0–255) at a third of the render time.
Other settings exist as environment variables for experiments; they change the pixels, so never
mix them inside a release.

**How every number in this section was measured** — thread scaling, parallel throughput, memory
per job, the render-setting experiments, and how prices are computed — is in
[performance.md](performance.md).

