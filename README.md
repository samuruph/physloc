# PhysLoc

**A spatio-temporally annotated physics-violation video dataset.** Every clip that breaks a
physical law ships with *where in the frame*, *exactly when*, *for how long*, and *how badly*
— all derived from the simulator, not from human annotation.

> **Status: every build cell generates, annotates and validates.** 13 built scenarios × 23
> violation families compose through the trajectory seam into **166 cells**, and
> `physloc generate` walks the whole matrix in one command. Each clip depicts *one*
> violation — that is asserted, not hoped for; see
> [§7](#7-orthogonality-what-the-labels-guarantee). Publishing:
> [§5b](#5b-publishing-a-release).
> Design doc: [docs/PLAN.md](docs/PLAN.md)

---

## 1. Setup (once)

```bash
# Host environment: annotation, severity, grids, validation, viz.
# Does NOT contain Kubric/Blender/PyBullet -- those live in the docker image.
conda env create -f environment.yml
conda activate physloc

# The render image (digest pinned in docker/IMAGE_DIGEST)
docker pull kubricdockerhub/kubruntu

# Optional: read-only Kubric source for reference (gitignored)
bash scripts/fetch_refs.sh
```

**Two environments, never mixed.** The pinned container holds Kubric 2022.4.1 / Blender
2.93.4 / PyBullet (Python 3.9) and does simulation + rendering. The `physloc` conda env
(Python 3.11) does everything else. They meet at the trajectory seam (`traj.npz`).

---

## 2. The runs

Three configs, one command each. `conda activate physloc` first.

| | config | what it is | cost |
|---|---|---|---|
| **review** | `configs/review.yaml` | every cell once, all bins, tier `debug`, L0 | **~35 min** |
| **ladder** | `configs/review_ladder.yaml` | the whole ladder, tier `debug`, 10 variants | ~3.1 h at 4 workers |
| **v0** | `configs/v0_release.yaml` | tier `release`, **whole ladder**, all bins, 3 variants | price with `taxonomy` |

**`v0` and `v1` are release names, not tiers.** There are two tiers, because there are two
geometries: `debug` (128², 25 f) and `release` (512², 89 f). Difficulty is the
[complexity ladder](#8-the-complexity-ladder), and what a published dataset is *called* comes
from `--outdir`, recorded as `release` in every `meta.json`.

### The review sweep — run this before anything else

Every scenario × family cell once, at debug size, `strong` only. This is the run to look at
when you want to know whether the dataset is right.

```bash
bash scripts/run.sh                      # generate + validate + every video
```

Once it finishes, two cheap follow-ups worth making a habit:

```bash
python -m physloc.cli randomisation --seeds 24    # is it actually varied? renders nothing
python -m physloc.cli export out/release --outdir out/hf/preview
```

To see whether instances actually *differ* — which one render per cell cannot show — use the
four-variant sweep instead:

```bash
bash scripts/run.sh review_ladder
```

That generates, validates, and builds every video: `coverage.mp4`, a `sheet` per scenario and
a `grid` per family. Then open `out/release/coverage.mp4` first.

The same thing by hand, if you want the steps separately:

```bash
python -m physloc.cli generate --config review
python -m physloc.cli validate out/release
python -m physloc.cli coverage out/release
```

### The v0 release

**Price it before you start it** — `taxonomy` reads the same config, so it sizes exactly the
run `generate` would perform.

```bash
python -m physloc.cli taxonomy --config v0_release
bash scripts/run.sh v0_release
```

### The v1 release

```bash
python -m physloc.cli taxonomy --config review_ladder
bash scripts/run.sh review_ladder
```

**The HDRI environment is the expensive dial, not the severity ladder.** It costs roughly
5.5× a plain background and arrives at L2, which is why the ladder's shares fall as realism
rises. The severity ladder is nearly free, because the valid twin and the scene build
are shared across bins.

---

## 2b. Narrowing a run

Everything below overrides whatever config you pass, so `--config review --scenario drop`
is the review settings on one scenario.

```bash
# ONE cell -- the fastest loop there is, ~14 s
python -m physloc.cli generate --config review --scenario occluder_pass --family permanence

# every family one scenario supports, in a single container run
python -m physloc.cli generate --config review --scenario drop

# one family everywhere it is meaningful
python -m physloc.cli generate --config review --family solidity

# stop after N cells, for a smoke test
python -m physloc.cli generate --config review -n 6

# the review matrix at release resolution
python -m physloc.cli generate --config review --tier release
```

`--keep-going` carries on past a cell that fails and lists the failures at the end; the
shipped configs set it.

### Parallelism

`--workers N` runs N container jobs at once. A job is one `(variant, scenario)` — so
parallelism only helps when a run spans several scenarios or several variants, and
`--scenario drop` on its own is a single job however high you set N.

**Output is byte-identical at any worker count**: each job writes its own directory, and the
per-clip rng is keyed by `(seed, family, severity)` through `crc32`, never by position in a
queue. Verified by hashing all 295 files of a run at `--workers 1` against the same run at
`--workers 4` — zero differences. Results are also printed in job order rather than
completion order, so two runs produce the same transcript.

### Config files

Every subcommand takes `--config NAME`, reading `configs/<NAME>.yaml`. **A flag you type
always beats the file.** A config is a `defaults:` block plus one block per subcommand;
keys are the long flag names with dashes as underscores (`--keep-going` is `keep_going`).
Anything both `generate` and `taxonomy` understand lives in `defaults:` once — which is why
`taxonomy --config X` prices exactly what `generate --config X` would build.

`configs/review.yaml` documents every available key and its alternatives inline. An unknown
key under a command's own block is an error, not a warning. `PHYSLOC_CONFIG=review` in the
environment does the same as passing `--config review`.

### Randomisation

**`--variants N` is the randomisation knob.** Each variant is a fresh seed, and a seed drives
every free parameter a scenario has. So six variants of `drop × solidity` are six visibly
different clips of the same violation, not one clip with a different random number in the
filename.

| axis | what varies |
|---|---|
| shape | `sphere`, `cube`, `cylinder`, `cone`, `torus` where a scenario allows it |
| material | cork, wood, plastic, rubber, ceramic, stone, steel — appearance **and** density |
| mass | `density × volume`, so a heavy object *looks* heavy; ~30× spread |
| size | drawn per instance; boxes also get independent half-extents |
| colour | actor hue within its material's band, plus **floor and backdrop** |
| camera | position and aim always; **1 variant in 5 also moves** — see below |
| timing | when the violation fires, and when the physical event it hangs on happens |
| violation | direction, magnitude and the body it acts on |

**Mass is derived, never drawn.** A randomly-drawn mass is invisible: a clip where the heavy
ball barely moves reads as a violation while being lawful physics, so the label would say
valid and the picture would say otherwise. Deriving it from a *visible* material keeps the
number and the picture in agreement.

**Camera motion is stratified across a scenario's variants, not drawn per scene.** Exactly
one variant in five moves — 2 of 10, 1 of 5 — and it is the *same* variant indices in every
scenario. An independent coin flip per scene gives the right share overall and an uneven one
per scenario: measured across thirteen it ran from 13% to 31%, so some scenarios effectively
had moving cameras and others did not, and any per-scenario comparison inherited that as a
confound.

The moving variant is the **last** of each block (index 4, 9, 14…), which is what makes a
short run entirely static: a four-variant run never reaches index 4. A run only spends clips
on camera motion once it is long enough to afford them. Three motions are drawn:

| motion | what changes | apparent size |
|---|---|---|
| `track` | slides across the view, aim held | unchanged |
| `orbit` | swings around the subject at fixed radius | unchanged |
| `dolly` | approaches or retreats, 6–12% | changes, deliberately capped |

`dolly` is capped because apparent size is the exact cue `immutability` and `deformation`
make their claim about. At 12% it is about a third of `deformation`'s weakest bin, and unlike
a deformation it scales the floor and every other body equally — so the scene still reads
"the camera moved" rather than "that object changed". There is no panning variant on purpose:
with the aim moving too, *"did the object move or did the camera?"* stops being answerable
from the clip.

`occluder_pass` never moves its camera. It precomputes its occlusion interval by intersecting
a camera→ball ray with the screen plane once, and that frame list is where every observability
label in the dataset comes from.

**Camera motion is an orthogonal axis, on 20% of every level's clips** — see [§8](#8-the-complexity-ladder). It is stratified by variant index, so a short run is the plain
baseline and never moves.

**Aspect ratio varies on boxes only**, and that is PyBullet, not a choice.
`kubric/simulator/pybullet.py` builds a `kb.Cube` from `halfExtents` (three independent
values), but asserts uniform scaling for spheres and for every mesh asset — a squashed
cylinder is a crashed render, not a subtly wrong collider.

Check it without rendering anything:

```bash
python -m physloc.cli randomisation --seeds 24      # distinct values per axis, in seconds
```

A column of `1` is an axis that is not varying. Some are legitimate — `collision` gives both
balls one material on purpose, because `newton2_mass` claims a mass ratio the image must not
justify.

To look at one camera motion without hunting for a seed that draws it:

```bash
PHYSLOC_CAMERA_MOTION=orbit python -m physloc.cli generate --debug \
    --scenario collision --family solidity
```

Values: `track`, `orbit`, `dolly`, `always`, `off`. Forwarded into the container by
`docker/kubric.sh`. **Never set it during a release run.**

One valid clip serves every family of the same scenario and seed: the twins are bit-identical
by construction, so rendering it once is both correct and cheaper. `validate` expects exactly
that shape — one `valid` and one or more `invalid` per `pair_uid`.

---

## 3. Look at what came out

**Nothing writes image files.** The container has no ffmpeg, so it writes arrays and every
mp4 is encoded host-side by `physloc/viz/video.py`.

```bash
# five-panel annotated video for ONE clip
python -m physloc.cli overlay out/release/clips/physloc_v0/drop/0777/invalid_solidity_strong

# the valid clip beside every severity bin of one family
python -m physloc.cli grid out/release/clips/physloc_v0/drop/0777 --family solidity

# ONE video tiling every invalid clip in the release -- the coverage check
python -m physloc.cli coverage out/release
```

`coverage.mp4` is the one to open first after a sweep: every cell at once, each tile captioned
with its scenario, family and observability lag, so a broken cell is obvious at a glance.

### Reading the overlay

| element | meaning |
|---|---|
| **red dot**, top right | the violation is active on *this* frame |
| **RGB** | the clip as released |
| **MASK** | red = `violation_mask` (the annotation); **green outline** = `reference_mask`, where the object *should* be per the valid twin |
| **SEVERITY MAP** | the residual painted into the culprit, with a live 0..1 scale |
| **CAUSAL MASK** | red = primary culprit, blue = other participants |
| **DIVERGENCE** | `\|valid - invalid\|` — shipped for analysis, **not** ground truth |
| red bar, timeline | `violation_windows` — when the law is actually broken |
| amber bar, timeline | `observable_windows` — when there is visual evidence |
| `t_event` / `t_obs` / `t_end` | the three clocks; they merge into one label when they coincide |

If `t_obs > t_event` the violation happened while the culprit was hidden — that gap is the
occlusion lag, and `occluder_pass` is the scenario built to produce it.

---

## 4. Command reference

Every subcommand takes `--config NAME`.

| command | what it does |
|---|---|
| `taxonomy` | print the taxonomy and **price a release** |
| `generate` | simulate + render + annotate, end to end |
| `annotate` | re-annotate a worker directory **without re-rendering** |
| `overlay` | the six-panel annotated video for one clip |
| `grid` | one family: valid vs every severity, every annotation view |
| `sheet` | one scenario: every family × every severity at once |
| `coverage` | every invalid clip in the release, tiled into one video |
| `validate` | schema + cross-checks over a whole release |
| `audit` | cells whose violation is **not visible** — candidates to drop |
| `config-path` | print the outdir a config resolves to |

```bash
# Taxonomy: 5 media, 23 families, 15 declared scenarios (13 built), 166 build cells
python -m physloc.cli taxonomy
python -m physloc.cli taxonomy -v                    # + every cell
python -m physloc.cli taxonomy --config v0_release   # + hours and clip counts

# How varied is the sampler, per axis? Renders nothing, runs in seconds.
python -m physloc.cli randomisation --seeds 24

# Package a release for distribution -- shards, index, card, splits.  Section 5b.
python -m physloc.cli export out/physloc_v0 --outdir out/hf/physloc_v0

# Re-annotate without re-rendering -- picks up any annotation change for free
python -m physloc.cli annotate out/work/drop/0777 --outdir out/release

# Videos
python -m physloc.cli overlay  out/release/clips/.../invalid_solidity_strong
python -m physloc.cli grid     out/release/clips/physloc_v0/drop/0777 --family solidity
python -m physloc.cli sheet    out/release/clips/physloc_v0/drop/0777 --view energy
python -m physloc.cli coverage out/release

python -m physloc.cli validate out/release

python -m pytest tests/ -q                    # 1012 tests, no docker needed
python -m pytest tests/test_all_cells.py -q   # plans and applies all 166 cells
```

### `generate` flags

| flag | |
|---|---|
| `--tier debug\|release` | the resolution/length ladder — section 9 |
| `--complexity L0..L3\|all` | how hard the scene is to parse — section 8. L0–L1 built |
| `--severity weak\|medium\|strong\|all` | which magnitude bins |
| `--variants N` | randomisations per cell |
| `--scenario X` / `--family Y` | restrict the matrix |
| `-n N` | stop after N cells |
| `--keep-going` | carry on past failures, list them at the end |
| `--window N` | force one violation duration; **leave unset** so each family scales with the clip |
| `--no-overlay` | skip the per-clip videos |
| `--outdir` / `--workdir` | where the release and the raw passes go |
| `--frames N` `--fps N` `--resolution N` `--spp N` | override one field of a tier — section 9 |

### The three videos

All three are laid out **wide**, and all three put every annotation view in one frame.

| video | rows | columns |
|---|---|---|
| `grid` | severity (valid, weak, medium, strong) | annotation view |
| `sheet` | annotation view | valid + every family of one scenario |
| `coverage` | scenario | family — **black where a cell is not built** |

Nine views, in the same order everywhere — `rgb`, `energy`, `seg`, `depth`, `flow`, `mask`,
`sev`, `causal`, `div` — so a grid cell, a sheet cell and an overlay panel read the same.
The order runs **evidence first** (what the renderer saw) then the **annotation** derived
from it.

Three timeline bars, keyed in every video that draws them:

| colour | window | meaning |
|---|---|---|
| **blue** | `intervening` | we are actively changing something |
| **red** | `consequence` | the scene is still wrong as a result |
| **amber** | `observable` | a viewer could tell |

A `sheet` column is the single-clip overlay turned on its side, so reading across a row
compares the same annotation over every violation at the same instant. `coverage` is a fixed
scenario × family lattice rather than a reflowed block, so a missing cell is a black square
in a known place instead of a gap the tiles close up around.

`grid --views a,b,c` restricts the columns; `sheet --severity BIN` and `coverage --severity BIN` pick which bin to show.

### Running a worker in the container directly

```bash
# throughput probe (no assets downloaded, pure render cost)
bash docker/kubric.sh physloc/render/worker_smoke.py --resolution 256 --frames 8

# the real worker: simulate + inject + render both twins.
# --family takes a comma list; they share one scene build and one valid render.
bash docker/kubric.sh physloc/render/worker.py \
    --scenario drop --seed 777 --tier debug --complexity L0 \
    --family solidity,antigravity --severity strong --outdir out/work
```

`docker/kubric.sh` mounts the repo at `/kubric`, runs as your uid so output is not
root-owned, and prefers the pinned digest over `:latest`.

---

## 5. What a clip contains

### The directory tree

Output lands under `<outdir>/clips/<release>/<level>/<scenario>/<seed>/`, gitignored so it
never enters version control. Concretely, after `--complexity all --variants 2 --severity all`:

```
out/release/
  coverage_strong.mp4                       every cell in the release, one video
  clips/
    physloc_v0/
      L0/                                   <- COMPLEXITY RUNG
        drop/                               <- SCENARIO
          0777/                             <- SEED = base seed + variant index
            valid/                          <- ONE valid twin, shared by every
            invalid_solidity_weak/             family and bin on this scene
            invalid_solidity_medium/        <- FAMILY _ SEVERITY BIN
            invalid_solidity_strong/
            invalid_continuity_weak/
            ...                                14 families x 3 bins on `drop`
            sheet_strong.mp4                every family of this scene, one frame
            grid_solidity.mp4               every severity of one family
          0778/                             <- variant 2: a different scene
            valid/
            invalid_solidity_weak/
            ...
        collision/
          0777/
          0778/
        ...                                    13 scenarios
      L1/                                   <- the same seeds, one axis changed
        drop/
          0777/                             <- pairs with L0/drop/0777
          ...
```

**The rung is part of a clip's identity.** The same seed and variant serve every rung on
purpose — that pairing is what makes "what did materials cost" a paired comparison rather
than two population averages — so the level has to be in the key, or L1 writes over L0 and
the index carries duplicate `clip_uid`s.

**How the four axes multiply.** A *cell* is one (scenario, family) pair — 166 of them, chosen
by `taxonomy.COMPATIBILITY`, which is why `drop` has 14 families and `pendulum_swing` has 8.
Each cell is rendered once per severity bin per variant:

```
clips = 166 cells x bins x variants          invalid
      + 13 scenarios x variants              valid  (one per scene, not per family)
```

So `--variants 2 --severity all` is `166×3×2 = 996` invalid + `13×2 = 26` valid = **1022
renders**. The valid twin is shared because the twins are bit-identical before `t_event` —
rendering it per family would be both wasteful and, if anything drifted, wrong.

**A variant is a fresh seed**, not a re-roll of one scene: variant *N* uses `seed + N`, and
the seed drives every free parameter (see [Randomisation](#randomisation)). So `0777/` and
`0778/` are different objects, materials, sizes, colours, floor, camera and event timing —
the same *cell*, a different *instance* of it.

### What one clip directory contains

```
meta.json            labels, taxonomy, violation windows, severity, provenance
rgb.mp4              the video
overlay.mp4          annotated: mask + severity + clocks + window bar
timelines.npz        active[T], intervening[T], consequence[T], observable[T],
                     occluded[T], severity_t[T]
violation_mask.npz   bool [T,H,W]  <- the primary annotation: union over both twins
mask_invalid.npz     bool [T,H,W]  the culprit in the INVALID render only
reference_mask.npz   bool [T,H,W]  where the culprit SHOULD be (valid twin); on both clips
causal_mask.npz      uint8 [T,H,W] 0=none, 1=culprit, 2+=participants
severity_map.npz     f16  [T,H,W]  how badly, localised in space and time
residuals.npz        r (physical units), z (vs noise floor), s (bounded [0,1])
divergence_map.npz   f16  [T,H,W]  NOT the violation region -- see below
seg.npz              uint16 [T,H,W] instance ids, stable across frames = per-object tracks
depth.npz            f32  [T,H,W,1] metres (background is a ~1e10 sentinel, mask it)
flow_fwd/bwd.npz     f32  [T,H,W,2] pixels, (row, col)
normals.npz  object_coords.npz     uint16, 0..65535
energy.npz           mechanical energy + its three anomaly channels
energy_map.npz       f32  [T,H,W]  per-body energy, painted through the segmentation
bodies.npz           mass, velocity, momentum, inertia, height, kinetic, potential
grids.npz            masks + severity reduced to the latent token grid
traj.npz             the seam file: poses, velocities, contacts
```

**Every object's mask for the whole clip** comes out of `seg.npz` in one comparison —
segmentation ids are the declared ones and are stable across frames, so they *are* the
tracks:

```python
seg  = np.load("seg.npz")["seg"]                 # [T,H,W]
bods = np.load("bodies.npz")                     # ids and names together
for bid, name in zip(bods["body_ids"], bods["body_names"]):
    track = (seg == bid)                         # [T,H,W] bool, this body, every frame
```

Field reference: [docs/schema.md](docs/schema.md).

**Three things to know before training on any of it:**

- **`divergence_map` is not the violation region.** It is `|valid − invalid|` in pixel space
  and it diverges *everywhere* downstream of the event. Use `violation_mask` and
  `severity_map`.
- **`violation_mask` is gated on visibility, not just on the window.** It answers "where can
  this be seen", so it is empty on frames where the violation is active but the culprit is
  hidden — or where the intervention has not yet moved a pixel. `timelines.active` is the
  ground-truth timeline; the gap between them is the observability lag, and it is the point.
- **Check `provenance.prefix_identical_verified`** rather than assuming twins are
  pixel-aligned.

---

## 5b. Publishing a release

`generate` writes a directory per clip, which is the right shape for producing and
inspecting them and the wrong shape for handing to anyone else. `export` turns that tree
into a dataset:

```bash
# package (local, repeatable)
python -m physloc.cli export out/physloc_v0 --outdir out/hf/physloc_v0

# package AND upload (deliberately a second step -- publishing is not repeatable)
python -m physloc.cli export out/physloc_v0 --outdir out/hf/physloc_v0 \
    --push-to <user>/physloc-v0
```

```
out/hf/physloc_v0/
  README.md                    dataset card, YAML front-matter first
  LICENSE
  taxonomy.json                what every scenario and family MEANS
  index.parquet                one row per clip, INCLUDING the video
  splits/{main,held_out,debug}.txt
  shards/<split>-*.tar         rgb + every annotation      <- the default download
  shards/<split>-passes-*.tar  depth/flow/normals/coords   <- only with --with-passes
```

### `index.parquet` — the table the hub renders

One row per clip, 26 metadata columns plus the video itself:

| group | columns |
|---|---|
| identity | `clip_uid`, `pair_uid`, `twin_uid`, `label`, `split` |
| taxonomy | `scenario`, `family`, `domain`, `medium` |
| violation | `severity_bin`, `magnitude`, `peak_severity`, `t_event_frame`, `violation_windows`, `observability_lag` |
| scene | `seed`, `variant`, `tier`, `complexity`, `n_distractors`, `num_frames`, `fps`, `camera_motion`, `actor_shape`, `actor_material`, `actor_mass` |
| text | `prompt` |
| **video** | **`rgb`** everywhere, **`overlay`** in the `debug` split |

The videos are **embedded bytes** typed as a HuggingFace `Video` feature, so the hub plays
them inline. A path would not work: it resolves on the machine that built the release and
nowhere else.

`overlay` is restricted to `debug` because it is nine panels wide and 24× the size of `rgb`
— 261 KB against 11 KB at debug geometry, about 5.3 MB against 0.2 MB at v0. Embedding it
everywhere would be 8.5 GB on a full v0 release against 0.3 GB for the RGB alone, and
`debug` is the calibration slice, which is exactly where "show me everything at once" earns
its bytes.

> `magnitude` and `peak_severity` are **not** the same quantity and are never comparable.
> `magnitude` is the knob turned, in per-family units — a volume ratio for `immutability`, an
> aspect ratio for `deformation` — and is known before simulating. `peak_severity` is the
> measured residual, z-scored against a noise floor from valid clips and bounded to `[0,1]`,
> which is what makes `friction 0.29` and `solidity 0.89` mean something side by side.

**Why the passes ship separately.** Measured on a debug sweep: `flow_fwd`, `depth` and
`object_coords` are **86% of the bytes** (~3.2 MB/clip against ~1 KB per annotation), and at
v0 geometry they are ~57× that. Someone training on `violation_mask` should not download a
hundred gigabytes of optical flow to get it.

### `taxonomy.json` — the per-scenario metadata

Generated from `physloc/taxonomy.py`, never hand-written, because five hand-copies of this
table in `docs/` already disagreed with each other and with the code. Per scenario:
`description`, `event_structure`, `physics_medium`, `grounded_in` (the prior-art scenario it
comes from), `has_occluder`, `provides` (the capabilities it offers injectors), the
`families` staged on it, and `clips_in_release`. Per family: `domain`, `law`,
`magnitude_unit`, `kind`, `detectable`, `graded`, `requires`, and its `intphys2` / `likephys`
mapping.

### The splits, and what they are for

Modelled on **IntPhys 2** (arXiv:2506.09849), which this project already takes its
debug/artifact split from. That paper releases 1416 videos as Debug (5 scenes, calibration),
Main (253 scenes, **with** metadata) and Held-Out (86 scenes, **without** metadata, "to avoid
training data contamination") — counted in *scenes*, not videos.

| split | share of pairs | ships |
|---|---|---|
| `main` | 75% | video + every annotation |
| `held_out` | 20% | video + every annotation |
| `debug` | 5% | video + every annotation, **plus the overlay video** |

**A release too small to split goes entirely to `debug`.** With fewer pairs than splits there
is nothing to hold out, and `debug` is the one that carries the overlay — so a release too
small to be a benchmark becomes the thing it can actually be: something to look at.

**Every split ships everything.** The split is a *label*, not a filter — so you can re-cut
the boundary later and you can score any clip in the release. IntPhys 2 withholds its
held-out metadata, and for a leaderboard other people submit to that is the right call; it
is the wrong one for a release that still has to be re-split and measured. Stripping
annotations is something you can do at publication time from `splits/held_out.txt` without
regenerating anything — putting the pairs on the wrong side of a boundary is not.

There is deliberately **no `train` split**. LikePhys (arXiv:2510.11512) does not split at
all — it is a training-free evaluator doing pairwise valid-versus-invalid comparison — and
PhysLoc's primary use is the same. Naming a split `train` would imply the opposite.

Three properties, each pinned by a test in `tests/test_export.py`:

1. **Grouped by pair, never by clip.** A valid twin and its invalid siblings are
   bit-identical up to `t_event`. Split them apart and the training set contains every frame
   of the test clip before the violation — the answer, in other words.
2. **Stratified within each scenario.** Cutting the whole population in one pass lets a
   scenario land entirely in one split, and then the held-out set measures *"have you seen
   `pour` before"* rather than *"do you understand pouring"*.
3. **Shards are written per split**, so one can be fetched without the others — but with
   identical contents, so the grouping above is the only thing that has to be decided up
   front.

Assignment is by hash of the `pair_uid`, ordered and cut at the quantiles — no rng, so
regenerating a release reproduces its splits exactly. **Adding clips moves the boundaries**,
so a release that grows should be re-split and re-reported rather than appended to.

> **A release needs several variants per scenario before it can be split.** With one pair
> per scenario there is nothing to hold out — three-way stratification of a single pair is
> impossible — and `export` says so in its summary rather than quietly shipping an empty
> held-out set.

---

## 6. How it is organised

Four levels — `python -m physloc.cli taxonomy` prints the live version, which is the
authority if this ever drifts from the counts below:

```
DOMAIN     8   which physical law is at stake      identity, kinematics, contact, dynamics, equilibrium, optical, appearance, global
FAMILY    23   the specific way it breaks          solidity, fission, colour_shift, ...
SCENARIO  15   the staged scene                    drop, occluder_pass, pour, ... (14 built; clutter_toss is deferred)
CELLS    166   scenario x family combinations actually built
INSTANCE       scenario x family x seed x severity -> one valid/invalid pair
```

| scenario | what it stages | why it is in the set |
|---|---|---|
| `drop` | sphere or cube falls and bounces | simplest scenario with a real contact instant |
| `collision` | two spheres roll together | two bodies that both *ought* to respond |
| `toss` | ballistic arc, no contact | the clean control: `t_event == t_observable` |
| `tumble` | cube tumbling in free flight | a sphere cannot show rotation |
| `occluder_pass` | body passes behind a screen | the only source of observability lag |
| `barrier_pass` | ball rolls into a solid wall, rebounds | a second, walled `solidity` host — `occluder_pass`'s only surface is the floor, so a disabled collision there reads as a vanish, not a pass-through |
| `ramp_slide` | block slides down an incline | sustained contact + friction |
| `rolling_ramp` | cube tumbles off a raised ramp | contact, then a short free flight |
| `stack_topple` | marginally stable stack | *surprising but lawful* — the control for false positives |
| `pyramid_impact` | cube dropped on a sphere pyramid | multi-body contact chain |
| `pendulum_swing` | bob on a rigid rod | constrained periodic motion (scripted; Kubric has no joints) |
| `resting_table` | bodies at rest on a table | static equilibrium; any motion is the violation |
| `shadow_track` | object translating under a key light | the only violation whose mask is not on the object |
| `pour` | 40 grains falling into an open box | **granular, never "fluid"** — see below |
| `clutter_toss` *(deferred)* | MOVi-style multi-object toss | declared in the taxonomy for its compatibility cells; no scenario file yet |

Two orthogonal augmentation axes:

- **severity** — `weak` / `medium` / `strong`, set by the *intervention magnitude*, which is
  exact by construction. Named for how hard the law is bent, not for how hard the clip is to
  classify — those are different things, and conflating them is how difficulty splits go bad.
- **complexity** — `L0`..`L3`, **one axis per rung**, with camera motion and distractors as
  orthogonal ratios inside each. See
  [§8](#8-the-complexity-ladder). L0–L1 are built; L2–L3 **raise if requested** rather
  than silently degrading.

**Why 22 files and not 345.** Scenarios and injectors are orthogonal and compose through the
trajectory seam — an injector edits `traj.npz`, per-body poses and velocities, which knows
nothing about the scene that produced it. So the project needs 13 scenario files + 8 injector
files (one per domain that owns a family — `global` shares its machinery with `kinematics`,
since `global_gravity` is `antigravity` turned up to the whole scene) + one shared geometry
helper, not one file per scenario × family pair. `antigravity` written once runs on every
airborne scenario; adding a scenario makes every compatible family available in it for free.

The corollary is that injectors must never branch on a scenario's *name*. They branch on
**state** — is this body moving, is it in contact, is it hidden — or on a **constraint the
scenario declares** in `spec.notes`. `newton1_inertia` halts a sliding block and shoves a
resting mug from one code path; `angular_momentum` asks the scenario to re-solve its own
pendulum arc rather than knowing what a pendulum is.

### No fluid at v0, and why

Tested rather than assumed: Blender 2.93.4 in the pinned image ships Mantaflow, but headless
scripted baking fails (`NameError: liquid_save_data_N` → `Manta::Error`), Kubric exposes no
fluid object, and a liquid's per-frame mesh state does not fit a pose-based trajectory seam.
`pour` is the v0 stand-in — a few dozen rigid grains that stream, pile and break up —
and it is labelled `physics_medium: "granular"`. `physloc validate` rejects any clip
claiming `"fluid"`. Real fluid and cloth are **Phase 3**, behind a newer Blender.

---

## 7. Orthogonality — what the labels guarantee

A clip labelled `solidity` has to contain solidity **and nothing else**, or the dataset
cannot support the claim "this model misses X". That is not free — it broke four separate
times while the injectors were being written, always the same way: an injector re-integrates a
body, the integrator does not know about the walls, and a `global_gravity` clip quietly
becomes a clip about solidity too.

So it is asserted. `taxonomy.EXCLUSIVE_LAWS` names the residuals with a clean zero baseline on
a lawful clip and the families entitled to move each one; `tests/test_orthogonality.py` fails
if any other family moves one, and fails again if an owner does *not* move the law it owns —
a tripwire nobody trips measures nothing.

```bash
python -m pytest tests/test_orthogonality.py -q     # every cell, no docker, ~10 s
```

**Where the guarantee stops.** Families with an exclusive tripwire are provably clean. The
rest are separated by *staging*, not by residual: `antigravity`, `phantom_impulse`,
`newton1_inertia` all move `linear_momentum`, because they must —
bend a body's gravity and its momentum residual moves with it. Physics is not separable there
and pretending otherwise would be the wrong fix. What separates them is the situation, which a
model has to read from the image. The dataset card written by `physloc export` says which is
which, and what that means for a confusion matrix.

## 8. The complexity ladder

**Severity asks how badly the law is broken. Complexity asks how hard the scene is to
parse.** Two independent axes; reporting across both is what separates "understands physics"
from "copes with clutter".

**The ladder is SCENE REALISM — four rungs, each the one below it plus one step.**

| level | adds | background | actors | surface | share | built |
|---|---|---|---|---|---|---|
| **L0** | *baseline* | solid | primitives | flat colour | **1.00** (50%) | ✅ |
| **L1** | materials | solid | primitives | **wood/steel/…** | 0.50 (25%) | ✅ |
| **L2** | HDRI environment | **hdri** | primitives | materials | 0.30 (15%) | ✗ |
| **L3** | GSO objects | hdri | **gso** | materials | 0.20 (10%) | ✗ |

### Camera motion and distractors are NOT rungs

They are **orthogonal axes applied inside every level** at declared ratios:

| axis | share of each level's clips | fires on variants (of 10) |
|---|---|---|
| moving camera | **20%** | 4, 9 |
| distractors (6, or 12 at L3) | **30%** | 3, 6, 9 |

They used to be rungs, and that was wrong twice over. A rung whose axis fires on only *some*
of its clips is not a stratum at all: with camera motion on 1 variant in 5, "L2 minus L1" was
not measuring materials, it was measuring materials plus whichever variants happened to draw
a camera move. And making them rungs forced a choice nobody wants — either every realistic
clip moves its camera, or none does.

As ratios, the dataset answers *"what does clutter cost at each realism level"* as well as
*"what does realism cost"*, and the overall share of moving-camera clips is a number you set
rather than an artefact of how the ladder was climbed.

### How the shares work

- **A run of `--complexity all` walks the built ladder**, giving each level
  `round(variants × share)` variants per cell. `--variants 10` → L0 ×10, L1 ×5.
- **A level that does not buy a whole variant is skipped.** `--variants 1` is L0 only. That
  is deliberate: a short run should be the easy case. **Naming a level explicitly overrides
  it** — `--complexity L2 --variants 1` gives you L2.
- **The orthogonal axes work the same way.** `floor(V × share)` clips get the axis, spread
  evenly across the variant *indices* rather than drawn per scene — so every scenario gets
  the same share instead of each flipping its own coin (measured across thirteen scenarios,
  independent draws ranged from 13% to 31%). Below 5 variants nothing moves; below 4 nothing
  is cluttered. `PHYSLOC_CAMERA_MOTION=orbit` forces one on demand.
- **The same seed and variant index serve every rung**, so each L1 clip has an L0 counterpart
  built from the same draw with one axis changed — a paired comparison, not two population
  averages.
- **Shares fall as realism rises** for two reasons: the baseline is what everything else is
  compared against so it should be the largest stratum, and an HDRI clip costs ~44 s against
  a solid background's ~8 s at the debug tier.

Everything above lives in `COMPLEXITY` in `physloc/scenarios/base.py` and nowhere else;
`python -m physloc.cli taxonomy --complexity all --variants 10` prices it per rung.

Three more things worth knowing:

- **A distractor never joins the physics.** It is placed clear of the actor's whole predicted
  path, never between the actor and the camera, and always inside the frame; it carries
  `role="distractor"`, which every injector query excludes, and `_geom.support_under` refuses
  it as a resting surface. `meta.json` records `n_distractors` — what actually went in, since
  the constraints can leave no room and a clip claiming six while containing four is a clip
  whose metadata lies.
- **Below L1 every object shares one density.** Mass is `density × volume`, so a level
  without materials would have mass varying *invisibly* — the confound materials exist to
  remove. With one density mass varies with **size**, which a viewer can see.
- **L0 and L1 are therefore not the same physics.** Materials change mass, so any "same
  physics, harder scene" pairing needs two levels on the same side of L1.

### Testing one level at a time

Each rung adds exactly one thing, so rendering them separately is how you find out *which*
thing broke. There is a config per rung, so the short form is:

```bash
for L in L0 L1 L2 L3; do
  python -m physloc.cli generate --config review_$L --scenario drop
done
```

or, through `run.sh` (which also packages the result for the hub):

```bash
for L in L0 L1 L2 L3; do
  bash scripts/run.sh review --complexity $L --scenario drop --variants 10 \
       --outdir out/$L --workdir out/work_$L
done
```

**Use ten variants when you want to see the orthogonal axes.** They are stratified by variant
index, so a run of one is entirely static and uncluttered — correct behaviour, but it means a
short run tells you nothing about the camera or the distractors. Ten gives two moving and
three cluttered clips per cell.

| level | what to look for | what would be wrong |
|---|---|---|
| **L0** | flat colours, one shared density, primitives on a solid ground | a material appearing; mass varying without size varying |
| **L1** | wood looks like wood, steel like steel; heavy things behave heavy | a material whose mass does not match its look |
| **L2** | a real environment, lit from an HDRI | *not built* — the config raises |
| **L3** | GSO objects in that environment | *not built* |
| *variants 4, 9* | the camera orbits, tracks or dollies; the actor stays in shot | motion on the wrong variant, or the actor drifting out of frame |
| *variants 3, 6, 9* | six distractors, in shot, clear of the action | one touching the actor, hiding it, or off-frame |

The severity of a given cell should be **identical with and without distractors** — they are
decoration, and if a number moves, one of them is taking part in the physics. Variants 2 and
3 of the same cell differ by exactly that:

```bash
python -m physloc.cli generate --config review_L0 --scenario drop --family solidity \
    --variants 4 --outdir out/cmp --workdir out/work_cmp
```

Compare the `sev=` on the variant-2 and variant-3 lines. That comparison is what caught a
distractor redefining `support`'s reference surface, which moved the score from 0.91 to 0.46.

### Generating the whole ladder in one run

`--complexity all` produces every rung in its declared proportion, which is what a release
is:

```bash
python -m physloc.cli taxonomy --config review_ladder    # price it, per rung
python -m physloc.cli generate --config review_ladder    # L0 x10, L1 x5
```

Each rung renders into its own subdirectory of the work tree, because the same seed serves
every rung by design — that pairing is the point, and without separate scratch L1 would
render over L0's passes.

### Publishing a level, or a whole run

`run.sh` packages every run into `out/hf/<name>/`. Set one environment variable and it
uploads too:

```bash
# one level, to its own dataset
PHYSLOC_PUSH_TO=<user>/physloc-l1 \
  bash scripts/run.sh review --complexity L1 --scenario drop --variants 10 \
       --outdir out/L1 --workdir out/work_L1

# a mini sample, private
PHYSLOC_PUSH_TO=<user>/physloc-mini PHYSLOC_PUSH_PRIVATE=1 \
  bash scripts/run.sh review -n 45 --variants 10 \
       --outdir out/physloc_mini --workdir out/work_mini

# every level as its own dataset
for L in L0 L1; do
  PHYSLOC_PUSH_TO=<user>/physloc-${L,,} \
    bash scripts/run.sh review --complexity $L --scenario drop --variants 10 \
         --outdir out/$L --workdir out/work_$L
done
```

| variable | effect |
|---|---|
| `PHYSLOC_PUSH_TO` | dataset repo id, e.g. `samueleruf/physloc-mini`. Created if absent |
| `PHYSLOC_PUSH_PRIVATE` | set to anything to create it private |

Packaging always happens; **uploading only when `PHYSLOC_PUSH_TO` is set**. That split is
deliberate: packaging is local and repeatable, uploading is neither — it puts the clips
somewhere other people can fetch, index and cache them.

Without the variable, or to publish a run you already have:

```bash
python -m physloc.cli export out/L1 --outdir out/hf/L1 --push-to <user>/physloc-l1
```

**A push replaces the card and index at that repo id**, so use a distinct name per artefact
rather than overwriting a release with a sample. What lands: `README.md` (the card),
`index.parquet` with the videos playable inline, `taxonomy.json`, `splits/`, `LICENSE`, and
the shards. See [§5b](#5b-publishing-a-release).

## 9. Tiers

A tier is a **geometry** — how big and how long. Nothing else.

| | `debug` | `release` |
|---|---|---|
| resolution | 128² | 512² |
| frames @ fps | 25 @ 12 | 89 @ 30 |
| duration | 2.08 s | 2.97 s |
| latent grid | 7×8×8 | 23×16×16 |
| cost | ~8 s/clip | ~637 s/clip |
| published | never | yes |

**`v0` and `v1` are not tiers.** They were, and differed in nothing but their name — what
actually separated them was complexity, which is [its own ladder](#8-the-complexity-ladder).
A tier that encodes a release number has to be renamed every release. So: the tier says how
big, the ladder says how hard, and `v0`/`v1` are what a published **dataset** is called — set
by `--outdir`, recorded as `release` in every `meta.json`.

Difficulty is deliberately not a resolution step: a model scoring worse on a harder release
could otherwise be failing at realism, at clutter, or merely at an unfamiliar render size.
Fixing the geometry is what makes two releases comparable.

**`release` is 30 fps** so it downsamples cleanly to 15 and 10, which a 12 fps master cannot
do. 89 frames rather than 90 because every frame count must be `4k+1` for exact VAE latent
alignment, and 89 is the nearest that is.

Frame counts are all `4k+1` so the token-grid reduction aligns exactly with a video-DiT VAE's
4× temporal binning. **`debug` is the default** — iterate there.

The letters this project used to use (A/B/D) are retired: there was no C, and the ordering
the alphabet implies ran backwards from the one that matters. `scenarios.base.tier()` and
`generate` both recognise the old letters and say what each was renamed to.

**Any single dial can be overridden without inventing a tier.** The name records what
changed, so `release+res128f25` never gets confused with `release` in `meta.json`:

```bash
python -m physloc.cli generate --tier release --frames 25 --resolution 128
```

| flag | overrides | note |
|---|---|---|
| `--frames N` | clip length | **must be `4k+1`** — 13, 17, 21, 25, 29… A bad value is refused with the nearest two legal ones |
| `--fps N` | frame rate | violation windows are a *fraction* of the clip, so they scale with it |
| `--resolution N` | square render size | render time is roughly linear in pixel count |
| `--spp N` | Cycles samples/pixel | frame time ≈ `1.29 + 0.0074·spp` at 256², so only ~26% is sampling |

**Rendering is CPU and stays that way.** Frame time fits `T = 1.29 + 0.0074·spp` at 256², so
only ~26% is sampling — the only part OptiX would accelerate, capping a GPU build at ~1.37×
by Amdahl. Clip-level parallelism measures **1.92×** at width 4, for free. Details in
[docs/PLAN.md](docs/PLAN.md) Part 0.

---

## 10. Repo layout

```
docs/PLAN.md          the design document -- start here
docs/schema.md        meta.json field reference
environment.yml       host conda env
docker/               kubric.sh wrapper + pinned image digest
scripts/run.sh        generate + validate + every video, from a config
scripts/fetch_refs.sh pinned read-only Kubric checkout -> refs/
configs/*.yaml        run settings: review, review_L0..L3, review_ladder, v0_release
physloc/
  taxonomy.py         Part 2 as data: domains, families, scenarios, compatibility
  scenarios/          seeded scene samplers (declarative SceneSpec, no Kubric import)
                      _common.py ground/lights/ramps/understudies; _hdri.py environments
  sim/trajectory.py   THE SEAM -- container writes it, host reads it
  injectors/          trajectory-level interventions, one file per domain
                      _geom.py the shared vocabulary they ask questions in
  residuals/laws.py   physical-law residuals, one per family
  render/worker.py    container-side: simulate + inject + render every variant
  annotate/           windows, masks, severity, grids, meta -> the released layout
  viz/                overlay.mp4, grid.mp4, coverage.mp4, one shared encoder
  schema/validate.py  cross-checks
  cli.py
tests/                prefix identity, mask union, windows, grids, taxonomy,
                      mockroll.py + test_all_cells.py (all 180 cells, no docker)
out/                  all generated output (gitignored)
```

## 11. Where it is going

[docs/roadmap.md](docs/roadmap.md) — the medium axis that maps onto LikePhys, the perceptual
families still missing from v0 (`illumination_shift`), and what v1 is:
population, a realistic twin of every clip, and deeper randomisation.

## 12. Evaluating on it

Report **per family (23)**, aggregate to **domain (8)**, and cross with **severity** and
**camera motion**. `index.parquet` from `physloc export` carries every one of those fields
per clip, so a breakdown is a groupby rather than a crawl over 1500 `meta.json` files.

Three things that will bite otherwise:

- **Do not pool families into one number.** Cell counts are uneven by **sixteen times** —
  `identity` has 48 build cells, `optical` has 3 — so a pooled score is largely a measurement
  of `identity`.
- **Never train on `divergence_map`.** It is `|valid − invalid|` and diverges everywhere
  downstream of the event, so a model trained on it learns to find the edit, not the physics.
  Train on `violation_mask` and `severity_map`.
- **`violation_mask` is not `timelines.active`.** The mask answers *where can this be seen*
  and is empty while the culprit is hidden; `active` is the unhedged truth about when the law
  is broken. Use `active` for temporal metrics, the mask for spatial ones.

## 13. Papers

[IntPhys 2](https://arxiv.org/abs/2506.09849) · [LikePhys](https://arxiv.org/abs/2510.11512) ·
[Kubric](https://github.com/google-research/kubric). PDFs of the first two live in
`context/` (gitignored — see [context/README.md](context/README.md)).
