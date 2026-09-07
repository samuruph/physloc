# PhysViol

**A spatio-temporally annotated physics-violation video dataset.** Every clip that breaks a
physical law ships with *where in the frame*, *exactly when*, *for how long*, and *how badly*
— all derived from the simulator, not from human annotation.

> **Status: every build cell generates, annotates and validates.** 13 built scenarios × 23
> violation families compose through the trajectory seam into **166 cells**, and
> `physviol generate` walks the whole matrix in one command. Each clip depicts *one*
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
conda activate physviol

# The render image (digest pinned in docker/IMAGE_DIGEST)
docker pull kubricdockerhub/kubruntu

# Optional: read-only Kubric source for reference (gitignored)
bash scripts/fetch_refs.sh
```

**Two environments, never mixed.** The pinned container holds Kubric 2022.4.1 / Blender
2.93.4 / PyBullet (Python 3.9) and does simulation + rendering. The `physviol` conda env
(Python 3.11) does everything else. They meet at the trajectory seam (`traj.npz`).

---

## 2. The runs

Three configs, one command each. `conda activate physviol` first.

| | config | what it is | cost |
|---|---|---|---|
| **review** | `configs/review.yaml` | every cell once, `strong`, 128², 25 f | **~25 min** |
| **v0** | `configs/v0_release.yaml` | 512², 30 fps, 89 f (2.97 s), all three bins, 3 variants | ~6.4 days |
| **v1** | `configs/v1_release.yaml` | **same geometry as v0**, photographic (L1) + multi-object | weeks |

### The review sweep — run this before anything else

Every scenario × family cell once, at debug size, `strong` only. This is the run to look at
when you want to know whether the dataset is right.

```bash
bash scripts/run.sh
```

That generates, validates, and builds every video: `coverage.mp4`, a `sheet` per scenario and
a `grid` per family. Then open `out/release/coverage.mp4` first.

The same thing by hand, if you want the steps separately:

```bash
python -m physviol.cli generate --config review
python -m physviol.cli validate out/release
python -m physviol.cli coverage out/release
```

### The v0 release

**Price it before you start it** — `taxonomy` reads the same config, so it sizes exactly the
run `generate` would perform.

```bash
python -m physviol.cli taxonomy --config v0_release
bash scripts/run.sh v0_release
```

### The v1 release

```bash
python -m physviol.cli taxonomy --config v1_release
bash scripts/run.sh v1_release
```

**Complexity is the expensive dial, not the severity ladder.** L1's photographic environment
costs roughly 5.5× an L0 render, which is the whole difference between v0 and v1's wall
clock. The ladder is nearly free, because the valid twin and the scene build are shared
across bins.

---

## 2b. Narrowing a run

Everything below overrides whatever config you pass, so `--config review --scenario drop`
is the review settings on one scenario.

```bash
# ONE cell -- the fastest loop there is, ~14 s
python -m physviol.cli generate --config review --scenario occluder_pass --family permanence

# every family one scenario supports, in a single container run
python -m physviol.cli generate --config review --scenario drop

# one family everywhere it is meaningful
python -m physviol.cli generate --config review --family solidity

# stop after N cells, for a smoke test
python -m physviol.cli generate --config review -n 6

# the review matrix at release resolution
python -m physviol.cli generate --config review --tier v0
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
key under a command's own block is an error, not a warning. `PHYSVIOL_CONFIG=review` in the
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
| camera | position and aim; and on ~20% of clips it **moves** (`track`, `orbit`, `dolly`) |
| timing | when the violation fires, and when the physical event it hangs on happens |
| violation | direction, magnitude and the body it acts on |

**Mass is derived, never drawn.** A randomly-drawn mass is invisible: a clip where the heavy
ball barely moves reads as a violation while being lawful physics, so the label would say
valid and the picture would say otherwise. Deriving it from a *visible* material keeps the
number and the picture in agreement.

**Aspect ratio varies on boxes only**, and that is PyBullet, not a choice.
`kubric/simulator/pybullet.py` builds a `kb.Cube` from `halfExtents` (three independent
values), but asserts uniform scaling for spheres and for every mesh asset — a squashed
cylinder is a crashed render, not a subtly wrong collider.

Check it without rendering anything:

```bash
python -m physviol.cli randomisation --seeds 24      # distinct values per axis, in seconds
```

A column of `1` is an axis that is not varying. Some are legitimate — `collision` gives both
balls one material on purpose, because `newton2_mass` claims a mass ratio the image must not
justify.

To look at one camera motion without hunting for a seed that draws it:

```bash
PHYSVIOL_CAMERA_MOTION=orbit python -m physviol.cli generate --debug \
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
mp4 is encoded host-side by `physviol/viz/video.py`.

```bash
# five-panel annotated video for ONE clip
python -m physviol.cli overlay out/release/clips/physviol_v0/drop/0777/invalid_solidity_strong

# the valid clip beside every severity bin of one family
python -m physviol.cli grid out/release/clips/physviol_v0/drop/0777 --family solidity

# ONE video tiling every invalid clip in the release -- the coverage check
python -m physviol.cli coverage out/release
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
python -m physviol.cli taxonomy
python -m physviol.cli taxonomy -v                    # + every cell
python -m physviol.cli taxonomy --config v0_release   # + hours and clip counts

# How varied is the sampler, per axis? Renders nothing, runs in seconds.
python -m physviol.cli randomisation --seeds 24

# Package a release for distribution -- shards, index, card, splits.  Section 5b.
python -m physviol.cli export out/physviol_v0 --outdir out/hf/physviol_v0

# Re-annotate without re-rendering -- picks up any annotation change for free
python -m physviol.cli annotate out/work/drop/0777 --outdir out/release

# Videos
python -m physviol.cli overlay  out/release/clips/.../invalid_solidity_strong
python -m physviol.cli grid     out/release/clips/physviol_v0/drop/0777 --family solidity
python -m physviol.cli sheet    out/release/clips/physviol_v0/drop/0777 --view energy
python -m physviol.cli coverage out/release

python -m physviol.cli validate out/release

python -m pytest tests/ -q                    # 1012 tests, no docker needed
python -m pytest tests/test_all_cells.py -q   # plans and applies all 166 cells
```

### `generate` flags

| flag | |
|---|---|
| `--tier debug\|v0\|v1` | the resolution/length ladder — section 8 |
| `--complexity L0\|L1` | L0 solid background, L1 photographic HDRI (~5.5× the cost) |
| `--severity weak\|medium\|strong\|all` | which magnitude bins |
| `--variants N` | randomisations per cell |
| `--scenario X` / `--family Y` | restrict the matrix |
| `-n N` | stop after N cells |
| `--keep-going` | carry on past failures, list them at the end |
| `--window N` | force one violation duration; **leave unset** so each family scales with the clip |
| `--no-overlay` | skip the per-clip videos |
| `--outdir` / `--workdir` | where the release and the raw passes go |
| `--frames N` `--fps N` `--resolution N` `--spp N` | override one field of a tier — section 8 |

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
bash docker/kubric.sh physviol/render/worker_smoke.py --resolution 256 --frames 8

# the real worker: simulate + inject + render both twins.
# --family takes a comma list; they share one scene build and one valid render.
bash docker/kubric.sh physviol/render/worker.py \
    --scenario drop --seed 777 --tier debug --complexity L0 \
    --family solidity,antigravity --severity strong --outdir out/work
```

`docker/kubric.sh` mounts the repo at `/kubric`, runs as your uid so output is not
root-owned, and prefers the pinned digest over `:latest`.

---

## 5. What a clip contains

### The directory tree

Output lands under `<outdir>/clips/<release>/<scenario>/<seed>/`, gitignored so it never
enters version control. Concretely, after `--variants 2 --severity all`:

```
out/release/
  coverage_strong.mp4                       every cell in the release, one video
  clips/
    physviol_v0/
      drop/                                 <- SCENARIO
        0777/                               <- SEED = base seed + variant index
          valid/                            <- ONE valid twin, shared by every
          invalid_solidity_weak/               family and bin on this scene
          invalid_solidity_medium/          <- FAMILY _ SEVERITY BIN
          invalid_solidity_strong/
          invalid_continuity_weak/
          ...                                  14 families x 3 bins on `drop`
          sheet_strong.mp4                  every family of this scene, one frame
          grid_solidity.mp4                 every severity of one family
        0778/                               <- variant 2: a different scene
          valid/
          invalid_solidity_weak/
          ...
      collision/
        0777/
        0778/
      ...                                      13 scenarios
```

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
python -m physviol.cli export out/physviol_v0 --outdir out/hf/physviol_v0

# package AND upload (deliberately a second step -- publishing is not repeatable)
python -m physviol.cli export out/physviol_v0 --outdir out/hf/physviol_v0 \
    --push-to <user>/physviol-v0
```

```
out/hf/physviol_v0/
  README.md                    dataset card, YAML front-matter first
  LICENSE
  taxonomy.json                what every scenario and family MEANS
  index.parquet                one row per clip (JSONL if pyarrow is absent)
  splits/{main,held_out,debug}.txt
  shards/<split>-*.tar         rgb + every annotation      <- the default download
  shards/<split>-passes-*.tar  depth/flow/normals/coords   <- only with --with-passes
```

**Why the passes ship separately.** Measured on a debug sweep: `flow_fwd`, `depth` and
`object_coords` are **86% of the bytes** (~3.2 MB/clip against ~1 KB per annotation), and at
v0 geometry they are ~57× that. Someone training on `violation_mask` should not download a
hundred gigabytes of optical flow to get it.

### `taxonomy.json` — the per-scenario metadata

Generated from `physviol/taxonomy.py`, never hand-written, because five hand-copies of this
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
| `debug` | 5% | video + every annotation |

**Every split ships everything.** The split is a *label*, not a filter — so you can re-cut
the boundary later and you can score any clip in the release. IntPhys 2 withholds its
held-out metadata, and for a leaderboard other people submit to that is the right call; it
is the wrong one for a release that still has to be re-split and measured. Stripping
annotations is something you can do at publication time from `splits/held_out.txt` without
regenerating anything — putting the pairs on the wrong side of a boundary is not.

There is deliberately **no `train` split**. LikePhys (arXiv:2510.11512) does not split at
all — it is a training-free evaluator doing pairwise valid-versus-invalid comparison — and
PhysViol's primary use is the same. Naming a split `train` would imply the opposite.

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

Four levels — `python -m physviol.cli taxonomy` prints the live version, which is the
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
- **complexity** — `L0`..`L4`, mirroring the MOVi ladder. **L0 (solid background + sun) and
  L1 (photographic HDRI environment + dome ground) are built; L1 is the default.** L2–L4 add
  GSO objects, distractors and camera motion, and **raise if requested** rather than silently
  degrading. GSO (1033 objects) and HDRI Haven (509 environments) are verified reachable from
  the pinned container, so those levels need scenario code only.

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
and it is labelled `physics_medium: "granular"`. `physviol validate` rejects any clip
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
model has to read from the image. The dataset card written by `physviol export` says which is
which, and what that means for a confusion matrix.

## 8. Tiers

| | `debug` | `v0` (`physviol_v0`) | `v1` (`physviol_v1`) |
|---|---|---|---|
| resolution | 128² | 512² | 512² |
| frames @ fps | 25 @ 12 | 89 @ 30 | 89 @ 30 |
| duration | 2.08 s | 2.97 s | 2.97 s |
| latent grid | 7×8×8 | 23×16×16 | 23×16×16 |
| complexity | `L0` | `L0` | **`L1`** |
| population | single | single | **single + multi** |
| published | never | yes | yes |

**v0 and v1 are the same tier geometry on purpose.** v1 is not a bigger render — it is the
same physics under harder conditions: photographic backgrounds and objects, and crowded
scenes. Making it a resolution step as well would confound the axes, since a model scoring
worse on v1 could be failing at realism, at clutter, or merely at an unfamiliar resolution.
Fixing the geometry is what makes the two **paired**. See [docs/roadmap.md](docs/roadmap.md).

**v0 is 30 fps** so the release downsamples cleanly to 15 and 10 without resampling, which a
12 fps master cannot do. 89 frames rather than 90 because every frame count must be `4k+1`
for exact VAE latent alignment, and 89 is the nearest that is.

Frame counts are all `4k+1` so the token-grid reduction aligns exactly with a video-DiT VAE's
4× temporal binning. **`debug` is the default** — iterate there.

The letters this project used to use (A/B/D) are retired: there was no C, and the ordering
the alphabet implies ran backwards from the one that matters. `scenarios.base.tier()` and
`generate` both recognise the old letters and say what each was renamed to.

**Any single dial can be overridden without inventing a tier.** The name records what
changed, so `v0+res128f25` never gets confused with `v0` in `meta.json`:

```bash
python -m physviol.cli generate --tier v0 --frames 25 --resolution 128
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

## 9. Repo layout

```
docs/PLAN.md          the design document -- start here
docs/schema.md        meta.json field reference
environment.yml       host conda env
docker/               kubric.sh wrapper + pinned image digest
scripts/run.sh        generate + validate + every video, from a config
scripts/fetch_refs.sh pinned read-only Kubric checkout -> refs/
configs/*.yaml        run settings: review, v0_release, v1_release
physviol/
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

## 10. Where it is going

[docs/roadmap.md](docs/roadmap.md) — the medium axis that maps onto LikePhys, the perceptual
families still missing from v0 (`illumination_shift`), and what v1 is:
population, a realistic twin of every clip, and deeper randomisation.

## 11. Evaluating on it

Report **per family (23)**, aggregate to **domain (8)**, and cross with **severity** and
**camera motion**. `index.parquet` from `physviol export` carries every one of those fields
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

## 12. Papers

[IntPhys 2](https://arxiv.org/abs/2506.09849) · [LikePhys](https://arxiv.org/abs/2510.11512) ·
[Kubric](https://github.com/google-research/kubric). PDFs of the first two live in
`context/` (gitignored — see [context/README.md](context/README.md)).
