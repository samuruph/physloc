# PhysLoc

**A physics-violation video dataset where every invalid clip ships *where* the violation is,
*when* it happens, and *how badly* — derived from the simulator, not annotated by hand.**

Most intuitive-physics benchmarks ask one bit per clip: *is this possible?* PhysLoc asks what a
model should be able to answer if it understands the scene — **which object is wrong, in which
pixels, during which frames, and by how much.** The labels come out of the simulator that
produced the violation, so they are exact and free.

- **Twins** — every invalid clip has a valid twin, bit-identical up to the violation.
- **Dense labels** — per-pixel violation masks, severity maps, causal masks and timelines.
- **A full taxonomy** — violation families grouped by domain, staged across scenarios, at three
  severities.
- **Two independent difficulty axes** — scene realism (L0–L3) and five named conditions — plus a
  measured easy / moderate / hard label per clip.

## Contents

- [Quick start](#quick-start)
- [The dataset](#the-dataset)
- [Taxonomy](#taxonomy)
- [Severity](#severity)
- [Complexity ladder](#complexity-ladder)
- [Difficulty conditions](#difficulty-conditions)
- [Detection difficulty](#detection-difficulty)
- [Scene variation](#scene-variation)
- [Splits, index and evaluation](#splits-index-and-evaluation)
- [Generating data](#generating-data)
- [Running the full release](#running-the-full-release)
- [Cost and performance](#cost-and-performance)
- [Publishing](#publishing)
- [Development](#development)
- [References](#references)

---

## Quick start

**Requirements:** Linux, Docker and conda. Rendering is CPU-only — the more physical cores the
better — and a release run wants about 2 GB of RAM per vCPU.

```bash
# Host environment: annotation, severity, validation, visualisation.
conda env create -f environment.yml
conda activate physloc

# Simulation and rendering run inside the pinned Kubric image.
docker pull kubricdockerhub/kubruntu      # digest pinned in docker/IMAGE_DIGEST
```

Generate a small review sweep and look at it:

```bash
python -m physloc.cli taxonomy --config review_severity   # what it produces and how long it takes
bash scripts/run.sh review_severity                       # generate, validate, visualise, package
```

`run.sh` ends by listing what to open, starting with `coverage_strong.mp4`: every cell of the run
tiled into one video.

To generate the full dataset, see [Running the full release](#running-the-full-release).

---

## The dataset

### Valid and invalid twins

Every invalid clip has a **valid twin**: the same scene, seed and objects, and a
**bit-identical prefix** up to the frame the violation is introduced (`t_event`). Any difference
between the two is attributable to the intervention alone — no lighting change, no re-rolled
object, no camera drift. Prefix identity is verified per clip and recorded as
`provenance.prefix_identical_verified`.

```
valid    ─────────────────────────●──────────────────────
                                  │ t_event      lawful
invalid  ─────────────────────────●╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌
         identical to the frame            the violation
         before t_event                    and its consequences
```

One valid twin is shared by every family and severity staged on that scene.

### What each clip contains

| file | what it is |
|---|---|
| `rgb.mp4` | the video |
| `violation_mask.npz` | **bool [T,H,W] — the primary annotation.** Where the violation can be seen |
| `severity_map.npz` | **f16 [T,H,W]** — how badly, per pixel per frame, bounded [0,1] |
| `causal_mask.npz` | uint8 [T,H,W] — 1 = the culprit, 2+ = bodies it disturbed |
| `reference_mask.npz` | bool [T,H,W] — where the culprit *should* have been (from the valid twin) |
| `timelines.npz` | per-frame flags: `active`, `observable`, `occluded`, `severity_t` |
| `meta.json` | labels, taxonomy, windows, magnitudes, provenance — see [docs/schema.md](docs/schema.md) |
| `seg.npz` | uint16 [T,H,W] — instance ids, stable across frames, so they are object tracks |
| `depth` · `flow_fwd/bwd` · `normals` · `object_coords` | geometry passes |
| `energy` · `bodies` · `residuals` | mechanical energy, per-body state, and the raw residuals |
| `overlay.mp4` | everything above burned into one annotated video |

Every object's track for the whole clip is one comparison:

```python
seg  = np.load("seg.npz")["seg"]                 # [T,H,W]
bods = np.load("bodies.npz")
for bid, name in zip(bods["body_ids"], bods["body_names"]):
    track = (seg == bid)                         # [T,H,W] bool
```

### Before you train on it

- **`divergence_map` is not the violation region.** It is `|valid − invalid|` in pixel space and
  diverges everywhere downstream of the event, so a model trained on it learns to find the edit,
  not the physics. Train on `violation_mask` and `severity_map`.
- **`violation_mask` is gated on visibility.** It answers *where can this be seen*, so it is
  empty while the culprit is hidden. `timelines.active` is the unhedged truth about *when*; the
  gap between them is the observability lag.
- **`permanence` and `dissolve` have an all-zero severity map** — the body is gone, so it has no
  pixels to score. `reference_mask` carries where it should have been.

### Layout on disk

```
out/<run>/clips/<release>/<level>/<scenario>/<seed>_<condition>/
    valid/                       the twin, shared by every family and severity
    invalid_<family>_<bin>/      one per cell per severity
```

A *cell* is one (scenario, family) pair. Per variant, each cell renders one invalid clip per
severity bin (families with no magnitude axis render only `strong`), and each scenario renders one
valid twin. A variant is a **fresh seed**, and each complexity level draws from its own seed block,
so no two levels share a scene.

---

## Taxonomy

Five levels: **medium → domain → family → scenario → instance.**
`python -m physloc.cli taxonomy` prints every cell.

### Medium — what the violation is made of

<!-- physloc:media -->
| medium | what it means | scenarios |
|---|---|---|
| **rigid** | solid bodies that keep their shape | 13 |
| **granular** | many small bodies behaving as a medium -- the v0 stand-in for fluid, and never labelled as one | 1 |
| **optical** | light and shadow rather than matter | 1 |
| **fluid** | Phase 3 -- needs a Blender with working headless Mantaflow | *not in v0* |
| **continuum** | Phase 3 -- cloth and soft bodies, behind a MuJoCo/MJX backend | *not in v0* |
<!-- /physloc:media -->

### Domain — the question the violation asks

<!-- physloc:domains -->
| domain | the question it asks | n | cells | families |
|---|---|---|---|---|
| **identity** | does the object persist and stay itself? | 5 | 48 | `dissolve`, `fission`, `fusion`, `immutability`, `permanence` |
| **kinematics** | is unsupported motion consistent with g? | 5 | 34 | `antigravity`, `continuity`, `newton1_inertia`, `non_parabolic`, `time_slip` |
| **contact** | do bodies interact legally when they touch? | 2 | 15 | `solidity`, `superelastic` |
| **dynamics** | do forces and masses behave? | 3 | 19 | `angular_momentum`, `newton2_mass`, `phantom_impulse` |
| **equilibrium** | do resting and supported bodies behave? | 2 | 16 | `friction`, `support` |
| **optical** | is light consistent with geometry? | 3 | 3 | `shadow`, `shadow_inverted`, `shadow_shape` |
| **appearance** | does the object look like itself from frame to frame? | 2 | 26 | `colour_shift`, `deformation` |
| **global** | are the scene's constants physical? | 1 | 5 | `global_gravity` |
<!-- /physloc:domains -->

### Scenario — the situation it happens in

<!-- physloc:scenarios -->
| scenario | medium | families | what happens |
|---|---|---|---|
| `barrier_pass` | rigid | 14 | ball rolls into a solid wall and bounces back |
| `collision` | rigid | 16 | two spheres roll toward each other |
| `drop` | rigid | 14 | sphere or cube falls to a floor and bounces |
| `occluder_pass` | rigid | 12 | body travels behind a screen and re-emerges |
| `pendulum_swing` | rigid | 8 | bob on a rigid rod, swung from a pivot (scripted, not solved) |
| `pour` | granular | 17 | a loose column of grains falls into an open box (96 at the debug tier, 212 above) |
| `pyramid_impact` | rigid | 13 | cube dropped onto a sphere pyramid |
| `ramp_slide` | rigid | 11 | block slides down an incline |
| `resting_table` | rigid | 11 | several bodies at rest on a surface |
| `rolling_ramp` | rigid | 16 | cube tumbles down a raised ramp and off its lip |
| `shadow_track` | optical | 10 | object translates under a fixed light |
| `stack_topple` | rigid | 13 | stacked bodies, marginally stable |
| `toss` | rigid | 11 | body thrown on a ballistic arc |
<!-- /physloc:scenarios -->

`clutter_toss` and `tumble` are declared but not built.

---

## Severity

Each cell is staged at three strengths: **`weak`, `medium`, `strong`.** Two numbers describe it,
and they are never the same thing:

| field | what it is | when it is known |
|---|---|---|
| `magnitude` | **the knob that was turned** — one exact scalar, in the family's own units | *before* simulating |
| `peak_severity` | **the measured effect** — the residual, z-scored against a noise floor, bounded to [0,1] | *after* |

`magnitude` says how hard the intervention pushed; `peak_severity` says how much came out, which
depends on the scene — a strong push into a wall can measure less than a medium push into open
space. `severity_map` is the spatial form of the second: `violation_mask` is binary *where*,
`severity_map` is continuous *how badly*.

Every family's ladder is monotone in distance from lawful, enforced by
`tests/test_severity_ladders.py`.

---

## Complexity ladder

Severity asks *how badly is the law broken*; complexity asks *how hard is the scene to parse*.
They are independent axes. Four levels of scene realism, each the one below plus exactly one
thing:

<!-- physloc:ladder -->
| level | adds | background | objects | materials | share | built |
|---|---|---|---|---|---|---|
| **L0** | *the baseline* | solid | primitive | one shared density | 50% | yes |
| **L1** | **materials** — wood, steel, rubber | solid | primitive | yes | 25% | yes |
| **L2** | **an HDRI environment** | hdri | primitive | yes | 15% | yes |
| **L3** | **GSO objects** — real 3D scans | hdri | gso | yes | 10% | yes |
<!-- /physloc:ladder -->

- Every clip carries its level in `complexity`, so a level is a filter, not a separate download.
- **Each level draws its own scenes** — an L1 clip is a different event, not an L0 clip in better
  materials — so the ladder adds breadth as well as difficulty.
- At L0 every object shares one density, so mass varies only with size. From L1 up mass is
  `density × volume`, and a heavy-looking object is heavy.

---

## Difficulty conditions

Every clip carries **exactly one** condition:

<!-- physloc:conditions -->
| condition | share | camera | extra objects | objects with invalid physics |
|---|---|---|---|---|
| `standard` | 60% | static | — | **1** |
| `camera` | 10% | **moves** | — | **1** |
| `distractors` | 10% | static | **3–10** | **1** |
| `multi` | 10% | static | **3–10** | **2 … N−1** |
| `camera+multi` | 10% | **moves** | **3–10** | **2 … N−1** |
<!-- /physloc:conditions -->

The camera moves on 20% of clips, 10% carry distractors and 20% have multiple culprits. Fields:
`condition`, `camera_motion`, `n_distractors`, `n_actors`, `n_culprits`. One condition per clip,
rather than independent coin flips per axis, keeps every count exact and makes every comparison
against `standard` isolate one change.

**`distractors` and `multi` differ only in how many objects violate:**

| | `distractors` | `multi` |
|---|---|---|
| extra objects | 3–10, some moving | 3–10, some moving |
| **objects violating** | **exactly 1** | **2 … N−1** |
| what the extras are | scenery no family can target (`role="distractor"`) | eligible culprits (`role="actor"`) |
| the question it asks | *is anything wrong?* | ***which** of these is wrong?* |

Only `multi` is a true localisation problem. The counts are drawn per clip so a model cannot learn
a layout, and the two conditions are never combined.

**Camera motion** is `track` (40%, slides with the aim held), `orbit` (40%, fixed radius) or
`dolly` (20%) — never a pan, which would make *did the object move or did the camera?*
unanswerable. Under a moving camera `flow` and `depth` include camera motion; per-frame
extrinsics ship in `meta.json`.

---

## Detection difficulty

Conditions are the **knob** — what was asked for. `difficulty` is the **measurement** — what came
out. A clip with eight distractors whose culprit fills a quarter of the frame is not hard; a
`standard` clip whose two-frame violation happens behind a screen is. `difficulty` is to
`condition` what `peak_severity` is to `magnitude`.

Every **invalid** clip carries one label (a valid twin has nothing to detect):

```json
"difficulty": {
  "level": "hard", "rank": 2,
  "binding_factors": ["footprint"],
  "factors": {"footprint": {"value": 0.0041, "level": "hard"},
              "severity":  {"value": 0.98,   "level": "easy"}, "...": {}}
}
```

### Seven factors; a clip takes its worst

<!-- physloc:difficulty -->
| factor | the question it asks | unit | easy | moderate | hard |
|---|---|---|---|---|---|
| `footprint` | how much of the frame does the violation cover, at its biggest? | fraction of frame | &ge; 0.05 | &ge; 0.012 | &lt; 0.012 |
| `occlusion` | how much of the violation happens while the culprit is hidden? | fraction of the violation window | &le; 0.05 | &le; 0.5 | &gt; 0.5 |
| `duration` | how long is the violation observable? | fraction of the clip | &ge; 0.35 | &ge; 0.15 | &lt; 0.15 |
| `severity` | how far from lawful does the physics actually get? | bounded residual, 0-1 | &ge; 0.9 | &ge; 0.4 | &lt; 0.4 |
| `clutter` | how many bodies must a model consider? | count | &le; 2 | &le; 6 | &gt; 6 |
| `culprits` | how many of them are violating? | count | &le; 1 | &le; 3 | &gt; 3 |
| `camera` | how far does the camera travel? | path length / standoff | &le; 0.02 | &le; 0.12 | &gt; 0.12 |
<!-- /physloc:difficulty -->

A clip is `easy` only when it is easy on **every** factor (KITTI's construction, not a weighted
score):

- **It says why** — `binding_factors` names the factors that set the label.
- **The sets nest** — easy ⊂ moderate ⊂ hard, so "moderate" means every clip with `rank <= 1`.
- **Nothing cancels** — a tiny footprint is not offset by a static camera.

```python
df[df.difficulty_rank <= 1]                        # the "moderate" evaluation set
df[df.difficulty == "hard"].binding_factors         # and what made them hard
```

**Not factors, on purpose:** the complexity level (its own axis — report
`difficulty × complexity` as a grid, which `physloc stats` plots), the family and scenario, and
`magnitude` (the knob; `severity` is its measurement). `pour`'s grains count as **one** body for
`clutter` and `culprits`, unless a family targets a genuine subset of them.

### Thresholds

The cuts live in **`configs/common.yaml`**, written `[easy, moderate]`; which direction is easier
belongs to the factor, not the config:

```yaml
difficulty:
  footprint: [0.05, 0.012]     # easy at or ABOVE 0.05 of the frame
  occlusion: [0.05, 0.5]       # easy at or BELOW 0.05 of the window
  clutter:   [2, 6]
```

Four are fitted at the tertiles of a review corpus and three are chosen; which is which is recorded
per factor in `physloc/annotate/difficulty.py`. To refit against your own runs:

```bash
python scripts/fit_difficulty.py out/review_conditions out/review_severity
```

> **Freeze them once you publish.** A benchmark whose labels move between releases cannot be
> compared with itself. The resolved values are recorded in every `meta.json`; changing them is a
> new release, not a bug fix.

---

## Scene variation

Every free parameter is drawn per clip from the seed: object shape, size, colour, mass, starting
position and velocity, floor and backdrop colour, camera pose, and the frame the violation fires
on. `python -m physloc.cli randomisation` reports the distinct values per axis.

**Materials** (from L1) make appearance and density agree, on the staging as well as the actors.
The draw is weighted so the mean density stays where the scenarios' contact parameters were tuned
(2256 kg/m³ against 2313; a uniform draw would give 3458):

<!-- physloc:materials -->
| material | density kg/m³ | share | surface |
|---|---|---|---|
| `cork` | 240 | 10% | rough 0.94, spec 0.18 |
| `wood` | 650 | 10% | rough 0.82, spec 0.30 |
| `cardboard` | 700 | 10% | rough 0.95, spec 0.10 |
| `ice` | 920 | 10% | **transmissive**, ior 1.31, rough 0.10, spec 0.90 |
| `plastic` | 1100 | 10% | rough 0.40, spec 0.55 |
| `rubber` | 1300 | 10% | rough 0.96, spec 0.12 |
| `ceramic` | 2400 | 6% | rough 0.15, spec 0.85 |
| `glass` | 2500 | 6% | **transmissive**, ior 1.46, rough 0.22, spec 0.90 |
| `marble` | 2700 | 6% | rough 0.16, spec 0.75 |
| `stone` | 2700 | 6% | rough 0.92, spec 0.20 |
| `aluminium` | 2700 | 5% | **metal**, rough 0.38, spec 0.60 |
| `steel` | 7800 | 5% | **metal**, rough 0.22, spec 0.60 |
| `brass` | 8500 | 4% | **metal**, rough 0.28, spec 0.60 |
| `copper` | 8900 | 4% | **metal**, rough 0.26, spec 0.60 |
<!-- /physloc:materials -->

- Glass and ice are **frosted** so a transparent culprit can still be pointed at, and the two
  transmissive materials have different densities so *transparent* is not a cue for *heavy*.
- Scenery draws from a narrower set — no glass ramp, no mirror floor — and the floor keeps its
  contrast-guarded colour, taking only the surface finish.
- **Environments** (from L2): 509 HDRI Haven captures.
- **Objects** (L3): 140 Google Scanned Objects across 14 categories, all CC BY-SA 4.0, curated
  squat and roughly isotropic so their lawful motion is predictable.

---

## Splits, index and evaluation

### Splits

`main` **75%** · `held_out` **20%** · `debug` **5%**, grouped by `pair_uid` so a valid twin and its
invalid siblings never land in different splits. Pairs are cut by a hash of their uid within each
scenario, so every split sees every scenario in the same proportions and a re-generated release
reproduces its splits. Every split ships every annotation — for a blind leaderboard set, strip
annotations at that point.

### The index

`index.parquet` has one row per clip, with the video embedded so it plays in the HuggingFace
viewer:

```python
import pandas as pd
df = pd.read_parquet("index.parquet")

df.groupby(["domain", "severity_bin"]).peak_severity.mean()
df.groupby("condition").size()                       # standard / camera / multi / ...
df[df.complexity == "L3"].groupby("family").size()
df[df.n_culprits > 1]                                # the multi-object clips
df.groupby(["complexity", "difficulty"]).size()      # the grid worth reporting
df[df.difficulty == "hard"].binding_factors.str.split(",").explode().value_counts()
```

| group | columns |
|---|---|
| identity | `clip_uid`, `pair_uid`, `twin_uid`, `label`, `split` |
| taxonomy | `scenario`, `family`, `domain`, `medium` |
| violation | `severity_bin`, `magnitude`, `peak_severity`, `t_event_frame`, `violation_windows`, `observability_lag` |
| difficulty | `difficulty`, `difficulty_rank`, `binding_factors` |
| scene | `complexity`, `condition`, `camera_motion`, `n_distractors`, `n_actors`, `n_culprits`, `actor_shape`, `actor_material`, `actor_mass` |
| geometry | `tier`, `num_frames`, `fps`, `seed`, `variant` |

Group by `difficulty`; filter by `difficulty_rank`.

### Evaluating

- **Report per family, aggregate to domain,** and cross with severity, complexity and condition.
  Do not pool families into one number: cell counts per domain are very uneven (see
  [Taxonomy](#taxonomy)), so a pooled score mostly measures the largest domain.
- **Use `timelines.active` for temporal metrics and `violation_mask` for spatial ones.** They
  disagree on purpose.
- **Some families are separable by residual, some only by situation.**
  `taxonomy.EXCLUSIVE_LAWS` names the ones with a clean tripwire (`tests/test_orthogonality.py`
  enforces it). `antigravity`, `phantom_impulse` and `newton1_inertia` all move linear momentum
  because they must; what separates them has to be read from the image.

---

## Generating data

Simulation and rendering run in the pinned Kubric container; everything else runs on the host.
`python -m physloc.cli <command> --help` documents every command.

### Configs

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

### Running and checking a config

`bash scripts/run.sh <config>` runs the whole pipeline — generate, validate, coverage video,
viz, export — and passes extra flags through to `generate`. The steps individually:

```bash
python -m physloc.cli generate --config review_severity
python -m physloc.cli validate  out/review_severity      # schema + cross-checks; must exit 0
python -m physloc.cli audit     out/review_severity      # cells whose violation is not visible
python -m physloc.cli stats     out/review_severity      # the distributions, plotted
python -m physloc.cli viz       out/review_severity      # every grid and sheet, one folder
python -m physloc.cli coverage  out/review_severity      # every invalid clip, one video
```

`stats` checks the run came out in the shape it declares; it reads only `meta.json`, so it takes
seconds over a full release:

```
out/review_severity/stats/
  difficulty.png          the three labels, and which factor set each one
  difficulty_factors.png  each factor's histogram with its two cuts drawn on it
  composition.png         levels; conditions measured vs declared; difficulty x complexity
  severity.png            measured peak score per declared bin, and the counts
  coverage.png            clips per family and per scenario
  stats.json              the numbers behind all five
```

`viz` re-reads finished clips — nothing is rendered again — and names its videos so the sort
order is the reading order:

```
out/review_severity/viz/
  L0_drop_0777_solidity.mp4        one family: the valid clip beside weak/medium/strong
  L0_drop_0777_sheet_strong.mp4    one scene: every family, at one bin
```

### Overriding a config

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

### Generating one level at a time

`v0_L0` … `v0_L3` **partition** `v0_release`: each carries the variants that level gets in a full
run, so all four together produce exactly what the release config produces.

```bash
python -m physloc.cli taxonomy --config v0_L0     # price the level
python -m physloc.cli generate --config v0_L0     # ...and generate only that level
```

Use them to spread a release across machines, regenerate one level after a fix, or ship an
L0-only dataset. Nothing collides: the clip path is keyed by level and every level draws from its
own seed block.

### Tiers

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

### Generation knobs

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
| `objects` | extra-object counts, culprit counts, distractor size / speed / clearance |
| `camera` | motion kinds and weights, travel and dolly ranges |
| `materials` | the mass scale |
| `difficulty` | the detection-difficulty thresholds |

Every layer is validated, so a typo is an error rather than a silently ignored value. The resolved
values are written to `params.json` and recorded in every `meta.json`.

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

- **The progress bar** counts renders, so it moves every few seconds, and shows a weighted ETA
  once the first renders finish:

  ```
  generate all:  12%|███▏                      | 1227/10220 [5h 14m<eta 1d 14h]
  ```

- **A status line every five minutes** appears above the bar:

  ```
    status: renders 1227/10220 | jobs 18/260 done, 96 running | elapsed 5h 14m | eta 1d 14h
  ```

- **`out/physloc_v0/progress.log`** records every finished job and every status line with a
  timestamp. Follow it from any terminal, whether or not the tmux session is attached:

  ```bash
  tail -f out/physloc_v0/progress.log
  ```

- **At the end**, a stage profile reports where the time went. Its `occupancy` line says how many
  workers were busy on average; far below the worker count means jobs were waiting on memory or
  on a long straggler, not on cores.

### Resuming, splitting and memory

- **Resuming.** Running the same command again continues where it stopped: a finished job is
  skipped when its recorded request — config, dials and render backend — matches, so an
  interruption loses only the jobs in flight and a changed setting re-renders rather than mixing.
  To start over, delete `out/physloc_v0` and `out/work_v0`.
- **Across machines.** Run one level per machine — `bash scripts/run.sh v0_L0` on one,
  `v0_L1` on the next, and so on; see [Generating one level at a time](#generating-one-level-at-a-time).
- **Memory.** A job starts only when its memory fits in host RAM, and every container is capped,
  so a crowded machine cannot OOM-kill its own jobs. One job type is heavy: release-size L3
  `pour`, set to an estimated ~60 GB in `JOB_MEMORY_GB` (`physloc/cli.py`). Measure it once on a
  new machine — run this and watch `docker stats` in a second terminal — and update the figure if
  it differs:

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
| `review_severity` | L0 | 166 | 511 | 8 | **36 min** |
| `review_conditions` | L0 | 6 | 80 | 8 | **6 min** |
| `review_L0` | L0 | 166 | 179 | 8 | **12 min** |
| `review_L1` | L1 | 166 | 179 | 8 | **12 min** |
| `review_L2` | L2 | 166 | 179 | 8 | **1.3 h** |
| `review_L3` | L3 | 166 | 179 | 8 | **1.3 h** |
| `review_ladder` | L0+L1+L2+L3 | 166 | 3580 | 8 | **9.4 h** |
| `review` | L0 | 166 | 511 | 8 | **36 min** |
| `v0_mini` | L0+L1+L2+L3 | 41 | 2520 | 96 | **6.2 h** |
| `v0_L0` | L0 | 166 | 5110 | 96 | 145 h (**6.0 days**) |
| `v0_L1` | L1 | 166 | 2555 | 96 | 72 h (**3.0 days**) |
| `v0_L2` | L2 | 166 | 1533 | 96 | 123 h (**5.1 days**) |
| `v0_L3` | L3 | 166 | 1022 | 96 | 82 h (**3.4 days**) |
| `v0_release` | L0+L1+L2+L3 | 166 | 10220 | 96 | 422 h (**17.6 days**) |
<!-- /physloc:costs -->

### Where a release's time goes

<!-- physloc:costs_ladder -->
| level | variants | renders | per render | at 96 workers | share of the run |
|---|---|---|---|---|---|
| **L0** | 10 | 5110 | 298 s | 145 h (**6.0 days**) | 34% |
| **L1** | 5 | 2555 | 298 s | 72 h (**3.0 days**) | 17% |
| **L2** | 3 | 1533 | 677 s | 123 h (**5.1 days**) | 29% |
| **L3** | 2 | 1022 | 677 s | 82 h (**3.4 days**) | 19% |
| **all four** | -- | 10220 | -- | 421 h (**17.6 days**) | 100% |
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
[docs/performance.md](docs/performance.md).

---

## Publishing

```bash
python -m physloc.cli export out/physloc_v0 --push-to <user>/physloc
```

Packaging always happens; **uploading only when asked**. `run.sh` does both when
`PHYSLOC_PUSH_TO` is set (`PHYSLOC_PUSH_PRIVATE=1` for a private repository), or set
`PHYSLOC_PUSH_OWNER` once and every run publishes as `<owner>/physloc-<run>`. A push replaces the
card and index at that repository id. What lands: the dataset card, `index.parquet` with inline
video, `taxonomy.json`, `splits/`, `LICENSE` and the WebDataset shards.

---

## Development

### Two environments

| | runs | contains |
|---|---|---|
| **container** — pinned Kubric image | scene sampling, simulation, rendering | Kubric 2022.4.1, Blender 2.93.4, PyBullet, Python 3.9 |
| **host** — `conda activate physloc` | annotation, residuals, masks, validation, visualisation | numpy, scipy, opencv, jsonschema, Python 3.11 |

They meet at the trajectory seam, `traj.npz`; never install Kubric, Blender or PyBullet on the
host. `docker/kubric.sh <script.py>` runs a script from this repo inside the container.
`bash scripts/fetch_refs.sh` checks out a read-only copy of the Kubric source for reference.

### Keeping the tables honest

The taxonomy, ladder, condition, difficulty, material, tier and cost tables in this README are
**generated** from `physloc/taxonomy.py`, `physloc/scenarios/base.py` and `physloc/cli.py`. After
changing any of them:

```bash
python -m physloc.reference            # is the README current? exits 1 if stale
python -m physloc.reference --write    # regenerate the tables in place
```

`tests/test_reference.py` fails when they are stale, and the HuggingFace card is generated from the
same functions.

### Tests

```bash
python -m pytest tests                 # the full suite: about 40 minutes, pour cells are slowest
python -m pytest tests/test_reference.py tests/test_cpu_slots.py   # a quick subset
```

### Repository layout

```
physloc/scenarios/    scenario builders, the complexity ladder and conditions (base.py)
physloc/injectors/    the violation families, one file per domain
physloc/render/       the container worker, and probes for render cost
physloc/sim/          trajectories and the simulation seam
physloc/residuals/    the physical residuals severity is measured from
physloc/annotate/     residuals -> masks, severity, timelines, difficulty, meta.json
physloc/release/      export, splits, dataset card
physloc/viz/          overlays, grids, sheets; every mp4 is written here
physloc/cli.py        the `physloc` command line
configs/              common.yaml and one file per run
scripts/              run.sh, run_fast.sh, probes and refresh tools
docs/schema.md        the meta.json field reference
docs/performance.md   how cost and performance were measured
docs/PLAN.md          the design document
docs/roadmap.md       what is next
```

---

## References

[IntPhys 2](https://arxiv.org/abs/2506.09849) · [LikePhys](https://arxiv.org/abs/2510.11512) ·
[Kubric](https://github.com/google-research/kubric). The IntPhys 2 category and LikePhys domain
each family maps to are data, on each family in `physloc/taxonomy.py`.
