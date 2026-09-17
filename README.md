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

Load a dataset — either downloaded from the Hub, exported locally, or still in
the generator output directory — and check every annotation by eye:

```bash
python test_dataset_loader.py                               # downloads samueleruf/physloc-mini into data/hub
python test_dataset_loader.py --repo <owner>/physloc-review_severity
python test_dataset_loader.py out/review_severity
python test_dataset_loader.py out/review_severity --gui       # browser viewer, http://localhost:8765
python test_dataset_loader.py out/review_severity --render 0 --layers violation,reference,bbox3d
```

From Python, [`physloc/loader.py`](physloc/loader.py) gives every sample and pair — see
[Loading the dataset](#loading-the-dataset).

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

### Layout on disk

PhysLoc has one public representation: schema v3.

```text
<root>/
├── dataset.json
├── schema.json
├── index.parquet
├── splits/
└── samples/<sample_uid>/
    ├── sample.json
    ├── rgb.mp4
    └── data.h5
```

RGB stays as a standalone H.264 MP4. `sample.json` is the readable manifest:
identity, taxonomy, scene/world settings, object definitions, events, causal
relations, provenance, and violation descriptions. Dense numeric tensors live
in one chunked, gzip-compressed, Fletcher32-protected HDF5 file per sample.

The main HDF5 groups are `/observations`, `/objects`, `/energy`, and
`/violations`. Stable positive object IDs join segmentation, trajectories,
energy, events, causal relations, and violation maps; ID 0 is background.
Every object is classified as `subject`, `context`, `support`, or
`background`. Floors and scenery are therefore excluded from the default
analysis view, while all violators are subjects.

Energy has an explicit release-wide accounting policy. The physics total
includes every eligible dynamic subject, distractor/context object, affected
object, and peer even while occluded; floors, backdrops, supports, barriers,
walls, occluders, and renderer helpers are excluded. A separate
`energy_in_frame` curve sums only eligible objects visible in segmentation and
is the primary overlay curve. Both scopes, plus the per-object tensor and
object-ID axis, are declared in `dataset.json.dataset_metadata.energy_accounting`.

Shadows are optical observations, not physical objects. Cycles renders them
from an internal camera-hidden caster; only the real actor is exported.
`shadow_strength` and `shadow_source_id` support soft shadow localisation,
and shadow violations map to the actor with component `shadow`.

The complete field, dtype, axis, unit, violation, energy, shadow, and validation
reference is [docs/schema.md](docs/schema.md).

### Loading the dataset

The generator, exporter, loader, visualiser, and GUI all consume the same
schema-v3 sample tree:

```python
from physloc.loader import (
    PhysLocDataset, LOCALIZATION_FIELDS, ENERGY_FIELDS, collate
)

dataset = PhysLocDataset("data/physloc", fields=LOCALIZATION_FIELDS)
sample = dataset.samples[0]             # one lazy Sample
same = dataset.get(sample.uid)

sample.video_path                       # RGB is not decoded
rgb = sample.decode_rgb()               # explicit uint8 [T,H,W,3]
sample.objects("subjects")              # default analysis actors
sample.objects("violators")
sample.objects("affected")
sample.object(2)                         # static + temporal + energy + violation
sample.spatial_mask("violators")        # bool [T,H,W]

pairs = PhysLocDataset("data/physloc", unit="pair")
pair = pairs[0]                          # valid Sample + invalid Samples

batch = collate(dataset.samples[:8])     # pads N and returns object_valid
```

Loader samples have the same logical shape as the files, with small metadata in
`sample.json` and dense arrays read lazily from `data.h5`:

```text
sample
├── metadata
│   ├── sample_info
│   └── scene
│       ├── info
│       └── world
├── observations
└── annotations
    ├── objects
    ├── scene_energy
    ├── maps
    ├── events
    ├── causal_relations
    └── violation_summary
```

Below, `T` is frames, `H,W` are image size, and `N` is the number of exported
objects. A review sample such as `out_old1/review_conditions_f37` uses
`T=37`, `H=W=128`; release tiers may be larger.

```text
sample
├── metadata
│   ├── sample_info
│   │   ├── sample_uid, pair_uid, valid_sample_uid, label
│   │   ├── schema_version, dataset_version, split
│   │   ├── seed, render_seed, variant, generation_config_id
│   │   ├── num_frames=T, fps, duration_seconds, resolution=[H,W]
│   │   ├── latent_frames, latent_hw, size_scale, framing_attempt
│   │   └── provenance
│   └── scene
│       ├── info
│       │   ├── level, complexity, condition, difficulty
│       │   ├── type/scenario, family, domain, physics_medium
│       │   ├── severity, label, variant, prompt
│       │   └── difficulty_analysis
│       └── world
│           ├── camera: intrinsics, extrinsics/projection, trajectory
│           ├── environment: background, HDRI, lights, floor/support datum
│           ├── physics: gravity, timestep, substeps, solver settings, units
│           └── objects_summary: counts by role, group, violator, affected
├── observations
│   ├── rgb_path: path to rgb.mp4; decode explicitly with sample.decode_rgb()
│   ├── rgb: uint8 [T,H,W,3], only when decoded
│   ├── segmentation: uint16 [T,H,W], object IDs, 0 = background
│   ├── depth: float32 [T,H,W,1], metres
│   ├── forward_flow, backward_flow: float32 [T,H,W,2], row/col pixels
│   ├── normal: [T,H,W,3]
│   ├── object_coordinates: [T,H,W,3]
│   ├── shadow_strength: float16 [T,H,W], optional true-shadow strength
│   └── shadow_source_id: uint16 [T,H,W], optional real actor casting shadow
└── annotations
    ├── objects
    │   ├── definitions: N dicts with id, name, category, role, asset,
    │   │   analysis_group, static_fields, temporal refs, energy refs
    │   ├── static_fields: mass, dimensions, inertia/material/friction,
    │   │   restitution, scales, static/scripted/collidable/visibility flags
    │   ├── ids: int32 [N], stable object axis shared by all arrays
    │   ├── analysis_group: list[N] of subject/context/support/background
    │   ├── positions: float32 [N,T,3], metres
    │   ├── quaternions: float32 [N,T,4], w,x,y,z
    │   ├── velocities, angular_velocities: float32 [N,T,3]
    │   ├── bboxes: float32 [N,T,4]
    │   ├── bboxes_3d: float32 [N,T,8,3]
    │   ├── image_positions: float32 [N,T,2]
    │   ├── visibility: int32 [N,T]
    │   ├── is_violator: bool [N]
    │   ├── active/intervening/consequence/observable/occluded/affected:
    │   │   bool [N,T]
    │   └── severity/residual/score: float32 [N,T]
    ├── scene_energy
    │   ├── total: float32 [T], eligible physical objects, visible or not
    │   ├── energy_in_frame: float32 [T], visible eligible objects
    │   ├── kinetic_translational, kinetic_rotational: float32 [T]
    │   └── residual/anomaly curves, such as excess_loss and contact_anomaly
    ├── object_energy
    │   ├── by_body: float32 [N,T], per-object total energy
    │   ├── kinetic, potential: float [N,T]
    │   └── momentum, angular_momentum: float [N,T,3]
    ├── maps
    │   ├── violation_object_id: uint16 [T,H,W], responsible object ID
    │   ├── violation_component: uint8 [T,H,W], body/shadow/trajectory/etc.
    │   ├── causal_level: uint8 [T,H,W], causal/affected level
    │   ├── causal_source_id: uint16 [T,H,W], source violator object ID
    │   └── severity: float16 [T,H,W], optional stored severity map
    ├── events: collisions, occlusions, energy spikes, custom events
    ├── causal_relations: sparse source/target object edges with intervals
    └── violation_summary: invalid-only type, intervals, intervention,
        responsible objects, affected objects, difficulty inputs
```

The convenience methods keep common analysis paths short:

| call | returns |
|---|---|
| `sample.object(object_id)` | one object's static fields plus `temporal`, `energy`, and `violation` arrays |
| `sample.objects("subjects")` | experiment-relevant actors; floor/background/support objects are excluded |
| `sample.objects("violators")` | responsible violator objects only |
| `sample.objects("affected")` | objects affected by a violation, including non-violators |
| `sample.spatial_mask("subjects")` | `bool [T,H,W]` mask for selected analysis group |
| `sample.visible_violation` | visible component-aware violation mask |
| `sample.reference_mask` | valid-twin footprint of the violators, when a twin is available |
| `sample.timeline` | unioned per-frame clocks over violators, plus max severity |
| `sample.latent_grid()` | violation mask and severity reduced to the video-token grid |
| `sample.divergence` | RGB difference from the valid twin, for inspection only |

HDF5 handles open lazily per process, so multi-worker loading is safe. Field
presets cover metadata, localisation, object localisation, energy, and
visualisation. Derived masks, timelines, severity maps, latent grids, and
valid/invalid divergence are computed lazily rather than stored as redundant
videos.

To inspect the same API visually:

```bash
python test_dataset_loader.py out/physloc_mini
python test_dataset_loader.py out/physloc_mini --gui
python test_dataset_loader.py out/physloc_mini --render 3 \
  --layers violation,reference,bbox3d,labels \
  --panels rgb,valid,segmentation,depth,camera
```

### Before you train on it

- **`divergence` is not the violation region.** It is `|valid − invalid|` in pixel space and
  diverges everywhere downstream of the event, so a model trained on it learns to find the edit,
  not the physics. Train on `violation_mask` and `severity_map`.
- **`violation_mask` is gated on visibility.** It answers *where can this be seen*, so it is
  empty while the violator is hidden. `timeline["active"]` is the unhedged truth about *when*; the
  gap between them is the observability lag.
- **Severity needs a visible body.** Once a body has vanished (`permanence`, `dissolve`) its
  severity is painted on its lawful footprint; a body that is merely hidden — behind a screen,
  under the floor — scores zero. `reference_mask` carries where it should have been.
- **Encodings that bite:** mask depth with `segmentation > 0`; flow is
  `(row, col)`, not `(x, y)`; all per-object HDF5 tensors use the stable
  object-first axis `[N,T,...]`.

### How much is generated

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

The camera moves on 20% of clips, 10% carry distractors and 20% have multiple violators. Fields:
`condition`, `camera_motion`, `n_distractors`, `n_actors`, `n_violators`. One condition per clip,
rather than independent coin flips per axis, keeps every count exact and makes every comparison
against `standard` isolate one change.

**`distractors` and `multi` differ only in how many objects violate:**

| | `distractors` | `multi` |
|---|---|---|
| extra objects | 3–10, some moving | 3–10, some moving |
| **objects violating** | **exactly 1** | **2 … N−1** |
| what the extras are | scenery no family can target (`role="distractor"`) | eligible violators (`role="actor"`) |
| the question it asks | *is anything wrong?* | ***which** of these is wrong?* |

Only `multi` is a true localisation problem. The counts are drawn per clip so a model cannot learn
a layout, and the two conditions are never combined.

**Camera motion** is `track` (40%, slides with the aim held), `orbit` (40%, fixed radius) or
`dolly` (20%) — never a pan, which would make *did the object move or did the camera?*
unanswerable. Under a moving camera `flow` and `depth` include camera motion; per-frame
camera poses and intrinsics ship in `sample.json`.

---

## Detection difficulty

Conditions are the **knob** — what was asked for. `difficulty` is the **measurement** — what came
out. A clip with eight distractors whose violator fills a quarter of the frame is not hard; a
`standard` clip whose two-frame violation happens behind a screen is. `difficulty` is to
`condition` what `peak_severity` is to `magnitude`.

Every **invalid** sample carries one label (a valid twin has nothing to detect)
under `metadata.scene.info.difficulty_analysis`:

```json
"difficulty_analysis": {
  "level": "hard", "rank": 2,
  "binding_factors": ["violation_area"],
  "factors": {"violation_area": {"value": 0.0041, "level": "hard"},
              "severity":  {"value": 0.98,   "level": "easy"}, "...": {}}
}
```

### Seven factors; a clip takes its worst

<!-- physloc:difficulty -->
| factor | the question it asks | unit | easy | moderate | hard |
|---|---|---|---|---|---|
| `violation_area` | how much of the frame does the violation cover, at its biggest? | fraction of frame | &ge; 0.008 | &ge; 0.003 | &lt; 0.003 |
| `occlusion` | how much of the violation happens while the violator is hidden? | fraction of the violation window | &le; 0.2 | &le; 0.5 | &gt; 0.5 |
| `duration` | how long is the violation observable? | fraction of the clip | &ge; 0.35 | &ge; 0.15 | &lt; 0.15 |
| `severity` | how far from lawful does the physics actually get? | bounded residual, 0-1 | &ge; 0.1 | &ge; 0.03 | &lt; 0.03 |
| `object_count` | how many objects must a model consider? | count | &le; 3 | &le; 10 | &gt; 10 |
| `violators` | how many of them are violating? | count | &le; 1 | &le; 2 | &gt; 2 |
| `camera_motion` | how far does the camera move? | path length / standoff | &le; 0.02 | &le; 0.2 | &gt; 0.2 |
<!-- /physloc:difficulty -->

A sample is `easy` only when it is easy on **every** factor (KITTI's construction, not a weighted
score):

- **It says why** — `binding_factors` names the factors that set the label.
- **The sets nest** — easy ⊂ moderate ⊂ hard, so "moderate" means every clip with `rank <= 1`.
- **Nothing cancels** — a tiny violation area is not offset by a static camera.

### Where the cuts come from

They are set so the **whole release lands near 30 / 40 / 30**, estimated by mixing the 872
violated clips of the review corpus in the release's own proportions (six in ten `standard`, one
in ten each of the other conditions, a third per severity bin). An earlier table, whose cuts sat
at that corpus's tertiles, produced **3 / 40 / 77** on the same estimate: with the worst factor
winning, seven factors that each fail a third of the time make nearly every clip hard. A 30% hard
share therefore means each factor alone may call only a few per cent hard.

The three scene factors are set from what the sampler draws, so each condition lands where it
should without the label ever reading the condition:

| condition | lands at | because |
|---|---|---|
| `standard` | ~50 / 37 / 13 | one or two bodies, a static camera: only the evidence factors bite |
| `camera` | ~0 / 88 / 12 | any real move is past `camera_motion`'s easy cut, and the sampler never reaches its hard one |
| `distractors` | ~0 / 86 / 14 | 3–10 extras put `object_count` past 3, and only the biggest draws pass 10 |
| `multi` | ~0 / 43 / 57 | two violators is moderate, three or more is hard |
| `camera+multi` | 0 / 0 / 100 | both apply |

Two violators is deliberately **not** hard: the pair families (`newton2_mass` exchanging momentum
between two balls, `fission`) name both bodies of a single event and are 15% of `standard` clips.

### Each violator also carries its own label

A `multi` clip whose violators are one large obvious body, one small one and one behind a screen
is not described by any single word, so every violator carries its own label under
`annotations.violation_summary.violators[k].difficulty`, measured on **its** mask, **its** occlusion, **its**
observable window and **its** residual — which is what an object detector is scored against.
`object_count` and `violators` are left out of it: they count what is in the scene, which is a
property of the clip and not of any one body in it.

The scalar label is searchable as `index.parquet:difficulty`; the per-factor
measurements and `binding_factors` remain in `sample.json`. Release-wide counts,
zones, thresholds, and label-setting factors are consolidated under
`dataset.json:dataset_metadata.difficulty_analysis`.

**Not factors, on purpose:** the complexity level (its own axis — report
`difficulty × complexity` as a grid, which `physloc stats` plots), the family and scenario, and
`magnitude` (the knob; `severity` is its measurement). `pour`'s grains count as **one** body for
`object_count` and `violators`, unless a family targets a genuine subset of them.

### Thresholds

The cuts live in **`configs/common.yaml`**, written `[easy, moderate]`; which direction is easier
belongs to the factor, not the config:

```yaml
difficulty:
  violation_area: [0.05, 0.012]   # easy at or ABOVE 0.05 of the frame
  occlusion:      [0.05, 0.5]     # easy at or BELOW 0.05 of the window
  object_count:   [2, 6]
```

Four are fitted at the tertiles of a review corpus and three are chosen; which is which is recorded
per factor in `physloc/annotate/difficulty.py`. To refit against your own runs:

```bash
python scripts/fit_difficulty.py out/review_conditions out/review_severity
```

> **Freeze them once you publish.** A benchmark whose labels move between releases cannot be
> compared with itself. The resolved values are recorded in every `sample.json`; changing them is a
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

- Glass and ice are **frosted** so a transparent violator can still be pointed at, and the two
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

`index.parquet` has one row per sample. It keeps relative paths to the MP4 and
HDF5 payloads instead of embedding duplicate bytes:

```python
import pandas as pd
df = pd.read_parquet("index.parquet")

df.groupby(["domain", "severity"]).size()
df.groupby("condition").size()                       # standard / camera / multi / ...
df[df.complexity == "L3"].groupby("family").size()
df[df.n_violators > 1]                                # multi-object samples
df.groupby(["complexity", "difficulty"]).size()      # the grid worth reporting
```

| group | columns |
|---|---|
| identity | `sample_uid`, `pair_uid`, `valid_sample_uid`, `label`, `split` |
| taxonomy | `scenario`, `family`, `domain`, `physics_medium` |
| violation | `severity`, `n_violators` |
| difficulty | `difficulty`, `complexity` |
| scene | `condition`, `n_distractors`, `n_actors`, `n_violators`, `prompt` |
| geometry | `num_frames`, `fps`, `resolution`, `seed`, `variant` |
| storage | `sample_path`, `rgb_path`, `data_path` (relative paths) |

Group by `difficulty` in the index. For the complete measured distribution,
zone counts, thresholds, and the factors that set each label, read
`dataset.json["dataset_metadata"]["difficulty_analysis"]`.

### Evaluating

- **Report per family, aggregate to domain,** and cross with severity, complexity and condition.
  Do not pool families into one number: cell counts per domain are very uneven (see
  [Taxonomy](#taxonomy)), so a pooled score mostly measures the largest domain.
- **Use `timeline["active"]` for temporal metrics and `violation_mask` for spatial ones.** They
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

### The whole sweep, in one command

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

**How a difficulty label is decided.** Each of the seven factors in `annotate/difficulty.py`
has two cuts, which make three zones; a clip's label is its WORST zone. So a clip is easy only
when every factor is easy, and one hard factor makes it hard. `4_difficulty.png` prints every
cut and whether it was fitted to the review corpus or chosen from what the quantity means.

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
| `review` | L0 | 166 | 511 | 32 | **30 min** |
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
[docs/performance.md](docs/performance.md).

---

## Publishing to Hugging Face

Generation already writes the canonical schema-v3 tree. Publishing is a
validation, packaging, inspection, and upload sequence.

### 1. Validate the generated dataset

```bash
conda activate physloc

python -m physloc.cli validate out/review_conditions_f37
python -m physloc.cli stats out/review_conditions_f37
```

Validation must report `"ok": true`. The stats command writes the six review
figures and `stats/stats.json`. At the end of generation, PhysLoc also writes
`dataset.json`, `schema.json`, `index.parquet`, and `splits/` directly
into the generated root.

### 2. Build a clean publication directory

```bash
python -m physloc.cli export out/review_conditions_f37 \
  --outdir out/review_conditions_f37_hf \
  --license CC-BY-4.0
```

The source and destination must differ. The exported directory contains the
complete dataset card, licence, global metadata, JSON Schema, Parquet index,
split lists, standalone loader, and every `sample.json`, `rgb.mp4`, and
`data.h5`.

### 3. Inspect the packaged dataset locally

```bash
python test_dataset_loader.py out/review_conditions_f37_hf
python test_dataset_loader.py out/review_conditions_f37_hf --gui --port 8765
```

### 4. Authenticate once

```bash
hf auth login
```

Use a Hugging Face write token. Authentication is handled by the current
`hf` CLI; do not use the deprecated `huggingface-cli`.

### 5. Upload

The one-command PhysLoc route packages and uploads:

```bash
python -m physloc.cli export out/review_conditions_f37 \
  --outdir out/review_conditions_f37_hf \
  --license CC-BY-4.0 \
  --push-to YOUR_USERNAME/physloc-review-conditions
```

Add `--private` to create a private dataset repository:

```bash
python -m physloc.cli export out/review_conditions_f37 \
  --outdir out/review_conditions_f37_hf \
  --push-to YOUR_USERNAME/physloc-review-conditions \
  --private
```

Or upload an already-packaged directory directly:

```bash
hf upload YOUR_USERNAME/physloc-review-conditions \
  out/review_conditions_f37_hf . \
  --type dataset \
  --commit-message "Publish PhysLoc schema v3"
```

### 6. Download and verify

```bash
hf download YOUR_USERNAME/physloc-review-conditions \
  --repo-type dataset \
  --local-dir data/physloc-review-conditions

python test_dataset_loader.py data/physloc-review-conditions
python -m physloc.cli validate data/physloc-review-conditions
```

The Parquet index stores relative paths only; MP4 and HDF5 bytes are not
duplicated inside it. For a metadata-only inspection, download
`dataset.json`, `schema.json`, `index.parquet`, `splits/**`, and
`samples/**/sample.json`. Full localisation or energy experiments also need
the corresponding `data.h5`; RGB experiments need `rgb.mp4`.

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
physloc/annotate/     residuals -> masks, severity, clocks, and schema-v3 samples
physloc/loader.py     reads a release and derives every annotation (numpy only)
physloc/release/      export, splits, dataset card
physloc/viz/          the overlay renderer, browser viewer, grids, sheets; every mp4
test_dataset_loader.py  load a dataset, print its structure, look at it
physloc/cli.py        the `physloc` command line
configs/              common.yaml and one file per run
scripts/              run.sh, run_fast.sh, probes and refresh tools
docs/schema.md        the sample layout, fields, axes, units, and loader contract
docs/performance.md   how cost and performance were measured
docs/PLAN.md          the design document
docs/roadmap.md       what is next
```

---

## References

[IntPhys 2](https://arxiv.org/abs/2506.09849) · [LikePhys](https://arxiv.org/abs/2510.11512) ·
[Kubric](https://github.com/google-research/kubric). The IntPhys 2 category and LikePhys domain
each family maps to are data, on each family in `physloc/taxonomy.py`.
