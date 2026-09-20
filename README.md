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

- **[Quick start](#quick-start)**
- **Using the dataset**
  - [What a sample is](#what-a-sample-is) — twins, files on disk
  - [Loading the dataset](#loading-the-dataset) — the `Sample` API and every field
  - [Before you train on it](#before-you-train-on-it)
- **What is in it**
  - [Taxonomy](#taxonomy) — medium → domain → family → scenario
  - [What varies, and how it is labelled](#what-varies-and-how-it-is-labelled) — severity,
    complexity ladder, difficulty conditions, detection difficulty, scene variation
  - [Splits, index and evaluation](#splits-index-and-evaluation)
- **Building and publishing it**
  - [Building it yourself](#building-it-yourself)
  - [Licensing and citation](#licensing-and-citation)
- [Project documentation](#project-documentation) · [References](#references)

---

## Quick start

**Requirements:** Linux and conda. To *generate* data you also need Docker; rendering is
CPU-only — the more physical cores the better — and a release run wants about 2 GB of RAM per vCPU.

```bash
# Host environment: loading, annotation, severity, validation, visualisation.
conda env create -f environment.yml
conda activate physloc
```

**Look at a dataset** — downloaded from the Hub, exported locally, or still in the generator's
output directory — and check every annotation by eye:

```bash
python test_dataset_loader.py                               # downloads samueleruf/physloc-mini into data/hub
python test_dataset_loader.py --repo <owner>/physloc-review_severity
python test_dataset_loader.py out/review_severity
python test_dataset_loader.py out/review_severity --gui       # browser viewer, http://localhost:8765
python test_dataset_loader.py out/review_severity --render 0 --layers violation,reference,bbox3d,labels \
  --panels rgb,valid,segmentation,depth,camera
```

**Load it from Python** with [`physloc/loader.py`](physloc/loader.py) — see
[Loading the dataset](#loading-the-dataset).

**Generate a small sample yourself.** Simulation and rendering run inside the pinned Kubric image:

```bash
docker pull kubricdockerhub/kubruntu      # digest pinned in docker/IMAGE_DIGEST
python -m physloc.cli taxonomy --config review_severity   # what it produces and how long it takes
bash scripts/run.sh review_severity                       # generate, validate, visualise, package
```

`run.sh` ends by listing what to open, starting with `coverage_strong.mp4`: every cell of the run
tiled into one video. The full dataset is [Building it yourself](#building-it-yourself).

---

## What a sample is

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

PhysLoc has one public representation: schema v4.

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

RGB stays as a standalone H.264 MP4. `sample.json` is the readable manifest, one
block per question -- `sample` (who), `video` (geometry), `scene` (what was
simulated and how it was shot), `objects` (one record per body), `violation`
(invalid samples only: what is wrong, per violator, with its five clocks as
frame intervals) and `provenance`. **Every fact is stored once**: clip-level
windows and times, `[N,T]` clocks, causal relations, counts and the body
severity map are derived by the loader rather than stored. Every dense or
per-frame array -- render passes, camera track, object trajectories, energy,
collisions, violation maps -- lives in one chunked, gzip-compressed,
Fletcher32-protected HDF5 file per sample, under `/observations`, `/camera`,
`/objects`, `/energy`, `/events` and `/violation`.

Stable positive object IDs join segmentation, trajectories, energy, events,
causal relations, and violation maps; ID 0 is background. Every object is
classified as `subject`, `context`, `support`, or `background`. Floors and
scenery are therefore excluded from the default analysis view, while all
violators are subjects.

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

The complete field, dtype, axis and unit reference -- and a table of every v3
field v4 stopped storing, with the loader derivation that replaced it -- is
[docs/schema.md](docs/schema.md).

## Loading the dataset

The generator, exporter, loader, visualiser, and GUI all consume the same
schema-v4 sample tree. A `Sample` is organised the way the files are, one
namespace per block, every array read lazily and cached:

```python
from physloc.loader import PhysLocDataset, collate

ds = PhysLocDataset("data/physloc", split="main")
s = ds[0]                                   # one lazy Sample

s.info            # uid, pair_uid, valid_uid, label, split, release, tier, seed, variant
s.video           # num_frames, fps, resolution, duration, latent_*, path, rgb
s.scene           # scenario, family, domain, level, condition, prompt, camera, physics
s.observations    # segmentation, depth, forward_flow, normal, ... [T,H,W,...]
s.objects         # ids, names, roles, groups, records, positions, bboxes ... [N,T,...]
s.violation       # what is wrong; empty (present=False) on a valid sample
s.energy          # scene [T], objects [N,T], map [T,H,W]
s.events          # collisions, one array per field
s.twin            # the valid Sample of this pair
```

Throughout, **`T`** is frames, **`H,W`** the image size, **`N`** every exported
object and **`V`** the violators. Every `[N,...]` array is in one row order --
`s.objects.ids[i]` owns row `i` -- shared by both twins, so row `i` is the same
body in the valid and the invalid clip. `s.objects.row(object_id)` maps an id to
its row.

### `s.info` -- who this sample is

Read from `sample.json`, no arrays.

| field | type | meaning |
|---|---|---|
| `uid` | str | `<release>/<level>/<scenario>/<seed>_<condition>/` then `valid` or `invalid_<family>_<bin>`; also the path under `samples/` |
| `pair_uid` | str | the scene. One valid twin and its invalid siblings share it |
| `valid_uid` | str | the valid twin's uid; a valid sample points at itself |
| `label` | str | `valid` or `invalid` |
| `is_valid` | bool | `label == "valid"` |
| `split` | str | `main`, `held_out` or `debug`. Assigned per `pair_uid`, so a twin never crosses a split |
| `release` | str | what the dataset is called, from `--outdir` |
| `tier` | str | `debug` (128², 12 fps, 25 frames) or `release` (512², 30 fps, 89 frames) |
| `seed` | int | the scene seed. The same seed at a different level is a DIFFERENT scene |
| `variant` | int | index in the ten-slot condition cycle; says why this clip got a camera move or clutter |
| `framing_attempt` | int | how many scenes were resampled before the actors stayed in shot |
| `provenance` | dict | `generator_commit`, `kubric_image_digest`, `blender_version`, and the measured `prefix_identical_verified` / `prefix_differing_pixels` / `prefix_identical_upto_frame` |

### `s.video` -- clip geometry and the RGB

| field | type | meaning |
|---|---|---|
| `num_frames` | int | `T`, always `4k+1` |
| `fps` | float | frames per second |
| `resolution` | (int, int) | `(H, W)` |
| `duration` | float | `T / fps`, seconds |
| `latent_frames`, `latent_hw` | int | the video-token grid `latent_grid()` reduces to |
| `path` | str | absolute path to `rgb.mp4`; **no decoding** |
| `rgb` | uint8 `[T,H,W,3]` | the decoded video, read on first access |

### `s.scene` -- what was simulated, and how it was shot

| field | type | meaning |
|---|---|---|
| `scenario` | str | the staging: `drop`, `collision`, `barrier_pass`, `pour`, ... (13 built) |
| `family` | str | the violated law: `continuity`, `solidity`, `shadow`, ... `None` on a valid twin, which belongs to every family staged on the scene |
| `domain` | str | the family's group: `kinematics`, `dynamics`, `optical`, ... |
| `physics_medium` | str | `rigid`, `granular`, `optical`, ... A granular medium counts as ONE thing for difficulty |
| `level` | str | complexity `L0`–`L3`: baseline, materials, HDRI, GSO scans. Scene realism only |
| `condition` | str | `standard`, `camera`, `distractors`, `multi`, `camera+multi` -- one per clip |
| `prompt` | str | a text description of the LAWFUL scene, for text-conditioned use |
| `size_scale` | float | the per-scene object size multiplier |
| `camera.motion` | str | `static`, `track`, `orbit` or `dolly` |
| `camera.K` | float64 `[3,3]` | normalised intrinsics (Kubric's convention) |
| `camera.field_of_view`, `focal_length`, `sensor_width` | float | the lens: degrees, mm, mm |
| `camera.look_at`, `position`, `end_position` | list[3] | the aim point held all clip, the start eye, and the end eye when it moves |
| `camera.positions` | float64 `[T,3]` | per-frame eye, camera-to-world, metres |
| `camera.quaternions` | float64 `[T,4]` | per-frame rotation, `w,x,y,z`, looking down −Z |
| `environment` | dict | `background` colour, `hdri` name (L2/L3), `lighting` |
| `physics` | dict | `engine`, `gravity` `[3]`, `step_rate` Hz, `substeps_per_frame`, and `params`: every resolved generation knob |

Under a moving camera `forward_flow`, `backward_flow` and `depth` include the
camera's own motion; the track above is what undoes it.

### `s.observations` -- the render passes, `[T,H,W,...]`

Each is read from `data.h5` on first access. `segmentation` is always present;
the rest depend on what the tier rendered, so test with `"depth" in s.observations`.

| pass | type | meaning |
|---|---|---|
| `segmentation` | uint16 `[T,H,W]` | object id per pixel, `0` is background. Ids match `s.objects.ids` |
| `depth` | float32 `[T,H,W,1]` | metres from the camera; background carries a huge sentinel, not a distance |
| `forward_flow` | float32 `[T,H,W,2]` | pixel motion to the next frame; undefined on the last |
| `backward_flow` | float32 `[T,H,W,2]` | pixel motion from the previous frame; undefined on the first |
| `normal` | uint16 `[T,H,W,3]` | surface normal, encoded |
| `object_coordinates` | uint16 `[T,H,W,3]` | each pixel's position in ITS OWN object's frame -- not world space |
| `shadow_strength` | float16 `[T,H,W]` | shadow scenarios only: the matched Cycles shadow-isolation pass |
| `shadow_source_id` | uint16 `[T,H,W]` | shadow scenarios only: which actor casts that shadow |

### `s.objects` -- who is in the scene, and every trajectory

Static facts come from `sample.json`, per-frame arrays from `data.h5`.

| field | type | meaning |
|---|---|---|
| `ids` | int32 `[N]` | the row order of every `[N,...]` array. `0` is never an object |
| `names`, `roles`, `groups` | list[N] | name; `role` (`actor`, `prop`, `distractor`, `floor`, `occluder`, `backdrop`); `analysis_group` (`subject`, `context`, `support`, `background`) |
| `is_violator` | bool `[N]` | rows this clip's violation names |
| `records` | list[N] of dict | the `sample.json` record: `id`, `name`, `category`, `role`, `analysis_group`, `asset{id, source, license, held_out}`, `physics{mass, friction, rolling_friction, restitution, static, collidable, scripted, dormant, energy_eligible}`, `render{material, color, scale}` |
| `positions` | float32 `[N,T,3]` | world position, metres |
| `quaternions` | float32 `[N,T,4]` | rotation, `w,x,y,z` |
| `velocities` | float32 `[N,T,3]` | m/s |
| `angular_velocities` | float32 `[N,T,3]` | rad/s |
| `bboxes` | float32 `[N,T,4]` | 2D box `(ymin, xmin, ymax, xmax)`, normalised to `[0,1]`, **NaN on frames where the body has no pixels** |
| `bboxes_3d` | float32 `[N,T,8,3]` | the 8 world-space corners |
| `image_positions` | float32 `[N,T,2]` | projected centre, normalised `(x, y)`; may fall outside `[0,1]` when off screen |
| `visibility` | int32 `[N,T]` | how many pixels the body occupies; `0` means fully hidden or absent |

```python
o = s.objects
o.positions[o.row(2)]            # object 2's trajectory, [T,3]
o.record(2)                      # its static record + its row of every array
o.select("violators")            # ids; subjects (default), affected, context, support, background, all
o.spatial_mask("subjects")       # bool [T,H,W] from the segmentation
```

### `s.violation` -- what is wrong, where, when and how badly

`present` is `False` on a valid sample and every array is then zeros of the
right shape, so a batch may mix valid and invalid clips without special cases.

**The description** (from `sample.json`):

| field | type | meaning |
|---|---|---|
| `severity_bin` | str | `weak`, `medium` or `strong`: the strength that was ASKED for |
| `kind` | str | `instant` (one frame, e.g. a teleport) or `sustained` (a window) |
| `component` | str | `body` or `shadow` -- what the violation is ON |
| `spatial_extent` | str | `local` (one body) or `global` (the whole scene, e.g. gravity) |
| `timing` | str | `shared` (violators act together), `independent` (each has its own moment) or `sync` (a `multi` clip that chose one moment) |
| `intervention` | dict | the knob that was turned: `type`, `magnitude`, `unit`, `params`. **Known before simulating** |
| `peak_residual` | dict | the effect that was MEASURED: `law`, `value` (in the law's own units), `score` (bounded 0–1), `z_vs_valid`, `frame` |
| `difficulty` | dict | `level` (`easy`/`moderate`/`hard`), `rank`, `binding_factors`, per-factor `factors`, and the measured `inputs` |
| `occluded_at_event` | bool | was the violator hidden when it fired |
| `causal_ids` | list[int] | every body the PLAN names, which for a two-body family is both |
| `shadow` | dict or None | shadow families only: caster, light, receivers, render method, threshold |
| `violators` | list[V] of dict | one record per violator, below |

Each **violator record**: `id`, `affected_ids` (bodies it disturbed),
`windows{active, intervening, consequence, observable, occluded}` as inclusive
`[[start, end], ...]` frame intervals, `t_event`, `t_observable`,
`observability_lag`, `peak_residual`, `peak_severity`,
`frames_visible_after_event`, `difficulty` (its own), `magnitude`.

**Derived by the loader** from those records -- nothing below is stored:

| field | type | meaning |
|---|---|---|
| `ids` | int32 `[V]` | the violators' object ids |
| `windows` | dict of intervals | the clip-level clocks: the union over violators |
| `t_event` | int | first frame the violation is active |
| `t_observable` | int | first frame a viewer could tell |
| `observability_lag` | int | `t_observable − t_event`; how long the violation hides |
| `t_end`, `t_intervention_end`, `t_consequence_end` | int | last active / last frame being changed / last frame still wrong |
| `causal_relations` | list[dict] | `source_object_id → target_object_id` over the source's consequence windows |

**The five clocks**, `bool [N,T]`, one row per object (zeros for non-violators):

| clock | true while |
|---|---|
| `active` | the violation is happening |
| `intervening` | the intervention is being applied |
| `consequence` | the scene is still wrong because of it |
| `observable` | a viewer could tell |
| `occluded` | the violator is FULLY hidden |
| `affected` | (not a clock) this body was disturbed BY a violator |

**Per-object measurements**, `float32 [N,T]`: `severity` (0–1, what
`severity_map` paints), `residual` (raw, in the law's units) and `score`
(the bounded 0–1 form of the residual).

**Maps**, `[T,H,W]`, invalid side unless noted:

| map | type | meaning |
|---|---|---|
| `object_id` | uint16 | which violator owns each pixel; `0` elsewhere |
| `component_map` | uint8 | `0` none, `1` body, `2` shadow, `3` trajectory, `4` interaction, `5` energy |
| `causal` | uint8 | `1` a violator, `2` a body it affected |
| `causal_source` | uint16 | which violator is responsible for that pixel |
| `mask` | bool | **a training target.** `object_id > 0` -- the union over BOTH twins, so a vanished body still has a mask |
| `visible` | bool | the part of `mask` a viewer can actually see |
| `severity_map` | float32 | **a training target.** 0–1 per pixel: how badly wrong, invalid side only |
| `reference_mask` | bool | where the violators lawfully ARE, from the valid twin. Needs `s.twin` |

**Also**: `timeline` -- the clocks and severity reduced to `[T]` (any violator,
max severity), for plotting; and `latent_grid()` -- `mask`, `severity_max` and
`severity_mean` on the `[latent_frames, latent_hw, latent_hw]` token grid.

### `s.energy`, `s.events`, `s.twin`, `s.divergence`

| field | type | meaning |
|---|---|---|
| `energy.scene` | dict of float32 `[T]` | `total`, `kinetic_translational`, `kinetic_rotational`, `dissipated`, `energy_in_frame` (visible bodies only) and the anomaly curves `free_anomaly`, `contact_anomaly`, `excess_loss` |
| `energy.objects` | dict of `[N,T,...]` | `kinetic`, `potential`, `by_body` (J), `mass` (kg), `height` (m), `momentum` `[N,T,3]`, `angular_momentum`, `inertia`, `in_frame` (bool), and the derived `momentum_magnitude` |
| `energy.map` | float32 `[T,H,W]` | per-pixel energy, J |
| `events.collisions` | dict of arrays `[E,...]` | one row per contact: `frame`, `instances` `[E,2]`, `force`, `position` `[E,3]`, `contact_normal` `[E,3]`, `image_position` `[E,2]` |
| `twin` | Sample or None | the valid sample of this pair |
| `divergence` | float32 `[T,H,W]` | the absolute RGB difference from the valid twin. **For inspection only -- never a training target**: it diverges everywhere after the event |

### Finding your way around a sample

Arrays are read on first access, so a debugger's variable pane shows only what
has already been touched. Two calls answer "what can I access here?" without
reading anything:

```python
print(s.describe())          # every block and field, with dtype and shape
print(s.objects.describe())  # or just one block
s.objects.keys()             # ['ids', 'names', ..., 'positions', 'velocities', ...]
dir(s.objects)               # the same fields, plus the methods
```

```text
Sample v0/L0/drop/91739_multi/invalid_continuity_strong
  s.objects
    ids                  int32 [8]
    names                list[8]
    positions            float32 [8, 25, 3]
    ...
  s.violation
    severity_bin         'strong'
    mask                 [T,H,W] on access
    severity_map         [T,H,W] on access
```

`on access` marks a field the loader derives or reads lazily: it has no shape
until you ask for it, so its axes are shown instead.

Every namespace also answers `ns["key"]`, `keys()` and `get()`. For training,
name the fields and let `to_dict`/`collate` build the same nesting as plain
dicts, padding every per-object field to the batch's largest N:

```python
ds = PhysLocDataset("data/physloc", fields=["video.rgb", "violation.mask",
                                            "violation.severity_map", "violation.severity"])
batch = collate([ds[i] for i in range(8)])
batch["video"]["rgb"]               # uint8 [B,T,H,W,3]
batch["violation"]["mask"]          # bool  [B,T,H,W]
batch["violation"]["severity"]      # float [B,N,T], NaN-padded
batch["objects"]["valid"]           # bool  [B,N], which rows are real
batch["info"]["uid"]                # list of B

pairs = PhysLocDataset("data/physloc", unit="pair")
pairs[0]                            # {"pair_uid", "prompt", "valid", "invalid": [...]}
```

`loader.FIELDS` lists every selector with its axes. HDF5 handles open lazily per
process, so multi-worker loading is safe (`loader.torch_dataset(ds)` wraps it for
a `DataLoader` with `collate_fn=collate`).

To look at the same API visually, see the [Quick start](#quick-start).

## Before you train on it

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

### How much is generated

A *cell* is one (scenario, family) pair. Per variant, each cell renders one invalid clip per
severity bin (families with no magnitude axis render only `strong`), and each scenario renders one
valid twin. A variant is a **fresh seed**, and each complexity level draws from its own seed block,
so no two levels share a scene.

---

## What varies, and how it is labelled

Four things vary between clips, and each is recorded on every one: **severity** (how badly the law
is broken), **complexity** (how realistic the scene is), **condition** (how many bodies, whether
the camera moves) and a measured **detection difficulty** (how hard the violation is to find).
Severity and complexity are independent knobs; the condition is a third; difficulty is what
actually came out.

### Severity

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

### Complexity ladder

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

### Difficulty conditions

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
unanswerable. Under a moving camera `flow` and `depth` include camera motion; the per-frame
camera poses ship in `data.h5` (`/camera`) and the intrinsics in `sample.json`
(`s.scene.camera` reads both).

---

### Detection difficulty

Conditions are the **knob** — what was asked for. `difficulty` is the **measurement** — what came
out. A clip with eight distractors whose violator fills a quarter of the frame is not hard; a
`standard` clip whose two-frame violation happens behind a screen is. `difficulty` is to
`condition` what `peak_severity` is to `magnitude`.

Every **invalid** sample carries one label (a valid twin has nothing to detect)
under `violation.difficulty` in `sample.json` (`s.violation.difficulty`), beside the two
array-measured inputs it was computed from:

```json
"difficulty": {
  "level": "hard", "rank": 2,
  "binding_factors": ["violation_area"],
  "factors": {"violation_area": {"value": 0.0041, "level": "hard"},
              "severity":  {"value": 0.98,   "level": "easy"}, "...": {}},
  "inputs": {"violation_area": 0.0041, "occlusion": 0.0}
}
```

#### Seven factors; a clip takes its worst

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

#### Where the cuts come from

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

#### Each violator also carries its own label

A `multi` clip whose violators are one large obvious body, one small one and one behind a screen
is not described by any single word, so every violator carries its own label under
`violation.violators[k].difficulty`, measured on **its** mask, **its** occlusion, **its**
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

#### Thresholds

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

### Scene variation

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
invalid siblings never land in different splits. The exporter groups pairs by scenario, orders the
pairs inside each scenario by a SHA-256 hash of the `pair_uid`, then cuts that ordered list by the
split fractions. That keeps every scenario in roughly the same proportions in each split, and a
re-generated release reproduces the same assignment. The `held_out` split is the dataset evaluation
split; it is unrelated to any asset-level `held_out` flag on scanned objects. Every split ships every
annotation — for a blind leaderboard set, strip annotations at that point.

### The index

`index.parquet` has one row per sample. It keeps relative paths to the MP4 and
HDF5 payloads instead of embedding duplicate bytes:

```python
import pandas as pd
df = pd.read_parquet("index.parquet")

df.groupby(["domain", "severity"]).size()
df.groupby("condition").size()                       # standard / camera / multi / ...
df[df.level == "L3"].groupby("family").size()
df[df.n_violators > 1]                                # multi-object samples
df.groupby(["level", "difficulty"]).size()           # the grid worth reporting
```

| group | columns |
|---|---|
| identity | `sample_uid`, `pair_uid`, `valid_uid`, `label`, `split`, `release`, `tier` |
| taxonomy | `scenario`, `family`, `domain`, `physics_medium` |
| violation | `severity`, `n_violators` (the violators the annotation marks) |
| difficulty | `difficulty`, `level` |
| scene | `condition`, `n_objects`, `n_actors`, `n_distractors`, `prompt` |
| geometry | `num_frames`, `fps`, `resolution`, `seed`, `variant` |
| storage | `sample_path`, `rgb_path`, `data_path` (relative paths) |

Group by `difficulty` in the index. For the complete measured distribution,
zone counts, thresholds, and the factors that set each label, read
`dataset.json["dataset_metadata"]["difficulty_analysis"]`.

### Evaluating

- **Report per family, aggregate to domain,** and cross with severity, complexity and condition.
  Do not pool families into one number: cell counts per domain are very uneven (see
  [Taxonomy](#taxonomy)), so a pooled score mostly measures the largest domain.
- **Use `violation.timeline["active"]` for temporal metrics and `violation.mask` for spatial ones.** They
  disagree on purpose.
- **Some families are separable by residual, some only by situation.**
  `taxonomy.EXCLUSIVE_LAWS` names the ones with a clean tripwire (`tests/test_orthogonality.py`
  enforces it). `antigravity`, `phantom_impulse` and `newton1_inertia` all move linear momentum
  because they must; what separates them has to be read from the image.

---

## Building it yourself

The published dataset was generated with this repository, and every clip records the commit,
Kubric image digest and Blender version that made it (`s.info.provenance`). Output is byte-identical
at any worker count, and a run resumes where it stopped.

```bash
python -m physloc.cli taxonomy --config v0_release   # price it first
bash scripts/run.sh v0_release                       # generate, validate, visualise, package
```

The full release takes days on a CPU box (`taxonomy --config v0_release` prices it); `v0_mini` is the same
structure in miniature, and `v0_L0` … `v0_L3` generate one complexity level each. Configs, tiers,
costs on other machines, following / stopping / resuming a run, and the Hugging Face upload are in:

- [docs/generating.md](docs/generating.md) — configs, running, cost and performance
- [docs/publishing.md](docs/publishing.md) — validate, export, upload, verify

---

## Licensing and citation

| what | licence | where it is recorded |
|---|---|---|
| annotations, renders and metadata | `CC-BY-4.0` by default; `export --license` overrides | `LICENSE`, `dataset.json` |
| scanned objects (**L3**) | Google Scanned Objects, `CC BY-SA 4.0` | per asset, `objects.records[i].asset.license` |
| HDRI environments (**L2, L3**) | HDRI Haven | environment name in `scene.environment.hdri`; **licence string not yet recorded** |

Two things to settle before v0 is published: L3 clips contain `CC BY-SA 4.0` assets, so check that
the licence you export under is compatible with that share-alike term; and HDRI environments do not
yet carry a licence string the way scanned objects do, which `physloc validate` should enforce for
every asset source.

To cite this dataset (provisional until a release exists — the repository and Hub id are
placeholders):

```bibtex
@misc{physloc,
  title  = {PhysLoc: a spatio-temporally annotated physics-violation video dataset},
  author = {Ruffino, Samuele},
  year   = {2026},
  note   = {Dataset, schema v4. <owner>/<repo>}
}
```

---

## Project documentation

| document | for |
|---|---|
| [docs/schema.md](docs/schema.md) | the sample layout, every field, axis and unit, and the loader contract |
| [docs/generating.md](docs/generating.md) | configs, running a release, cost and performance |
| [docs/publishing.md](docs/publishing.md) | validating, packaging and uploading to Hugging Face |
| [docs/development.md](docs/development.md) | the two environments, tests, repository layout |
| [docs/energy.md](docs/energy.md) | how the energy annotation is computed, and its limits |
| [docs/performance.md](docs/performance.md) | how every cost and speed number was measured |
| [docs/PLAN.md](docs/PLAN.md) | the design reasoning behind the annotations |
| [docs/roadmap.md](docs/roadmap.md) | what is next |

---

## References

[IntPhys 2](https://arxiv.org/abs/2506.09849) · [LikePhys](https://arxiv.org/abs/2510.11512) ·
[Kubric](https://github.com/google-research/kubric). The IntPhys 2 category and LikePhys domain
each family maps to are data, on each family in `physloc/taxonomy.py`.
