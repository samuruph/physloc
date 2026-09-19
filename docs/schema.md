# PhysLoc dataset schema

PhysLoc has one dataset format: **schema version 4**. Generation, export,
loading, validation, visualisation and the browser GUI all use it directly.
There is no reader for older versions; regenerate instead.

**Every fact is stored once**, at the most specific level where it varies.
Anything that can be computed from what is stored -- clip-level windows and
times, the `[N,T]` clocks, causal relations, object counts, the severity map --
is computed by `physloc/loader.py`, the one implementation of each. Every
derivation that replaced a v3 field was checked against all 2,135 v3 samples in
`out/` before the field was dropped; the fields whose derivation did not match
everywhere are still stored (see [What v4 stopped storing](#what-v4-stopped-storing)).

## Directory layout

```text
dataset-root/
├── README.md  LICENSE  loader.py
├── dataset.json
├── schema.json
├── index.parquet                 # index.jsonl only when PyArrow is unavailable
├── splits/{main,held_out,debug}.txt
└── samples/
    └── <sample_uid>/             # <release>/<level>/<scenario>/<seed>_<condition>/<valid|invalid_*>
        ├── sample.json           # metadata and annotations
        ├── rgb.mp4               # H.264, standalone
        └── data.h5               # every dense or per-frame array
```

A sample directory is self-contained: copy one out and it loads. Twins share
camera, physics and object statics, and each carries its own copy on purpose.

## Dataset metadata and index

`dataset.json.dataset_metadata` records what is constant across a release:
counts, split definitions and distributions, complexity distribution, taxonomy,
units, coordinate convention, floor datum, storage format, generation config
ids (`<release>:<tier>`), analysis groups, violation-component codes, the
shadow threshold, licence, `difficulty_analysis` (the release-wide difficulty
report) and `energy_accounting` (which bodies contribute to which total).

`index.parquet` has one row per sample: `sample_uid`, `pair_uid`, `valid_uid`,
`label`, `split`, `scenario`, `family`, `domain`, `physics_medium`, `level`,
`condition`, `severity`, `difficulty`, `seed`, `variant`, `num_frames`, `fps`,
`resolution`, `n_objects`, `n_actors`, `n_distractors`, `n_violators`,
`prompt`, `release`, `tier`, and relative `sample_path`, `rgb_path`,
`data_path`. Video bytes are never in Parquet.

Splits are deterministic, stratified by scenario, and assigned per `pair_uid`,
so a valid sample and its invalid siblings never cross a split.

## `sample.json`

Keys are sorted; a list of numbers (a window, a vector, a matrix row) is kept on
one line.

```text
schema_version            4
sample                    who it is
  uid  pair_uid  valid_uid  label  split  release  tier  seed  variant  framing_attempt
video                     clip geometry
  num_frames  fps  resolution  latent{frames, hw}
scene                     what was simulated and how it was shot
  scenario  family  domain  physics_medium  level  condition  prompt  size_scale
  camera{motion, K, field_of_view, focal_length, sensor_width, look_at, position, end_position}
  environment{hdri, background, lighting}
  physics{engine, gravity, step_rate, substeps_per_frame, params}
objects[]                 list order == row order of every [N,...] array in data.h5
  id  name  category  role  analysis_group
  asset{id, source, license, held_out}
  physics{mass, friction, rolling_friction, restitution, static, collidable,
          scripted, dormant, energy_eligible}
  render{material, color, scale}
violation                 invalid samples only
  severity_bin  kind  component  spatial_extent  timing  causal_ids
  intervention{type, magnitude, unit, params}
  difficulty{level, rank, binding_factors, factors, inputs{violation_area, occlusion}}
  peak_residual{law, value, z_vs_valid, score, frame}
  occluded_at_event  consequences  shadow?
  violators[]
    id  affected_ids
    windows{active, intervening, consequence, observable, occluded}   # inclusive [a, b] frames
    peak_residual  peak_severity  occluded_at_event  frames_visible_after_event
    difficulty  magnitude?        # magnitude only where it differs from the plan's
provenance                generator_commit  kubric_image_digest  blender_version  prefix_*
```

A valid sample points `valid_uid` at itself and has no `violation` block.
`scene.physics.params` is the resolved generation knobs (`configs/*.yaml`),
kept per sample so a sample records what it was made under.

### Objects and analysis groups

Pixel id `0` is background and is not an object. Every public object has one
positive `id`, shared across twins and used by segmentation, trajectories,
energy, events, violations and causal relations. The four `analysis_group`s:

- `subject`: experiment-relevant actors; every violator is a subject.
- `context`: distractors and other dynamic context.
- `support`: floors, ramps, tables, barriers, and passive receiving geometry.
- `background`: scenery not normally analysed.

## `data.h5`

Every dataset carries `axes` (named leading axes, then the size of each
trailing one, e.g. `N,T,8,3`) and `units` where it has one. Non-scalar arrays
use shuffle, gzip 4 and Fletcher32; images are chunked per frame, per-object
arrays per object and 32 frames.

```text
/observations                               render passes
  segmentation              uint16 [T,H,W], background = 0
  depth                     float  [T,H,W,1], m
  forward_flow backward_flow float [T,H,W,2], pixels
  normal object_coordinates uint16 [T,H,W,3]
  shadow_strength           float16[T,H,W]      shadow scenarios only
  shadow_source_id          uint16 [T,H,W]      shadow scenarios only
/camera
  positions                 float  [T,3], m     camera-to-world, Kubric convention
  quaternions               float  [T,4]        w,x,y,z, looking down -Z
/objects
  ids                       int32  [N]          == sample.json objects order
  positions velocities angular_velocities  [N,T,3]
  quaternions [N,T,4]  bboxes [N,T,4]  bboxes_3d [N,T,8,3]
  image_positions [N,T,2]  visibility [N,T]
/energy
  scene/*                   float  [T], J       total, kinetic_*, dissipated, anomalies, energy_in_frame
  objects/*                 [N,T,...]           kinetic potential by_body (J), mass (kg), height (m),
                                                inertia, momentum, angular_momentum, in_frame
  map                       float  [T,H,W], J
/events/collisions                              one array per field, E collisions
  frame [E]  instances [E,2]  force [E]  position [E,3]  contact_normal [E,3]  image_position [E,2]
/violation                                      invalid samples only
  maps/object_id            uint16 [T,H,W]      violator id per pixel; > 0 is the violation mask
  maps/component            uint8  [T,H,W]      0 none 1 body 2 shadow 3 trajectory 4 interaction 5 energy
  maps/causal_level         uint8  [T,H,W]      1 violator, 2 a body it affected
  maps/causal_source_id     uint16 [T,H,W]      the violator responsible
  maps/severity             float16[T,H,W]      SHADOW components only (see below)
  objects/severity residual score  float [N,T]  zero rows for non-violators
```

## What v4 stopped storing

| v3 field | why | in v4 |
|---|---|---|
| `sample_info.schema_version`, `render_seed`, `dataset_version`, `generation_config_id`, `duration_seconds` | repeated root / `seed` / `release` / `release:tier` / `T/fps` | `sample`, `video.duration` |
| `scene.info.label`, `variant`, `complexity`, `severity`, `difficulty` | repeated `sample`, `level`, `violation.severity_bin`, `difficulty.level` | one copy each |
| `difficulty_analysis` + `difficulty_inputs` | two halves of one measurement | `violation.difficulty` (with `inputs`) |
| `objects[].temporal_info/energy/violation` `{store, group, row}` | row is always the list index | list order |
| `objects[].violation.*` | re-keyed copy of the violator record | `violation.violators[]` |
| `static_fields.draw_scale`, `camera_visible`, `shadow_visible` | equal to `dimensions`, always true | `render.scale` |
| `objects_summary` | derivable; its `n_violators` was `len(causal_body_ids)`, e.g. 5 on a clip with 1 violator | counts from `objects` / `violators` |
| summary `t_*_frame`, `*_windows`, `observability_lag_frames` | min / max / union of the violators' windows | `Violation.t_event`, `.windows`, ... |
| violator `t_event_frame`, `t_observable_frame`, `observability_lag_frames` | first `active` frame; first `observable` frame at or after it (else `t_event`) | derived per violator |
| `causal_relations` | violator -> each `affected_ids`, over its consequence windows | `Violation.causal_relations` |
| `physics.units`, `timestep_seconds`, `floor_datum`, `storage`, `observations.{rgb,dense}` | constant | `dataset.json`, loader constants |
| `events.collisions` (85% of a v3 `sample.json`) | per-frame data | `/events/collisions` |
| camera `positions`, `quaternions` | per-frame data | `/camera` |
| `/violations/objects/{is_violator, active, intervening, consequence, observable, affected}` | rasterised from the violator records | `Violation.<clock>`, `.affected` |
| `/violations/maps/severity` for a body | equal to painting `objects/severity` over the mask | `Violation.severity_map` |
| `/energy/objects/momentum_magnitude` | `|momentum|` | `Energy.objects["momentum_magnitude"]` |
| all-zero `/violations` on a valid sample | nothing to say | absent; the loader returns zeros |

**Still stored**, because the derivation did not match on every v3 sample:
`/energy/objects/by_body` (not `kinetic + potential` on a third of samples),
`/energy/objects/mass` (varies over time under deformation and fission),
`causal_ids` (the plan's causal bodies, not violators plus affected),
per-violator `magnitude` on `multi` clips, the summary `peak_residual` and
`occluded_at_event`, per-violator `frames_visible_after_event`, and the severity
map of a shadow component. `occluded` is now stored as a violator window, which
it never was in v3 JSON.

Two v3 inconsistencies this resolves: a violator whose shadow caster was
remapped onto itself listed itself as `affected`, while the HDF5 `affected`
array did not; and one of 1,939 v3 summaries had `observable_windows` that were
not the union of its violators' windows. v4 uses the per-violator record in
both cases. The writer checks every stored window against the dense clocks it
was computed from, and refuses to write a sample where they disagree.

## Shadows

A shadow is an optical observation, never a physical object. The renderer's
camera-hidden Cycles caster is excluded from public objects, segmentation,
boxes, physics and energy, and its id is remapped to the real actor everywhere
(violator records, causal ids, collisions). For a shadow family
`violation.component` is `"shadow"` and `violation.shadow` records the caster,
light, receivers, render method and the `shadow_strength > 1/255` threshold.

## Loader

```python
from physloc.loader import PhysLocDataset, collate

ds = PhysLocDataset(root, split="main", family="solidity")
s = ds[0]                                  # a lazy Sample
s.info.uid, s.info.label, s.info.split
s.video.num_frames, s.video.fps, s.video.rgb       # rgb decoded on access
s.scene.scenario, s.scene.level, s.scene.camera.positions
s.observations.segmentation, s.observations.depth
s.objects.ids, s.objects.records, s.objects.positions
s.objects.select("violators"), s.objects.spatial_mask("support"), s.objects.record(2)
v = s.violation                            # empty (present=False) on a valid sample
v.severity_bin, v.violators, v.windows, v.t_event, v.observability_lag
v.mask, v.visible, v.severity_map, v.reference_mask, v.causal
v.active, v.observable, v.severity         # [N,T]
v.timeline, v.latent_grid()
s.energy.scene, s.energy.objects, s.energy.map
s.events.collisions
s.twin, s.divergence                       # divergence: inspection only, never a target
```

Every namespace supports `ns.key` and `ns["key"]`, `keys()`, `get()` and
`as_dict()`. `s.to_dict(fields)` returns the same nesting as a plain dict --
`{"info": {...}, "violation": {"mask": ...}, ...}`, always with `info` -- and
`collate` batches those, stacking equal shapes and padding every per-object
field (axes starting with `N` in `loader.FIELDS`) to the batch's largest N,
with `objects.valid` marking real rows. `fields=` selectors are listed in
`loader.FIELDS`; naming a whole namespace (`"scene"`, `"violation"`) gives its
metadata, not its arrays.

HDF5 opens lazily with process-local handles, so DataLoader workers are safe.
The visualiser, GUI, validator, exporter and release stats all read samples
through this interface.

## Validation rules

`physloc validate` checks the version, required blocks and fields, safe paths,
the three files, `/objects/ids` against `objects` order, segmentation shape,
dtype and ownership, one camera pose per frame, collision ids, that a valid
sample has neither a `violation` block nor a `/violation` group, that every
violator is a declared `subject` with windows inside `[0, T)` and declared
affected ids, map shapes, that `object_id` names only violators, component
codes, shadow invariants, and pair membership. An older layout fails with
"no schema-v4 samples"; it is never interpreted.
