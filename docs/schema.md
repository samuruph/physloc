# PhysLoc dataset schema

PhysLoc has one dataset format: schema version 3. Generation, export, loading,
validation, visualisation, and the browser GUI all use this format directly.
There is no v2 reader or compatibility mode.

## Directory layout

```text
dataset-root/
├── README.md
├── LICENSE
├── dataset.json
├── schema.json
├── index.parquet                 # index.jsonl only when PyArrow is unavailable
├── splits/
│   ├── main.txt
│   ├── held_out.txt
│   └── debug.txt
└── samples/
    └── <sample_uid>/
        ├── sample.json
        ├── rgb.mp4
        └── data.h5
```

`sample_uid` may contain path components. All stored paths are relative to the
dataset root. RGB remains a normal H.264 MP4, making it streamable and easy to
inspect. Numeric annotations are consolidated into one chunked HDF5 file per
sample; there is no collection of small NPZ files.

## Dataset metadata and index

`dataset.json.dataset_metadata` records the dataset version, schema version,
sample and pair counts, split definitions and distributions, complexity
distribution, scenario/family/domain taxonomy, units, coordinate convention,
generation configuration IDs, analysis groups, violation-component codes,
shadow threshold, licence, checksum policy, `difficulty_analysis`, and the
explicit `energy_accounting` policy.

`difficulty_analysis` is the authoritative release-wide difficulty report. It
contains the invalid-sample count, easy/moderate/hard distribution, labels by
complexity, the factors that set labels, all factor thresholds, and—for each
factor—count, zone counts, min, p05, median, mean, p95, and max. The full
per-sample measurement remains in
`sample.json:metadata.scene.info.difficulty_analysis`; its scalar label is
`metadata.scene.info.difficulty`.

`energy_accounting` distinguishes the physics-wide `world_total` from the
camera-grounded `visible_scene_total` and records the per-object tensor and its
object-ID axis. Dynamic subjects, distractors/context, affected objects, and
peers contribute. Floors, backdrops, supports, barriers, walls, occluders, and
renderer-only shadow casters do not. This policy is stored in the release—not
left as an implicit visualizer convention.

`index.parquet` has one row per sample. It contains searchable scalar values:
sample, pair, and valid-reference IDs; split and label; scenario taxonomy;
condition, complexity, difficulty, and severity; seed and variant; frame
geometry; object counts; prompt; generation configuration; and relative paths
to the sample directory, `rgb.mp4`, and `data.h5`. Video bytes are never
duplicated in Parquet.

Split assignment is deterministic, stratified by scenario, and performed on
`pair_uid`, so a valid sample and its invalid siblings cannot cross a split.

## `sample.json`

```text
sample
├── metadata
│   ├── sample_info
│   └── scene
│       ├── info
│       └── world
├── observations
├── annotations
│   ├── objects
│   ├── scene_energy
│   ├── maps
│   ├── events
│   ├── causal_relations
│   └── violation_summary       # invalid samples only
└── storage
```

### `metadata.sample_info`

Required fields are `sample_uid`, `pair_uid`, `valid_sample_uid`, `label`,
`num_frames`, `fps`, and `resolution`. It also records schema and dataset
versions, split, seed and render seed, variant, generation-config ID,
provenance, duration, latent dimensions, framing attempt, and scene scale.

A valid sample points `valid_sample_uid` to itself. Every invalid sample points
to the valid sample for its pair.

### `metadata.scene`

`scene.info` contains level, scenario type, family, domain, physics medium,
condition, complexity, difficulty, severity, label, variant, and prompt.

`scene.world` contains:

- `camera`: intrinsics, projection information, coordinate convention, and the
  complete per-frame camera trajectory.
- `environment`: background, HDRI, lights, and floor/support datum.
- `physics`: gravity, timestep, step rate, substeps, solver and generation
  settings, and units.
- `objects_summary`: counts by analysis group, actor/distractor status,
  violator status, and affected status.

### Objects and analysis groups

Background pixel ID `0` is reserved and is not an object row. Every exported
object has one positive stable `id`, shared across valid/invalid siblings and
used by segmentation, trajectories, energy, events, violations, and causal
relations.

Each object definition contains identity (`id`, name, category, role), an asset
record (source and licence), `analysis_group`, `static_fields`, and row
references into the temporal, energy, and violation HDF5 groups.

The four analysis groups are:

- `subject`: experiment-relevant actors; every violator must be a subject.
- `context`: distractors and other potentially relevant dynamic context.
- `support`: floors, ramps, tables, barriers, and passive receiving geometry.
- `background`: exported scenery that is not normally analysed.

The loader defaults to subjects and exposes named object and spatial masks for
subjects, contexts, supports, violators, affected objects, and all objects.

## `data.h5`

Arrays carry `axes`, `units` where applicable, and a documented fill value.
Non-scalar arrays use shuffle, gzip level 4, and Fletcher32. Image tensors are
chunked by frame; object trajectories are chunked by object and time.

```text
/observations
  segmentation              uint16 [T,H,W], background = 0
  depth                     float  [T,H,W] or [T,H,W,1], metres
  forward_flow              float  [T,H,W,2], pixels
  backward_flow             float  [T,H,W,2], pixels
  normal                    float  [T,H,W,3]
  object_coordinates        float  [T,H,W,3]
  shadow_strength           float16[T,H,W], optional
  shadow_source_id          uint16 [T,H,W], optional

/objects
  ids                       int32  [N]
  positions                 float  [N,T,3], metres
  quaternions               float  [N,T,4], w,x,y,z
  velocities                float  [N,T,3], metres/second
  angular_velocities        float  [N,T,3], radians/second
  bboxes                    float  [N,T,4]
  bboxes_3d                 float  [N,T,8,3]
  image_positions           float  [N,T,2]
  visibility                int    [N,T]

/energy/scene               scene time series and residuals
/energy/objects             per-object kinetic, potential, momentum, etc. [N,T,...]
/energy/map                 float [T,H,W]

/violations/objects
  is_violator               bool  [N]
  active                    bool  [N,T]
  intervening               bool  [N,T]
  consequence              bool  [N,T]
  observable                bool  [N,T]
  occluded                  bool  [N,T]
  affected                  bool  [N,T]
  severity                  float [N,T]
  residual                  float [N,T]
  score                     float [N,T]

/violations/maps
  violation_object_id       uint16[T,H,W]
  violation_component       uint8 [T,H,W]
  causal_level              uint8 [T,H,W]
  causal_source_id          uint16[T,H,W]
  severity                  float16[T,H,W]
```

Violation component codes are `0 none`, `1 body`, `2 shadow`, `3 trajectory`,
`4 interaction`, and `5 energy`. This explicitly permits a violation footprint
to differ from visible body segmentation.

## Per-object violations and causality

An invalid sample's `violation_summary` describes the violation type,
intervention, global intervals, first observable frame, peak, and its list of
violators. Every violator record names its responsible `instance_id`, target
component, severity, active/intervention/consequence/observable intervals, and
affected object IDs. The dense `[N,T]` tensors make per-object temporal
experiments direct; non-violators retain zero rows. The spatial maps make
selected-object and union localisation direct.

`causal_relations` stores source and target object IDs, relation type,
intervals, and confidence. An affected non-violator is therefore distinct from
the responsible object.

Valid samples contain all ordinary observations, objects, trajectories,
events, and energy, omit `violation_summary`, and carry zero violation tensors.

## Shadows

A shadow is an optical observation, never a physical object. The renderer uses
a camera-hidden Cycles caster associated with the real actor. That internal
caster is excluded from public objects, segmentation, bounding boxes, physics,
and energy.

For shadow experiments, a matched Cycles render isolates continuous
`shadow_strength`; `shadow_source_id` names the real actor. The dataset-wide
binary threshold is `shadow_strength > 1/255`. Shadow violation pixels use the
actor's ID in `violation_object_id` and component code `shadow`; they need not
match the support object's segmentation. Metadata records the actor, light,
receivers, render method, and threshold.

## Loader contract

```python
from physloc.loader import PhysLocDataset

dataset = PhysLocDataset("/path/to/dataset")
sample = dataset.samples[0]
same = dataset.get(sample.uid)

sample.video_path                 # no decoding
rgb = sample.decode_rgb()         # explicit decoding
sample.objects("subjects")
sample.objects("violators")
sample.objects("affected")
sample.object(2)
sample.spatial_mask("subjects")
```

HDF5 files open lazily with process-local handles, making worker processes safe.
Field presets support metadata, localisation, object localisation, energy, and
visualisation. Batch collation pads the object dimension and returns
`object_valid`. Pair access is an explicit dataset view based on `pair_uid` and
`valid_sample_uid`.

The visualiser and GUI consume this same `Sample` interface. They never open
HDF5 directly. Overlays are derived in memory; redundant overlay videos are
not part of a published dataset.

## Validation rules

`physloc validate` checks the version, required fields, safe paths, required
files, axes and shapes, dtypes, IDs, HDF5 references, pair membership, stable
object order, segmentation ownership, causal references, analysis groups, and
shadow invariants. Any old layout fails with a clear “no schema-v3 samples”
error; it is not interpreted or migrated implicitly.
