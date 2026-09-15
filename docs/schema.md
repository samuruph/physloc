# PhysLoc clip layout — `metadata.json` and the arrays beside it

The human reference for one released clip. `physloc validate` enforces the cross-checks at the
bottom. The annotation *design* — why each field exists — is [PLAN.md](PLAN.md) Part 3.

**Schema version: 1.** Version 0 was a flat `meta.json` with PhysLoc-only file names.
Version 1 follows Kubric's **MOVi** datasets
([`refs/kubric/challenges/movi/README.md`](../refs/kubric/challenges/movi/README.md)) wherever
MOVi has a name for a thing, so a MOVi reader finds the passes where it expects them, and adds
PhysLoc's annotations beside MOVi's blocks. File names live in `physloc/annotate/layout.py`.

## One clip directory

`clips/<release>/<level>/<scenario>/<seed>_<condition>/{valid | invalid_<family>_<bin>}/`

| file | MOVi | valid | invalid | contents |
|---|---|---|---|---|
| `metadata.json` | ✓ | ✓ | ✓ | everything below |
| `video.mp4` | ✓ (`video`) | ✓ | ✓ | RGB |
| `segmentations.npz` | ✓ | ✓ | ✓ | `segmentations` uint16 `[T,H,W]`, declared instance ids, **0 = background** |
| `depth.npz` | ✓ | ✓ | ✓ | `depth` float32 `[T,H,W,1]`, metres from the camera centre |
| `forward_flow.npz` | ✓ | ✓ | ✓ | `forward_flow` float32 `[T,H,W,2]`, pixels, `(row, col)`, `t → t+1` |
| `backward_flow.npz` | ✓ | ✓ | ✓ | `backward_flow` float32 `[T,H,W,2]`, pixels, `(row, col)`, `t → t-1` |
| `normal.npz` | ✓ | ✓ | ✓ | `normal` uint16 `[T,H,W,3]`, `0..65535` maps to `-1..1` |
| `object_coordinates.npz` | ✓ | ✓ | ✓ | `object_coordinates` uint16 `[T,H,W,3]` |
| `instances.npz` | ✓ (`instances[*]` tensors) | ✓ | ✓ | per-instance, per-frame arrays — see below |
| `traj.npz` | | ✓ | ✓ | the simulator trajectory (the seam file) |
| `bodies.npz`, `energy.npz`, `energy_map.npz` | | ✓ | ✓ | physical quantities and mechanical energy |
| `reference_mask.npz` | | ✓ | ✓ | where the violators lawfully are |
| `timelines.npz`, `residuals.npz` | | ✓ | ✓ | per-frame timelines and residuals (zeros on a valid clip) |
| `violation_mask.npz`, `mask_invalid.npz`, `causal_mask.npz` | | | ✓ | the masks |
| `violation_ids.npz`, `causal_ids.npz` | | | ✓ | which violator each masked pixel belongs to |
| `severity_map.npz`, `divergence_map.npz`, `grids.npz` | | | ✓ | severity, divergence (not GT), latent grids |
| `overlay.mp4` | | | optional | nine-panel review video |

MOVi's raw data (`rgba`, `data_ranges.json`) is not shipped: depth and flow are stored as float32
in real units, so no range table is needed.

## `metadata.json`

```json
{
  "metadata":    { "schema_version": 1, "clip_uid": "...", "label": "invalid", ... },
  "camera":      { "focal_length": 50.0, "sensor_width": 36.0, "K": [[...]], "positions": [...], ... },
  "instances":   [ { "id": 2, "name": "ball", "asset_id": "sphere", "mass": 1.0, ... } ],
  "events":      { "collisions": [ { "instances": [1, 2], "frame": 9, "force": 12.3, ... } ] },
  "segmentation": { "encoding": "instance", "background_id": 0, "id_to_name": {...} },
  "violation":   { "t_event_frame": 7, "violator_timing": "independent", "violators": [...], ... },
  "difficulty":  { "level": "moderate", "rank": 1, ... },
  "energy":      { "E0": ..., ... },
  "noise_floor": { "position_continuity": {...} },
  "provenance":  { "prefix_identical_verified": true, ... },
  "real2sim":    null,
  "files":       { "segmentations.npz": {"arrays": {"segmentations": {"dtype": "uint16", "shape": [25, 128, 128]}}}, ... }
}
```

### `metadata` — identity, taxonomy, geometry

| field | notes |
|---|---|
| `schema_version` | `1`. Check it before trusting anything else. |
| `clip_uid`, `pair_uid`, `twin_uid` | `twin_uid` is null on a valid clip, which is shared by every family on its scene. **Never split a pair across splits.** |
| `label` | `valid` \| `invalid` |
| `tier`, `release` | geometry (`debug` / `release`) and what the published dataset is called |
| `domain`, `family` | null on a valid clip |
| `scenario`, `seed`, `variant`, `condition` | `condition` is one of `standard`, `camera`, `distractors`, `multi`, `camera+multi` |
| `n_distractors`, `n_actors`, `n_violators` | what landed in the scene |
| `physics_medium` | `rigid` \| `granular` (`pour`). Never `fluid`. |
| `complexity` | the realism level block (`L0`..`L3`) |
| `params` | every generation knob in force |
| `size_scale` | the per-scene size multiplier (`params.objects.size_scale`); absent on `pour` |
| `framing_attempt` | which resampled scene this is when earlier draws let the actors leave the frame (0 = the seed's first scene) |
| `background` | `hdri_id` (L2+) and the flat background `color` |
| `intphys2_category`, `likephys_domain` | cross-references |
| `frame_rate`, `num_frames`, `resolution`, `step_rate`, `gravity` | MOVi's geometry fields |
| `prompt` | caption of the clip's *valid* physics |
| `controls` | `is_surprising_but_valid`, `is_artifact_probe` |

### `camera` — MOVi's, plus the aim point

| field | notes |
|---|---|
| `focal_length`, `sensor_width` | mm; Kubric's defaults, 50 and 36 |
| `field_of_view` | horizontal, degrees |
| `K` | 3×3, Kubric's **normalised** intrinsics (`PerspectiveCamera.intrinsics`); multiply by the resolution for pixels |
| `positions` | `[s,3]` world, per frame |
| `quaternions` | `[s,4]` `(w,x,y,z)`, camera-to-world, looking down −Z (`look_at_quat`) |
| `motion` | `static`, `track`, `orbit` or `dolly` |
| `look_at` | the aim point, fixed for the whole clip |
| `position`, `end_position` | where a moving camera starts and ends |

The eye is always at least 0.6 m above the floor (`base.CAMERA_MIN_HEIGHT`).

### `instances` — one record per body, in `spec.bodies` order

`id` (= `track_id`, the pixel value in `segmentations`), `name`, `category`, `asset_id`,
`source`, **`license`** (mandatory), `held_out`, `material`, `color`, `role`, `scale`,
`static`, `dormant`, `collides`, `mass`, `friction`, `restitution`, `is_violator`,
`violator_index` (position in `violation.causal_body_ids`), `bbox_frames`, `first_frame`,
`last_frame`, `frames_visible`, `pixels_peak`.

**Not re-sorted by visibility, unlike MOVi**: the valid and invalid twins must agree on who is
who. Row `i` of `instances` is row `i` of every array in `instances.npz`.

`role`: `actor` (a family may target it), `floor`, `prop`, `occluder`, `distractor` (scenery no
family targets), `shadow` (the scripted stand-in on `shadow_track`, carrying no energy),
`backdrop` (the HDRI dome at L2–L3, which covers the floor, so there the floor has no pixels).

### `instances.npz` — MOVi's per-frame instance tensors

| key | shape | |
|---|---|---|
| `ids` | `[k]` | the instance ids, in order |
| `positions`, `quaternions` | `[k,s,3]`, `[k,s,4]` | world; `(w,x,y,z)` |
| `velocities`, `angular_velocities` | `[k,s,3]` | |
| `bboxes_3d` | `[k,s,8,3]` | world corners, Kubric's corner order, following any resize |
| `image_positions` | `[k,s,2]` | `(x, y)` in `[0,1]`, y down (`Camera.project_point`) |
| `bboxes` | `[k,s,4]` | `(ymin, xmin, ymax, xmax)` in `[0,1]` from the segmentation; **NaN** where not visible |
| `visibility` | `[k,s]` | pixels per frame |

### `events.collisions`

One record per **(frame, pair of bodies)**: `instances` `[a, b]` (instance ids, not indices),
`frame`, `force` (largest over the frame's contact rows), `position` (force-weighted mean),
`image_position`, `contact_normal`. The trajectory stores a row per solver substep and contact
point; the events group them.

### `violation` — null on valid clips

| field | notes |
|---|---|
| `kind` | `instant` \| `sustained` \| `repeated` |
| `t_event_frame` | the **earliest** violator's moment; the law breaks in simulator state |
| `t_observable_frame`, `observability_lag_frames` | first frame with visual evidence, and the lag |
| `t_end_frame`, `t_intervention_end_frame`, `t_consequence_end_frame` | ends of the three window families |
| `violation_windows`, `intervention_windows`, `consequence_windows`, `observable_windows` | lists of inclusive `[s, e]`; the clip's are the union over violators |
| `occluded_at_event` | whether the primary violator was hidden |
| `causal_body_ids` | instance ids; `[0]` is the primary violator the residual is scored on |
| `spatial_extent` | `global` for `global_gravity` only |
| `intervention` | `type`, `params`, `magnitude`, `magnitude_unit`, `severity_bin` |
| `peak_residual` | `law`, `value`, `z_vs_valid`, `score`, `frame` |
| **`violator_timing`** | `independent`, `sync` or `shared` — see below |
| **`violators`** | one record per dynamic violator, below |
| `difficulty_inputs` | the two array-derived difficulty values |

**Each violator on its own clock.** In a `multi` clip whose family acts on each body separately,
every violator gets its own moment: `violator_timing: "independent"`. A quarter of such clips
(`params.objects.multi_sync_share`) deliberately give all violators one moment: `"sync"`.
Scene-wide families, granular media and families that act on a pair together are `"shared"`.

Each `violators[]` record: `instance_id`, `t_event_frame`, `t_observable_frame`,
`observability_lag_frames`, `violation_windows`, `intervention_windows`,
`consequence_windows`, `observable_windows`, `occluded_at_event`,
`frames_visible_after_event`, `magnitude`, `peak_residual`, `peak_severity`,
`affected_instance_ids` (level-2 bodies this violator reached first).

**When events happen.** Absent a physical cue, a moment is drawn per (scene, family, violator)
from 15–70% of the clip, never per severity bin, leaving at least 35% of the clip (0.8 s) for
the effect. Families about motion fire only before their actor comes to rest. The worker
rejects a moment after which the violators leave the shot and tries another; a scene whose
actors leave the frame early is resampled before anything renders.

**Why windows are lists.** `superelastic` fires once per bounce; an object can appear, re-hide
and re-emerge; staggered violators leave gaps. `t_event_frame` / `t_end_frame` are the min and
max across windows.

**Three clocks.** `intervention_windows` is when we are actively changing something;
`consequence_windows` is when the scene differs as a result (and may run past the evidence);
`violation_windows` is **where the evidence is** — the intervention window for a family whose
evidence is the change itself (`detectable == "event"`), the consequence window otherwise. So
`intervention_windows ⊆ violation_windows`.

`magnitude` is the knob turned (exact); `peak_residual.value` is the measured consequence.

### `segmentation`, `difficulty`, `energy`, `noise_floor`, `provenance`, `files`

- `segmentation`: `encoding`, `dtype`, `background_id`, `ids_are_declared`, `id_to_name`.
- `difficulty` (invalid only): `level`, `rank`, `binding_factors`, `factors` — seven factors,
  worst one wins; thresholds in `configs/common.yaml`, frozen per release.
- `energy`: `E0`, `E_end`, dissipation and the three anomaly peaks, as fractions of `E0`.
- `provenance`: `generator_commit`, `kubric_image_digest`, `blender_version`, `render_seed`,
  `prefix_identical_verified` (**measured** pixel by pixel over `[0, t_event)`),
  `prefix_differing_pixels`, `prefix_identical_upto_frame`.
- `files`: every file beside the metadata, with each array's key, dtype and shape.

## Masks — which side of the twin each one lives on

| array | side | |
|---|---|---|
| `violation_mask` | **union** of both twins | the training target; the only mask with pixels for a vanished body |
| `mask_invalid` | invalid only | where the wrong thing is, in the video a model sees |
| `reference_mask` | valid only | where the violators should have been; ungated in time |
| `severity_map` | invalid only | how badly, per pixel; each violator painted with its own score |
| `causal_mask` | invalid only | uint8: `1` a violator, `2` a body it measurably disturbed |
| `violation_ids` | as `violation_mask` | uint16 violator id per masked pixel; the invalid-side footprint wins on overlap |
| `causal_ids` | as `causal_mask` | uint16: the violator whose violation or consequence the pixel is |

`violation_mask[t] = footprint(violator, invalid, t) ∪ footprint(violator, valid, t)`, per
violator gated to that violator's own active **and** observable frames, then combined. A body that
vanished has no invalid footprint, so its severity falls back to its lawful footprint on those
frames only. `causal_mask` level 2 is measured: bodies that provably depart from the valid twin
*and* that a violator reached at or after its moment.

`divergence_map` is `|valid − invalid|` and diverges everywhere downstream of the event. **Never
a training target.**

## `timelines.npz` and `residuals.npz`

| key | shape | |
|---|---|---|
| `active`, `intervening`, `consequence`, `observable`, `occluded` | `[T]` bool | clip-level |
| `severity_t` | `[T]` f32 | `severity_map[t].max()` |
| `violator_ids` | `[K]` | violator instance ids, in `violation.violators` order |
| `violator_active`, `violator_intervening`, `violator_consequence`, `violator_observable`, `violator_occluded` | `[K,T]` bool | each violator's own timelines |
| `violator_severity_t` | `[K,T]` f32 | each violator's peak severity per frame |

`residuals.npz`: `r`, `z`, `s` `[T]` for the primary violator and `law`; plus `violator_ids`,
`violator_r`, `violator_s` `[K,T]`.

## `grids.npz`, `traj.npz`, `bodies.npz`, `energy.npz`

- `grids.npz`: masks and severity reduced to the latent token grid, time-major
  `[F_lat, H_lat, W_lat]`, `F_lat = (T - 1) / 4 + 1`.
- `traj.npz`: the seam — `pos`, `quat`, `lin_vel`, `ang_vel`, `present`, `scale_mul`,
  `colour`, `opacity`, contacts, `fps`, `gravity`, and the spec as JSON.
- `bodies.npz`: the quantities the energy was computed from; `mass [T,B]` follows volume.
- `energy.npz` / `energy_map.npz`: the trace and the per-pixel map. See [energy.md](energy.md).

**Two encodings that bite:** depth's background is a sentinel (~`1.1e10`), so mask with
`segmentations > 0`; flow is `(row, col)`, not `(x, y)`.

## Cross-checks enforced by `physloc validate`

0. `metadata.schema_version == 1`
1. every instance carries a non-empty `license` (both twins)
2. `t_event_frame ≤ t_observable_frame` and `t_event_frame ≤ t_end_frame`
3. `violation_windows` sorted, non-overlapping, within `[0, num_frames)`; `t_event_frame` and
   `t_end_frame` equal their extremes
4. `timelines.active` is exactly the rasterisation of `violation_windows`; likewise `observable`
5. `violation_mask` is non-empty on every `active ∧ observable` frame, and empty on
   `active ∧ ¬observable` frames
6. `severity_t[t] == severity_map[t].max()`
7. every `causal_body_id` is an instance; `newton3_reaction` has ≥ 2
8. `spatial_extent == "global"` iff `family == "global_gravity"`
9. `domain` matches `family`; `(scenario, family)` is in the compatibility matrix;
   `physics_medium == "granular"` iff `scenario == "pour"`
10. `intervention_windows ⊆ violation_windows`; `mask_invalid ⊆ violation_mask`
11. `provenance.prefix_identical_verified` is true
12. `violation` is null iff `label == "valid"`
13. every pair has one valid and at least one invalid clip
14. each violator is a causal body, with `t_event ≤ t_observable` and in-range windows; the
    earliest violator's moment is the clip's
15. `violation_ids` covers exactly `violation_mask` and names only violators
16. `instances.npz` has `k` rows in instance order and `num_frames` columns
