# PhysLoc `meta.json` schema

Versioned separately from [PLAN.md](PLAN.md) because the schema will evolve while the design
does not. The machine-readable copy lives at `physloc/schema/meta.schema.json`; this file is
the human reference. `physloc validate` enforces both this schema and the cross-checks at
the bottom. The annotation *design* — why each field exists — is [PLAN.md](PLAN.md) Part 3.

**Schema version: 0 (draft — frozen at the end of Phase 1).**

## Example

```json
{
  "clip_uid": "physloc_v0/collision/0173/invalid_solidity_a",
  "pair_uid": "physloc_v0/collision/0173",
  "twin_uid": "physloc_v0/collision/0173/valid",
  "label": "invalid",
  "tier": "v0", "release": "physloc_v0",
  "domain": "contact", "family": "solidity", "scenario": "collision", "seed": 91731,
  "intphys2_category": "solidity", "likephys_domain": "rigid_body",
  "fps": 12, "num_frames": 25, "resolution": [256, 256],
  "prompt": "two red cubes rolling toward each other and colliding head-on",
  "camera": {"intrinsics": [], "extrinsics_per_frame": [], "motion": "static",
             "position": [0.3, -6.0, 1.7], "look_at": [0.1, 0.0, 0.35]},
  "violation": {
    "kind": "sustained",
    "t_event_frame": 9, "t_observable_frame": 13, "observability_lag_frames": 4,
    "t_end_frame": 16, "occluded_at_event": true,
    "violation_windows": [[9, 16]],
    "observable_windows": [[13, 16]],
    "causal_body_ids": [3, 5],
    "spatial_extent": "local",
    "intervention": {"type": "disable_collision_pair", "params": {"pair": [3, 5]},
                     "magnitude": 0.043, "magnitude_unit": "m_penetration_depth",
                     "severity_bin": "medium"},
    "consequences": [{"body_id": 7, "t_diverge_frame": 14, "t_observable_frame": 15,
                      "displacement_m": 0.21, "relation": "struck_by_causal_body"}],
    "peak_residual": {"law": "penetration", "value": 0.043, "z_vs_valid": 41.2,
                      "score": 0.62, "frame": 12}
  },
  "controls": {"is_surprising_but_valid": false, "is_artifact_probe": false},
  "assets": [{"name": "...", "source": "polyhaven", "license": "CC0"}],
  "provenance": {"generator_commit": "…", "kubric_image_digest": "sha256:…",
                 "blender_version": "2.93.4", "render_seed": 91731,
                 "prefix_identical_verified": true, "prefix_identical_upto_frame": 9},
  "real2sim": null
}
```

## Fields

### Identity

| field | type | notes |
|---|---|---|
| `clip_uid` | str | `<release>/<scenario>/<seed:04d>/<label>[_<family>][_<variant>]`; also the directory name |
| `pair_uid` | str | shared by a valid/invalid twin pair |
| `twin_uid` | str | the counterpart clip. **Never split a pair across train/val/test.** Several invalid variants may share one valid twin (same scenario+seed → bit-identical valid render), so `pair_uid` groups one valid with N invalids. |
| `label` | `"valid"` \| `"invalid"` | clip-level ground truth |
| `tier` | `"v0"` \| `"v1"` | `v0` = 256²/12 fps/49 f; `v1` = 512²/24 fps/97 f. `debug` (128²/25 f) is the iteration loop and is **never published**. Renamed from the old A/B/D letters, which had no C and ran backwards. |
| `release` | str | what the published dataset is CALLED, e.g. `physloc_v0`. Set by `--outdir`. **Not a tier** -- the tier is the geometry (`debug` or `release`) and difficulty is the complexity ladder |
| `seed` | int | the seed for the whole sampling + render path |

### Taxonomy (PLAN Part 2)

| field | type | notes |
|---|---|---|
| `domain` | str | one of 7: `identity`, `kinematics`, `contact`, `dynamics`, `equilibrium`, `optical`, `global` |
| `family` | str | one of 16, e.g. `solidity`, `non_parabolic`, `newton3_reaction` |
| `scenario` | str | one of 13, e.g. `collision`, `occluder_pass`, `pour` |
| `complexity` | object | `{name, background, actor_assets, materials, motion_blur, share, camera_share, distractor_share, n_distractors}` -- the scene-realism rung (L0 baseline .. L3 HDRI+GSO). Orthogonal to severity. Camera motion and distractors are ratios INSIDE each rung, not rungs; what this clip actually got is the top-level `camera_motion` and `n_distractors`. |
| `hdri_id` | str \| null | HDRI Haven environment id, from complexity L1 up |
| `physics_medium` | `rigid` \| `granular` | `granular` for `pour`. **Never `fluid` in v0** — see PLAN Part 2. Prevents a granular scenario being mistaken for an SPH benchmark. |
| `intphys2_category` | str \| null | cross-reference; null is a claim of novelty |
| `likephys_domain` | str \| null | cross-reference |

`domain` is derivable from `family`, and `(scenario, family)` must be a `●` cell in the
compatibility matrix. Both are checked by `physloc validate` against `physloc/taxonomy.py`.
Novelty claims (a `null` cross-reference) must be justified in [prior_art.md](prior_art.md).

### Clip properties

`fps` (int), `num_frames` (int), `resolution` (`[H, W]`), and `camera` with `intrinsics`,
`extrinsics_per_frame` (length `num_frames`), `motion` (always `"static"` in v0 -- the camera
does not move during a clip; `orbit`/`linear` are declared on `Complexity.camera_motion` for a
later complexity tier and are not yet built), and the actual sampled `position`/`look_at` the
scenario framed the shot from (world-space, metres -- see `physloc/camera.py`).

`prompt` (str) is a textual caption of the clip's *valid* physics, composed per clip from a
per-scenario template plus whatever shape and color that instance actually sampled --
`physloc/prompts.py::compose_prompt`. E.g. `"a red cube dropping and colliding with the
ground, in an empty background"`. Describes the lawful scene only; it says nothing about
`family`/`violation` even on an invalid clip; see `physloc/prompts.py`'s docstring for why
invalid captions are not composed the same way.

### `violation` — null on valid clips

| field | type | notes |
|---|---|---|
| `kind` | `instant` \| `sustained` \| `repeated` | shape of the intervention in time |
| `t_event_frame` | int | when the law breaks **in simulator state**. Exact by construction. |
| `t_observable_frame` | int | first frame with **visual evidence**. Exact, via prefix identity. |
| `t_end_frame` | int | last frame the violation is active |
| `observability_lag_frames` | int | `t_observable − t_event`. The occlusion-lag metric. |
| **`violation_windows`** | `[[s,e], …]` | **every interval where the law is actively broken** |
| **`observable_windows`** | `[[s,e], …]` | every interval where visual evidence is present |
| `occluded_at_event` | bool | whether the culprit was hidden when the law broke |
| `causal_body_ids` | int[] | instance ids in `seg.npz`. **`newton3_reaction` requires both bodies.** |
| `spatial_extent` | `local` \| `global` | `global` for `global_gravity`, whose mask is the whole frame |
| `intervention` | object | `type`, `params`, `magnitude`, `magnitude_unit`, `severity_bin` |
| `consequences` | object[] | `body_id`, `t_diverge_frame`, `t_observable_frame`, `displacement_m`, `relation` |
| `peak_residual` | object | `law`, `value` (physical units), `z_vs_valid`, `score` (bounded), `frame` |

**Why windows are lists.** `superelastic` fires once per bounce
(`[[12,13],[31,32],[47,48]]`); an object behind an occluder can appear, re-hide and re-emerge
(`observable_windows = [[19,26],[38,50]]`). `instant` violations are the degenerate single
window of length 1. `t_event_frame` / `t_end_frame` are the min and max across all windows
and are retained for consumers that only want a coarse interval.

**Invariant:** `t_event_frame ≤ t_observable_frame` and `t_event_frame ≤ t_end_frame`.
Observability arriving *after* `t_end` is ordinary rather than a bug: a super-elastic
bounce adds its energy on one frame and the body has not visibly moved by the time the
window shuts. `t_end` is when the intervention stops acting; when it becomes visible is
a separate question, which is why there are three clocks and not one.

`magnitude` is the knob we turned (exact); `peak_residual.value` is the measured consequence.
`severity_bin` (`weak`/`medium`/`strong`) is derived from `magnitude`, not from model
performance — splits are principled, not tuned. The names describe **how hard the law is
bent**, not how hard the clip is to classify. Those are different things: a large
intervention behind an occluder can be far harder to detect than a small one in plain view,
which is exactly what `observability_lag_frames` measures.

### `controls`

`is_surprising_but_valid` and `is_artifact_probe`. Both are `label: "valid"`; both exist so a
detector cannot score by flagging weirdness or by keying on the renderer.

### `assets`

Array of `{name, source, license}`. **`license` is mandatory and non-empty** — `physloc
validate` fails otherwise. This is what makes the Phase 5 audit a check rather than an
archaeology project.

### `provenance`

`generator_commit`, `kubric_image_digest`, `blender_version`, `render_seed`,
### `energy.npz` and `energy_map.npz`

Shipped on **both** twins, because the valid clip's trace is the baseline every anomaly is
judged against. `energy.npz`: `total[T]`, `kinetic_translational[T]`,
`kinetic_rotational[T]`, `potential[T]`, `by_body[T,B]`, `body_ids[B]`, `dissipated[T]`,
`free_anomaly[T]`, `contact_anomaly[T]`, `excess_loss[T]`. `energy_map.npz`: `energy[T,H,W]`,
each body's energy painted onto its own pixels through the segmentation pass — constant
inside a rigid body's silhouette, which is the honest spatial resolution.

### Three timelines, not two

A violation has three distinct clocks and the release used to carry two, conflating the first
with the second:

| field | meaning |
|---|---|
| `intervention_windows` | when we are **actively changing something** — the colour ramping, the body shrinking, the teleport happening. Ends when the change completes |
| `consequence_windows` | when the scene **differs from lawful as a result**. Runs on afterwards, and for `permanence` never ends |
| `observable_windows` | when a viewer **could tell**. Gated by occlusion; the gap from `t_event` is the observability lag |

`violation_windows` is **where the evidence is** — the frames `violation_mask` and
`severity_map` actually cover. It is *not* the union of the first two, which is what it used
to be, and the difference is not cosmetic: `annotate/pipeline.py` paints the mask on the
intervention window for a family whose evidence is the change itself
(`FAMILIES[f].detectable == "event"`), and on the consequence window otherwise. Declaring
the union made every `event` clip claim a violation on frames whose mask was provably empty
— `drop × colour_shift` shipped `violation_windows [[7, 24]]` with pixels on 7–12 — and
`validate` rejected all of them.

So the invariant is:

- `intervention_windows ⊆ violation_windows` — we cannot be changing something on a frame we
  are not calling violating.
- `consequence_windows` **may extend past** `violation_windows`. A teleported ball is
  somewhere else for the rest of the clip, but only the jump is visible *as a violation*.
  Consequences are not lost: they drive the blue `causal_mask` through
  `(consequence | diverged) & observable`, which is what keeps the mask alive for as long as
  behaviour is still different.

`timelines.npz` carries `intervening[T]` and `consequence[T]` alongside `active[T]`, and
`meta.json` adds `t_intervention_end_frame` and `t_consequence_end_frame`.

Concretely, `colour_shift` at strong on a 25-frame clip: `t_event` 9, intervention *and
violation* end at 14 when the colour has finished turning, consequence runs to 24. `friction`
and `support` are the other shape — they change a property that keeps acting, so their
intervention runs the whole clip and all three windows coincide. `permanence` is a third
shape: the body is *held* absent every frame, so its intervention is the whole absence rather
than the instant of disappearance.

### Masks — which side of the twin each one lives on

| array | side | |
|---|---|---|
| `violation_mask` | **union** of both twins | the documented training target. The only mask with pixels for a vanished body |
| `mask_invalid` | invalid only | where the wrong thing actually is, in the video a model will see |
| `reference_mask` | valid only | where it *should* have been. Ungated in time |
| `severity_map` | **invalid only** | how badly, per pixel |
| `causal_mask` | **invalid only** | `1` = culprit, `2` = a body it measurably disturbed |

Severity and causal moved off the union deliberately. They answer "where is the thing that is
wrong", and at inference a model only ever has the invalid video — marking the object's
lawful footprint asks it for pixels the question does not contain.

**One exception, where the invalid side has nothing at all.** A body that *moved* has pixels
in the invalid render, so its lawful footprint stays unpainted — `continuity` gets severity
only where the object actually is. A body that *vanished* has no invalid footprint anywhere,
and since `severity_t[t] == severity_map[t].max()` is a schema guarantee, an empty map is a
zero timeline: `permanence` came out unscoreable on all fourteen of its cells. Where there is
no wrong place to confuse it with, "where it should have been" is the only honest
localisation, so severity falls back to `reference_mask` on exactly those frames.

`violation_mask` keeps the union throughout, so the training target is unaffected.

`causal_mask` level 2 is **measured**: bodies whose trajectory provably departs from the valid
twin without being culprits themselves, which is the same comparison the bystander guard
runs. It is no longer a list a plan has to enumerate in advance.

### `instances` and `segmentation` — the label space

A segmentation map is unusable without the id → name table beside it. `meta.json` carries
both, so a consumer never has to reconstruct the mapping:

```json
"segmentation": {
  "encoding": "instance", "dtype": "uint16", "background_id": 0,
  "ids_are_declared": true,
  "id_to_name": {"0": "background", "1": "floor", "2": "ball_a", "4": "ball_b"}
},
"instances": [
  {"id": 2, "track_id": 2, "name": "ball_a", "category": "sphere", "role": "actor",
   "static": false, "dormant": false, "is_culprit": true,
   "mass_kg": 1.0, "friction": 0.05, "restitution": 0.75,
   "first_frame": 0, "last_frame": 24, "frames_visible": 25, "pixels_peak": 209}
]
```

`id` is the pixel value in `seg.npz`, and it is stable for the whole clip — so **`id` is also
the `track_id`** and there is no association step. `is_culprit` marks the bodies the violation
acts on, which is `violation.causal_body_ids` denormalised onto the instance for convenience.

Visibility is **measured from the rendered map**, not assumed: a dormant understudy reports
`frames_visible: 0` and `first_frame: null` rather than looking present because the scene
declared it.

### Render passes — segmentation tracks, depth, flow

All shipped for **both** twins, straight from the renderer with no re-encoding. They cost
nothing extra: the seven passes are what the measured per-frame render time already includes.

| file | array | shape | |
|---|---|---|---|
| `seg.npz` | `seg` | `[T,H,W]` uint16 | instance ids, **0 = background** |
| `depth.npz` | `depth` | `[T,H,W,1]` float32 | metres from the camera |
| `flow_fwd.npz` | `flow_fwd` | `[T,H,W,2]` float32 | pixels, `(row, col)`, frame `t → t+1` |
| `flow_bwd.npz` | `flow_bwd` | `[T,H,W,2]` float32 | pixels, `(row, col)`, frame `t → t-1` |
| `normals.npz` | `normals` | `[T,H,W,3]` uint16 | surface normals, `0..65535` maps to `-1..1` |
| `object_coords.npz` | `object_coords` | `[T,H,W,3]` uint16 | normalised object-space coordinates |

**Per-object masks for the whole clip.** Segmentation ids are the *declared*
`segmentation_id`s and they are stable across every frame, so one comparison gives a body's
track:

```python
seg  = np.load("seg.npz")["seg"]                 # [T,H,W]
bods = np.load("bodies.npz")                     # ids and names, self-describing
for bid, name in zip(bods["body_ids"], bods["body_names"]):
    track = (seg == bid)                         # [T,H,W] bool -- this body, every frame
```

That the ids are the declared ones and not the renderer's asset order is not free — it
depends on `kb.adjust_segmentation_idxs()` running after every render, and the worker asserts
the rendered ids are a subset of the declared ones. See CLAUDE.md.

**Two encodings that will bite a naive reader:**

- **Depth's background is a sentinel, not a distance.** Foreground runs a few metres;
  background comes back as ~`1.1e10`. Mask it with `seg > 0` before doing statistics, or a
  mean depth is meaningless.
- **Flow is in pixels and in `(row, col)` order**, not `(x, y)`. Verified against measured
  centroid displacement: a body whose centroid moved 2.96 columns between frames reports a
  mean flow of 3.33 columns over its mask — the residual is rotation, which flow sees per
  pixel and a centroid cannot.

### `bodies.npz` — the physical quantities as the energy computation saw them

**Derived from `traj.npz`, not additional to it.** Every clip directory already ships
`traj.npz` with positions, orientations, both velocities, radius, gravity and the contact
list. Two things make this worth a separate file:

- `traj.npz`'s `mass` is the **declared** mass, one constant per body. `bodies.npz`'s is
  `[T,B]` and **follows volume**, which is what the energy is computed against and what makes
  `immutability` a mass violation while `deformation` is not.
- Every column comes off the same code path as the energy trace — the same `inertia_diag`,
  the same floor datum, the same volume rule — so the two cannot disagree. A consumer can
  recompute `E` from these columns and get the shipped trace back to floating-point noise
  (`tests/test_energy.py` pins that).

| key | shape | |
|---|---|---|
| `body_ids` | `[B]` | matches `seg.npz` instance ids |
| `static`, `present` | `[B]`, `[T,B]` | static bodies carry `mass == 0`, PyBullet's convention |
| `mass` | `[T,B]` | kg, **follows volume** — see [energy.md](energy.md) |
| `radius` | `[B]` | m |
| `inertia` | `[T,B,3]` | body-frame principal moments |
| `position`, `quaternion` | `[T,B,3]`, `[T,B,4]` | m; `(w,x,y,z)` |
| `velocity`, `speed` | `[T,B,3]`, `[T,B]` | m/s |
| `angular_velocity`, `angular_speed` | `[T,B,3]`, `[T,B]` | rad/s |
| `momentum`, `momentum_magnitude` | `[T,B,3]`, `[T,B]` | kg·m/s |
| `angular_momentum` | `[T,B,3]` | body frame |
| `height` | `[T,B]` | m above the scene's floor datum |
| `kinetic`, `potential` | `[T,B]` | J |
| `gravity`, `dt` | `[3]`, scalar | the constants the rest is measured against |

`meta.json` carries an `energy` block: `E0`, `E_end`, `total_dissipated`,
`total_dissipated_fraction`, `peak_free_anomaly`, `peak_contact_anomaly`, `peak_excess_loss`.
Anomalies are fractions of `E0`. See [energy.md](energy.md) for the three channels and the
measured floors.

`prefix_identical_verified` (bool), `prefix_differing_pixels` (int),
`prefix_identical_upto_frame` (int).

`prefix_identical_verified` is **measured** at annotation time by comparing the two renders
pixel by pixel over `[0, t_event)`, and `prefix_differing_pixels` reports the count. It used
to be a hardcoded `true`, which is how 29 of 176 clips in one sweep shipped with renders that
differed before `t_event` while the validator cheerfully checked the constant. A provenance
field that cannot be false is not provenance.

`prefix_identical_verified` is the dataset's central integrity claim: valid and invalid
renders are **bit-identical** before `t_event`. Consumers relying on pixel-aligned twins
should check this flag rather than assuming it.

### `real2sim`

`null` in v0. Reserved for Phase 4:
`{real_twin_uid, capture_rig, sysid_params, twin_gap_lpips, twin_gap_reproj_px}`.

## Array files

### Temporal — `timelines.npz` (PLAN §3.2)

The rasterised form of the windows, so consumers never expand intervals themselves.

| array | shape | dtype | meaning |
|---|---|---|---|
| `active` | `[T]` | bool | is the violation active at frame `t` |
| `observable` | `[T]` | bool | is there visual evidence at frame `t` |
| `occluded` | `[T]` | bool | is the primary culprit hidden at frame `t` |
| `severity_t` | `[T]` | f32 | `max_b s(b,t)` — the magnitude curve over time |

### Spatiotemporal masks (PLAN §3.3)

| file | dtype / shape | notes |
|---|---|---|
| `seg.npz` | uint16 `[T,H,W]` | instance ids; `causal_body_ids` index into these |
| **`violation_mask.npz`** | bool `[T,H,W]` | **the primary annotation** — culprit pixels while the violation is active *and visible* |
| **`reference_mask.npz`** | bool `[T,H,W]` | where the culprit *should* be: its footprint in the valid twin, ungated in time. Shipped on **both** clips of a pair |
| `causal_mask.npz` | uint8 `[T,H,W]` | `0` = none, `1` = primary culprit, `k ≥ 2` = consequence `k−1` |
| `divergence_map.npz` | f16 `[T,H,W]` | **NOT the violation region** — see below |

> **The mask union rule.** `violation_mask[t]` is the union of the culprit's footprint in the
> **invalid** render *and* in the **valid** twin. This is the only rule that gives non-empty
> masks for `permanence`-vanish (the body has no invalid-side pixels precisely because it
> vanished) and correct two-lobed masks for `continuity`-teleport. It is well-defined only
> because the twins are pixel-aligned and share instance indexing.

> **The visibility gate.** The mask is gated on `active AND observable`, not on `active`
> alone, because it answers *where can this be seen*. The two differ on real frames: a
> super-elastic bounce is unlawful the instant the contact resolves, but the body is still in
> exactly the same place in both twins, so that frame contains no evidence. Marking pixels
> there would ask a model to localise something the image does not contain. `timelines.active`
> remains the ground-truth timeline, and the gap between the two *is* the observability lag.

> **`reference_mask` is the counterfactual, not a violation.** It says where the lawful
> trajectory would have put the culprit, with no claim that anything is wrong — which is why
> it ships on the valid clip too, and why it is ungated in time. For a vanished body it is the
> only mask with any pixels at all, and it is what makes the union rule legible rather than
> mysterious. Visualisers draw it as an outline *on top of* the violation mask; drawing it
> underneath lets the filled mask hide it in exactly the cases that matter.

> **`divergence_map` warning.** This is `|valid − invalid|` in pixel space. After `t_event` a
> twin pair diverges *everywhere* downstream — shadows, contact chains, secondary collisions
> — so the pixels that differ are **not** the pixels where the violation is. A
> different-but-valid rollout diverges too. Shipped for analysis; **never a training target.**

### Severity (PLAN §3.4)

| file | dtype / shape | notes |
|---|---|---|
| `severity_map.npz` | f16 `[T,H,W]` | bounded score `s` painted into every dynamic culprit, under the same visibility gate as the mask |
| `residuals.npz` | f32 `[T, n_bodies, n_laws]` × 3 | `r` (physical units), `z` (vs noise floor), `s` (bounded `[0,1]`) |

Three representations, all shipped: `r` for interpretability, `z` for calibration against the
per-`(scenario, law)` noise floor, `s = clip(z / z_ref(family), 0, 1)` for training, since
metres of penetration and energy ratios are not otherwise comparable.

`violation_mask` and `severity_map` are derived **differently on purpose**: the mask is the
injector's exact ground truth (binary), the severity field is measured from residuals
(continuous, with a noise floor). Where they disagree — a sustained violation whose residual
dips mid-window — that is real physics, not a bug.

**Consistency guarantee:** `severity_t[t] == severity_map[t].max()`.

### Geometry (PLAN §3.5)

`depth`, `flow_fwd`, `flow_bwd`, `normals`, `object_coords`, all `[T,H,W,·]`, straight from
Kubric's exporters.

### `grids.npz` (PLAN §3.6)

Masks and severity pre-reduced to the latent token grid. **The grid timeline is always
`4k + 1` frames** so it aligns exactly with a video-DiT VAE's 4× temporal binning.

| tier | source timeline | latent grid |
|---|---|---|
| A | 25 frames @ 256² | `7 × 16 × 16` |
| B | 97 frames @ 512² | `25 × 16 × 16` |

- `mask_<F>x16x16` — bool, reduced by **max** (a violation in any contributing source frame
  marks the latent frame)
- `severity_max_<F>x16x16`, `severity_mean_<F>x16x16` — f16; peak and average are different
  questions
- `mask_Tx32x32`, `severity_{max,mean}_Tx32x32` — for consumers on a different tokenizer

**Ordering guarantee:** time-major, `[F_lat, H_lat, W_lat]` with the latent frame slowest.
A schema guarantee, not an implementation detail — flattening to `[F_lat·H_lat·W_lat]` must
match a transformer's token order.

### `traj.npz` (PLAN §3.7)

The seam file: per-body `pos[T,3]`, `quat[T,4]`, `lin_vel`, `ang_vel`, `applied_force`;
`present[T,B]` bool (False = the body is not in the scene at all, which is how `permanence`
is expressed -- the body is *gone*, not moved somewhere odd);
`contacts[T]` as `(bodyA, bodyB, point, normal, impulse, penetration)`; and the event block
(`violation_windows`, `causal_body_ids`, intervention params). Shipped so renders reproduce
without re-simulating.

## Cross-checks enforced by `physloc validate`

1. jsonschema conformance against `meta.schema.json`
2. `t_event_frame ≤ t_observable_frame` and `t_event_frame ≤ t_end_frame`
3. `violation_windows` sorted, non-overlapping, within `[0, num_frames)`; `t_event_frame` and
   `t_end_frame` equal their min and max
4. `timelines.active` is exactly the rasterisation of `violation_windows`; likewise
   `observable` and `observable_windows`
5. `violation_mask` is non-empty on every frame that is **`active` AND `observable`** — the
   check that catches a regression in the mask union rule. Deliberately *not* every active
   frame: while the culprit is fully occluded the violation is active but has no visible
   extent at all, so an empty mask is the truthful annotation. Conversely the mask must be
   empty on active-but-unobservable frames
6. `severity_t[t] == severity_map[t].max()` for all `t`
7. every id in `causal_body_ids` appears in `seg.npz`; `newton3_reaction` has ≥ 2
8. `spatial_extent == "global"` iff `family == "global_gravity"`
9. `domain` matches `family`, and `(scenario, family)` is a `●` cell in `taxonomy.py`
9b. `physics_medium == "granular"` iff `scenario == "pour"`; `physics_medium` is
    never `"fluid"` at schema version 0
10. every entry in `assets` carries a non-empty `license`
11. `provenance.prefix_identical_verified` is true
12. `violation` is null iff `label == "valid"`
13. `twin_uid` resolves to an existing clip whose `pair_uid` matches
