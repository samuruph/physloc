# PhysLoc — orientation

A spatio-temporally annotated physics-violation video dataset: every invalid clip ships
*where*, *when* and *how badly*, derived from the simulator rather than annotated by hand.

**Read [README.md](README.md) first** — it is the operational reference and the one kept
current. [docs/PLAN.md](docs/PLAN.md) is the design document: its *reasoning* is sound, its
*numbers* have drifted, so trust the code over any count you read there.
[docs/schema.md](docs/schema.md) holds the `meta.json` reference. The IntPhys 2 and
LikePhys mapping is DATA, on each family in `physloc/taxonomy.py` — the prose copy of it was
deleted because it drifted from the data it described.

**Counts live in `physloc/taxonomy.py` and nowhere else.** `python -m physloc.cli taxonomy`
prints them. Prose copies of these tables have drifted five different ways; do not add a
sixth.

**Status: the whole matrix builds.** 13 scenarios x 23 families compose into **166 cells**;
`physloc generate` walks them, `validate` checks them, `export` packages them for
publication. Nothing has been published yet.

## Locked decisions — don't relitigate without saying so

- **Kubric** (Blender + PyBullet in docker) for v0. MuJoCo replaces PyBullet later, behind
  the trajectory seam.
- **Two tiers, because there are only two geometries.** `debug`: 128×128, 12 fps, 25
  frames, 16 spp — **~8 s per clip**, never published; this is what you iterate on.
  `release`: **512×512, 30 fps, 89 frames = 2.97 s**, ~637 s/clip. 30 fps so a release
  downsamples cleanly to 15 and 10; 89 rather than 90 because every frame count must be
  `4k+1` for VAE latent alignment.

  **`v0` and `v1` are NOT tiers.** They used to be, and differed in nothing but their name —
  what actually separated them was complexity, which is its own axis and its own ladder. A
  tier that encodes a release number has to be renamed every release. So: the tier says how
  big and how long, the **complexity ladder (L0–L3)** says how hard, and `v0`/`v1` are what
  a published dataset is CALLED — set by `--outdir`, recorded as `release` in every
  `meta.json`. **Debug at the debug tier; a bug found there is fixed for both.**
- **The complexity ladder is SCENE REALISM, and nothing else.** Four levels -- L0 baseline,
  L1 materials, L2 HDRI, L3 GSO -- each with a declared `share` of a full generation
  (1.00/0.50/0.30/0.20). **Difficulty is FIVE NAMED CONDITIONS, one per clip**, applied
  inside every level and never levels of their own -- `CONDITION_CYCLE`, a ten-variant cycle:
  `standard` 60%, then `camera`, `distractors`, `multi` and `camera+multi` at 10% each
  (marginals: camera 20%, distractors 10%, multi 20%). One condition per clip rather than
  independent per-axis coin flips, so every count is exact and every comparison against
  `standard` isolates one change.

  **`distractors` and `multi` differ in how many objects get INVALID PHYSICS, not in how
  many objects there are.** Both add `EXTRA_OBJECTS` = 3-10 extra bodies, some moving and
  some inert. `distractors` leaves exactly one culprit and its extras are `role="distractor"`
  scenery no family can target; `multi` makes them `role="actor"` peers and 2..N-1 violate.
  Only the second asks "which of these is wrong". They are never combined.

  Conditions used to be levels, firing on 1 variant in 5 *within* a level, which meant
  "L2 minus L1" measured materials plus whichever variants happened to draw a camera move --
  a level whose axis fires on only some of its clips is not a stratum. A level that does not
  buy a whole variant is SKIPPED, so a short run is all baseline; naming a level explicitly
  is the override, and `v0_L0`..`v0_L3` partition `v0_release` for exactly that. `--complexity all` walks the ladder in one run, and **every level draws its OWN scenes**
  from its own seed block (`cli.LEVEL_SEED_STRIDE`) — an L1 clip is not an L0 clip in better
  materials. Reusing one seed block across levels was tried and rejected: it pairs clips
  neatly and buys an ablation, at the cost of the breadth the dataset exists for. The level is
  part of a clip's identity (`clips/<release>/<level>/<scenario>/<seed>/`) so a level is a
  directory you can hold up on its own. All of it lives in `COMPLEXITY` in
  `physloc/scenarios/base.py`.

  **"Twin" means the valid/invalid pair and nothing else** — one scene, bit-identical prefix
  up to `t_event`. Levels are not twins of each other and must not be described as such.
- **The README's tables are GENERATED** by `physloc/reference.py`, from `taxonomy.py` and
  `scenarios/base.py`, and the HuggingFace card calls the same functions. Run
  `python -m physloc.reference --write` after changing any count; `tests/test_reference.py`
  fails if the README is stale. This generator has silently done nothing twice, both times
  a splice regex that matched neither the empty nor the filled form -- the test compares
  each block's CONTENT against `render(name)` for that reason, not `splice(text) == text`.
- **Scenarios and injectors compose; never write per-combination code.** An injector edits
  `traj.npz`, which is scenario-agnostic, so 13 scenarios x 23 families needs 13 + 8 files,
  not 299. `taxonomy.COMPATIBILITY` selects the 166 meaningful cells.
- **No fluid in v0.** Tested: Blender 2.93.4 has Mantaflow but headless baking fails
  (`NameError: liquid_save_data_N` → `Manta::Error`), Kubric exposes no fluid objects, and a
  liquid does not fit the pose-based seam. `pour` (80 grains at the debug tier, 176 above)
  is the v0
  stand-in and is labelled `physics_medium: "granular"` — never call it fluid. True fluid and
  cloth are Phase 3.

  **The count and the box are ONE decision, and the quantity that matters is how many
  grains deep the medium settles.** Forty grains in a 1.64 m box covered under a third of
  the floor and settled exactly one grain deep — measured, every grain ended at z = 0.073 —
  and a medium with no interior is one `newton2_mass` cannot stratify and one `friction`
  cannot shape. Eighty grains in a 0.68 m box settle about four deep. Grains also carry
  `rolling_friction`, without which they are frictionless rollers with no angle of repose
  and no pile forms however narrowly they are poured; Kubric's constructors have no argument
  for it, so `render.worker` applies it through `changeDynamics`.
- **CPU rendering, upstream image, unchanged.** GPU/OptiX caps at ~1.37× (only 26% of frame
  time is sampling); clip-level parallelism measures **2.50× at four workers** and flattens
  after that — eight buys 7% more. Do not build a GPU image without a new measurement.
- **Two different magnitudes, never conflated.** `intervention.magnitude` is the knob we
  turned (one scalar, exact, known *before* simulating, drives weak/medium/strong splits). The
  residual `r`/`z`/`s` is the measured effect (per body per frame, known *after*, becomes
  `severity_map`). There is no "severity mask": `violation_mask` is binary *where*,
  `severity_map` is continuous *how badly*. PLAN §3.4 works a full numeric example.
- **Taxonomy is five levels**: medium (5) → domain (8) → family (23) → scenario (15
  declared, 13 built) → instance. Encoded as data in `physloc/taxonomy.py`, including the
  scenario × family compatibility matrix. `clutter_toss` and `tumble` are `UNBUILT`.
- **A scenario's driven bodies are re-derived on every INVALID clip, not just the
  valid one.** `shadow_track`'s cast shadow is a projection of its actor, so
  `Scenario.rescript` re-casts it from the trajectory the intervention actually
  produced -- position, footprint, presence and opacity -- for every family whose
  culprit is not the shadow itself. Without it, every family on that scenario shipped a
  *detached shadow*, which is the `shadow` family, inside a clip claiming and
  annotating something else. The three optical families own the shadow and are left
  alone. **A shadow also carries no energy**: it is a picture of an absence, not matter,
  and `role == "shadow"` is skipped by `residuals.energy`.
- **`reference_mask` ships on both twins** -- the culprit's lawful footprint, taken from the
  valid render and ungated in time. It is the counterfactual "where it should be", and for a
  vanished body it is the only mask with any pixels. Visualisers draw it as a green outline
  *on top of* the red violation mask (drawing it under lets the fill hide it).
- **Debug generation defaults to one `strong` variant per cell**, with `--severity all` for
  the ladder and `--window N` for a uniform duration. Breadth of coverage is what you check
  first; three strengths of one cell is what you check second.
- **Violations are windows, not onsets.** `violation_windows` is a *list* of intervals —
  `superelastic` fires once per bounce, and observability can be interrupted by re-occlusion.
- **`causal_mask` lasts as long as the consequences do, and that is MEASURED.** Level 1 is
  red (the culprit the plan names), level 2 is blue (a body it disturbed) — both drawn
  invalid-side only, both labelled in the overlay's causal panel. The gate is the declared
  consequence window *unioned with* the frames where the invalid trajectory provably
  departs from the valid one, so a two-frame `newton2_mass` exchange that leaves both balls
  travelling wrong keeps its mask for the rest of the clip without any family having to
  declare it. Rule of thumb: if behaviour is still different because of the violation, the
  causal mask is still on.
- **Decouple dynamics from rendering.** Simulate → write `traj.npz` → render by replaying
  keyframes. This is the seam; it is what makes valid/invalid twins bit-identical before
  `t_event`.

## Non-negotiables

1. **Prefix identity.** Valid and invalid renders must be bit-identical for every frame
   before `t_event`. If that assert fails, every downstream annotation is suspect. Anything
   that perturbs the render path — resolution, `samples_per_pixel`, denoising, motion blur,
   seeds — must be identical across a twin pair.
2. **`divergence_map` is not the violation region.** It is `|valid − invalid|` in pixel
   space and it diverges everywhere downstream of the event. Ship it, label it, never train
   on it. The training targets are `violation_mask` and `severity_map`.
3. **`violation_mask` is the union over BOTH twins**:
   `violation_mask[t] = footprint(culprit, invalid, t) ∪ footprint(culprit, valid, t)`.
   Without this, vanish and teleport violations produce empty or half-empty masks — the body
   has no pixels in the invalid render precisely because it vanished. Guarded by
   `test_mask_union.py`.
   **`severity_map` and `causal_mask` are the invalid side only**, and `mask_invalid` ships
   that footprint on its own. They answer "where is the thing that is wrong", and at
   inference a model only has the invalid video. The cost is accepted and documented:
   `permanence` and `dissolve` get an all-zero severity map, with `reference_mask` carrying
   where the body should have been.
4. **Injectors never touch the sim rng.** Otherwise the twin diverges from frame 0 and (1)
   breaks. A family may take either of two paths after `t_event`:
   - **staged** (`Injector.simulates(plan)`): the world is reset to the valid state at
     `t_event`, the intervention is applied as something PyBullet honours — a velocity, a
     mass, a disabled collision pair, a resized collision shape — and the simulator runs
     forward. Real contacts. `stage()` must have a matching `unstage()`, because one scene
     serves every family in a run and a `changeDynamics` persists exactly the way a stray
     keyframe does. **This is the default and the goal**; a family stays off it only when
     nothing in the simulator corresponds to what it changes.
   - **edited**: the finished trajectory is rewritten and re-integrated by `_rewrite_from`.
     Approximate contacts. **Six families remain here**, and they are the ones whose subject
     the simulator does not own: `colour_shift` (a material property), `shadow`,
     `shadow_inverted` and `shadow_shape` (a scripted stand-in body, since Blender's shadow
     is not an object PyBullet knows about), `time_slip` (a reparameterisation of time
     itself) and `fusion` (its draw-in is prescribed motion). `solidity` decides **per
     plan**: a two-body pass-through stages as a disabled collision pair, while its granular
     `sink_group` mode edits, because removing the floor under forty grains is a scene edit
     rather than one pair. The other sixteen stage.

     Do not take this list on trust — it has been wrong before. It is derivable:
     `simulated` on the class, `simulates()` when a family decides per plan.
     Every staged family keeps its `_apply`
     as a **host-side approximation**, because that is what the mock rollout in `tests/`
     exercises without a docker round trip — so `_apply` and `stage()` must describe the
     same intervention, and where they can share a profile function they do.
   - **Resizing IS staged.** PyBullet has no in-place collision rescale, which is why
     `immutability` and `deformation` used to be edited — and it showed: a shrunken cube
     stopped touching the floor and a swollen ball grew into the barrier beside it.
     `stepper.ShapeSwap` parks the declared body and stands a correctly-sized proxy in its
     place, rebuilt a few times per frame through the ramp. A squashed sphere is an
     ellipsoid, which PyBullet also has no primitive for, so the proxy is a convex hull of
     a per-axis-scaled unit sphere — measured in the pinned image, a 0.3 × 0.3 × 0.6 hull
     comes to rest at z = 0.601. Consequence, accepted: a shape change the physics honours
     cannot leave the path lawful, because an ellipsoid tumbles where a ball rolled.

   Either way frames before `t_event` are the valid rollout verbatim, so (1) holds by
   construction.
5. **Look at the clips before scaling.** Overlay video with mask, severity and all three
   clocks burned in, every phase, before generating more.
6. **Generated output never enters git.** `data/ renders/ out/ clips/ *.npz *.mp4` are
   gitignored; keep it that way.
7. **Every asset carries a license string.** Enforced by `physloc validate`. Record it when
   you enable an asset source, not at release.

## Two environments, never mixed

- **container** (pinned image): Kubric 2022.4.1, Blender 2.93.4, PyBullet, Python 3.9 —
  scene sampling, simulation, rendering. Writes `traj.npz` + raw passes.
- **host** (`conda activate physloc`, Python 3.11): numpy/scipy/opencv/jsonschema —
  residuals, annotation, masks, severity, grids, validation, viz.

Never install Kubric, Blender or PyBullet on the host. The trajectory seam is the boundary.

## Two traps that fail silently

1. **Always run `kb.adjust_segmentation_idxs()` after `renderer.render()`.** Kubric numbers
   the raw segmentation by scene-asset order; declared `segmentation_id`s are honoured only
   by that post-process. Skipping it relabels instances whenever declaration order differs
   from insertion order and quietly corrupts every mask and residual. The worker asserts the
   rendered ids are a subset of the declared ones.
2. **"Occluded" means *fully* occluded.** A few visible actor pixels make a violation
   instantly observable and collapse the observability lag to zero. `occluder_pass` shrinks
   the occluder extents by the actor's projected silhouette radius and fires mid-run.

## Visualisation

`overlay.mp4` only -- **no image files anywhere**. The container has no ffmpeg, so it writes
arrays and every mp4 comes from `physloc/viz/video.py`. Nine panels in one order everywhere -- RGB,
energy, segmentation, depth, optical flow, mask, severity, causal, divergence: evidence
first, then the annotation derived from it. A red dot while the violation is active, and a
timeline with both window families and the three clocks. `grid`, `sheet` and `coverage` use
the same nine and the same order.

When drawing text on frames, use `viz.overlay._text`: it draws a dark backing box rather
than a thick outline, because OpenCV's Hershey glyph advance grows with stroke thickness, so
an outline pass drawn at `thick+2` is wider than the fill and leaves dark ghost glyphs.

## Working with Kubric

Kubric is **not** a Python dependency — it lives in the docker image. Pattern is *your
script, their container*:

```bash
bash docker/kubric.sh physloc/render/worker_smoke.py --frames 4
```

`refs/kubric` is a **pinned, gitignored, read-only reference checkout** (`bash
scripts/fetch_refs.sh`). Read `refs/kubric/challenges/movi/movi_def_worker.py` — it is the
template this project adapts (PLAN Part 0.5). The clone is ~4 years newer than the image;
**the installed copy inside the container is the API authority.**

## Measured, not assumed

Blender 2.93.4 / Python 3.9.5 / kubric 2022.4.1 in the image. **1.75 s per 256² frame,
7.16 s per 512²**, linear in frames, all seven passes. Frame time fits
`T = 1.29 + 0.0074·spp` at 256² → only ~26% is sampling, so OptiX caps at ~1.37×.
`sim_seconds` is 0.0 — physics is free at v0 scale.

Clip-level parallelism, measured on this box (8 cores) over 8 jobs of 14 cells each:
**1826 s at one worker, 729 s at four, 685 s at eight** — so 2.50× at four, and doubling to
eight buys 7%. Blender already uses every core per render, so workers oversubscribe and the
curve flattens hard. `physloc/cli.py` holds these as live constants (`SPEEDUP`,
`SECONDS_PER_CLIP`) and prices a run from them: `physloc taxonomy --config <name>`.

An older 1.92× figure appears in `docs/`; it was four clips at 256² and is superseded.
