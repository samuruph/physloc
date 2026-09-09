# PhysLoc

**A physics-violation video dataset where every invalid clip ships *where* the violation is,
*when* it happens, and *how badly* — derived from the simulator, not annotated by hand.**

Most intuitive-physics benchmarks ask one bit per clip: *is this possible?* PhysLoc asks the
harder question a model should be able to answer if it understands the scene at all — **which
object is wrong, in which pixels, during which frames, and by how much.** Because the labels
come out of the simulator that produced the violation, they are exact and free.

---

# Part I — Using the dataset

## 1. Clips come in twins

Every invalid clip has a **valid twin**: the same scene, the same seed, the same objects, and
a **bit-identical prefix** up to the moment the violation is introduced (`t_event`). Only
after that do they differ.

That is the whole design. It means the difference between the two clips is attributable to
the intervention and nothing else — no lighting change, no re-rolled object, no camera drift.
It is verified per clip and recorded as `provenance.prefix_identical_verified`.

```
valid    ─────────────────────────●──────────────────────
                                  │ t_event      lawful
invalid  ─────────────────────────●╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌
         identical to the frame            the violation
         before t_event                    and its consequences
```

One valid twin is shared by every family and severity bin staged on that scene, because the
prefix is identical — rendering it per family would be waste, and any drift would be a bug.

## 2. What one clip gives you

| file | what it is |
|---|---|
| `rgb.mp4` | the video |
| `violation_mask.npz` | **bool [T,H,W] — the primary annotation.** Where the violation can be seen |
| `severity_map.npz` | **f16 [T,H,W]** — how badly, per pixel per frame, bounded [0,1] |
| `causal_mask.npz` | uint8 [T,H,W] — 1 = the culprit, 2+ = bodies it disturbed |
| `reference_mask.npz` | bool [T,H,W] — where the culprit *should* have been (from the valid twin) |
| `timelines.npz` | per-frame flags: `active`, `observable`, `occluded`, `severity_t` |
| `meta.json` | labels, taxonomy, windows, magnitudes, provenance |
| `seg.npz` | uint16 [T,H,W] — instance ids, stable across frames, so they *are* object tracks |
| `depth` · `flow_fwd/bwd` · `normals` · `object_coords` | the usual geometry passes |
| `energy` · `bodies` · `residuals` | mechanical energy, per-body state, and the raw residuals |
| `overlay.mp4` | everything above burned into one annotated video, for looking at |

Every object's mask for the whole clip is one comparison:

```python
seg  = np.load("seg.npz")["seg"]                 # [T,H,W]
bods = np.load("bodies.npz")
for bid, name in zip(bods["body_ids"], bods["body_names"]):
    track = (seg == bid)                         # [T,H,W] bool
```

**Three traps, worth knowing before you train on any of it:**

- **`divergence_map` is not the violation region.** It is `|valid − invalid|` in pixel space
  and diverges *everywhere* downstream of the event. A model trained on it learns to find the
  edit, not the physics. Train on `violation_mask` and `severity_map`.
- **`violation_mask` is gated on visibility.** It answers *where can this be seen*, so it is
  empty on frames where the violation is active but the culprit is hidden.
  `timelines.active` is the unhedged truth about *when*. The gap between them is the
  observability lag, and it is deliberate.
- **`permanence` and `dissolve` have an all-zero severity map** — the body is gone, so it has
  no pixels to score. `reference_mask` carries where it should have been.

## 3. The taxonomy

Five levels: **medium → domain → family → scenario → instance.** A *cell* is one
(scenario, family) pair; there are **166** of them.

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
| `pour` | granular | 17 | a loose column of grains falls into an open box (40 at the debug tier, 96 above) |
| `pyramid_impact` | rigid | 13 | cube dropped onto a sphere pyramid |
| `ramp_slide` | rigid | 11 | block slides down an incline |
| `resting_table` | rigid | 11 | several bodies at rest on a surface |
| `rolling_ramp` | rigid | 16 | cube tumbles down a raised ramp and off its lip |
| `shadow_track` | optical | 10 | object translates under a fixed light |
| `stack_topple` | rigid | 13 | stacked bodies, marginally stable |
| `toss` | rigid | 11 | body thrown on a ballistic arc |
<!-- /physloc:scenarios -->

`clutter_toss` and `tumble` are declared but not built.

## 4. Severity — two numbers that are never the same thing

Each cell is staged at three strengths: **`weak`, `medium`, `strong`.**

| field | what it is | when it is known |
|---|---|---|
| `magnitude` | **the knob we turned** — one scalar, exact, in the family's own units | *before* simulating |
| `peak_severity` | **the measured effect** — the residual, z-scored against a noise floor and bounded to [0,1] | *after* |

They answer different questions and must not be conflated. `magnitude` says how hard the
intervention pushed; `peak_severity` says how much came out, which depends on the scene. A
strong push into a wall can produce less measured effect than a medium push into open space.

`severity_map` is the spatial version of the second: `violation_mask` is binary *where*,
`severity_map` is continuous *how badly*.

**Every ladder is monotone in distance-from-lawful** — checked by
`tests/test_severity_ladders.py`, which refuses any family whose `medium` is milder than its
`weak`.

## 5. The complexity ladder — how hard the scene is to look at

Severity asks *how badly is the law broken*. Complexity asks *how hard is the scene to
parse*. They are independent axes, and reporting across both is what separates "understands
physics" from "copes with clutter".

**Four levels of scene realism**, each the one below it plus exactly one thing:

<!-- physloc:ladder -->
| level | adds | background | objects | materials | share | built |
|---|---|---|---|---|---|---|
| **L0** | *the baseline* | solid | primitive | one shared density | 50% | yes |
| **L1** | **materials** — wood, steel, rubber | solid | primitive | yes | 25% | yes |
| **L2** | **an HDRI environment** | hdri | primitive | yes | 15% | yes |
| **L3** | **GSO objects** — real 3D scans | hdri | gso | yes | 10% | yes |
<!-- /physloc:ladder -->

Every clip carries its level in `complexity`, so a level is a filter rather than a separate
download. **Each level draws its own scenes** — an L1 clip is not an L0 clip in better
materials, it is a different event — so the ladder buys breadth as well as difficulty.

Below L1 every object shares **one density**, so mass varies only with size, which a viewer
can see. From L1 up mass is `density × volume` and a heavy-looking object is heavy.

## 6. Difficulty conditions — what else is in the scene

Camera motion, distractors and multiple culprits are three ways to make a clip harder. Every
clip carries **exactly one** condition:

<!-- physloc:conditions -->
| condition | share | camera | extra objects | objects with invalid physics |
|---|---|---|---|---|
| `standard` | 60% | static | — | **1** |
| `camera` | 10% | **moves** | — | **1** |
| `distractors` | 10% | static | **3–10** | **1** |
| `multi` | 10% | static | **3–10** | **2 … N−1** |
| `camera+multi` | 10% | **moves** | **3–10** | **2 … N−1** |
<!-- /physloc:conditions -->

Marginals: the camera moves on **20%** of clips, **10%** carry distractors, **20%** have
multiple culprits. Fields: `condition`, `camera_motion`, `n_distractors`, `n_actors`,
`n_culprits`.

**One condition per clip, not independent coin flips.** Independent per-axis ratios blur what
a benchmark reports — a "moving camera" clip that also carries clutter mixes two effects —
so every count is exact and every comparison against `standard` isolates one change.

**`distractors` and `multi` differ in one thing: how many objects have invalid physics.**
Both put **N ∈ [3,10]** extra objects in the scene, drawn per clip, and in both some of those
objects move and some are still. What separates them:

| | `distractors` | `multi` |
|---|---|---|
| extra objects | 3–10 | 3–10 |
| **objects violating** | **exactly 1** | **2 … N−1** |
| what the extras are | scenery no family can target (`role="distractor"`) | eligible culprits (`role="actor"`) |
| the question it asks | *is anything wrong?* | ***which** of these is wrong?* |

That is the distinction worth drawing, because only the second is a localisation problem.
With one culprit, *which object is wrong* has a trivial answer — there is one candidate — so
a model can score by detecting that *something* is off and pointing at it. With 2 of 7
violating among 5 that are fine, the spatial annotation has to be earned.

The counts are drawn rather than fixed for the same reason in both cases: a scene that is
always six objects with two wrong teaches the layout, not the physics.

**They are never combined.** A scene with both would ask you to separate inert clutter from
lawful peers from culprits — three distinctions where the label makes one.

**Camera motion** is one of three kinds, never a pan: `track` (40%) slides across with the
aim held, `orbit` (40%) swings around the subject at fixed radius, `dolly` (20%) approaches
or retreats. A panning camera would make *"did the object move or did the camera?"*
unanswerable from the clip, and every violation here is a claim about object motion. Under a
moving camera, `flow` and `depth` stop being pure object motion — the per-frame extrinsics
ship in `meta.json` so you can undo it.

## 7. Detection difficulty — easy, moderate, hard

Section 6 is the **knob**: we asked for a moving camera, or for clutter, and the sampler
delivered it. This is the **measurement**: what actually came out. A clip built with eight
distractors whose culprit still fills a quarter of the frame is not hard, and a `standard`
clip whose two-frame violation happens behind a screen is.

> **`difficulty` is to `condition` what `peak_severity` is to `magnitude`.** The same
> refusal to conflate the knob with the measurement, one axis over.

Every **invalid** clip carries one label. A valid twin has no violation to detect, so it has
no difficulty and belongs to no evaluation set.

```json
"difficulty": {
  "level": "hard", "rank": 2,
  "binding_factors": ["footprint"],
  "factors": {"footprint": {"value": 0.0041, "level": "hard"},
              "severity":  {"value": 0.98,   "level": "easy"}, "...": {}}
}
```

### Seven factors, and the clip takes its worst

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

A clip is `easy` only when it is easy on **every** axis; one small footprint makes it hard
however clean the rest of it is. That is KITTI's Easy/Moderate/Hard construction, and it is
chosen over a weighted score for three reasons:

- **It says why.** `binding_factors` names the axes that set the label. A model that fails on
  occlusion-bound clips and passes on footprint-bound ones has told you something; a single
  number has not.
- **The sets nest.** easy ⊂ moderate ⊂ hard, so *"at moderate"* means every clip of
  `rank <= 1` and the three numbers are comparable to each other. A weighted score gives three
  disjoint buckets whose members share nothing.
- **A sum hides trade-offs.** Averaging a tiny footprint against a static camera claims the
  two cancel. They do not.

```python
df[df.difficulty_rank <= 1]                       # the "moderate" evaluation set
df[df.difficulty == "hard"].binding_factors        # and what made them hard
```

### What is deliberately *not* a factor

**The complexity level.** L3 is harder to parse than L0 — and it is already its own axis,
with its own share of the release and its own directory. Folding it in would correlate the two
and destroy the ablation both exist for. Report **`difficulty × complexity` as a grid**; that
grid is the interesting result, and it only exists if the two are measured apart.
`physloc stats` plots it. **The family and the scenario** are excluded for the same reason,
and **`magnitude`** because it is the knob — `severity` is its measured counterpart and is the
one that belongs here.

### Where the thresholds come from, and how to change them

Four are **fitted**, at the tertiles of 431 invalid clips from the review corpus. Three are
**chosen**, because the corpus could not answer: every review config holds one window setting,
so `duration`'s tertiles would encode the config rather than the difficulty, and 60% of clips
are `standard`, so `clutter` and `culprits` mostly report 1. Which is which is recorded per
factor in `physloc/annotate/difficulty.py` — a fitted threshold describes *this* dataset, a
chosen one makes a claim about detection, and the two age differently.

They live in **`configs/common.yaml`**, written `[easy, moderate]` and always in the
easier-is-better direction:

```yaml
difficulty:
  footprint: [0.05, 0.012]     # easy at or ABOVE 0.05 of the frame
  occlusion: [0.05, 0.5]       # easy at or BELOW 0.05 of the window
  clutter:   [2, 6]
```

Which way a factor runs belongs to the factor, not the config. To refit against your own
corpus:

```bash
python scripts/fit_difficulty.py out/review_conditions out/review_severity
```

It prints each factor's tertiles and what the current cuts do to that corpus, so a change
starts from data rather than from taste.

> **Freeze them once you publish.** A benchmark whose difficulty labels move between releases
> cannot be compared with itself. They are editable because a dataset with different geometry
> or a different window policy will want different cuts — and because the resolved values ride
> in every `meta.json`, so a clip always says what it was labelled under. Changing them is a
> new release, not a bug fix.

**A granular medium counts once.** `pour` has 80 grains and nobody is asked which grain is
wrong, so `clutter` and `culprits` see one thing rather than eighty. A family that acts on a
genuine *subset* of the medium keeps its count, because then the question really is "which".

## 8. What else varies

Every free parameter is drawn per clip from the seed: object shape, size, colour, mass,
starting position and velocity, floor and backdrop colour, camera pose, and **the frame the
violation fires on**.

**Materials** (from L1) give appearance and density that agree, so a heavy-looking object is
heavy and the resulting motion is legible rather than arbitrary.

They apply to the **staging** as well as the actors — a ramp is wooden, a pendulum post is
steel — because half of what is on screen is staging, and leaving it as untextured blocks
meant the level changed only a fraction of the frame. Four materials are metallic and two
transmissive, which matters most from L2 up: a metal or glass body *reflects and refracts the
environment*, and that is what makes an object look like it belongs in the scene rather than
composited onto it.

**The draw is weighted, not uniform**, and the `share` column is why. The palette is not
uniform in density — the metals that make a level legible are also 7800–8900 kg/m³ — so
drawing evenly would put the mean density at 3458 against the 2313 the scenarios' contact
parameters were tuned against, and double the median. That is a change to the *physics*
smuggled in by a change to the *appearance*. Weighting the light end up puts it back at 2256.

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

**Environments** (from L2): **509** HDRI Haven captures — every one the manifest has, since
an environment map has no geometry to get wrong. **Objects** (L3): **140** Google Scanned
Objects across 14 categories, all CC BY-SA 4.0, curated squat and roughly isotropic so their
lawful motion is predictable — an object that topples unexpectedly reads as the violation.
13 of them are in Kubric's own held-out split.

The floor is the one exception to materials: it keeps the colour
`_recolour_scenery` chose for it, because that colour is guarded for contrast against every
actor in the scene, and takes only the surface finish.

Two deliberate limits. Glass and ice are **frosted** rather than clear, because a transparent
culprit is hard to point at and pointing at it is the task; and there are two transmissive
materials at different densities (920 and 2500) so that *transparent* does not become a cue
for *heavy*. Scenery draws from a **narrower set** — no glass ramp, no mirror floor.

The floor is the one body that keeps its own colour: `_recolour_scenery` chooses it with a
contrast guard against every actor in the scene, so the floor takes only the surface finish
from its material.

`python -m physloc.cli randomisation` reports distinct values per axis, so *"is it actually
varied"* is a number rather than an impression.

## 9. Splits

`main` **75%** · `held_out` **20%** · `debug` **5%**.

**Grouped by `pair_uid`, never by clip.** A valid twin and its invalid siblings share every
frame before `t_event`, so splitting them apart would put the answer on the other side.
Pairs are ordered by a hash of their uid and cut at the quantiles *within each scenario*, so
every split sees every scenario in the same proportions. There is no rng — reproducing the
release reproduces its splits.

**Every split ships every annotation.** The split is a label, not a filter: you can re-cut it,
and you can score any clip. If you need a genuinely blind held-out set for a leaderboard,
strip the annotations at that point — nothing has to be regenerated.

## 10. The index

`index.parquet` is one row per clip, with the video embedded so it plays inline in the
HuggingFace viewer. A breakdown is a groupby, not a crawl over thousands of `meta.json`:

```python
import pandas as pd
df = pd.read_parquet("index.parquet")

df.groupby(["domain", "severity_bin"]).peak_severity.mean()
df.groupby("condition").size()                       # standard / camera / multi / ...
df[df.complexity == "L3"].groupby("family").size()   # what survives the hardest level
df[df.n_culprits > 1]                                # the multi-object clips

df[df.difficulty_rank <= 1]                          # the "moderate" evaluation set
df.groupby(["complexity", "difficulty"]).size()      # the grid worth reporting
df[df.difficulty == "hard"].binding_factors.str.split(",").explode().value_counts()
```

Columns: identity (`clip_uid`, `pair_uid`, `twin_uid`, `label`, `split`), taxonomy
(`scenario`, `family`, `domain`, `medium`), violation (`severity_bin`, `magnitude`,
`peak_severity`, `t_event_frame`, `violation_windows`, `observability_lag`), difficulty
(`difficulty`, `difficulty_rank`, `binding_factors`), scene
(`complexity`, `condition`, `camera_motion`, `n_distractors`, `n_actors`, `n_culprits`,
`actor_shape`, `actor_material`, `actor_mass`), and geometry (`tier`, `num_frames`, `fps`,
`seed`, `variant`).

`difficulty` is what you group by; **`difficulty_rank` is what you filter by**, because the
sets nest and a string comparison cannot say `rank <= 1`.

## 11. Evaluating on it

Report **per family (23)**, aggregate to **domain (8)**, and cross with **severity**,
**complexity** and **condition**.

- **Do not pool families into one number.** Cell counts are uneven by sixteen times —
  `identity` has 48 cells, `optical` has 3 — so a pooled score is largely a measurement of
  `identity`.
- **Use `timelines.active` for temporal metrics and `violation_mask` for spatial ones.** They
  disagree on purpose; see §2.
- **Some families are separable by residual, some only by situation.** `taxonomy.EXCLUSIVE_LAWS`
  names the ones with a clean tripwire, and `tests/test_orthogonality.py` fails if any other
  family moves one. The rest — `antigravity`, `phantom_impulse`, `newton1_inertia` — all move
  `linear_momentum` because they must: bend a body's gravity and its momentum residual moves
  with it. What separates those is the situation, which a model has to read from the image.

---

# Part II — Building it

## 12. Setup (once)

```bash
# Host environment: annotation, severity, grids, validation, viz.
# Does NOT contain Kubric/Blender/PyBullet -- those live in the docker image.
conda env create -f environment.yml
conda activate physloc

docker pull kubricdockerhub/kubruntu      # digest pinned in docker/IMAGE_DIGEST
bash scripts/fetch_refs.sh                # optional, read-only Kubric source
```

**Two environments, never mixed.** The pinned container holds Kubric 2022.4.1 / Blender
2.93.4 / PyBullet (Python 3.9) and does simulation + rendering. The `physloc` conda env
(Python 3.11) does everything else. They meet at the trajectory seam, `traj.npz`.

## 13. The runs

**Each config answers ONE question**, which is what keeps them all minutes rather than hours —
a check you will not run is a check you do not have.
`python -m physloc.cli taxonomy --config <name>` prices any of them exactly, first.

| config | the question it answers |
|---|---|
| `review_severity` | do **weak / medium / strong** differ, for every family? |
| `review_conditions` | do the **five conditions** do what they claim? |
| `review_L0` … `review_L3` | does **this level** render every scene correctly? |
| `review_ladder` | do the levels come out in their **declared proportions**? |
| `review` | does **every cell** build? |
| `v0_mini` | **the whole dataset in miniature** — every family, level, condition and bin |
| `v0_release` | the published dataset, whole ladder, 10 variants |
| `v0_L0` … `v0_L3` | one level of that release, on its own — the four **sum to** `v0_release` |

**What each costs is [one section down](#what-each-config-costs)**, and generated there from
the measured constants. It used to be a column here as well; the two copies disagreed within
a day of being written, which is the whole reason the tables in this file are generated.

**`v0_mini` is not a review sweep** — it is the same structure as the release, made small, so
what you learn from it transfers. It is small in *cells*, not variants, and that is forced:
the condition cycle has ten slots and each level takes a share of them, so it takes **ten
variants** before all four levels and all five conditions appear at all. Three scenarios
(`pour`, `shadow_track`, `drop`) cover all 23 families between them, which is 41 cells
instead of 166.

**`v0_release` is weeks on one box**, and embarrassingly parallel: jobs are independent by
`(scenario, seed, level)` and the per-clip rng is keyed by content rather than queue position,
so N machines is N× faster. Split with `--scenario a,b,c` per machine, or a level apiece with
`v0_L0`…`v0_L3`, and merge the clip trees — nothing collides.

```bash
python -m physloc.cli generate --config review_severity
python -m physloc.cli validate  out/review_severity      # must exit 0
python -m physloc.cli audit     out/review_severity      # cells depicting nothing
python -m physloc.cli stats     out/review_severity      # the distributions, plotted
python -m physloc.cli viz       out/review_severity      # grids + sheets, one folder
python -m physloc.cli coverage  out/review_severity      # every cell, one video
```

**`stats` is how you check the run came out in the shape it declares.** `validate` says a
release is well formed and `audit` says every cell depicts something; neither says what the
*distributions* look like, and those are what a benchmark is judged on. It reads `meta.json`
and nothing else, so it runs in seconds over a full release.

```
out/review_severity/stats/
  difficulty.png          the three labels, and which factor set each one
  difficulty_factors.png  each factor's histogram with its two cuts drawn on it
  composition.png         levels; conditions measured vs declared; difficulty x complexity
  severity.png            measured peak score per declared bin, and the counts
  coverage.png            clips per family and per scenario
  stats.json              the numbers behind all five, so a regression is diffed
```

It earns its place immediately. On `review_ladder` the levels come out **80 / 40 / 24 / 16** —
exactly the declared 1.00 / 0.50 / 0.30 / 0.20 — while the conditions come out
**60 / 5 / 10 / 15 / 10** against a declared 60/10/10/10/10: the per-level spread does not hit
its marginals when each level gets only a few variants. One glance at the middle panel; not
visible in any amount of log reading.

**`viz` is how you look at a finished run**, and it re-reads clips already on
disk — nothing is rendered again, so it takes seconds and can be re-run after
any change to the visualisers.

```
out/review_severity/viz/
  L0_drop_0777_solidity.mp4        one family: the valid clip beside weak/medium/strong
  L0_drop_0777_sheet_strong.mp4    one scene: every family, at one bin
  L0_pour_0777_continuity.mp4
```

`grid` and `sheet` write beside the clips they came from, which is right for a
single look and wrong for reviewing a sweep — a twenty-pair run scatters them
four levels deep across twenty directories, putting the videos you most want to
compare furthest apart. `viz` collects them and names them so the sort order is
the reading order.

```bash
python -m physloc.cli viz out/review_conditions --outdir out/inspect
python -m physloc.cli viz out/review_conditions --severity strong   # sheets: one bin
```

`review_severity` reaches all 23 families in 41 cells rather than 166, because three
scenarios — `pour`, `shadow_track`, `drop` — cover the whole taxonomy between them.
`review_conditions` runs ten variants, exactly one turn of `CONDITION_CYCLE`, so it contains
every condition at least once.

Anything typed on the command line overrides the config:

```bash
python -m physloc.cli generate --config review --scenario drop --family solidity
python -m physloc.cli generate --config review --scenario drop,collision -n 6
python -m physloc.cli generate --config review --complexity all --variants 10
PHYSLOC_CAMERA_MOTION=orbit python -m physloc.cli generate --config review --scenario drop
```

`--workers N` runs N container jobs at once. Measured on an 8-core box: **2.50× at four**,
and eight buys 7% more — Blender already uses every core per render, so workers
oversubscribe. Output is byte-identical at any worker count.

### What each config costs

Every number below is **priced from measured constants**, not estimated — and generated from
them, so it cannot go stale silently. `python -m physloc.cli taxonomy --config <name>` prints
the same figure for any config, with the per-level split.

<!-- physloc:costs -->
| config | levels | cells | renders | at 4 workers |
|---|---|---|---|---|
| `review_severity` | L0 | 41 | 126 | **6 min** |
| `review_conditions` | L0 | 6 | 80 | **6 min** |
| `review_L0` | L0 | 32 | 45 | **6 min** |
| `review_L1` | L1 | 32 | 45 | **6 min** |
| `review_L2` | L2 | 32 | 45 | **18 min** |
| `review_L3` | L3 | 32 | 45 | **18 min** |
| `review_ladder` | L0+L1+L2+L3 | 6 | 160 | **24 min** |
| `review` | L0 | 166 | 511 | **36 min** |
| `v0_mini` | L0+L1+L2+L3 | 41 | 2520 | **6.2 h** |
| `v0_L0` | L0 | 166 | 5110 | 515 h (**21.5 days**) |
| `v0_L1` | L1 | 166 | 2555 | 258 h (**10.7 days**) |
| `v0_L2` | L2 | 166 | 1533 | 405 h (**16.9 days**) |
| `v0_L3` | L3 | 166 | 1022 | 270 h (**11.3 days**) |
| `v0_release` | L0+L1+L2+L3 | 166 | 10220 | 1448 h (**60.4 days**) |
<!-- /physloc:costs -->

Scale by workers: these assume four, which measures **2.50×** on an 8-core box. Eight buys 7%
more, because Blender already uses every core per render. Across N machines it is genuinely
N× — jobs are independent by `(scenario, seed, level)`.

### Where a release's time goes

A level's cost is its variant count times its per-render rate, and those pull in opposite
directions: L0 gets ten variants at the cheap solid rate, L3 two at the expensive HDRI one.
Neither the declared share nor the rate predicts the answer alone, so here it is.

<!-- physloc:costs_ladder -->
| level | variants | renders | per render | at 4 workers | share of the run |
|---|---|---|---|---|---|
| **L0** | 10 | 5110 | 908 s | 515 h (**21.5 days**) | 36% |
| **L1** | 5 | 2555 | 908 s | 258 h (**10.7 days**) | 18% |
| **L2** | 3 | 1533 | 2378 s | 405 h (**16.9 days**) | 28% |
| **L3** | 2 | 1022 | 2378 s | 270 h (**11.3 days**) | 19% |
| **all four** | -- | 10220 | -- | 1448 h (**60.3 days**) | 100% |
<!-- /physloc:costs_ladder -->

**Three quarters of the renders are the cheap half of the bill.** L0 and L1 are 7665 of the
10220 renders and 54% of the time; L2 and L3 are the remaining quarter and 46%, because the
dome charges each of them 2.6× a slab. The ladder's shares put the breadth where it is
cheapest, deliberately — the baseline is what every other level is compared against, so it
should be the largest stratum.

On one machine, `v0_L0` alone is 21 days and is a complete, publishable dataset by itself;
`v0_L2` and `v0_L3` are 675 h together and are what you add when there is capacity for them.
The four **partition** the release, so nothing is wasted and nothing collides.

**Where the numbers come from.** `SECONDS_PER_CLIP` in `physloc/cli.py` is a tier's frame
count times the measured per-frame cost of its background, and `physloc/render/probe_cost.py`
reproduces every per-frame number in it:

| | s/render | frames × measured s/frame |
|---|---|---|
| `debug` solid | **8** | 25 × 0.32, at 128², 16 spp |
| `debug` hdri | **44** | 25 × 1.76 — at 128² the environment map's fixed cost dominates |
| `release` solid | **695** | 89 × 7.81, the mean of L0 and L1 below |
| `release` hdri | **1821** | 89 × 20.46, the mean of L2 and L3 |

The per-render rates in the two tables above are higher than these — 908 s where the constant
says 695 — because `DISTRACTOR_COST` scales every estimate by the extra bodies the average
clip carries: `distractors` and `multi` add 3–10 of them, on 30% of the clips.

**This prices the render and nothing else.** Build, simulation, annotation and the overlay
video are real and are not in the constant, so what `taxonomy` prints is a floor and a run
comes in over it. That is the trade for a number `probe_cost` can reproduce in four frames.

Per frame, at 512², spp 64, all seven passes, on an idle box:

| | L0 | L1 | L2 | L3 |
|---|---|---|---|---|
| s/frame | **7.69** | **7.93** | **19.31** | **21.61** |

The step is the **dome**, not the environment map. L2 and L3 project their HDRI onto a
KuBasic dome, and a dome *encloses* the scene: every ray that misses an object hits it and
bounces, where a flat slab lets those rays escape. L0 and L1 have no dome and cost a third as
much.

**The HDRI rate was the stale one.** It was 2930 s a clip and had never been measured at
release geometry at all — it was the debug tier's 5.5× ratio scaled up, which is exactly the
kind of copy this file's generated tables exist to prevent. At 128² that ratio is real,
because an environment map's fixed per-frame cost dominates a cheap frame; at 512² sampling
dominates and the true ratio is 2.6×. The guess was 60% high, on the quarter of the dataset
that is L2 and L3.

**And the size of the dome is why the levels split at all.** The ground used to be that dome at
*every* level — the fix for a real bug, that a cube below the HDRI level and a dome at it are
different collision shapes, so the same seed did not roll the same way and a level comparison
measured lighting plus a changed floor. It worked, and it charged L0 and L1 **27.54 s/frame
against a slab's 8.69** for a backdrop they are not even lit by. The dome is now a render-only
backdrop at L2 and L3 (`_common.backdrop`, collisions disabled), the ground is a slab
everywhere, and level isolation is unchanged: 0 of 13 scenarios differ.

## 14. Tiers

A tier is a **geometry** — how big and how long. Nothing else. Difficulty is the complexity
ladder; `v0`/`v1` are what a published dataset is *called*, set by `--outdir`.

<!-- physloc:tiers -->
| tier | resolution | frames | duration | spp | latent grid |
|---|---|---|---|---|---|
| `debug` | 128² | 25 @ 12 fps | 2.08 s | 16 | 7×8×8 |
| `release` | 512² | 89 @ 30 fps | 2.97 s | 64 | 23×16×16 |
<!-- /physloc:tiers -->

`debug` is never published. Frame counts are `4k+1` so they map exactly onto a video VAE's
temporal stride.

## 15. Output layout

```
out/release/
  clips/<release>/<level>/<scenario>/<seed>/
      valid/                       one twin, shared by every family and bin
      invalid_<family>_<bin>/      one per cell per severity
```

A *cell* is one (scenario, family) pair. Each is rendered once per severity bin per variant:

```
clips = 166 cells × bins × variants        invalid
      + 13 scenarios × variants            valid
```

A variant is a **fresh seed**, not a re-roll: variant *N* uses `seed + N`, and each complexity
level draws from its own seed block, so no two levels share a scene.

### Generating one level at a time

`v0_L0` … `v0_L3` are `v0_release` split by level, and they **partition** it: each carries the
number of variants that level would get in a full run, so generating all four produces exactly
what `--complexity all` produces, and generating one produces exactly that level's share.

```bash
python -m physloc.cli taxonomy --config v0_L0     # price the level
python -m physloc.cli generate --config v0_L0     # ...and only that level
```

Useful for spreading a release across machines a level at a time, regenerating one level after
a fix, or shipping a smaller dataset that is only ever L0. Nothing collides when the trees
are merged: the clip path is keyed by level already, and every level draws from its own seed
block.

## 16. Publishing

```bash
python -m physloc.cli export out/physloc_v0 --push-to <user>/physloc
```

Packaging always happens; **uploading only when you ask**, because packaging is local and
repeatable and uploading is neither. `run.sh` does both if `PHYSLOC_PUSH_TO` is set
(`PHYSLOC_PUSH_PRIVATE=1` for a private repo). What lands: the dataset card, `index.parquet`
with videos playable inline, `taxonomy.json`, `splits/`, `LICENSE` and the WebDataset shards.

## 17. The generation knobs

Shares, counts and bands used to be module constants spread across three files. They are now
**`configs/common.yaml`** — one place to see them and one place to change them.

```bash
python -m physloc.cli params                    # what is in force, and what differs
python -m physloc.cli params --config v0_mini   # ...for one run
```

| section | what it holds |
|---|---|
| `ladder` | each level's share of a full generation |
| `conditions` | the difficulty cycle — its length is the period, its contents are the shares |
| `objects` | extra-object count, culprit counts, distractor size/speed/clearance |
| `camera` | motion kinds and weights, travel and dolly ranges |
| `materials` | the mass scale |

Any config may override part of it in its own `params:` block, so `common.yaml` holds the
defaults and a run states its differences. Three layers — shipped defaults, `common.yaml`,
the run — each validated against the known tree, so **a typo is an error rather than a value
that silently does nothing**.

The resolved values are written to `params.json` and recorded in every `meta.json`, so a clip
says what it was generated under: a tunable nobody can reproduce is worse than a constant
nobody can change.

They cross the container seam as **JSON, not YAML** — the render container has Kubric's
pinned packages and no PyYAML, and scene sampling happens there.

## 18. Keeping the tables honest

The taxonomy, ladder, condition and tier tables in Part I are **generated** from
`physloc/taxonomy.py` and `physloc/scenarios/base.py` — prose copies of these numbers have
drifted five separate ways before.

```bash
python -m physloc.reference            # is the README current? exits 1 if stale
python -m physloc.reference --write    # regenerate the tables in place
```

`tests/test_reference.py` fails if they are stale, and the HuggingFace card is generated from
the same functions, so the two documents cannot disagree.

## 19. Repo layout

```
physloc/scenarios/    13 scenario builders + the ladder and conditions (base.py)
physloc/injectors/    23 violation families, one file per domain
physloc/render/       the container worker; probes that measured the hard numbers
physloc/annotate/     residuals -> masks, severity, timelines, meta.json
physloc/release/      export, splits, dataset card
configs/*.yaml        the runs above, each documenting every key
docs/schema.md        the meta.json field reference
docs/PLAN.md          the design document
docs/roadmap.md       what is next and why
```

## 20. Papers

[IntPhys 2](https://arxiv.org/abs/2506.09849) · [LikePhys](https://arxiv.org/abs/2510.11512) ·
[Kubric](https://github.com/google-research/kubric). The IntPhys 2 category and LikePhys
domain each family maps to are **data**, on the family in `physloc/taxonomy.py`.
