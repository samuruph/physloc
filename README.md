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

**Four rungs of scene realism**, each the one below it plus exactly one thing:

<!-- physloc:ladder -->
| level | adds | background | objects | materials | share | built |
|---|---|---|---|---|---|---|
| **L0** | *the baseline* | solid | primitive | one shared density | 50% | yes |
| **L1** | **materials** — wood, steel, rubber | solid | primitive | yes | 25% | yes |
| **L2** | **an HDRI environment** | hdri | primitive | yes | 15% | yes |
| **L3** | **GSO objects** — real 3D scans | hdri | gso | yes | 10% | yes |
<!-- /physloc:ladder -->

Every clip carries its rung in `complexity`, so a rung is a filter rather than a separate
download. **Each rung draws its own scenes** — an L1 clip is not an L0 clip in better
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

## 7. What else varies

Every free parameter is drawn per clip from the seed: object shape, size, colour, mass,
starting position and velocity, floor and backdrop colour, camera pose, and **the frame the
violation fires on**.

**Materials** (from L1) give appearance and density that agree:

<!-- physloc:materials -->
| material | density kg/m³ |
|---|---|
| `cork` | 240 |
| `wood` | 650 |
| `plastic` | 1100 |
| `rubber` | 1300 |
| `ceramic` | 2400 |
| `stone` | 2700 |
| `steel` | 7800 |
<!-- /physloc:materials -->

**Environments** (from L2): 36 HDRI Haven captures. **Objects** (L3): 48 Google Scanned
Objects, all CC BY-SA 4.0, curated squat and roughly isotropic so their lawful motion is
predictable — an object that topples unexpectedly reads as the violation.

`python -m physloc.cli randomisation` reports distinct values per axis, so *"is it actually
varied"* is a number rather than an impression.

## 8. Splits

`main` **75%** · `held_out` **20%** · `debug` **5%**.

**Grouped by `pair_uid`, never by clip.** A valid twin and its invalid siblings share every
frame before `t_event`, so splitting them apart would put the answer on the other side.
Pairs are ordered by a hash of their uid and cut at the quantiles *within each scenario*, so
every split sees every scenario in the same proportions. There is no rng — reproducing the
release reproduces its splits.

**Every split ships every annotation.** The split is a label, not a filter: you can re-cut it,
and you can score any clip. If you need a genuinely blind held-out set for a leaderboard,
strip the annotations at that point — nothing has to be regenerated.

## 9. The index

`index.parquet` is one row per clip, with the video embedded so it plays inline in the
HuggingFace viewer. A breakdown is a groupby, not a crawl over thousands of `meta.json`:

```python
import pandas as pd
df = pd.read_parquet("index.parquet")

df.groupby(["domain", "severity_bin"]).peak_severity.mean()
df.groupby("condition").size()                       # standard / camera / multi / ...
df[df.complexity == "L3"].groupby("family").size()   # what survives the hardest rung
df[df.n_culprits > 1]                                # the multi-object clips
```

Columns: identity (`clip_uid`, `pair_uid`, `twin_uid`, `label`, `split`), taxonomy
(`scenario`, `family`, `domain`, `medium`), violation (`severity_bin`, `magnitude`,
`peak_severity`, `t_event_frame`, `violation_windows`, `observability_lag`), scene
(`complexity`, `condition`, `camera_motion`, `n_distractors`, `n_actors`, `n_culprits`,
`actor_shape`, `actor_material`, `actor_mass`), and geometry (`tier`, `num_frames`, `fps`,
`seed`, `variant`).

## 10. Evaluating on it

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

## 11. Setup (once)

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

## 12. The runs

**Each config answers ONE question**, which is what keeps them all minutes rather than hours —
a check you will not run is a check you do not have.
`python -m physloc.cli taxonomy --config <name>` prices any of them exactly, first.

| config | the question it answers | cost |
|---|---|---|
| `review_severity` | do **weak / medium / strong** differ, for every family? | ~8 min |
| `review_conditions` | do the **five conditions** do what they claim? | ~5 min |
| `review_L0` … `review_L3` | does **this rung** render every scene correctly? | ~3–18 min |
| `review_ladder` | do the rungs come out in their **declared proportions**? | ~18 min |
| `review` | does **every cell** build? | ~35 min |
| `v0_release` | the published dataset, whole ladder | **price it first** |
| `v0_L0` … `v0_L3` | one rung of that release, on its own | the four **sum to** `v0_release` |

```bash
python -m physloc.cli generate --config review_severity
python -m physloc.cli validate  out/review_severity      # must exit 0
python -m physloc.cli audit     out/review_severity      # cells depicting nothing
python -m physloc.cli viz       out/review_severity      # grids + sheets, one folder
python -m physloc.cli coverage  out/review_severity      # every cell, one video
```

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

## 13. Tiers

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

## 14. Output layout

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
rung draws from its own seed block, so no two rungs share a scene.

### Generating one rung at a time

`v0_L0` … `v0_L3` are `v0_release` split by rung, and they **partition** it: each carries the
number of variants that rung would get in a full run, so generating all four produces exactly
what `--complexity all` produces, and generating one produces exactly that rung's share.

```bash
python -m physloc.cli taxonomy --config v0_L0     # price the rung
python -m physloc.cli generate --config v0_L0     # ...and only that rung
```

Useful for spreading a release across machines a rung at a time, regenerating one rung after
a fix, or shipping a smaller dataset that is only ever L0. Nothing collides when the trees
are merged: the clip path is keyed by rung already, and every rung draws from its own seed
block.

## 15. Publishing

```bash
python -m physloc.cli export out/physloc_v0 --push-to <user>/physloc
```

Packaging always happens; **uploading only when you ask**, because packaging is local and
repeatable and uploading is neither. `run.sh` does both if `PHYSLOC_PUSH_TO` is set
(`PHYSLOC_PUSH_PRIVATE=1` for a private repo). What lands: the dataset card, `index.parquet`
with videos playable inline, `taxonomy.json`, `splits/`, `LICENSE` and the WebDataset shards.

## 16. Keeping the tables honest

The taxonomy, ladder, condition and tier tables in Part I are **generated** from
`physloc/taxonomy.py` and `physloc/scenarios/base.py` — prose copies of these numbers have
drifted five separate ways before.

```bash
python -m physloc.reference            # is the README current? exits 1 if stale
python -m physloc.reference --write    # regenerate the tables in place
```

`tests/test_reference.py` fails if they are stale, and the HuggingFace card is generated from
the same functions, so the two documents cannot disagree.

## 17. Repo layout

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

## 18. Papers

[IntPhys 2](https://arxiv.org/abs/2506.09849) · [LikePhys](https://arxiv.org/abs/2510.11512) ·
[Kubric](https://github.com/google-research/kubric). The IntPhys 2 category and LikePhys
domain each family maps to are **data**, on the family in `physloc/taxonomy.py`.
