# Roadmap — macro-categories, perception, and the realistic twin

Three questions came out of reviewing the v0 clips, and they have different answers.

---

## 1. Macro-categories like LikePhys — yes, and they cost almost nothing

LikePhys groups by **medium**: Rigid Body, Optical, Fluid, Continuum. PhysLoc groups by
**law**: identity, kinematics, contact, dynamics, equilibrium, optical, appearance, global.

These are not competing schemes. They are two independent groupings of the same cells, and
that is exactly what makes them worth having both of:

- **medium** answers *what kind of matter is misbehaving* — the axis that maps onto prior art
  and onto a capability question ("does this model handle deformables at all?")
- **domain** answers *which principle is broken* — the axis severity is measured against,
  since the domain determines the residual

So the taxonomy grows a level at the top rather than being reorganised:

```
Level 0  MEDIUM    what kind of matter        (5)   <- new, maps to LikePhys
Level 1  DOMAIN    which law                  (8)
Level 2  FAMILY    how it breaks             (20)
Level 3  SCENARIO  the staged scene          (15)
Level 4  INSTANCE  scenario x family x seed x severity
```

| medium | scenarios | status |
|---|---|---|
| `rigid` | 12 of the 15 | built |
| `granular` | `pour` | built — the fluid stand-in, never labelled fluid |
| `optical` | `shadow_track` | built |
| `fluid` | — | **Phase 3**: needs a Blender with working headless Mantaflow |
| `continuum` | — | **Phase 3**: cloth and soft bodies, needs MuJoCo/MJX behind the seam |

`physics_medium` already exists on every scenario and is already cross-checked by
`physloc validate`; today it only takes `rigid` and `granular`. Promoting it means widening
the enum, tagging `shadow_track` as `optical`, and reporting on it. **Half a day**, and it
makes head-to-head comparison with LikePhys a table rather than an argument.

The two empty rows are the honest part: declaring `fluid` and `continuum` as known-empty is
more useful than omitting them, because it says what the dataset does *not* cover without a
reader having to infer it from absence.

---

## 2. Perceptual violations — the half of v0 that is still missing

`deformation` and `shadow_shape` landed. Two more belong in v0, and both are mechanically
cheap because the render path already animates what they need.

### `colour_shift` — the object changes colour — **BUILT**

**Verified working in the pinned image.** `material.keyframe_insert("color", frame)` creates
one fcurve per RGBA channel on the Principled BSDF's Base Colour, and the rendered pixels
follow. Needs a `Trajectory.colour[T,B,3]` channel, keyframed in `replay` exactly the way
`scale_mul` already is.

Bins by perceptual distance: a hue nudge, a clear shift, a jump to the complementary colour.
Law: `colour_continuity`, the CIE-Lab distance from the body's own frame 0 — Lab rather than
RGB so "how different does this look" is measured the way a viewer would rank it.

Ramped, not switched, for the same reason `immutability` is: a colour that changes between
two frames is a cut, one that visibly shifts is a violation.

### `illumination_shift` — the light changes with nothing to change it

Light `color` and `intensity` are both keyframable traits. The shadows and shading follow
automatically, which is what makes it a *scene-wide* perceptual violation rather than an
object one — the counterpart to `global_gravity` in the appearance domain.

Needs a `LightSpec` animation channel. Note it necessarily co-moves with `shadow`, since
moving the light moves the shadow, so the two must not be built on the same scenario.

### Deliberately not in v0

| | why |
|---|---|
| `texture_swap` | needs GSO assets, which arrive with the complexity ladder in v1 |
| `reflection` | no scenario has a reflective surface, and adding one is a lighting project |
| `transparency` | **built** as the `dissolve` family, though it took three attempts. The first two probes concluded it was impossible; both were measuring the *floor*, because they never called `kb.adjust_segmentation_idxs`. Mixing a Transparent BSDF into the shader works cleanly — see `render/probe_opacity.py`, which keeps the mistake on record |

---

## 3. v1 — the same physics under harder conditions

**v1 is not a bigger render.** It has exactly the same tier geometry as v0 — 512×512, 30 fps,
89 frames — and differs on two other axes:

| axis | v0 | v1 |
|---|---|---|
| tier | 512², 30 fps, 89 f | **the same** |
| complexity | `L0` — solid background, one sun lamp | `L1` — HDRI environment, photographic objects |
| population | `single` | `single` **and** `multi` |

Making v1 a resolution step *as well* would confound the axes. A model that scored worse on
v1 could be failing at realism, at clutter, or merely at a resolution it had not been trained
on, and the release could not say which. Fixing the geometry is what makes v0 and v1
**paired**, which is the entire point of the axis.

### 3a. The complexity twin — and the one thing blocking it

The design is that `(scenario, seed)` names *the same physical event*, rendered plainly in v0
and photographically in v1. A benchmark can then ask the question that matters: does a model's
grasp of the physics survive the realism, or was it reading the plain background?

That requires the rollout to be **bit-identical across complexity**, and today it is not.

- **Fixed.** `pick_hdri(rng)` drew from the physics stream, and because it only fires at L1
  the extra draw shifted every physics value after it. Appearance now has its own salted
  stream (`_common.appearance_rng`), so the draw order no longer depends on complexity.
- **Still open.** `C.ground` returns a **cube** at L0 and a KuBasic **dome** at L1. That is a
  genuine geometry change — different collision surface, different support height — so the
  same seed still produces a different rollout.

  The fix is to make the collision geometry identical at both levels and let complexity vary
  only the *material*, the lighting and the backdrop. Two candidates, neither yet chosen:
  use the dome at L0 too, shaded flat instead of with an HDRI; or keep the cube at L1 and add
  the dome as a non-colliding backdrop body. The second changes body counts and segmentation
  ids across complexity, which the first does not, so the first is probably right.

  **Until this is fixed there is no complexity twin**, only two independent releases that
  happen to share a seed. `tests/` should grow a check that
  `sample(seed, tier, "L0")` and `sample(seed, tier, "L1")` roll identically, so the property
  cannot regress once it holds.

### 3b. Population — single vs multi

A third orthogonal axis beside severity and complexity, in `physloc/scenarios/base.py`
next to `TIERS` and `COMPLEXITY`:

```python
POPULATIONS = {
    "single": Population("single", n_actors=(1, 1), varied=False),
    "multi":  Population("multi",  n_actors=(3, 6), varied=True),
}
```

Two settings, not three. A `crowd` level would add render cost without adding a distinct kind
of violation — `pour` and `clutter_toss` are purpose-built for dense scenes and cover that
ground better.

`Scenario.sample` is already a template method, so the expansion happens in one place:
`_sample` stages the physics, `_vary` moves the camera, and a new `_populate` clones the
primary actor into N siblings with their own kinds, sizes, colours and jittered initial
state, drawing from a scenario-declared `notes["spawn_box"]`. Scenarios needing bespoke
placement (`resting_table`, `stack_topple`, `collision`) override it; inherently fixed ones
(`pendulum_swing`, `shadow_track`, `pour`) set `supports_population = False`.

**Varied object kinds.** `BodySpec.kind` gains a `"kubasic:<id>"` form, built with the same
`AssetSource.create(...)` call the dome already uses. Usable ids, confirmed against the
manifest: `cone, cube, cylinder, gear, sphere, sponge, spot, suzanne, teapot, torus,
torus_knot`. Two traps, both verified:

- **`torus_knot` takes an underscore.** Kubric's own `KUBASIC_OBJECTS` lists `torusknot`,
  which does not exist — `get_random_kubasic_object` crashes about one time in eleven. Ship
  our own tuple.
- **Bounds are not all ±0.5.** `cone`, `torus`, `gear`, `spot`, `suzanne`, `teapot` and
  `torus_knot` are asymmetric, and `bounding_radius` feeds residual normalisation, every
  support test, and now the inertia tensor in `residuals/energy.py`. Bake an id → (bounds,
  mass) table into `scenarios/_kubasic.py`, the way `_hdri.py` bakes HDRI ids, so host-side
  sampling needs no network.

### 3c. Multi-culprit annotation

`multi` is what forces it: three of five objects starting to slide at frames 3, 5 and 8 is
one clip with three culprits and three different windows.

**A culprit becomes a record inside the plan; the clip-level fields become derived reductions
over those records.** `plan.windows` is *defined* as the merged union, `t_event` as the min
over culprits, `causal_body_ids` as the list of their ids. A third role, `participant`,
covers a body that is in `causal_body_ids` but is not independently scored — the static floor
a ball sinks through, the victim of the second half of a `fission`. It
inherits the primary's windows and score, **which is exactly what the pipeline does today**,
so the first migration step is a no-op provable by byte-diffing a regenerated release.

The released format grows a body axis **beside** the union, never instead of it:
`timelines.npz` keeps `active[T]` and gains `active_by_body[T,N]`; `violation_mask.npz` keeps
`mask[T,H,W]` and gains `mask_by_body[T,N,H,W]`. Additive, because N is not a fixed axis
(`pour` has ~40 culprits, `global_gravity` has none), because the union is the documented
training target and a CLAUDE.md non-negotiable, and because two independently computed views
can be **checked against each other** — `mask == mask_by_body.any(1)`. A single source of
truth would make a window attached to the wrong body id perfectly self-consistent and
undetectable.

### 3d. Randomisation depth

`--variants N` already redraws every free parameter a scenario has. What v1 adds is *what
there is to draw from*: GSO objects instead of primitives, the full HDRI set instead of the
baked shortlist, and wider camera and layout sampling. This is the cheapest of the three
pieces and should land last, because it multiplies whatever the other two produce.

### Order

1. **Make the complexity twin real** — the floor-geometry fix in 3a, plus the regression test.
   Nothing else in v1 means anything until `(scenario, seed)` names one physical event.
2. **Population axis in the scenes**, still one culprit per clip. All 180 cells run unchanged
   in a busier scene, which is what makes this safe to land on its own.
3. **Multi-culprit annotation**, one family at a time. `support` and `friction` first — they
   *are* the "three of five objects start sliding" case.
4. **Randomisation depth.**

## Where this stands

**Done (v0 complete):** `colour_shift`, `deformation`, `shadow_shape`, `dissolve` (optical,
via a Transparent BSDF mix), `fusion`, medium as Level 0, and the compatibility matrix
derived from declared capabilities rather than written out by hand. Also: materials with
matching density, five actor shapes, per-scene floor and backdrop colour, camera motion on
about a fifth of clips, and `physloc export` for publication.

For the current counts run `python -m physloc.cli taxonomy` — they are not repeated here,
because every prose copy of them in this repository drifted.

**Still open, in order:**

1. **`illumination_shift`** — the scene's light changes with nothing to change it. Light
   `color` and `intensity` are both keyframable traits, so the mechanism is proven; it needs
   a `LightSpec` animation channel. Completes the appearance domain. Note it necessarily
   co-moves with `shadow`, so the two must not share a scenario.
2. **Population + multi-culprit** — the big structural piece (§3a).
3. **Complexity ladder L2–L4** — the realistic twin (§3b).
4. **Randomisation depth** — mostly falls out of 3 (§3c).

1 is the last of v0. 2–4 are v1.

---

## Open questions

Carried over from `docs/decisions_pending.md`, which was deleted once most of its contents
had shipped. These are the parts that had not.

**Does `causal_mask` earn its place?** Raised and then deferred by the user. What it is for,
whether consumers can use it, and whether it is distinct enough from `violation_mask` and
`reference_mask` to justify another array per clip. Decide it against real clips, not in the
abstract.

**Release configuration.** Which severity bins and how many variants per cell the published
run uses. `configs/v0_release.yaml` proposes tier v0 / L0 / all three bins / 3 variants;
`physloc taxonomy --config v0_release` prices it exactly. **Settle this with the user before
starting** — it is the difference between an overnight job and a week.

**Taxonomy v2.** A proposal to replace `domain` with a derived `principle`, agreed in outline
and never built; the file arguing for it was deleted with the rest of the speculative docs.
It is re-labelling only — no re-rendering — so it stays cheap for as long as it is unbuilt.

**Fission is still `geometric`.** Its draw-in is prescribed motion rather than something the
simulator produces.
