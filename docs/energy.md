# Energy annotation

**Status: built and shipping.** Numbers here are from real PyBullet rollouts at tier
`debug`, not estimates. `tests/test_energy.py` covers it host-side, no docker.

## Why

The taxonomy's third principle is *conservation of momentum and energy*, and until now the
only thing scoring it was `energy_at_contact`, which fires on one family. Energy is
computable **exactly** from the trajectory seam, for both twins, at negligible cost -- so it
can be both a descriptive channel a benchmark consumer reads and a second, independent
scorer for that principle.

It also cross-checks the taxonomy. `deformation` is volume-preserving so its **mass** stays
at exactly 1.0; `immutability` scales volume and its mass moves with it. That is the
mass-vs-shape split of [taxonomy_v2.md](taxonomy_v2.md) showing up in an independent
measurement.

**Volume-preserving is not the same as energy-preserving.** `deformation` does move the
energy budget, because the inertia tensor scales with the shape and squashing a spinning body
changes its rotational kinetic energy — which a rigid body cannot do, and is precisely the
claim the family makes. So the energy signature there is part of the violation, not noise.

What *was* noise, and is now prevented: a body resting on a surface stretched along the
vertical. That lifts its centre of mass, which is real work done from nowhere and has nothing
to do with the aspect ratio, so the clip depicted a shape violation *and* an energy violation
while claiming one. `_Squash` restricts a resting body to horizontal axes for that reason —
the same one-violation-per-clip rule that makes the scaling volume-preserving in the first
place.

## What energy

Total **mechanical** energy of the dynamic bodies:

```
E(t) = Σ_b  ½·m_b(t)·|v_b(t)|²  +  ½·ω_b(t)ᵀ·I_b(t)·ω_b(t)  +  m_b(t)·g·h_b(t)
```

- `v`, `ω`, `pos`, `quat` come straight off the trajectory. No estimation, no fitting.
- `h` is height along `-ĝ` from the scene's `floor_level`, so the datum is fixed per scene
  and absolute values are comparable between twins.
- `I` is the analytic inertia tensor of the primitive -- `⅖mr²` for a sphere,
  `⅓m(b²+c²)` per axis for a box of half-extents `(a,b,c)` -- rotated into world frame by
  the body's quaternion.
- Bodies with `present == False` contribute nothing. That is what makes `permanence` read
  −100%.
- **A body contributes in proportion to how much of it is still there**, `present × opacity`,
  not `present` alone. `present` is a single flip, which is exactly right for `permanence`
  -- a cut is a step -- and wrong for `dissolve`, where the matter goes away over several
  frames. Weighted by `present` alone, an eleven-frame fade and a four-frame fade produced
  the identical one-frame cliff at different moments, so the three severity bins were
  indistinguishable in this channel. Only `dissolve` writes `opacity`; everything else sits
  at 1.0 and is unaffected. `bodies.npz` ships the column, so the trace is still
  recomputable from that file alone (`tests/test_energy.py`).
- **A cast shadow carries no energy at all.** `shadow_track` gives its shadow to a scripted
  stand-in body so that it can have a segmentation id and therefore a mask (see that
  scenario's docstring); the stand-in was declared with a mass like anything else, so it
  contributed a constant 0.33 J to every clip in the scenario and *moving* it moved the
  scene's energy -- the `shadow` family at its strong bin slid the shadow a body-width over
  two frames and the total jumped from 4.12 J to 54.63 J. A shadow is where light is not.
  Bodies with `role == "shadow"` are skipped, and `body_state` reports them as static so the
  two files cannot disagree. The violation is still fully annotated; it just has no joules.

**Mass follows volume**: `m_b(t) = m_b(0) · Π scale_mul[t,b]`. This is a deliberate choice
and it is the one that keeps the annotation consistent with the taxonomy. A body that
doubles in size is depicting twice the matter, so it must carry twice the energy;
`deformation`'s squash is volume-preserving, so its product is exactly 1 and its energy does
not move. Every other family leaves `scale_mul` at 1 and is unaffected either way.

## The two channels

A passive rigid-body scene under gravity, contacts and friction can only **lose** mechanical
energy, and only **at a contact**. That gives two independent anomaly channels, and the
measurement shows they separate different families rather than restating each other.

### 1. Free-energy anomaly -- ΔE on frames with no contact

Energy that changes with nothing touching. No budget model needed; the lawful value is zero.

| | free-loss | free-gain |
|---|---|---|
| **valid twins** | **0.00%** | **0.00%** |
| solidity | 146% | 61% |
| non_parabolic | 48% | 32% |
| support | 47% | — |
| antigravity | 25% | 13% |
| angular_momentum | — | 12% |
| immutability | — | 9% |
| phantom_impulse | 9% | — |

### 2. Contact-budget anomaly -- ΔE at a contact beyond what is allowed

Energy may be lost at a contact, but only within a budget:

- **Restitution.** Normal-direction kinetic energy after a bounce is at most `e²` times
  before it. Gaining any is impossible. `superelastic` reads **−190%** here (a *gain*) while
  sitting at 0.06% in the free channel -- the exact inverse of `support`.
- **Friction.** Frictional dissipation over a step is bounded by `µ·m·g·Δs` for a body on a
  surface -- textbook, and it needs only µ, m, g and path length, all of which the seam
  carries exactly.

Contact *count* is not a signal on its own: valid clips dissipate 27% of `E₀` at contacts,
which is simply what inelastic collisions do. Only the excess over budget means anything.

## Noise floor

The largest frame-to-frame energy **increase** on a valid clip is **0.005% of E₀** -- that is
integrator drift and nothing else. Every violating family above is two to five orders of
magnitude clear of it, so this scores without a tuned threshold. It still goes through
`bounded_score(baseline=r_valid)` like every other law, so a scenario with unusual drift is
scored against its own twin rather than against a global constant.

## What ships

| file | contents |
|---|---|
| `energy.npz` | `total[T]`, `kinetic_translational[T]`, `kinetic_rotational[T]`, `potential[T]`, `by_body[T,B]`, `body_ids[B]`, `dissipated[T]`, `free_anomaly[T]`, `contact_anomaly[T]`, `excess_loss[T]` |
| `energy_map.npz` | `energy[T,H,W]` -- each body's energy painted onto its pixels through the segmentation pass, the same mechanism `severity_map` already uses |
| `bodies.npz` | the physical quantities the energy was computed from — `mass[T,B]`, `velocity[T,B,3]`, `speed`, `momentum`, `angular_momentum`, `inertia`, `height`, `kinetic`, `potential`, plus `gravity` and `dt`. See [schema.md](schema.md) |
| `meta.json` | an `energy` block: `E0`, `E_end`, `peak_free_anomaly`, `peak_contact_anomaly`, `total_dissipated`, all normalised by `E₀` as well as in joules |

`energy_map` is per-body constant within a body's silhouette. For a rigid body that is the
honest spatial resolution -- energy is not a field inside a rigid body, and pretending
otherwise would invent structure the physics does not have.

## Shipped for every family, including the ones that should read zero

A flat trace on `colour_shift` is a **positive, falsifiable statement**: this violation
provably does not touch energy. Omitting it would turn a measured fact into an untested
assumption, and it is what lets the orthogonality test check energy bidirectionally the way
it already checks the exclusive laws.

## Known limits

- **Scripted scenarios.** `pendulum_swing` and `shadow_track` drive a body kinematically, so
  the conservation invariant does not apply to those bodies and the clip-level claim must
  exclude them. Handled by flagging, not by hiding the numbers. (`shadow_track`'s *shadow*
  is a separate case and is excluded outright -- see above. Its actor is still counted.)
- **`total` is the scene; `energy_in_frame` is the evidence.** Energy comes off the
  trajectory, not the pixels, so a body that has left the shot still has a real energy and a
  super-elastic bounce really does add that much. What a viewer -- or a model with only the
  invalid video -- can account for is `energy_in_frame`, which sums the bodies inside the
  frustum and goes to zero when they leave it. Both ship, and `overlay.mp4` plots
  `energy_in_frame` solid with `total` dimmed behind it, so the gap between the two reads as
  "this much energy left the frame rather than the scene". Measured on `drop`:
  `superelastic` at medium holds 19.8 J to the last frame while every pixel of evidence for
  it is gone from frame 14; `solidity` at strong holds 7.9 J while the ball is 17 m below
  the floor.
- **PyBullet's `impulse` field may be a force.** Noted in CLAUDE.md and the cause of an
  earlier residual bimodality. The free-energy channel deliberately needs no force at all,
  and the restitution budget uses velocities rather than the reported impulse, so neither
  depends on resolving this.


## The third channel: excess loss

Energy may be lost at a contact, but only up to the kinetic energy actually available at the
**start** of the step. A perfectly inelastic stop takes all of it and no more; potential
energy cannot be dissipated at all, only converted by the body moving, which the balance
already accounts for.

Getting that end wrong is worth recording: budgeting against the kinetic energy at the *end*
of the step reads 77% excess loss on a perfectly lawful `drop`, because at the impact frame
the ball has already stopped and every joule the floor legitimately absorbed looks
unexplained.

This channel is deliberately not tuned to catch everything. `newton1_inertia` halts a body on
the ground, which costs exactly its kinetic energy and is therefore within budget — a body
stopping dead is energetically identical to one that hit something. That is a momentum
violation and `linear_momentum` is what scores it.

## Measured, per family

Peak anomaly as % of `E0`, worst scenario per family, from real rollouts:

| family | free | gain at contact | excess loss |
|---|---|---|---|
| immutability | — | **768** | — |
| support | — | **400** | — |
| fission | — | **304** | — |
| superelastic | — | **191** | — |
| non_parabolic | **48** | 157 | 1 |
| solidity | **146** | — | — |
| global_gravity | — | **145** | — |
| phantom_impulse | — | **111** | — |
| dissolve / permanence | — | — | **70** |
| fusion | — | 52 | 36 |
| time_slip | — | **38** | — |
| newton2_mass | — | 25 | — |
| antigravity | 20 | 13 | 4 |
| friction | — | 14 | — |
| angular_momentum | **14** | — | — |
| newton1_inertia | — | 7 | — |
| **colour_shift, continuity, deformation** | **at the valid floor** | | |

Valid-clip floors, worst of all three channels: `occluder_pass` 0.000%, `barrier_pass`
0.005%, `collision` 0.007%, `pendulum_swing` 0.31%, `drop` 1.7% (a hard impact strains the
discrete-time balance).

Four families sit at the floor and three of them should: `colour_shift` and `deformation` are
appearance and volume-preserving shape, and `continuity` teleports a body without changing
its height or speed. `newton3_reaction` is the interesting one — suppressing one body's
response breaks momentum without necessarily moving the energy budget, which is a real
statement about the violation rather than a gap in the measurement.

## The scored law is narrower than the shipped trace

`energy_balance` is the **per-body free-energy anomaly**: this body's energy moved while
nothing was touching it. That is the only one of the three channels honestly attributable to
a single body — a body's energy may legitimately jump at a contact because a partner supplied
it, and untangling that needs pairwise bookkeeping the seam does not carry.

So the law claims less than `energy.npz` shows, on purpose. `permanence` and `dissolve` are a
scene-level excess loss and the law does not claim them; `energy.npz` and the `meta.json`
block do.


## Visualisation

The `ENERGY` panel appears in three places, all driven by the same arrays:

- **`physloc overlay`** — panel order is RGB / ENERGY / MASK / SEVERITY / CAUSAL /
  DIVERGENCE. The energy panel carries a VIRIDIS key running `0 → 4x E0` with a tick at the
  frame's largest per-body value, because a field of colours with no key tells you two bodies
  differ but not by how much, which is most of what the panel is for. `E(t)` is drawn inside
  it as a two-line curve: the clip in red, its valid twin in green, playhead on the current
  frame, and the live anomaly percentage when it exceeds 1%. The map answers "where is the
  energy"; the curve answers "what did it do", which is the question the annotation exists
  for. A violation that creates energy is a *step in a line*, and no amount of colouring
  pixels shows a step.
- **`physloc grid --views energy`** — an ENERGY row beside the other annotation rows.
- **`physloc sheet --view energy`** — every family of a scenario at once, each cell showing
  its own `E(t)` in joules.

The map uses VIRIDIS, deliberately not the severity map's INFERNO: the two sit side by side
and answer different questions, so they must not be mistakable for one another at a glance.
