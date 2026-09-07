# PhysLoc — the design

**What this file is for.** The *reasoning* behind the annotations: why a violation is a set
of intervals rather than an onset, and how severity is derived. That reasoning is still
current and is not written down anywhere else.

**What it is not.** It is not a status report, a repo map, or a source of counts. It used to
be all three and every one of them drifted — five different cell counts appeared across the
docs, and the repo skeleton listed fifteen files that do not exist. For anything countable:

| you want | look at |
|---|---|
| how to run anything | [../README.md](../README.md) |
| counts, families, scenarios, cells | `python -m physloc.cli taxonomy` |
| what a `meta.json` field means | [schema.md](schema.md) |
| locked decisions and traps | [../CLAUDE.md](../CLAUDE.md) |
| what is next, and what is undecided | [roadmap.md](roadmap.md) |

---

## Part 1 — The design that makes this a contribution

Three ideas carry the whole dataset. Everything else is plumbing.

### 1.1 Clocks and windows, not a single onset

Existing benchmarks have no onset at all; the naive version of this dataset would have one
number. A violation is really an **interval** — often several — observed through a second,
lagging interval.

**Three clocks:**

- **`t_event`** — the frame the law breaks *in simulator state*. Known exactly: it is when
  the injector fires.
- **`t_observable`** — the first frame carrying *visual evidence*. If the violation happens
  behind an occluder these differ by a second or more. Computed exactly, not estimated:
  because valid and invalid twins go through a bit-identical render path with the same seed,
  the pixel difference before `t_event` is **exactly zero**, so `t_observable` is the first
  frame where the causal bodies' rendered footprint differs at all.
- **`t_consequence[k]`** — when each *downstream* body first diverges, each with its own
  observability frame.

**Two families of windows.** A single onset cannot describe a super-elastic ball that gains
energy on bounce 1, 2 and 3, nor an object that becomes visible, hides behind the occluder
again, and re-emerges. So both are stored as **lists of intervals**:

- **`violation_windows`** — `[[start, end], …]` in simulator time. The frames during which
  the intervention is actively breaking the law.
  - `instant` (teleport, vanish) → one window of length 1
  - `sustained` (collision pair disabled for N frames, per-body `αg`) → one window of length N
  - `repeated` (restitution `e>1` firing on each of three bounces) → **three windows**
- **`observable_windows`** — `[[start, end], …]`, the frames during which visual evidence is
  actually present. Derived from pixel evidence, so it lags, and it can be interrupted by
  re-occlusion. Not simply a shifted copy of `violation_windows`.

The two are not interchangeable, and the difference is what the spatial annotations are gated
on. `active` is ground truth about the *world*: the frames on which the law is being broken.
`observable` is about the *image*: the frames on which the two renders differ at all. A
super-elastic bounce is unlawful the instant the contact resolves, while the body is still in
exactly the same place in both twins — active, not yet observable. `violation_mask` and
`severity_map` answer "where can this be seen" and so are gated on `active AND observable`;
`timelines.active` keeps the unhedged truth. Severity accumulated while nothing is observable
is carried to the next observable frame inside the same window, so a violation whose residual
is a single invisible spike is not annotated as harmless.

Both are also rasterised to per-frame boolean timelines (`active[T]`, `observable[T]`) so a
consumer never has to expand intervals itself.

This yields metrics no benchmark currently has: **detection latency**
(`predicted_onset − t_observable`), **occlusion lag** (`t_observable − t_event`), and
**duration IoU** (predicted vs true active interval). It also lets us build a split where the
violation is *never* directly observable and only its consequences are — the hardest honest
test in the space.

### 1.2 Violation is a law residual, never "difference from the valid rollout"

This is the trap to design around. After `t_event` a valid/invalid pair diverges
*everywhere* downstream — shadows, contact chains, secondary collisions — and in chaotic
scenes it diverges enormously. **The pixels that differ are not the pixels where the
violation is.** Worse, a merely different-but-valid rollout also diverges, so divergence
cannot define violation at all.

So severity is computed **per body per frame from simulator state**, as a dimensionless
residual against the law that was broken:

| law | residual `r(b,t)` |
|---|---|
| linear momentum | `‖m·Δv − (ΣF_contact + m·g)·Δt‖ / (m·‖g‖·Δt)` |
| angular momentum | same form with torques and `I·Δω` |
| energy at contact | `max(0, E_after − E_before) / E_before` |
| solidity | max signed penetration depth (m), normalised by body radius |
| free fall / support | `‖a − g‖ / ‖g‖` for an unsupported body |
| trajectory shape | RMS deviation (m) of flight path from the fitted `g`-parabola |
| identity | Δvolume ratio, ΔLab, or mass discontinuity |

Residuals are computed for **valid clips too**. They are not exactly zero there — solver
error is real — so a **noise-floor calibration pass** over valid clips is a required
deliverable (Part 3, step 2).

The pixel-space `|valid − invalid|` map is still exported, as `divergence_map`, with an
explicit schema note that it is **not** the violation region — so nobody trains on it by
accident.

### 1.3 Severity is a field, not a flag

**There are two different "magnitudes", and keeping them separate is the point.**

| | **intervention magnitude** | **measured residual** |
|---|---|---|
| what it is | the knob we turned | the effect that knob had |
| known | **before** simulating — we chose it | **after** simulating, from `traj.npz` |
| exact? | yes, by construction | up to the noise floor |
| shape | one scalar per clip | per body, per frame → `[T,H,W]` field |
| lives in | `meta.json:violation.intervention.magnitude` | `residuals.npz`, `severity_map.npz`, `severity_t` |
| used for | building weak/medium/strong splits | training targets, difficulty analysis |

They are **not** the same number and the mapping between them is not the identity — a large
teleport that happens behind an occluder produces a large intervention magnitude and a small
*observable* consequence. That gap is itself a research object, which is why both ship.

There is no "severity mask". There are two distinct arrays: **`violation_mask`** (binary —
*where*) and **`severity_map`** (continuous — *how badly*). Part 3.3 and 3.4 build them, and
§3.4 works a full numeric example end to end.

---

## Part 3 — Annotations: exactly what ships with every clip

This is the contribution. Grouped by the question each annotation answers.

| group | answers | files |
|---|---|---|
| 3.1 labels | *what* and *whether* | `meta.json` |
| 3.2 temporal | ***when***, and ***for how long*** | `meta.json`, `timelines.npz` |
| 3.3 spatiotemporal masks | ***where***, per frame | `violation_mask.npz`, `causal_mask.npz`, `seg.npz` |
| 3.4 severity fields | ***how badly***, localised in space and time | `severity_map.npz`, `severity_t`, `residuals.npz` |
| 3.5 geometry | scene structure, free from Kubric | `depth`, `flow_fwd`, `flow_bwd`, `normals`, `object_coords` |
| 3.6 token grids | ready-to-train reductions | `grids.npz` |
| 3.7 raw physics + provenance | reproducibility | `traj.npz`, `meta.json:provenance` |

### 3.1 Clip-level labels

`label` (valid/invalid), `domain`, `family`, `scenario`, `seed`, `severity_bin`
(weak/medium/strong), the `intervention` block (type, params, `magnitude`, `magnitude_unit`),
`peak_residual` (law, value, frame, z-score), the `controls` flags, `assets` with licenses,
and the cross-reference fields `intphys2_category` / `likephys_domain`.

### 3.2 Temporal annotations — when, and for how long

The part that answers *"a window of when the physics is violated"*.

| annotation | type | meaning |
|---|---|---|
| `t_event_frame` | int | law breaks in simulator state — exact, by construction |
| `t_observable_frame` | int | first frame with visual evidence — exact, via prefix identity |
| `t_end_frame` | int | last frame the intervention is active |
| `observability_lag_frames` | int | `t_observable − t_event` — the occlusion-lag metric |
| **`violation_windows`** | `[[s,e], …]` | **every interval during which the law is actively broken** |
| **`observable_windows`** | `[[s,e], …]` | every interval during which visual evidence is present |
| `consequences[k]` | list | per downstream body: `t_diverge_frame`, `t_observable_frame`, `displacement_m`, `relation` |

Rasterised into **`timelines.npz`** so consumers never expand intervals themselves:

| array | shape | dtype | meaning |
|---|---|---|---|
| `active` | `[T]` | bool | is the violation active at frame `t` |
| `observable` | `[T]` | bool | is there visual evidence at frame `t` |
| `occluded` | `[T]` | bool | is the primary culprit hidden at frame `t` |
| `severity_t` | `[T]` | f32 | violation magnitude over time — `max_b s(b,t)` (§3.4) |

**Why lists of intervals and not a single `[start, end]`:**

- `superelastic` fires on *each* bounce → `violation_windows = [[12,13],[31,32],[47,48]]`
- an object behind an occluder can become visible, re-hide, and re-emerge →
  `observable_windows = [[19,26],[38,50]]`
- `instant` violations are the degenerate case, a single window of length 1

**Invariant, enforced by `physloc validate`:** `t_event ≤ t_observable` and `t_event ≤ t_end`;
`violation_windows` are sorted, non-overlapping, and within `[0, T)`; `active` is exactly the
rasterisation of `violation_windows`.

### 3.3 Spatiotemporal masks — where, per frame

**Yes — spatiotemporal localisation is the core annotation.** Every mask is `[T, H, W]`: a
value per pixel per frame, so "where" and "when" are answered by the same array.

| array | shape | dtype | meaning |
|---|---|---|---|
| **`violation_mask`** | `[T,H,W]` | bool | **the primary annotation.** True on the culprit body's pixels, on frames where the violation is active **and visible**. |
| **`reference_mask`** | `[T,H,W]` | bool | where the culprit *should* be — its footprint in the valid twin, ungated in time. Shipped on **both** clips. |
| `causal_mask` | `[T,H,W]` | uint8 | `0` = nothing, `1` = primary culprit, `k ≥ 2` = consequence body `k−1`. Separates cause from effect spatially. |
| `seg` | `[T,H,W]` | uint16 | instance ids — the substrate every other mask is painted into |
| `divergence_map` | `[T,H,W]` | f16 | `\|valid − invalid\|` in pixel space. **Shipped for analysis, NOT the violation region, never a training target.** |

**How `violation_mask` is built — and the subtlety that makes it correct.** The naive rule
("pixels of the culprit body in the invalid render") is wrong for half our families:

- `permanence` (vanish): the body has **no pixels** in the invalid render. The violation is
  precisely that it is absent.
- `permanence` (duplicate) / `continuity` (teleport): the body is in the **wrong place**, so
  both the place it should be and the place it is are relevant.

So the rule is:

> **`violation_mask[t] = footprint(culprit, invalid, t) ∪ footprint(culprit, valid, t)`**

The union over both twins. This is only well-defined *because* the twins are pixel-aligned
and share instance indexing — the prefix-identity property paying off directly. It handles
vanish (only the valid side contributes), spawn (only the invalid side), and teleport (both,
disjoint) with one rule and no special cases.

**Per-family exceptions, stated explicitly:**

| family | mask region | why |
|---|---|---|
| `newton2_mass` | **both** bodies in the pair | the violation is in the interaction, not in one body |
| `shadow*` | the **shadow** region, not the caster | the shadow is staged as a body of its own, so it segments like any other |
| `global_gravity` | every dynamic culprit, with `spatial_extent: "global"` | there is no single localised culprit; the flag lets consumers exclude these from localisation metrics |
| `solidity` | union of both bodies, restricted to the **overlap region** where available | the violation is the interpenetration itself |

> `newton3_reaction` used to head this table and is **retired** — see
> `taxonomy.RETIRED` for why. There is no shadow-only render pass either; that was planned
> and never built, and the staged-shadow-body approach replaced it.

### 3.4 Severity fields — how badly, localised in space and time

The design the rest of the dataset hangs on. Six steps, from simulator state to a trainable
`[T,H,W]` field.

**Step 1 — per-body, per-frame scalar residual.** For body `b` at frame `t`, compute
`r(b,t)` for the law the injector broke, using the table in §1.2. Dimensionless by
construction. Computed from `traj.npz`, never from pixels.

**Step 2 — calibrate the noise floor.** Solver error means `r > 0` even on valid clips. Over
the valid arm of each `(scenario, law)` pair, collect the residual distribution and record
`μ`, `σ`. This is a required deliverable, not an optimisation — without it "severity" is
uncalibrated. Then

```
z(b,t) = (r(b,t) − μ_scenario,law) / σ_scenario,law
```

**Step 3 — bounded, cross-family-comparable score.** Physical units are not comparable across
families (metres of penetration vs an energy ratio). So map to `[0,1]`:

```
s(b,t) = clip( z(b,t) / z_ref(family), 0, 1 )
```

where `z_ref(family)` is the z-score at that family's `strong` severity bin. **All three of
`r`, `z` and `s` are stored** — physical units for interpretability, z for calibration, `s`
for training.

**Step 4 — paint into pixels.** For each frame, each pixel takes the score of the body
occupying it:

```
severity_map[t, y, x] = s(b, t)   where  b = seg[t, y, x],  b ∈ causal_body_ids
                      = 0         otherwise
```

Occlusion is already resolved by `seg`, so no depth reasoning is needed. Where the mask rule
in §3.3 pulls in the valid twin's footprint (vanish, teleport), the score is painted there
too. Overlaps resolve by `max`. Every *dynamic* culprit is painted, not only the primary one:
`global_gravity` acts on the whole scene and `fission` on both halves, and painting one body
would describe a fraction of the violation.

Painting is gated on `active AND observable`, like the mask. Severity that accrues while
nothing is observable is carried forward to the next observable frame **within the same
window** — otherwise a violation whose residual is a single spike on an invisible frame
(a super-elastic bounce changes velocity while the pixels are still identical; a pendulum's
angular momentum reverses a frame before the arc turns) paints its whole magnitude where
nobody can see it and reads as `0.00` on every frame that shows anything. Where nothing is
hidden the carry is the identity, so a shaped intervention keeps its rise and fall rather
than becoming a running maximum.

**Step 5 — the temporal profile.** `severity_t[t] = max_b s(b,t)`, stored in
`timelines.npz`. This is the 1-D curve a temporal model regresses; `severity_map` is the 3-D
field a spatial model regresses. They are consistent by construction:
`severity_t[t] == severity_map[t].max()`.

**Step 6 — special cases.** `global_gravity` is flagged `spatial_extent: "global"` and paints
its culprits; `shadow*` paints the staged shadow body; `newton2_mass` paints the same
imbalance into both bodies of the pair.

> The code paints through `seg` for every family, including these — see
> `annotate/pipeline.py`. An earlier plan to paint `global_gravity` uniformly over the whole
> frame was not built.

#### The intervention should have a *shape*

A violation implemented as a step — flip a parameter at `t_event` and leave it flipped —
produces a residual that is constant for the rest of the clip, so `severity_map` holds one
value everywhere and the "field" is a flag wearing a costume. It also means there is no
*after*: the clip never returns to legal physics, so nothing shows what recovery looks like.

So a sustained intervention ramps: `antigravity` drives its gravity scale from 1 up to the
bin's peak and back to 1 over the window (a raised cosine), after which the actor obeys real
physics again. The residual traces the same curve, and `severity_t` becomes a genuine
profile — e.g. `0.13 → 0.40 → 0.40 → 0.13 → 0` for the medium bin.

**Two traps this exposed, both of which produce confident nonsense:**

1. **An injector must not smuggle in a second violation.** Bending gravity with plain
   ballistic integration drops the actor straight through the floor — a *solidity* failure
   inside a clip labelled `antigravity`, with only one of the two annotated. Every injector
   that re-integrates motion does so against a solid ground plane
   (`Injector._integrate_profile`). Guarded by
   `test_antigravity_does_not_smuggle_in_a_solidity_violation`.
2. **An intervention must be sustained, not merely peaked.** A raised-cosine ramp touches
   its peak instantaneously, so the *mean* effect over the window is `(1 + peak) / 2` — half
   what the bin advertises. On a body already moving fast that is not enough to change its
   visible motion, and the strongest bin ends up looking like the weakest. The profile is a
   **trapezoid**: ramp in, hold at the peak, ramp out. Ramp values sit strictly between 1 and
   the peak so every frame inside the window is genuinely violating — an `active` frame where
   `alpha == 1` would be a frame marked wrong with nothing wrong in it.
3. **A bounce is a velocity reversal *at the floor*.** The free-fall law asks "is this
   The free-fall law asks "is this body accelerating like gravity?", which is only fair while
   nothing holds it up, so contact frames must be excluded. Two ways to get that gate wrong,
   and we hit both:

   * Gating on "is it resting on the floor" misses bounces, which complete *between* sampled
     frames — the actor is airborne on both neighbours while its velocity reversed in
     between, and the central-difference acceleration straddling that contact reported
     residuals of ~6 against a physical maximum of 2.6. Those spikes also contaminate the
     noise floor measured on the valid arm and swamp the weak bin entirely.
   * Gating on velocity reversal *alone* is worse, and fails silently: reversed gravity flips
     the actor's vertical velocity too, so the gate deletes exactly the frames where the
     violation peaks and reports **zero severity at its strongest moment**.

   The gate therefore requires both — a reversal **and** proximity to the surface, with the
   clearance allowance scaled by `|v_z| · dt` since the contact happens between samples.
   Guarded by `test_bounce_gate_does_not_eat_the_antigravity_signal`.

#### Worked example, end to end — `antigravity` on `drop`

Every number below is either chosen by us or computed from `traj.npz`. Nothing comes from
pixels.

| step | quantity | value | where it comes from |
|---|---|---|---|
| **choose** | gravity scale `α` | `0.3` | the sampler picks it from the `medium` bin |
| **choose** | `intervention.magnitude` | `\|1 − α\| = 0.70` | pure arithmetic on the knob; unit `gravity_scale_deviation` |
| **inject** | `violation_windows` | `[[9, 24]]` | frames the injector applies `g → αg` to body 3 |
| **simulate** | acceleration `a(3,t)` | `≈ 0.3·g` | second difference of `pos[t,3]` from `traj.npz` |
| **measure** | residual `r(3,t)` | `‖a − g‖/‖g‖ = 0.70` | the free-fall law from §1.2 |
| **calibrate** | noise floor `μ, σ` | `0.004, 0.002` | over the *valid* arm of `drop` |
| **normalise** | `z(3,t)` | `(0.70 − 0.004)/0.002 = 348` | z-score vs that floor |
| **bound** | `z_ref(antigravity)` | `498` | the z of the `strong` bin (`α = 0`, so `r = 1.0`) |
| **bound** | `s(3,t)` | `clip(348/498, 0, 1) = 0.70` | the trainable score |
| **paint** | `severity_map[t,y,x]` | `0.70` where `seg[t,y,x] == 3`, else `0` | §3.4 step 4 |
| **reduce** | `severity_t[t]` | `0.70` for `t ∈ [9,24]`, else `0` | `max_b s(b,t)` |

Read the last three rows together and the design becomes concrete: **the magnitude is a
number attached to a body, and the mask is what turns that number into a location.** The
body occupies certain pixels on certain frames (`seg`), so painting its score into those
pixels produces a quantity that is simultaneously *how badly* (`0.70`), *where* (the ball's
footprint) and *when* (frames 9-24).

Note that `r = 0.70` coincides with `magnitude = 0.70` here only because `antigravity` is a
family whose knob and whose law residual happen to share units. For `solidity` the knob is
"disable the contact pair for 15 frames" and the residual is a penetration depth in metres —
related, but not equal, and only knowable after simulating. That is exactly why both are
stored.

**`violation_mask` and `severity_map` are deliberately derived differently**, and a consumer
should know which they are using:

| | `violation_mask` | `severity_map` |
|---|---|---|
| source | the injector's ground truth | measured law residuals |
| value | binary | continuous `[0,1]` |
| exact? | **yes, by construction** | up to the noise floor |
| use for | localisation, detection | magnitude regression, difficulty |

They will *mostly* agree. Where they disagree — a sustained violation whose residual dips
mid-window — that disagreement is real physics, not a bug.

### 3.5 Geometry passes

`depth`, `flow_fwd`, `flow_bwd`, `normals`, `object_coords`, all `[T,H,W,·]`, straight from
Kubric's exporters at no extra cost. Shipped because they make the dataset useful for work
that is not about violations at all.

### 3.6 Token grids

`grids.npz` pre-reduces masks and severity to the latent token grid so a consumer never
re-derives a VAE's binning. the release tier: `7×16×16`. (the 81-frame derivative was never built) `21×16×16` on the 81-frame derivative.

- `mask_<F>x16x16` — bool, reduced by **max** (a violation in any contributing source frame
  marks the latent frame)
- `severity_max_<F>x16x16`, `severity_mean_<F>x16x16` — f16. Peak and average are different
  questions; both ship.
- a `T×32×32` reduction alongside, for consumers on a different tokenizer

**Ordering guarantee:** time-major, `[F_lat, H_lat, W_lat]`, latent frame slowest. This is a
schema guarantee, not an implementation detail — flattening must match transformer token
order.

### 3.7 Raw physics and provenance

`traj.npz` (the seam file: per-body pose, velocity, applied force, contacts, residuals,
events) and the `provenance` block (`generator_commit`, `kubric_image_digest`,
`blender_version`, `render_seed`, `prefix_identical_verified`,
`prefix_identical_upto_frame`). Shipping `traj.npz` means renders reproduce without
re-simulating.

---

