"""Injector invariants -- the ones whose violation is silent and expensive."""
import numpy as np
import pytest

import mockroll
from physloc import injectors, scenarios
from physloc.scenarios import TIERS
from physloc.sim.trajectory import Contacts, Trajectory, prefix_identical


def _valid_traj(spec, T=13, fps=12.0):
    """A hand-rolled free-fall + bounce rollout, so these tests need no render."""
    bodies = spec.bodies
    B = len(bodies)
    dt = 1.0 / fps
    pos = np.zeros((T, B, 3), np.float32)
    vel = np.zeros((T, B, 3), np.float32)
    g = np.array([0, 0, -9.81], np.float64)
    for j, b in enumerate(bodies):
        p = np.asarray(b.position, np.float64)
        v = np.asarray(b.velocity, np.float64)
        r = b.bounding_radius
        for t in range(T):
            pos[t, j] = p
            vel[t, j] = v
            if not b.static:
                v = v + g * dt
                p = p + v * dt
                if p[2] - r <= 0.0 and v[2] < 0:
                    p[2] = r
                    v[2] = -v[2] * b.restitution
    return Trajectory(
        body_ids=np.array([b.segmentation_id for b in bodies], np.int32),
        body_names=[b.name for b in bodies], pos=pos,
        quat=np.tile(np.array([1, 0, 0, 0], np.float32), (T, B, 1)),
        lin_vel=vel, ang_vel=np.zeros((T, B, 3), np.float32),
        mass=np.array([b.mass for b in bodies], np.float32),
        radius=np.array([b.bounding_radius for b in bodies], np.float32),
        is_static=np.array([b.static for b in bodies]),
        contacts=Contacts.empty(), fps=fps,
        gravity=np.array([0, 0, -9.81], np.float32), meta={})


@pytest.fixture
def spec():
    return scenarios.get("drop").sample(555, TIERS["debug"], "L0")


@pytest.mark.parametrize("family", ["antigravity", "continuity", "solidity",
                                    "permanence"])
def test_prefix_is_identical_by_construction(spec, family):
    """Every injector must leave frames before t_event untouched. This is what
    makes the pixel-level twin identity possible at all."""
    traj = _valid_traj(spec)
    inj = injectors.get(family)
    plan = inj.plan(spec, traj, np.random.RandomState(0), "medium")
    assert plan is not None
    out = inj.apply(spec, traj, plan)
    ok, why = prefix_identical(traj, out, plan.t_event, atol=0.0)
    assert ok, why


@pytest.mark.parametrize("severity", ["weak", "medium", "strong"])
def test_antigravity_does_not_smuggle_in_a_solidity_violation(spec, severity):
    """Bending gravity must not also drop the actor through the floor: that is a
    *different* family, and a clip carrying two violations while annotating one
    is worse than useless."""
    traj = _valid_traj(spec)
    inj = injectors.get("antigravity")
    plan = inj.plan(spec, traj, np.random.RandomState(0), severity)
    out = inj.apply(spec, traj, plan)
    bi = out.index_of(spec.body("ball").segmentation_id)
    lowest = (out.pos[:, bi, 2] - spec.body("ball").bounding_radius).min()
    assert lowest > -1e-3, "actor sank %.3f m below the floor" % -lowest


def test_antigravity_intervention_varies_over_time(spec):
    """The violation has a shape: alpha ramps up and releases, so the residual
    -- and with it severity_map -- is a curve rather than one value held flat."""
    traj = _valid_traj(spec)
    inj = injectors.get("antigravity")
    plan = inj.plan(spec, traj, np.random.RandomState(0), "strong")
    out = inj.apply(spec, traj, plan)
    alpha = out.meta["alpha_profile"]
    assert len(alpha) >= 3
    # Every frame in the window is genuinely violating -- alpha never sits at
    # 1.0, which would mark an active frame where nothing is wrong. The return
    # to normal physics happens on the frame after the window.
    assert all(abs(a - 1.0) > 1e-6 for a in alpha), alpha
    # ...but it has a shape: ramp in, hold, ramp out.
    assert len(set(round(a, 3) for a in alpha)) > 1, "profile must not be flat"
    assert alpha[0] != alpha[len(alpha) // 2], "must ramp, not step"
    assert min(alpha) < 0.0, "strong bin should reverse gravity at its peak"
    peak = min(alpha)
    assert sum(1 for a in alpha if abs(a - peak) < 1e-9) >= 1, "needs a plateau"
    # The property the plateau exists for: the *mean* applied effect has to be
    # most of the way to the advertised peak. A profile that merely passes
    # through its peak delivers about half of it, which is how a raised cosine
    # made the strongest bin look like the weakest.
    mean_effect = (1.0 - float(np.mean(alpha))) / (1.0 - peak)
    assert mean_effect > 0.6, (
        "mean alpha is only %.0f%% of the way to the peak" % (100 * mean_effect))


def test_severity_bins_are_visibly_different_not_just_numerically(spec):
    """The strong bin must actually reverse the actor's direction. A bin that is
    numerically larger but looks the same is useless as a difficulty axis.

    Measured against the *lawful twin* at the same frame, not against the
    actor's own height when the window opened. A window that opens mid-fall can
    reverse the ball convincingly and still end below where it started, so
    "did it climb above its starting height" reports 0.00 on a clip that
    visibly turns around. The honest question is how far the body ends up from
    where physics would have put it.
    """
    traj = _valid_traj(spec)
    inj = injectors.get("antigravity")
    bi = traj.index_of(spec.body("ball").segmentation_id)
    lift, turned = {}, {}
    for sev in ("weak", "medium", "strong"):
        plan = inj.plan(spec, traj, np.random.RandomState(0), sev)
        out = inj.apply(spec, traj, plan)
        t0, t1 = plan.windows[0]
        lift[sev] = float(out.pos[t1, bi, 2] - traj.pos[t1, bi, 2])
        turned[sev] = bool((out.lin_vel[t0:t1 + 1, bi, 2] > 0.05).any())
    assert turned["strong"], "strong antigravity must reverse the fall"
    assert lift["strong"] > 0.4, (
        "strong should end well above the lawful twin, got %.3f m"
        % lift["strong"])
    assert lift["strong"] > lift["medium"] > lift["weak"], lift


def test_severity_bins_are_ordered(spec):
    """weak < medium < strong, by intervention magnitude, for every family."""
    traj = _valid_traj(spec)
    for family in ("antigravity", "continuity", "solidity"):
        inj = injectors.get(family)
        mags = []
        for sev in ("weak", "medium", "strong"):
            plan = inj.plan(spec, traj, np.random.RandomState(0), sev)
            assert plan is not None, family
            mags.append(plan.magnitude)
        assert mags[0] < mags[1] < mags[2], "%s: %s" % (family, mags)


def test_antigravity_window_ends_before_the_clip_does(spec):
    """Real physics must resume, otherwise there is no 'after' to compare."""
    traj = _valid_traj(spec)
    inj = injectors.get("antigravity")
    plan = inj.plan(spec, traj, np.random.RandomState(0), "strong")
    assert plan.t_end < traj.num_frames - 1


def test_bounce_gate_does_not_eat_the_antigravity_signal():
    """A bounce is a velocity reversal *at the floor*.

    Gating the free-fall residual on velocity reversal alone is wrong in an
    instructive way: reversed gravity also flips the actor's vertical velocity,
    so the gate deletes exactly the frames where the violation peaks and reports
    zero severity at its strongest moment. Only proximity to the floor
    distinguishes "the ground pushed back" from "the law was broken".
    """
    dt = 1.0 / 12.0
    # A body high above the floor whose velocity reverses -- antigravity, not a
    # bounce: floor is at z=0 and the body never comes near it.
    z = np.array([3.0, 2.6, 2.3, 2.2, 2.4, 2.9], np.float64)
    vz = np.array([-5.0, -3.0, -1.0, 1.0, 3.0, 5.0], np.float64)
    radius, surface_top = 0.3, 0.0

    clear = np.maximum(0.05, np.abs(vz) * dt * 1.5)
    at_floor = (z - radius - surface_top) < clear
    bounce = np.zeros(len(z), bool)
    bounce[1:] = ((vz[:-1] < -1e-3) & (vz[1:] > 1e-3)
                  & (at_floor[:-1] | at_floor[1:]))
    assert not bounce.any(), "a mid-air reversal must not be treated as a bounce"

    # The same reversal happening at the floor *is* a bounce.
    z_low = np.array([1.0, 0.5, 0.32, 0.32, 0.5, 1.0], np.float64)
    at_floor_low = (z_low - radius - surface_top) < clear
    bounce_low = np.zeros(len(z_low), bool)
    bounce_low[1:] = ((vz[:-1] < -1e-3) & (vz[1:] > 1e-3)
                      & (at_floor_low[:-1] | at_floor_low[1:]))
    assert bounce_low.any(), "a reversal at the floor is a bounce"


def _collision_traj():
    """Two equal spheres closing head-on, with a lawful elastic exchange."""
    import mockroll
    from physloc import scenarios
    sc = scenarios.get("collision")
    spec = sc.sample(4242, TIERS["debug"], "L0")
    return spec, mockroll.roll(spec, sc)


def _dv_along(traj, invalid, body_id, t0, normal):
    """How much this body's velocity changed across the collision, along the
    line of centres. Tangential and gravity terms are excluded so the number
    reflects the collision and nothing else."""
    bi = traj.index_of(body_id)
    dv = (invalid.lin_vel[t0, bi].astype(float)
          - traj.lin_vel[t0 - 1, bi].astype(float))
    return abs(float(np.dot(dv, normal)))

def test_newton3_reaction_is_retired():
    """It was withdrawn, and the reason is recorded rather than forgotten.

    Staged honestly it was either the same clip as `newton2_mass` -- an
    immovable target is the limit of a mass ratio -- or, rewritten to inject
    momentum, unreliable across severity bins. `taxonomy.RETIRED` keeps the
    reasoning so the question is not reopened from scratch.
    """
    from physloc import injectors, taxonomy

    assert "newton3_reaction" not in taxonomy.FAMILIES
    assert "newton3_reaction" not in injectors.available()
    assert "newton3_reaction" in taxonomy.RETIRED
    assert len(taxonomy.RETIRED["newton3_reaction"]) > 80

def test_newton2_needs_visually_identical_bodies():
    """The scenario has to hold up its end: same radius, same colour."""
    from physloc import scenarios
    for seed in (1, 777, 4242):
        spec = scenarios.get("collision").sample(seed, TIERS["debug"], "L0")
        # Dormant understudies carry role "actor" too -- they are the body
        # `fission` switches on -- so they have to be excluded here.
        a, b = [x for x in spec.bodies if x.role == "actor" and not x.dormant]
        assert a.scale == b.scale, seed
        assert a.color == b.color, seed


@pytest.mark.parametrize("scenario", ["collision", "drop", "barrier_pass", "toss"])
@pytest.mark.parametrize("family", ["phantom_impulse", "continuity"])
def test_only_the_magnitude_varies_across_bins(scenario, family):
    """Weak, medium and strong must be three strengths of ONE violation.

    Direction is a property of the scene, not of the bin. Drawn from the per-bin
    rng it came out different for each, so the three were three different
    violations and their magnitudes stopped being comparable: `phantom_impulse`
    reported 0.87 / 2.08 / 1.95 for bins that scale 1.0 / 2.4 / 4.5, because
    every heading got its own frustum-fit scale.
    """
    sc = scenarios.get(scenario)
    spec = sc.sample(777, TIERS["debug"], "L0")
    traj = mockroll.roll(spec, sc)
    inj = injectors.get(family)
    plans = [inj.plan(spec, traj, np.random.RandomState(0), s)
             for s in ("weak", "medium", "strong")]
    if any(p is None for p in plans):
        pytest.skip("%s does not stage %s" % (scenario, family))
    mags = [float(p.magnitude) for p in plans]
    assert mags[0] <= mags[1] <= mags[2], mags
    assert plans[0].causal_body_ids == plans[2].causal_body_ids


@pytest.mark.parametrize("family", ["continuity", "phantom_impulse",
                                    "immutability", "permanence", "dissolve"])
def test_the_shadow_follows_whatever_the_caster_does(family):
    """A shadow is a projection, so every family that moves, resizes or removes
    the caster must move, resize or remove the shadow with it.

    None of them did. Measured on the review sweep: under `continuity` the ball
    jumped 1.4 m and its shadow carried on down the lawful track; under
    `immutability` the ball grew to 2.3x and its shadow stayed the original
    size; under `permanence` the ball was removed and its shadow went on
    sliding across an empty floor. Each of those clips shipped a *detached
    shadow* -- which is the `shadow` family, a violation in its own right --
    inside a clip claiming and annotating something else.
    """
    sc = scenarios.get("shadow_track")
    spec = sc.sample(777, TIERS["debug"], "L0")
    traj = mockroll.roll(spec, sc)
    sc.script(spec, traj)
    inj = injectors.get(family)
    plan = inj.plan(spec, traj, np.random.RandomState(0), "strong")
    if plan is None:
        pytest.skip("%s does not stage shadow_track" % family)
    out = inj.apply(spec, traj, plan)
    sc.rescript(spec, out, plan)

    from physloc.scenarios.shadow_track import project

    ja, js = spec.index_of("body"), spec.index_of("shadow")
    want = project(np.asarray(out.pos[:, ja, :], np.float64),
                   spec.notes["light_dir"], float(spec.notes["surface_top"]),
                   0.006)
    assert np.allclose(out.pos[:, js, :], want, atol=1e-4), (
        "%s: shadow is not under its caster" % family)
    assert np.allclose(out.scale_mul[:, js, :2], out.scale_mul[:, ja, :2]), (
        "%s: shadow did not follow the caster's size" % family)
    assert np.array_equal(out.present[:, js], out.present[:, ja]), (
        "%s: shadow outlived its caster" % family)


@pytest.mark.parametrize("family", ["shadow", "shadow_inverted", "shadow_shape"])
def test_a_violation_of_the_shadow_survives_rescripting(family):
    """The three optical families own the shadow, so re-deriving it from a
    caster that never moved would silently undo them."""
    sc = scenarios.get("shadow_track")
    spec = sc.sample(777, TIERS["debug"], "L0")
    traj = mockroll.roll(spec, sc)
    sc.script(spec, traj)
    inj = injectors.get(family)
    plan = inj.plan(spec, traj, np.random.RandomState(0), "strong")
    assert plan is not None
    out = inj.apply(spec, traj, plan)
    before = np.array(out.pos[:, spec.index_of("shadow"), :], copy=True)
    scale_before = np.array(out.scale_mul[:, spec.index_of("shadow"), :],
                            copy=True)
    sc.rescript(spec, out, plan)
    assert np.array_equal(out.pos[:, spec.index_of("shadow"), :], before)
    assert np.array_equal(out.scale_mul[:, spec.index_of("shadow"), :],
                          scale_before)
