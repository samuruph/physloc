"""Mechanical energy annotation -- docs/energy.md.

Host-side only: every number here comes off a trajectory, so none of it needs
docker.
"""
from __future__ import annotations

import numpy as np
import pytest

import mockroll
from physloc import scenarios
from physloc.injectors import get as get_injector
from physloc.residuals import energy as E
from physloc.scenarios import TIERS

SEED = 777


def _scene(name, seed=SEED):
    sc = scenarios.get(name)
    spec = sc.sample(seed, TIERS["debug"], "L0")
    return sc, spec, mockroll.roll(spec, sc)


def test_free_fall_trades_potential_for_kinetic():
    """No contact and no friction, so kinetic and potential trade one for one --
    to the integrator's order, which is the honest claim.

    A semi-implicit Euler step conserves discrete energy only to O(dt), and
    `mockroll` is exactly that, so the total drifts ~0.15% per frame here.
    Real PyBullet rollouts drift 0.005% (docs/energy.md), which is why the
    anomaly channels have the headroom they do. What must hold in both is the
    *direction*: free flight never gains energy.
    """
    sc, spec, traj = _scene("drop")
    trace = E.compute(traj, spec)
    early = trace.total[:4]
    assert np.allclose(early, early[0], rtol=1e-2), early
    assert np.all(np.diff(early) <= 1e-9), "free flight gained energy"
    # The trade is real, not both terms sitting still.
    ke = trace.kinetic_translational[:4]
    pe = trace.potential[:4]
    assert ke[-1] > ke[0] and pe[-1] < pe[0]


@pytest.mark.parametrize("name", ["drop", "collision", "barrier_pass",
                                  "occluder_pass", "ramp_slide"])
def test_valid_clips_never_gain_energy(name):
    """A passive scene under gravity, contacts and friction can only lose
    mechanical energy. This is the invariant every anomaly channel is measured
    against, so if it does not hold the annotation means nothing."""
    sc, spec, traj = _scene(name)
    trace = E.compute(traj, spec)
    assert trace.contact_anomaly.max() < 0.05, (
        "%s valid clip gains %.1f%% of E0 at a contact"
        % (name, 100 * trace.contact_anomaly.max()))
    assert trace.free_anomaly.max() < 0.05, (
        "%s valid clip moves %.1f%% of E0 with nothing touching it"
        % (name, 100 * trace.free_anomaly.max()))


def _peak(trace):
    """Largest anomaly over all three channels.

    Which channel a family lands in depends on whether the body happens to be
    touching something when the intervention fires -- `immutability` on a ball
    in flight is a free-energy event, and on a ball resting on the floor it is a
    gain at contact. That is correct behaviour and not something a test of "does
    this family move the energy budget" should pin.
    """
    return max(trace.free_anomaly.max(), trace.contact_anomaly.max(),
               trace.excess_loss.max())


def test_superelastic_creates_energy_and_valid_twin_does_not():
    sc, spec, traj = _scene("drop")
    inj = get_injector("superelastic")
    plan = inj.plan(spec, traj, np.random.RandomState(0), "strong")
    assert plan is not None
    invalid = inj.apply(spec, traj, plan)
    v = E.compute(traj, spec)
    i = E.compute(invalid, spec)
    assert _peak(i) > 20 * max(_peak(v), 1e-4)
    assert i.total[-1] > v.total[-1]


def test_deformation_is_mass_neutral_but_immutability_is_not():
    """The taxonomy's mass-vs-shape split, measured rather than asserted.

    `deformation` is volume-preserving, so mass follows volume to exactly 1 and
    no matter is created. `immutability` scales volume uniformly, so it does.
    If this ever fails, one of the two injectors has stopped meaning what
    the taxonomy says it means.

    **Mass, not energy.** Deformation does move the energy budget, and for a
    reason that belongs to the violation rather than to how it is staged: the
    inertia tensor scales with the shape, so squashing a spinning body changes
    its rotational kinetic energy. A rigid body cannot do that, which is
    precisely the claim the family makes.

    What *was* an artefact -- and is now prevented -- is a resting body being
    stretched vertically. That lifts its centre of mass, which is real work
    done from nowhere and has nothing to do with the aspect ratio; the clip
    would depict a shape violation and an energy violation while claiming one.
    `_Squash` restricts a resting body to horizontal axes for that reason.
    """
    sc, spec, traj = _scene("drop")
    base = E.body_state(traj, spec)["mass"]
    for family, grows in (("immutability", True), ("deformation", False)):
        inj = get_injector(family)
        plan = inj.plan(spec, traj, np.random.RandomState(0), "strong")
        assert plan is not None, family
        m = E.body_state(inj.apply(spec, traj, plan), spec)["mass"]
        changed = bool(np.abs(m - base).max() > 1e-3 * base.max())
        assert changed is grows, "%s: mass changed=%s" % (family, changed)
    # And the volume ratio itself, which is what "volume-preserving" means.
    inj = get_injector("deformation")
    plan = inj.plan(spec, traj, np.random.RandomState(0), "strong")
    out = inj.apply(spec, traj, plan)
    vol = np.prod(out.scale_mul, axis=2)
    assert np.allclose(vol, 1.0, atol=1e-5), "squash changed the volume"


@pytest.mark.parametrize("name", ["drop", "collision", "resting_table",
                                  "barrier_pass", "stack_topple"])
def test_deformation_never_lifts_a_resting_body(name):
    """The staging rule, checked where it matters: on the scenarios whose actors
    are in contact with a surface when the squash fires.

    The invariant is that no work appears from nowhere -- so the body's centre
    of mass must never RISE. It used to be checked as "z never changes at all",
    and that was the bug rather than the guarantee: a volume-preserving stretch
    along x shortens the body vertically, so holding its centre still lifts it
    clear of the floor by the difference. That is what shipped, and what you
    see in a clip is a cube that stops touching the ground the moment it
    squashes. Seating it back on the surface *lowers* the centre of mass, which
    releases energy rather than creating it, and is what a real object does.
    """
    sc, spec, traj = _scene(name)
    inj = get_injector("deformation")
    plan = inj.plan(spec, traj, np.random.RandomState(0), "strong")
    if plan is None:
        pytest.skip("%s does not stage deformation" % name)
    if not plan.notes.get("resting"):
        pytest.skip("%s deforms an airborne body" % name)
    assert plan.notes["axis"] in (0, 1), (
        "resting body stretched along z: that lifts its centre of mass, which "
        "is an energy violation this family never claimed")
    out = inj.apply(spec, traj, plan)
    j = traj.index_of(int(plan.causal_body_ids[0]))
    t0 = plan.t_event
    assert np.all(out.pos[t0:, j, 2] <= traj.pos[t0:, j, 2] + 1e-6), (
        "deformation raised the body's centre of mass")

    # And it stays ON the surface: the bottom of the squashed body tracks the
    # bottom of the lawful one, rather than floating above it.
    r = float(traj.radius[j])
    sz = out.scale_mul[t0:, j, 2].astype(np.float64)
    bottom_valid = traj.pos[t0:, j, 2].astype(np.float64) - r
    bottom_squashed = out.pos[t0:, j, 2].astype(np.float64) - r * sz
    assert np.allclose(bottom_valid, bottom_squashed, atol=1e-5), (
        "deformation changed the body's clearance to its support")


def test_vanishing_body_shows_up_as_excess_loss():
    """`permanence` takes a body's potential energy with it, which no contact
    can account for."""
    sc, spec, traj = _scene("barrier_pass")
    inj = get_injector("permanence")
    plan = inj.plan(spec, traj, np.random.RandomState(0), "strong")
    assert plan is not None
    i = E.compute(inj.apply(spec, traj, plan), spec)
    v = E.compute(traj, spec)
    assert i.excess_loss.max() > 0.2
    assert v.excess_loss.max() < 0.05


def test_energy_map_paints_only_the_bodies_pixels():
    sc, spec, traj = _scene("drop")
    trace = E.compute(traj, spec)
    T = traj.num_frames
    seg = np.zeros((T, 8, 8), np.uint16)
    bid = int(traj.body_ids[0])
    seg[:, 2:5, 2:5] = bid
    emap = E.energy_map(trace, seg)
    assert emap.shape == (T, 8, 8)
    for t in range(T):
        outside = emap[t][seg[t] != bid]
        assert np.all(outside == 0.0)


def test_inertia_matches_the_textbook_forms():
    sphere = E.inertia_diag("sphere", (0.5, 0.5, 0.5), 2.0)
    assert np.allclose(sphere, 0.4 * 2.0 * 0.25)
    cube = E.inertia_diag("cube", (0.5, 0.5, 0.5), 3.0)     # side 1.0
    assert np.allclose(cube, 3.0 / 6.0 * 1.0 ** 2)


def test_law_is_silent_without_a_spec_rather_than_wrong():
    from physloc.residuals import laws
    sc, spec, traj = _scene("drop")
    got = laws._LAWS["energy_balance"](traj, 0, {})
    assert got.shape == (traj.num_frames,)
    assert np.all(got == 0.0)


def test_body_state_reproduces_the_shipped_energy():
    """The published clip must be self-contained: a consumer with `bodies.npz`
    and nothing else must be able to recompute the energy trace and get the
    shipped one back."""
    sc, spec, traj = _scene("drop")
    trace = E.compute(traj, spec)
    st = E.body_state(traj, spec)
    here = st["present"] & ~st["static"][None, :]
    recomputed = ((st["kinetic"] + st["potential"]) * here).sum(axis=1)
    assert np.allclose(recomputed, trace.total, rtol=1e-6, atol=1e-6)


def test_body_state_columns_are_physically_consistent():
    sc, spec, traj = _scene("collision")
    st = E.body_state(traj, spec)
    # momentum = m*v, and its magnitude agrees with the vector
    assert np.allclose(st["momentum"], st["mass"][..., None] * st["velocity"],
                       rtol=1e-5, atol=1e-6)
    assert np.allclose(st["momentum_magnitude"],
                       np.linalg.norm(st["momentum"], axis=-1), rtol=1e-5)
    assert np.allclose(st["speed"], np.linalg.norm(st["velocity"], axis=-1),
                       rtol=1e-5)
    # potential = m*g*h against the same datum the trace uses
    g = float(np.linalg.norm(st["gravity"]))
    assert np.allclose(st["potential"], st["mass"] * g * st["height"],
                       rtol=1e-5, atol=1e-6)
    # Dynamic bodies only: PyBullet spells "static" as mass 0, and the floor
    # keeps that convention all the way through the seam.
    assert np.all(st["mass"][:, ~st["static"]] > 0)


def test_mass_follows_volume_in_the_shipped_columns():
    """`bodies.npz` must tell the same story the energy does: uniform growth
    adds matter, volume-preserving squash does not."""
    sc, spec, traj = _scene("drop")
    base = E.body_state(traj, spec)["mass"]
    for family, grows in (("immutability", True), ("deformation", False)):
        inj = get_injector(family)
        plan = inj.plan(spec, traj, np.random.RandomState(0), "strong")
        assert plan is not None, family
        m = E.body_state(inj.apply(spec, traj, plan), spec)["mass"]
        changed = bool(np.abs(m - base).max() > 1e-3 * base.max())
        assert changed is grows, "%s: mass changed=%s" % (family, changed)


def test_bodies_npz_is_self_describing():
    """`seg == body_ids[j]` must be the mask for `body_names[j]`.

    Shipping ids without names forces every consumer to cross-reference
    meta.json to find out which track is which.
    """
    sc, spec, traj = _scene("collision")
    st = E.body_state(traj, spec)
    assert len(st["body_names"]) == len(st["body_ids"])
    assert list(st["body_names"]) == list(traj.body_names)
    declared = {int(b.segmentation_id): b.name for b in spec.bodies}
    for bid, name in zip(st["body_ids"], st["body_names"]):
        assert declared[int(bid)] == str(name)
