"""When a violation fires, and whether it can be seen afterwards.

Feedback: event moments barely varied -- one draw per scene, a quarter to 45%
of the way in, shared by every family -- and nothing stopped a violator leaving
the frame right after its event, so a clip could claim a violation whose
effects nobody could see. These pin the replacement: a draw per (scene,
family, violator, attempt) inside `EVENT_BAND`, a visible-span requirement,
and a framing check on the lawful rollout.
"""
from __future__ import annotations

import copy

import numpy as np
import pytest

import mockroll
from physloc import injectors, scenarios
from physloc.injectors import _geom
from physloc.injectors.base import Injector
from physloc.scenarios import TIERS

TIER = TIERS["debug"]


def _spec(seed=5, name="drop"):
    sc = scenarios.get(name)
    return sc, sc.sample(seed, TIER, "L0")


def test_band_frame_stays_in_band_and_leaves_room_to_see_the_effect():
    T = TIER.num_frames
    frames = set()
    for seed in range(60):
        _, spec = _spec(seed)
        t = _geom.band_frame(spec, T)
        assert t is not None
        lo = round(_geom.EVENT_BAND[0] * T)
        assert lo <= t <= round(_geom.EVENT_BAND[1] * T)
        assert T - 1 - t >= _geom.min_visible_frames(spec, T)
        frames.add(t)
    # Spread across the band rather than clustered on a few frames.
    assert len(frames) >= 6


def test_event_draw_is_keyed_on_family_violator_and_attempt():
    _, spec = _spec(9)
    with _geom.event_context("permanence"):
        a = _geom.event_fraction(spec)
        assert a == _geom.event_fraction(spec)
        per_body = {_geom.event_fraction(spec, body_id=i) for i in range(5)}
    with _geom.event_context("immutability"):
        b = _geom.event_fraction(spec)
    with _geom.event_context("permanence", attempt=1):
        c = _geom.event_fraction(spec)
    assert len({a, b, c}) == 3
    assert len(per_body) == 5


def test_every_family_plan_runs_in_its_own_event_context():
    for family in ("permanence", "continuity", "phantom_impulse", "solidity"):
        plan = type(injectors.get(family)).plan
        assert getattr(plan, "_event_keyed", False), family
    assert issubclass(type(injectors.get("permanence")), Injector)


def test_severity_bins_share_a_moment_and_attempts_move_it():
    sc, spec = _spec(21)
    traj = mockroll.roll(spec, sc)
    inj = injectors.get("permanence")
    rng = lambda: np.random.RandomState(3)                      # noqa: E731
    moments = {inj.plan(spec, traj, rng(), sev).t_event
               for sev in ("weak", "medium", "strong")}
    assert len(moments) == 1
    moved = set()
    try:
        for attempt in range(4):
            inj.event_attempt = attempt
            moved.add(inj.plan(spec, traj, rng(), "strong").t_event)
    finally:
        inj.event_attempt = 0
    assert len(moved) > 1


@pytest.mark.parametrize("family", ["phantom_impulse", "continuity", "permanence"])
def test_retries_move_the_moment_even_where_an_occlusion_is_preferred(family):
    """`occluder_pass` prefers its occluded midpoint, a FIXED frame. All four
    attempts used to choose it, so a retry could never escape a moment that had
    just failed the visibility gate."""
    sc = scenarios.get("occluder_pass")
    spec = sc.sample(22260826, TIER, "L0")
    traj = mockroll.roll(spec, sc)
    inj = injectors.get(family)
    moments = []
    try:
        for attempt in range(4):
            inj.event_attempt = attempt
            plan = inj.plan(spec, traj, np.random.RandomState(1), "strong")
            if plan is not None:
                moments.append(plan.t_event)
    finally:
        inj.event_attempt = 0
    if len(moments) < 2:
        pytest.skip("%s plans on too few attempts here" % family)
    assert len(set(moments)) > 1, moments


def test_families_on_one_scene_no_longer_all_fire_together():
    differ = 0
    for seed in range(12):
        sc, spec = _spec(seed)
        traj = mockroll.roll(spec, sc)
        a = injectors.get("permanence").plan(spec, traj,
                                             np.random.RandomState(0), "strong")
        b = injectors.get("immutability").plan(spec, traj,
                                               np.random.RandomState(0), "strong")
        differ += int(a.t_event != b.t_event)
    assert differ >= 6


@pytest.mark.parametrize("name", ["drop", "collision", "pyramid_impact"])
def test_declared_physical_anchor_keeps_appearance_events_near_contact(name):
    sc, spec = _spec(7, name)
    traj = mockroll.roll(spec, sc)
    cfg = spec.notes["event_anchor"]
    bodies, partners = set(cfg["body_ids"]), set(cfg["partner_ids"])
    contacts = [int(traj.contacts.frame[k]) for k in range(len(traj.contacts))
                if ((int(traj.contacts.body_a[k]) in bodies
                     and int(traj.contacts.body_b[k]) in partners)
                    or (int(traj.contacts.body_b[k]) in bodies
                        and int(traj.contacts.body_a[k]) in partners))]
    if not contacts:
        pytest.skip("mock rollout did not produce the declared contact")
    plan = injectors.get("deformation").plan(
        spec, traj, np.random.RandomState(1), "strong")
    assert plan is not None
    centre = min(contacts) + int(cfg["offset"])
    assert abs(plan.t_event - centre) <= int(cfg["jitter"])


def test_physical_anchor_mixes_near_and_broad_timing():
    """Anchored scenes keep a modal contact cue without freezing on it."""
    near = broad = 0
    for seed in range(40):
        sc, spec = _spec(seed, "drop")
        traj = mockroll.roll(spec, sc)
        cfg = spec.notes["event_anchor"]
        bodies = set(cfg["body_ids"])
        partners = set(cfg["partner_ids"])
        contacts = [int(traj.contacts.frame[k]) for k in range(len(traj.contacts))
                    if ((int(traj.contacts.body_a[k]) in bodies
                         and int(traj.contacts.body_b[k]) in partners)
                        or (int(traj.contacts.body_b[k]) in bodies
                            and int(traj.contacts.body_a[k]) in partners))]
        if not contacts:
            continue
        plan = injectors.get("deformation").plan(
            spec, traj, np.random.RandomState(1), "strong")
        if plan is None:
            continue
        centre = min(contacts) + int(cfg["offset"])
        if abs(plan.t_event - centre) <= int(cfg["jitter"]):
            near += 1
        else:
            broad += 1
    assert near >= 12
    assert broad >= 6


def test_collision_anchor_allows_late_broad_branch():
    """The collision cue is modal, not a hard upper bound on timing."""
    late = 0
    for seed in range(40):
        sc, spec = _spec(seed, "collision")
        traj = mockroll.roll(spec, sc)
        cfg = spec.notes["event_anchor"]
        bodies, partners = set(cfg["body_ids"]), set(cfg["partner_ids"])
        contacts = [int(traj.contacts.frame[k]) for k in range(len(traj.contacts))
                    if ((int(traj.contacts.body_a[k]) in bodies
                         and int(traj.contacts.body_b[k]) in partners)
                        or (int(traj.contacts.body_b[k]) in bodies
                            and int(traj.contacts.body_a[k]) in partners))]
        if not contacts:
            continue
        plan = injectors.get("deformation").plan(
            spec, traj, np.random.RandomState(1), "strong")
        if plan is not None and plan.t_event > min(contacts):
            late += 1
    assert late >= 4


def test_permanence_scores_only_disappearance_and_return_transitions():
    sc, spec = _spec(4, "barrier_pass")
    traj = mockroll.roll(spec, sc)
    plan = injectors.get("permanence").plan(
        spec, traj, np.random.RandomState(1), "medium")
    assert plan is not None
    t0, t1 = plan.consequence_windows[0]
    assert plan.intervention_windows == [(t0, t0), (t1 + 1, t1 + 1)]
    invalid = injectors.get("permanence").apply(spec, traj, plan)
    from physloc.residuals import laws
    residual = laws.get("mass_continuity")(
        invalid, invalid.index_of(plan.causal_body_ids[0]), {})
    assert residual[t0] == 1.0 and residual[t1 + 1] == 1.0


def _thrown_out(traj, body_id, frame):
    out = copy.deepcopy(traj)
    bi = out.index_of(int(body_id))
    out.pos[frame:, bi, :] = np.array([500.0, 500.0, 500.0], np.float32)
    return out


def test_eligible_frames_end_where_the_body_must_still_be_seen():
    sc, spec = _spec(4)
    traj = mockroll.roll(spec, sc)
    ball = spec.actors[0]
    T = traj.num_frames
    need = _geom.min_visible_frames(spec, T)
    gone = 18
    lost = _thrown_out(traj, ball.segmentation_id, gone)
    ok = _geom.eligible_event_frames(spec, lost, [ball], 0, T - 1)
    assert ok.size and ok.max() <= gone - need
    assert _geom.violators_visible(spec, traj, [ball], 5)
    assert not _geom.violators_visible(spec, lost, [ball], gone - 2)


def test_framing_rejects_a_scene_whose_actor_leaves_early():
    sc, spec = _spec(4)
    traj = mockroll.roll(spec, sc)
    assert sc.framing_ok(spec, traj)
    assert not sc.framing_ok(spec, _thrown_out(traj, spec.actors[0].segmentation_id,
                                               TIER.num_frames // 5))


def test_occluder_pass_frames_its_ball_as_visible():
    """Its ball passes behind a screen by design. Testing only the centre called
    it hidden whenever the middle was covered, and framing failed every seed."""
    sc = scenarios.get("occluder_pass")
    for seed in range(5):
        spec = sc.sample(seed, TIER, "L0")
        assert sc.framing_ok(spec, mockroll.roll(spec, sc)), seed


def test_a_body_is_hidden_only_when_all_of_it_is():
    sc = scenarios.get("occluder_pass")
    spec = sc.sample(3, TIER, "L0")
    eye = np.asarray(spec.camera_position, np.float64)
    screen = np.asarray(spec.body("screen").position, np.float64)
    behind = eye + (screen - eye) * 1.3          # on the ray, past the screen
    assert _geom.fully_hidden_behind_static(spec, behind, 0.05)
    # Wide enough that its silhouette pokes out past the screen's edges.
    assert not _geom.fully_hidden_behind_static(spec, behind, 5.0)


def test_a_brief_exit_counts_as_visible_but_leaving_at_once_does_not():
    sc, spec = _spec(4)
    traj = mockroll.roll(spec, sc)
    ball = spec.actors[0]
    bi = traj.index_of(int(ball.segmentation_id))
    need = _geom.min_visible_frames(spec, traj.num_frames)
    head = max(1, int(round(_geom.EVIDENCE_SECONDS * TIER.fps)))
    t = 5
    assert _geom.violators_visible(spec, traj, [ball], t)
    # Climbs out of shot after the evidence and comes back.
    wide = copy.deepcopy(traj)
    wide.pos[t + head + 1:t + head + 1 + int(0.3 * need), bi, :] = 500.0
    assert _geom.violators_visible(spec, wide, [ball], t)
    # Gone the moment the violation happens: nothing to see.
    gone = copy.deepcopy(traj)
    gone.pos[t:t + head, bi, :] = 500.0
    assert not _geom.violators_visible(spec, gone, [ball], t)


def test_framing_attempt_zero_is_the_scene_the_seed_always_made():
    sc = scenarios.get("collision")
    base = sc.sample(8, TIER, "L0")
    same = sc.sample(8, TIER, "L0", attempt=0)
    other = sc.sample(8, TIER, "L0", attempt=2)
    assert base.to_dict() == same.to_dict()
    assert other.seed == base.seed
    assert other.notes["framing_attempt"] == 2
    assert other.to_dict()["bodies"] != base.to_dict()["bodies"]
    # And it round-trips: the same attempt rebuilds the same scene.
    assert sc.sample(8, TIER, "L0", attempt=2).to_dict() == other.to_dict()


def test_motion_families_fire_before_the_actor_comes_to_rest():
    """Antigravity on a ball already lying on the floor shows nothing."""
    sc, spec = _spec(6)
    traj = mockroll.roll(spec, sc)
    T = traj.num_frames
    bi = traj.index_of(int(spec.actors[0].segmentation_id))
    speed = np.linalg.norm(traj.lin_vel[:, bi, :], axis=1)
    # Make the ball settle a third of the way in, whatever the mock did.
    still = copy.deepcopy(traj)
    rest = T // 3 + 3
    still.lin_vel[rest:, bi, :] = 0.0
    with _geom.event_context("antigravity", traj=still):
        limit = _geom.motion_limit(spec)
        assert limit is not None and limit < rest
        for seed in range(30):
            spec.seed = seed
            t = _geom.band_frame(spec, T)
            assert t is None or t <= max(limit, round(_geom.EVENT_BAND[0] * T))
    # Identity and appearance families may act on a body at rest.
    with _geom.event_context("permanence", traj=still):
        assert _geom.motion_limit(spec) is None
    assert speed.size == T


@pytest.mark.parametrize("fps,frames", [(12, 25), (12, 37), (12, 61), (30, 89)])
def test_motion_events_land_before_rest_at_every_clip_length(fps, frames):
    """The limit was a share of the clip, so it outgrew the motion on long
    clips and was dropped: antigravity then fired seconds after the drop.

    The ball here falls, bounces, and by one second is only ROLLING slowly --
    which is not motion anything a gravity violation does could be seen on."""
    base = TIERS["debug"] if fps == 12 else TIERS["release"]
    tier = base.override(fps=fps, num_frames=frames)
    sc = scenarios.get("drop")
    spec = sc.sample(6, tier, "L0")
    traj = mockroll.roll(spec, sc)
    bi = traj.index_of(int(spec.actors[0].segmentation_id))
    settle = int(round(1.0 * fps))
    still = copy.deepcopy(traj)
    still.lin_vel[:, bi, :] = 0.0
    still.lin_vel[:settle, bi, 2] = -6.0              # the fall and bounces
    still.lin_vel[settle:, bi, 0] = 0.4               # then a slow roll
    for seed in range(40):
        spec.seed = seed
        with _geom.event_context("antigravity", traj=still):
            t = _geom.band_frame(spec, frames)
        assert t is not None and 1 <= t < settle, (fps, frames, seed, t)


@pytest.mark.parametrize("name", ["drop", "collision", "rolling_ramp"])
def test_default_event_frames_respect_the_clip(name):
    sc = scenarios.get(name)
    for seed in range(5):
        spec = sc.sample(seed, TIER, "L0")
        traj = mockroll.roll(spec, sc)
        for family in ("permanence", "continuity"):
            plan = injectors.get(family).plan(spec, traj,
                                              np.random.RandomState(1), "strong")
            if plan is None:
                continue
            assert 1 <= plan.t_event < traj.num_frames - 1
