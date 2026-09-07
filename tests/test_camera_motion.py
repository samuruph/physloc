"""The camera track: how often it moves, what it does, and what it must not do.

A moving camera is the one scene property that changes what every frame looks
like without changing the physics, so the things worth pinning are the ones a
render would only reveal expensively.
"""
import os

import numpy as np
import pytest

from physviol import scenarios
from physviol.scenarios import TIERS
from physviol.scenarios.base import (CAMERA_MOTION_KINDS, MOVING_CAMERA_SHARE,
                                     DOLLY_RANGE)

NAMES = sorted(scenarios.available())
SEEDS = range(120)


def _specs(name, complexity="L0"):
    sc = scenarios.get(name)
    return [sc.sample(s, TIERS["debug"], complexity) for s in SEEDS]


def test_about_a_fifth_of_clips_move_the_camera():
    """The share the dataset promises, within sampling noise.

    Checked in aggregate rather than per scenario: at 120 seeds a single
    scenario's count has a standard deviation of about 4.4 points, so a
    per-scenario bound tight enough to be meaningful would flake.
    """
    moving = sum(sp.camera_moves for name in NAMES for sp in _specs(name))
    total = len(NAMES) * len(SEEDS)
    share = moving / float(total)
    assert abs(share - MOVING_CAMERA_SHARE) < 0.04, (
        "%.1f%% of clips move the camera, expected about %.0f%%"
        % (100 * share, 100 * MOVING_CAMERA_SHARE))


def test_the_decision_is_not_shared_across_scenarios():
    """Each scenario flips its own coin.

    Salted on the seed alone, every scenario made the same decision for a given
    seed -- measured at exactly 22% for all thirteen, which is one coin flip
    reported thirteen times. It looks like a healthy 22% until you notice the
    number is identical everywhere.
    """
    per = {name: sum(sp.camera_moves for sp in _specs(name)) for name in NAMES}
    counts = [c for n, c in per.items() if n != "occluder_pass"]
    assert len(set(counts)) > 1, (
        "every scenario moves the camera on exactly the same seeds: %s" % per)


def test_occluder_pass_never_moves_the_camera():
    """Its occlusion interval is precomputed from one pose.

    `_occluded_frames` intersects a camera->ball ray with the screen plane once,
    at sample time, and the frame list it returns is where every observability
    label in the dataset comes from. A camera that moved would change which
    frames are hidden and the list would describe a different clip.
    """
    assert not any(sp.camera_moves for sp in _specs("occluder_pass"))


@pytest.mark.parametrize("name", NAMES)
def test_the_camera_track_is_identical_across_complexity(name):
    """L0 and L1 must film the same clip from the same place.

    v1 exists to ask whether a model's grasp of the physics survives realism,
    which only means anything if the two renders are otherwise the same shot.
    """
    sc = scenarios.get(name)
    for seed in (0, 7, 4242):
        a = sc.sample(seed, TIERS["v0"], "L0")
        b = sc.sample(seed, TIERS["v0"], "L1")
        assert a.camera_motion_kind == b.camera_motion_kind, (name, seed)
        assert a.camera_end_position == b.camera_end_position, (name, seed)


def test_orbit_holds_its_distance_and_dolly_does_not():
    """The two motions have to differ in exactly the way their names claim.

    An orbit that drifts in distance is a dolly with extra steps, and it would
    swell and shrink the actor mid-arc -- which is the cue `immutability` and
    `deformation` make their claim about. Interpolating the two endpoint
    POSITIONS does exactly that, because the straight line between them cuts
    the chord; the angle has to be interpolated instead.
    """
    T = 25
    seen = {}
    for name in NAMES:
        for sp in _specs(name):
            if sp.camera_moves:
                seen.setdefault(sp.camera_motion_kind, sp)
    for kind in CAMERA_MOTION_KINDS:
        assert kind in seen, "no %s clip in %d samples" % (kind, len(NAMES) * len(SEEDS))

    def radii(sp):
        aim = np.asarray(sp.camera_look_at, np.float64)
        return np.array([np.linalg.norm(np.asarray(sp.camera_at(f, T)[0]) - aim)
                         for f in range(T)])

    r = radii(seen["orbit"])
    assert (r.max() - r.min()) / r[0] < 1e-6, (
        "orbit changed its distance to the subject by %.2f%%"
        % (100 * (r.max() - r.min()) / r[0]))

    r = radii(seen["dolly"])
    change = abs(r[-1] - r[0]) / r[0]
    assert DOLLY_RANGE[0] - 1e-6 <= change <= DOLLY_RANGE[1] + 1e-6, (
        "dolly moved %.1f%%, outside its declared %s" % (100 * change, DOLLY_RANGE))


def test_a_static_camera_really_is_constant():
    """`camera_at` must not drift on a clip that declares no motion."""
    sp = next(sp for name in NAMES for sp in _specs(name) if not sp.camera_moves)
    poses = {sp.camera_at(f, 25)[0] for f in range(25)}
    assert len(poses) == 1, "a static camera reported %d poses" % len(poses)


def test_the_debug_override_forces_a_kind():
    """`PHYSVIOL_CAMERA_MOTION` is how a motion gets looked at without hunting
    for a seed that draws it. It must never be set during a release run, so it
    is worth a test that says out loud what it does."""
    old = os.environ.get("PHYSVIOL_CAMERA_MOTION")
    try:
        for kind in CAMERA_MOTION_KINDS:
            os.environ["PHYSVIOL_CAMERA_MOTION"] = kind
            kinds = {scenarios.get("drop").sample(s, TIERS["debug"], "L0")
                     .camera_motion_kind for s in range(8)}
            assert kinds == {kind}, (kind, kinds)
        os.environ["PHYSVIOL_CAMERA_MOTION"] = "off"
        assert not any(scenarios.get("drop").sample(s, TIERS["debug"], "L0")
                       .camera_moves for s in range(30))
    finally:
        os.environ.pop("PHYSVIOL_CAMERA_MOTION", None)
        if old is not None:
            os.environ["PHYSVIOL_CAMERA_MOTION"] = old
