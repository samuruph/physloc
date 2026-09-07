"""The camera track: how often it moves, what it does, and what it must not do.

A moving camera is the one scene property that changes what every frame looks
like without changing the physics, so the things worth pinning are the ones a
render would only reveal expensively.
"""
import os

import numpy as np
import pytest

from physloc import scenarios
from physloc.scenarios import TIERS
from physloc.scenarios.base import (CAMERA_MOTION_KINDS, COMPLEXITY,
                                    DOLLY_RANGE, stratify)

NAMES = sorted(scenarios.available())
BASE = 777
N_VARIANTS = 20


#: Camera motion is an ORTHOGONAL AXIS, not a rung: every level moves
#: `camera_share` of its clips, L0 included. Tested at L0 because that is where
#: the rest of the scene is held stillest, so anything the camera does is the
#: only thing that changed.
CAMERA_LEVEL = "L0"
CAMERA_SHARE = COMPLEXITY[CAMERA_LEVEL].camera_share


def _specs(name, complexity=CAMERA_LEVEL, n=N_VARIANTS):
    """One spec per VARIANT, which is what the camera rule is defined over."""
    sc = scenarios.get(name)
    return [sc.sample(BASE + v, TIERS["debug"], complexity, variant=v)
            for v in range(n)]


def test_every_scenario_moves_the_camera_on_the_same_share_of_variants():
    """`camera_share` of them, PER SCENARIO -- not a coin flip per scene.

    An independent flip gives the right share overall and an uneven one per
    scenario: measured across thirteen scenarios it ranged from 13% to 31%, so
    some scenarios had moving cameras and others effectively did not, and any
    per-scenario comparison inherited that as a confound.
    """
    expected = sum(stratify(v, CAMERA_SHARE) for v in range(N_VARIANTS))
    assert expected == round(N_VARIANTS * CAMERA_SHARE), (
        "the stratifier must deliver the declared share exactly")
    for name in NAMES:
        if name == "occluder_pass":
            continue                       # opts out entirely; see below
        moving = [sp.variant for sp in _specs(name) if sp.camera_moves]
        assert len(moving) == expected, (
            "%s moved on %d of %d variants, expected %d"
            % (name, len(moving), N_VARIANTS, expected))


def test_the_same_variants_move_in_every_scenario():
    """Which variants move is a property of the INDEX, not of the scenario.

    That is what makes the share identical everywhere, and it also means a
    consumer can say "variant 4 is the moving-camera one" without consulting a
    table.
    """
    sets = {name: tuple(sp.variant for sp in _specs(name) if sp.camera_moves)
            for name in NAMES if name != "occluder_pass"}
    assert len(set(sets.values())) == 1, sets


def test_a_short_run_is_entirely_static():
    """Fewer variants than the period means no camera motion at all.

    Asked for explicitly: a short run should be the easy case unless motion is
    requested. `stratify` fires on the LAST variant of each block, so a run of
    four never reaches it -- a run only spends clips on camera motion once it
    is long enough to afford them.
    """
    for n in range(1, int(1.0 / CAMERA_SHARE)):
        for name in NAMES:
            assert not any(sp.camera_moves for sp in _specs(name, n=n)), (
                "%s moved the camera in a %d-variant run" % (name, n))


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
    """Two rungs must film the same clip from the same place.

    The ladder exists to ask whether a model's grasp of the physics survives
    realism, which only means anything if the two renders are otherwise the
    same shot. Camera motion is an orthogonal axis with the same share at every
    level, so the answer must not depend on the rung -- and comparing the two
    BUILT rungs is what pins that.
    """
    sc = scenarios.get(name)
    for seed in (0, 7, 4242):
        a = sc.sample(seed, TIERS["release"], "L0")
        b = sc.sample(seed, TIERS["release"], "L1")
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
        for sp in _specs(name, n=60):
            if sp.camera_moves:
                seen.setdefault(sp.camera_motion_kind, sp)
    for kind in CAMERA_MOTION_KINDS:
        assert kind in seen, "no %s clip sampled" % kind

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
    sp = next(sp for name in NAMES for sp in _specs(name)
              if not sp.camera_moves)
    poses = {sp.camera_at(f, 25)[0] for f in range(25)}
    assert len(poses) == 1, "a static camera reported %d poses" % len(poses)


def test_the_debug_override_forces_a_kind():
    """`PHYSLOC_CAMERA_MOTION` is how a motion gets looked at without hunting
    for a seed that draws it. It must never be set during a release run, so it
    is worth a test that says out loud what it does."""
    old = os.environ.get("PHYSLOC_CAMERA_MOTION")
    try:
        for kind in CAMERA_MOTION_KINDS:
            os.environ["PHYSLOC_CAMERA_MOTION"] = kind
            kinds = {scenarios.get("drop")
                     .sample(s, TIERS["debug"], CAMERA_LEVEL, variant=s)
                     .camera_motion_kind for s in range(8)}
            assert kinds == {kind}, (kind, kinds)
        os.environ["PHYSLOC_CAMERA_MOTION"] = "off"
        assert not any(scenarios.get("drop")
                       .sample(s, TIERS["debug"], CAMERA_LEVEL, variant=s)
                       .camera_moves for s in range(30))
    finally:
        os.environ.pop("PHYSLOC_CAMERA_MOTION", None)
        if old is not None:
            os.environ["PHYSLOC_CAMERA_MOTION"] = old
