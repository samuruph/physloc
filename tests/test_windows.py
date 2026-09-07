import pytest
import numpy as np

from physloc.annotate import windows as win


def test_rasterise_to_windows_roundtrip():
    for w in ([(1, 2)], [(0, 0)], [(1, 2), (5, 7)], [(0, 3), (5, 5), (9, 12)]):
        a = win.rasterise(w, 13)
        assert win.to_windows(a) == [tuple(x) for x in w]


def test_repeated_family_produces_several_windows():
    """A super-elastic bounce fires once per contact -- the reason windows are
    a list rather than a single interval."""
    a = np.zeros(20, bool)
    for s, e in [(3, 4), (9, 10), (15, 16)]:
        a[s:e + 1] = True
    assert win.to_windows(a) == [(3, 4), (9, 10), (15, 16)]


def test_observability_can_be_interrupted():
    """Re-occlusion splits observability into several windows."""
    a = np.array([0, 0, 1, 1, 0, 0, 1, 1, 1, 0], bool)
    assert win.to_windows(a) == [(2, 3), (6, 8)]


def test_observable_frames_needs_dynamic_ids_only():
    """Including a static participant destroys the signal: pixels move from the
    ball id to the floor id and the union footprint never changes."""
    T, H, W = 3, 4, 4
    sv = np.ones((T, H, W), np.uint16)          # all floor
    si = np.ones((T, H, W), np.uint16)
    sv[1, 0, 0] = 2                             # ball visible in valid only
    assert win.observable_frames(sv, si, [2]).tolist() == [False, True, False]
    # with the floor included, the union {1,2} is identical every frame
    assert win.observable_frames(sv, si, [1, 2]).tolist() == [False, False, False]


def test_occlusion_lag_is_reported_when_the_culprit_is_hidden():
    """The three-clocks payoff: a violation fired while the actor is fully
    occluded is not observable until it fails to re-emerge."""
    T = 12
    sv = np.zeros((T, 4, 4), np.uint16)
    si = np.zeros((T, 4, 4), np.uint16)
    # Actor visible except frames 5-8, when it is behind an occluder.
    for t in range(T):
        if not (5 <= t <= 8):
            sv[t, 0, 0] = 2
            si[t, 0, 0] = 2
    # Removed from frame 6; in the invalid twin it never comes back.
    for t in range(9, T):
        si[t, 0, 0] = 0
    obs = win.observable_frames(sv, si, [2])
    assert not obs[:9].any(), "must not be observable while hidden"
    assert obs[9:].all()
    assert win.to_windows(obs) == [(9, T - 1)]


def test_severity_is_attributed_to_the_frame_it_becomes_visible():
    """A residual spike on an invisible frame must not annotate as harmless.

    `superelastic` changes a body's velocity at the contact frame while its
    pixels are still identical in both twins, so the whole magnitude lands on a
    frame that shows nothing. Carrying it to the next observable frame is what
    keeps the severity field honest about how bad the violation was.
    """
    from physloc.annotate.severity import attribute_to_evidence

    active = np.array([0, 0, 1, 1, 1, 1, 0, 0], bool)
    observable = np.array([0, 0, 0, 1, 1, 1, 1, 1], bool)
    s = np.array([0.0, 0.0, 0.9, 0.2, 0.3, 0.1, 0.0, 0.0])
    gate, out = attribute_to_evidence(s, active, observable)
    assert out[2] == 0.0, "an invisible frame must not carry severity"
    assert out[3] == pytest.approx(0.9), "the spike must surface where it can be seen"
    assert out[4] == pytest.approx(0.3)
    assert not out[~active].any(), "severity outside the window"


def test_carry_to_visible_is_the_identity_when_nothing_is_hidden():
    """Which is most cells -- and it is what keeps a shaped intervention shaped.

    A running maximum over the window would flatten `antigravity`'s ramp-up and
    release into a monotone step, losing exactly the time variation the
    severity field exists to express.
    """
    from physloc.annotate.severity import attribute_to_evidence

    active = np.array([0, 1, 1, 1, 1, 0], bool)
    s = np.array([0.0, 0.5, 1.0, 1.0, 0.5, 0.0])
    gate, out = attribute_to_evidence(s, active, np.ones(6, bool))
    assert out == pytest.approx(s * active)
    assert list(gate) == list(active)


def test_evidence_after_the_window_still_gets_annotated():
    """A violation whose consequence only shows up once the window has closed.

    `collision x superelastic` adds its energy over two frames and the
    body has not visibly moved by the time the window shuts, so gating strictly
    on `active AND observable` left the clip with an empty mask and an
    all-zero severity field -- internally consistent and useless. The severity
    spills to the first observable frame instead.
    """
    from physloc.annotate.severity import attribute_to_evidence

    active = np.array([0, 0, 0, 0, 1, 1, 0, 0], bool)
    observable = np.array([0, 0, 0, 0, 0, 0, 1, 1], bool)
    s = np.array([0.0, 0.0, 0.0, 0.0, 0.8, 0.2, 0.0, 0.0])
    gate, out = attribute_to_evidence(s, active, observable)
    assert gate[6] and out[6] == pytest.approx(0.8)
    assert not gate[:6].any() and not gate[7]


def test_a_never_observable_window_annotates_nothing():
    """Removed while fully occluded: no evidence anywhere, and the annotation
    must say so rather than invent a region."""
    from physloc.annotate.severity import attribute_to_evidence

    active = np.array([0, 1, 1, 1, 0], bool)
    gate, out = attribute_to_evidence(np.array([0, 1.0, 1.0, 1.0, 0]),
                                      active, np.zeros(5, bool))
    assert not gate.any() and not out.any()
