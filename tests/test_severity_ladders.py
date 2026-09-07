"""A severity ladder must point one way.

`weak`, `medium` and `strong` are a claim that the same violation is being
turned up, so the knob that expresses it has to move monotonically. Which
DIRECTION is per family -- a shorter fade is a more abrupt `dissolve`, a lower
coefficient is a more slippery `friction` -- so the rule is not "increasing",
it is "does not turn around".

The bug this exists for: `global_gravity` declared
`{weak: 0.45, medium: 0.05, strong: -1.2}`. As a deviation from lawful gravity
that is 0.45, 0.05, 1.20 -- a V. Medium was a five per cent change, milder than
weak's forty-five, so it was invisible. Measured over a 166-cell L0 sweep the
family scored weak 0.000 / medium 0.000 / strong 1.000 on four of its five
scenarios: a binary family wearing three labels, with a third of its clips
depicting nothing.

Static, over the declared tables, so it costs nothing and covers every family
that declares one.
"""
import pytest

from physloc import injectors

BINS = ("weak", "medium", "strong")


def _ladders():
    """Every `*_BY_BIN` numeric table on every injector, as (family, name, values)."""
    out = []
    for family in sorted(injectors.available()):
        cls = type(injectors.get(family))
        for attr in dir(cls):
            if not attr.endswith("_BY_BIN"):
                continue
            table = getattr(cls, attr)
            if not isinstance(table, dict) or set(table) != set(BINS):
                continue
            vals = [table[b] for b in BINS]
            if all(isinstance(v, (int, float)) and not isinstance(v, bool)
                   for v in vals):
                out.append((family, attr, vals))
    return out


def test_there_are_ladders_to_check():
    """A guard on the guard: if `available()` or the naming convention changes,
    this file would silently pass by checking nothing."""
    found = _ladders()
    assert len(found) >= 15, found
    assert any(f == "global_gravity" for f, _, _ in found)


@pytest.mark.parametrize("family,attr,vals", _ladders(),
                         ids=lambda x: x if isinstance(x, str) else "")
def test_a_ladder_never_turns_around(family, attr, vals):
    """Monotone in MAGNITUDE, in whichever direction the family means.

    Magnitude rather than signed value because several families cross zero on
    purpose -- `angular_momentum` spins one way at `weak` and the other at
    `strong`, and `global_gravity` goes from slower-than-Earth to faster. What
    is not allowed is a bin that is milder than the one below it.
    """
    a = [abs(v) for v in vals]
    up = a[0] <= a[1] <= a[2]
    down = a[0] >= a[1] >= a[2]
    assert up or down, (
        "%s.%s is not monotone: %s -> magnitudes %s. `medium` must not be "
        "milder than `weak`." % (family, attr, vals, [round(x, 4) for x in a]))


@pytest.mark.parametrize("family,attr,vals", _ladders(),
                         ids=lambda x: x if isinstance(x, str) else "")
def test_a_ladder_actually_climbs(family, attr, vals):
    """Three bins that are all the same value are one bin with three names.

    Allowed only where the class says so: `fission` holds its cleave scale
    constant across bins on purpose, because what its severity varies is how
    far the halves end up, which is a different table.
    """
    a = [abs(v) for v in vals]
    if a[0] == a[1] == a[2]:
        assert (family, attr) in {("fission", "SCALE_BY_BIN")}, (
            "%s.%s is flat across all three bins -- either vary it or record "
            "here why it is deliberate" % (family, attr))


# --------------------------------------------------------------------------
# `peak_residual` -- the block a consumer trains against
# --------------------------------------------------------------------------

def _floor():
    from physloc.annotate.severity import NoiseFloor

    return NoiseFloor(mu=0.0, sigma=0.0, n_samples=25)


def test_the_peak_never_comes_from_the_bit_identical_prefix():
    """The bug that produced EVERY zero-severity clip in the dataset.

    `peak()` took `argmax` over the whole clip. Frames before `t_event` are the
    valid rollout verbatim, so a residual there is the law measuring a lawful
    scene -- and on `rolling_ramp x support` that is 8.16 radii at frame 0,
    because the block starts on a RAISED ramp and the support law reads its
    height above the datum. That frame beat every frame of the real violation,
    and the score is gated to the violation window, so the clip shipped
    `peak_severity: 0.0` while its own summary line read `sev=1.00`.

    Measured over a 166-cell L0 sweep: 41 of 495 invalid clips peaked in the
    prefix, and all 34 clips scoring zero were exactly those.
    """
    import numpy as np

    from physloc.annotate.severity import peak

    # A big lawful residual at frame 0, a real violation from frame 3.
    residual = np.array([8.16, 8.0, 7.9, 1.2, 2.0, 1.5])
    score = np.array([0.0, 0.0, 0.0, 0.4, 0.9, 0.6])
    out = peak(residual, score, _floor(), "support", t_event=3)
    assert out["frame"] == 4, out
    assert out["score"] == pytest.approx(0.9)
    assert out["value"] == pytest.approx(2.0), "value follows the chosen frame"


def test_a_flat_score_falls_back_to_the_residual_inside_the_tail():
    """A violation genuinely at the noise floor still has a best frame, and it
    is still not allowed to be a prefix one."""
    import numpy as np

    from physloc.annotate.severity import peak

    residual = np.array([9.0, 9.0, 0.5, 3.0, 1.0])
    score = np.zeros(5)
    out = peak(residual, score, _floor(), "support", t_event=2)
    assert out["frame"] == 3, out
    assert out["value"] == pytest.approx(3.0)


def test_peak_tolerates_an_event_on_the_last_frame():
    import numpy as np

    from physloc.annotate.severity import peak

    out = peak(np.arange(4.0), np.zeros(4), _floor(), "support", t_event=99)
    assert out["frame"] == 3
