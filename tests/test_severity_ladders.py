"""A severity ladder must point one way.

`weak`, `medium` and `strong` are a claim that the same violation is being
turned up, so the knob that expresses it has to move monotonically. Which
DIRECTION is per family -- a shorter fade is a more abrupt `dissolve`, a lower
coefficient is a more slippery `friction` -- so the rule is not "increasing",
it is "does not turn around".

The bug this exists for: `global_gravity` declared
`{weak: 0.45, medium: 0.05, strong: -1.2}` and scored weak 0.000 / medium 0.000
/ strong 1.000 over a 166-cell L0 sweep -- a binary family wearing three
labels, with a third of its clips depicting nothing.

**Distance from LAWFUL, not distance from zero**, and getting that wrong is how
this file let the same family through a second time. A gravity scale is a
ratio: `alpha = 1` is Earth and lawful, `alpha = 0` is weightlessness and about
as wrong as it gets. Comparing `abs(alpha)` therefore measures nothing the
family is scored on -- `_GravityScale.magnitude` is `abs(1 - alpha)` -- and it
passed `{0.25, 0.55, -1.2}`, whose deviations are **0.75, 0.45, 2.20**: a V,
with medium a milder violation than weak. That reached the review sweep as
"weak and medium are invisible on pour", and the shipped `meta.json` said so
outright, reporting magnitudes of 0.75 / 0.45 / 2.2 in that order.

So `LAWFUL` records, per table, the value that means "nothing is wrong".
Default 0, because most knobs are already deviations; 1.0 for the ratio knobs,
where doubling and halving are equally far from lawful and neither is near
zero. `tests/test_all_cells.py` carries the same table for the same reason.

Static, over the declared tables, so it costs nothing and covers every family
that declares one.
"""
import pytest

from physloc import injectors

BINS = ("weak", "medium", "strong")

#: (family, table) -> the value of that table meaning "lawful". See the module
#: docstring. Anything unlisted is a deviation already, so its lawful value is 0.
LAWFUL = {
    ("antigravity", "ALPHA_BY_BIN"): 1.0,
    ("global_gravity", "ALPHA_BY_BIN"): 1.0,
    ("shadow_inverted", "FRACTION_BY_BIN"): 1.0,
    ("superelastic", "GAIN_BY_BIN"): 1.0,
    ("newton2_mass", "RATIO_BY_BIN"): 1.0,
    ("newton3_reaction", "RATIO_BY_BIN"): 1.0,
    ("solidity", "RETENTION_BY_BIN"): 1.0,
    ("friction", "RATE_BY_BIN"): 1.0,
    ("friction", "TRAVEL_BY_BIN"): 1.0,
}


def _distance(family, attr, vals):
    """How far each bin sits from the value that means nothing is wrong."""
    lawful = LAWFUL.get((family, attr), 0.0)
    return [abs(v - lawful) for v in vals]


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
    a = _distance(family, attr, vals)
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
    a = _distance(family, attr, vals)
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
