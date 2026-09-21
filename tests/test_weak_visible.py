"""The weakest bin is still a violation: small, but seen and scored.

A weak clip fails in one of two ways, and they need different fixes. Too faint
-- the eye cannot find it -- calls for a stronger ladder. Unscored -- plainly
visible, severity zero -- calls for a residual law that sees what the picture
shows. The first audit of the review sweep found 65 weak clips failing, every
one of them the second kind; that is only visible once the audit counts the
pixels that change rather than the single pixel that changes most.
"""
from physloc.annotate import audit


def _row(severity, share, frames):
    return {"peak_severity": severity, "visible_share": share,
            "visible_frames": frames, "severity_bin": "weak"}


def test_a_seen_and_scored_weak_clip_passes():
    assert audit.weak_failure(_row(0.10, 0.01, 12)) == ""


def test_one_flickering_pixel_is_not_a_visible_violation():
    """What `evidence` alone cannot tell apart: a peak difference with almost
    no pixels behind it."""
    tiny = audit.MIN_VISIBLE_SHARE / 10.0
    assert audit.weak_failure(_row(0.50, tiny, 30)) == "too faint"


def test_a_violation_seen_on_a_single_frame_is_too_faint():
    assert audit.weak_failure(_row(0.50, 0.05, 1)) == "too faint"


def test_a_visible_clip_that_scores_zero_is_unscored():
    assert audit.weak_failure(_row(0.0, 0.05, 25)) == "unscored"
    assert audit.weak_failure(_row(audit.MIN_WEAK_SEVERITY - 1e-3, 0.05, 25)) \
        == "unscored"


def _bins(pair, family, weak, medium, strong):
    return [{"pair_uid": pair, "family": family, "scenario": "drop",
             "severity_bin": b, "peak_severity": s}
            for b, s in (("weak", weak), ("medium", medium), ("strong", strong))]


def test_a_climbing_ladder_passes():
    assert audit.ladder_failures(_bins("p", "f", 0.1, 0.4, 0.9)) == []


def test_a_bin_milder_than_the_one_below_is_out_of_order():
    out = audit.ladder_failures(_bins("p", "f", 0.5, 0.3, 0.9))
    assert [r["why"] for r in out] == ["out of order"]


def test_three_saturated_bins_are_one_bin():
    out = audit.ladder_failures(_bins("p", "f", 1.0, 1.0, 1.0))
    assert [r["why"] for r in out] == ["all saturated"]


def test_a_scene_missing_a_bin_is_not_judged():
    rows = [r for r in _bins("p", "f", 0.5, 0.3, 0.9) if r["severity_bin"] != "medium"]
    assert audit.ladder_failures(rows) == []
