"""The ETA has to be right in PROPORTION early, not just correct at the end.

An estimate that only converges once a run is nearly over is an estimate nobody
can act on -- the whole reason to print one is to decide, in the first minutes,
whether to let a job run or kill it and change the machine. These tests pin the
property that makes that possible: a ladder's cheap levels must not be
extrapolated onto its expensive ones.
"""
import io

from physloc.progress import Profile, Progress, _fmt, job_weight


class _FakeCx:
    def __init__(self, background):
        self.background = background


COMPLEXITY = {"L0": _FakeCx("solid"), "L1": _FakeCx("solid"),
              "L2": _FakeCx("hdri"), "L3": _FakeCx("hdri")}
RATES = {("release", "solid"): 695.0, ("release", "hdri"): 1821.0}


def _sink():
    return io.StringIO()


def test_weight_follows_the_background_not_the_level_name():
    """L2 costs more than L0 because of its DOME, and the weight must say so."""
    l0 = job_weight("L0", "release", RATES, COMPLEXITY, n_families=12, n_bins=3)
    l2 = job_weight("L2", "release", RATES, COMPLEXITY, n_families=12, n_bins=3)
    assert l2 > l0
    assert abs(l2 / l0 - 1821.0 / 695.0) < 1e-9
    # L1 is solid-background, so it prices exactly as L0 does -- the ladder
    # steps on the dome, not on the level number.
    assert job_weight("L1", "release", RATES, COMPLEXITY, 12, 3) == l0


def test_unmeasured_tier_falls_back_instead_of_raising():
    w = job_weight("L0", "no-such-tier", RATES, COMPLEXITY, n_families=1,
                   n_bins=1)
    assert w == 60.0 * 2


def test_eta_is_not_fooled_by_a_cheap_first_block():
    """The regression the weighting exists for.

    Ten jobs: five cheap then five that cost ten times as much. After the cheap
    block, an unweighted rate would predict the whole remainder at cheap prices
    and promise an ending it cannot make. The weighted ETA must know the
    expensive block is coming.
    """
    weights = [1.0] * 5 + [10.0] * 5
    p = Progress(10, weights=weights, stream=_sink(), use_bar=False)
    p.t0 = p.t0 - 50.0                      # the cheap block took 50 s
    for _ in range(5):
        p.update("cheap")

    # Half the JOBS are done but only 5 of 55 units of WORK.
    assert p.n == 5
    assert abs(p.done_weight - 5.0) < 1e-9

    elapsed = 50.0
    naive = elapsed / p.n * (p.total - p.n)          # 50 s: job-count rate
    weighted = p.eta()
    # 50 units left at 10 s/unit -> ~500 s. The naive estimate is 10x short,
    # which on a real ladder is the difference between "done tonight" and
    # "done in three days".
    assert 450.0 < weighted < 550.0
    assert weighted > 9 * naive


def test_eta_matches_a_known_rate():
    """With a controlled clock the ETA is exactly remaining/rate."""
    weights = [2.0] * 4
    p = Progress(4, weights=weights, stream=_sink(), use_bar=False)
    p.t0 = p.t0 - 10.0          # pretend 10 s have passed
    p.update("one")             # 2 of 8 units done in ~10 s -> 5 s/unit
    # 6 units remain at ~5 s each.
    assert 25.0 < p.eta() < 35.0


def test_retry_jobs_do_not_collapse_the_eta():
    """A retry has no declared weight; falling back to 1.0 would be a lie.

    The declared weights here are in the hundreds, so a 1.0 fallback would read
    as a job that finished instantly and drag the estimate toward zero exactly
    when the run is ADDING work.
    """
    weights = [600.0] * 4
    p = Progress(4, weights=weights, stream=_sink(), use_bar=False)
    p.bump(2)
    assert p.total == 6
    # The two extra jobs are priced at the mean of what is known, not at 1.0.
    assert abs(p.total_weight - (4 * 600.0 + 2 * 600.0)) < 1e-6
    for _ in range(6):
        p.update("job")
    assert abs(p.done_weight - 6 * 600.0) < 1e-6


def test_progress_writes_a_line_per_job_without_a_tty():
    out = _sink()
    p = Progress(3, stream=out, use_bar=False)
    p.update("drop", ok=True)
    p.update("toss", ok=False)
    text = out.getvalue()
    assert "[1/3]" in text and "[2/3]" in text
    assert "FAILED" in text and "eta" in text


def test_profile_totals_and_overlap():
    prof = Profile()
    prof.add("worker", 100.0, n=2)
    prof.add("render", 70.0, n=2)
    prof.add("annotate", 10.0, n=2)
    out = _sink()
    prof.report(out=out, workers=2)
    text = out.getvalue()
    assert "worker" in text and "render" in text
    # render is NESTED inside worker, so the report must not imply they sum.
    assert "overlap" in text
    assert "occupancy" in text


def test_profile_timer_records_the_stage():
    prof = Profile()
    with prof.timer("annotate"):
        pass
    assert "annotate" in prof.totals
    assert prof.counts["annotate"] == 1


def test_fmt_is_readable_at_every_scale():
    assert _fmt(4.2).endswith("s")
    assert _fmt(125) == "2m 05s"
    assert _fmt(3 * 3600 + 12 * 60) == "3h 12m"
    assert _fmt(4 * 86400 + 3 * 3600) == "4d 03h"
    assert _fmt(float("nan")) == "?"
