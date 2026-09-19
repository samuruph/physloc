"""The release report: the numbers behind the figures, and the files it writes.

Light on purpose. The figures themselves are checked by LOOKING at them --
non-negotiable 5, and the reason this project has an overlay video rather than
a metrics table. What a test can hold is that the counts are right, that every
factor reaches the histogram panel, and that the command writes what it says it
writes.
"""
from physloc.annotate import difficulty as D
from tests.test_difficulty import _meta


def test_summarise_counts_what_the_figures_draw(tmp_path):
    from physloc.release import stats

    # One clip in each zone, placed from the cuts so a recalibration does not
    # silently move a clip across one.
    area = D.BY_NAME["violation_area"]
    metas = []
    for foot, cond in ((area.easy * 2.0, "standard"),
                       ((area.easy + area.moderate) / 2.0, "camera"),
                       (area.moderate / 2.0, "multi")):
        m = _meta(condition=cond, scenario="drop", family="antigravity",
                  complexity={"name": "L0"})
        m["violation"]["difficulty_inputs"]["violation_area"] = foot
        m["violation"]["intervention"] = {"severity_bin": "strong"}
        m["difficulty"] = D.assess(m)
        metas.append(m)
    valid = _meta(condition="standard", scenario="drop",
                  complexity={"name": "L0"})
    valid["violation"] = None
    metas.append(valid)

    s = stats.summarise(metas)
    assert s["samples"] == 4 and s["invalid"] == 3 and s["valid"] == 1
    assert s["difficulty"] == {"easy": 1, "moderate": 1, "hard": 1}
    assert s["conditions"] == {"standard": 2, "camera": 1, "multi": 1}
    assert s["difficulty_by_level"]["L0"]["hard"] == 1
    # Every factor is represented, so the histogram panel cannot silently lose
    # one when a factor is added to the table.
    assert set(s["factor_values"]) == {f.name for f in D.FACTORS}
    assert set(s["thresholds"]) == {f.name for f in D.FACTORS}


def test_report_writes_the_figures_and_the_json(tmp_path):
    import json
    import os

    from physloc.release import stats

    metas = []
    for foot in (0.3, 0.02, 0.001):
        m = _meta(condition="standard", scenario="drop", family="antigravity",
                  complexity={"name": "L0"})
        m["violation"]["difficulty_inputs"]["violation_area"] = foot
        m["violation"]["intervention"] = {"severity_bin": "strong"}
        m["difficulty"] = D.assess(m)
        metas.append(m)
    got = stats.report(str(tmp_path), metas=metas)
    out = got["outdir"]
    for name in [n for n, _ in stats.FIGURES] + ["stats.json"]:
        p = os.path.join(out, name)
        assert os.path.exists(p) and os.path.getsize(p) > 0, name
    with open(os.path.join(out, "stats.json")) as fh:
        assert json.load(fh)["invalid"] == 3


def test_stats_reconstructs_early_v3_factor_blocks(tmp_path):
    """Early v3 JSON retained all measurements but only the label string."""
    from physloc.release import stats
    from sample_fixture import make_pair

    make_pair(tmp_path)
    loaded = stats.load(str(tmp_path))
    loaded = [row for row in loaded if row["violation"]]
    assert loaded[0]["difficulty"] is not None
    assert loaded[0]["difficulty"]["level"] in D.LEVELS
    assert set(loaded[0]["difficulty"]["factors"]) == {factor.name for factor in D.FACTORS}
