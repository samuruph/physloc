"""The release report: the numbers behind the figures, and the files it writes.

Light on purpose. The figures themselves are checked by LOOKING at them --
non-negotiable 5, and the reason this project has an overlay video rather than
a metrics table. What a test can hold is that the counts are right, that every
factor reaches the histogram panel, and that the command writes what it says it
writes.
"""
from physloc.annotate import difficulty as D
from tests.test_difficulty import _meta


def _release(tmp_path, metas):
    import json
    import os

    for i, m in enumerate(metas):
        cdir = tmp_path / "clips" / "rel" / "L0" / "drop" / ("%04d" % i) / "x"
        os.makedirs(cdir, exist_ok=True)
        with open(cdir / "meta.json", "w") as fh:
            json.dump(m, fh)
    return str(tmp_path)


def test_summarise_counts_what_the_figures_draw(tmp_path):
    from physloc.release import stats

    metas = []
    for foot, cond in ((0.3, "standard"), (0.02, "camera"), (0.001, "multi")):
        m = _meta(condition=cond, scenario="drop", family="antigravity",
                  complexity={"name": "L0"})
        m["violation"]["difficulty_inputs"]["footprint"] = foot
        m["violation"]["intervention"] = {"severity_bin": "strong"}
        m["difficulty"] = D.assess(m)
        metas.append(m)
    valid = _meta(condition="standard", scenario="drop",
                  complexity={"name": "L0"})
    valid["violation"] = None
    metas.append(valid)

    s = stats.summarise(metas)
    assert s["clips"] == 4 and s["invalid"] == 3 and s["valid"] == 1
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
        m["violation"]["difficulty_inputs"]["footprint"] = foot
        m["violation"]["intervention"] = {"severity_bin": "strong"}
        m["difficulty"] = D.assess(m)
        metas.append(m)
    root = _release(tmp_path, metas)

    got = stats.report(root)
    out = got["outdir"]
    for name in ("difficulty.png", "difficulty_factors.png", "composition.png",
                 "severity.png", "coverage.png", "stats.json"):
        p = os.path.join(out, name)
        assert os.path.exists(p) and os.path.getsize(p) > 0, name
    with open(os.path.join(out, "stats.json")) as fh:
        assert json.load(fh)["invalid"] == 3


def test_stats_backfills_a_release_that_predates_the_label(tmp_path):
    """Several runs on disk were generated before `difficulty` existed, and
    they are the corpus its thresholds were fitted to. The report has to be
    able to read them."""
    from physloc.release import stats

    m = _meta(condition="standard", scenario="drop", family="antigravity",
              complexity={"name": "L0"})
    m["violation"]["intervention"] = {"severity_bin": "strong"}
    m.pop("difficulty", None)
    root = _release(tmp_path, [m])
    loaded = stats.load(root)
    assert loaded[0]["difficulty"] is not None
    assert loaded[0]["difficulty"]["level"] in D.LEVELS
