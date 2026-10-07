"""Presentation videos: picking, layout, the wipe, and the picks file.

Built on tiny schema-v4 samples, so these run without a generated release.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

from physloc import taxonomy
from physloc.viz import compare, showcase
from sample_fixture import make_sample

T, SIZE = 9, 32


def _pair(root, scenario, family, seed, level="L0", condition="standard",
          bins=("strong",)):
    """A valid twin and one invalid clip per bin, as `generate` lays them out."""
    pair = "rel/%s/%s/%04d_%s" % (level, scenario, seed,
                                  condition.replace("+", "-"))
    make_sample(root, pair + "/valid", "valid", family=None, pair_uid=pair,
                frames=T, side=SIZE, level=level, scenario=scenario,
                condition=condition, seed=seed)
    for sev in bins:
        make_sample(root, pair + "/invalid_%s_%s" % (family, sev), "invalid",
                    family=family, pair_uid=pair, valid_uid=pair + "/valid",
                    frames=T, side=SIZE, level=level, scenario=scenario,
                    condition=condition, seed=seed, severity=sev)


def _frames(path):
    import imageio.v2 as imageio
    r = imageio.get_reader(path)
    out = [np.asarray(x) for x in r]
    r.close()
    return np.stack(out)


@pytest.fixture
def root(tmp_path):
    root = str(tmp_path / "rel")
    _pair(root, "drop", "support", 1, bins=taxonomy.SEVERITY_BINS)
    _pair(root, "collision", "solidity", 2)
    _pair(root, "drop", "solidity", 3, level="L1")
    _pair(root, "drop", "solidity", 4, condition="multi")
    return root


@pytest.mark.parametrize("n", list(range(1, 41)))
def test_layout_fits_the_slide(n):
    cols, rows, cell = showcase.layout(n)
    assert cols * rows >= n and cell > 0
    for x, y in showcase.grid_positions(n, cols, rows, cell):
        assert showcase.MARGIN <= x and x + cell <= showcase.W - showcase.MARGIN
        assert showcase.TOP <= y and y + cell <= showcase.H


def test_rank_prefers_strong_standard_L0_then_the_more_visible():
    base = dict(severity="strong", condition="standard", level="L0",
                area=0.01, occlusion=0.0, seed=1, dir="a")
    recs = [dict(base, dir="weak", severity="weak", area=0.5),
            dict(base, dir="multi", condition="multi", area=0.5),
            dict(base, dir="L2", level="L2", area=0.5),
            dict(base, dir="small"),
            dict(base, dir="hidden", area=0.05, occlusion=0.9),
            dict(base, dir="big", area=0.03)]
    assert [r["dir"] for r in sorted(recs, key=showcase.rank)] == [
        "big", "small", "hidden", "L2", "multi", "weak"]


def test_a_family_video_is_one_scene_of_its_scenario(root):
    recs = compare.index([root])
    for scen, fams in (("drop", ["support"]), ("collision", ["solidity"])):
        slots = showcase.pick_family_scene(recs, scenario=scen)
        assert slots[0]["label"] == "valid" and slots[0]["valid"]
        assert [s["label"] for s in slots[1:]] == fams
        assert len({os.path.dirname(s["dir"]) for s in slots}) == 1
    assert showcase.pick_family_scene(recs, scenario="toss") == []


def test_axis_videos_stay_on_one_cell(root):
    recs = compare.index([root])
    sev = showcase.pick_severity(recs)
    assert [s["label"] for s in sev] == ["valid"] + list(taxonomy.SEVERITY_BINS)
    assert len({os.path.dirname(s["dir"]) for s in sev}) == 1
    for pick in (showcase.pick_levels, showcase.pick_conditions):
        cells = {tuple(s["dir"].split("/")[-4::3]) for s in pick(recs) if s["dir"]}
        assert len({c[0] for c in cells}) == 1      # one scenario
    levels = showcase.pick_levels(recs, scenario="drop", family="solidity")
    # The only L0 drop x solidity is a multi clip; levels are standard only.
    assert [s["dir"] is not None for s in levels] == [False, True, False, False]
    conds = showcase.pick_conditions(recs, scenario="drop", family="solidity")
    assert [s["label"] for s in conds if s["dir"]] == ["multi-violation"]


def test_wipe_reveals_the_annotation_left_of_the_line():
    a = np.zeros((40, 40, 3), np.uint8)
    b = np.full((40, 40, 3), 200, np.uint8)
    assert (showcase.wiped(a, b, 0.0) == a).all()
    assert (showcase.wiped(a, b, 1.0) == b).all()
    half = showcase.wiped(a, b, 0.5)           # the line, with its glow, at 20
    assert (half[:, :14] == 200).all() and (half[:, 26:] == 0).all()
    assert (half[:, 20] == 255).all()


def test_teaser_slots_follow_the_story(root):
    slots = showcase.pick_teaser(compare.index([root]), mosaic=4)
    roles = [s["role"] for s in slots]
    built = [x for x in taxonomy.SCENARIOS if x not in taxonomy.UNBUILT]
    assert roles == (["scenario"] * (len(built) - 1) + ["hero"]
                     + ["family"] * len(taxonomy.FAMILIES)
                     + ["condition"] * len(compare.CONDITIONS)
                     + ["level"] * len(compare.LEVELS) + ["mosaic"] * 15)
    hero = slots[roles.index("hero")]
    # The family carousel opens on the clip the scenario carousel stopped on.
    assert slots[roles.index("family")]["dir"] == hero["dir"]
    # Every family has a slot; the ones not generated are placeholders.
    fams = [s for s in slots if s["role"] == "family"]
    assert {s["label"] for s in fams} == {showcase.nice(f) for f in taxonomy.FAMILIES}


def test_teaser_is_square(root, tmp_path, monkeypatch):
    monkeypatch.setattr(showcase, "TEASER_MOSAIC", 4)
    monkeypatch.setattr(showcase, "TEASER_SIZE", 256)
    got = showcase.build([root], str(tmp_path / "s"), only=["teaser"])
    frames = _frames(got["made"]["teaser"]["path"])
    assert frames.shape[1:3] == (256, 256)
    assert frames.shape[0] == got["made"]["teaser"]["frames"] > 0


def test_build_writes_slide_sized_videos_and_replays_its_picks(root, tmp_path):
    out = str(tmp_path / "showcase")
    got = showcase.build([root], out, only=["families", "severity"])
    assert set(got["made"]) == {"families/drop", "families/collision",
                                "severity/drop"}
    assert os.path.exists(os.path.join(out, "severity", "drop.mp4"))
    for made in got["made"].values():
        frames = _frames(made["path"])
        h, w = frames.shape[1:3]                 # the grid, edge to edge
        assert h % 2 == 0 and w % 2 == 0 and h <= showcase.H and w <= showcase.W
        assert frames.shape[0] == made["frames"]
    picks = os.path.join(out, "picks.json")
    with open(picks) as fh:
        first = json.load(fh)

    # Curate: empty the severity video's strong tile, and replay only that
    # video -- the other videos' picks must survive.
    first["severity/drop"][-1]["dir"] = first["severity/drop"][-1]["uid"] = None
    with open(picks, "w") as fh:
        json.dump(first, fh)
    again = showcase.build([root], out, only=["severity"], picks=picks,
                           scenario="drop")
    assert again["made"]["severity/drop"]["tiles"] == 3
    with open(picks) as fh:
        assert json.load(fh) == first


def test_clips_of_different_rates_play_at_real_speed():
    """A 25-frame 12 fps clip beside a 61-frame 30 fps one: both last about
    two seconds, so in a 30 fps video both take about 60 frames per play."""
    def clip(T, fps):
        c = showcase.Clip.__new__(showcase.Clip)
        c.T, c.fps, c.fps_out = T, float(fps), 30.0
        c.n = max(1, int(round(T * 30.0 / fps)))
        return c

    slow, fast = clip(25, 12), clip(61, 30)
    assert (slow.n, fast.n) == (62, 61)
    assert [fast.at(g) for g in (0, 30, 60, 61)] == [0, 30, 60, 0]
    # One second in, both show the frame one second into their clip.
    assert slow.at(30) == 12 and fast.at(30) == 30
    assert slow.at(slow.n) == 0


def test_carousels_start_slow_cruise_and_settle():
    step, fps, pr = 100.0, 30.0, showcase.SCROLL
    off = showcase.scroll_offsets(13, step, fps)
    v = np.diff(off)
    cruise = step / pr["seconds"] / fps
    assert off[0] == 0 and off[-1] == 12 * step
    assert (v > 0).all()                                   # never stops early
    assert v[0] == pytest.approx(cruise * pr["start"], rel=0.05)
    first_full = off[int(np.argmax(v >= 0.999 * cruise))] / step
    assert first_full == pytest.approx(pr["accel"], abs=0.1)   # the fifth video
    assert v[-1] < 0.2 * cruise                            # settles into the stop
    assert showcase.scroll_offsets(1, step, fps) == [0.0]


def test_family_reel_starts_on_the_hero_scene_and_covers_the_rest(root):
    strong = [r for r in compare.index([root]) if r["severity"] == "strong"]
    hero = next(r for r in strong if r["family"] == "support")
    reel = showcase._family_reel(strong, hero)
    assert reel[0] == ("support", hero)
    assert [f for f, _ in reel].count("solidity") == 1
    assert {f for f, _ in reel} == set(taxonomy.FAMILIES)
    assert all(r is None for f, r in reel if f not in ("support", "solidity"))


@pytest.mark.parametrize("n, shape", [(4, (2, 2)), (5, (5, 1)), (6, (3, 2)),
                                      (13, (7, 2)), (18, (6, 3)), (1, (1, 1))])
def test_tight_grid_has_no_gaps_and_stays_near_square(n, shape):
    cols, rows, cell = showcase.tight_grid(n)
    assert (cols, rows) == shape
    assert cell % 2 == 0 and cols * cell <= showcase.W and rows * cell <= showcase.H
