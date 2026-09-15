"""Dataset-structure videos: levels, variants and conditions side by side.

Built on tiny synthetic clips laid out the way a release is, so these run
without a generated review on disk.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

from physloc.annotate import layout
from physloc.viz import compare
from physloc.viz import video as vid

T, SIZE = 9, 32


def _clip(root, level, scenario, family, seed, variant=0,
          condition="standard", severity="strong", release="rel"):
    d = os.path.join(root, "clips", release, level, scenario,
                     "%04d_%s" % (seed, condition.replace("+", "-")),
                     "invalid_%s_%s" % (family, severity))
    os.makedirs(d)
    meta = {"metadata": {"label": "invalid", "scenario": scenario,
                         "family": family, "complexity": {"name": level},
                         "condition": condition, "seed": seed,
                         "variant": variant, "num_frames": T,
                         "frame_rate": 12, "release": release},
            "violation": {"intervention": {"severity_bin": severity},
                          "t_event_frame": 3,
                          "violation_windows": [[3, 6]],
                          "observable_windows": [[4, 8]]}}
    with open(os.path.join(d, layout.METADATA), "w") as fh:
        json.dump(meta, fh)
    vid.write(np.full((T, SIZE, SIZE, 3), 30 + 20 * variant, np.uint8),
              os.path.join(d, layout.VIDEO), fps=12)
    mask = np.zeros((T, SIZE, SIZE), bool)
    mask[3:7, 8:16, 8:16] = True
    np.savez_compressed(os.path.join(d, "violation_mask.npz"), mask=mask)
    np.savez_compressed(os.path.join(d, "timelines.npz"),
                        active=np.arange(T) >= 3)
    return d


def _frames(path):
    import imageio.v2 as imageio
    r = imageio.get_reader(path)
    out = [np.asarray(x) for x in r]
    r.close()
    return np.stack(out)


def test_levels_draw_every_level_and_blank_the_missing(tmp_path):
    a, b = str(tmp_path / "L0run"), str(tmp_path / "L3run")
    _clip(a, "L0", "drop", "antigravity", 777)
    _clip(b, "L3", "drop", "antigravity", 20777)
    recs = compare.index([a, b])
    tiles = compare.pick_levels(recs, "drop", "antigravity")
    assert [t[0] for t in tiles] == list(compare.LEVELS)
    assert [t[2] is not None for t in tiles] == [True, False, False, True]
    out = compare.one("levels", recs, "drop", "antigravity",
                      str(tmp_path / "levels.mp4"))
    frames = _frames(out)
    assert frames.shape[0] == T
    assert frames.shape[2] >= compare.PAD + 4 * (compare.CELL + compare.PAD)


def test_variants_take_at_most_n_distinct_in_variant_order(tmp_path):
    root = str(tmp_path / "run")
    for v in (6, 0, 3, 1, 5, 2, 4):
        _clip(root, "L0", "toss", "continuity", 777 + v, variant=v)
    tiles = compare.pick_variants(compare.index([root]), "toss", "continuity",
                                  n=5)
    assert [t[2]["variant"] for t in tiles] == [0, 1, 2, 3, 4]


def test_the_same_scene_in_two_roots_is_one_variant(tmp_path):
    """`review_L0` and `review_conditions` both hold seed 777 variant 0."""
    a, b = str(tmp_path / "L0run"), str(tmp_path / "condrun")
    _clip(a, "L0", "drop", "solidity", 777, variant=0)
    for v in range(3):
        _clip(b, "L0", "drop", "solidity", 777 + v, variant=v)
    tiles = compare.pick_variants(compare.index([a, b]), "drop", "solidity")
    assert [(t[2]["seed"], t[2]["variant"]) for t in tiles] == [
        (777, 0), (778, 1), (779, 2)]


def test_conditions_follow_the_declared_order(tmp_path):
    root = str(tmp_path / "run")
    for v, cond in enumerate(("multi", "standard", "camera+multi", "camera")):
        _clip(root, "L0", "drop", "solidity", 777 + v, variant=v, condition=cond)
    tiles = compare.pick_conditions(compare.index([root]), "drop", "solidity")
    assert [t[0] for t in tiles] == list(compare.CONDITIONS)
    assert [t[2] is not None for t in tiles] == [True, True, False, True, True]


def test_a_subset_covers_every_scenario():
    pairs = [(s, f) for s in ("drop", "toss", "pour")
             for f in ("a", "b", "c", "d", "e", "f")]
    got = compare.spread(pairs, 6)
    assert len(got) == 6
    assert {s for s, _ in got} == {"drop", "toss", "pour"}
    assert compare.spread(pairs, 0) == pairs


def test_batch_only_draws_cells_with_something_to_compare(tmp_path):
    root = str(tmp_path / "run")
    _clip(root, "L0", "drop", "solidity", 777, variant=0)
    _clip(root, "L0", "drop", "solidity", 778, variant=1, condition="camera")
    _clip(root, "L0", "toss", "continuity", 777, variant=0)   # one variant only
    got = compare.batch([root], str(tmp_path / "out"))
    assert got["made"] == {"levels": 0, "variants": 1, "conditions": 1}
    for kind in ("variants", "conditions"):
        assert os.path.exists(os.path.join(str(tmp_path / "out"), kind,
                                           "drop__solidity.mp4"))


def test_nothing_generated_is_an_error_not_an_empty_video(tmp_path):
    with pytest.raises(ValueError):
        compare.render("empty", "", [("L0", "", None)], str(tmp_path / "x.mp4"))
