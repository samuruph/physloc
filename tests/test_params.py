"""The generation knobs: one source, validated, recorded, and across the seam.

Shares, counts and bands were module constants spread over three files. They
are now `configs/common.yaml`, and the value of that is entirely in the
guarantees around it -- a knob you cannot validate, cannot reproduce and cannot
get into the container is worse than the constant it replaced.
"""
import json
import os

import pytest

from physloc import params

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMMON = os.path.join(REPO, "configs", "common.yaml")


def test_common_yaml_matches_the_shipped_defaults():
    """The file DOCUMENTS the code rather than diverging from it.

    If they disagree, one of them is lying to whoever reads it, and the file is
    the one people read.
    """
    yaml = pytest.importorskip("yaml")
    with open(COMMON) as fh:
        on_disk = yaml.safe_load(fh)
    assert params.resolved(on_disk) == params.DEFAULTS, (
        "configs/common.yaml has drifted from params.DEFAULTS")


def test_an_unknown_key_is_an_error():
    """Not ignored. A typo in a config is the one failure mode that costs a
    whole run: it looks like it worked and quietly used the default."""
    with pytest.raises(KeyError):
        params.resolved({"objects": {"extra_maxx": 4}})
    with pytest.raises(KeyError):
        params.resolved({"obejcts": {}})
    with pytest.raises(TypeError):
        params.resolved({"objects": 4})


def test_a_partial_override_keeps_everything_else():
    got = params.resolved({"objects": {"extra_max": 5}})
    assert got["objects"]["extra_max"] == 5
    assert got["objects"]["extra_min"] == params.DEFAULTS["objects"]["extra_min"]
    assert got["ladder"] == params.DEFAULTS["ladder"]


def test_layers_stack_in_order():
    """Shipped defaults, then `common.yaml`, then a run's own block."""
    mid = params.resolved({"objects": {"extra_min": 4, "extra_max": 6}})
    top = params.resolved({"objects": {"extra_max": 9}}, base=mid)
    assert (top["objects"]["extra_min"], top["objects"]["extra_max"]) == (4, 9)


def test_it_crosses_the_seam_as_json(tmp_path):
    """The container has Kubric's pinned set and no PyYAML -- `config.py` says
    so. So the knobs travel as JSON, which is stdlib on both sides."""
    values = params.resolved({"objects": {"extra_max": 7}})
    path = params.write(values, str(tmp_path))
    assert os.path.basename(path) == params.FILENAME
    with open(path) as fh:
        json.load(fh)                       # plain JSON, no custom types
    assert params.read(path) == values


def test_params_is_importable_without_yaml():
    """It is imported inside the container. A stray `import yaml` at module
    level would break every render, and only in the container."""
    src = open(os.path.join(REPO, "physloc", "params.py")).read()
    assert "import yaml" not in src, "params.py must not need PyYAML"


def test_apply_actually_moves_the_behaviour():
    """A knob that does not reach the code is a comment.

    Every value in `DEFAULTS` should have a home; this checks the ones whose
    effect is visible without rendering, including one that flows all the way
    through to how many variants a rung is allocated.
    """
    from physloc.scenarios import _common as C
    from physloc.scenarios import base as B

    before = params.CURRENT
    try:
        params.apply({"objects": {"extra_min": 5, "extra_max": 7,
                                  "distractor_moving": 0.9},
                      "ladder": {"shares": {"L1": 0.8}}})
        assert B.EXTRA_OBJECTS == (5, 7)
        assert B.MULTI_ACTORS == (5, 7)
        assert C.DISTRACTOR_MOVING == pytest.approx(0.9)
        assert B.COMPLEXITY["L1"].share == pytest.approx(0.8)
        assert B.variants_at("L1", 10) == 8, "the share did not reach the split"
    finally:
        params.apply(before)
    assert B.EXTRA_OBJECTS == (params.DEFAULTS["objects"]["extra_min"],
                               params.DEFAULTS["objects"]["extra_max"])
    assert B.variants_at("L1", 10) == 5


def test_the_defaults_are_internally_consistent():
    d = params.DEFAULTS
    assert d["objects"]["extra_min"] >= 3, (
        "a multi scene needs three bodies to pose its question")
    assert d["objects"]["extra_min"] < d["objects"]["extra_max"]
    assert d["objects"]["multi_culprits_min"] >= 2, (
        "one culprit is what `standard` already is")
    assert 0.0 <= d["objects"]["distractor_moving"] <= 1.0
    assert len(d["camera"]["kinds"]) == len(d["camera"]["weights"])
    assert sum(d["camera"]["weights"]) == pytest.approx(1.0)
    assert set(d["ladder"]["shares"]) == {"L0", "L1", "L2", "L3"}
    assert d["ladder"]["shares"]["L0"] == max(d["ladder"]["shares"].values()), (
        "the baseline must be the largest stratum")
    assert d["conditions"]["cycle"].count("standard") * 2 > len(
        d["conditions"]["cycle"]), "standard must stay the majority"
