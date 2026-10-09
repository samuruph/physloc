"""Packaging the one schema-v4 representation for local and Hub use."""
from __future__ import annotations

import importlib.util
import json
import os

import pytest

from physloc.release import export as X
from sample_fixture import make_pair


@pytest.fixture
def release(tmp_path):
    root = tmp_path / "generated"
    for index in range(12):
        make_pair(root, "test/L0/drop/%04d_standard" % index, write_rgb=False)
    return str(root)


def _rows(root):
    path = os.path.join(root, "index.jsonl")
    if os.path.exists(path):
        return [json.loads(line) for line in open(path, encoding="utf-8")]
    import pyarrow.parquet as pq
    return pq.read_table(os.path.join(root, "index.parquet")).to_pylist()


def test_export_preserves_the_v4_sample_payload(release, tmp_path):
    out = str(tmp_path / "pack")
    result = X.export(release, out)
    assert result["samples"] == 24 and result["pairs"] == 12
    for name in ("README.md", "LICENSE", "loader.py", "dataset.json", "schema.json"):
        assert os.path.exists(os.path.join(out, name))
    sample = os.path.join(out, "samples", "test", "L0", "drop",
                          "0000_standard", "invalid_solidity_strong")
    assert set(os.listdir(sample)) == {"sample.json", "rgb.mp4", "data.h5"}
    assert not any(name.endswith(".npz") for _, _, files in os.walk(out) for name in files)


def test_shipped_loader_reads_the_export(release, tmp_path):
    out = str(tmp_path / "pack")
    X.export(release, out)
    spec = importlib.util.spec_from_file_location("shipped_loader", os.path.join(out, "loader.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    dataset = module.PhysLocDataset(out)
    assert len(dataset.samples) == 24
    splits = {sample.info.split for sample in dataset.samples}
    assert splits <= {"main", "held_out"}
    assert len(module.PhysLocDataset(out, unit="pair")) == 12
    # `debug` labels no sample: it is the list in splits/debug.txt, inside main.
    debug = module.PhysLocDataset(out, split="debug")
    assert len(debug) == 2 and {s.info.split for s in debug.samples} == {"main"}


def test_pairs_never_cross_splits(release, tmp_path):
    out = str(tmp_path / "pack")
    X.export(release, out)
    membership = {}
    for name, _fraction in X.SPLIT_FRACTIONS:
        with open(os.path.join(out, "splits", name + ".txt"), encoding="utf-8") as handle:
            for uid in handle.read().split():
                membership.setdefault(uid.rsplit("/", 1)[0], set()).add(name)
    assert all(len(splits) == 1 for splits in membership.values())


def test_index_uses_relative_paths_without_video_bytes(release, tmp_path):
    out = str(tmp_path / "pack")
    X.export(release, out)
    row = next(value for value in _rows(out) if value["label"] == "invalid")
    for key in ("sample_uid", "pair_uid", "valid_uid", "scenario", "family",
                "condition", "level", "severity", "split", "rgb_path", "data_path"):
        assert row.get(key) is not None, key
    assert not os.path.isabs(row["rgb_path"])
    assert row["rgb_path"].endswith("/rgb.mp4")
    assert "bytes" not in row
    assert row["n_violators"] == 1 and row["n_objects"] == 2


def test_dataset_metadata_is_complete(release, tmp_path):
    out = str(tmp_path / "pack")
    X.export(release, out)
    with open(os.path.join(out, "dataset.json"), encoding="utf-8") as handle:
        metadata = json.load(handle)["dataset_metadata"]
    assert metadata["number_of_samples"] == 24
    assert metadata["number_of_pairs"] == 12
    assert metadata["split_ratios"] == {"main": 0.75, "held_out": 0.25}
    assert metadata["split_design"]["unit"] == "pair_uid"
    assert sum(metadata["splits"].values()) == metadata["number_of_samples"]
    assert metadata["taxonomy"]["scenarios"] == ["drop"]
    assert metadata["generation_config_ids"] == ["test:debug"]
    assert metadata["coordinate_convention"].startswith("right-handed")
    analysis = metadata["difficulty_analysis"]
    assert analysis["number_of_invalid_samples"] == 12
    assert set(analysis["thresholds"]) == {
        "violation_area", "occlusion", "duration", "severity",
        "object_count", "violators", "camera_motion"}
    assert "label_setting_factors" in analysis
    accounting = metadata["energy_accounting"]
    assert accounting["world_total"]["hdf5_path"] == "/energy/scene/total"
    assert accounting["visible_scene_total"]["hdf5_path"].endswith("energy_in_frame")
    assert accounting["per_object"]["object_axis"] == "/objects/ids"
    assert {"floor", "occluder", "barrier", "shadow_caster"} <= set(
        accounting["excluded_roles"])


def test_export_rejects_v2_and_in_place_targets(tmp_path):
    old = tmp_path / "old" / "clips" / "valid"
    old.mkdir(parents=True)
    (old / "metadata.json").write_text("{}")
    with pytest.raises(FileNotFoundError, match="schema-v4"):
        X.export(str(tmp_path / "old"), str(tmp_path / "pack"))
    release = tmp_path / "generated"
    make_pair(release, write_rgb=False)
    with pytest.raises(ValueError, match="different"):
        X.export(str(release), str(release))


def test_split_assignment_is_reproducible_and_populated():
    for count in (6, 13, 40):
        values = ["dataset/drop/%04d" % index for index in range(count)]
        got = X.assign_splits(values)
        assert got == X.assign_splits(reversed(values))
        assert all(any(split == name for split in got.values())
                   for name, _fraction in X.SPLIT_FRACTIONS)


def _layout():
    """v0's shape: per scenario 10 / 5 / 3 / 2 pairs at L0..L3, conditions mixed."""
    conds = {"L0": ("standard", "standard", "standard", "camera", "distractors",
                    "multi", "camera-multi"),
             "L1": ("standard", "standard", "distractors", "camera-multi"),
             "L2": ("standard", "multi"), "L3": ("standard", "multi")}
    uids = []
    for scenario in ("drop", "toss", "pour", "collision", "shadow_track"):
        for level, n in (("L0", 10), ("L1", 5), ("L2", 3), ("L3", 2)):
            uids += ["v0/%s/%s/%04d_%s" % (level, scenario, i, conds[level][i % len(conds[level])])
                     for i in range(n)]
    return uids


def test_held_out_covers_every_level_scenario_cell():
    """The guarantee the split makes: each level's held_out holds every
    scenario, and no cell strays more than a pair from its share."""
    got = X.assign_splits(_layout())
    cells = {}
    for uid, split in got.items():
        level, scenario, _ = X.strata(uid)
        cells.setdefault((level, scenario), []).append(split)
    for key, splits in cells.items():
        held, n = splits.count("held_out"), len(splits)
        assert 1 <= held < n, key
        assert held <= max(1, X.HELD_OUT_SHARE * n + 1), key
    for level, n in (("L0", 50), ("L1", 25)):
        held = sum(s == "held_out" for u, s in got.items() if X.strata(u)[0] == level)
        assert abs(held - X.HELD_OUT_SHARE * n) <= 1, level


def test_split_depends_on_the_set_not_the_order():
    uids = _layout()
    assert X.assign_splits(uids) == X.assign_splits(list(reversed(uids)))


def test_a_frozen_assignment_survives_new_pairs():
    uids = _layout()
    first = X.assign_splits(uids)
    grown = uids + ["v0/L0/drop/%04d_standard" % i for i in range(100, 110)]
    again = X.assign_splits(grown, frozen=first)
    assert all(again[uid] == first[uid] for uid in uids)
    assert any(again[uid] == "held_out" for uid in grown[len(uids):])


def test_debug_is_one_main_pair_per_scenario():
    got = X.assign_splits(_layout())
    debug = X.debug_subset(got)
    assert all(got[uid] == "main" for uid in debug)
    assert sorted(X.strata(uid)[1] for uid in debug) == sorted(
        {X.strata(uid)[1] for uid in got})
