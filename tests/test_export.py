"""Packaging the one schema-v3 representation for local and Hub use."""
from __future__ import annotations

import importlib.util
import json
import os

import pytest

from physloc.release import export as X
from v3_fixture import make_pair


@pytest.fixture
def release(tmp_path):
    root = tmp_path / "generated"
    for index in range(12):
        make_pair(root, "test-v3/L0/drop/%04d_standard" % index, write_rgb=False)
    return str(root)


def _rows(root):
    path = os.path.join(root, "index.jsonl")
    if os.path.exists(path):
        return [json.loads(line) for line in open(path, encoding="utf-8")]
    import pyarrow.parquet as pq
    return pq.read_table(os.path.join(root, "index.parquet")).to_pylist()


def test_export_preserves_the_v3_sample_payload(release, tmp_path):
    out = str(tmp_path / "pack")
    result = X.export(release, out)
    assert result["samples"] == 24 and result["pairs"] == 12
    for name in ("README.md", "LICENSE", "loader.py", "dataset.json", "schema.json"):
        assert os.path.exists(os.path.join(out, name))
    sample = os.path.join(out, "samples", "test-v3", "L0", "drop",
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
    assert len(dataset.samples) == 24 and not hasattr(dataset, "clips")
    assert len(module.PhysLocDataset(out, unit="pair")) == 12


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
    for key in ("sample_uid", "pair_uid", "valid_sample_uid", "scenario", "family",
                "condition", "complexity", "split", "rgb_path", "data_path"):
        assert row.get(key) is not None, key
    assert not os.path.isabs(row["rgb_path"])
    assert row["rgb_path"].endswith("/rgb.mp4")
    assert "bytes" not in row


def test_dataset_metadata_is_complete(release, tmp_path):
    out = str(tmp_path / "pack")
    X.export(release, out)
    with open(os.path.join(out, "dataset.json"), encoding="utf-8") as handle:
        metadata = json.load(handle)["dataset_metadata"]
    assert metadata["number_of_samples"] == 24
    assert metadata["number_of_pairs"] == 12
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
    with pytest.raises(FileNotFoundError, match="schema-v3"):
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
