"""The public schema-v3 Sample and PhysLocDataset API."""
from __future__ import annotations

import ast
import importlib.util
import pickle

import numpy as np
import pytest

from physloc import loader as L
from v3_fixture import make_pair, make_sample


def test_component_aware_visible_violation():
    seg = np.array([[[1, 2], [1, 0]]], np.uint16)
    violation = np.array([[[2, 2], [0, 0]]], np.uint16)
    component = np.array([[[2, 1], [0, 0]]], np.uint8)
    shadow_source = np.array([[[2, 0], [0, 0]]], np.uint16)
    assert L.visible_violation(violation, seg, component, shadow_source).tolist() == [
        [[True, True], [False, False]]]


def test_sample_timeline_unions_only_violators():
    table = {"ids": np.array([1, 2]), "is_violator": np.array([False, True]),
             "severity": np.array([[1, 1, 1], [0, .5, .9]], np.float32)}
    table.update({key: np.zeros((2, 3), bool) for key in L.CLOCKS})
    table["active"][1, 1:] = True
    timeline = L.sample_timeline(table, 3)
    assert timeline["active"].tolist() == [False, True, True]
    assert timeline["severity"].tolist() == pytest.approx([0, .5, .9])


def test_loader_is_standalone():
    with open(L.__file__, encoding="utf-8") as handle:
        tree = ast.parse(handle.read())
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert node.level == 0 and not (node.module or "").startswith("physloc")
    spec = importlib.util.spec_from_file_location("shipped_physloc_loader", L.__file__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.SCHEMA_VERSION == 3
    assert not hasattr(module.PhysLocDataset, "clips")


@pytest.fixture
def dataset_root(tmp_path):
    make_pair(tmp_path)
    return str(tmp_path)


def test_dataset_exposes_samples_and_explicit_pairs(dataset_root):
    dataset = L.PhysLocDataset(dataset_root)
    assert len(dataset.samples) == len(dataset) == 2
    assert not hasattr(dataset, "clips")
    invalid = next(sample for sample in dataset.samples if not sample.is_valid)
    assert dataset.get(invalid.uid) is invalid
    assert invalid.twin is not None and invalid.twin.is_valid
    assert invalid.video_path.endswith("rgb.mp4")
    assert invalid.decode_rgb().shape == (5, 16, 16, 3)
    pair = L.PhysLocDataset(dataset_root, unit="pair")[0]
    assert pair["valid"].is_valid and len(pair["invalid"]) == 1


def test_objects_are_aligned_and_subjects_are_default(dataset_root):
    sample = L.PhysLocDataset(dataset_root, label="invalid")[0]
    assert sample.object_ids.tolist() == [1, 2]
    assert [obj["id"] for obj in sample.objects()] == [2]
    assert [obj["id"] for obj in sample.objects("support")] == [1]
    assert [obj["id"] for obj in sample.objects("violators")] == [2]
    assert sample.object(2)["temporal"]["positions"].shape == (5, 3)
    assert sample.object(2)["violation"]["severity"].shape == (5,)
    assert sample.subject_mask.shape == sample.segmentations.shape
    assert not (sample.subject_mask & sample.support_mask).any()


def test_per_object_spatiotemporal_violation(dataset_root):
    sample = L.PhysLocDataset(dataset_root, label="invalid")[0]
    assert sample.violation_arrays["active"].shape == (2, 5)
    assert not sample.violation_arrays["active"][0].any()
    assert sample.violation_arrays["active"][1, 1:].all()
    assert set(np.unique(sample.violation)) == {0, 2}
    assert sample.timeline["active"].tolist() == [False, True, True, True, True]
    assert sample.violation_summary["violators"][0]["instance_id"] == 2


def test_shadow_is_an_actor_component_not_an_object(tmp_path):
    pair = "test-v3/L0/shadow_track/0007_standard"
    make_sample(tmp_path, pair + "/valid", "valid", pair_uid=pair,
                valid_uid=pair + "/valid", component=2)
    make_sample(tmp_path, pair + "/invalid_shadow_strong", "invalid",
                family="shadow", pair_uid=pair, valid_uid=pair + "/valid", component=2)
    sample = L.PhysLocDataset(str(tmp_path), label="invalid")[0]
    assert all(obj["role"] not in {"shadow", "shadow_caster"}
               for obj in sample.object_definitions)
    assert set(np.unique(sample.violation_component)) == {0, 2}
    assert sample.visible_violation.any()
    assert sample.objects("violators")[0]["id"] == 2

    # The lawful (green) reference is the cast-shadow footprint from the
    # isolation pass, never the visible actor segmentation.
    ref = sample.reference_mask
    assert ref[:, 0, 0].all()  # fixture's lawful shadow is the floor region
    assert not ref[:, 4:12, 4:12].any()  # actor pixels must stay unlabeled


def test_hdf5_handle_is_not_pickled(dataset_root):
    sample = L.PhysLocDataset(dataset_root)[0]
    _ = sample.segmentations
    assert sample._h5 is not None
    restored = pickle.loads(pickle.dumps(sample))
    assert restored._h5 is None
    np.testing.assert_array_equal(restored.segmentations, sample.segmentations)


def test_selected_object_fields_are_padded(tmp_path):
    make_sample(tmp_path, "a/pair/valid", "valid", pair_uid="a/pair",
                valid_uid="a/pair/valid", n_objects=2)
    make_sample(tmp_path, "b/pair/valid", "valid", pair_uid="b/pair",
                valid_uid="b/pair/valid", n_objects=3)
    dataset = L.PhysLocDataset(str(tmp_path),
        fields=("annotations.objects.ids", "annotations.objects.energy"))
    batch = L.collate(dataset.samples)
    assert batch["object_ids"].shape == (2, 3)
    assert batch["object_valid"].tolist() == [[True, True, False], [True, True, True]]


def test_missing_old_layout_is_not_interpreted(tmp_path):
    old = tmp_path / "clips" / "old" / "valid"
    old.mkdir(parents=True)
    (old / "metadata.json").write_text("{}")
    with pytest.raises(FileNotFoundError, match="schema-v3"):
        L.PhysLocDataset(str(tmp_path))
