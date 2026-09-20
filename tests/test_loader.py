"""The public schema-v4 Sample and PhysLocDataset API."""
from __future__ import annotations

import ast
import importlib.util
import pickle

import numpy as np
import pytest

from physloc import loader as L
from sample_fixture import make_pair, make_sample


def test_component_aware_visible_violation():
    seg = np.array([[[1, 2], [1, 0]]], np.uint16)
    violation = np.array([[[2, 2], [0, 0]]], np.uint16)
    component = np.array([[[2, 1], [0, 0]]], np.uint8)
    shadow_source = np.array([[[2, 0], [0, 0]]], np.uint16)
    assert L.visible_violation(violation, seg, component, shadow_source).tolist() == [
        [[True, True], [False, False]]]


def test_sample_timeline_unions_only_violators():
    clocks = {key: np.zeros((2, 3), bool) for key in L.CLOCKS}
    clocks["active"][1, 1:] = True
    severity = np.array([[1, 1, 1], [0, .5, .9]], np.float32)
    timeline = L.sample_timeline(clocks, severity, np.array([False, True]), 3)
    assert timeline["active"].tolist() == [False, True, True]
    assert timeline["severity"].tolist() == pytest.approx([0, .5, .9])


def test_interval_helpers_round_trip():
    mask = np.array([0, 1, 1, 0, 1, 0, 0, 1], bool)
    assert L.mask_to_intervals(mask) == [[1, 2], [4, 4], [7, 7]]
    assert L.intervals_to_mask([[1, 2], [4, 4], [7, 7]], 8).tolist() == mask.tolist()
    assert L.union_intervals([[[0, 1]], [[1, 3]], [[6, 6]]], 8) == [[0, 3], [6, 6]]


def test_loader_is_standalone():
    with open(L.__file__, encoding="utf-8") as handle:
        tree = ast.parse(handle.read())
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert node.level == 0 and not (node.module or "").startswith("physloc")
    spec = importlib.util.spec_from_file_location("shipped_physloc_loader", L.__file__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.SCHEMA_VERSION == 4


@pytest.fixture
def dataset_root(tmp_path):
    make_pair(tmp_path)
    return str(tmp_path)


def test_dataset_exposes_samples_and_explicit_pairs(dataset_root):
    dataset = L.PhysLocDataset(dataset_root)
    assert len(dataset.samples) == len(dataset) == 2
    invalid = next(sample for sample in dataset.samples if not sample.info.is_valid)
    assert dataset.get(invalid.uid) is invalid
    assert invalid.twin is not None and invalid.twin.info.is_valid
    assert invalid.video.path.endswith("rgb.mp4")
    assert invalid.video.rgb.shape == (5, 16, 16, 3)
    pair = L.PhysLocDataset(dataset_root, unit="pair")[0]
    assert pair["valid"].info.is_valid and len(pair["invalid"]) == 1


def test_namespaces_mirror_the_document(dataset_root):
    sample = L.PhysLocDataset(dataset_root, label="invalid")[0]
    assert sample.info.label == "invalid" and sample.info.tier == "debug"
    assert sample.video.num_frames == 5 and sample.video.duration == 1.0
    assert sample.scene.scenario == "drop" and sample.scene.level == "L0"
    assert sample.scene.camera.positions.shape == (5, 3)
    assert sample.scene.camera["K"].shape == (3, 3)
    assert "segmentation" in sample.observations
    assert sample.events.collisions["instances"].tolist() == [[1, 2]]
    # Attribute and key access are the same thing.
    assert sample.violation["t_event"] == sample.violation.t_event == 1


def test_objects_are_aligned_and_selectable(dataset_root):
    sample = L.PhysLocDataset(dataset_root, label="invalid")[0]
    objects = sample.objects
    assert objects.ids.tolist() == [1, 2] and len(objects) == 2
    assert objects.select().tolist() == [2]
    assert objects.select("support").tolist() == [1]
    assert objects.select("violators").tolist() == [2]
    assert objects.positions.shape == (2, 5, 3)
    record = objects.record(2)
    assert record["arrays"]["positions"].shape == (5, 3)
    assert record["violation"]["severity"].shape == (5,)
    assert record["violation"]["record"]["id"] == 2
    subjects, support = objects.spatial_mask("subjects"), objects.spatial_mask("support")
    assert subjects.shape == sample.observations.segmentation.shape
    assert not (subjects & support).any()


def test_clip_level_times_and_clocks_are_derived(dataset_root):
    v = L.PhysLocDataset(dataset_root, label="invalid")[0].violation
    assert v.present and v.severity_bin == "strong"
    assert v.ids.tolist() == [2]
    assert v.active.shape == (2, 5)
    assert not v.active[0].any() and v.active[1, 1:].all()
    assert v.intervening[1].tolist() == [False, True, False, False, False]
    assert v.windows["active"] == [[1, 4]]
    assert (v.t_event, v.t_observable, v.t_end, v.observability_lag) == (1, 1, 4, 0)
    assert set(np.unique(v.object_id)) == {0, 2}
    assert v.timeline["active"].tolist() == [False, True, True, True, True]
    assert v.violators[0]["magnitude"] == 1.0     # filled from the intervention


def test_affected_and_causal_relations_come_from_violator_records(tmp_path):
    # Three objects: floor 1, bystander 2, violator 3 -- which disturbs 2.
    make_sample(tmp_path, "t/p/invalid_x", n_objects=3, affected=(2,))
    v = L.Sample(str(tmp_path / "samples" / "t" / "p" / "invalid_x")).violation
    assert v.affected[1, 1:].all() and not v.affected[[0, 2]].any()
    assert v.causal_relations == [{"source_object_id": 3, "target_object_id": 2,
                                   "relation_type": "physical_consequence",
                                   "intervals": [[1, 4]]}]


def test_valid_sample_has_an_empty_violation(dataset_root):
    valid = next(s for s in L.PhysLocDataset(dataset_root).samples if s.info.is_valid)
    v = valid.violation
    assert not v.present and v.ids.size == 0 and v.t_event is None
    assert not v.mask.any() and v.active.shape == (2, 5) and not v.active.any()
    assert not valid._h5_has("/violation")


def test_shadow_is_an_actor_component_not_an_object(tmp_path):
    pair = "test/L0/shadow_track/0007_standard"
    make_sample(tmp_path, pair + "/valid", "valid", pair_uid=pair,
                valid_uid=pair + "/valid", component=2)
    make_sample(tmp_path, pair + "/invalid_shadow_strong", "invalid",
                family="shadow", pair_uid=pair, valid_uid=pair + "/valid", component=2)
    sample = L.PhysLocDataset(str(tmp_path), label="invalid")[0]
    # A label-filtered dataset intentionally contains no valid twin; attach it
    # here because the reference-mask assertion exercises the pair API.
    sample._twin = L.Sample.from_dir(str(tmp_path / "samples" / pair / "valid"))
    assert all(role not in {"shadow", "shadow_caster"} for role in sample.objects.roles)
    v = sample.violation
    assert v.component == "shadow" and v.shadow["caster_object_id"] == 2
    assert set(np.unique(v.component_map)) == {0, 2}
    assert v.visible.any()
    assert sample.objects.select("violators").tolist() == [2]
    # The shadow severity map cannot be painted from the body, so it is stored.
    assert sample._h5_has("/violation/maps/severity")

    # The lawful (green) reference is the cast-shadow footprint from the
    # isolation pass, never the visible actor segmentation.
    ref = v.reference_mask
    assert ref[:, 0, 0].all()  # fixture's lawful shadow is the floor region
    assert not ref[:, 4:12, 4:12].any()  # actor pixels must stay unlabeled


def test_body_severity_map_is_painted_not_stored(dataset_root):
    sample = L.PhysLocDataset(dataset_root, label="invalid")[0]
    assert not sample._h5_has("/violation/maps/severity")
    severity_map = sample.violation.severity_map
    assert severity_map[sample.violation.mask].min() == pytest.approx(0.8)
    assert not severity_map[~sample.violation.mask].any()


def test_shadow_reference_mask_is_receiver_only():
    strength = np.array([[[0.5, 0.5], [0.5, 0.0]]], np.float32)
    source = np.array([[[2, 2], [0, 0]]], np.uint16)
    receiver_seg = np.array([[[2, 1], [1, 1]]], np.uint16)
    lawful = L.shadow_receiver_mask(strength, source, receiver_seg, [2], guard_px=0)
    assert lawful.tolist() == [[[False, True], [False, False]]]


def test_shadow_receiver_mask_removes_the_two_pixel_caster_halo():
    strength = np.ones((1, 9, 9), np.float32)
    source = np.full((1, 9, 9), 2, np.uint16)
    seg = np.zeros((1, 9, 9), np.uint16)
    seg[:, 4, 4] = 2
    mask = L.shadow_receiver_mask(strength, source, seg, [2])
    assert not mask[:, 2:7, 2:7].any()
    assert mask[:, 1, 1].all()


def test_hdf5_handle_is_not_pickled(dataset_root):
    sample = L.PhysLocDataset(dataset_root)[0]
    _ = sample.observations.segmentation
    assert sample._h5 is not None
    restored = pickle.loads(pickle.dumps(sample))
    assert restored._h5 is None
    np.testing.assert_array_equal(restored.observations.segmentation,
                                  sample.observations.segmentation)


def test_to_dict_is_nested_like_the_sample(dataset_root):
    sample = L.PhysLocDataset(dataset_root, label="invalid")[0]
    item = sample.to_dict(["violation.mask", "violation", "objects.positions", "scene"])
    assert set(item) == {"info", "violation", "objects", "scene"}
    assert item["info"]["uid"] == sample.uid
    assert item["violation"]["mask"].shape == (5, 16, 16)      # merged, not replaced
    assert item["violation"]["severity_bin"] == "strong"
    assert "object_id" not in item["violation"]                  # arrays only by name
    assert item["objects"]["positions"].shape == (2, 5, 3)
    assert item["scene"]["camera"]["motion"] == "static"


def test_unknown_field_is_rejected(dataset_root):
    with pytest.raises(KeyError, match="unknown field"):
        L.PhysLocDataset(dataset_root, fields=("annotations.maps.violation_object_id",))


def test_collate_pads_per_object_fields(tmp_path):
    make_sample(tmp_path, "a/pair/valid", "valid", pair_uid="a/pair",
                valid_uid="a/pair/valid", n_objects=2)
    make_sample(tmp_path, "b/pair/valid", "valid", pair_uid="b/pair",
                valid_uid="b/pair/valid", n_objects=3)
    dataset = L.PhysLocDataset(str(tmp_path), fields=(
        "objects.ids", "violation.severity", "violation.mask"))
    batch = L.collate(dataset.samples)
    assert batch["objects"]["ids"].shape == (2, 3)
    assert batch["objects"]["valid"].tolist() == [[True, True, False], [True, True, True]]
    assert batch["violation"]["severity"].shape == (2, 3, 5)
    assert np.isnan(batch["violation"]["severity"][0, 2]).all()
    assert batch["violation"]["mask"].shape == (2, 5, 16, 16)
    assert batch["info"]["uid"] == ["a/pair/valid", "b/pair/valid"]


def test_every_field_is_discoverable_without_reading_it(dataset_root):
    """What a debugger shows must be what the block holds.

    `objects.positions` arrives through `__getattr__` from HDF5, so `dir()` --
    and every IDE variable pane -- listed the handful of JSON-backed fields and
    none of the arrays, which reads as "this sample has no trajectories".
    """
    sample = L.PhysLocDataset(dataset_root, label="invalid")[0]
    for namespace in (sample.objects, sample.observations, sample.violation,
                      sample.info, sample.scene):
        listed = set(dir(namespace))
        assert set(namespace.keys()) <= listed, namespace
    assert "positions" in dir(sample.objects)
    assert "segmentation" in dir(sample.observations)

    # describe() answers the same question in one call, and reads no array.
    text = sample.describe()
    for line in ("s.objects", "positions", "s.violation", "severity_map", "s.twin"):
        assert line in text, line
    assert not [key for key in sample._cache if key.startswith("h5:")]
    assert "positions" in sample.objects.describe()


def test_collating_plain_dicts_matches_collating_samples(tmp_path):
    """A DataLoader's workers hand collate `to_dict()` rows, not Samples; the
    batch -- `objects.valid` included -- must not depend on which it got."""
    make_sample(tmp_path, "a/pair/valid", "valid", pair_uid="a/pair", n_objects=2)
    make_sample(tmp_path, "b/pair/valid", "valid", pair_uid="b/pair", n_objects=3)
    ds = L.PhysLocDataset(str(tmp_path), fields=("objects.positions", "violation.mask"))
    from_samples = L.collate(ds.samples)
    from_dicts = L.collate([pickle.loads(pickle.dumps(s)).to_dict() for s in ds.samples])
    assert from_dicts["objects"]["valid"].tolist() == [[True, True, False], [True, True, True]]
    np.testing.assert_array_equal(from_dicts["objects"]["valid"], from_samples["objects"]["valid"])
    np.testing.assert_array_equal(from_dicts["objects"]["positions"],
                                  from_samples["objects"]["positions"])
    assert "valid" not in L.collate([s.to_dict(["violation.mask"]) for s in ds.samples]).get(
        "objects", {})


def test_missing_old_layout_is_not_interpreted(tmp_path):
    old = tmp_path / "clips" / "old" / "valid"
    old.mkdir(parents=True)
    (old / "metadata.json").write_text("{}")
    with pytest.raises(FileNotFoundError, match="schema-v4"):
        L.PhysLocDataset(str(tmp_path))
