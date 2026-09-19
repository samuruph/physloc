"""Schema-v4 storage invariants and validation."""
from __future__ import annotations

import json
import os

import h5py
import numpy as np

from physloc import loader
from physloc.schema.validate import validate_release, validate_sample
from sample_fixture import make_pair


def test_v4_sample_has_only_three_payload_files(tmp_path):
    _valid, invalid = make_pair(tmp_path)
    assert set(os.listdir(invalid)) == {"sample.json", "rgb.mp4", "data.h5"}
    with open(os.path.join(invalid, "sample.json"), encoding="utf-8") as handle:
        document = json.load(handle)
    assert document["schema_version"] == 4
    assert set(document) == {"schema_version", "sample", "video", "scene", "objects",
                             "violation", "provenance"}


#: Keys v4 stopped storing because the loader derives them, or because they
#: repeated another field. None of them may creep back into a sample.
REMOVED = {"metadata", "annotations", "storage", "observations", "sample_info",
           "objects_summary", "n_violators", "n_actors", "n_distractors",
           "duration_seconds", "timestep_seconds", "generation_config_id",
           "dataset_version", "render_seed", "complexity", "difficulty_analysis",
           "difficulty_inputs", "causal_relations", "violation_summary",
           "t_event_frame", "t_observable_frame", "t_end_frame",
           "observability_lag_frames", "violation_windows", "observable_windows",
           "intervention_windows", "consequence_windows", "temporal_info",
           "static_fields", "draw_scale", "dimensions", "camera_visible",
           "shadow_visible", "positions", "quaternions", "events", "instance_id",
           "affected_instance_ids", "magnitude_unit"}


def _keys(value, path=""):
    if isinstance(value, dict):
        for key, item in value.items():
            yield path + "." + key, key
            yield from _keys(item, path + "." + key)
    elif isinstance(value, list):
        for item in value:
            yield from _keys(item, path + "[]")


def test_v4_document_stores_each_fact_once(tmp_path):
    valid, invalid = make_pair(tmp_path)
    for sample_dir in (valid, invalid):
        with open(os.path.join(sample_dir, "sample.json"), encoding="utf-8") as handle:
            document = json.load(handle)
        leaked = sorted(path for path, key in _keys(document)
                        if key in REMOVED and not path.startswith(".scene.physics.params"))
        assert not leaked, leaked
    with open(os.path.join(valid, "sample.json"), encoding="utf-8") as handle:
        assert "violation" not in json.load(handle)


def test_valid_sample_stores_no_violation_arrays(tmp_path):
    valid, _invalid = make_pair(tmp_path)
    with h5py.File(os.path.join(valid, "data.h5"), "r") as handle:
        assert "violation" not in handle
        assert "events/collisions/instances" in handle
        assert "camera/positions" in handle


def test_writer_rejects_times_that_disagree_with_windows():
    import pytest
    from physloc.schema import write
    from sample_fixture import make_meta

    meta = make_meta("t/p/invalid_x")
    meta["violation"]["violators"][0]["t_event_frame"] = 3   # windows say 1
    with pytest.raises(ValueError, match="t_event_frame"):
        write.document_from_generation(meta)


def test_hdf5_is_chunked_compressed_and_self_describing(tmp_path):
    _valid, invalid = make_pair(tmp_path)
    with h5py.File(os.path.join(invalid, "data.h5"), "r") as handle:
        segmentation = handle["/observations/segmentation"]
        positions = handle["/objects/positions"]
        assert segmentation.compression == "gzip" and segmentation.compression_opts == 4
        assert segmentation.shuffle and segmentation.fletcher32
        assert segmentation.chunks[0] == 1
        assert positions.chunks[:2] == (1, 5)
        assert segmentation.attrs["axes"] == "T,H,W"
        assert positions.attrs["axes"] == "N,T,3"
        assert handle["/energy/objects/momentum"].attrs["units"] == "kg*m/s"
        assert "momentum_magnitude" not in handle["/energy/objects"]


def test_release_validation_checks_pairs_ids_and_arrays(tmp_path):
    make_pair(tmp_path)
    report = validate_release(str(tmp_path))
    assert report["ok"], report["errors"]
    assert report["samples"] == 2 and report["pairs"] == 1


def test_validator_rejects_a_public_shadow_object(tmp_path):
    _valid, invalid = make_pair(tmp_path)
    path = os.path.join(invalid, loader.SAMPLE_METADATA)
    with open(path, encoding="utf-8") as handle:
        document = json.load(handle)
    document["objects"][1]["role"] = "shadow_caster"
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(document, handle)
    assert any("shadow caster" in error for error in validate_sample(invalid))


def test_validator_rejects_undeclared_segmentation_id(tmp_path):
    _valid, invalid = make_pair(tmp_path)
    with h5py.File(os.path.join(invalid, loader.DATA), "r+") as handle:
        handle["/observations/segmentation"][0, 0, 0] = np.uint16(99)
    assert any("undeclared" in error for error in validate_sample(invalid))
