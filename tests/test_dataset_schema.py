"""Schema-v3 storage invariants and validation."""
from __future__ import annotations

import json
import os

import h5py
import numpy as np

from physloc import loader
from physloc.schema.validate import validate_release, validate_sample
from v3_fixture import make_pair


def test_v3_sample_has_only_three_payload_files(tmp_path):
    _valid, invalid = make_pair(tmp_path)
    assert set(os.listdir(invalid)) == {"sample.json", "rgb.mp4", "data.h5"}
    with open(os.path.join(invalid, "sample.json"), encoding="utf-8") as handle:
        document = json.load(handle)
    assert document["schema_version"] == 3
    assert "sample_info" in document["metadata"]
    assert "clip_properties" not in document["metadata"]
    assert document["observations"]["rgb"]["path"] == "rgb.mp4"


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
        assert positions.attrs["axes"] == "N,T"


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
    document["annotations"]["objects"][1]["role"] = "shadow_caster"
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(document, handle)
    assert any("shadow caster" in error for error in validate_sample(invalid))


def test_validator_rejects_undeclared_segmentation_id(tmp_path):
    _valid, invalid = make_pair(tmp_path)
    with h5py.File(os.path.join(invalid, loader.DATA), "r+") as handle:
        handle["/observations/segmentation"][0, 0, 0] = np.uint16(99)
    assert any("undeclared" in error for error in validate_sample(invalid))
