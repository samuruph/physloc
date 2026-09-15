"""`physloc/loader.py`: the derivations on hand-built arrays, then a real release.

Schema v2 stores `masks.npz` and `objects.npz` and derives everything else, so
these are the tests that the derived annotations mean what v1's stored ones did.
The last group reads a generated release and skips without one.
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pytest

from conftest import REPO
from physloc import loader as L

RELEASE = os.environ.get("PHYSLOC_V2_RELEASE", os.path.join(REPO, "out/physloc_mini"))


# ---- derivations ------------------------------------------------------------

def test_visible_violation_is_the_part_rendered_in_this_video():
    seg = np.array([[[0, 7], [7, 3]]], np.uint16)
    vio = np.array([[[7, 7], [0, 3]]], np.uint16)
    assert L.visible_violation(vio, seg).tolist() == [[[False, True], [False, True]]]


def test_severity_is_one_value_per_violator_per_frame():
    seg = np.array([[[2, 2], [5, 5]]], np.uint16)
    vio = seg.copy()
    m = L.paint_severity(vio, seg, [2, 5], np.array([[0.4], [0.9]], np.float32))
    np.testing.assert_allclose(m, [[[0.4, 0.4], [0.9, 0.9]]], rtol=1e-6)


def test_severity_stays_off_the_lawful_footprint_while_the_body_is_seen():
    """A teleport: the violation covers both twins' footprints, but the body is
    rendered at only one of them, and only that one is wrong to look at."""
    seg = np.array([[[0, 7], [0, 0]]], np.uint16)
    vio = np.array([[[0, 7], [7, 0]]], np.uint16)
    m = L.paint_severity(vio, seg, [7], np.array([[1.0]], np.float32))
    assert m[0, 0, 1] == 1.0 and m[0, 1, 0] == 0.0


def test_severity_falls_back_to_the_lawful_footprint_once_the_body_vanished():
    seg = np.array([[[0, 7], [0, 0]], [[0, 0], [0, 0]]], np.uint16)
    vio = np.array([[[0, 7], [0, 0]], [[0, 0], [7, 0]]], np.uint16)
    m = L.paint_severity(vio, seg, [7], np.array([[0.5, 0.25]], np.float32))
    assert m[0, 0, 1] == 0.5 and m[1, 1, 0] == 0.25
    assert m.sum() == pytest.approx(0.75)


def test_clip_timeline_unions_violators_and_takes_the_primary_occlusion():
    T = 3
    obj = {"ids": np.array([2, 5]),
           "severity": np.array([[0, .5, 0], [0, .2, .9]], np.float32)}
    obj.update({k: np.zeros((2, T), bool) for k in L.CLOCKS})
    obj["active"][0, 1] = obj["active"][1, 2] = True
    obj["occluded"][1] = True
    tl = L.clip_timeline(obj, T)
    assert tl["active"].tolist() == [False, True, True]
    assert not tl["occluded"].any()
    assert tl["severity"].tolist() == pytest.approx([0, .5, .9])


def test_latent_grid_matches_the_generator_reduction():
    from physloc.annotate import grids

    rng = np.random.default_rng(0)
    mask = rng.random((25, 16, 16)) > 0.9
    sev = np.where(mask, rng.random((25, 16, 16)), 0).astype(np.float32)
    got = L.latent_grid(mask, sev, 7, 8)
    ref = grids.reduce_all(mask, sev, 7, 8)
    assert np.array_equal(got["mask"], ref["mask_7x8x8"])
    np.testing.assert_allclose(got["severity_max"],
                               ref["severity_max_7x8x8"].astype(np.float32), atol=1e-3)


def test_collate_pads_objects_to_the_largest_K():
    a = {"uid": "a", "video": np.zeros((2, 2, 2, 3)),
         "objects": {"ids": np.array([1]), "severity": np.ones((1, 2))}}
    b = {"uid": "b", "video": np.zeros((2, 2, 2, 3)),
         "objects": {"ids": np.array([1, 2]), "severity": np.ones((2, 2))}}
    out = L.collate([a, b])
    assert out["video"].shape == (2, 2, 2, 2, 3)
    assert out["objects"]["severity"].shape == (2, 2, 2)
    assert out["objects"]["valid"].tolist() == [[True, False], [True, True]]
    assert out["uid"] == ["a", "b"]


def test_a_clip_path_names_its_identity():
    f = L._path_fields("mini/L1/drop/0005_camera-multi/invalid_newton2_mass_strong")
    assert f == {"release": "mini", "level": "L1", "scenario": "drop", "seed": "0005",
                 "condition": "camera+multi", "label": "invalid",
                 "family": "newton2_mass", "severity_bin": "strong",
                 "pair_uid": "mini/L1/drop/0005_camera-multi"}


# ---- a generated release ----------------------------------------------------

@pytest.fixture(scope="module")
def dataset():
    if not glob.glob(os.path.join(RELEASE, "clips", "**", L.METADATA), recursive=True):
        pytest.skip("no v2 release at %s" % RELEASE)
    return L.PhysLocDataset(RELEASE)


def test_the_release_validates(dataset):
    from physloc.schema.validate import validate_release

    report = validate_release(RELEASE)
    assert report["ok"], report["errors"]


def test_every_pair_has_its_valid_twin_and_shapes_agree(dataset):
    for pair in dataset.pairs():
        assert pair.valid is not None and pair.invalids, pair.pair_uid
        T, H, W = pair.valid.segmentations.shape
        assert pair.valid.video.shape == (T, H, W, 3)
        assert not pair.valid.violation_mask.any()
        for clip in pair.invalids:
            K = len(clip.violator_ids)
            assert clip.violation.shape == (T, H, W)
            assert clip.objects["severity"].shape == (K, T)
            assert clip.twin is pair.valid
            assert clip.violation_mask.any(), clip.uid
            assert not (clip.visible_violation & ~clip.violation_mask).any()


def test_the_objects_clocks_are_the_metadata_windows(dataset):
    from physloc.annotate.windows import rasterise

    for clip in dataset.clips:
        if clip.is_valid:
            continue
        T = clip.num_frames
        for k, v in enumerate(clip.metadata["violation"]["violators"]):
            want = rasterise([tuple(w) for w in v["violation_windows"]], T)
            assert np.array_equal(clip.objects["active"][k], want), clip.uid


def test_a_batch_collates(dataset):
    batch = L.collate([dataset[i] for i in range(min(4, len(dataset)))])
    assert batch["video"].ndim == 5
    assert batch["objects"]["valid"].ndim == 2


def test_exported_shards_read_back_identically(dataset, tmp_path):
    from physloc.release import export

    # WITH the passes: a clip then spans two tars, its core and its passes
    # shard, and reading every member from the last tar opened returned garbage.
    out = str(tmp_path / "pack")
    export.export(RELEASE, out, with_passes=True)
    packed = L.PhysLocDataset(out)
    assert len(packed) == len(dataset)
    assert sum(len(L.PhysLocDataset(out, split=name))
               for name, _ in export.SPLIT_FRACTIONS) == len(dataset)
    by_uid = {c.uid: c for c in packed.clips}
    for clip in dataset.clips:
        other = by_uid[clip.uid]
        assert other.metadata == clip.metadata
        if clip.has("depth.npz"):
            assert np.array_equal(other.pass_("depth"), clip.pass_("depth"))
        assert np.array_equal(other.violation, clip.violation)
        assert np.array_equal(other.severity_map, clip.severity_map)
        assert np.array_equal(other.reference_mask, clip.reference_mask)
