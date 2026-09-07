"""Packaging a release: the shards, the index, and the split that must not leak.

Built on a synthetic clip tree rather than a generated release, so the test
runs anywhere and does not need docker, a renderer, or an hour.
"""
import json
import os
import tarfile

import numpy as np
import pytest

import physviol.release.export as X


def _clip(root, pair, name, label, family=None, seed=7):
    cdir = os.path.join(root, "clips", pair, name)
    os.makedirs(cdir, exist_ok=True)
    meta = {
        "clip_uid": "%s/%s" % (pair, name),
        "pair_uid": pair,
        "twin_uid": "%s/valid" % pair,
        "label": label,
        "scenario": pair.split("/")[1],
        "family": family,
        "domain": "identity",
        "physics_medium": "rigid",
        "seed": seed,
        "tier": "debug",
        "num_frames": 25,
        "fps": 12,
        "camera": {"motion": "orbit"},
        "instances": [
            {"role": "floor", "category": "cube", "dormant": False},
            {"role": "actor", "category": "cone", "material": "steel",
             "mass_kg": 3.5, "dormant": False},
            {"role": "actor", "category": "cone", "dormant": True},
        ],
        "violation": ({"t_event_frame": 8, "violation_windows": [[8, 12]],
                       "intervention": {"severity_bin": "strong",
                                        "magnitude": 1.5},
                       "peak_residual": {"score": 0.9}}
                      if label == "invalid" else None),
    }
    with open(os.path.join(cdir, "meta.json"), "w") as fh:
        json.dump(meta, fh)
    with open(os.path.join(cdir, "rgb.mp4"), "wb") as fh:
        fh.write(b"\0" * 64)
    np.savez_compressed(os.path.join(cdir, "violation_mask.npz"),
                        mask=np.zeros((2, 4, 4), bool))
    np.savez_compressed(os.path.join(cdir, "depth.npz"),
                        depth=np.zeros((2, 4, 4), np.float32))
    return cdir


@pytest.fixture
def release(tmp_path):
    root = str(tmp_path / "rel")
    for i in range(12):
        pair = "physviol_v0/drop/%04d" % i
        _clip(root, pair, "valid", "valid")
        _clip(root, pair, "invalid_permanence_strong", "invalid", "permanence")
    return root


def test_export_writes_shards_index_card_and_splits(release, tmp_path):
    out = str(tmp_path / "pack")
    res = X.export(release, out)
    assert res["clips"] == 24 and res["pairs"] == 12
    for name in ("README.md", "LICENSE"):
        assert os.path.exists(os.path.join(out, name))
    assert any(os.path.exists(os.path.join(out, n))
               for n in ("index.parquet", "index.jsonl"))
    assert res["shards"]["core"] >= 1


def test_a_twin_pair_is_never_split_apart(release, tmp_path):
    """The leak this grouping exists to prevent.

    A valid clip and its invalid siblings share every frame before `t_event`.
    Put them on opposite sides and the training set contains the answer.
    """
    out = str(tmp_path / "pack")
    X.export(release, out)
    where = {}
    for name, _ in X.SPLIT_FRACTIONS:
        with open(os.path.join(out, "splits", "%s.txt" % name)) as fh:
            for uid in fh.read().split():
                where.setdefault(uid.rsplit("/", 1)[0], set()).add(name)
    bad = {p: s for p, s in where.items() if len(s) > 1}
    assert not bad, "these pairs straddle a split: %s" % bad


def test_every_split_gets_pairs_even_on_a_small_release():
    """Independent hash bucketing does not hold its proportions when there are
    few pairs -- thirteen came out 11/2/0, and a benchmark whose test split is
    empty is not a benchmark. Ordering by hash and cutting at the quantiles
    does."""
    for n in (6, 13, 40, 400):
        got = X.assign_splits("pair/%d" % i for i in range(n))
        counts = {s: sum(1 for v in got.values() if v == s)
                  for s, _ in X.SPLIT_FRACTIONS}
        assert all(c > 0 for c in counts.values()), (n, counts)
        assert sum(counts.values()) == n


def test_splits_are_reproducible():
    """No rng anywhere: the same pairs must always land the same way, or a
    number reported against one build cannot be compared to the next."""
    uids = ["physviol_v0/drop/%04d" % i for i in range(50)]
    assert X.assign_splits(uids) == X.assign_splits(reversed(uids))


def test_raw_passes_stay_out_of_the_core_shards(release, tmp_path):
    """`depth`, `flow` and `object_coords` are ~86% of the bytes. Someone
    training on the masks should not download them to get there."""
    out = str(tmp_path / "pack")
    X.export(release, out)
    names = []
    for shard in sorted(os.listdir(os.path.join(out, "shards"))):
        with tarfile.open(os.path.join(out, "shards", shard)) as tf:
            names += tf.getnames()
    assert any(n.endswith(".violation_mask.npz") for n in names)
    assert not any(n.endswith(".depth.npz") for n in names), (
        "a raw geometry pass leaked into the core shards")


def test_with_passes_ships_them_separately(release, tmp_path):
    out = str(tmp_path / "pack")
    res = X.export(release, out, with_passes=True)
    assert res["shards"].get("passes", 0) >= 1
    with tarfile.open(os.path.join(out, "shards", "passes-000.tar")) as tf:
        assert any(n.endswith(".depth.npz") for n in tf.getnames())


def test_the_index_carries_what_a_filter_needs(release, tmp_path):
    """A consumer picking "moving camera, steel actors, strong bin" should not
    have to open 1500 meta.json files to do it."""
    out = str(tmp_path / "pack")
    X.export(release, out)
    path = os.path.join(out, "index.jsonl")
    if not os.path.exists(path):
        pytest.skip("parquet index; covered by the export summary")
    rows = [json.loads(l) for l in open(path)]
    inv = next(r for r in rows if r["label"] == "invalid")
    for field in ("scenario", "family", "severity_bin", "t_event_frame",
                  "camera_motion", "actor_shape", "actor_material",
                  "actor_mass", "split", "pair_uid"):
        assert inv.get(field) is not None, field
    assert inv["actor_shape"] == "cone" and inv["actor_material"] == "steel"


def test_the_held_out_split_ships_no_answers(release, tmp_path):
    """The mechanism, not the request.

    IntPhys 2 releases its held-out videos without metadata "to avoid training
    data contamination", and that is the only version of a held-out split that
    means anything -- one whose annotations sit in the same download measures
    whoever remembered not to look at them.
    """
    out = str(tmp_path / "pack")
    res = X.export(release, out)
    assert res["shards"].get("held_out"), "no held-out shard was written"

    held = [n for n in sorted(os.listdir(os.path.join(out, "shards")))
            if n.startswith("held_out")]
    for shard in held:
        with tarfile.open(os.path.join(out, "shards", shard)) as tf:
            names = tf.getnames()
            assert not any(n.endswith(".violation_mask.npz") for n in names)
            assert not any(n.endswith(".severity_map.npz") for n in names)
            assert not any(n.endswith(".timelines.npz") for n in names)
            for n in names:
                if not n.endswith(".meta.json"):
                    continue
                meta = json.loads(tf.extractfile(n).read())
                for leaked in ("violation", "family", "label", "instances"):
                    assert leaked not in meta, (
                        "held-out meta.json still carries %r" % leaked)


def test_every_split_sees_every_scenario(tmp_path):
    """A split missing a whole scenario measures familiarity, not physics.

    Cutting the population in one pass lets that happen: with thirteen
    scenarios and a 5% slice, the small splits would be a couple of scenarios
    each. Stratifying within each scenario is what prevents it.
    """
    uids = ["physviol_v0/%s/%04d" % (s, i)
            for s in ("drop", "collision", "pour", "toss") for i in range(20)]
    got = X.assign_splits(uids)
    for name, _ in X.SPLIT_FRACTIONS:
        scen = {u.split("/")[1] for u, v in got.items() if v == name}
        assert scen == {"drop", "collision", "pour", "toss"}, (name, scen)
