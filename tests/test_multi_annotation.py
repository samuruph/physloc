"""Per-violator annotation, end to end, on a real worker output.

Needs a container workdir whose plan gave each violator of a `multi` clip its
own moment -- `PHYSLOC_MULTI_WORKDIR`, or `out/smoke_multi/drop/0005`, made by

    bash docker/kubric.sh physloc/render/worker.py --scenario drop --seed 5 \
        --tier debug --family continuity,antigravity --severity strong \
        --complexity L0 --variant 8 --n-variants 10 --outdir out/smoke_multi

and skips without one, like every other test that reads rendered output.
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np
import pytest

from conftest import REPO

WORKDIR = os.environ.get("PHYSLOC_MULTI_WORKDIR",
                         os.path.join(REPO, "out/smoke_multi/drop/0005"))


@pytest.fixture(scope="module")
def release(tmp_path_factory):
    if not glob.glob(os.path.join(WORKDIR, "variants", "*", "plan.json")):
        pytest.skip("no multi-violator worker output at %s" % WORKDIR)
    from physloc.annotate.pipeline import annotate_work

    out = str(tmp_path_factory.mktemp("multi_release"))
    annotate_work(WORKDIR, out, write_video=False)
    return out


def _invalid_clips(root):
    for mp in sorted(glob.glob(os.path.join(root, "clips", "**", "metadata.json"),
                               recursive=True)):
        with open(mp) as fh:
            m = json.load(fh)
        if m["metadata"]["label"] == "invalid":
            yield os.path.dirname(mp), m


def test_the_release_validates(release):
    from physloc.schema.validate import validate_release

    report = validate_release(release)
    assert report["ok"], report["errors"]


def test_every_invalid_clip_describes_each_violator(release):
    for cdir, m in _invalid_clips(release):
        v = m["violation"]
        violators = v["violators"]
        assert v["violator_timing"] in ("independent", "sync", "shared")
        assert [c["instance_id"] for c in violators] == [
            i for i in v["causal_body_ids"]
            if any(c["instance_id"] == i for c in violators)]
        assert min(c["t_event_frame"] for c in violators) == v["t_event_frame"]
        for c in violators:
            assert c["t_event_frame"] <= c["t_observable_frame"]
            for s, e in c["violation_windows"]:
                assert 0 <= s <= e < m["metadata"]["num_frames"]


def test_independent_violators_keep_their_own_windows(release):
    seen = 0
    for cdir, m in _invalid_clips(release):
        v = m["violation"]
        if v["violator_timing"] != "independent":
            continue
        seen += 1
        T = m["metadata"]["num_frames"]
        tl = np.load(os.path.join(cdir, "timelines.npz"))
        assert list(tl["violator_ids"]) == [c["instance_id"] for c in v["violators"]]
        for k, c in enumerate(v["violators"]):
            want = np.zeros(T, bool)
            for s, e in c["violation_windows"]:
                want[s:e + 1] = True
            assert np.array_equal(tl["violator_active"][k], want)
        # The clip-level timeline is the union of the violators' own.
        assert np.array_equal(tl["active"], tl["violator_active"].any(axis=0))
    if not seen:
        pytest.skip("this workdir has no independently timed violators")


def test_pixel_attribution_names_only_violators_and_their_consequences(release):
    for cdir, m in _invalid_clips(release):
        v = m["violation"]
        violator_ids = {c["instance_id"] for c in v["violators"]}
        vids = np.load(os.path.join(cdir, "violation_ids.npz"))["ids"]
        vmask = np.load(os.path.join(cdir, "violation_mask.npz"))["mask"]
        assert np.array_equal(vids > 0, vmask)
        assert set(np.unique(vids[vids > 0]).tolist()) <= violator_ids
        cids = np.load(os.path.join(cdir, "causal_ids.npz"))["ids"]
        cmask = np.load(os.path.join(cdir, "causal_mask.npz"))["mask"]
        assert np.array_equal(cids > 0, cmask > 0)
        assert set(np.unique(cids[cids > 0]).tolist()) <= violator_ids
