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
import os

import numpy as np
import pytest

from conftest import REPO
from physloc import loader
from physloc.annotate.windows import rasterise

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
    for mp in sorted(glob.glob(os.path.join(root, "samples", "**", "sample.json"),
                               recursive=True)):
        sample = loader.Sample.from_dir(os.path.dirname(mp))
        if not sample.info.is_valid:
            yield sample


def test_the_release_validates(release):
    from physloc.schema.validate import validate_release

    report = validate_release(release)
    assert report["ok"], report["errors"]


def test_every_invalid_clip_describes_each_violator(release):
    for sample in _invalid_clips(release):
        v = sample.violation
        assert v.timing in ("independent", "sync", "shared")
        assert [c["id"] for c in v.violators] == [
            i for i in v.causal_ids if any(c["id"] == i for c in v.violators)]
        assert min(c["t_event"] for c in v.violators) == v.t_event
        for c in v.violators:
            assert c["t_event"] <= c["t_observable"]
            for s, e in c["windows"]["active"]:
                assert 0 <= s <= e < sample.video.num_frames


def test_independent_violators_keep_their_own_windows(release):
    seen = 0
    for sample in _invalid_clips(release):
        v = sample.violation
        if v.timing != "independent":
            continue
        seen += 1
        T = sample.video.num_frames
        rows = [sample.objects.row(c["id"]) for c in v.violators]
        for k, c in enumerate(v.violators):
            want = rasterise([tuple(w) for w in c["windows"]["active"]], T)
            assert np.array_equal(v.active[rows[k]], want)
        # The clip's windows are the union of the violators' own.
        clip_active = rasterise([tuple(w) for w in v.windows["active"]], T)
        assert np.array_equal(clip_active, v.active[rows].any(axis=0))
    if not seen:
        pytest.skip("this workdir has no independently timed violators")


def test_pixel_attribution_names_only_violators_and_their_consequences(release):
    for sample in _invalid_clips(release):
        v = sample.violation
        violator_ids = set(v.ids.tolist())
        vids, cids, cmask = v.object_id, v.causal_source, v.causal
        assert set(np.unique(vids[vids > 0]).tolist()) <= violator_ids
        assert np.array_equal(cids > 0, cmask > 0)
        assert set(np.unique(cids[cids > 0]).tolist()) <= violator_ids
