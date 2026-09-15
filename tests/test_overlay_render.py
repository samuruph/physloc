"""The renderer draws every layer and every panel on a real release.

Whether a picture is RIGHT is checked by eye (CLAUDE.md: look at the clips
before scaling). This makes sure no layer or panel has quietly stopped working
on some kind of clip -- a valid twin, a vanished body, several violators.
Skips without a generated v2 release.
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pytest

from conftest import REPO
from physloc import loader
from physloc.viz import overlay

RELEASE = os.environ.get("PHYSLOC_V2_RELEASE", os.path.join(REPO, "out/physloc_mini"))


@pytest.fixture(scope="module")
def clips():
    if not glob.glob(os.path.join(RELEASE, "clips", "**", loader.METADATA), recursive=True):
        pytest.skip("no v2 release at %s" % RELEASE)
    return loader.PhysLocDataset(RELEASE).clips


def test_every_layer_and_panel_renders_on_every_clip(clips):
    for clip in clips:
        r = overlay.Renderer(clip, list(overlay.LAYERS), list(overlay.PANELS), panel=128)
        t_event = int((clip.metadata.get("violation") or {}).get("t_event_frame", 0))
        for t in (0, t_event, clip.num_frames - 1):
            img = r.frame(t)
            assert img.shape == (r.height, r.width, 3) and img.dtype == np.uint8


def test_the_violation_layer_draws_on_the_rgb_panel(clips):
    clip = next(c for c in clips if not c.is_valid and c.violation_mask.any())
    t = int(np.flatnonzero(clip.violation_mask.reshape(clip.num_frames, -1).any(axis=1))[0])
    plain = overlay.Renderer(clip, [], ["rgb"], panel=128).frame(t)
    drawn = overlay.Renderer(clip, ["violation"], ["rgb"], panel=128).frame(t)
    assert (plain != drawn).any()


def test_no_box_is_drawn_around_a_body_that_is_not_there(clips):
    clip = next((c for c in clips if c.family == "permanence"), None)
    if clip is None:
        pytest.skip("no permanence clip")
    r = overlay.Renderer(clip, ["bbox3d"], ["rgb"], panel=128)
    gone = [iid for iid in clip.violator_ids
            if not clip.trajectory["present"][-1, list(clip.trajectory["body_ids"]).index(iid)]]
    assert gone, "the permanence violator should be absent by the last frame"
    assert all(iid not in dict((b, a) for a, b in r._boxed(clip.num_frames - 1))
               for iid in gone)


def test_an_unknown_layer_is_an_error(clips):
    with pytest.raises(KeyError):
        overlay.Renderer(clips[0], ["nope"])
