"""The union rule -- docs/PLAN.md 3.3.

    violation_mask[t] = footprint(culprit, invalid, t) | footprint(culprit, valid, t)

Without it, vanish- and teleport-type violations produce empty or half-empty
masks: the body has no invalid-side pixels precisely *because* it vanished. A
regression here silently empties the masks on a sixth of the taxonomy, so this
tests the rule directly rather than only through a rendered clip.
"""
import glob
import json
import os

import numpy as np
import pytest

from conftest import find_release
from physloc.annotate import masks


def _seg(T=3, H=4, W=4, fill=0):
    return np.full((T, H, W), fill, np.uint16)


def test_vanish_mask_comes_from_the_valid_twin():
    """The culprit is absent from the invalid render entirely."""
    sv, si = _seg(), _seg()
    sv[:, 1, 1] = 7                       # present in valid
    active = np.ones(3, bool)
    m = masks.violation_mask(sv, si, [7], active)
    assert m[:, 1, 1].all(), "vanish produced an empty mask"
    assert m.sum() == 3


def test_spawn_mask_comes_from_the_invalid_twin():
    sv, si = _seg(), _seg()
    si[:, 2, 2] = 7
    m = masks.violation_mask(sv, si, [7], np.ones(3, bool))
    assert m[:, 2, 2].all()


def test_teleport_mask_covers_both_places():
    sv, si = _seg(), _seg()
    sv[:, 0, 0] = 7                       # where it should be
    si[:, 3, 3] = 7                       # where it wrongly is
    m = masks.violation_mask(sv, si, [7], np.ones(3, bool))
    assert m[:, 0, 0].all() and m[:, 3, 3].all()
    assert m.sum() == 6, "both lobes must be marked"


def test_mask_is_gated_to_active_frames():
    sv, si = _seg(), _seg()
    sv[:, 1, 1] = 7
    active = np.array([False, True, False])
    m = masks.violation_mask(sv, si, [7], active)
    assert not m[0].any() and m[1].any() and not m[2].any()


def test_static_participant_does_not_flood_the_mask():
    """A floor is a causal body for solidity, but the mask must not become the
    whole plane; static participants belong in causal_mask level 2."""
    sv, si = _seg(fill=1), _seg(fill=1)    # 1 == floor everywhere
    sv[:, 1, 1] = 2                        # 2 == ball, present in both twins
    si[:, 1, 1] = 2
    m = masks.violation_mask(sv, si, [2], np.ones(3, bool))
    assert m.sum() == 3, "only the dynamic body should contribute"
    c = masks.causal_mask(sv, si, [2], [1], np.ones(3, bool), neighbourhood=1,
                          static_ids=[1])
    assert (c == 1).sum() == 3
    assert (c == 2).sum() > 0, "the floor should appear near the culprit"
    assert (c == 2).sum() < sv.size, "but not everywhere"


def test_causal_and_invalid_masks_are_invalid_side_only():
    """A body that is not in the invalid render has no causal mask.

    The deliberate consequence of moving causal_mask and severity_map off the
    union: they answer "where is the thing that is wrong", and at inference a
    model only ever has the invalid video. For `permanence` and `dissolve` that
    means an empty mask once the body goes -- honest, because there is nothing
    rendered there -- and `reference_mask` still carries where it should have
    been. `violation_mask` keeps the union and keeps its pixels.
    """
    sv, si = _seg(fill=1), _seg(fill=1)
    sv[:, 1, 1] = 2                        # in the valid twin only: it vanished
    active = np.ones(3, bool)

    assert masks.violation_mask(sv, si, [2], active).sum() == 3
    assert masks.reference_mask(sv, [2]).sum() == 3
    assert masks.invalid_mask(si, [2], active).sum() == 0
    assert masks.causal_mask(sv, si, [2], [1], active,
                             static_ids=[1]).sum() == 0


def test_invalid_mask_is_a_subset_of_the_union():
    sv, si = _seg(fill=1), _seg(fill=1)
    sv[:, 1, 1] = 2
    si[:, 2, 2] = 2                        # moved: both twins contribute
    active = np.ones(3, bool)
    u = masks.violation_mask(sv, si, [2], active)
    i = masks.invalid_mask(si, [2], active)
    assert i.sum() == 3 and u.sum() == 6
    assert np.all(u[i]), "invalid-side pixels must all be inside the union"


@pytest.mark.parametrize("which", ["violation_mask"])
def test_released_clips_have_nonempty_masks_while_active(which):
    root = find_release()
    if root is None:
        pytest.skip("no release; run `python -m physloc.cli generate --debug`")
    n = 0
    for mp in glob.glob(os.path.join(root, "clips", "**", "meta.json"),
                        recursive=True):
        cdir = os.path.dirname(mp)
        with open(mp) as fh:
            meta = json.load(fh)
        if meta.get("label") != "invalid":
            continue
        mask = np.load(os.path.join(cdir, "%s.npz" % which))["mask"]
        tl = np.load(os.path.join(cdir, "timelines.npz"))
        # WHICH window the mask answers to depends on where the violation is
        # detectable. An `event` family -- a colour that finishes changing, a
        # body that finishes growing -- is only wrong ACROSS the change: a
        # recoloured cube is a normal cube, and no frame afterwards contains
        # the violation. A `state` family is wrong in every frame it lasts.
        #
        # Asserting against the union `active` failed on `colour_shift` for
        # exactly that reason, and the mask was right.
        from physloc.taxonomy import FAMILIES

        fam = FAMILIES.get(meta.get("family"))
        key = ("intervening" if fam is not None and fam.detectable == "event"
               else "consequence")
        scored = np.asarray(tl[key] if key in tl.files else tl["active"], bool)
        observable = np.asarray(tl["observable"], bool)
        per = mask.reshape(mask.shape[0], -1).any(axis=1)
        # Non-empty exactly where the violation is both scored and visible.
        # While the culprit is fully occluded the violation is active but has no
        # visible extent, so an empty mask is correct there -- that is the case
        # the observability lag describes.
        assert not (scored & observable & ~per).any(), (
            "%s: empty mask on a scored+observable frame" % cdir)
        assert not (scored & ~observable & per).any(), (
            "%s: mask has pixels while nothing is observable" % cdir)
        n += 1
    assert n > 0


def test_a_fully_occluded_violation_has_an_empty_mask():
    """Active but invisible: the culprit is hidden in the valid twin and gone
    from the invalid one, so the union is legitimately empty. The annotation
    must say "nothing to see", not invent a region."""
    sv, si = _seg(), _seg()
    # culprit has no pixels in either twin -- fully occluded, then removed
    m = masks.violation_mask(sv, si, [7], np.ones(3, bool))
    assert not m.any()


def test_reference_mask_is_the_lawful_footprint():
    """`reference_mask` says where the culprit *should* be, taken from the valid
    twin and ungated in time. For a vanished body it is the only mask with any
    pixels at all, which is what makes the union rule legible."""
    sv, si = _seg(), _seg()
    sv[:, 1, 1] = 7                       # lawful position, every frame
    # invalid twin: the body is gone entirely
    ref = masks.reference_mask(sv, [7])
    assert ref[:, 1, 1].all()
    assert ref.sum() == 3, "ungated: present on every frame, not just active ones"
    # ...and it is not the violation mask, which is gated to the active window
    vm = masks.violation_mask(sv, si, [7], np.array([False, True, False]))
    assert vm.sum() == 1
