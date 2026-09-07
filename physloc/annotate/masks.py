"""Spatiotemporal masks -- docs/PLAN.md 3.3.

The primary annotation. Every mask is [T, H, W]: a value per pixel per frame, so
"where" and "when" are answered by one array.

**The union rule.** A naive "pixels of the culprit in the invalid render" is
wrong for a sixth of the taxonomy -- a vanished body has *no* invalid-side
pixels precisely because it vanished, and a teleported one is in two places. So

    violation_mask[t] = footprint(culprit, invalid, t) | footprint(culprit, valid, t)

which handles vanish (valid side only), spawn (invalid side only) and teleport
(both, disjoint) with one rule. It is well-defined only because the twins are
pixel-aligned and share instance indexing -- prefix identity paying off.
"""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np


def footprint(seg: np.ndarray, ids: Sequence[int]) -> np.ndarray:
    """[T,H,W] bool: pixels belonging to any of `ids`."""
    s = seg[..., 0] if seg.ndim == 4 else seg
    return np.isin(s, np.asarray(list(ids), dtype=s.dtype))


def violation_mask(seg_valid: np.ndarray, seg_invalid: np.ndarray,
                   dynamic_causal_ids: Sequence[int],
                   active: np.ndarray) -> np.ndarray:
    """The union rule, gated to the frames where the violation *can be seen*.

    Only *dynamic* causal bodies contribute. A static participant (the floor a
    ball sinks through) would otherwise flood the mask with the entire surface;
    it is recorded in `causal_mask` level 2 instead.

    Callers pass `active & observable`, not `active` alone. The two differ on
    real frames: a super-elastic bounce is unlawful the instant the contact
    resolves, but the body is still in exactly the same place in both twins, so
    that frame carries no visual evidence at all. Marking pixels there would ask
    a model to localise something the image does not contain -- and the three
    clocks already record that the violation began before it became visible.
    Where the culprit is fully occluded the two gates agree, because the union
    of two hidden footprints is empty either way.
    """
    u = footprint(seg_valid, dynamic_causal_ids) | footprint(seg_invalid,
                                                             dynamic_causal_ids)
    return u & np.asarray(active, bool)[:, None, None]


def invalid_mask(seg_invalid: np.ndarray, dynamic_causal_ids: Sequence[int],
                 active: np.ndarray) -> np.ndarray:
    """The culprit's footprint in the INVALID render alone, gated to `active`.

    The union in `violation_mask` exists so that a vanished body still has a
    mask, and it stays the documented training target. But it also marks pixels
    where the object lawfully *is*, and at inference a model only ever sees the
    invalid video -- so for anything that asks "where is the thing that is
    wrong", the union is asking for pixels the question does not contain.

    Empty for `permanence` and `dissolve` after the body goes, which is correct
    rather than unfortunate: there is no evidence in the invalid frame at a
    place where nothing is rendered. `reference_mask` still says where it
    should have been.
    """
    return (footprint(seg_invalid, dynamic_causal_ids)
            & np.asarray(active, bool)[:, None, None])


def causal_mask(seg_valid: np.ndarray, seg_invalid: np.ndarray,
                primary_ids: Sequence[int], secondary_ids: Sequence[int],
                active: np.ndarray, neighbourhood: int = 3,
                static_ids: Sequence[int] = ()) -> np.ndarray:
    """uint8 [T,H,W]: 0 = nothing, 1 = the culprit, 2 = a body it disturbed.

    **Invalid side only.** The question this array answers is "who did it, and
    what did it disturb", and both halves of that are about the clip in front of
    you -- not about where a body would have been in a rollout the viewer never
    sees.

    Level 2 is *measured*, not declared: `secondary_ids` are the bodies whose
    trajectory provably differs from the valid twin as a consequence, which is
    the same comparison `Injector._settle_bystanders` runs. Static participants
    are included only near the culprit, so "the floor" is the bit of floor being
    passed through rather than the whole plane.
    """
    prim = invalid_mask(seg_invalid, primary_ids, active)
    out = np.zeros(prim.shape, np.uint8)
    out[prim] = 1
    if len(secondary_ids):
        gate = np.asarray(active, bool)[:, None, None]
        static_near, moving = _split_participants(secondary_ids, static_ids)
        sec = np.zeros(prim.shape, bool)
        if static_near:
            # A surface only counts where the culprit is passing through it:
            # otherwise "the floor" becomes the whole plane.
            sec |= (footprint(seg_invalid, static_near)
                    & _dilate(prim, neighbourhood))
        if moving:
            # A body that was knocked away is a consequence *wherever it went*.
            # Restricting it to a neighbourhood of the culprit deleted it
            # exactly when it mattered -- the further a struck ball travels, the
            # more clearly it is an effect, and the more certainly it fell
            # outside the box.
            sec |= footprint(seg_invalid, moving)
        out[sec & gate & ~prim] = 2
    return out


def _split_participants(secondary_ids, static_ids):
    st = {int(i) for i in (static_ids or ())}
    near = [int(i) for i in secondary_ids if int(i) in st]
    moving = [int(i) for i in secondary_ids if int(i) not in st]
    return near, moving


def reference_mask(seg_valid: np.ndarray,
                   dynamic_causal_ids: Sequence[int]) -> np.ndarray:
    """Where the culprit *should* be -- its footprint in the valid twin.

    Shipped on both clips of a pair and ungated in time, so a consumer always
    has the counterfactual: the lawful trajectory the invalid clip departed
    from. It is what makes the union rule legible rather than mysterious -- for
    a vanished body this mask is the only place the object appears at all.

    Distinct from `violation_mask`, which is the union over both twins gated to
    the active window. This one is pure "should", with no claim about wrongness.
    """
    return footprint(seg_valid, dynamic_causal_ids)


def divergence_map(rgb_valid: np.ndarray, rgb_invalid: np.ndarray) -> np.ndarray:
    """|valid - invalid| in pixel space, normalised to [0,1].

    NOT the violation region -- after t_event a twin pair diverges everywhere
    downstream. Shipped for analysis and labelled as such; never a target.
    """
    a = rgb_valid[..., :3].astype(np.float32)
    b = rgb_invalid[..., :3].astype(np.float32)
    return (np.abs(a - b).max(axis=-1) / 255.0).astype(np.float16)


def _dilate(mask: np.ndarray, k: int) -> np.ndarray:
    """Square dilation by `k` px, per frame, without a SciPy dependency."""
    out = mask.copy()
    for dy in range(-k, k + 1):
        for dx in range(-k, k + 1):
            if dy == 0 and dx == 0:
                continue
            out |= np.roll(np.roll(mask, dy, axis=1), dx, axis=2)
    return out


def summarise(mask: np.ndarray) -> Dict[str, object]:
    T = mask.shape[0]
    per_frame = mask.reshape(T, -1).sum(axis=1)
    return {"frames_nonempty": int((per_frame > 0).sum()),
            "peak_pixels": int(per_frame.max()) if T else 0,
            "peak_frame": int(np.argmax(per_frame)) if T else -1}
