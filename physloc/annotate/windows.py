"""Temporal annotations -- docs/PLAN.md 3.2.

Violations are *windows*, not onsets: a list of intervals, because a repeated
family fires once per bounce and observability can be interrupted by
re-occlusion. This module turns the injector's plan plus the two renders into
`violation_windows`, `observable_windows` and the rasterised timelines.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

Window = Tuple[int, int]


def rasterise(windows: Sequence[Window], num_frames: int) -> np.ndarray:
    a = np.zeros((num_frames,), bool)
    for s, e in windows:
        a[max(0, int(s)):min(num_frames, int(e) + 1)] = True
    return a


def to_windows(active: np.ndarray) -> List[Window]:
    """Inverse of `rasterise`: the maximal True runs, as inclusive intervals."""
    out: List[Window] = []
    idx = np.flatnonzero(active.astype(bool))
    if idx.size == 0:
        return out
    start = prev = int(idx[0])
    for i in idx[1:]:
        i = int(i)
        if i == prev + 1:
            prev = i
            continue
        out.append((start, prev))
        start = prev = i
    out.append((start, prev))
    return out


def observable_frames(seg_valid: np.ndarray, seg_invalid: np.ndarray,
                      causal_ids: Sequence[int],
                      rgb_valid: Optional[np.ndarray] = None,
                      rgb_invalid: Optional[np.ndarray] = None,
                      level: int = 8, min_pixels: int = 4) -> np.ndarray:
    """Frames carrying visual evidence, computed exactly rather than estimated.

    The twins share a bit-identical render path, so before the intervention the
    causal bodies' footprints are *identical*. A frame is observable exactly
    when the two renders disagree about them -- which is why this needs no
    heuristic.

    Two ways they can disagree, and for a long time only the first was checked:

    * **occupancy** -- the culprit covers different pixels. Catches everything
      that moves, vanishes or changes size.
    * **appearance** -- it covers the same pixels and they *look* different.
      Catches everything that does not move: a body changing colour, a shadow
      changing shape, a body fading out.

    Missing the second was not a subtle loss. Segmentation records which
    instance owns a pixel and nothing about what that pixel looks like, so a
    `colour_shift` clip -- an object plainly turning from red to green --
    reported zero observable frames, and shipped with an empty mask and a
    severity of exactly zero. Any family whose whole subject is appearance was
    unmeasurable by construction.

    Pass only the **dynamic** causal bodies. Including a static participant
    silently destroys the occupancy signal: when a ball sinks into a floor,
    pixels move from the ball id to the floor id, so the *union* footprint
    {ball, floor} is unchanged and every frame looks identical.
    """
    T = seg_valid.shape[0]
    ids = np.asarray(list(causal_ids), dtype=seg_valid.dtype)
    fv = np.isin(seg_valid, ids)
    fi = np.isin(seg_invalid, ids)
    diff = (fv != fi).reshape(T, -1).any(axis=1)

    if rgb_valid is None or rgb_invalid is None:
        return diff

    # Appearance, restricted to where the culprit is in either twin, so a
    # shadow moving elsewhere in the frame is not mistaken for evidence about
    # this body. `level` is in 0-255; a few pixels' worth of path-tracing noise
    # should not count as a violation becoming visible.
    here = (fv | fi)
    if here.ndim == 4:
        here = here[..., 0]
    a = np.asarray(rgb_valid)[..., :3].astype(np.int16)
    b = np.asarray(rgb_invalid)[..., :3].astype(np.int16)
    changed = (np.abs(a - b).max(axis=-1) >= level) & here
    return diff | (changed.reshape(T, -1).sum(axis=1) >= min_pixels)


def occluded_frames(seg: np.ndarray, body_id: int) -> np.ndarray:
    """Frames where `body_id` contributes no pixels at all -- fully hidden
    (behind an occluder, off-screen, or absent)."""
    T = seg.shape[0]
    return ~(seg.reshape(T, -1) == body_id).any(axis=1)


def build(plan_windows: Sequence[Window], num_frames: int,
          seg_valid: np.ndarray, seg_invalid: np.ndarray,
          causal_ids: Sequence[int], primary_id: int,
          severity_t: Optional[np.ndarray] = None,
          observable: Optional[np.ndarray] = None) -> Dict[str, object]:
    """Everything in PLAN 3.2, ready for meta.json + timelines.npz."""
    active = rasterise(plan_windows, num_frames)
    if observable is None:
        observable = observable_frames(seg_valid, seg_invalid, causal_ids)
    occluded = occluded_frames(seg_invalid, primary_id)

    obs_windows = to_windows(observable)
    t_event = int(min(s for s, _ in plan_windows))
    t_end = int(max(e for _, e in plan_windows))
    t_obs = int(np.flatnonzero(observable)[0]) if observable.any() else t_event

    if severity_t is None:
        severity_t = np.zeros((num_frames,), np.float32)

    return {
        "t_event_frame": t_event,
        "t_observable_frame": t_obs,
        "t_end_frame": t_end,
        "observability_lag_frames": int(t_obs - t_event),
        "violation_windows": [[int(s), int(e)] for s, e in plan_windows],
        "observable_windows": [[int(s), int(e)] for s, e in obs_windows],
        "occluded_at_event": bool(occluded[t_event])
        if 0 <= t_event < num_frames else False,
        "_arrays": {
            "active": active,
            "observable": observable,
            "occluded": occluded,
            "severity_t": np.asarray(severity_t, np.float32),
        },
    }
