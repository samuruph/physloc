"""One encoder, one set of settings -- every mp4 in the project comes from here.

Lives host-side: the Kubric container has no ffmpeg, so the worker writes only
arrays and all video is produced here.
"""
from __future__ import annotations

import os
from typing import Iterable, Optional, Sequence

import numpy as np

CRF = "14"
#: H.264 macroblock. Canvases are padded up to a multiple of this.
MACROBLOCK = 16


def to_uint8_rgb(frames: np.ndarray) -> np.ndarray:
    a = np.asarray(frames)
    if a.ndim == 4 and a.shape[-1] == 4:
        a = a[..., :3]
    if a.ndim == 3:                                   # [T,H,W] grayscale
        a = np.repeat(a[..., None], 3, axis=-1)
    if a.dtype != np.uint8:
        a = np.clip(a, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(a)


def write(frames: np.ndarray, path: str, fps: int = 12,
          upscale: Optional[int] = None) -> Optional[str]:
    """Encode [T,H,W,C] to H.264. `upscale` nearest-neighbours small tiers so a
    128px debug clip is actually watchable."""
    try:
        import imageio.v2 as imageio
    except ImportError:
        return None
    a = to_uint8_rgb(frames)
    if upscale and upscale > 1:
        a = np.repeat(np.repeat(a, upscale, axis=1), upscale, axis=2)

    # Pad up to a multiple of the H.264 macroblock, rather than let imageio do
    # it. Left alone, imageio *resizes* a canvas that is not divisible by 16 --
    # 1770x412 became 1776x416 -- which resamples every mask edge and severity
    # value in the published file, and says so once per clip on stderr. Padding
    # by edge replication changes nothing inside the frame and keeps the encoder
    # quiet. Clip renders are 128/256/512 square and already divisible, so this
    # only ever fires on composed canvases: overlays, grids and sheets.
    pad_h = (-a.shape[1]) % MACROBLOCK
    pad_w = (-a.shape[2]) % MACROBLOCK
    if pad_h or pad_w:
        a = np.pad(a, ((0, 0), (0, pad_h), (0, pad_w), (0, 0)), mode="edge")

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    # `pixelformat` rather than a -pix_fmt in `output_params`: imageio passes
    # its own -pix_fmt too, and ffmpeg then warns "Multiple -pix_fmt options
    # specified" on every single encode.
    imageio.mimwrite(path, a, fps=fps, codec="libx264",
                     pixelformat="yuv420p", macro_block_size=MACROBLOCK,
                     output_params=["-crf", CRF])
    return path


def hstack_labelled(strips: Sequence[np.ndarray], labels: Sequence[str],
                    colors: Optional[Sequence[tuple]] = None) -> np.ndarray:
    """Side-by-side comparison video: [T,H,W*n,3] with a caption per panel."""
    import cv2
    n = len(strips)
    colors = colors or [(255, 255, 255)] * n
    outs = []
    T = min(s.shape[0] for s in strips)
    for t in range(T):
        panels = []
        for s, lab, col in zip(strips, labels, colors):
            im = to_uint8_rgb(s)[t].copy()
            cv2.rectangle(im, (0, im.shape[0] - 14), (im.shape[1], im.shape[0]),
                          (0, 0, 0), -1)
            cv2.putText(im, lab, (3, im.shape[0] - 4), cv2.FONT_HERSHEY_SIMPLEX,
                        0.30, col, 1, cv2.LINE_AA)
            panels.append(im)
        outs.append(np.hstack(panels))
    return np.stack(outs)
