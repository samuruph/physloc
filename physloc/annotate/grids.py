"""Token-grid reduction -- docs/PLAN.md 3.6.

Pre-reduces masks and severity to a video-DiT latent grid so a consumer never
re-derives a VAE's binning.

Ordering guarantee: time-major, [F_lat, H_lat, W_lat], latent frame slowest.
This is a schema guarantee, not an implementation detail -- flattening to
[F_lat*H_lat*W_lat] must match a transformer's token order.

Reduction rule: masks by max (a violation in any contributing source frame marks
the latent frame); severity by both max and mean, since peak and average are
different questions.
"""
from __future__ import annotations

from typing import Dict

import numpy as np


def temporal_bins(num_frames: int, latent_frames: int):
    """Which source frames feed each latent frame, under 4x VAE binning.

    A 4k+1 timeline maps as: latent 0 <- source 0; latent i>0 <- sources
    [4i-3 .. 4i]. Requiring 4k+1 is exactly what makes this exact.
    """
    if (num_frames - 1) % 4 != 0 or (num_frames - 1) // 4 + 1 != latent_frames:
        raise ValueError("num_frames=%d does not bin to %d latent frames (need 4k+1)"
                         % (num_frames, latent_frames))
    bins = [np.array([0])]
    for i in range(1, latent_frames):
        bins.append(np.arange(4 * i - 3, 4 * i + 1))
    return bins


def _spatial_reduce(x: np.ndarray, out_hw: int, how: str) -> np.ndarray:
    """[H,W] -> [out_hw,out_hw] by exact block reduction."""
    H, W = x.shape
    if H % out_hw or W % out_hw:
        raise ValueError("%dx%d does not divide into %d" % (H, W, out_hw))
    bh, bw = H // out_hw, W // out_hw
    b = x.reshape(out_hw, bh, out_hw, bw)
    return b.max(axis=(1, 3)) if how == "max" else b.mean(axis=(1, 3))


def reduce_all(mask: np.ndarray, severity: np.ndarray,
               latent_frames: int, latent_hw: int) -> Dict[str, np.ndarray]:
    T = mask.shape[0]
    bins = temporal_bins(T, latent_frames)
    tag = "%dx%dx%d" % (latent_frames, latent_hw, latent_hw)

    m = np.zeros((latent_frames, latent_hw, latent_hw), bool)
    smax = np.zeros((latent_frames, latent_hw, latent_hw), np.float32)
    smean = np.zeros((latent_frames, latent_hw, latent_hw), np.float32)
    sev = severity.astype(np.float32)

    for i, src in enumerate(bins):
        mm = mask[src].any(axis=0)
        m[i] = _spatial_reduce(mm.astype(np.float32), latent_hw, "max") > 0
        smax[i] = _spatial_reduce(sev[src].max(axis=0), latent_hw, "max")
        smean[i] = _spatial_reduce(sev[src].mean(axis=0), latent_hw, "mean")

    return {"mask_%s" % tag: m,
            "severity_max_%s" % tag: smax.astype(np.float16),
            "severity_mean_%s" % tag: smean.astype(np.float16),
            "latent_frames": np.int32(latent_frames),
            "latent_hw": np.int32(latent_hw),
            "ordering": np.array("time_major_F_H_W")}
