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

from .. import loader as _loader


#: The implementation lives in `physloc/loader.py`, which a consumer imports
#: without the generator. Schema v3 derives token grids on demand instead of
#: publishing another dense file; these aliases keep generator and loader math
#: identical.
temporal_bins = _loader.temporal_bins
_spatial_reduce = _loader._block


def reduce_all(mask: np.ndarray, severity: np.ndarray,
               latent_frames: int, latent_hw: int) -> Dict[str, np.ndarray]:
    g = _loader.latent_grid(mask, severity, latent_frames, latent_hw)
    tag = "%dx%dx%d" % (latent_frames, latent_hw, latent_hw)
    return {"mask_%s" % tag: g["mask"],
            "severity_max_%s" % tag: g["severity_max"].astype(np.float16),
            "severity_mean_%s" % tag: g["severity_mean"].astype(np.float16),
            "latent_frames": np.int32(latent_frames),
            "latent_hw": np.int32(latent_hw),
            "ordering": np.array("time_major_F_H_W")}
