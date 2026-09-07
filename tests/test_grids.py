import numpy as np
import pytest

from physloc.annotate import grids
from physloc.scenarios import TIERS


def test_every_tier_bins_exactly():
    for tier in TIERS.values():
        bins = grids.temporal_bins(tier.num_frames, tier.latent_frames)
        assert len(bins) == tier.latent_frames
        covered = np.concatenate(bins)
        assert covered.min() == 0 and covered.max() == tier.num_frames - 1
        assert len(np.unique(covered)) == tier.num_frames


def test_non_4k1_timeline_is_rejected():
    with pytest.raises(ValueError):
        grids.temporal_bins(24, 6)


def test_mask_reduces_by_max_and_is_time_major():
    T, H, W, F, L = 13, 128, 128, 4, 8
    mask = np.zeros((T, H, W), bool)
    mask[6, 0, 0] = True                      # one pixel, one frame
    sev = mask.astype(np.float32)
    g = grids.reduce_all(mask, sev, F, L)
    m = g["mask_%dx%dx%d" % (F, L, L)]
    assert m.shape == (F, L, L)
    assert m[2, 0, 0]                          # frame 6 lands in latent bin 2
    assert m.sum() == 1                        # max, so exactly one cell marked
    assert str(g["ordering"]) == "time_major_F_H_W"
