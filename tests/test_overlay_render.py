"""The visualiser consumes the same schema-v4 Sample API as experiments."""
from __future__ import annotations

import numpy as np
import pytest

from physloc import loader
from physloc.viz import overlay
from sample_fixture import make_pair


@pytest.fixture
def samples(tmp_path):
    make_pair(tmp_path)
    return loader.PhysLocDataset(str(tmp_path)).samples


def test_every_layer_and_panel_renders_on_every_sample(samples):
    for sample in samples:
        renderer = overlay.Renderer(sample, list(overlay.LAYERS),
                                    list(overlay.PANELS), panel=128)
        event = sample.violation.t_event or 0
        for frame in (0, event, sample.video.num_frames - 1):
            image = renderer.frame(frame)
            assert image.shape == (renderer.height, renderer.width, 3)
            assert image.dtype == np.uint8


def test_violation_layer_draws_on_rgb(samples):
    sample = next(value for value in samples if not value.info.is_valid)
    frame = int(np.flatnonzero(sample.violation.mask.reshape(
        sample.video.num_frames, -1).any(axis=1))[0])
    plain = overlay.Renderer(sample, [], ["rgb"], panel=128).frame(frame)
    drawn = overlay.Renderer(sample, ["violation"], ["rgb"], panel=128).frame(frame)
    assert (plain != drawn).any()


def test_direct_sample_loading_resolves_reference_twin(samples):
    invalid = next(value for value in samples if not value.info.is_valid)
    sample = loader.Sample.from_dir(invalid.path)
    assert sample.twin is not None
    assert sample.twin.info.is_valid
    assert sample.violation.reference_mask.any()


def test_unknown_layer_is_an_error(samples):
    with pytest.raises(KeyError):
        overlay.Renderer(samples[0], ["nope"])


def test_grid_and_sheet_read_a_pair_directory(tmp_path):
    """`physloc grid`/`sheet` are handed a pair directory, whose samples sit
    directly inside it; they once searched it for a `samples/` tree and found
    nothing, then passed `_panel` arguments one slot late (depth as seg)."""
    from physloc.viz import grid

    valid, _invalid = make_pair(tmp_path)
    pair_dir = valid.rsplit("/", 1)[0]
    out = grid.build(pair_dir, out_path=str(tmp_path / "grid.mp4"), cell=48)
    assert out["frames"] == 5
    sheet = grid.sheet(pair_dir, out_path=str(tmp_path / "sheet.mp4"), cell=48)
    assert sheet["columns"] == ["valid", "solidity"]
