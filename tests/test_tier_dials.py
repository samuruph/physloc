"""Every config spells out its tier's geometry, and restating it changes nothing.

`resolution`, `fps`, `frames` and `spp` sit in each config's `defaults:` block so
a run is resized in the config. As written they equal the tier's own values, and
that must be a no-op: same tier name in every clip, same resume request, same
price. Only a value that differs is a change.
"""
import glob
import os
import re

import pytest

from physloc import cli, config, scenarios
from physloc.scenarios import TIERS

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIGS = sorted(os.path.basename(p)[:-5]
                 for p in glob.glob(os.path.join(HERE, "configs", "*.yaml"))
                 if not p.endswith("common.yaml"))


def test_restating_the_tier_leaves_it_alone():
    t = TIERS["release"]
    assert t.override(resolution=512, fps=30, num_frames=89,
                      samples_per_pixel=64) is t


def test_only_a_real_difference_is_named():
    assert TIERS["release"].override(resolution=512, num_frames=49).name == "release+f49"
    assert TIERS["debug"].override(samples_per_pixel=64).name == "debug+spp64"


def test_restated_dials_are_dropped_before_they_reach_the_run():
    assert cli._changed_dials(TIERS["release"], 512, 30, 49, 64) == (None, None, 49, None)
    assert cli._changed_dials(TIERS["debug"], None, None, None, None) == (None,) * 4


def test_pour_keeps_the_debug_medium_under_a_dial():
    """A renamed debug tier (`debug+f49`) must not get the release grain count."""
    spec = scenarios.get("pour").sample(777, TIERS["debug"].override(num_frames=49), "L0")
    grains = [b for b in spec.bodies if b.name.startswith("grain_")]
    assert len(grains) == 96


def _geometry(name):
    _, subs = cli._build()
    valid = {ac.dest for ac in subs["generate"]._actions} - {"help", "config"}
    got = config.load(name, "generate", valid)
    return got["tier"], tuple(got.get(k) for k in ("resolution", "fps", "frames", "spp"))


@pytest.mark.parametrize("name", CONFIGS)
def test_every_config_spells_out_a_valid_geometry(name):
    tier, (resolution, fps, frames, spp) = _geometry(name)
    assert None not in (resolution, fps, frames, spp), \
        "%s: resolution, fps, frames and spp must all be spelled out" % name
    # Raises for an illegal geometry, e.g. a frame count that is not 4k+1.
    TIERS[tier].override(resolution=resolution, fps=fps, num_frames=frames,
                         samples_per_pixel=spp)


def test_the_release_partition_shares_one_geometry():
    """v0_L0..v0_L3 sum to v0_release, so a level with a different clip length
    or size would silently mix geometries inside one published dataset."""
    release = _geometry("v0_release")
    for level in ("v0_L0", "v0_L1", "v0_L2", "v0_L3"):
        assert _geometry(level) == release, level


def _serial_hours(capsys, *extra):
    argv = ["taxonomy", "--tier", "release", "--complexity", "L0", "--variants", "1",
            "--severity", "strong", "--scenario", "drop"] + list(extra)
    assert cli.main(argv) == 0
    return float(re.search(r"~([\d.]+) h serial", capsys.readouterr().out).group(1))


def test_taxonomy_prices_a_changed_frame_count(capsys):
    full = _serial_hours(capsys)
    same = _serial_hours(capsys, "--frames", "89")
    half = _serial_hours(capsys, "--frames", "45")
    assert same == full
    assert abs(half / full - 45.0 / 89.0) < 0.05
