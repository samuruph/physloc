"""The camera's eye stays above the ground on every frame.

`occluder_pass` seed 22260826 at L2 was rendered from 0.32 m UNDER the floor
slab: `_vary`'s elevation swing took a low, close camera below the aim point
and through the ground, and the clip showed the slab's underside and the dome
with no actor pixel on any frame. Across 1040 sampled release scenes, 42 put
the eye below 0.4 m.
"""
from __future__ import annotations

import pytest

from physloc import scenarios
from physloc.scenarios import TIERS
from physloc.scenarios.base import CAMERA_MIN_HEIGHT, camera_clear_of_ground

RELEASE = TIERS["release"]


@pytest.mark.parametrize("name", sorted(scenarios.available()))
def test_camera_never_goes_below_the_ground(name):
    sc = scenarios.get(name)
    T = RELEASE.num_frames
    for seed in range(40):
        for level in ("L0", "L2"):
            spec = sc.sample(22260800 + seed, RELEASE, level,
                             variant=seed % 10, n_variants=10)
            assert camera_clear_of_ground(spec, T), (
                name, seed, level, spec.camera_position, spec.camera_end_position)
            # The opening pose itself honours the full height, not only half.
            assert (spec.camera_position[2] - spec.floor_level
                    >= CAMERA_MIN_HEIGHT - 1e-6), (name, seed, level)


def test_the_reported_clip_is_now_filmed_from_above():
    spec = scenarios.get("occluder_pass").sample(22260826, RELEASE, "L2",
                                                 variant=8, n_variants=10)
    assert spec.camera_position[2] >= CAMERA_MIN_HEIGHT - 1e-6
