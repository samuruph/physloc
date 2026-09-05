"""Scenario registry. Importing this module registers every scenario."""
from .base import (COMPLEXITY, DEFAULT_COMPLEXITY, DEFAULT_TIER,  # noqa: F401
                   BodySpec, Complexity, LightSpec, Scenario, SceneSpec, Tier,
                   TIERS, available, get, implemented_complexities, register)
from . import (collision, drop, barrier_pass,  # noqa: F401
               pour,
               occluder_pass, pendulum_swing, toss, pyramid_impact,
               ramp_slide, resting_table, rolling_ramp, shadow_track,
               stack_topple)
