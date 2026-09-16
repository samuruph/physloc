"""Captions: every built scenario composes one, at every level.

The subject is found by segmentation id because a body's NAME follows the shape
it drew and, at L3, the scan it became -- looking it up as "ball" failed every
clip once names stopped being fixed.
"""
import pytest

from physloc import scenarios
from physloc.prompts import SCENARIO_PROMPTS, SUBJECTS, compose_prompt
from physloc.scenarios import TIERS


@pytest.mark.parametrize("name", sorted(SCENARIO_PROMPTS))
@pytest.mark.parametrize("level", ["L0", "L3"])
def test_every_scenario_captions_at_every_level(name, level):
    sc = scenarios.get(name)
    for seed in (777, 778, 779):
        spec = sc.sample(seed, TIERS["debug"], level)
        text = compose_prompt(name, spec.to_dict())
        assert "{" not in text


@pytest.mark.parametrize("name", sorted(SUBJECTS))
def test_the_subject_is_the_scenario_actor(name):
    spec = scenarios.get(name).sample(777, TIERS["debug"], "L0")
    body = next(b for b in spec.bodies
                if int(b.segmentation_id) == SUBJECTS[name])
    assert body.role == "actor"
