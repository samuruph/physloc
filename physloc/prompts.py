"""Textual prompts for v0 -- docs/PLAN.md is silent on captions, this is new.

One valid-physics caption template per scenario (LikePhys style: a short noun
phrase naming the scene and what happens in it, e.g. their "two balls
colliding with each other"), with `{color}`/`{shape}` slots that
`compose_prompt` fills from whatever a given clip actually sampled -- see
`physloc/scenarios/base.py::SceneSpec.to_dict` for where `color` and `kind`
reach `spec_d`. `physloc/annotate/pipeline.py::_build_meta` calls
`compose_prompt` once per clip and writes the result to `meta.json["prompt"]`,
so two clips of the same scenario read differently when they sampled
differently ("a red cube dropping..." vs "a blue ball dropping...").

`SUBJECTS` says which body's shape/color a scenario's template cares about.
Left out of `SUBJECTS` entirely (`stack_topple`, `pour`) means the template
has no `{color}`/`{shape}` slot -- naming one block's color in a stack of
differently-colored ones, or one grain's out of a pour, does not read as a
description of the scene. `collision` names only `ball_a`: its two balls are
sampled from one shared shape/color draw on purpose (see the "IDENTICAL"
comment in `collision.py`), so either one describes both.

Invalid-clip captions are deliberately not generated here. A hand-written
generic violation clause per family does not know which body in a
multi-body scenario the violation acts on, so it reads as vague at best and
wrong at worst -- e.g. `solidity` on `collision` cannot say which of the two
balls passes through the other. That precision already lives in
`meta.json`'s `violation` block. The plan is to caption invalid clips with a
VLM against the rendered frames instead, later; see docs/PLAN.md.

Run as a script to (re)write `docs/prompts_v0.json` -- the templates
themselves, placeholders and all, not any one clip's filled-in prompt:

    python -m physloc.prompts
"""
from __future__ import annotations

import colorsys
import json
import os
from typing import Dict, Sequence, Tuple

from .taxonomy import SCENARIOS, UNBUILT

#: One noun-phrase caption template per built scenario, describing the
#: *valid* physics only -- no violation vocabulary belongs here. Written in
#: the style of LikePhys's own examples (arXiv:2510.11512 Table 1). Scenarios
#: with a shape that varies per clip carry a `{shape}` slot; every scenario
#: with a colored subject carries `{color}` and `{article}` ("a"/"an",
#: `compose_prompt` picks it from the color's first letter -- "orange" is the
#: one of the 8 `COLOR_NAMES` that needs it). `collision` uses `{shape_pl}`
#: (`compose_prompt` derives it from `{shape}`) since it names two bodies.
SCENARIO_PROMPTS: Dict[str, str] = {
    "drop": "{article} {color} {shape} dropping and colliding with the "
            "ground, in an empty background",
    "collision": "two {color} {shape_pl} rolling toward each other and "
                 "colliding head-on",
    "ramp_slide": "{article} {color} block sliding down a ramp under friction",
    "toss": "{article} {color} {shape} thrown through the air on a free "
            "ballistic arc, with no contact",
    "occluder_pass": "{article} {color} {shape} rolling behind a screen and "
                      "re-emerging on the other side",
    "barrier_pass": "{article} {color} {shape} rolling into a solid wall "
                     "and bouncing back",
    "stack_topple": "a stack of blocks balanced at the edge of stability, "
                     "standing or toppling over",
    "pyramid_impact": "{article} {color} cube crashing down onto a pile of "
                       "spheres stacked in a pyramid",
    "pendulum_swing": "{article} {color} {shape} swinging back and forth "
                       "from a fixed pivot",
    "resting_table": "several objects resting still on a tabletop, "
                      "including {article} {color} {shape}",
    "rolling_ramp": "{article} {color} cube rolling down a raised ramp and "
                     "tumbling off its lip into free fall",
    "shadow_track": "{article} {color} ball moving across the ground under "
                     "a fixed light, its shadow tracking it faithfully",
    "pour": "a stream of grains pouring from above into an open box and "
            "piling up",
}

#: scenario -> the one body whose `kind`/`color` fill that scenario's
#: template. A scenario absent here has a template with no `{color}`/`{shape}`
#: slot at all -- see the module docstring for why `stack_topple` and `pour`
#: are the two.
SUBJECTS: Dict[str, str] = {
    "drop": "ball",
    "collision": "ball_a",
    "ramp_slide": "block",
    "toss": "ball",
    "occluder_pass": "ball",
    "barrier_pass": "ball",
    "pyramid_impact": "cube",
    "pendulum_swing": "bob",
    "resting_table": "mug",
    "rolling_ramp": "block",
    "shadow_track": "body",
}

#: `kind` -> the noun a prompt uses for it. Kubric's two primitives, so two
#: entries; extend this, not a hardcoded string, if a third primitive ever
#: joins the actor set.
SHAPE_NOUN: Dict[str, str] = {"sphere": "ball", "cube": "cube",
                              "cylinder": "cylinder",
                              "cone": "cone", "torus": "ring"}

#: 8 equal 45-degree hue bands. `hue_rgb` (`scenarios/_common.py`) always
#: draws `s=0.62, v=0.88` -- a fixed, vivid, non-gray, non-white color -- so
#: bucketing on hue alone is enough; finer bands would split hairs a viewer
#: could not reliably name back either.
COLOR_NAMES: Tuple[str, ...] = (
    "red", "orange", "yellow", "green", "cyan", "blue", "purple", "pink")


def color_name(rgb: Sequence[float]) -> str:
    """The nearest named color for an `(r, g, b)` in [0, 1]."""
    h, _s, _v = colorsys.rgb_to_hsv(*rgb[:3])
    return COLOR_NAMES[int(round(h * len(COLOR_NAMES))) % len(COLOR_NAMES)]


def compose_prompt(scenario: str, spec_d: dict) -> str:
    """The caption for one sampled clip: `SCENARIO_PROMPTS[scenario]` filled
    in from whichever body `SUBJECTS[scenario]` names, if any."""
    template = SCENARIO_PROMPTS[scenario]
    subject_name = SUBJECTS.get(scenario)
    if subject_name is None:
        return template
    bodies = {b["name"]: b for b in spec_d.get("bodies", [])}
    body = bodies[subject_name]
    shape = SHAPE_NOUN.get(body["kind"], body["kind"])
    color = color_name(body["color"])
    article = "an" if color[0] in "aeiou" else "a"
    return template.format(shape=shape, shape_pl=shape + "s", color=color,
                           article=article)


def build_prompts() -> Dict[str, str]:
    """Every template for v0: one per built scenario, placeholders intact."""
    built = set(s for s in SCENARIOS if s not in UNBUILT)
    assert set(SCENARIO_PROMPTS) == built, (
        "SCENARIO_PROMPTS is out of sync with taxonomy.SCENARIOS")
    assert set(SUBJECTS) <= built, (
        "SUBJECTS names a scenario SCENARIO_PROMPTS does not have")
    return dict(SCENARIO_PROMPTS)


def main() -> None:
    out_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "docs", "prompts_v0.json")
    with open(out_path, "w") as fh:
        json.dump(build_prompts(), fh, indent=2, sort_keys=True)
        fh.write("\n")
    print("wrote %s" % out_path)


if __name__ == "__main__":
    main()
