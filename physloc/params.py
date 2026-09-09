"""The knobs, in one place, editable without touching code.

Shares, counts and bands were spread across `scenarios/base.py`,
`scenarios/_common.py` and `scenarios/materials.py` as module constants -- each
well documented where it sat, and collectively impossible to see or change as a
set. This is the set. `configs/common.yaml` is the human-editable form.

**STDLIB ONLY, because this crosses the seam.** Scene sampling runs in the
container, whose Python is 3.9 with Kubric's pinned packages and no PyYAML --
`config.py` says so and it is right. So the flow is:

    configs/common.yaml   (host reads it, with PyYAML)
        -> params.json    (written into the workdir, which is mounted)
        -> params.apply() (container reads it, with json)

which also means the resolved values land in `meta.json`, so a clip records
what it was generated under. A tunable nobody can reproduce is worse than a
constant nobody can change.

`apply()` rewrites the module constants rather than making every caller ask
this module: the constants are the API that `reference.py` renders from and the
tests assert against, and keeping one read-time source is worth more than
purity about where the value came from.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

#: Every tunable, with the value the code shipped with. The shape of this dict
#: IS the shape of `common.yaml`; anything not here cannot be set from a file,
#: which is deliberate -- a knob is a promise to keep it working.
DEFAULTS: Dict[str, Any] = {
    "ladder": {
        # What share of a full generation each level gets, relative to L0.
        # Falling as realism rises: the baseline is what everything is compared
        # against, and an HDRI clip costs several times a solid-background one.
        "shares": {"L0": 1.00, "L1": 0.50, "L2": 0.30, "L3": 0.20},
    },
    "conditions": {
        # The difficulty cycle, one entry per variant slot. Its LENGTH is the
        # period and its contents are the shares -- six `standard` in ten is
        # 60%. Editing this edits the metadata, the card, the README table and
        # the cost estimate, because nothing else stores the shares.
        "cycle": ["standard", "standard", "standard", "standard", "standard",
                  "standard", "camera", "distractors", "multi",
                  "camera+multi"],
    },
    "objects": {
        # How many EXTRA bodies a crowded clip holds, drawn per clip, for
        # `distractors` and `multi` alike.
        "extra_min": 3,
        "extra_max": 10,
        # `multi`: at least this many culprits, and always this many bodies
        # left lawful, so there is something to contrast against.
        "multi_culprits_min": 2,
        "multi_lawful_min": 1,
        # A distractor's size as a fraction of the actor's, how fast a MOVING
        # one goes as a fraction of the actor's speed, what share move at all,
        # and what share start airborne.
        "distractor_size": [0.35, 1.60],
        "distractor_speed": [0.25, 0.85],
        "distractor_moving": 0.5,
        "distractor_airborne": 0.35,
        # Clearance margins, in actor radii: how far a distractor stays from
        # the actor's path, and how much of its silhouette it must clear.
        "keep_clear_radii": 5.0,
        "sightline_radii": 1.35,
    },
    "camera": {
        # The three motions and how often each is chosen among moving clips.
        # A panning variant is deliberately absent -- with both the camera and
        # the aim moving, "did the object move or did the camera" stops being
        # answerable from the clip.
        "kinds": ["track", "orbit", "dolly"],
        "weights": [0.40, 0.40, 0.20],
        # How far a track or orbit travels, as a fraction of the standoff, and
        # how much a dolly changes its distance. Dolly is tighter because it is
        # the motion that changes APPARENT SIZE, which is the cue
        # `immutability` and `deformation` make their claim about.
        "travel": [0.10, 0.22],
        "dolly": [0.06, 0.12],
    },
    "difficulty": {
        # THE EASY / MODERATE / HARD CUTS, two per factor, in the order
        # [easy, moderate] and always stated in the easier-is-better direction
        # -- so `footprint` reads "easy at or above 0.05" and `occlusion`
        # "easy at or below 0.05". Which direction that is belongs to the
        # factor, not to the config, and lives in `annotate.difficulty`.
        #
        # THESE ARE FROZEN ONCE A RELEASE IS PUBLISHED. A benchmark whose
        # difficulty labels move between releases cannot be compared with
        # itself. They are editable because a future dataset with a different
        # geometry or a different window policy will want different cuts --
        # and because the resolved values ride in every `meta.json`, so a clip
        # always says what it was labelled under. Changing them is a new
        # release, not a bug fix.
        #
        # Four were fitted on the 431 invalid clips of the review corpus and
        # three were chosen from what the quantity means; which is which is
        # recorded per factor in `annotate/difficulty.py`.
        # `scripts/fit_difficulty.py` prints the tertiles of a corpus so a
        # refit starts from data rather than from taste.
        "footprint": [0.05, 0.012],
        "occlusion": [0.05, 0.5],
        "duration": [0.35, 0.15],
        "severity": [0.90, 0.40],
        "clutter": [2, 6],
        "culprits": [1, 3],
        "camera": [0.02, 0.12],
    },
    "materials": {
        # Divides every density. Anchors a median wooden actor near 1 kg, which
        # is the mass every body carried before materials existed -- so a
        # scenario stays in the contact regime its restitution and friction
        # were tuned in.
        "mass_scale": 250.0,
    },
}

#: Where the resolved values are written for the container to read.
FILENAME = "params.json"


def _merge(base: Dict[str, Any], over: Dict[str, Any]) -> Dict[str, Any]:
    """`over` wins, one level into each section, so a file may set one key."""
    out = {k: dict(v) if isinstance(v, dict) else v for k, v in base.items()}
    for section, values in (over or {}).items():
        if section not in out:
            raise KeyError("unknown params section %r; known: %s"
                           % (section, ", ".join(sorted(out))))
        if not isinstance(values, dict):
            raise TypeError("params section %r must be a mapping" % section)
        unknown = set(values) - set(out[section])
        if unknown:
            raise KeyError("unknown params key(s) in %r: %s"
                           % (section, ", ".join(sorted(unknown))))
        out[section] = dict(out[section], **values)
    return out


def resolved(over: Optional[Dict[str, Any]] = None,
             base: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """`base` (default: `DEFAULTS`) with `over` applied and validated.

    `base` lets the layers stack -- shipped defaults, then `common.yaml`, then
    a run's own `params:` block -- while every layer is checked against the
    known tree, so a typo in any of them is an error rather than a value that
    silently does nothing.
    """
    return _merge(base or DEFAULTS, over or {})


def write(values: Dict[str, Any], workdir: str) -> str:
    os.makedirs(workdir, exist_ok=True)
    path = os.path.join(workdir, FILENAME)
    with open(path, "w") as fh:
        json.dump(values, fh, indent=2, sort_keys=True)
    return path


def read(path: str) -> Dict[str, Any]:
    with open(path) as fh:
        return resolved(json.load(fh))


#: The values in force, for `meta.json` and for anything that wants to report
#: them. Replaced wholesale by `apply`.
CURRENT: Dict[str, Any] = resolved()


def apply(values: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Push `values` into the module constants that read them.

    Explicit rather than clever: every assignment below is a tunable finding
    its home, and a knob that is not listed here does nothing, which is the
    behaviour a reader should be able to verify by looking.
    """
    global CURRENT

    from .scenarios import _common as C
    from .scenarios import base as B
    from .scenarios import materials as M

    v = CURRENT = resolved(values)

    shares = v["ladder"]["shares"]
    for name, cx in B.COMPLEXITY.items():
        if name in shares:
            B.COMPLEXITY[name] = _replace(cx, share=float(shares[name]))

    B.CONDITION_CYCLE = tuple(v["conditions"]["cycle"])
    B.LEVEL_PHASE = {n: i for i, n in enumerate(B.COMPLEXITY)}

    o = v["objects"]
    B.EXTRA_OBJECTS = (int(o["extra_min"]), int(o["extra_max"]))
    B.MULTI_ACTORS = B.EXTRA_OBJECTS
    B.MULTI_CULPRIT_RANGE = (int(o["multi_culprits_min"]),
                             int(o["multi_lawful_min"]))
    C.DISTRACTOR_SIZE = tuple(o["distractor_size"])
    C.DISTRACTOR_SPEED = tuple(o["distractor_speed"])
    C.DISTRACTOR_MOVING = float(o["distractor_moving"])
    C.DISTRACTOR_AIRBORNE = float(o["distractor_airborne"])
    C.KEEP_CLEAR_RADII = float(o["keep_clear_radii"])
    C.SIGHTLINE_RADII = float(o["sightline_radii"])

    # The difficulty cuts, rebuilt as a whole table so a factor cannot end up
    # with one threshold from the config and one from the shipped default.
    from .annotate import difficulty as D

    cuts = v["difficulty"]
    D.FACTORS = tuple(
        _replace(f, easy=float(cuts[f.name][0]), moderate=float(cuts[f.name][1]))
        if f.name in cuts else f
        for f in D.FACTORS)
    D.BY_NAME = {f.name: f for f in D.FACTORS}

    c = v["camera"]
    B.CAMERA_MOTION_KINDS = tuple(c["kinds"])
    B.CAMERA_MOTION_WEIGHTS = tuple(float(x) for x in c["weights"])
    B.CAMERA_TRAVEL = tuple(float(x) for x in c["travel"])
    B.DOLLY_RANGE = tuple(float(x) for x in c["dolly"])

    M.MASS_SCALE = float(v["materials"]["mass_scale"])
    return v


def _replace(obj, **kw):
    import dataclasses

    return dataclasses.replace(obj, **kw)
