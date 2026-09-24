"""Each violator on its own clock, in clips with several.

A `multi` clip makes two or more of its actors violate. They used to share one
plan: one `t_event`, one window list, one severity timeline painted into every
violator -- so every violator in the clip broke the law on the same frame, and a
model could find all of them by finding one.

Now, for most such clips, each violator is planned on its own -- the family is
asked to act on that body alone, with an event draw keyed on it -- and the
plans are merged into one whose `violators` carry each body's own moment,
windows and magnitude. A share of clips (`MULTI_SYNC_SHARE`) keeps the shared
moment on purpose, so simultaneity is represented and countable rather than an
accident of the draw.

Only where splitting means something, and only where it can be honoured:

* the family acts on its requested number of bodies, not a coupled group it
  stages together -- `newton2_mass` exchanges momentum between two bodies and
  has no per-body version. Some families (notably `antigravity`) rank eligible
  actors and choose that many, so the selected ids need not be the random ids
  returned by `_group`;
* the plan STAGES, so the worker can apply each violator's intervention at its
  own frame in one simulation (`stepper.run_segments`);
* the violation is not scene-wide (`spatial_extent == "global"`) and not a
  granular medium, where "each grain on its own clock" is not a picture anyone
  can read.

Everything else keeps the shared plan, marked `violator_timing: "shared"`.

py3.9-compatible: runs inside the container.
"""
from __future__ import annotations

import zlib
from typing import Callable, List, Optional, Tuple

import numpy as np

from . import _geom
from .base import InterventionPlan

#: Share of eligible `multi` clips whose violators all fire at one moment.
#: Set from `params.objects.multi_sync_share`.
MULTI_SYNC_SHARE = 0.25

_MISSING = object()


def timing_mode(spec, family: str) -> str:
    """"sync" or "independent", drawn per (scene, family) off a salted stream."""
    key = (int(spec.seed) * 2654435761 + 0x5C7C
           + zlib.crc32(str(family).encode())) % (2 ** 31 - 1)
    u = float(np.random.RandomState(key).uniform())
    return "sync" if u < float(MULTI_SYNC_SHARE) else "independent"


def _dynamic_ids(spec, ids) -> List[int]:
    dynamic = {int(b.segmentation_id) for b in spec.bodies if not b.static}
    return [int(i) for i in ids if int(i) in dynamic]


def _seen_enough(spec, traj, body_id: int) -> bool:
    """Is this body on screen for enough of the clip to carry a violation?

    The share the worker's gate asks of a violator after its event
    (`_geom.VISIBLE_AFTER_SHARE`), over the whole stretch events are drawn from
    -- asked before any moment is drawn, so of the lawful rollout.
    """
    body = next((b for b in spec.bodies
                 if int(b.segmentation_id) == int(body_id)), None)
    if body is None:
        return False
    T = int(traj.num_frames)
    lo = max(1, int(round(_geom.EVENT_BAND[0] * T)))
    on = _geom.violators_on_screen(spec, traj, [body])[lo:]
    return bool(on.size and on.mean() >= _geom.VISIBLE_AFTER_SHARE)


def splittable(inj, spec, plan: InterventionPlan, staged: bool) -> bool:
    """Can this plan's violators be given moments of their own? See the module."""
    if "multi" not in str(getattr(spec, "condition", "") or ""):
        return False
    if not staged or plan.spatial_extent == "global":
        return False
    if getattr(spec, "physics_medium", "rigid") == "granular":
        return False
    group = inj._group(spec)
    violators = set(_dynamic_ids(spec, plan.causal_body_ids))
    # `_group` is the requested group SIZE. A family can select the actual
    # members from that many eligible actors (antigravity ranks airborne runs),
    # so requiring exact id equality incorrectly disabled its local per-body
    # timing. Each split sub-plan below is still checked to target exactly its
    # requested body; coupled plans fail that check and stay shared.
    return len(group) >= 2 and len(violators) == len(group)


def violator_plans(inj, spec, traj, make_rng: Callable[[], np.random.RandomState],
                  severity_bin: str,
                  is_staged: Callable[[InterventionPlan], bool]
                  ) -> Tuple[Optional[InterventionPlan], List[InterventionPlan]]:
    """(plan, sub_plans). `sub_plans` is empty unless violators got own moments.

    `make_rng` returns a FRESH generator each call, seeded as the worker seeds
    this (family, severity), so every sub-plan draws exactly as a standalone
    plan of that bin would. The scene's `family_targets` override is always
    restored: one scene serves every family in a worker run.
    """
    plan = inj.plan(spec, traj, make_rng(), severity_bin)
    if plan is None:
        return None, []
    if not splittable(inj, spec, plan, bool(is_staged(plan))):
        plan.notes.setdefault("violator_timing", "shared")
        return plan, []
    if timing_mode(spec, inj.family) == "sync":
        plan.notes["violator_timing"] = "sync"
        return plan, []

    # A PEER NOBODY CAN SEE IS NOT A VIOLATOR. The framing check lets a scene
    # keep a share of its actors out of shot, but a split plan is judged on
    # EVERY violator, so one peer spawned outside the frame declined the whole
    # clip at every event moment -- measured in the container on `drop` 786
    # (camera+multi) and `stack_topple` 785 (multi), each x solidity, where
    # every other violator was in plain view. Such a peer keeps its lawful
    # motion and simply is not asked to break the law. Too few left to split,
    # and the clip keeps its shared plan and the group's share rule.
    order = [bid for bid in _dynamic_ids(spec, plan.causal_body_ids)
             if _seen_enough(spec, traj, bid)]
    if len(order) < 2:
        plan.notes["violator_timing"] = "shared"
        return plan, []
    had_targets = "family_targets" in spec.notes
    targets = spec.notes.setdefault("family_targets", {})
    saved = targets.get(inj.family, _MISSING)
    subs: List[InterventionPlan] = []
    try:
        for bid in order:
            targets[inj.family] = [bid]
            with _geom.violator_context(bid):
                sub = inj.plan(spec, traj, make_rng(), severity_bin)
            if (sub is None or not is_staged(sub)
                    or _dynamic_ids(spec, sub.causal_body_ids)[:1] != [bid]):
                plan.notes["violator_timing"] = "shared"
                return plan, []
            subs.append(sub)
    finally:
        if saved is _MISSING:
            targets.pop(inj.family, None)
        else:
            targets[inj.family] = saved
        if not had_targets and not targets:
            spec.notes.pop("family_targets", None)

    merged = InterventionPlan.merge(subs, [int(s.causal_body_ids[0]) for s in subs])
    merged.notes["violator_timing"] = "independent"
    return merged, subs


def by_moment(subs: List[InterventionPlan]) -> List[InterventionPlan]:
    """Sub-plans in the order the worker stages them: earliest moment first,
    ties in the order the plan named its violators."""
    return [s for _, s in sorted(enumerate(subs),
                                 key=lambda p: (p[1].t_event, p[0]))]
