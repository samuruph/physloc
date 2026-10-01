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
from .base import InterventionPlan, ViolatorTiming

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


#: Families whose law is about how a body MOVES while it moves: applied to a
#: body at rest they change nothing a viewer can see. A `multi` member must be
#: moving at its own event moment to carry them -- `friction` handed to two
#: resting peers on `collision` camera+multi was an invisible clip with an
#: empty severity map.
NEEDS_MOTION = frozenset({"friction", "newton1_inertia", "time_slip",
                          "non_parabolic", "angular_momentum"})


def _main_ids(inj, spec) -> List[int]:
    """The scenario's own actors, the one it names for this family first."""
    named = [int(i) for i in
             ((spec.notes.get("family_targets") or {}).get(inj.family) or [])]
    own = [int(b.segmentation_id) for b in _geom.scene_actors(spec)]
    dynamic = set(_dynamic_ids(spec, named + own))
    return [i for i in dict.fromkeys(named + own) if i in dynamic]


def _multi_pool(inj, spec) -> Tuple[List[int], int]:
    """(candidates in order, how many violators the clip wants).

    The scenario's own actors first -- the clip is about the ball that strikes
    or the body that drops, and a `multi` clip whose violators were all extras
    left that body lawful in a clip labelled with its law -- then the extras
    the condition added, in an order fixed per scene and family, as backups for
    members the family cannot act on. The count is the condition's own draw
    (`n_violators_wanted`), never all actors.
    """
    main = _main_ids(inj, spec)
    peers = [int(b.segmentation_id) for b in _geom.actors(spec)
             if not b.dormant and not b.static
             and int(b.segmentation_id) not in main]
    order = inj._instance_rng(spec).permutation(len(peers)) if peers else []
    peers = [peers[int(i)] for i in order]
    total = len(main) + len(peers)
    want = int(spec.notes.get("n_violators_wanted")
               or max(2, len(inj._group(spec))))
    want = max(2, min(want, max(2, total - 1)))
    return main + peers, want


def _moving_at(traj, body_id: int, frame: int) -> bool:
    j = traj.index_of(int(body_id))
    f = int(np.clip(frame, 0, traj.num_frames - 1))
    return float(np.linalg.norm(traj.lin_vel[f, j])) > _geom.REST_SPEED


def violator_plans(inj, spec, traj, make_rng: Callable[[], np.random.RandomState],
                  severity_bin: str,
                  is_staged: Callable[[InterventionPlan], bool]
                  ) -> Tuple[Optional[InterventionPlan], List[InterventionPlan]]:
    """(plan, sub_plans). `sub_plans` is empty unless violators got own plans.

    In a `multi` clip every family is asked to act on each candidate body in
    turn, exactly as it would act on a lone actor, and the plans are merged:
    every violator breaks the SAME law through its own plan, at its own
    physical moment -- a super-elastic body rebounds at ITS impact, not at the
    main actor's. The scenario's own actor is always one of them; a family
    that cannot act on it, or on two bodies, keeps the single plan and the
    worker declines the clip.

    `make_rng` returns a FRESH generator each call, seeded as the worker seeds
    this (family, severity), so every sub-plan draws exactly as a standalone
    plan of that bin would. The scene's `family_targets` override is always
    restored: one scene serves every family in a worker run.
    """
    plan = inj.plan(spec, traj, make_rng(), severity_bin)
    if plan is None:
        return None, []
    if "multi" not in str(getattr(spec, "condition", "") or ""):
        plan.notes.setdefault("violator_timing", "shared")
        return plan, []
    # Scene-wide by nature: gravity for everyone, or a medium of grains.
    if (plan.spatial_extent == "global"
            or getattr(spec, "physics_medium", "rigid") == "granular"):
        plan.notes.setdefault("violator_timing", "shared")
        return plan, []
    sync = timing_mode(spec, inj.family) == "sync"
    earliest = max(1, int(round(_geom.EVENT_EARLIEST_SECONDS * _geom._fps(spec))))
    pool, want = _multi_pool(inj, spec)
    main = set(_main_ids(inj, spec))
    # Peers only: the scenario's own actor may be hidden ON PURPOSE --
    # `occluder_pass` puts it behind a screen -- and whether its violation can
    # be seen is the worker's gate to decide, which knows the declared
    # occlusion. Dropping it here declined every camera+multi `occluder_pass`.
    unseen = [bid for bid in pool
              if bid not in main and not _seen_enough(spec, traj, bid)]
    pool = [bid for bid in pool if bid not in unseen]

    had_targets = "family_targets" in spec.notes
    targets = spec.notes.setdefault("family_targets", {})
    saved = targets.get(inj.family, _MISSING)
    made: List[Tuple[int, InterventionPlan, bool]] = []
    import os as _os
    import sys as _sys
    debug = bool(_os.environ.get("PHYSLOC_DEBUG"))

    def say(msg):
        if debug:
            print("MULTI %s/%s %s" % (inj.family, severity_bin, msg),
                  file=_sys.stderr, flush=True)
    say("main=%s pool=%s want=%d unseen=%s" % (sorted(main), pool, want,
                                                unseen))
    try:
        for bid in pool:
            if len(made) >= want:
                break
            targets[inj.family] = [bid]
            # A sync clip draws every violator's moment from one stream, so
            # the bodies break the law together wherever the family's own
            # physics allows; otherwise each is keyed on its own body.
            ctx = (_geom.violator_context(None) if sync
                   else _geom.violator_context(bid))
            with ctx:
                sub = inj.plan(spec, traj, make_rng(), severity_bin)
            # A PAIR FAMILY (`newton2_mass`, `fusion`, `solidity` between two
            # moving bodies) may list the named body second: it only has to
            # be one of them. Two members whose plans would act on the same
            # body -- the two halves of one collision -- are one violation,
            # not two, so a member overlapping an accepted one is skipped.
            ids = _dynamic_ids(spec, sub.causal_body_ids) if sub else []
            if sub is None or bid not in ids:
                say("%d: %s" % (bid, "no plan" if sub is None else
                                "plan targets %s" % sub.causal_body_ids))
                continue        # this body cannot carry it; the next may
            taken = {int(i) for _, s, _ in made
                     for i in _dynamic_ids(spec, s.causal_body_ids)}
            if taken.intersection(ids):
                say("%d: overlaps %s" % (bid, sorted(taken.intersection(ids))))
                continue
            if (inj.family in NEEDS_MOTION
                    and not _moving_at(traj, bid, sub.t_event)):
                say("%d: at rest at %d" % (bid, sub.t_event))
                continue
            # A PEER HAS NO STORY OF ITS OWN to time an event by: its first
            # contacts are the settle after it is placed, and a violation
            # fired there -- `solidity` sinking peers on frames 1-4 -- starts
            # before anything lawful has been shown. The scenario's own actor
            # keeps its physical moment (the cube striking the pyramid at
            # frame 5); a peer waits for the band like any undirected event.
            if bid not in main and sub.t_event < earliest:
                say("%d: early %d" % (bid, sub.t_event))
                continue
            made.append((bid, sub, bool(is_staged(sub))))
            say("%d: ok t=%d staged=%s" % (bid, sub.t_event, made[-1][2]))
    finally:
        if saved is _MISSING:
            targets.pop(inj.family, None)
        else:
            targets[inj.family] = saved
        if not had_targets and not targets:
            spec.notes.pop("family_targets", None)

    # Staged and edited members may mix: the worker simulates the staged ones
    # and applies the edits on top (`render.worker._invalid_variant`).
    # VIOLATORS ARE BODIES, NOT PLANS. A pair family's single plan can make
    # two bodies break the law at once -- `superelastic` bouncing both cubes
    # of a `collision` -- and that is already a two-violator clip: returned
    # as it is, every body it acts on named a violator.
    if len(made) == 1 and any(b in main for b, _, _ in made):
        only = made[0][1]
        acted = [i for i in _dynamic_ids(spec, only.causal_body_ids)
                 if not next(b for b in spec.bodies
                             if int(b.segmentation_id) == i).dormant]
        if len(acted) >= 2:
            only.notes["violator_timing"] = "shared"
            say("one plan acting on %s" % acted)
            return only, []
    if len(made) < 2 or not any(b in main for b, _, _ in made):
        plan.notes["violator_timing"] = "shared"
        # Said plainly, for the worker's decline: which of the two things a
        # multi clip needs this family could not provide.
        plan.notes["multi_shortfall"] = (
            "%s can act on %d bod%s here (the scenario's own actor %s)"
            % (inj.family, len(made), "y" if len(made) == 1 else "ies",
               "included" if any(b in main for b, _, _ in made)
               else "not among them"))
        return plan, []
    owners = [b for b, _, _ in made]
    subs = [s for _, s, _ in made]
    merged = InterventionPlan.merge(subs, owners)
    merged.notes["violator_timing"] = "sync" if sync else "independent"
    _name_every_body(spec, merged, subs, owners)
    return merged, subs


def _name_every_body(spec, merged: InterventionPlan, subs, owners) -> None:
    """Every moving body a member's plan acts on is a named violator.

    A member is planned for ONE body, but its plan can act on two: a
    super-elastic collision rebounds both bodies, so on `drop` multi the peer a
    violator struck flew off at the gained speed while only the striker was
    labelled -- a law broken by a body the clip called lawful. The extra body
    gets the member's own timing.
    """
    dormant = {int(b.segmentation_id) for b in spec.bodies if b.dormant}
    named = {int(c.body_id) for c in merged.violators}
    for owner, s in zip(owners, subs):
        for bid in _dynamic_ids(spec, s.causal_body_ids):
            if bid in named or bid in dormant:
                continue
            named.add(bid)
            merged.violators.append(ViolatorTiming(
                body_id=int(bid), t_event=int(s.t_event),
                windows=[tuple(w) for w in s.windows],
                intervention_windows=[tuple(w) for w in s.intervention_windows],
                consequence_windows=[tuple(w) for w in s.consequence_windows],
                magnitude=float(s.magnitude),
                notes=dict(s.notes, acted_on_by=int(owner))))


def by_moment(subs: List[InterventionPlan]) -> List[InterventionPlan]:
    """Sub-plans in the order the worker stages them: earliest moment first,
    ties in the order the plan named its violators."""
    return [s for _, s in sorted(enumerate(subs),
                                 key=lambda p: (p[1].t_event, p[0]))]
