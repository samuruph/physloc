"""Injectors -- docs/PLAN.md Part 2 / Part 4.

An injector edits a **trajectory**, never the simulator's rng. That is what makes
the valid prefix bit-identical by construction rather than by hope: the invalid
trajectory literally *is* the valid one up to `t_event`.

Two-phase API, because an injector generally has to look at the valid rollout
before it can choose a sensible `t_event` (you cannot break a contact before the
contact happens):

    plan(spec, valid_traj, rng, severity_bin) -> InterventionPlan | None
    apply(spec, valid_traj, plan)             -> invalid Trajectory

py3.9-compatible: runs inside the container.
"""
from __future__ import annotations

import copy
import zlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..sim.trajectory import Trajectory
from . import _geom


@dataclass
class InterventionPlan:
    """What we are about to do, decided before we do it."""

    family: str
    kind: str                                   # instant | sustained | repeated
    t_event: int
    windows: List[Tuple[int, int]]              # inclusive [start, end] frame ranges
    causal_body_ids: List[int]
    params: Dict[str, Any]
    magnitude: float                            # the knob we turned -- exact
    magnitude_unit: str
    severity_bin: str
    spatial_extent: str = "local"               # local | global
    notes: Dict[str, Any] = field(default_factory=dict)

    #: When we are ACTIVELY changing something -- the colour ramping, the body
    #: shrinking, the teleport happening. Ends when the change is complete.
    intervention_windows: Optional[List[Tuple[int, int]]] = None
    #: When the scene differs from lawful AS A RESULT. Runs on after the
    #: intervention finishes, and for `permanence` never ends.
    consequence_windows: Optional[List[Tuple[int, int]]] = None

    def __post_init__(self) -> None:
        # A family that says nothing declares an intervention that lasts the
        # whole window, which is what every family meant before the split
        # existed.
        if self.intervention_windows is None:
            self.intervention_windows = list(self.windows)
        if self.consequence_windows is None:
            self.consequence_windows = list(self.windows)
        self.windows = self._detectable_windows()

    def _detectable_windows(self) -> List[Tuple[int, int]]:
        """`windows` is WHERE THE EVIDENCE IS, not the union of the two.

        It used to be the union, and that quietly contradicted the annotation
        shipped beside it. `annotate.pipeline` paints the mask and the severity
        on the *scored* window, which for a family whose evidence is the change
        itself (`detectable == "event"`) is the intervention: a cube that has
        finished turning green is an ordinary green cube, and no later frame
        contains anything to see. So `drop x colour_shift` declared
        `violation_windows [[7, 24]]` and shipped a mask with pixels on 7-12,
        and `validate` was right to call it -- for all seventeen `event`
        families, on every clip.

        Reporting the union asks a model to flag frames that provably contain
        no evidence, which is worse than the narrower claim being occasionally
        conservative. `consequence_windows` still runs to the end of the clip
        and still drives the blue `causal_mask` through
        `(consequence | diverged) & observable`, so a violation whose effects
        outlive it keeps saying so -- just not in the channel that means "look
        here".
        """
        from ..taxonomy import FAMILIES
        fam = FAMILIES.get(self.family)
        if fam is None:                       # a family the taxonomy has not
            return list(self.windows)         # heard of: leave it alone
        wins = (self.intervention_windows if fam.detectable == "event"
                else self.consequence_windows)
        return list(wins) if wins else list(self.windows)

    @property
    def t_end(self) -> int:
        return max(e for _, e in self.windows)

    @property
    def t_intervention_end(self) -> int:
        return max(e for _, e in self.intervention_windows)

    @property
    def t_consequence_end(self) -> int:
        return max(e for _, e in self.consequence_windows)

    def active_mask(self, num_frames: int) -> np.ndarray:
        """Rasterise `windows` into the per-frame boolean timeline (PLAN 3.2)."""
        a = np.zeros((num_frames,), dtype=bool)
        for s, e in self.windows:
            a[max(0, s):min(num_frames, e + 1)] = True
        return a

    def to_dict(self) -> Dict[str, Any]:
        return {
            "family": self.family, "kind": self.kind,
            "t_event_frame": self.t_event, "t_end_frame": self.t_end,
            "violation_windows": [list(w) for w in self.windows],
            "intervention_windows": [list(w) for w in self.intervention_windows],
            "consequence_windows": [list(w) for w in self.consequence_windows],
            "t_intervention_end_frame": self.t_intervention_end,
            "t_consequence_end_frame": self.t_consequence_end,
            "causal_body_ids": list(self.causal_body_ids),
            "spatial_extent": self.spatial_extent,
            "intervention": {
                "type": self.params.get("type", self.family),
                "params": {k: v for k, v in self.params.items() if k != "type"},
                "magnitude": self.magnitude,
                "magnitude_unit": self.magnitude_unit,
                "severity_bin": self.severity_bin,
            },
            "notes": self.notes,
        }


class Injector:
    """Base class. `family` must be a key in taxonomy.FAMILIES."""

    family: str = "unnamed"

    #: Set by the caller to request a uniform violation duration, in frames.
    #: Honoured by `sustained` families; `instant` ones ignore it, because
    #: stretching a teleport's window would conflate the law breach with its
    #: consequences -- the object stays displaced afterwards, but the
    #: continuity law is broken only at the jump.
    window_frames: Optional[int] = None

    #: The violation is a *state*, not an event: once it starts it holds to the
    #: end of the clip and `--window` does not apply. A detached shadow does not
    #: reattach; a hovering body does not settle; a body that split stays split.
    #: Truncating the window on these families does not shorten the violation,
    #: it just stops annotating the tail of it -- which is how `shadow` came to
    #: ship a detached shadow across frames `active` said were lawful.
    persistent: bool = False

    @staticmethod
    def _frames_for(spec, seconds: float, minimum: int = 2) -> int:
        """A duration written in SECONDS, converted to this tier's frames.

        The tiers differ by 2.5x in frame rate, so any constant expressed in
        frames describes a different violation at each of them: a five-frame
        ramp is 0.42 s of visible distortion at the debug tier and 0.17 s at
        v0, and a ten-frame pass-through is 0.83 s against 0.33 s -- which is
        the difference between a ball that clears a wall and a ball that gets
        ejected back out of it. Anything a viewer perceives as a duration
        belongs in seconds; only counts that are genuinely about frames (a
        single-frame teleport) stay in frames.
        """
        return int(max(minimum, round(float(seconds) * float(spec.tier.fps))))

    @staticmethod
    def _split_windows(t0: int, T: int, applied_frames: int):
        """(union, intervention, consequence) for "change for N frames, then it
        stays changed".

        The shape most of the taxonomy has and the one the released windows got
        wrong: a colour that finishes turning green at frame 12 is not still
        *being changed* at frame 24, even though the scene is still unlawful
        there. Callers that genuinely change something for the whole clip --
        `antigravity` holds a body up every frame it is airborne -- keep the
        default and declare nothing.
        """
        end = min(T - 1, t0 + max(1, int(applied_frames)) - 1)
        union = [(t0, T - 1)]
        return union, [(t0, end)], [(t0, T - 1)]

    def _window_len(self, default: int, t0: int, num_frames: int) -> int:
        if self.persistent:
            return max(1, num_frames - t0)
        n = self.window_frames or default
        return max(1, min(int(n), num_frames - t0))

    def plan(self, spec, traj: Trajectory, rng: np.random.RandomState,
             severity_bin: str) -> Optional[InterventionPlan]:
        raise NotImplementedError

    #: True when this family can express itself as a change the SIMULATOR
    #: honours, so the worker re-runs PyBullet from `t_event` instead of
    #: re-integrating the edited trajectory by hand. See `stage`.
    simulated = False

    def simulates(self, plan: InterventionPlan) -> bool:
        """Whether THIS plan takes the simulated path.

        Per plan, not per class: `solidity` stages a two-body pass-through as a
        disabled collision pair, but its granular `sink_group` mode removes the
        floor under forty grains at once, which is a scene edit rather than one
        pair. A class-level flag would have sent that plan down the simulated
        path to a stage() that does nothing, and the violation would simply not
        have happened.
        """
        return self.simulated

    def available_at(self, spec) -> bool:
        """Can this family express itself in THIS scene at all?

        Not "did it decline this seed" -- that is `plan` returning None, and a
        seed retry answers it. This is the stronger claim that no seed will
        work, because the rung itself has removed what the family acts on.

        One family needs it today: `colour_shift` has nothing to shift once
        actors are scanned GSO assets, which carry a texture rather than a flat
        colour. Without this the cell was still walked, and it produced a fully
        annotated clip in which nothing changed -- `sev=0.00` with a complete
        set of labels, which is worse than a missing clip.
        """
        return True

    def revives(self, spec, plan: InterventionPlan) -> frozenset:
        """Scripted causal bodies this family will make DYNAMIC when it stages.

        The worker refuses the staged path for a plan whose causal body is
        `scripted`, and it is right to: such a body is pinned at mass 0, so
        resetting the world and running forward leaves it sitting exactly where
        it was, and five families on `pendulum_swing` all shipped the same
        picture of a stopped pendulum that way.

        `fission` is the exception the rule did not know about. Its understudy
        is *deliberately* scripted -- parked and invisible from frame 0 -- and
        `ShapeSwap(dynamic=True)` exists precisely to stand a genuinely dynamic
        proxy in its place. So the blanket guard silently sent the one family
        equipped to overcome it down the edited path instead, where the halves
        were re-integrated by hand and came to rest 0.41 m in the air: the
        floating cones you reported. Nothing failed and nothing was logged --
        `_apply` is a legitimate fallback, just not this family's.

        Declared here rather than inferred, so the worker asks the injector
        what it is going to do instead of guessing from the spec.
        """
        return frozenset()

    def stage(self, spec, simulator, objs, plan: InterventionPlan):
        """Apply the intervention to the live physics world at `t_event`.

        Returns a list of per-step hooks (possibly empty). The world has already
        been reset to the valid state at `plan.t_event`, and the caller runs
        PyBullet forward afterwards, so anything set here is the *initial
        condition* of the violation rather than a prescribed outcome.

        The point of the seam. A family that stages itself gets real contacts:
        a teleported body genuinely misses the wall it was moved past, two balls
        genuinely touch before they exchange momentum, and a pass-through is a
        disabled collision pair rather than a written-in overlap.
        """
        raise NotImplementedError

    def unstage(self, spec, simulator, objs, plan: InterventionPlan) -> None:
        """Undo whatever `stage` changed about the world.

        **Not optional.** One scene serves every family and every severity in a
        worker run, and a `changeDynamics` or a disabled collision pair persists
        exactly the way a stray keyframe does. Three separate cross-family leaks
        have already been traced to state nobody reset; this is the same hazard
        one layer down.
        """
        return None

    def refine_windows(self, spec, traj_valid: Trajectory,
                       traj_invalid: Trajectory,
                       plan: InterventionPlan) -> None:
        """Correct the plan's windows once the trajectory actually exists.

        `plan()` has to guess durations, because it runs before anything has
        happened. For most families a fraction of the clip is the right guess.
        For a family whose window has a *geometric* end -- `solidity` is over
        when the bodies stop overlapping, not after N frames -- the honest
        length can only be measured afterwards, and the worker writes plan.json
        after this runs.

        Mutates `plan` in place; default does nothing.
        """
        return None

    def post_simulate(self, spec, traj_valid: Trajectory,
                      traj_invalid: Trajectory,
                      plan: InterventionPlan) -> Trajectory:
        """Non-physical channels, applied after the re-simulation.

        PyBullet knows nothing about colour, opacity or a body's declared
        presence, so a family whose violation has both a physical and a visual
        part writes the visual part here. Most simulated families need nothing.
        """
        traj_invalid.meta = dict(traj_invalid.meta)
        traj_invalid.meta["intervention"] = plan.to_dict()
        traj_invalid.meta["label"] = "invalid"
        return traj_invalid

    def _apply(self, spec, traj: Trajectory, plan: InterventionPlan) -> Trajectory:
        """Family-specific edit. Implemented by every subclass."""
        raise NotImplementedError

    def apply(self, spec, traj: Trajectory, plan: InterventionPlan) -> Trajectory:
        """`_apply`, then the two things every family owes the clip.

        A template method rather than a rule each injector has to remember,
        because both of these were being got wrong silently and identically
        everywhere.
        """
        out = self._apply(spec, traj, plan)
        out = self._settle_bystanders(spec, traj, out, plan)
        # The contact list belongs to the trajectory it describes. `_clone`
        # copies the simulator's, and nothing re-runs the simulator, so without
        # this an invalid clip claims contacts from a rollout that no longer
        # happened -- including the very collision the intervention prevented.
        out.contacts = _geom.geometric_contacts(spec, out)
        out.meta = dict(out.meta)
        out.meta["contacts"] = "geometric"
        return out

    #: Speed change, in m/s, big enough to count as "this body was pushed".
    BYSTANDER_DV = 0.15
    #: Frames either side of a contact that still count as causing it.
    BYSTANDER_SLACK = 2

    def _settle_bystanders(self, spec, traj: Trajectory, out: Trajectory,
                           plan: InterventionPlan) -> Trajectory:
        """Stop uninvolved bodies reacting to collisions that no longer happen.

        An injector edits the culprit and leaves everyone else on their original
        path. When the edit prevents a collision, the body that *was* going to
        be struck still departs on schedule, hit by nothing -- which is a second,
        unlabelled violation in a clip that claims one.

        Found in `collision x fission`: the striker splits at frame 9 and both
        halves go elsewhere, yet the target still starts moving at frame 12 at
        exactly the speed the original impact would have given it.

        The rule is the one physics already implies: a body accelerates only if
        something touches it. So any non-culprit whose speed jumps while nothing
        is in contact with it gets re-integrated from just before the jump,
        carrying on with whatever it was lawfully doing -- which, for a ball
        waiting to be hit, is nothing at all.
        """
        # A family that changes no POSE and removes no BODY cannot have made
        # anything else move without cause. `colour_shift` and `deformation`
        # edit colour, opacity and scale and leave every trajectory untouched --
        # yet the guard was re-settling the ball their culprit lawfully struck,
        # displacing it by 0.44 m in a clip whose only claim is that a ball
        # changed colour.
        #
        # `dissolve` and `permanence` DO belong here even though they move
        # nothing, because a body that stops existing stops striking things.
        if not self._changes_dynamics(traj, out, plan):
            return out

        culprits = {int(b) for b in plan.causal_body_ids}
        # Scripted bodies are driven by a constraint the seam does not model --
        # a pendulum bob accelerates every frame with nothing touching it, which
        # is exactly right for a body on a string and exactly what this guard
        # must not "correct".
        movers = [b for b in spec.bodies
                  if not b.static and not b.scripted
                  and int(b.segmentation_id) not in culprits]
        if not movers:
            return out

        contacts = _geom.geometric_contacts(spec, out)
        T = out.num_frames
        for body in movers:
            bid = int(body.segmentation_id)
            bi = out.index_of(bid)
            bad = np.flatnonzero(self._uncaused_frames(
                out, spec, contacts, bid, from_frame=max(1, plan.t_event)))
            if not bad.size:
                continue
            t0 = int(bad[0])
            # Obstacles come from the EDITED trajectory, not the spec. In
            # `pyramid_impact x solidity` the struck balls have already sunk
            # through the floor by the time the cube arrives, so re-integrating
            # against the scene as declared lands the cube on a pyramid that is
            # no longer there -- swapping one uncaused motion for another.
            self._rewrite_from(spec, traj, out, body, t0,
                               v0=np.asarray(out.lin_vel[t0 - 1, bi], np.float64),
                               p0=np.asarray(out.pos[t0 - 1, bi], np.float64),
                               obstacles=_geom.Obstacles(spec, out,
                                                         exclude_ids=[bid]))
        return out

    @staticmethod
    def _changes_dynamics(traj: Trajectory, out: Trajectory,
                          plan: InterventionPlan) -> bool:
        """Did the intervention change a culprit's motion, or remove it?

        Pose, orientation, both velocities -- and `present`, because a body that
        ceases to exist stops striking whatever it was about to strike, which is
        every bit as much a dynamics change as moving it.
        """
        for bid in plan.causal_body_ids:
            try:
                j = traj.index_of(int(bid))
            except Exception:                                 # noqa: BLE001
                continue
            for attr in ("pos", "quat", "lin_vel", "ang_vel"):
                a = np.asarray(getattr(traj, attr)[:, j], np.float64)
                b = np.asarray(getattr(out, attr)[:, j], np.float64)
                if a.shape == b.shape and float(np.abs(a - b).max()) > 1e-9:
                    return True
            if not np.array_equal(traj.present[:, j], out.present[:, j]):
                return True
        return False

    @staticmethod
    def _uncaused_frames(traj: Trajectory, spec, contacts, body_id: int,
                         from_frame: int = 1) -> np.ndarray:
        """Frames where a body's velocity changes by more than gravity explains,
        with no dynamic body touching it.

        Two corrections, both of which the first version got wrong and both of
        which matter:

        **Gravity is a cause.** A body in free flight gains `g*dt` -- 0.82 m/s a
        frame at 12 fps -- with nothing touching it, and that is the most lawful
        motion there is. Comparing raw speed change flagged every falling grain
        in `pour`. The test is on the residual acceleration, `dv - g*dt`.

        **Static geometry is not a cause.** A floor can hold a body up or slow it
        down; it cannot accelerate one. A resting ball that suddenly departs is
        unexplained even though it has been in contact throughout, and a
        detector that accepted any contact at all reported nothing.
        """
        T = traj.num_frames
        bi = traj.index_of(int(body_id))
        v = np.asarray(traj.lin_vel[:, bi, :], np.float64)
        g_step = np.asarray(traj.gravity, np.float64) * traj.dt
        dv = np.zeros_like(v)
        dv[1:] = v[1:] - v[:-1] - g_step[None, :]
        residual = np.linalg.norm(dv, axis=1)

        dyn = {int(b.segmentation_id) for b in spec.bodies
               if not b.static and not b.scripted}
        touching = np.zeros(T, bool)
        if len(contacts):
            a = np.asarray(contacts.body_a, int)
            b = np.asarray(contacts.body_b, int)
            m = (((a == body_id) & np.isin(b, list(dyn)))
                 | ((b == body_id) & np.isin(a, list(dyn))))
            if m.any():
                touching[np.clip(np.asarray(contacts.frame, int)[m], 0, T - 1)] = True
        k = Injector.BYSTANDER_SLACK
        pad = np.concatenate([np.zeros(k, bool), touching, np.zeros(k, bool)])
        wide = pad.copy()
        for s in range(1, k + 1):
            wide[s:] |= pad[:-s]
            wide[:-s] |= pad[s:]
        touching = wide[k:k + T]

        # A body resting on a surface is held there by a normal force gravity
        # does not account for, so its residual is ~g*dt every frame it sits
        # still. Only count frames where it is actually gaining speed.
        speed = np.linalg.norm(v, axis=1)
        gaining = np.zeros(T, bool)
        gaining[1:] = speed[1:] > speed[:-1] + 1e-6

        bad = (residual > Injector.BYSTANDER_DV) & ~touching & gaining
        bad[:max(1, from_frame)] = False
        return bad

    def strong_residual_reference(self, spec) -> float:
        """The residual this family produces at its `strong` bin, in the units of
        its law. This is `r_strong` in the bounded score of PLAN 3.4 step 3, and
        it is what makes severity comparable across families whose physical
        units are not.

        Families whose knob is exact by construction -- a prescribed penetration
        depth, a gravity scale -- can answer from the spec alone. Families whose
        effect depends on the rollout (how fast was it going when it stopped?)
        cannot, so they measure it instead: `plan()` builds the strong-bin edit,
        runs the law over it, and records the answer as `notes["r_strong"]`,
        which the annotation pipeline prefers over this method. Measuring it
        even while planning a *weak* clip is the point -- the reference has to be
        the same for all three bins or the bins are not comparable.
        """
        raise NotImplementedError

    def _measure(self, traj: Trajectory, body_id: int, law_name: str,
                 ctx: Dict[str, Any]) -> float:
        """Peak residual of `law_name` on this trajectory. See above."""
        from ..residuals import laws
        r = laws.get(law_name)(traj, traj.index_of(int(body_id)), dict(ctx))
        return float(np.max(r)) if r.size else 0.0

    # -- helpers shared by subclasses -------------------------------------
    @staticmethod
    def _clone(traj: Trajectory) -> Trajectory:
        out = copy.copy(traj)
        out.pos = traj.pos.copy()
        out.quat = traj.quat.copy()
        out.lin_vel = traj.lin_vel.copy()
        out.ang_vel = traj.ang_vel.copy()
        out.present = traj.present.copy()
        out.scale_mul = traj.scale_mul.copy()
        out.colour = traj.colour.copy()
        out.opacity = traj.opacity.copy()
        out.meta = dict(traj.meta)
        return out

    # -- who the intervention acts on -------------------------------------
    def _instance_rng(self, spec) -> np.random.RandomState:
        """A generator keyed to the *scene*, not to the severity bin.

        `plan()` is handed an rng seeded per (family, severity) so that adding a
        family to a batched run cannot perturb the others. That is right for
        anything the bin should vary, and wrong for anything it must not: a
        choice like "does this body grow or shrink" has to come out the same for
        weak, medium and strong, or the three bins of one cell are three
        different violations and their magnitudes stop being comparable.
        """
        key = (int(spec.seed) * 2654435761
               + zlib.crc32(self.family.encode())) % (2 ** 31 - 1)
        return np.random.RandomState(key)

    def _primary(self, spec):
        """The one body most families act on. `None` if the scenario has no actor.

        A scenario may override the choice per family through
        `notes["family_targets"]`, which is how a scene says something about
        itself that the injector could not work out: `pyramid_impact` points
        `solidity` at a struck sphere rather than the falling cube, and
        `pour` points every single-grain family at the grains that
        start highest, because any other grain spends the clip buried and its
        violation would be perfectly scored and completely invisible.

        Dormant understudies are skipped: they exist in the scene graph so
        `fission` has something to switch on, and until it does they are not
        part of the scene.
        """
        want = (spec.notes.get("family_targets") or {}).get(self.family)
        if want:
            by_id = {int(b.segmentation_id): b for b in spec.bodies}
            preferred = [by_id[int(i)] for i in want if int(i) in by_id]
            if preferred:
                return preferred[0]
        live = [b for b in _geom.actors(spec) if not b.dormant]
        return live[0] if live else None

    def _group(self, spec):
        """The bodies this family should act on: one, or several.

        A violation applied to one grain of forty is perfectly annotated and
        impossible to see -- the mask is a handful of pixels somewhere in a
        pile. Scenes that are made of many interchangeable bodies say so, and
        families that can act on a set do.

        Two ways a scenario can ask. `family_targets[family]` names the bodies
        outright, which is how `pour` steers single-body families onto
        the grains that end up on top of the pile rather than buried in it.
        `group_fraction` asks for a share of the actors, chosen per scene so the
        three severity bins act on the same bodies.
        """
        live = self._all_actors(spec)
        if not live:
            return []
        named = (spec.notes.get("family_targets") or {}).get(self.family)
        if named:
            by_id = {int(b.segmentation_id): b for b in spec.bodies}
            chosen = [by_id[int(i)] for i in named if int(i) in by_id]
            if chosen:
                return chosen
        frac = float(spec.notes.get("group_fraction") or 0.0)
        # TWO, not four. The threshold used to be four live actors, from when
        # the only caller was `pour` asking for a share of forty grains -- but
        # the `multi` condition draws its object count from three upward, and
        # at N = 3 the guard silently handed back ONE culprit on a clip whose
        # metadata said two. A fraction is only ever set deliberately (by a
        # scenario, or by the condition), so where one exists it should be
        # honoured rather than second-guessed.
        if frac <= 0.0 or len(live) < 2:
            return live[:1]
        k = int(np.clip(round(frac * len(live)), 2, len(live)))
        picks = self._instance_rng(spec).choice(len(live), size=k, replace=False)
        return [live[i] for i in sorted(int(x) for x in picks)]

    def _medium_event_frame(self, spec, traj, bodies, fallback):
        """Fire on a MEDIUM while it is still in the air, not once it has piled up.

        A scene of many interchangeable falling bodies has a moment, and it is
        not "a third of the way in". Measured on `pour` at the debug tier, the
        grains reach the floor between frames 3 and 8 of 25 -- so every family
        whose event frame came from a fraction of the clip was intervening on a
        pile that had already settled. You reported the consequences one by
        one: `continuity` teleporting grains that were lying on the floor,
        `newton1_inertia` stopping bodies that had already stopped,
        `global_gravity` bending gravity for a pile at rest, `newton2_mass`
        exchanging momentum between two grains in a heap. A pour that
        misbehaves in flight is visible; a pile that twitches is not.

        Only for a genuine medium -- two bodies or fewer keep whatever moment
        the family chose, so no single-actor scenario is affected.
        """
        if len(bodies) <= 2:
            return fallback
        t = _geom.before_medium_lands(spec, traj, bodies)
        if t is None or not (1 <= t < traj.num_frames - 2):
            return fallback
        return int(t)

    @staticmethod
    def _medium_radius(traj, bodies, t0: int) -> float:
        """The size of the THING, when the thing is made of many bodies.

        A single actor's bounding radius is the right reference length for a
        family that acts on a single actor. For a medium it is the wrong one by
        an order of magnitude -- a pour is a decimetre-scale object made of
        centimetre-scale grains -- so any displacement written in radii comes
        out invisible. Returns the RMS spread of the group about its own
        centroid, floored at the largest member so a two-body group is never
        smaller than its parts.
        """
        idx = []
        for b in bodies:
            try:
                idx.append(traj.index_of(int(b.segmentation_id)))
            except Exception:                                 # noqa: BLE001
                continue
        biggest = max([float(b.bounding_radius) for b in bodies] or [0.1])
        if len(idx) < 2:
            return biggest
        f = int(np.clip(t0, 0, traj.num_frames - 1))
        p = np.asarray(traj.pos[f, idx, :], np.float64)
        spread = float(np.sqrt(np.mean(np.sum(
            (p - p.mean(axis=0)[None, :]) ** 2, axis=1))))
        return max(biggest, spread)

    @staticmethod
    def _all_actors(spec):
        """Every actor, for the families that act on a whole medium rather than
        on one culprit -- `antigravity` over a pour, `global_gravity` over a
        scene, an assembly whose parts must move together."""
        return [b for b in _geom.actors(spec) if not b.dormant]

    @staticmethod
    def _ballistic(p0: np.ndarray, v0: np.ndarray, g: np.ndarray,
                   dt: float, n: int) -> Tuple[np.ndarray, np.ndarray]:
        """Free-flight continuation: `n` frames of pos/vel starting one step after p0."""
        t = (np.arange(1, n + 1, dtype=np.float64) * dt)[:, None]
        pos = p0[None, :] + v0[None, :] * t + 0.5 * g[None, :] * t ** 2
        vel = v0[None, :] + g[None, :] * t
        return pos.astype(np.float32), vel.astype(np.float32)

    @staticmethod
    def _ballistic_with_floor(p0: np.ndarray, v0: np.ndarray, g: np.ndarray,
                              dt: float, n: int, floor_z, radius: float,
                              restitution: float, substeps: int = 24):
        """Free flight that still respects the ground.

        Needed after a violation window closes: the actor is legal again, so it
        must bounce rather than sink through on a later frame. Sub-steps so the
        bounce instant is not quantised to the frame rate.

        `floor_z` may be a callable (x, y) -> z. Scenarios with a raised surface
        need it: a mug kicked off a table must fall to the floor once it clears
        the edge, not glide along an infinite plane at table height, which is a
        solidity violation nobody asked for.
        """
        return Injector._integrate_profile(
            p0, v0, np.tile(np.asarray(g, np.float64)[None, :], (n, 1)),
            dt, floor_z, radius, restitution, substeps)

    @staticmethod
    def _integrate_profile(p0: np.ndarray, v0: np.ndarray, g_per_frame: np.ndarray,
                           dt: float, floor_z, radius: float,
                           restitution: float, substeps: int = 24,
                           obstacles=None, t_start: float = 0.0,
                           friction: float = 0.0):
        """Integrate under a *time-varying* gravity, respecting a ground plane.

        Two things this buys over plain ballistics:

        * the floor stays solid. An injector that only meant to change gravity
          must not also drop the actor through the table -- that would put two
          violations in one clip and annotate only one of them.
        * gravity may differ per frame, so an intervention can ramp in and out
          instead of being a step that lasts to the end of the clip. That is
          what makes the residual -- and therefore the severity map -- vary over
          time rather than sitting at one value.
        * **a body on the ground slows down.** Without this a shoved body slides
          at constant speed for the rest of the clip and leaves frame, which is
          not what the simulator does and not what the render shows -- but it IS
          what `_fit_to_frame` measures, because the fit is decided on this
          integrator. So every family that keeps its actor in shot by weakening
          itself was weakening itself against a body that never stops. Measured
          on `drop x fission`: the fit clamped to 0.34 of the nominal push, and
          the staged halves it produced came to rest 0.50 m apart when they
          needed 0.78 m to be two objects rather than one blurred one.
        """
        p = np.asarray(p0, np.float64).copy()
        v = np.asarray(v0, np.float64).copy()
        n = int(g_per_frame.shape[0])
        h = dt / float(substeps)
        ground = floor_z if callable(floor_z) else (lambda x, y: float(floor_z))
        pos = np.zeros((n, 3), np.float64)
        vel = np.zeros((n, 3), np.float64)
        for f in range(n):
            g = np.asarray(g_per_frame[f], np.float64)
            for k in range(substeps):
                v = v + g * h
                p = p + v * h
                fz = ground(p[0], p[1])
                if p[2] - radius <= fz and v[2] < 0.0:
                    p[2] = fz + radius
                    v[2] = -v[2] * restitution
                    if abs(v[2]) < 0.05:
                        v[2] = 0.0
                # The contact band has to allow for the bounce completing
                # between substeps: a body settling on the floor spends most of
                # them a little above it, and an exact test finds it in contact
                # almost never -- which is how the first version of this changed
                # nothing at all.
                if friction > 0.0 and (p[2] - radius) <= fz + max(
                        2e-3, abs(float(v[2])) * h):
                    speed = float(np.linalg.norm(v[:2]))
                    if speed > 1e-9:
                        drop = friction * abs(float(g[2])) * h
                        v[:2] *= max(0.0, 1.0 - drop / speed)
                if obstacles is not None:
                    p, v = obstacles.resolve(
                        p, v, radius, restitution,
                        t_start + f + (k + 1) / float(substeps))
            pos[f] = p
            vel[f] = v
        return pos.astype(np.float32), vel.astype(np.float32)

    # ------------------------------------------------------------------ #
    def _offscreen_frames(self, spec, traj: Trajectory, bodies,
                          from_frame: int, visible_fraction: float = 0.6) -> int:
        """How many frames after `from_frame` lose sight of the culprit.

        A frame counts as lost when fewer than `visible_fraction` of the culprit
        bodies are inside the frustum, so one grain of forty drifting out of a
        `pour` is not treated the same as the whole pour leaving.
        """
        idx = [traj.index_of(int(b.segmentation_id)) for b in bodies]
        if not idx:
            return 0
        pts = traj.pos[from_frame:, idx, :]
        # Per frame, because the camera may be moving. This block is the sole
        # visibility measure behind the fit ladder, so testing it against one
        # pose would fit every intervention to a frustum a fifth of the clips
        # do not have.
        vis = _geom.in_frame(spec, pts, from_frame=from_frame,
                             num_frames=int(traj.num_frames))
        # Only bodies that are actually in the scene count. A dormant
        # understudy is parked far below the world until something summons it,
        # so including it made the *valid* baseline "half the culprits are off
        # camera on every frame" -- which set the budget so high that `fission`
        # passed any separation at all and flung its halves clean out of frame.
        here = np.asarray(traj.present[from_frame:, idx], bool)
        counted = here.sum(axis=1)
        seen = (vis & here).sum(axis=1)
        frac = np.where(counted > 0, seen / np.maximum(counted, 1), 1.0)
        return int((frac < visible_fraction).sum())

    #: Fractions of the nominal intervention to try, strongest first.
    #:
    #: Finer than the 1.0 / 0.72 / 0.52 / 0.36 / 0.24 it started as, and that
    #: coarseness was costing real strength: a clip whose geometry could host
    #: 0.70 of the nominal shove was offered 0.72, failed it by one frame, and
    #: got 0.52 -- a quarter of the violation thrown away to a rounding of the
    #: search grid. You reported `phantom_impulse` as barely visible at its
    #: strongest bin, which is what that looks like from the outside.
    FIT_LADDER = (1.0, 0.90, 0.80, 0.71, 0.63, 0.55, 0.48, 0.41, 0.34, 0.28)

    def _fit_to_frame(self, spec, traj: Trajectory, bodies, t0: int, knob,
                      build, tolerance: int = 1, ladder=None):
        """Weaken `knob` until the culprit stays on screen, and return what stuck.

        The severity bins are chosen for visual legibility on a typical clip,
        but "typical" is a per-seed claim: the same reversed gravity that makes
        a nicely rising ball in one sample sends it out of the top of the frame
        in another with a higher toss. Rather than tuning the constants down
        until nothing ever escapes -- which makes every clip weaker to fix a
        few -- each clip keeps the strongest setting its own geometry allows.

        `magnitude` then reports what was actually applied, so it stays exact,
        and the severity a clip earns honestly reflects the smaller
        intervention. `build(scale)` returns a candidate trajectory.

        Callers fit on the **strong** bin and apply the resulting scale to
        whichever bin they are planning. Fitting each bin separately clamps
        `strong` hardest -- it is the one that overshoots -- and can leave it
        weaker than `medium`, which would make the bin labels a lie. One scale
        per clip says something defensible instead: *this* geometry supports at
        most this fraction of the nominal intervention, and the three bins keep
        their spacing inside it.
        """
        # Measured against the valid twin, not against zero. If the scenario
        # already lets the actor drift out of shot, weakening the intervention
        # cannot fix that, and clamping to the floor of the ladder would turn a
        # framing problem into a violation nobody can see.
        ladder = self.FIT_LADDER if ladder is None else ladder
        budget = self._offscreen_frames(spec, traj, bodies, t0) + tolerance
        candidate = None
        for scale in ladder:
            candidate = build(scale)
            if self._offscreen_frames(spec, candidate, bodies, t0) <= budget:
                return scale, candidate
        return ladder[-1], candidate

    def _fit_window_to_frame(self, spec, traj: Trajectory, bodies, t0: int,
                             n_win: int, build, tolerance: int = 1,
                             floor: int = 3) -> int:
        """Shorten a window until the culprit stays in shot, keeping its strength.

        The counterpart to `_fit_to_frame`, and the right one whenever the bin
        is a *qualitative* claim. Weakening the knob to keep a body on screen
        turns "gravity reverses" into "gravity is slightly reduced" -- the clip
        no longer shows what its label says, which is the failure the user
        reported on `antigravity`. Shortening the window keeps the reversal and
        just gives it less time to carry the body out of frame.

        Never goes below `floor` frames, because `_pulse` needs a ramp on each
        side and a plateau between them: shorter than three and the profile
        degenerates into a step, losing the shape that makes `severity_map` a
        field rather than a flag.
        """
        budget = self._offscreen_frames(spec, traj, bodies, t0) + tolerance
        for n in range(int(n_win), floor - 1, -1):
            if self._offscreen_frames(spec, build(n), bodies, t0) <= budget:
                return n
        return floor

    # ------------------------------------------------------------------ #
    def _rewrite_from(self, spec, traj: Trajectory, out: Trajectory, body,
                      t0: int, v0=None, p0=None, g_per_frame=None,
                      restitution=None, obstacles=None, solid=True,
                      floorless=False) -> None:
        """Re-integrate one body from frame `t0`, on the surface it belongs to.

        The workhorse behind most families: an intervention is usually "change
        the state at one instant, then let physics resume", and doing that by
        hand in each injector is how a clip ends up with two violations and one
        annotation.
        """
        bi = traj.index_of(int(body.segmentation_id))
        n = traj.num_frames - t0
        if n <= 0:
            return
        if g_per_frame is None:
            g_per_frame = np.tile(traj.gravity.astype(np.float64)[None, :], (n, 1))
        start_v = traj.lin_vel[t0 - 1, bi] if v0 is None else np.asarray(v0, np.float64)
        # `p0` exists for `fission`: the understudy has to start from the
        # *original's* last lawful pose, and writing that into frame t0-1 would
        # break prefix identity on the frame before the event.
        start_p = traj.pos[t0 - 1, bi] if p0 is None else np.asarray(p0, np.float64)
        pos, vel = self._integrate_profile(
            start_p, start_v, np.asarray(g_per_frame, np.float64),
            traj.dt, (self._no_floor(spec, body) if floorless
                      else _geom.floor_fn(spec, body)), float(traj.radius[bi]),
            float(body.restitution if restitution is None else restitution),
            obstacles=self._obstacles(spec, traj, body, obstacles, solid),
            t_start=float(t0 - 1),
            friction=float(getattr(body, "friction", 0.0)))
        out.pos[t0:, bi, :] = pos
        out.lin_vel[t0:, bi, :] = vel

    @staticmethod
    def _obstacles(spec, traj, body, given, solid):
        """The scene as seen by a body being re-integrated.

        `solid=False` is for the one family that means to pass through things --
        `solidity` -- where an obstacle set would undo the entire intervention.
        Everywhere else the default is on, because a family that bends gravity
        should not also be a family that walks through walls.
        """
        if not solid:
            return None
        if given is not None:
            return given
        return _geom.Obstacles(spec, traj, exclude_ids=[body.segmentation_id])

    @staticmethod
    def _no_floor(spec, body):
        """A ground plane far below everything, for the one case that removes it.

        Kept separate from `solid`, which governs *obstacles*. Conflating the
        two was a real regression: `solidity` passing a ball through a wall also
        turned the ground off, so the ball sank through the floor mid-window and
        was bounced back out when contact resumed -- a clip that was supposed to
        show one thing showing three.
        """
        return lambda x, y: -1.0e4

    @staticmethod
    def _sync_velocity(traj: Trajectory, out: Trajectory, bi: int,
                       t0: int) -> None:
        """Make the recorded velocity agree with the positions actually written.

        Anything that edits `pos` without doing this leaves the two telling
        different stories, and `position_continuity` believes the velocity: it
        asks whether a step is larger than the speed in play could produce, so a
        body nudged upward at a stale velocity reads as having teleported.
        That is how the vertical clamp on `immutability` -- there only to stop a
        growing body sinking into the floor -- made half the appearance families
        trip the teleport detector.
        """
        dt = traj.dt
        # `t0 - 1` is a backward difference, so it needs clamping rather than
        # letting Python wrap it: at t0 == 0 `out.pos[-1:]` is the *last* frame,
        # a one-row slice, and this returned silently having synced nothing.
        # Harmless until the first family with `t_event == 0` -- `shadow_inverted`
        # -- at which point it is a body whose recorded velocity describes a
        # path it is not on.
        lo = max(t0 - 1, 0)
        pos = out.pos[lo:, bi, :].astype(np.float64)
        if pos.shape[0] < 2:
            return
        v = ((pos[1:] - pos[:-1]) / dt).astype(np.float32)
        out.lin_vel[lo + 1:, bi, :] = v
        if t0 == 0:                      # no frame -1 to difference against
            out.lin_vel[0, bi, :] = v[0]

    def _rewrite_group(self, spec, traj: Trajectory, out: Trajectory, bodies,
                       t0: int, g_by_body=None, v0_by_body=None,
                       substeps: int = 24) -> None:
        """Re-integrate several bodies *together*, so they collide with each other.

        Stepping them one at a time and treating the others' lawful paths as
        obstacles is right for the bodies an injector leaves alone, and wrong
        for the ones it is moving: under `global_gravity` every sphere in a
        pyramid is being re-integrated at once, so each would dodge where its
        neighbours *would have been* rather than where they now are, and they
        end up sharing space.
        """
        n = traj.num_frames - t0
        if n <= 0 or not bodies:
            return
        ids = [int(b.segmentation_id) for b in bodies]
        obstacles = _geom.Obstacles(spec, traj, exclude_ids=ids)
        idx = [traj.index_of(i) for i in ids]
        rad = [float(traj.radius[j]) for j in idx]
        rest = [float(b.restitution) for b in bodies]
        ground = [_geom.floor_fn(spec, b) for b in bodies]
        g_def = np.tile(traj.gravity.astype(np.float64)[None, :], (n, 1))
        g_seq = [np.asarray((g_by_body or {}).get(i, g_def), np.float64)
                 for i in ids]
        p = [traj.pos[t0 - 1, j].astype(np.float64).copy() for j in idx]
        v = [np.asarray((v0_by_body or {}).get(i, traj.lin_vel[t0 - 1, j]),
                        np.float64).copy() for i, j in zip(ids, idx)]

        h = traj.dt / float(substeps)
        pos = np.zeros((len(idx), n, 3), np.float64)
        vel = np.zeros((len(idx), n, 3), np.float64)
        for f in range(n):
            for k in range(substeps):
                t = float(t0 - 1) + f + (k + 1) / float(substeps)
                for a in range(len(idx)):
                    v[a] = v[a] + g_seq[a][f] * h
                    p[a] = p[a] + v[a] * h
                    fz = ground[a](p[a][0], p[a][1])
                    if p[a][2] - rad[a] <= fz and v[a][2] < 0.0:
                        p[a][2] = fz + rad[a]
                        v[a][2] = -v[a][2] * rest[a]
                        if abs(v[a][2]) < 0.05:
                            v[a][2] = 0.0
                    p[a], v[a] = obstacles.resolve(p[a], v[a], rad[a], rest[a], t)
                for a in range(len(idx)):
                    for b in range(a + 1, len(idx)):
                        d = p[b] - p[a]
                        dist = float(np.linalg.norm(d))
                        reach = rad[a] + rad[b]
                        if dist >= reach or dist < 1e-9:
                            continue
                        nrm = d / dist
                        push = 0.5 * (reach - dist)
                        p[a] -= nrm * push
                        p[b] += nrm * push
                        rel = float(np.dot(v[b] - v[a], nrm))
                        if rel < 0.0:
                            e = 0.5 * (rest[a] + rest[b])
                            imp = -(1.0 + e) * rel * 0.5
                            v[a] -= nrm * imp
                            v[b] += nrm * imp
            for a in range(len(idx)):
                pos[a, f] = p[a]
                vel[a, f] = v[a]
        for a, j in enumerate(idx):
            out.pos[t0:, j, :] = pos[a].astype(np.float32)
            out.lin_vel[t0:, j, :] = vel[a].astype(np.float32)

    @staticmethod
    def _pulse(n: int, peak: float, base: float = 1.0,
               hold: float = 0.42) -> np.ndarray:
        """Trapezoid: ramp `base` -> `peak`, hold, ramp back to `base`.

        A violation with a shape, not a step: it comes on, sustains, releases,
        and the actor obeys real physics afterwards. The default hold is under
        half the window, so most of it is ramp -- with the longer clips the
        tiers now use, that reads as a body drifting up and settling back rather
        than being yanked. The residual traces the
        same shape, which is what a per-frame severity field is for.

        Deliberately a trapezoid rather than a raised cosine. A cosine only
        touches its peak instantaneously, so the *mean* effect over the window
        is (base + peak) / 2 -- half the strength the bin advertises. On a body
        already moving fast that is not enough to visibly change its motion, and
        the strongest bin ends up looking like the weakest. Holding at the peak
        makes the advertised magnitude the one actually applied.
        """
        if n <= 1:
            return np.full((max(n, 1),), peak, np.float64)
        # The plateau is the load-bearing part and must never vanish: a profile
        # that only *passes through* its peak applies a mean effect of about
        # (base + peak) / 2, which is half what the bin advertises. The old
        # clamp kept a ramp frame on each side at all costs, so a 2-frame
        # window came out as two ramp values and never reached `peak` at all --
        # which only became reachable once the frustum fit started shortening
        # windows instead of weakening them.
        n_ramp = max(1, int(round((1.0 - hold) * n / 2.0)))
        while n_ramp > 0 and 2 * n_ramp >= n:
            n_ramp -= 1
        n_hold = n - 2 * n_ramp
        # Ramp values are strictly *between* base and peak: every frame inside
        # the window must actually be violating, or `active[t]` would mark
        # frames where alpha == 1 and nothing is wrong. The return to base
        # happens on the frame after the window, where normal physics resumes.
        up = np.linspace(base, peak, n_ramp + 2)[1:-1]
        down = up[::-1]
        return np.concatenate([up, np.full((n_hold,), peak, np.float64), down])

    @staticmethod
    def _first_contact(traj: Trajectory, a_id: int, b_id: int) -> Optional[int]:
        """First frame where bodies `a_id` and `b_id` are recorded in contact."""
        c = traj.contacts
        if len(c) == 0:
            return None
        m = (((c.body_a == a_id) & (c.body_b == b_id)) |
             ((c.body_a == b_id) & (c.body_b == a_id)))
        if not m.any():
            return None
        return int(c.frame[m].min())


_REGISTRY: Dict[str, Injector] = {}


def register(inj: Injector) -> Injector:
    _REGISTRY[inj.family] = inj
    return inj


def get(family: str) -> Injector:
    if family not in _REGISTRY:
        raise KeyError("no injector for family %r; have %s"
                       % (family, sorted(_REGISTRY)))
    return _REGISTRY[family]


def available() -> List[str]:
    return sorted(_REGISTRY)
