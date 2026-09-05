"""The dataset taxonomy as data — docs/PLAN.md Part 2.

Four levels: DOMAIN -> FAMILY -> SCENARIO -> INSTANCE.

This module is imported by BOTH environments (Python 3.9 inside the Kubric
container and Python 3.11 on the host), so it must stay pure stdlib with no
third-party imports and no 3.10+ syntax.
"""
from __future__ import annotations

from typing import Dict, List, NamedTuple, Optional, Tuple

# --------------------------------------------------------------------------
# Level 0 -- medium: what kind of matter is misbehaving
# --------------------------------------------------------------------------
# The macro-category, and the axis that lines up with LikePhys's four domains.
# It is *not* a competing scheme to `DOMAINS` -- the two are independent
# groupings of the same cells, and both are worth reporting:
#
#   medium  answers "what kind of matter", which is a capability question --
#           does this model handle deformables at all?
#   domain  answers "which principle broke", which is what severity is
#           measured against, since the domain picks the residual.
#
# The two empty rows are deliberate. Declaring `fluid` and `continuum` as
# known-empty says what the dataset does not cover; omitting them would leave a
# reader to infer it from absence.
MEDIA: Dict[str, str] = {
    "rigid":     "solid bodies that keep their shape",
    "granular":  "many small bodies behaving as a medium -- the v0 stand-in "
                 "for fluid, and never labelled as one",
    "optical":   "light and shadow rather than matter",
    "fluid":     "Phase 3 -- needs a Blender with working headless Mantaflow",
    "continuum": "Phase 3 -- cloth and soft bodies, behind a MuJoCo/MJX backend",
}

#: Media that no scenario stages yet. Named so the gap is legible.
EMPTY_MEDIA: Tuple[str, ...] = ("fluid", "continuum")


# --------------------------------------------------------------------------
# Level 1 -- domains: which physical law is at stake
# --------------------------------------------------------------------------
DOMAINS: Dict[str, str] = {
    "identity": "does the object persist and stay itself?",
    "kinematics": "is unsupported motion consistent with g?",
    "contact": "do bodies interact legally when they touch?",
    "dynamics": "do forces and masses behave?",
    "equilibrium": "do resting and supported bodies behave?",
    "optical": "is light consistent with geometry?",
    "appearance": "does the object look like itself from frame to frame?",
    "global": "are the scene's constants physical?",
}


class Family(NamedTuple):
    """Level 2 -- a specific way a law breaks."""

    domain: str
    injection: str
    magnitude_unit: str
    kind: str  # instant | sustained | repeated
    law: str  # which residual in residuals/laws.py scores it
    intphys2: Optional[str]  # cross-reference; None is a novelty claim
    likephys: Optional[str]
    #: What a scenario must offer for this family to be stageable there. The
    #: compatibility matrix is *derived* from these against each scenario's
    #: `provides`, rather than written out cell by cell -- see CAPABILITIES.
    requires: Tuple[str, ...] = ()
    #: Where the violation is detectable, and therefore what the severity field
    #: should be gated on.
    #:
    #:   "event"  only ACROSS the change. A recoloured cube is a perfectly
    #:            normal cube; the violation is that it used to be another
    #:            colour, which no single frame afterwards contains. Severity is
    #:            gated on the intervention window.
    #:   "state"  in ANY frame while it lasts. A hovering body is wrong whether
    #:            or not you saw it rise, and two bodies sharing space are wrong
    #:            on sight. Severity is gated on the consequence window.
    #:
    #: This is not a presentation detail. At inference a model has one video and
    #: no counterfactual, so marking an event family as violating long after the
    #: change asks it to report a violation from a frame that does not contain
    #: one -- and rewards a model that guesses from context instead of looking.
    detectable: str = "state"
    #: Whether the family has a real magnitude axis. `newton3_reaction` does
    #: not: staged honestly it is `mass = 0`, and immovable has no degrees --
    #: all three bins came out at severity 1.00 because they are one violation
    #: rendered three times. A simulator conserves momentum by construction, so
    #: non-conservation can only be said by handing the momentum to the world,
    #: and there is no partial version of that. Ungraded families generate their
    #: `strong` bin only, which is honest and saves two thirds of their budget.
    graded: bool = True


# Category of violations
FAMILIES: Dict[str, Family] = {
    # -- identity ----------------------------------------------------------
    "permanence": Family(
        "identity", "a body is simply gone between one frame and the next",
        "mass_ratio", "sustained", "mass_continuity", "permanence", "rigid_body",
        requires=('actor',), detectable="event"),
    "dissolve": Family(
        "identity", "a body fades out of visibility over several frames",
        "opacity_lost", "sustained", "mass_dissolution", "permanence",
        "rigid_body",
        requires=('actor',), detectable="event"),
    "immutability": Family(
        "identity", "the body grows or shrinks, ideally behind an occluder",
        "volume_ratio", "sustained", "shape_continuity", "immutability", "rigid_body",
        requires=('actor',), detectable="event"),
    "fission": Family(
        "identity", "one body becomes two, which fly apart",
        "count_ratio", "sustained", "object_count", "permanence", "rigid_body",
        requires=('actor', 'understudy'), detectable="event"),
    "fusion": Family(
        "identity", "bodies converge and come out as one, same size",
        "count_ratio", "sustained", "object_count", "permanence", "rigid_body",
        requires=('converging',), detectable="event"),
    # -- kinematics --------------------------------------------------------
    "continuity": Family(
        "kinematics", "discontinuous position set",
        "m_jump_distance", "instant", "position_continuity",
        "spatio_temporal_continuity", "rigid_body",
        requires=('actor',), detectable="event"),
    "non_parabolic": Family(
        "kinematics", "replace the free-flight arc with a non-g curve",
        "m_rms_from_parabola", "sustained", "trajectory_shape", None, None,
        requires=('flight',)),
    "antigravity": Family(
        "kinematics", "per-body gravity scale g -> alpha*g",
        "gravity_scale_deviation", "sustained", "free_fall", None, "rigid_body",
        requires=('flight',)),
    "time_slip": Family(
        "kinematics", "a body stalls in place, then resumes where it left off",
        "slip_frames", "sustained", "phase_consistency", None, "rigid_body",
        requires=('sliding',), detectable="event"),
    "newton1_inertia": Family(
        "kinematics", "a moving body stops dead and stays stopped",
        "dv_over_g_dt", "sustained", "linear_momentum", None, None,
        requires=('sliding',), detectable="event"),
    # -- contact -----------------------------------------------------------
    "solidity": Family(
        "contact", "disable a collision pair for N frames",
        "m_penetration_depth", "sustained", "penetration", "solidity", "rigid_body",
        requires=('contact',)),
    "superelastic": Family(
        "contact", "restitution e > 1",
        "energy_gain_ratio", "repeated", "energy_at_contact", None, "rigid_body",
        requires=('impact',), detectable="event"),
    # -- dynamics ----------------------------------------------------------
    "phantom_impulse": Family(
        "dynamics", "impulse applied with no contact",
        "impulse_over_m_vtyp", "instant", "linear_momentum", None, "rigid_body",
        requires=('actor',), detectable="event"),
    "newton2_mass": Family(
        "dynamics", "identical-looking bodies respond differently to equal impulse",
        "effective_mass_ratio", "instant", "linear_momentum", None, None,
        requires=('identical_pair',), detectable="event"),
    "angular_momentum": Family(
        "dynamics", "spin reverses, or torque appears with no contact",
        "angular_momentum_defect", "instant", "angular_momentum", None, None,
        requires=('spin',), detectable="event"),
    # -- equilibrium -------------------------------------------------------
    "support": Family(
        "equilibrium", "an unsupported body hovers, or an unstable stack holds",
        "m_support_clearance", "sustained", "support", None, None,
        requires=('contact',)),
    "friction": Family(
        "equilibrium", "a body accelerates against its motion, or slides forever",
        "effective_mu_ratio", "sustained", "friction", None, None,
        requires=('sliding',)),
    # -- optical -----------------------------------------------------------
    "shadow": Family(
        "optical", "the shadow detaches from its caster and slides away",
        "shadow_caster_offset_radii", "sustained", "shadow_consistency", None, "optical",
        requires=('cast_shadow',)),
    "shadow_inverted": Family(
        "optical", "the shadow sits on the lit side of its caster, all clip",
        "shadow_caster_offset_radii", "sustained", "shadow_consistency", None,
        "optical",
        requires=('cast_shadow',)),
    "shadow_shape": Family(
        "optical", "the shadow stays put but stops matching the caster's shape",
        "shadow_aspect_ratio", "sustained", "shape_anisotropy", None, "optical",
        requires=('cast_shadow',)),
    # -- appearance --------------------------------------------------------
    "colour_shift": Family(
        "appearance", "a body changes colour with nothing to explain it",
        "lab_distance", "sustained", "colour_continuity", "immutability", None,
        requires=('actor',), detectable="event"),
    "deformation": Family(
        "appearance", "a rigid body squashes or stretches out of proportion",
        "aspect_ratio_change", "sustained", "shape_anisotropy", None, None,
        requires=('actor',), detectable="event"),
    # -- global ------------------------------------------------------------
    "global_gravity": Family(
        "global", "the whole scene runs at alpha*g, internally consistent",
        "gravity_scale_deviation", "sustained", "free_fall", None, None,
        requires=('multi_body',)),
}


# --------------------------------------------------------------------------
# What a scene has to offer for a family to be stageable in it
# --------------------------------------------------------------------------
# The compatibility matrix used to be written out cell by cell, and it showed:
# `colour_shift` was built on twelve scenarios and `continuity` on six, for no
# reason except which cells someone had got round to. Both need exactly the same
# thing -- a visible actor -- so both should be everywhere.
#
# Families declare what they need, scenarios declare what they have, and the
# matrix is derived. Adding a scenario then lights up every family it can host,
# and adding a family lights up every scenario that can host it, without anyone
# revisiting a grid of 300 cells.
CAPABILITIES: Dict[str, str] = {
    "actor":          "a visible body the violation can act on",
    "flight":         "an actor spends several frames airborne",
    "contact":        "an actor touches something solid",
    "impact":         "an actor arrives at a contact with real closing speed",
    "identical_pair": "two visually indistinguishable dynamic bodies collide",
    "multi_body":     "two or more bodies move independently",
    "converging":     "two actors approach closely enough to plausibly merge",
    "resting":        "something sits supported and still",
    "sliding":        "something travels along a surface",
    "spin":           "rotation is visible, or motion is constrained to a pivot",
    "understudy":     "the scene declares a dormant duplicate (for fission)",
    "cast_shadow":    "the scene stages a shadow as a body of its own",
}


class Scenario(NamedTuple):
    """Level 3 -- the staged scene."""

    description: str
    event_structure: str
    has_occluder: bool
    physics_medium: str  # a key of MEDIA; never "fluid" at schema v0
    grounded_in: Optional[str]
    #: Which of the CAPABILITIES this scene actually offers. Declared once here
    #: rather than implied by a hand-written matrix row.
    provides: Tuple[str, ...] = ()


# Supported scenarios, with a short description and the event structure. The `provides`
# tuple is what a family must declare in `requires` to be stageable here.
SCENARIOS: Dict[str, Scenario] = {
    "drop": Scenario(
        "sphere or cube falls to a floor and bounces",
        "free fall -> contact -> rebound", False, "rigid", "LikePhys Ball Drop",
        provides=('actor', 'flight', 'contact', 'impact', 'spin', 'understudy')),
    "collision": Scenario(
        "two spheres roll toward each other",
        "approach -> collision -> separation", False, "rigid", "LikePhys Ball Collision",
        provides=('actor', 'contact', 'impact', 'identical_pair', 'multi_body',
                  'converging', 'sliding', 'understudy')),
    "ramp_slide": Scenario(
        "block slides down an incline",
        "sustained contact + friction", False, "rigid", "LikePhys Block Slide",
        provides=('actor', 'contact', 'sliding', 'understudy')),
    "toss": Scenario(
        "body thrown on a ballistic arc",
        "pure free flight, no contact", False, "rigid", None,
        # `spin` is promised conditionally, and the condition is the shape.
        # A featureless sphere's rotation is invisible even when it has some --
        # an audit found `toss x angular_momentum` shipping 0 observable frames
        # with segmentation byte-identical between the twins, a label with no
        # picture. This scenario randomises shape and throws a cube spinning,
        # so the promise holds on the cube draws; `angular_momentum` declines
        # the sphere draws for itself (see its `plan`). Declaring it here is
        # what lets the family be offered at all.
        provides=('actor', 'flight', 'spin', 'understudy')),
    "tumble": Scenario(
        # Retired into `UNBUILT`, not deleted -- see the note there. Kept
        # declared so a clip generated under the old taxonomy still resolves.
        "cube tumbling through the air, thrown with heavy spin",
        "free flight with visible rotation", False, "rigid", None,
        provides=('actor', 'flight', 'spin', 'understudy')),
    "occluder_pass": Scenario(
        "body travels behind a screen and re-emerges",
        "occlusion interval of known length", True, "rigid", "IntPhys 2 occlusion",
        provides=('actor', 'contact', 'sliding', 'understudy')),
    "barrier_pass": Scenario(
        "ball rolls into a solid wall and bounces back",
        "approach -> rebound off a static barrier", False, "rigid", None,
        provides=('actor', 'contact', 'impact', 'sliding', 'understudy')),
    "stack_topple": Scenario(
        "stacked bodies, marginally stable",
        "static equilibrium -> topple", False, "rigid", "IntPhys 2 / LikePhys",
        provides=('actor', 'contact', 'resting', 'multi_body', 'impact', 'flight')),
    "pyramid_impact": Scenario(
        "cube dropped onto a sphere pyramid",
        "multi-body contact chain", False, "rigid", "LikePhys Pyramid Impact",
        provides=('actor', 'flight', 'contact', 'impact', 'multi_body', 'resting')),
    "pendulum_swing": Scenario(
        "bob on a rigid rod, swung from a pivot (scripted, not solved)",
        "constrained periodic arc", False, "rigid", "LikePhys Pendulum",
        provides=('actor', 'spin', 'multi_body')),
    "resting_table": Scenario(
        "several bodies at rest on a surface",
        "sustained static equilibrium", False, "rigid", "IntPhys 2 permanence",
        provides=('actor', 'contact', 'resting', 'multi_body', 'understudy')),
    "rolling_ramp": Scenario(
        "cube tumbles down a raised ramp and off its lip",
        "rolling contact, then a short free flight", False, "rigid", None,
        provides=('actor', 'contact', 'sliding', 'flight', 'spin', 'understudy')),
    "shadow_track": Scenario(
        "object translates under a fixed light",
        "a clean, trackable cast shadow", False, "optical", "LikePhys Moving Shadow",
        provides=('actor', 'cast_shadow')),
    "clutter_toss": Scenario(
        "MOVi-style multi-object toss",
        "dense collisions, heavy occlusion", True, "rigid", "MOVi baseline",
        provides=('actor', 'flight', 'contact', 'impact', 'multi_body')),
    "pour": Scenario(
        "a loose column of grains falls into an open box (40 at the debug tier, 96 above)",
        "streaming flow, accumulation, break-up", False, "granular",
        "LikePhys Faucet Flow (as granular, not fluid)",
        provides=('actor', 'flight', 'contact', 'impact', 'multi_body', 'converging', 'identical_pair', 'resting', 'sliding')),
}

# Level 3 x Level 2 -- which families are meaningful in which scenarios.
# "build" == a v0 deliverable; "defer" == valid but not scheduled.
BUILD, DEFER = "build", "defer"

# Cells that the capability rules would allow but which are not meaningful, with
# the reason recorded. Every entry here is a measurement or an argument, not an
# omission -- an empty exception list would mean the derivation is trusted
# completely, and it nearly is.
#: Deferred *for now*, not on principle -- see docs/decisions_pending.md §3. The
#: tier lengthening left several scenarios with their actors at rest for most of
#: the clip, and `time_slip` refuses to stage a stall when the body has no
#: motion left to resume into, which is the right behaviour and makes it the
#: first family to notice the problem. These become BUILD when the timing is
#: fixed; nothing about the pairing itself is wrong.
TIME_SLIP_NEEDS_MOTION = (
    "%s's actor comes to rest well before the clip ends, so there is no window "
    "where it is moving AND has motion left to resume into. A stall staged "
    "there would be indistinguishable from the body simply having stopped. "
    "Temporary: see docs/decisions_pending.md section 3")

#: Cells that are stageable but should not be built, each with a written reason.
#:
#: Two kinds live here. Some are wrong in principle -- `newton2_mass` on
#: `pyramid_impact` pits a cube against a sphere, so "the cube is heavy" is a
#: lawful reading and there is no violation for a viewer to catch. Others are
#: **measured invisible**: the family has nothing to act on in that scenario, so
#: the clip carries a full set of labels describing a violation the video does
#: not contain, which teaches a model that nothing is something.
#:
#: The second kind is found, not guessed. `physviol audit <release>` measures
#: severity, observable frames and peak pixel evidence inside the mask for every
#: cell, and `tests/test_visible_violation.py` fails the build if one that is
#: still marked BUILD depicts nothing. Move a flagged cell here with its
#: numbers, or fix the scenario so the family has something to act on.
_NOT_MEANINGFUL_DOC = None


#: Families that were built and then withdrawn, with the reason. Kept as a
#: record so the question does not get reopened from scratch.
RETIRED: Dict[str, str] = {
    "newton3_reaction":
        "indistinguishable from `newton2_mass` once both are staged honestly. "
        "Expressed as an immovable target (`mass = 0`) it is the limit of "
        "newton2's mass ratio as that ratio goes to infinity, and at newton2's "
        "strong bin of 25 the two rendered the same clip: struck ball barely "
        "moves, striker rebounds. Expressed instead as momentum injected into "
        "the striker -- the one direction newton2 cannot imitate -- it fired "
        "on two severity bins out of three and could not be made reliable. "
        "`newton2_mass` covers 'the collision came out wrong' with a working "
        "ladder; a second family that sometimes depicts the same thing and "
        "sometimes depicts nothing is worse than no second family.",
}

NOT_MEANINGFUL: Dict[Tuple[str, str], str] = {
    ("global_gravity", "pendulum_swing"):
        "the pendulum has exactly one body gravity acts on -- the rod is "
        "kinematic geometry hung between the pivot and the bob -- and this "
        "family's whole content is that everything falls wrong TOGETHER. With "
        "one falling body, scaling gravity for the scene and scaling it for "
        "that body produce identical pixels, so the clip would be an "
        "`antigravity` clip with a different label. `antigravity` is built here "
        "and says the same thing honestly",
    ("newton2_mass", "pyramid_impact"):
        "the spheres are identical to each other but never strike one another "
        "-- they touch from frame 0 and settle, so there is no reaction to "
        "suppress; measured at severity 0.01",
    ("solidity", "occluder_pass"):
        "the only surface is the floor, so the violation sank the ball into the "
        "ground while it was hidden and read as a vanish -- which is what "
        "permanence does on the same scenario. `barrier_pass` stages it instead",
            ("time_slip", "collision"): TIME_SLIP_NEEDS_MOTION % "collision",
    ("time_slip", "ramp_slide"): TIME_SLIP_NEEDS_MOTION % "ramp_slide",
    ("time_slip", "pour"):
        "forty grains in constant mutual contact: stalling one desynchronises "
        "it from its neighbours, which then pass through where it is standing "
        "-- a penetration violation inside a clip labelled as a temporal one",
    ("fission", "pour"):
        "forty grains already; splitting one more is not a legible change in "
        "object count",
    ("fusion", "clutter_toss"):
        "clutter_toss is not built",
    ("fission", "ramp_slide"):
        "the halves are pushed apart horizontally and a tilted slab gives them "
        "nowhere to land: one leaves the ramp's surface and drops to the floor, "
        "which reads as a solidity failure inside a fission clip. Splitting "
        "needs either free flight or a level support",
}

#: Scenarios declared in the taxonomy but not implemented yet.
#: `tumble` is retired rather than deleted. It was a near-duplicate of `toss`
#: -- identical `provides`, medium and family list, launch velocities within 2%
#: -- differing only in that it forced a cube and threw it spinning. `toss` now
#: throws with spin too and randomises shape, so it covers both draws and the
#: pair was redundant. The declaration stays so that a clip generated before
#: the retirement still resolves its scenario; the implementation is gone.
UNBUILT: Tuple[str, ...] = ("clutter_toss", "tumble")


def _derive_compatibility() -> Dict[str, Dict[str, str]]:
    """Level 2 x Level 3, computed from what each side declares.

    Written out by hand this drifted badly: `colour_shift` was built on twelve
    scenarios and `continuity` on six, though both need exactly the same thing.
    Deriving it means a family is offered wherever it can be staged, and the
    only judgement calls left are the ones in NOT_MEANINGFUL, which have to be
    argued for in writing.
    """
    out: Dict[str, Dict[str, str]] = {}
    for fam, meta in FAMILIES.items():
        row: Dict[str, str] = {}
        for scen, sc in SCENARIOS.items():
            if not set(meta.requires).issubset(set(sc.provides)):
                continue
            if (fam, scen) in NOT_MEANINGFUL or scen in UNBUILT:
                row[scen] = DEFER
            else:
                row[scen] = BUILD
        out[fam] = row
    return out


COMPATIBILITY: Dict[str, Dict[str, str]] = _derive_compatibility()


def why_deferred(family: str, scenario: str) -> Optional[str]:
    """Why a cell is not built, when there is a reason worth reading."""
    if scenario in UNBUILT:
        return "%s is declared in the taxonomy but not implemented" % scenario
    return NOT_MEANINGFUL.get((family, scenario))


# --------------------------------------------------------------------------
# Orthogonality -- which family is allowed to move which law
# --------------------------------------------------------------------------
# The dataset only supports the claim "this model cannot detect X" if clips
# labelled X do not also contain Y. That is not automatic: an injector that
# re-integrates a body will happily send it through a wall unless something
# stops it, and then a `global_gravity` clip is also a `solidity` clip while
# being scored as neither.
#
# These laws are the ones with a clean zero baseline on a lawful clip -- a body
# either passed through something or it did not -- which makes them usable as
# tripwires. For each, the families that are *entitled* to move it. Any other
# family moving it is contamination, and `tests/test_orthogonality.py` fails.
#
# The continuous dynamical laws (`free_fall`, `linear_momentum`,
# `energy_at_contact`, `friction`, `support`, `trajectory_shape`,
# `angular_momentum`) are deliberately absent. They co-move by physics -- change
# a body's gravity and its momentum residual moves too -- so demanding
# orthogonality there would be demanding that physics be separable, which it is
# not. Orthogonality is a claim about *staging*, not about mechanics.
EXCLUSIVE_LAWS: Dict[str, Tuple[str, ...]] = {
    "penetration":         ("solidity",),
    "position_continuity": ("continuity",),
    "mass_continuity":     ("permanence",),
    "object_count":        ("fission", "fusion", "permanence", "dissolve"),
    # `dissolve` is NOT here. It fades optically -- a Transparent BSDF mixed
    # into the shader -- so it changes neither shape nor position and moves no
    # law but its own.
    # `fusion` is here as well, and it is the one entry that is a TRADE rather
    # than a fact. A merge that conserves mass must make the survivor bigger --
    # two bodies of volume V becoming one of volume 2V is what "they became one
    # thing" means, and without it the clip shows one object quietly deleted.
    # So the family unavoidably moves the shape law too.
    #
    # What that costs: a `fusion` clip also reads as a shape violation, so the
    # shape law alone no longer attributes to `immutability`. What it does not
    # cost: the two stay separable, because `fusion` moves `object_count` and
    # `immutability` does not. The alternative -- a survivor that swallows
    # another body and stays exactly the same size -- keeps the law clean by
    # depicting something that cannot happen, which is the worse trade for a
    # dataset whose whole claim is that its physics is real.
    "shape_continuity":    ("immutability", "fusion"),
    "shape_anisotropy":    ("deformation", "shadow_shape"),
    "colour_continuity":   ("colour_shift",),
    "mass_dissolution":    ("dissolve",),
}

#: How far an unrelated law may move before it counts as contamination, in body
#: radii. A sub-radius step is not something a viewer can see and not something
#: a detector would fire on; a whole-radius one is a different violation.
ORTHOGONALITY_TOLERANCE = 1.0


SEVERITY_BINS: Tuple[str, ...] = ("weak", "medium", "strong")


def domain_of(family: str) -> str:
    return FAMILIES[family].domain


def build_cells() -> List[Tuple[str, str]]:
    """Every (scenario, family) pair scheduled for v0. The generator's job list."""
    return sorted(
        (scenario, family)
        for family, scen in COMPATIBILITY.items()
        for scenario, status in scen.items()
        if status == BUILD
    )


def is_compatible(scenario: str, family: str, require_build: bool = False) -> bool:
    status = COMPATIBILITY.get(family, {}).get(scenario)
    return status == BUILD if require_build else status in (BUILD, DEFER)


def validate_taxonomy() -> None:
    """Internal consistency. Called by tests and by `physviol validate`."""
    for name, fam in FAMILIES.items():
        assert fam.domain in DOMAINS, "%s: unknown domain %s" % (name, fam.domain)
        assert fam.kind in ("instant", "sustained", "repeated"), name
    for family, scen in COMPATIBILITY.items():
        assert family in FAMILIES, "compatibility for unknown family %s" % family
        for scenario, status in scen.items():
            assert scenario in SCENARIOS, "%s: unknown scenario %s" % (family, scenario)
            assert status in (BUILD, DEFER), "%s/%s: %s" % (family, scenario, status)
    missing = set(FAMILIES) - set(COMPATIBILITY)
    assert not missing, "families with no scenarios: %s" % sorted(missing)
    for name, sc in SCENARIOS.items():
        assert sc.physics_medium in MEDIA, "%s: unknown medium %s" % (
            name, sc.physics_medium)
        assert sc.physics_medium not in EMPTY_MEDIA, (
            "%s claims medium %r, which schema v0 does not stage"
            % (name, sc.physics_medium))
        for cap in sc.provides:
            assert cap in CAPABILITIES, "%s: unknown capability %s" % (name, cap)
    for name, fam in FAMILIES.items():
        for cap in fam.requires:
            assert cap in CAPABILITIES, "%s: unknown capability %s" % (name, cap)
    for (fam, scen) in NOT_MEANINGFUL:
        assert fam in FAMILIES, fam
        assert scen in SCENARIOS, scen
