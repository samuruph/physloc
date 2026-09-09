"""Shared scenario building blocks.

Scenario files stay short because everything generic lives here: the ground
slab, the HDRI dome that backs it from complexity L2 up, standard lighting and
a colour helper. A new scenario is then mostly a description of *what is staged*,
not renderer plumbing.
"""
from __future__ import annotations

import colorsys
import math
from typing import Optional, List, Tuple

from .base import BodySpec, Complexity, LightSpec


def appearance_rng(seed: int, salt: str = "") -> "np.random.RandomState":
    """A stream for how a scene LOOKS, independent of how it behaves.

    Appearance draws must not come off the physics stream. `pick_hdri(rng)` did,
    and because it only fires at L1 the extra draw shifted every physics value
    after it -- so the same seed produced a different rollout at L0 and L1, and
    the two could not be paired.

    That pairing is what v1 is for: the same physics rendered plainly and
    photographically, so a benchmark can ask whether a model's grasp of the
    physics survives the realism. A salted, separate `RandomState` costs nothing
    and makes the two streams independent by construction.

    `salt` is normally the scenario's name. Without it every scenario draws the
    same first number for a given seed, so seed 0 produced a stone actor in
    `drop`, in `collision`, in `toss` and in `ramp_slide` alike -- a correlation
    across the release that buys nothing.

    Call this ONCE per sample and thread the result. It builds a fresh
    RandomState every call, so asking for it per draw hands every draw the same
    first number, which is how material, colour and proportions all ended up
    perfectly correlated.

    The HDRI is the exception and takes a stream of its own. It is drawn only
    from L1 up, so sharing the scenario's stream would let that one extra draw
    shift every material and dimension after it -- and then the same seed would
    produce a different body at L0 and at L1, which is precisely the pairing
    this function was written to protect. `test_complexity_isolation` catches it.
    """
    import numpy as np
    import zlib

    mixed = int(seed) * 2654435761 + 0x9E3779B9
    if salt:
        mixed += int(zlib.crc32(salt.encode()))
    return np.random.RandomState(mixed % (2 ** 31 - 1))


#: Where the HDRI backdrop's segmentation id sits. Above every scenario's own
#: and above the extras, so adding it cannot renumber anything else.
SEG_BACKDROP = 900


def ground(cx: Complexity, seg_id: int, size: float = 6.0) -> BodySpec:
    """A plain cube slab, at EVERY level. THE THING THE OBJECTS LAND ON.

    It was briefly the KuBasic dome instead -- MOVi's trick, where one body is
    both the ground and the HDRI backdrop -- because a cube and a dome are
    different collision shapes, so the same seed rolled differently at the HDRI
    level and no two levels could be compared. That fixed the comparison and cost
    a great deal: a dome ENCLOSES the scene, so every ray that misses an object
    hits it and bounces, where a cube lets those rays escape. Measured on one
    scene at release geometry, all seven passes: **8.69 s/frame with the cube,
    27.54 with the dome** -- 3.2x, on the 75% of the dataset that is L0 and L1,
    and about two months of compute on a full release.

    The dome was doing two jobs and only one of them had to be uniform. The
    COLLIDER must match across levels, because that is what the rollout depends
    on; the BACKDROP need not, because nothing physical reads it. So the cube
    collides everywhere and `backdrop()` adds the dome, render-only, at the
    levels that light themselves from an HDRI.
    """
    return BodySpec(name="floor", kind="cube", position=(0.0, 0.0, -0.1),
                    scale=(size, size, 0.1), mass=0.0, static=True,
                    friction=0.6, restitution=0.4, color=(0.32, 0.33, 0.36),
                    segmentation_id=seg_id, role="floor")


def backdrop(cx: Complexity) -> Optional[BodySpec]:
    """The dome the HDRI is projected onto, or `None` below L2.

    RENDER-ONLY. It is added to the scene like anything else -- the trajectory
    is built from `spec.bodies` and would fail on a body the simulator has
    never heard of -- but it is declared `collides=False`, so the cube slab
    stays the only ground and the rollout is identical to the level below.

    `role="backdrop"` keeps it out of everything that reasons about what the
    scene DEPICTS -- it is not an actor, not something a family can target and
    not something a distractor must clear -- and `collides=False` keeps it out
    of everything that reasons about what the scene TOUCHES. The second is not
    a restatement of the first: the role was there from the start and the
    geometry search that put `solidity` through the floor at L2 and L3 did not
    consult it, because "is there a static surface under this body" is a
    question about colliders, not about staging.

    ONE VISIBLE CONSEQUENCE, checked in a render rather than assumed. The dome's
    inner surface sits at z = 0, exactly where the slab's top is, and the dome
    wins: at L2 and L3 the ground you SEE is this body, and the slab reports
    `frames_visible: 0` and holds no pixels. That is the MOVi look and it is
    what an HDRI environment should give you -- the ground belongs to the
    capture rather than being a grey rectangle floating in it. It is also
    uniform, not speckled: the two surfaces are coplanar but the dome is drawn
    as the background, so there is no z-fighting to see.
    """
    if cx.background != "hdri":
        return None
    return BodySpec(name="backdrop", kind="dome", position=(0.0, 0.0, 0.0),
                    mass=0.0, static=True, collides=False,
                    friction=0.6, restitution=0.4,
                    segmentation_id=SEG_BACKDROP, role="backdrop")


def lights(cx: Complexity, look_at=(0.0, 0.0, 0.6),
           scale: float = 1.0) -> List[LightSpec]:
    """THE SAME KEY LIGHT AT EVERY LEVEL. The environment is what changes.

    An HDRI was allowed to light the scene on its own, on the reasoning that an
    environment map is a complete lighting rig. It is -- for illumination. It is
    not for CONTACT SHADOW, because most of HDRI Haven is overcast, indoor or
    shaded, and a soft dome throws nothing a 128 px frame can resolve. Measured
    on `stack_topple`, seed 777, by differencing a render against the same
    render with the actors removed: at L0 the slab under the stack darkened by
    up to 72/255, and at L2 with the dome and no sun by under 8 -- invisible in
    the clip, and the blocks read as pasted onto the street rather than
    standing on it. Adding this sun back put a 188/255 cast shadow on the dome,
    which does receive one perfectly well once something directional is there
    to cast it.

    That matters more here than the small realism cost of a key light whose
    direction need not agree with the environment's own. **The complexity
    ladder is scene realism and nothing else**, so a level may change how a
    scene is lit and may not change whether the physics is legible in it.
    Grounding is physics: `support` hovers a body a few centimetres off a
    surface, and without a contact shadow there is nothing in the frame that
    says where the surface was.

    `scale` is the scene's linear size relative to the hand-tuned default of
    roughly four metres across -- see `camera.REFERENCE_HALF_EXTENT`. The
    free-flight scenarios size themselves from the clip length, and a sun left
    at 4.5 m while the actor arcs twenty metres up would be a lamp *inside* the
    trajectory, lighting the underside of everything.
    """
    s = float(scale)
    return [LightSpec("sun", position=(-2.2 * s, -1.6 * s, 4.5 * s),
                      look_at=look_at, intensity=2.6)]


def hue_rgb(h: float, s: float = 0.62, v: float = 0.88) -> Tuple[float, float, float]:
    return tuple(float(c) for c in colorsys.hsv_to_rgb(h, s, v))


# --------------------------------------------------------------------------
# Inclines. Two scenarios stage one, and getting a body to *sit* on a tilted
# slab is fiddly enough that doing it twice by hand invites a body embedded in
# its own ramp -- which reads as a solidity violation nobody injected.
# --------------------------------------------------------------------------


def ramp_axes(tilt: float) -> Tuple[Tuple[float, float, float],
                                    Tuple[float, float, float]]:
    """(down-slope, surface-normal) unit vectors for a slab tilted about +Y.

    A rotation of `tilt` about +Y takes +X to (cos, 0, -sin), so that is the
    down-slope direction, and +Z to (sin, 0, cos), the outward normal.
    """
    c, s = math.cos(tilt), math.sin(tilt)
    return (c, 0.0, -s), (s, 0.0, c)


def ramp(seg_id: int, tilt: float, center, half_len: float, half_wid: float,
         thick: float, friction: float = 0.3, name: str = "ramp") -> BodySpec:
    return BodySpec(
        name=name, kind="cube", position=tuple(float(x) for x in center),
        scale=(half_len, half_wid, thick),
        quaternion=(math.cos(tilt / 2.0), 0.0, math.sin(tilt / 2.0), 0.0),
        mass=0.0, static=True, friction=friction, restitution=0.15,
        color=(0.38, 0.36, 0.42), segmentation_id=seg_id, role="prop")


def on_ramp(center, tilt: float, along: float, half_size: float,
            clearance: float = 0.02) -> Tuple[float, float, float]:
    """World position of a body of half-extent `half_size` resting on a ramp.

    `along` is signed distance from the slab's centre down the slope. The
    `clearance` lifts it a hair off the surface so the simulator settles it into
    contact rather than starting it interpenetrating.
    """
    d, n = ramp_axes(tilt)
    lift = half_size + clearance
    return tuple(float(center[i] + along * d[i] + lift * n[i]) for i in range(3))


def understudy(actor: BodySpec, seg_id: int) -> BodySpec:
    """A dormant duplicate of `actor`, parked out of the render until summoned.

    `fission` needs a second body to spawn, and a body cannot be added to a
    Kubric scene part-way through a render without perturbing the very render
    path prefix identity depends on. So the understudy is declared from frame 0,
    marked `dormant`, and contributes no pixels until an intervention switches
    it on. It costs one scene asset and nothing at all in the valid twin.
    """
    # Parked far below the scene, not at the actor's own position: the
    # understudy is `static` to the simulator but still a collidable body, and
    # sharing the actor's start pose made PyBullet report a contact between the
    # two on frame 0 -- which then became the "first contact" that `solidity`
    # and `superelastic` hung their violation on.
    return BodySpec(
        name=actor.name + "_split", kind=actor.kind, position=(0.0, 0.0, -1000.0),
        scale=actor.scale, quaternion=actor.quaternion, mass=actor.mass,
        friction=actor.friction, restitution=actor.restitution,
        color=actor.color, segmentation_id=seg_id, role="actor",
        scripted=True, dormant=True)


def with_material(body: BodySpec, name: str, rng, look=None,
                  mass_from: str = "volume") -> BodySpec:
    """Give a body a material, deriving its mass and look from it.

    The single place a scenario opts in. Returns a NEW spec rather than
    mutating, so `understudy()` clones stay correct whichever order they are
    built in.

    Draw `rng` from `appearance_rng`, never the physics stream -- see that
    function. The mass this produces is physics, but it is a deterministic
    function of the scene's appearance, so the two streams stay independent.
    """
    from . import materials as M
    import dataclasses

    # `look` lets two bodies share one draw. `collision` needs it: the pair
    # must be INDISTINGUISHABLE for `newton2_mass` to mean anything, and giving
    # them the same material off the same stream is not enough -- the second
    # call advances the stream and comes back a different colour.
    look = M.appearance(name, rng) if look is None else look
    rgb, roughness, metallic, specular, transmission, ior = look
    # `mass_from`:
    #   "volume" -- density x volume, the honest default.
    #   "ratio"  -- scale the mass the scenario already chose by how much
    #               denser this material is than wood. For the scenarios whose
    #               mass is a TUNED quantity rather than an incidental one: a
    #               pour's grains were picked at 0.12 kg so the pile settles
    #               instead of jittering, and deriving 0.02 kg from a 6 cm
    #               sphere's volume would quietly retune the medium. This keeps
    #               the tuning and still varies the material.
    if mass_from == "ratio":
        mass = float(body.mass) * M.density_ratio(name)
    else:
        mass = M.mass_for(name, body.scale, body.kind)
    return dataclasses.replace(
        body, material=name, color=rgb, roughness=roughness, metallic=metallic,
        specular=specular, transmission=transmission, ior=ior,
        mass=mass)


#: How far a body's proportions may stray from cubic, as a ratio between its
#: longest and shortest axis. 2.2 gives slabs and columns that are obviously
#: not cubes without producing splinters that tunnel through the floor or
#: topple the moment they are placed.
MAX_ASPECT = 2.2


def vary_dims(body: BodySpec, rng, max_aspect: float = MAX_ASPECT) -> BodySpec:
    """Give a box body independent half-extents, and reseat it.

    Sizes already varied per seed, but only as ONE number: `scale=(r,)*3`
    everywhere, so every cube was a perfect cube and every sphere a perfect
    sphere. Two clips of the same scenario differed in how big the object was
    and never in what shape it was.

    Reseating is the part that is easy to get wrong. A body's z is written by
    the scenario as "resting on the surface", which for a cube means
    `z = scale[2]`; change the height without moving the body and it is either
    buried in the floor or hovering above it, and `support` -- whose whole
    subject is whether a thing is held up -- would be measuring an artefact.
    So the shift is applied to z by exactly the change in half-height.

    EVERY OTHER SHAPE IS LEFT ALONE, and not by preference -- see below. A
    sphere also could not take it for a second reason: the rolling scenarios
    set its spin as `omega = v / r`, which has no meaning for a body with two
    radii.
    """
    # BOXES ONLY, and this is a simulator limit rather than a choice.
    # `kubric/simulator/pybullet.py` builds a `kb.Cube` as `GEOM_BOX` with
    # `halfExtents=obj.scale`, so a box may have three different half-extents.
    # A `kb.Sphere` asserts `scale[0] == scale[1] == scale[2]`, and every
    # KuBasic mesh -- cylinder, cone, torus -- asserts the same with the
    # message "Pybullet does not support non-uniform scaling". Those are hard
    # asserts inside the container, so squashing one is not a subtly wrong
    # collider, it is a crashed render.
    if body.kind != "cube":
        return body
    import dataclasses

    sx, sy, sz = (float(v) for v in body.scale)
    lo, hi = 1.0 / float(max_aspect) ** 0.5, float(max_aspect) ** 0.5
    scale = (sx * float(rng.uniform(lo, hi)),
             sy * float(rng.uniform(lo, hi)),
             sz * float(rng.uniform(lo, hi)))
    x, y, z = (float(v) for v in body.position)
    out = dataclasses.replace(body, scale=scale,
                              position=(x, y, z + scale[2] - sz))
    # Mass follows the new volume -- SCALED, not recomputed. Recomputing from
    # `density x volume` silently discarded a `mass_from="ratio"` mass, which
    # is how `pyramid_impact` ended up with a striker LIGHTER than the pile it
    # exists to scatter. Scaling by the volume change is right whichever way
    # the mass was arrived at.
    if body.material is not None:
        old_v = sx * sy * sz
        if old_v > 1e-12:
            out = dataclasses.replace(
                out, mass=float(body.mass) * (scale[0] * scale[1] * scale[2]) / old_v)
    return out


#: Shapes whose motion a viewer can predict, which is the premise of asking
#: whether a clip obeyed physics. KuBasic also ships `gear`, `torus_knot`,
#: `sponge`, `spot`, `teapot` and `suzanne`; a tumbling Suzanne head is a fine
#: picture and a poor test.
#:
#: `sphere` and `cube` are Kubric primitives; the rest are KuBasic meshes, and
#: PyBullet will only take a UNIFORM scale for those (see `vary_dims`).
FREE_SHAPES = ("sphere", "cube", "cylinder", "cone", "torus")

#: For scenarios that roll or slide their actor along a surface. A sphere rolls
#: and a box slides, and both scenarios' spin is written for exactly those two
#: cases -- `omega = v / r` means nothing for a cone, and a torus on its side
#: rolls in a way the approach solve does not model.
SURFACE_SHAPES = ("sphere", "cube")


def pick_shape(rng, choices=FREE_SHAPES) -> str:
    """One shape name, drawn off the APPEARANCE stream.

    Shape is physics as much as appearance -- a cone topples where a ball rolls
    -- but it is drawn here for the same reason the material is: so that
    adding shapes to the menu cannot shift the physics draws a scenario already
    made, and the same seed keeps producing the same rollout at L0 and L1.
    """
    return str(choices[int(rng.randint(0, len(choices)))])


#: Segmentation ids for distractors start here. Clear of every scenario's own
#: ids, and clear of `pour`, whose grains run to 111 at release size.
SEG_DISTRACTOR_BASE = 500

#: Attempts per distractor before giving up on it. The constraints pull
#: against each other -- clear of the path, not in front of it, inside the
#: frame -- so a scene can genuinely have room for fewer than were asked for.
#: `distractors()` returns what it managed, and the caller records it, because
#: silently placing four when six were requested is the kind of thing that
#: turns into a puzzling number in a results table months later.
PLACEMENT_TRIES = 400

#: How many points along the actor's predicted path are protected. Six is
#: enough to cover a fall or a roll at this scale without the rejection
#: sampler running out of room to place anything.
PATH_SAMPLES = 6

#: A distractor's size, as a fraction of the actor's. Comparable on purpose:
#: something a tenth the size is a speck, and a benchmark level called
#: "distractors" should contain things that actually compete for attention.
DISTRACTOR_SIZE = (0.35, 1.60)

#: How fast a MOVING distractor goes, as a fraction of the actor's own speed
#: (or of a walking pace when the actor starts at rest). Static clutter is easy
#: clutter: a model can learn "the thing that moves is the subject" and never
#: look at the physics at all. Moving distractors take that shortcut away.
#:
#: The lower bound is no longer zero, because "moving" now means moving. It
#: used to be a uniform draw from 0, so a distractor's speed was a continuum
#: and roughly a third of them were near-stationary by accident rather than by
#: choice; `DISTRACTOR_MOVING` decides that explicitly now.
DISTRACTOR_SPEED = (0.25, 0.85)

#: Share of distractors that move at all. The rest are inert -- genuinely
#: still, not slow. A scene of uniformly drifting clutter is as learnable as a
#: scene of uniformly still clutter; a mixture is neither.
DISTRACTOR_MOVING = 0.5

#: Share of distractors that start ABOVE the floor and fall into the scene,
#: rather than resting on it. Another shortcut removed -- if only the actor
#: ever falls, falling identifies the actor.
DISTRACTOR_AIRBORNE = 0.35

#: How much of the actor's silhouette a distractor must clear, in actor radii.
#: Smaller than the physical margin on purpose -- see the note at its use.
SIGHTLINE_RADII = 1.35

#: How far a distractor must stay from the actor's path, in actor radii. The
#: point of a distractor is to be *distracting*, not to take part: one that
#: wanders into the collision changes the physics the clip is a claim about,
#: and there is no label saying it did.
#:
#: Five rather than three because the protected path is BALLISTIC and a body
#: does not stop when it lands -- it bounces and rolls, into places the sweep
#: never predicted. On `drop x support` the actor came to rest against a
#: distractor three radii away, and the violation was then scored against that
#: cylinder's top instead of the floor.
#:
#: The margin cannot be the whole answer, because post-bounce motion is not
#: predictable from the declared start state. It is paired with a rule in
#: `_geom.support_under`: a distractor is never a support surface, whatever the
#: body ends up next to.
KEEP_CLEAR_RADII = 5.0


#: Where a multi-object peer's segmentation ids start. Distinct from the
#: distractor block so a reader can tell the two conditions apart from ids
#: alone, and far from any scenario's own.
SEG_PEER_BASE = 600


def distractors(spec, n: int, rng, floor_top: float = 0.0, role: str = "distractor"):
    """`n` extra bodies placed around the scene, clear of the action.

    ONE PLACER, TWO CONDITIONS. With `role="distractor"` the extras are inert
    scenery that no family can target. With `role="actor"` the very same bodies
    become lawful PEERS -- eligible culprits, which is what the `multi`
    condition is made of. The placement problem is identical either way (extra
    bodies, clear of the action, inside the frame), and the only thing that
    differs is whether the physics is allowed to notice them, so writing it
    twice would be writing the same rejection sampler twice and letting the two
    copies drift.

    MOVi places its distractors with `kb.move_until_no_overlap`, which
    resamples a pose until the SIMULATOR reports no overlap
    (`refs/kubric/kubric/randomness.py:119`). That is not available here: a
    `SceneSpec` is declarative and is built host-side, with no simulator in
    reach, and it has to stay that way because the annotator reconstructs the
    scene from a seed to read the camera and the bodies back. So placement is
    geometric -- rejection sampling against bounding radii, which for convex
    primitives on a floor is the same test the simulator would do.

    Two rules the placement has to respect, and they pull in opposite
    directions:

    * FAR ENOUGH from the actor and its path that it cannot join the physics.
      A distractor that rolls into the collision changes the event the clip is
      labelled for, and nothing in the annotation says so.
    * NEAR ENOUGH to be in shot. A distractor outside the frustum is not a
      distractor, it is a body that costs render time.

    `role="distractor"` is excluded by every actor query -- the injectors select
    on `role == "actor"` -- so no family can target one by accident. That is
    exactly the line the `multi` condition crosses on purpose.
    """
    import numpy as np

    from . import materials as M
    from .base import BodySpec

    actors = [b for b in spec.bodies if b.role == "actor" and not b.dormant]
    if not actors or n <= 0:
        return []
    actor_r = float(np.median([a.bounding_radius for a in actors]))
    actor_v = np.median([np.asarray(a.velocity, np.float64) for a in actors],
                        axis=0)
    # The band to place in: outside the action, inside the shot. Sized from the
    # camera's own framing so it holds whatever scale the scenario works at.
    from .. import camera as cam
    extent = cam.frame_extent(spec.camera_position, spec.camera_look_at)
    aim = np.asarray(spec.camera_look_at, np.float64)

    eye = np.asarray(spec.camera_position, np.float64)
    # THE WHOLE PATH, not the starting point. The first version protected the
    # start and one second of travel, which for `drop` -- where the actor
    # begins three metres up and falls -- guards empty air and leaves the
    # landing site, the part the violation is about, completely unprotected. A
    # distractor placed there occluded the actor and its `severity_map` went
    # from 0.23 to empty.
    #
    # A coarse ballistic sweep is enough: exact enough to know where the body
    # spends its time, and it costs nothing because it is arithmetic on the
    # declared start state rather than a rollout.
    duration = float(spec.tier.num_frames) / float(spec.tier.fps)
    g = np.asarray(spec.gravity, np.float64)
    keep_clear, sightlines = [], []
    for a in actors:
        p0 = np.asarray(a.position, np.float64)
        v0 = np.asarray(a.velocity, np.float64)
        keep = float(a.bounding_radius) * KEEP_CLEAR_RADII
        for k in range(PATH_SAMPLES):
            t = duration * k / float(PATH_SAMPLES - 1)
            p = p0 + v0 * t + 0.5 * g * t * t
            # A PREDICTED sample cannot go below the floor, and a body resting
            # on it is exactly where a distractor must not stand -- so the
            # ballistic guess is lifted to resting height.
            #
            # The FIRST sample is not a guess. It is where the body actually
            # is, and clamping it protected a point the actor was not at:
            # `toss` starts its ball at z = 0.558 with a 0.652 radius, so the
            # clamp moved the protected point 9 cm up and a distractor was
            # admitted into the gap -- 0.137 rad from the ball against the
            # 0.139 it needed. Marginal, and marginal is what a guard is for.
            if k:
                p[2] = max(float(p[2]), floor_top + float(a.bounding_radius))
            keep_clear.append((p, keep))
            # TWO DIFFERENT MARGINS, because they answer different questions.
            # Physical clearance wants room for a body to move without meeting
            # anything; a sightline only has to clear the actor's silhouette.
            # Using the physical margin for both excluded most of the visible
            # floor -- `drop` placed zero distractors on some seeds, because a
            # three-metre fall sampled six times casts a very wide shadow.
            sightlines.append((p, float(a.bounding_radius) * SIGHTLINE_RADII))

    # How far the exclusion reaches from the aim point, in the ground plane.
    aim_xy = aim[:2]
    exclusion = max(
        [float(np.linalg.norm(np.asarray(c)[:2] - aim_xy)) + keep
         for c, keep in keep_clear] or [0.0])

    out = []
    for i in range(int(n)):
        for _ in range(PLACEMENT_TRIES):
            ang = float(rng.uniform(0.0, 2.0 * np.pi))
            # SIZED AGAINST THE ACTOR, not the frame. Sizing off the frustum
            # gave distractors 6 to 34 pixels against the actor's 83 at debug
            # resolution -- specks, not distractions. A distractor has to be
            # comparable to the thing it competes with for attention, which is
            # what MOVi does by drawing its distractors from the same size
            # distribution as its objects.
            r = actor_r * float(rng.uniform(*DISTRACTOR_SIZE))
            # The band starts OUTSIDE the exclusion, not at a fixed fraction of
            # the frame. A falling actor's keep-clear column can be wider than
            # the inner edge of a fixed band, and then most candidates are
            # rejected before they are even considered -- `drop` and
            # `pyramid_impact` placed zero distractors on some seeds that way.
            # Deriving the inner radius from the exclusion means the sampler is
            # always drawing from somewhere it can succeed.
            # The body is BUILT FIRST, then tested, because `vary_dims` can
            # grow it: checking clearance against the drawn radius and then
            # stretching the body afterwards let an approved distractor end up
            # occluding the actor after all.
            kind = pick_shape(rng)
            probe = vary_dims(BodySpec(
                name="probe", kind=kind, position=(0.0, 0.0, 0.0),
                scale=(r,) * 3, material=None), rng)
            r = float(probe.bounding_radius)
            inner = max(0.30 * extent, exclusion + r)
            outer = max(inner * 1.25, 0.75 * extent)
            rad = float(rng.uniform(inner, outer))
            pos = np.array([aim[0] + rad * np.cos(ang),
                            aim[1] + rad * np.sin(ang), floor_top + r])
            # AIRBORNE SOMETIMES, and decided HERE -- before anything is
            # checked. If only the actor ever falls, "the thing that falls"
            # identifies the actor without looking at physics.
            #
            # It used to be decided after the sightline test, and lifting a
            # body changes where it is on screen: a candidate cleared at floor
            # height rose straight into the actor's line, which is the same
            # mistake as approving a clearance and then growing the body.
            # Measured on `toss`: `distractor_04` ended 0.109 rad from the ball
            # against the 0.131 it needed.
            if float(rng.uniform()) < DISTRACTOR_AIRBORNE:
                pos[2] = floor_top + r + float(rng.uniform(0.5, 2.5)) * r
            if any(float(np.linalg.norm(pos - c)) < (keep + r)
                   for c, keep in keep_clear):
                continue
            # AND NOT IN FRONT OF IT. Distance in world space is not enough: a
            # distractor three metres from the actor can still sit squarely
            # between the actor and the camera, and then it does not distract,
            # it OCCLUDES. Measured when this check was missing -- the actor
            # lost a frame of visibility and its `severity_map` went from 0.23
            # to empty, because severity is painted through the segmentation of
            # a body the camera can no longer see.
            if _occludes(eye, pos, r, sightlines):
                continue
            # AND VISIBLE. A distractor the camera cannot see is not a
            # distractor, it is render time -- one placed off the edge of the
            # frame came back with zero pixels.
            if not cam.visible(spec.camera_position, spec.camera_look_at,
                               pos[None, :])[0]:
                continue
            if any(float(np.linalg.norm(pos - np.asarray(o.position))) <
                   (r + float(o.bounding_radius)) * 1.15 for o in out):
                continue
            # MOVING OR INERT, decided per body rather than drawn as a speed
            # that happens to be small. Static clutter lets a model find the
            # subject by asking what moves; uniformly drifting clutter lets it
            # ask the same question the other way round. A mixture answers
            # neither.
            if float(rng.uniform()) < DISTRACTOR_MOVING:
                ref = max(float(np.linalg.norm(actor_v)), 0.6)
                speed = ref * float(rng.uniform(*DISTRACTOR_SPEED))
                heading = float(rng.uniform(0.0, 2.0 * np.pi))
                vel = (speed * np.cos(heading), speed * np.sin(heading), 0.0)
                spin = tuple(float(rng.uniform(-2.5, 2.5)) for _ in range(3))
            else:
                vel, spin = (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)

            peer = role == "actor"
            body = BodySpec(
                name=("peer_%02d" if peer else "distractor_%02d") % i,
                kind=kind, position=tuple(pos),
                scale=probe.scale, velocity=vel, angular_velocity=spin,
                mass=1.0,
                friction=float(rng.uniform(0.2, 0.8)),
                restitution=float(rng.uniform(0.1, 0.6)),
                color=hue_rgb(float(rng.uniform(0, 1))),
                segmentation_id=(SEG_PEER_BASE if peer else SEG_DISTRACTOR_BASE) + i,
                role=role)
            out.append(with_material(body, M.pick(rng), rng))
            break
    return out


def _occludes(eye, pos, radius: float, keep_clear) -> bool:
    """Would a body at `pos` hide any of the protected points from `eye`?

    An angular test rather than a projection: a candidate occludes when it sits
    NEARER the camera than the thing it would hide, and its angular radius
    overlaps that thing's. Nearer matters -- a body behind the actor is a
    backdrop, which is what a distractor is for.
    """
    import numpy as np

    to_c = np.asarray(pos, np.float64) - eye
    d_c = float(np.linalg.norm(to_c))
    if d_c < 1e-6:
        return True
    for centre, keep in keep_clear:
        to_a = np.asarray(centre, np.float64) - eye
        d_a = float(np.linalg.norm(to_a))
        if d_a < 1e-6 or d_c >= d_a:
            continue                      # behind the actor: harmless backdrop
        cos = float(np.clip(np.dot(to_c / d_c, to_a / d_a), -1.0, 1.0))
        sep = np.arccos(cos)
        if sep < (np.arctan2(radius, d_c) + np.arctan2(keep, d_a)):
            return True
    return False
