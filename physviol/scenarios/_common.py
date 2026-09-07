"""Shared scenario building blocks.

Scenario files stay short because everything generic lives here: the ground
plane (which becomes an HDRI dome from complexity L1 up), standard lighting and
a colour helper. A new scenario is then mostly a description of *what is staged*,
not renderer plumbing.
"""
from __future__ import annotations

import colorsys
import math
from typing import List, Tuple

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
    this function was written to protect. `test_complexity_twin` catches it.
    """
    import numpy as np
    import zlib

    mixed = int(seed) * 2654435761 + 0x9E3779B9
    if salt:
        mixed += int(zlib.crc32(salt.encode()))
    return np.random.RandomState(mixed % (2 ** 31 - 1))


def ground(cx: Complexity, seg_id: int, size: float = 6.0) -> BodySpec:
    """At L0 a plain cube; from L1 up KuBasic's `dome`, which doubles as the
    HDRI backdrop -- the same trick MOVi uses."""
    if cx.background == "hdri":
        return BodySpec(name="floor", kind="dome", position=(0.0, 0.0, 0.0),
                        mass=0.0, static=True, friction=0.6, restitution=0.4,
                        segmentation_id=seg_id, role="floor")
    return BodySpec(name="floor", kind="cube", position=(0.0, 0.0, -0.1),
                    scale=(size, size, 0.1), mass=0.0, static=True,
                    friction=0.6, restitution=0.4, color=(0.32, 0.33, 0.36),
                    segmentation_id=seg_id, role="floor")


def lights(cx: Complexity, look_at=(0.0, 0.0, 0.6),
           scale: float = 1.0) -> List[LightSpec]:
    """An HDRI environment lights the scene on its own; only L0 needs a sun.

    `scale` is the scene's linear size relative to the hand-tuned default of
    roughly four metres across -- see `camera.REFERENCE_HALF_EXTENT`. The
    free-flight scenarios size themselves from the clip length, and a sun left
    at 4.5 m while the actor arcs twenty metres up would be a lamp *inside* the
    trajectory, lighting the underside of everything.
    """
    if cx.background == "hdri":
        return []
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


def with_material(body: BodySpec, name: str, rng, look=None) -> BodySpec:
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
    rgb, roughness, metallic = M.appearance(name, rng) if look is None else look
    return dataclasses.replace(
        body, material=name, color=rgb, roughness=roughness, metallic=metallic,
        mass=M.mass_for(name, body.scale, body.kind))


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

    SPHERES ARE LEFT ALONE, deliberately. PyBullet has no ellipsoid primitive:
    a non-uniformly scaled `kb.Sphere` renders as an ellipsoid but keeps a
    round collider, so the picture and the physics would disagree -- exactly
    the bug `stepper.ShapeSwap` exists to avoid, and it can only do so because
    it builds a convex hull per intervention. Anisotropy here would also break
    the rolling-without-slipping spin the rolling scenarios set as
    `omega = v / r`, which has no meaning for a body with two radii.
    """
    if body.kind not in ("cube", "cylinder", "cone"):
        return body
    import dataclasses

    sx, sy, sz = (float(v) for v in body.scale)
    lo, hi = 1.0 / float(max_aspect) ** 0.5, float(max_aspect) ** 0.5
    if body.kind == "cube":
        scale = (sx * float(rng.uniform(lo, hi)),
                 sy * float(rng.uniform(lo, hi)),
                 sz * float(rng.uniform(lo, hi)))
    else:
        # A cylinder or cone has one radius and a height; varying x and y
        # independently would make it an elliptical cylinder, which Kubric's
        # primitive is not.
        r = sx * float(rng.uniform(lo, hi))
        scale = (r, r, sz * float(rng.uniform(lo, hi)))
    x, y, z = (float(v) for v in body.position)
    out = dataclasses.replace(body, scale=scale,
                              position=(x, y, z + scale[2] - sz))
    # Mass follows the new volume when the body is made of something.
    if body.material is not None:
        from . import materials as M
        out = dataclasses.replace(
            out, mass=M.mass_for(body.material, scale, body.kind))
    return out
