"""Scenario scaffolding -- docs/PLAN.md Part 2 Level 3.

A scenario emits a declarative SceneSpec, never Kubric objects directly. That
keeps scenario sampling testable on the host (no Kubric import) and leaves the
container worker as the only place that touches `bpy`/`kb`.

py3.9-compatible: imported inside the container.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# --------------------------------------------------------------------------
# Tiers -- docs/PLAN.md "The resolution ladder"
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Tier:
    name: str
    resolution: int
    fps: int
    num_frames: int
    samples_per_pixel: int
    latent_frames: int      # (num_frames - 1) // 4 + 1, must be exact
    latent_hw: int
    publishable: bool

    def validate(self) -> None:
        assert (self.num_frames - 1) % 4 == 0, (
            "%s: num_frames must be 4k+1 for exact VAE latent alignment, got %d"
            % (self.name, self.num_frames))
        assert (self.num_frames - 1) // 4 + 1 == self.latent_frames, self.name

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "resolution": self.resolution,
                "fps": self.fps, "num_frames": self.num_frames,
                "samples_per_pixel": self.samples_per_pixel,
                "latent_frames": self.latent_frames,
                "latent_hw": self.latent_hw, "publishable": self.publishable}

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Tier":
        """Rebuild a tier from a spec, overrides and all.

        Annotation used to recover the tier with `TIERS[spec["tier"]]`, which
        works only while every tier is one of the three named ones. The moment
        `--frames`/`--resolution` produce a `v0+f25`, that lookup raises -- so
        the spec carries the whole record and the name is just a label.
        """
        return Tier(**{k: d[k] for k in
                       ("name", "resolution", "fps", "num_frames",
                        "samples_per_pixel", "latent_frames", "latent_hw",
                        "publishable")})

    def override(self, resolution: Optional[int] = None,
                 fps: Optional[int] = None, num_frames: Optional[int] = None,
                 samples_per_pixel: Optional[int] = None) -> "Tier":
        """A tier with individual dials changed, keeping the rest.

        For sweeping one knob without inventing a whole new tier -- "the v0
        settings but 25 frames", "debug but at 64 spp". `latent_frames` is
        recomputed rather than carried over, so a frame count that does not
        align to the VAE stride fails here with an explanation instead of
        silently shipping clips no latent model can consume evenly.

        The name records what was changed, so it lands in `meta.json` as e.g.
        `v0+f25` and two clips from different overrides are never confused.
        """
        changes = []
        for key, value in (("res", resolution), ("fps", fps),
                           ("f", num_frames), ("spp", samples_per_pixel)):
            if value is not None:
                changes.append("%s%d" % (key, value))
        if not changes:
            return self
        frames = self.num_frames if num_frames is None else int(num_frames)
        if (frames - 1) % 4 != 0:
            raise ValueError(
                "num_frames must be 4k+1 for exact VAE latent alignment "
                "(…, 13, 17, 21, 25, 29, …); got %d. Nearest are %d and %d."
                % (frames, frames - (frames - 1) % 4,
                   frames + 4 - (frames - 1) % 4))
        out = Tier(
            name="%s+%s" % (self.name, "".join(changes)),
            resolution=self.resolution if resolution is None else int(resolution),
            fps=self.fps if fps is None else int(fps),
            num_frames=frames,
            samples_per_pixel=(self.samples_per_pixel
                               if samples_per_pixel is None
                               else int(samples_per_pixel)),
            latent_frames=(frames - 1) // 4 + 1,
            latent_hw=self.latent_hw if resolution is None else max(
                1, int(resolution) // 16),
            publishable=self.publishable)
        out.validate()
        return out


# Named for what they are, not lettered. The letters were inherited and made no
# sense: A was the first release, B the second, D the debug size, and there was
# no C at all -- so the ordering implied by the alphabet was backwards from the
# ordering that matters, and every reader had to memorise a lookup. These are
# the release names already used for the output directories.
TIERS: Dict[str, Tier] = {
    #                             res  fps  frames  spp  F_lat  HW_lat  publish
    "debug": Tier("debug", 128, 12, 25, 16, 7, 8, False),
    "v0":    Tier("v0",    512, 30, 89, 64, 23, 16, True),
    "v1":    Tier("v1",    512, 30, 89, 64, 23, 16, True),
}
# v0 is 512x512 / 30 fps / 89 frames = 2.97 s.
#
# 30 fps so the release downsamples cleanly to 15 and 10 without resampling,
# which a 12 fps master cannot do. 89 frames rather than 90 because every tier's
# frame count must be 4k+1 for exact VAE latent alignment -- 90 is not, and 89
# is the closest that is.
#
# `latent_hw` stays 16 at 512 rather than following the /16 rule the debug tier
# uses: the token grid is a fixed 16x16 for both published tiers so a model
# trained on one transfers to the other without reshaping.
#
# **v0 and v1 are the same tier geometry on purpose.** v1 is not a bigger
# render, it is the same physics under harder conditions -- photographic
# backgrounds and objects (complexity L1) and crowded scenes (population
# multi). Making it a resolution step as well would confound the two: a model
# that scored worse on v1 could be failing at realism, at clutter, or merely at
# a resolution it had not been trained on, and the release could not say which.
# Keeping the geometry fixed makes v0 and v1 *paired*, which is the comparison
# the axis exists for. See docs/roadmap.md.

#: The letters this project used until 2026-08-24. Accepted with a pointer to
#: the new name rather than silently, so an old script or a stale note fails
#: loudly and readably instead of building the wrong size.
LEGACY_TIER_NAMES = {"D": "debug", "A": "v0", "B": "v1"}


def tier(name: str) -> Tier:
    """A tier by name, with a readable error for the retired letters."""
    if name in TIERS:
        return TIERS[name]
    if name in LEGACY_TIER_NAMES:
        raise KeyError("tier %r was renamed to %r (the letters had no C and "
                       "ran backwards); valid tiers: %s"
                       % (name, LEGACY_TIER_NAMES[name], ", ".join(TIERS)))
    raise KeyError("unknown tier %r; valid tiers: %s" % (name, ", ".join(TIERS)))
# Frame counts went up across the board (13/25/97 -> 25/49/97). At thirteen
# frames a violation that fires a third of the way in has eight frames to play
# out, which is not enough to see a body rise and fall, or a pour drain through
# a floor, or a stack finish toppling -- and every window had to be squeezed
# into the space left over. Timing is expressed as a fraction of the clip
# (see `EVENT_FRACTION`), so lengthening the tier lengthens the violations too
# rather than leaving them as brief events in a longer static shot.
DEFAULT_TIER = "debug"

for _t in TIERS.values():
    _t.validate()


# --------------------------------------------------------------------------
# Complexity -- an augmentation axis orthogonal to severity
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Complexity:
    """How realistic the *scene* is, independent of how wrong the physics is.

    Severity asks "how badly is the law broken"; complexity asks "how hard is
    the scene to parse". They are orthogonal, and reporting accuracy across
    both is what separates "the model understands physics" from "the model
    copes with clutter". Mirrors the MOVi ladder.

    Asset availability was measured against the pinned image, not assumed:
    GSO 1033 objects (~0.9 s/fetch), HDRI Haven 509 environments (~1.8 s), and
    KuBasic 15 primitives including the `dome` used as an HDRI backdrop.
    """

    name: str
    background: str        # "solid" | "hdri"
    actor_assets: str      # "primitive" | "gso"
    n_distractors: int
    camera_motion: str     # "static" | "orbit" | "linear"
    motion_blur: float
    movi_analogue: str
    implemented: bool

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "background": self.background,
                "actor_assets": self.actor_assets,
                "n_distractors": self.n_distractors,
                "camera_motion": self.camera_motion,
                "motion_blur": self.motion_blur}


COMPLEXITY: Dict[str, Complexity] = {
    "L0": Complexity("L0", "solid", "primitive", 0, "static", 0.0, "MOVi-A", True),
    "L1": Complexity("L1", "hdri", "primitive", 0, "static", 0.0, "MOVi-B", True),
    "L2": Complexity("L2", "hdri", "gso", 6, "static", 0.0, "MOVi-C", False),
    "L3": Complexity("L3", "hdri", "gso", 12, "linear", 0.0, "MOVi-D/E", False),
    "L4": Complexity("L4", "hdri", "gso", 20, "linear", 0.5, "MOVi-F", False),
}
DEFAULT_COMPLEXITY = "L0"


def implemented_complexities() -> List[str]:
    return [k for k, v in COMPLEXITY.items() if v.implemented]


# --------------------------------------------------------------------------
# Declarative scene description
# --------------------------------------------------------------------------


@dataclass
class BodySpec:
    """One rigid body. `segmentation_id` is what shows up in seg.npz."""

    name: str
    kind: str                                  # "sphere" | "cube" | "dome"
    position: Tuple[float, float, float]
    scale: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    quaternion: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    angular_velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    mass: float = 1.0
    friction: float = 0.5
    restitution: float = 0.5
    static: bool = False
    color: Tuple[float, float, float] = (0.7, 0.7, 0.75)
    segmentation_id: int = 0
    role: str = "prop"        # prop | floor | occluder | actor | shadow

    #: What the body is made of -- see `scenarios/materials.py`. `None` keeps
    #: the literal `mass`, `color` and default shading, which is what floors,
    #: ramps, walls and screens want: they are scenery, not objects, and giving
    #: them materials would put metallic backdrops in the dataset for no gain.
    #:
    #: When set, the scenario derives `mass` from `density x volume` and takes
    #: `color`/`roughness`/`metallic` from the material, so a heavy object
    #: LOOKS heavy. That agreement is the whole point: a randomly-drawn mass a
    #: viewer cannot see turns lawful motion into something that reads as a
    #: violation.
    material: Optional[str] = None
    roughness: float = 0.55
    metallic: float = 0.0

    # Motion is written by the scenario rather than solved by the simulator --
    # a pendulum bob on a rigid rod, say, which PyBullet would need a joint for
    # and Kubric exposes no joints. The body is `static` to the simulator so it
    # neither falls nor generates contacts, but it counts as *dynamic*
    # everywhere downstream so residuals and masks still treat it as an actor.
    scripted: bool = False

    # Declared but absent: the body is in the scene graph from frame 0 and
    # parked off camera, contributing no pixels. `fission` needs a second body
    # to spawn, and a body cannot be added to a Kubric scene mid-render without
    # perturbing the very render path prefix identity depends on -- so the
    # understudy is always there, and the intervention only switches it on.
    dormant: bool = False

    # Cycles ray visibility. `shadow_track` uses these to hand the cast shadow
    # to a body of its own, which is the only way a shadow can carry a
    # segmentation id and therefore a mask.
    visible_camera: bool = True
    visible_shadow: bool = True

    # What the renderer draws, when that differs from what the simulator is
    # given. Kubric's PyBullet wrapper asserts uniform scaling on spheres, so a
    # flattened disc -- which is what a round object's shadow looks like -- can
    # only be declared as a uniform sphere and squashed afterwards. Applied
    # once the body has joined the scene, and only the renderer observes
    # `scale`, so the physics never sees it.
    render_scale: Optional[Tuple[float, float, float]] = None

    @property
    def draw_scale(self) -> Tuple[float, float, float]:
        return self.render_scale or self.scale

    @property
    def bounding_radius(self) -> float:
        return float(max(self.scale))

    @property
    def sim_static(self) -> bool:
        """What the simulator is told. Scripted bodies are pinned there and
        animated from the trajectory instead."""
        return bool(self.static or self.scripted)

    @property
    def dynamic(self) -> bool:
        """What annotation believes. A scripted actor is a moving body."""
        return not bool(self.static)


@dataclass
class LightSpec:
    name: str
    position: Tuple[float, float, float]
    look_at: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    intensity: float = 1.5
    kind: str = "directional"


@dataclass
class SceneSpec:
    """Everything needed to build, simulate and render a clip -- deterministically."""

    scenario: str
    seed: int
    tier: Tier
    bodies: List[BodySpec]
    lights: List[LightSpec]
    camera_position: Tuple[float, float, float]
    camera_look_at: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    gravity: Tuple[float, float, float] = (0.0, 0.0, -9.81)
    background_color: Tuple[float, float, float] = (0.05, 0.05, 0.06)
    floor_level: float = 0.0
    physics_medium: str = "rigid"
    complexity: str = DEFAULT_COMPLEXITY
    hdri_id: Optional[str] = None
    notes: Dict[str, Any] = field(default_factory=dict)
    #: (azimuth, elevation) degrees the camera may swing around its hand-framed
    #: position -- see `_vary`. The default suits a grounded scenario, where the
    #: frustum has slack; flight scenarios override it tighter, because
    #: `camera.frame_flight` already spends most of that slack fitting a tall,
    #: thin arc from one specific side-on angle (see its docstring for the
    #: `toss`/`tumble` regression a wide swing would reintroduce).
    camera_jitter_deg: Tuple[float, float] = (35.0, 12.0)

    #: Where the camera ENDS, when it moves. `None` -- the usual case -- is a
    #: static camera and the whole rest of the pipeline's assumption.
    #:
    #: A linear translation with the aim point held fixed, which is MOVi-D/E's
    #: `linear_movement`. The aim stays put on purpose: a camera that also pans
    #: makes "did the object move, or did the camera?" a question the clip
    #: cannot answer, and every violation here is a claim about object motion.
    #:
    #: Drawn in `_vary`, so it is a pure function of the seed and the host can
    #: reconstruct it without reading anything the container wrote.
    camera_end_position: Optional[Tuple[float, float, float]] = None

    #: Set False by a scenario that cannot tolerate a moving camera. Only
    #: `occluder_pass` does: it precomputes its occlusion interval from a
    #: single camera pose, and that list is where every observability label in
    #: the dataset comes from.
    camera_motion: bool = True

    #: Which motion this clip uses: "static", "track", "orbit" or "dolly".
    #: Shipped in `meta.json` so a consumer can condition on it.
    camera_motion_kind: str = "static"

    #: Which randomisation of this cell this is: variant 0 is the first. Needed
    #: because some choices have to be spread ACROSS a scenario's variants
    #: rather than drawn independently per scene -- see `_maybe_move_camera`.
    variant: int = 0

    @property
    def camera_moves(self) -> bool:
        return self.camera_end_position is not None

    def camera_at(self, frame: int, num_frames: int
                  ) -> Tuple[Tuple[float, float, float],
                             Tuple[float, float, float]]:
        """The camera pose on one frame. Constant unless the camera moves.

        The single source of truth for where the camera was: the renderer
        keyframes from it, the framing guards evaluate against it, and
        `meta.json` ships what it returns. If any of those computed the path
        separately they could disagree, and a violation would be fitted to a
        frustum the clip never had.

        `track` and `dolly` interpolate position linearly. `orbit` interpolates
        the ANGLE about the aim point instead: lerping its endpoints would cut
        the chord, so the camera would dip towards the subject mid-arc and the
        actor would swell and shrink again -- turning the one motion that
        preserves apparent size into the one that does not.
        """
        if self.camera_end_position is None or num_frames <= 1:
            return self.camera_position, self.camera_look_at
        t = min(max(int(frame), 0), num_frames - 1) / float(num_frames - 1)
        start = np.asarray(self.camera_position, np.float64)
        end = np.asarray(self.camera_end_position, np.float64)
        if self.camera_motion_kind != "orbit":
            return (tuple(float(v) for v in (1.0 - t) * start + t * end),
                    self.camera_look_at)

        aim = np.asarray(self.camera_look_at, np.float64)
        u, v = start - aim, end - aim
        ru, rv = float(np.linalg.norm(u)), float(np.linalg.norm(v))
        if ru < 1e-9 or rv < 1e-9:
            return self.camera_position, self.camera_look_at
        cos = float(np.clip(np.dot(u / ru, v / rv), -1.0, 1.0))
        ang = math.acos(cos)
        if ang < 1e-6:
            eye = (1.0 - t) * start + t * end
        else:
            # Slerp on the direction, lerp on the radius. The radius is equal
            # at both ends by construction, so this holds it fixed.
            s = math.sin(ang)
            eye = aim + ((math.sin((1.0 - t) * ang) / s) * u
                         + (math.sin(t * ang) / s) * v)
        return tuple(float(x) for x in eye), self.camera_look_at

    @property
    def actors(self) -> List[BodySpec]:
        """Every body the violation may act on, in declaration order."""
        return [b for b in self.bodies if b.role == "actor"]

    def body(self, name: str) -> BodySpec:
        for b in self.bodies:
            if b.name == name:
                return b
        raise KeyError("no body named %r in %s" % (name, self.scenario))

    def index_of(self, name: str) -> int:
        for i, b in enumerate(self.bodies):
            if b.name == name:
                return i
        raise KeyError(name)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario": self.scenario, "seed": self.seed, "tier": self.tier.name,
            "tier_spec": self.tier.to_dict(),
            "resolution": [self.tier.resolution] * 2, "fps": self.tier.fps,
            "num_frames": self.tier.num_frames, "gravity": list(self.gravity),
            "physics_medium": self.physics_medium,
            "complexity": COMPLEXITY[self.complexity].to_dict(),
            "hdri_id": self.hdri_id,
            "camera_position": list(self.camera_position),
        "camera_end_position": (list(self.camera_end_position)
                                if self.camera_end_position else None),
        "camera_motion_kind": self.camera_motion_kind,
        "variant": int(self.variant),
            "camera_look_at": list(self.camera_look_at),
            "bodies": [{"name": b.name, "kind": b.kind, "role": b.role,
                        "segmentation_id": b.segmentation_id, "mass": b.mass,
                        "restitution": b.restitution, "friction": b.friction,
                        "static": b.static, "scripted": b.scripted,
                        "dormant": b.dormant, "color": list(b.color), "material": b.material,
                        "render_scale": list(b.draw_scale),
                        "scale": list(b.scale)} for b in self.bodies],
            "notes": self.notes,
        }


def _vary(spec: SceneSpec, seed: int) -> SceneSpec:
    """Per-instance appearance variation: viewpoint, and light direction at L0.

    Drawn from its own generator seeded off the scenario's seed, so adding a
    variation axis here never shifts the random stream a scenario staged its
    physics from -- otherwise every existing clip would resample the moment
    this function grew a line.

    The camera swings around `camera_look_at` at fixed *distance* -- an azimuth
    and elevation rotation, not an xyz nudge -- so `camera.frame_extent()`
    (which depends only on that distance) cannot drift and accidentally shrink
    or blow out the frustum: only the viewing angle changes, which is the
    point. The swing is still bounded by `spec.camera_jitter_deg`: the camera
    must keep framing the event, and every scenario's position was hand-derived
    for one side-on angle -- see that field's docstring.
    """
    rng = np.random.RandomState((seed * 2654435761 + 0x5EED) % (2 ** 31 - 1))

    eye = np.asarray(spec.camera_position, np.float64)
    target = np.asarray(spec.camera_look_at, np.float64)
    v = eye - target
    radius = float(np.linalg.norm(v))
    if radius > 1e-9:
        az0 = math.atan2(v[1], v[0])
        el0 = math.asin(float(np.clip(v[2] / radius, -1.0, 1.0)))
        az_max, el_max = spec.camera_jitter_deg
        az = az0 + math.radians(float(rng.uniform(-az_max, az_max)))
        el = el0 + math.radians(float(rng.uniform(-el_max, el_max)))
        el = float(np.clip(el, math.radians(-85.0), math.radians(85.0)))
        v = radius * np.array([math.cos(el) * math.cos(az),
                               math.cos(el) * math.sin(az),
                               math.sin(el)])
        eye = target + v
    spec.camera_position = tuple(float(x) for x in eye)

    aim = rng.uniform(-1.0, 1.0, size=3) * 0.10
    spec.camera_look_at = tuple(float(a + b) for a, b in
                                zip(spec.camera_look_at, aim))
    for light in spec.lights:
        light.position = tuple(float(a + b) for a, b in
                               zip(light.position,
                                   rng.uniform(-1.0, 1.0, size=3) * 0.9))
    _recolour_scenery(spec, seed)
    _maybe_move_camera(spec, seed)
    return spec


#: One variant in every `CAMERA_MOTION_PERIOD` moves its camera -- so 2 of 10,
#: 1 of 5, PER SCENARIO.
#:
#: Spread across a scenario's variants rather than drawn independently per
#: scene. An independent coin flip gives the right share overall and an uneven
#: one per scenario: measured across thirteen scenarios it ranged from 13% to
#: 31%, so some scenarios had moving cameras and others effectively did not,
#: and any per-scenario comparison inherited that as a confound.
#:
#: The variant INDEX decides, not the seed, and the moving one is the LAST of
#: each block. That is what makes a short run entirely static: four variants
#: are 0-3 and none of them is index 4. A run only starts spending clips on
#: camera motion once it is long enough to afford them, and `--camera-motion`
#: overrides this for anyone who wants one on purpose.
CAMERA_MOTION_PERIOD = 5
MOVING_CAMERA_SHARE = 1.0 / CAMERA_MOTION_PERIOD

#: The motions, and how often each is chosen among the clips that move.
#:
#: `track`  slides across the view with the aim held  -- parallax, no size change
#: `orbit`  swings around the subject at fixed radius -- new angle, no size change
#: `dolly`  approaches or retreats along the view axis
#:
#: A panning variant (translate AND swing the aim, MOVi's
#: `linear_movement_linear_lookat`) is deliberately absent: with both moving,
#: "did the object move or did the camera?" stops being answerable from the
#: clip, and every violation here is a claim about object motion.
CAMERA_MOTION_KINDS = ("track", "orbit", "dolly")
CAMERA_MOTION_WEIGHTS = (0.40, 0.40, 0.20)

#: How far a `track` or `orbit` camera travels, as a fraction of its distance
#: to the aim point (`orbit` reads it as an arc length). Small on purpose:
#: every scenario's viewpoint was derived for one shot, and a camera that
#: travels a third of its own standoff has turned it into a different shot --
#: the actor drifts out of frame, and every guard that fitted a violation to
#: the frustum fitted it to one the clip no longer has.
CAMERA_TRAVEL = (0.10, 0.22)

#: How much a `dolly` changes its distance to the subject, as a fraction.
#:
#: Tighter than the others, because this is the motion that changes APPARENT
#: SIZE -- the exact cue `immutability` and `deformation` make their claim
#: about. At 12% the size change is a third of `deformation`'s weakest bin
#: (1.35 aspect), and unlike a deformation it scales the floor and every other
#: body by the same amount, so the scene still says "the camera moved" rather
#: than "that object changed".
DOLLY_RANGE = (0.06, 0.12)


def _maybe_move_camera(spec: SceneSpec, seed: int) -> None:
    """One clip in five gets a moving camera, of one of three kinds.

    Deliberately NOT on the complexity ladder. L3 and L4 declare
    `camera_motion="linear"` and neither is built, so tying motion to them
    would mean no moving-camera clips until GSO assets and distractors arrive
    as well -- and the axis being exercised here is viewpoint, not realism.

    The aim point never moves; see `camera_end_position`.
    """
    # A debug override, for looking at one motion without hunting for a seed
    # that happens to draw it:
    #
    #     PHYSLOC_CAMERA_MOTION=orbit  python -m physloc.cli generate ...
    #
    # `off` forces every clip static; a kind name forces that kind on every
    # clip that is allowed to move; `always` picks among the kinds as usual but
    # never declines. Unset is the real behaviour, and this is the only thing
    # in the sampler that reads the environment -- it must never be set during
    # a release run, which is why it is named this loudly.
    import os

    forced = os.environ.get("PHYSLOC_CAMERA_MOTION", "").strip().lower()
    if forced in ("off", "static", "none"):
        return
    if not spec.camera_motion:
        return
    # Its own stream, salted by scenario. Sharing `_vary`'s would do two bad
    # things: appending a draw there shifts every camera angle already
    # sampled, and a stream keyed on the seed alone makes the decision
    # identical across scenarios -- measured at exactly 22% for all thirteen,
    # which is one coin flip reported thirteen times, not thirteen flips.
    import zlib

    rng = np.random.RandomState(
        (int(seed) * 2654435761 + 0xCA31 + zlib.crc32(spec.scenario.encode()))
        % (2 ** 31 - 1))
    if not forced and (int(spec.variant) % CAMERA_MOTION_PERIOD
                       != CAMERA_MOTION_PERIOD - 1):
        return

    eye = np.asarray(spec.camera_position, np.float64)
    target = np.asarray(spec.camera_look_at, np.float64)
    view = eye - target
    standoff = float(np.linalg.norm(view))
    if standoff < 1e-6:
        return
    view = view / standoff
    right = np.cross(np.array([0.0, 0.0, 1.0]), view)
    if float(np.linalg.norm(right)) < 1e-6:
        right = np.array([1.0, 0.0, 0.0])
    right = right / float(np.linalg.norm(right))
    up = np.cross(view, right)

    kind = str(CAMERA_MOTION_KINDS[int(np.searchsorted(
        np.cumsum(CAMERA_MOTION_WEIGHTS), float(rng.uniform())))])
    if forced in CAMERA_MOTION_KINDS:
        kind = forced
    theta = float(rng.uniform(0.0, 2.0 * math.pi))

    if kind == "dolly":
        # Signed, so the camera pulls back as often as it closes in. Pulling
        # back is the safer half and there is no reason to prefer one.
        frac = float(rng.uniform(*DOLLY_RANGE)) * float(rng.choice([-1.0, 1.0]))
        end = target + view * (standoff * (1.0 + frac))
    elif kind == "orbit":
        # An arc of the same length a `track` would travel, so the two motions
        # are comparable in how much the view changes. The end point sits on
        # the same sphere; `camera_at` interpolates the angle, not the chord.
        arc = float(rng.uniform(*CAMERA_TRAVEL))
        axis = math.cos(theta) * right + math.sin(theta) * up
        axis = axis / float(np.linalg.norm(axis))
        end = target + standoff * (math.cos(arc) * view + math.sin(arc) * axis)
    else:                                                        # "track"
        travel = standoff * float(rng.uniform(*CAMERA_TRAVEL))
        end = eye + travel * (math.cos(theta) * right + math.sin(theta) * up)

    spec.camera_motion_kind = kind
    spec.camera_end_position = tuple(float(v) for v in end)


#: How far the scenery must stay from every actor, in CIE-Lab distance. Below
#: about 25 two colours read as "the same, slightly off" rather than as
#: different things, and an actor that close to the floor it is rolling on is
#: hard to pick out by eye -- which matters when reviewing whether a mask
#: landed on the right object is done by looking.
MIN_SCENERY_SEPARATION = 26.0

#: The floor's lightness band. Not the full range: a black floor swallows
#: `shadow_track`'s cast shadow, whose entire subject is a shadow you can see,
#: and a white one blows out under the L0 sun and takes the contact region
#: with it. Saturation stays low because a vivid floor competes with the actor
#: for attention and the actor is the thing being annotated.
#: Wide, because this is the axis separation is bought on: with a mid-grey
#: actor a narrow band leaves nowhere to go, and the floor ends up 22 Lab from
#: the ball rolling over it. The scan below picks a point in this box, so the
#: extremes are only reached when the actors force it.
FLOOR_VALUE = (0.10, 0.82)
FLOOR_SATURATION = (0.03, 0.34)

#: The void behind the scene at L0. Kept darker than the floor so the horizon
#: still reads as a horizon rather than the ground dissolving into the sky.
BACKDROP_VALUE = (0.04, 0.30)
BACKDROP_SATURATION = (0.02, 0.25)


def _recolour_scenery(spec: SceneSpec, seed: int) -> None:
    """Vary the floor and the backdrop, keeping them clear of the actors.

    Both were constants -- the floor a hard-coded grey in `_common.ground`, the
    backdrop the `SceneSpec` default that no scenario ever overrode -- so every
    clip in the dataset shared a background. A model can learn a fixed
    background exactly as easily as it can learn a fixed event frame, and
    neither one is physics.

    Done here rather than in `ground()` because the constraint is about the
    whole scene: the floor has to stay clear of every actor colour, and the
    actors do not exist yet at the point `ground()` is called. Once in `_vary`
    also means all thirteen scenarios get it without thirteen edits.

    Skipped from L1 up, where the floor is the HDRI dome and the environment
    supplies the ground and the sky together.
    """
    import colorsys

    from ..residuals.laws import _srgb_to_lab

    floors = [b for b in spec.bodies if b.role == "floor" and b.kind != "dome"]
    if not floors:
        return
    import zlib

    # Salted by scenario as well as seed, for the reason `appearance_rng`
    # documents: salted by seed alone, every scenario picks the same floor on
    # a given seed and the release ships a correlation nobody asked for.
    rng = np.random.RandomState(
        (seed * 2654435761 + 0xF100D + zlib.crc32(spec.scenario.encode()))
        % (2 ** 31 - 1))
    # The cast shadow counts too. `shadow_track` stages its shadow as a real
    # near-black body, and the family is about whether that shadow tracks the
    # object faithfully -- on a floor the same darkness there is nothing to
    # judge.
    lab_actors = [_srgb_to_lab(np.asarray(b.color, np.float64))
                  for b in spec.bodies if b.role in ("actor", "shadow")]

    def separation(rgb) -> float:
        """Lab distance to the nearest actor -- bigger is better."""
        lab = _srgb_to_lab(np.asarray(rgb, np.float64))
        if not lab_actors:
            return 1e9
        return min(float(np.linalg.norm(lab - la)) for la in lab_actors)

    # CHOOSE the lightness rather than gambling on it. Rejection sampling
    # looked fine and quietly failed: the floor palette is deliberately
    # desaturated, and so are stone, steel, ceramic and cork, so on a scene
    # with a grey actor all twenty-four draws landed inside the exclusion zone
    # and the floor fell back to the hard-coded grey it was trying to replace
    # -- 4.6 Lab from the ball it sat under, on `collision` and `barrier_pass`.
    #
    # Hue and saturation are still drawn; the value is then scanned across its
    # band and the most separated one wins. Lab distance between two
    # desaturated colours is dominated by L*, so this is the axis that actually
    # buys separation, and scanning it always finds the best available instead
    # of giving up.
    hue = float(rng.uniform(0.0, 1.0))
    best, best_gap = floors[0].color, -1.0
    for k in range(17):
        v = FLOOR_VALUE[0] + (FLOOR_VALUE[1] - FLOOR_VALUE[0]) * k / 16.0
        for j in range(3):
            sat = FLOOR_SATURATION[0] + (
                FLOOR_SATURATION[1] - FLOOR_SATURATION[0]) * j / 2.0
            cand = tuple(float(c) for c in colorsys.hsv_to_rgb(hue, sat, v))
            gap = separation(cand)
            if gap > best_gap:
                best, best_gap = cand, gap
    for b in floors:
        b.color = best
    spec.background_color = tuple(float(c) for c in colorsys.hsv_to_rgb(
        float(rng.uniform(0.0, 1.0)),
        float(rng.uniform(*BACKDROP_SATURATION)),
        float(rng.uniform(*BACKDROP_VALUE))))


class Scenario:
    """Base class. Subclasses implement `_sample`."""

    name: str = "unnamed"

    def sample(self, seed: int, tier: Tier,
               complexity: str = DEFAULT_COMPLEXITY,
               variant: int = 0) -> SceneSpec:
        """Sample one instance, then vary how it looks. Do not override.

        Level 4 of the taxonomy is the *instance*, and two instances of one
        scenario should differ in more than their random seed's effect on a
        radius. Subclasses stage the physics in `_sample`; the appearance
        variation that is common to all of them -- where the camera stands, how
        the scene is lit -- is applied here so it cannot drift between thirteen
        files.
        """
        spec = self._sample(seed, tier, complexity)
        spec.variant = int(variant)
        return _vary(spec, seed)

    def _sample(self, seed: int, tier: Tier,
                complexity: str = DEFAULT_COMPLEXITY) -> SceneSpec:
        raise NotImplementedError

    def script(self, spec: "SceneSpec", traj) -> None:
        """Overwrite the simulated rollout for bodies the scenario drives itself.

        Called once, on the *valid* trajectory, right after simulation. Default
        is a no-op; only constrained scenarios (`pendulum_swing`) need it. It
        runs before any injector, so the seam's guarantee is untouched: an
        injector still edits a finished trajectory and the prefix stays
        bit-identical.
        """
        return None

    def sim_hooks(self, spec: "SceneSpec", simulator, objs):
        """Per-substep hooks the SIMULATOR runs, for constraints it has no joint
        for. Default is none, and most scenarios need none.

        The honest alternative to scripting a constrained scenario. A pendulum
        or a rope is a distance constraint, and PyBullet's point-to-point joints
        are not usable for it in the pinned build -- measured, a single rigid-rod
        constraint drifts 18% and dies out within a second, and an 8-link chain
        stretches 250%. What *is* exact is enforcing the constraint directly:
        put the body back on the sphere of radius L about the pivot and take out
        the radial velocity, every substep. Measured, the rod then holds its
        length to six decimal places at 0.19 ms/frame.

        The point is what it leaves alone. The bob stays an ordinary dynamic
        body that Kubric owns, one asset to one PyBullet body, so every staged
        injector addresses it the way it addresses any other body and nothing
        needs a second code path. Scripting the motion instead buys the same
        picture and costs exactly that.

        Declared here rather than in the worker because the constraint belongs
        to the scenario, the same way `family_targets` and the occlusion
        interval do.
        """
        return ()

    @staticmethod
    def rng(seed: int) -> np.random.RandomState:
        # RandomState (not default_rng) so the stream is identical under
        # numpy 1.21 in the container and numpy 2.x on the host.
        return np.random.RandomState(seed)


_REGISTRY: Dict[str, Scenario] = {}


def register(scenario: Scenario) -> Scenario:
    _REGISTRY[scenario.name] = scenario
    return scenario


def get(name: str) -> Scenario:
    if name not in _REGISTRY:
        raise KeyError("unknown scenario %r; have %s" % (name, sorted(_REGISTRY)))
    return _REGISTRY[name]


def available() -> Sequence[str]:
    return sorted(_REGISTRY)
