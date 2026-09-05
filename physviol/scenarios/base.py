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
            "camera_look_at": list(self.camera_look_at),
            "bodies": [{"name": b.name, "kind": b.kind, "role": b.role,
                        "segmentation_id": b.segmentation_id, "mass": b.mass,
                        "restitution": b.restitution, "friction": b.friction,
                        "static": b.static, "scripted": b.scripted,
                        "dormant": b.dormant, "color": list(b.color),
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
    return spec


class Scenario:
    """Base class. Subclasses implement `_sample`."""

    name: str = "unnamed"

    def sample(self, seed: int, tier: Tier,
               complexity: str = DEFAULT_COMPLEXITY) -> SceneSpec:
        """Sample one instance, then vary how it looks. Do not override.

        Level 4 of the taxonomy is the *instance*, and two instances of one
        scenario should differ in more than their random seed's effect on a
        radius. Subclasses stage the physics in `_sample`; the appearance
        variation that is common to all of them -- where the camera stands, how
        the scene is lit -- is applied here so it cannot drift between thirteen
        files.
        """
        return _vary(self._sample(seed, tier, complexity), seed)

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
