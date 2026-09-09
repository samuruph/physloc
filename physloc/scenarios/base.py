"""Scenario scaffolding -- docs/PLAN.md Part 2 Level 3.

A scenario emits a declarative SceneSpec, never Kubric objects directly. That
keeps scenario sampling testable on the host (no Kubric import) and leaves the
container worker as the only place that touches `bpy`/`kb`.

py3.9-compatible: imported inside the container.
"""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# --------------------------------------------------------------------------
# Tiers -- docs/PLAN.md "The resolution ladder"
# --------------------------------------------------------------------------


#: The LOCAL-MESH BOUNDS of every kind a body can be -- the extent of the mesh
#: itself, before `scale` multiplies it. Measured in the pinned image with
#: `AssetSource.from_manifest(KUBASIC).create(...).bounds`, reproduced by
#: `physloc/render/probe_fission.py`.
#:
#: `scale` is NOT a half-extent for anything but a cube and a sphere. A KuBasic
#: cylinder's mesh is half the size of the [-1, +1] cube everything assumed, a
#: torus is a disc a fifth as thick as it is wide, and a cone is not centred on
#: its own origin at all: base at z = -0.306, apex at +0.900. Two things read
#: this -- the collision hull a resize stands in the body's place, and how far
#: apart `fission` sets its halves down.
KIND_BOUNDS = {
    "cube": ((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0)),
    "sphere": ((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0)),
    "cylinder": ((-0.5, -0.5, -0.5), (0.5, 0.5, 0.5)),
    "cone": ((-0.6007, -0.6007, -0.3062), (0.6007, 0.6007, 0.8997)),
    "torus": ((-0.75, -0.75, -0.15), (0.75, 0.75, 0.15)),
}


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
#: TWO GEOMETRIES, because there are only two. `v0` and `v1` used to sit here
#: as separate tiers and differed in NOTHING but their name -- both 512 x 512,
#: 30 fps, 89 frames -- because what actually separated them was complexity,
#: which is its own axis and now its own ladder. A tier that encodes a release
#: number is a tier that has to be renamed every release.
#:
#: So the tier says how big and how long, the complexity ladder says how hard,
#: and `v0`/`v1` are what a published DATASET is called -- set by the outdir,
#: recorded as `release` in every `meta.json`.
TIERS: Dict[str, Tier] = {
    #                              res  fps  frames  spp  F_lat  HW_lat  publish
    "debug":   Tier("debug",   128, 12, 25, 16, 7, 8, False),
    "release": Tier("release", 512, 30, 89, 64, 23, 16, True),
}
# `release` is 512x512 / 30 fps / 89 frames = 2.97 s.
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
# ONE publishable geometry, on purpose. A harder release is not a bigger
# render -- it is the same physics under harder conditions, which is what the
# complexity ladder is for. Making difficulty a resolution step as well would
# confound the two: a model scoring worse could be failing at realism, at
# clutter, or merely at a resolution it had not been trained on, and the
# release could not say which. Fixing the geometry is what makes two releases
# comparable at all.

#: The letters this project used until 2026-08-24. Accepted with a pointer to
#: the new name rather than silently, so an old script or a stale note fails
#: loudly and readably instead of building the wrong size.
LEGACY_TIER_NAMES = {"D": "debug", "A": "release", "B": "release",
                     "v0": "release", "v1": "release"}


def tier(name: str) -> Tier:
    """A tier by name, with a readable error for the retired letters."""
    if name in TIERS:
        return TIERS[name]
    if name in LEGACY_TIER_NAMES:
        why = ("the letters had no C and ran backwards"
               if name in ("A", "B", "D") else
               "a tier is a geometry, not a release number -- `v0` is what a "
               "published dataset is CALLED, set by --outdir")
        raise KeyError("tier %r was renamed to %r (%s); valid tiers: %s"
                       % (name, LEGACY_TIER_NAMES[name], why, ", ".join(TIERS)))
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
    materials: bool        # do bodies have a material, or a flat colour?
    motion_blur: float
    #: This level's share of a full generation, RELATIVE TO L0. A run asking
    #: for `V` variants per cell gives this level `round(V * share)` of them,
    #: and skips it entirely when that rounds to zero -- so a short run is all
    #: baseline and only a long one starts spending clips on realism. Naming a
    #: level explicitly (`--complexity L2`) overrides that.
    share: float
    #: May a clip at this level move its camera at all? WHICH clips do is not a
    #: property of the level -- it is the `CONDITION_CYCLE`, which applies
    #: identically at every level. This is only an off switch, for a level whose
    #: framing cannot survive a moving camera.
    camera_moves: bool
    #: How many extra objects a crowded clip holds is `EXTRA_OBJECTS`, drawn
    #: per clip and identical at every level -- it was a per-level field, and a
    #: quantity of the same thing is not a step of realism.

    implemented: bool

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "background": self.background,
                "actor_assets": self.actor_assets,
                "materials": self.materials,
                "motion_blur": self.motion_blur,
                "share": self.share,
                "camera_moves": self.camera_moves}


#: THE LADDER IS SCENE REALISM. Four levels, each the one below it plus one step
#: of how hard the scene is to look at:
#:
#:   L0  primitives, flat colours, a solid background   -- the baseline
#:   L1  + materials: wood, steel, rubber, whose appearance and density agree
#:   L2  + a real environment, lit by an HDRI
#:   L3  + GSO objects in that environment              -- the hardest
#:
#: **Camera motion and distractors are NOT levels.** They used to be, and it was
#: wrong twice over. A level whose axis fires on only some of its clips is not a
#: stratum at all: with camera motion on 1 variant in 5, "L2 minus L1" was not
#: measuring materials, it was measuring materials plus whichever variants
#: happened to draw a camera move. And making them levels forced a choice nobody
#: wants -- either every realistic clip moves its camera, or none does.
#:
#: They are DIFFICULTY CONDITIONS, one per clip, applied identically inside
#: every level -- see `CONDITION_CYCLE`. So the dataset answers "how much does
#: clutter cost at each realism level" as well as "how much does realism cost",
#: and the share of each condition is a number you set rather than an artefact
#: of how the ladder was climbed.
#:
#: SHARES FALL AS REALISM RISES, for two reasons. The baseline is what everything
#: else is compared against, so it should be the largest stratum; and an HDRI
#: clip costs 44 s against a solid background's 8 s at the debug tier, so the
#: expensive levels are also the ones a fixed budget can afford least of.
#: Normalised, the four come to 50% / 25% / 15% / 10% of a full generation.
COMPLEXITY: Dict[str, Complexity] = {
    #                  bg      actors      mat   blur  share  cam   built
    "L0": Complexity("L0", "solid", "primitive", False, 0.0, 1.00, True, True),
    "L1": Complexity("L1", "solid", "primitive", True,  0.0, 0.50, True, True),
    "L2": Complexity("L2", "hdri",  "primitive", True,  0.0, 0.30, True, True),
    "L3": Complexity("L3", "hdri",  "gso",       True,  0.0, 0.20, True, True),
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
    #: Resistance to ROLLING and spinning, which Kubric's constructor has no
    #: argument for and which therefore defaulted to zero on every body.
    #:
    #: Zero is right for a ball that is supposed to roll and wrong for a
    #: granular medium, because it is the property that lets a pile stand up. A
    #: perfectly smooth sphere with no rolling resistance does not have an angle
    #: of repose: it rolls until something stops it. Measured on `pour`, forty
    #: grains released down a narrow column still ended spread across a 1.4 m
    #: circle exactly ONE GRAIN DEEP -- they rolled apart until they reached the
    #: walls -- and a medium with no interior is one `newton2_mass` cannot
    #: stratify and one `friction` cannot shape. Applied by
    #: `render.worker.build_scene` through `changeDynamics`, since it has to be
    #: set after the body exists.
    rolling_friction: float = 0.0
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
    #: The rest of the PrincipledBSDF surface. Defaults are Blender's, so a
    #: body that never asked for a material renders exactly as it did before.
    #: `transmission` above zero is glass, and `ior` is what bends through it.
    specular: float = 0.5
    transmission: float = 0.0
    ior: float = 1.45

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
    #: Which scanned asset this body IS, when `kind == "gso"`. The id keys
    #: `_gso.GSO_ASSETS`, which carries the mesh bounds the host needs and the
    #: licence every asset is required to ship.
    asset_id: Optional[str] = None

    @property
    def draw_scale(self) -> Tuple[float, float, float]:
        return self.render_scale or self.scale

    @property
    def bounding_radius(self) -> float:
        """A CONSERVATIVE radius, in METRES.

        For a primitive that is `max(scale)`: `scale` is already a length there,
        and since no KuBasic mesh reaches 1.0 in its own coordinates this
        over-estimates for a cylinder, cone or torus. Deliberately left that
        way -- every caller is a clearance or framing margin, where erring
        large is safe.

        For a GSO asset it CANNOT be `scale`. Following MOVi, a scanned object
        is normalised -- `scale = target / longest mesh axis` -- so `scale` is a
        unitless factor that says nothing about how big the thing is drawn: a
        30 mm toy block scaled to 0.4 m carries `scale = 13`. Its real
        half-extent is the answer, and it is still an over-estimate of the
        radius in any direction but the longest.
        """
        if self.kind == "gso":
            return float(max(self.extents))
        return float(max(self.scale))

    @property
    def mesh_bounds(self) -> Tuple[Tuple[float, float, float],
                                   Tuple[float, float, float]]:
        """The body's own mesh extent, before `scale`.

        A primitive's comes from `KIND_BOUNDS`; a GSO asset's comes from the
        baked manifest, because a scanned object has whatever shape it has and
        the host has no container to ask.
        """
        if self.kind == "gso" and self.asset_id:
            from ._gso import GSO_ASSETS

            entry = GSO_ASSETS.get(self.asset_id)
            if entry is not None:
                lo, hi = entry["bounds"]
                return tuple(lo), tuple(hi)
        return KIND_BOUNDS.get(self.kind, KIND_BOUNDS["sphere"])

    @property
    def extents(self) -> Tuple[float, float, float]:
        """Half-extents in the body's own frame, as DRAWN."""
        lo, hi = self.mesh_bounds
        return tuple(float(s) * (float(h) - float(l)) / 2.0
                     for s, l, h in zip(self.draw_scale, lo, hi))

    def half_extent_along(self, direction) -> float:
        """How far the body reaches from its centre along `direction`.

        The support of its bounding ELLIPSOID, which is exact for a sphere and
        for the case that actually needs it: a horizontal direction across an
        upright cone, cylinder or torus, where it returns the mesh's own
        radius. A cube's corner is under-reported, which is why this is not
        what clearance margins use.
        """
        u = np.asarray(direction, np.float64)
        n = float(np.linalg.norm(u))
        if n < 1e-12:
            return float(max(self.extents))
        return float(np.linalg.norm(np.asarray(self.extents) * (u / n)))

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
    #: How many variants this level was allocated, and which difficulty
    #: condition this clip therefore carries. See `condition_for`.
    n_variants: Optional[int] = None
    condition: str = "standard"
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
            "n_variants": self.n_variants,
            "condition": self.condition,
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
    _flatten_materials(spec)
    _add_distractors(spec, seed)
    _add_peers(spec, seed)
    _match_understudies(spec)
    _recolour_scenery(spec, seed)
    _material_scenery(spec, seed)
    _swap_in_gso(spec, seed)
    _pick_hdri(spec, seed)
    _add_backdrop(spec)
    _maybe_move_camera(spec, seed)
    return spec


def _match_understudies(spec: SceneSpec) -> None:
    """A dormant stand-in must LOOK like the body it stands in for.

    `fission` needs a duplicate in the scene graph from frame 0 -- adding an
    object part-way through a render would perturb the render path, and prefix
    identity is the one thing that cannot be traded away. Each scenario stages
    that as `<parent>_split`, and copied the parent's colour but not the rest
    of its appearance.

    So from L1 up the two halves rendered with different SURFACE FINISH: on
    `drop` the parent drew cork at roughness 0.9 and its understudy kept the
    0.55 default, which is one object splitting into two that are not quite the
    same object. It also split the pair in `_swap_in_gso`, which groups bodies
    by what a viewer can tell apart -- `drop` came out as a shark splitting
    into an unrelated scan.

    Appearance only. Mass stays the understudy's own, because `fission.stage`
    deliberately gives BOTH halves half the parent's mass and reading it from
    here would just be a second answer to the same question.
    """
    by_name = {b.name: b for b in spec.bodies}
    for body in spec.bodies:
        if not body.dormant or not body.name.endswith("_split"):
            continue
        parent = by_name.get(body.name[:-len("_split")])
        if parent is None:
            continue
        body.color = parent.color
        body.roughness = parent.roughness
        body.metallic = parent.metallic
        body.specular = parent.specular
        body.transmission = parent.transmission
        body.ior = parent.ior
        body.material = parent.material


def _material_scenery(spec: SceneSpec, seed: int) -> None:
    """From L1 up, the STAGING is made of something too.

    Materials arrived on actors alone, and you reported the consequence: L1
    looked like L0. Half of what is on screen is a ramp, a barrier, a table, a
    pendulum post and a floor -- and every one of them stayed an untextured
    block of flat colour while one object in the middle got a material. The
    level changed a fraction of the frame.

    Two different treatments, because the two kinds of scenery answer to
    different constraints:

    * **Props and occluders take a whole material.** A wooden ramp, a stone
      barrier, a steel post. Drawn from `SCENERY_MATERIALS`, which is narrower
      than the actor set on purpose -- a glass ramp is a puzzle rather than a
      surface, and a mirror-finish table throws the actor's reflection across
      the scene. Neither is the difficulty being measured.
    * **The floor keeps its COLOUR and takes only the surface.** Its colour is
      chosen by `_recolour_scenery` with a contrast guard against every actor,
      measured and re-measured -- a material's own colour band would throw that
      away. What it gains is roughness, specular and the rest, so the ground
      reads as stone or wood rather than as matte nothing.

    Mass is untouched throughout: scenery is static or scenario-tuned, and
    `density x volume` on a six-metre floor slab is a number nobody wants.

    Its own salted stream, so dressing the staging cannot shift a physics draw
    a scenario already made.
    """
    import zlib

    from . import materials as M

    if not COMPLEXITY[spec.complexity].materials:
        return
    rng = np.random.RandomState(
        (int(seed) * 2654435761 + 0x57A6E + zlib.crc32(spec.scenario.encode()))
        % (2 ** 31 - 1))
    for body in spec.bodies:
        if body.role in ("actor", "distractor", "backdrop") or body.material:
            continue
        # The shadow stand-in is a picture of an absence -- see
        # `shadow_track`. Giving it a surface would make it an object.
        if body.role == "shadow":
            continue
        name = M.pick(rng, M.SCENERY_MATERIALS)
        m = M.get(name)
        rgb, rough, metal, spec_, trans, ior = M.appearance(name, rng)
        keep = body.role == "floor"
        body.material = name
        body.roughness = rough
        body.metallic = metal
        body.specular = spec_
        body.transmission = trans
        body.ior = ior
        if not keep:
            body.color = rgb


def _swap_in_gso(spec: SceneSpec, seed: int) -> None:
    """From L3 up, actors become scanned objects rather than primitives.

    The last level, and the one that changes what an object IS rather than how
    it is lit. A GSO asset is real photogrammetry: irregular, textured, and
    shaped like nothing the physics has a primitive for.

    THE DRAWN SIZE IS PRESERVED. Following MOVi
    (`movi_c_worker.py:167-170`), the asset is normalised by its own longest
    axis -- `scale = target / max(bounds[1] - bounds[0])` -- so a body that was
    a 0.4 m sphere becomes a 0.4 m teapot rather than whatever size the scan
    happened to be. Without that the level would change the scene's SCALE as
    well as its geometry, and two axes would move at once.

    Scenery is left alone: a floor, a ramp or a barrier is staging, and the
    violation is defined against it. Swapping those for scanned meshes would
    change what the physics means, not how hard it is to look at.

    Its own salted stream, so turning the level up cannot shift a physics draw
    a scenario already made.
    """
    import zlib

    from ._gso import GSO_ASSETS, GSO_IDS

    if COMPLEXITY[spec.complexity].actor_assets != "gso" or not GSO_IDS:
        return
    rng = np.random.RandomState(
        (int(seed) * 2654435761 + 0x6507 + zlib.crc32(spec.scenario.encode()))
        % (2 ** 31 - 1))
    # ONE ASSET PER LOOK. Bodies that were indistinguishable as primitives must
    # stay indistinguishable as scans, and two scenarios depend on it outright:
    # `fission`'s dormant understudy has to be the same object its parent
    # splits into, or the clip shows one thing becoming a different thing; and
    # `collision`'s two balls must match in every respect or `newton2_mass`
    # loses its meaning. Drawing per body gave `drop` a shark that split into
    # an unrelated scan.
    #
    # The signature is what a viewer could tell apart before the swap, so the
    # grouping is exactly as fine as the distinction it has to preserve.
    picked = {}
    for body in spec.bodies:
        if body.role not in ("actor", "distractor") or body.static:
            continue
        # Keyed on what is VISIBLE, not on the material NAME. `fission`'s
        # understudy is drawn with its parent's colour but carries no material
        # of its own -- it is a stand-in the simulator holds parked -- so
        # keying on the name split the very pair this exists to hold together,
        # and `drop` got a shark that split into an unrelated scan. Colour,
        # roughness and metallic are what a material actually shows.
        look = (body.kind, tuple(np.round(body.scale, 9)),
                tuple(np.round(body.color, 6)),
                round(float(body.roughness), 6), round(float(body.metallic), 6),
                body.role)
        if look not in picked:
            picked[look] = str(GSO_IDS[int(rng.randint(0, len(GSO_IDS)))])
        aid = picked[look]
        lo, hi = np.asarray(GSO_ASSETS[aid]["bounds"], np.float64)
        longest = float(np.max(hi - lo))
        if longest <= 1e-6:
            continue
        # The size it was DRAWN at, not its `scale` -- for a primitive those
        # agree, and for what it is about to become they do not.
        target = 2.0 * float(max(body.extents))
        f = target / longest
        body.kind = "gso"
        body.asset_id = aid
        body.scale = (f, f, f)
        body.render_scale = None
    spec.notes["gso_assets"] = sorted(
        {b.asset_id for b in spec.bodies if b.kind == "gso" and b.asset_id})


def _add_backdrop(spec: SceneSpec) -> None:
    """The HDRI dome, at the levels that have an HDRI, and nowhere else.

    Added here rather than by each scenario for the same reason the
    environment itself is: it is a property of the LEVEL, and a property of the
    level that lives in thirteen files is one that will eventually differ in one
    of them.

    Added LAST, so its segmentation id cannot renumber anything -- which is the
    objection that kept the dome out of the higher levels originally, and it is
    answered by ordering rather than by making the dome the ground everywhere.
    """
    from . import _common as C

    if any(b.role == "backdrop" for b in spec.bodies):
        return
    body = C.backdrop(COMPLEXITY[spec.complexity])
    if body is not None:
        spec.bodies.append(body)


def _pick_hdri(spec: SceneSpec, seed: int) -> None:
    """Choose the environment map, from L2 up.

    Here rather than in each scenario for the reason every other level-wide
    property is here: it was copied into all THIRTEEN of them, identically, and
    a property of the level that lives in thirteen files is a property that will
    eventually differ in one of them.

    Its own salted stream, so turning the level up cannot shift a physics draw
    -- the same rule `appearance_rng` exists for, and the exact bug it was
    written after: `pick_hdri(rng)` once drew from the physics stream, and
    because it only fires at the realistic level the extra draw shifted every
    physics value after it.
    """
    from ._common import appearance_rng
    from ._hdri import pick as pick_hdri

    if COMPLEXITY[spec.complexity].background != "hdri":
        spec.hdri_id = None
        return
    # SALTED BY SCENARIO, like every other appearance draw. Salted by the bare
    # word "hdri" -- which is what all thirteen copies did -- every scenario
    # picks the same environment on a given seed, and a release ships one
    # backdrop per seed repeated thirteen times. Measured: seed 777 gave
    # `killesberg_park` to all thirteen.
    spec.hdri_id = pick_hdri(appearance_rng(seed, spec.scenario + "|hdri"))


def _add_peers(spec: SceneSpec, seed: int) -> None:
    """The `multi` condition: several actors, only some of them violating.

    The hard version of the task. With one actor, "which object is wrong" has a
    trivial answer -- there is only one object it could be -- so a model can
    score by detecting that SOMETHING is off and pointing at the only candidate.
    With N bodies behaving lawfully except for M of them, the clip asks *which*,
    and a spatial annotation finally has to be earned rather than inferred.

    BOTH COUNTS ARE DRAWN PER CLIP -- N over `MULTI_ACTORS`, M from two up to
    N-1. Fixing either would make it a cue: a model that learns "five objects,
    two wrong" is not reading physics, and the condition is about there being
    SEVERAL, not about there being five.

    At least two culprits, because one is what `standard` already is. At most
    N-1, so there is always a lawful body to contrast against -- which is the
    whole task. Note what the upper bound allows at N = 3: M = 2 is the only
    legal draw, a violating majority. Deliberate: it makes the small scenes the
    hardest, because "most things are wrong" is a different perceptual claim
    from "one thing is wrong".

    Reuses the distractor placer with `role="actor"` -- see `_common.distractors`.
    The extras are then ordinary actors: `_geom.actors` returns them, every
    injector's `_group` can pick them, and `group_fraction` decides how many
    are culprits. `_group` clamps its pick to at least two and needs at least
    four live actors to engage at all, which `MULTI_ACTORS` satisfies.

    Its own salted stream, so a `multi` clip's scenario draws are the same ones
    a `standard` clip of that scenario would have made.
    """
    import zlib

    from . import _common as C

    if not has_multi(spec.condition):
        spec.notes["n_peers_placed"] = 0
        return
    live = [b for b in spec.bodies if b.role == "actor" and not b.dormant]
    # A SCENARIO THAT IS ALREADY A CROWD NEEDS NOTHING. `pour` stages forty
    # grains and declares `group_fraction: 1.0` on purpose -- one grain of
    # forty hovering is perfectly annotated and impossible to see, so its
    # families act on the whole medium. Adding more bodies to that, and then
    # overwriting its fraction, would replace a deliberate choice with this
    # condition's default and make the violation invisible.
    spec.notes["n_peers_placed"] = 0
    if spec.notes.get("group_fraction"):
        spec.notes["n_actors"] = len(live)
        return
    rng = np.random.RandomState(
        (int(seed) * 2654435761 + 0xBEEF + zlib.crc32(spec.scenario.encode()))
        % (2 ** 31 - 1))
    # HOW MANY, drawn per clip. A fixed count is a cue: five objects at every
    # seed teaches the count rather than the physics, and the condition is
    # about there being SEVERAL, not about there being five.
    n_want = int(rng.randint(MULTI_ACTORS[0], MULTI_ACTORS[1] + 1))
    want = max(0, n_want - len(live))
    floor = next((b for b in spec.bodies if b.role == "floor"), None)
    top = 0.0 if floor is None else (
        float(floor.position[2] + floor.scale[2]) if floor.kind == "cube"
        else float(spec.floor_level))
    placed = C.distractors(spec, want, rng, floor_top=top, role="actor")
    spec.bodies.extend(placed)
    spec.notes["n_peers_placed"] = len(placed)
    # WHAT ACTUALLY LANDED decides the count, not what was asked for. The
    # placement constraints can leave no room, and a scene that ended up with
    # three actors must not claim eight.
    total = len(live) + len(placed)
    spec.notes["n_actors"] = total
    if total < 3:
        return                       # too few to pose the question at all
    # HOW MANY VIOLATE, also drawn per clip: at least two, and never all of
    # them, so there is always a lawful body to contrast against.
    lo = min(MULTI_CULPRIT_RANGE[0], total - MULTI_CULPRIT_RANGE[1])
    hi = total - MULTI_CULPRIT_RANGE[1]
    m = int(rng.randint(lo, hi + 1)) if hi > lo else hi
    spec.notes["n_culprits_wanted"] = m
    # `_group` takes a FRACTION and rounds, so the fraction is chosen to round
    # back to exactly `m` -- storing the count directly would mean teaching
    # every injector a second way to ask the same question.
    spec.notes["group_fraction"] = float(m) / float(total)


def _add_distractors(spec: SceneSpec, seed: int) -> None:
    """Populate the scene with bodies that take no part in the violation.

    On the clips whose CONDITION calls for them -- not a level, so "what does
    clutter cost" is answerable at each realism level rather than only at the
    top of the ladder. See `CONDITION_CYCLE`.

    Added here rather than in each scenario because the placement rule is a
    property of the LEVEL, and because it needs the finished scene -- where the
    actor is, where it is heading, and what the camera can see -- all of which
    exist only once `_sample` has run.

    Drawn off its own salted stream so that adding distractors cannot shift a
    single physics or appearance draw that a scenario already made: a cluttered
    scene is a clean scene with more bodies in it, and nothing else different.
    """
    import zlib

    from . import _common as C

    cx = COMPLEXITY[spec.complexity]
    if not has_distractors(spec.condition):
        # Recorded as zero rather than left absent: a clip with no distractors
        # is a fact about that clip, and a reader filtering on the axis needs
        # both sides of it.
        spec.notes["n_distractors_placed"] = 0
        return
    rng = np.random.RandomState(
        (int(seed) * 2654435761 + 0xD157 + zlib.crc32(spec.scenario.encode()))
        % (2 ** 31 - 1))
    # HOW MANY, drawn per clip, from the same range `multi` uses. It was a
    # fixed six (twelve at the GSO level), which is a count a model can learn
    # instead of the physics -- the same reason `multi` draws its own.
    n = int(rng.randint(EXTRA_OBJECTS[0], EXTRA_OBJECTS[1] + 1))
    floor = next((b for b in spec.bodies if b.role == "floor"), None)
    top = 0.0 if floor is None else (
        float(floor.position[2] + floor.scale[2]) if floor.kind == "cube"
        else float(spec.floor_level))
    placed = C.distractors(spec, n, rng, floor_top=top)
    spec.bodies.extend(placed)
    # What was actually placed, not what was asked for. The constraints can
    # genuinely leave no room, and a clip that says `n_distractors: 8` while
    # containing four is a clip whose metadata lies.
    spec.notes["n_distractors_placed"] = len(placed)


#: What a body looks like before any material touches it, read off `BodySpec`
#: rather than written down again -- the first copy of these numbers fell two
#: traits behind the dataclass and took L0's whole appearance guarantee with it.
_BODY_DEFAULTS = {
    f.name: f.default
    for f in dataclasses.fields(BodySpec)
    if f.name in ("roughness", "metallic", "specular", "transmission", "ior")}


def _flatten_materials(spec: SceneSpec) -> None:
    """Below L2, bodies have a flat colour and ONE shared density.

    Applied here rather than in each scenario for the same reason the scenery
    recolouring is: it is a property of the level, and thirteen files should
    not each have to know about it.

    THE SHARED DENSITY IS THE POINT, not a simplification. Mass is derived from
    the material, so stripping materials without also fixing the density would
    leave mass varying invisibly -- and an invisible mass difference is exactly
    the confound materials were introduced to remove: the heavy ball barely
    moves, nothing in the picture says why, and lawful physics reads as a
    violation. With one density, mass still varies -- with SIZE, which a viewer
    can see.

    Hue is preserved so the objects stay tellable apart; only saturation and
    value are pinned, which is what makes them read as plain colours rather
    than as steel or rubber.

    ALL FIVE OPTICAL TRAITS ARE RESET, not just the two this started with.
    `roughness` and `metallic` were the whole surface when materials were
    introduced; `specular`, `transmission` and `ior` arrived with glass and ice
    and this function was not told about them. So L0 -- the level whose entire
    definition is "flat colour, one density" -- shipped a random subset of
    clips whose actors were TRANSPARENT: `drop` drew glass on 5 seeds in 6 and
    rendered a see-through ball, `pour` on 2 in 6 and rendered 144 see-through
    grains, `ramp_slide` ice on 4 in 6. That is the appearance confound the
    level exists to remove, arriving at the level meant to be free of it, and
    it also charged L0 for the most expensive shading in the table.

    The defaults are `BodySpec`'s own, so this cannot drift from them again
    without the dataclass changing underneath it.
    """
    import colorsys

    from . import materials as M

    if COMPLEXITY[spec.complexity].materials:
        return
    for b in spec.bodies:
        if b.material is None:
            continue
        h, _, _ = colorsys.rgb_to_hsv(*b.color)
        b.color = tuple(float(c) for c in colorsys.hsv_to_rgb(h, 0.62, 0.88))
        for trait in ("roughness", "metallic", "specular", "transmission",
                      "ior"):
            setattr(b, trait, _BODY_DEFAULTS[trait])
        b.mass = M.mass_for(M.REFERENCE_MATERIAL, b.scale, b.kind)
        b.material = None


#: THE CONDITION EACH VARIANT CARRIES, as a cycle over variant index.
#:
#: Camera motion, distractors and multiple culprits are three ways to make a
#: clip harder, and this table says which combinations the dataset actually
#: contains and in what proportion. Read down the cycle: six plain clips, then
#: one of each condition, then the one combination worth having.
#:
#:   standard       static camera, one culprit, nothing else in shot
#:   camera         the camera moves
#:   distractors    extra bodies that take NO part in the physics
#:   multi          extra bodies that DO -- N actors, M of them violating
#:   camera+multi   both
#:
#: **Named cells, not independent coin flips.** Independent ratios were the
#: first design and they blur the thing a benchmark reports: a "moving camera"
#: clip might also carry clutter, so the marginal comparison mixes two effects
#: and the per-condition counts are only exact in expectation. One condition
#: per clip makes every count exact and every comparison against `standard`
#: clean.
#:
#: **Distractors and multi are alternatives, not a pair.** They are the same
#: placement machinery -- extra bodies, cleared of the action and inside the
#: frame -- differing only in whether the extras take part. A scene with both
#: asks the viewer to sort inert clutter from lawful peers from culprits, which
#: is three distinctions where the family only makes one.
#:
#: **The plain clips come FIRST**, so a short run is the easy case: eight
#: variants get camera and distractors but never `multi`, and six get nothing
#: at all. A run spends clips on difficulty only once it is long enough to
#: afford them.
#:
#: Six plain clips, then one of each condition. Marginals over the cycle:
#: camera 20%, distractors 10%, multi 20%.
#:
#: This tuple IS the policy -- change it and everything follows, including the
#: metadata, the card, the README's table and the cost estimate. Nothing reads
#: the shares from anywhere else.
CONDITION_CYCLE = ("standard", "standard", "standard", "standard", "standard",
                   "standard", "camera", "distractors", "multi",
                   "camera+multi")

#: How many EXTRA OBJECTS a scene holds, drawn per clip, for `distractors` and
#: `multi` alike. Randomised rather than fixed, so neither condition varies in
#: the thing it is about while holding a count a model could learn instead: a
#: six-object scene at every seed teaches the count.
#:
#: **The two conditions differ in how many objects get INVALID PHYSICS, not in
#: how many objects there are.** Under `distractors` exactly one body violates
#: -- the scenario's own actor -- and the extras are scenery no family can
#: target. Under `multi` the extras are eligible culprits and 2..N-1 of them
#: violate. That is the whole distinction, and it is the one worth drawing:
#: "which of these is wrong" is a different question from "is anything wrong",
#: and only `multi` asks it.
EXTRA_OBJECTS = (3, 10)

#: Kept as the name the multi path reads, because the count means the same
#: thing there: how many actors are in shot.
MULTI_ACTORS = EXTRA_OBJECTS

#: How many of them violate: at least two -- one culprit is what `standard`
#: already is -- and at most all but one, so there is always a lawful object to
#: contrast against.
#:
#: Note what the upper bound allows: at N = 3 the only legal draw is M = 2, a
#: violating MAJORITY. That is deliberate and it is your call; it makes the
#: small scenes the hardest ones, because "most things are wrong" is a
#: different perceptual claim from "one thing is wrong".
MULTI_CULPRIT_RANGE = (2, 1)   # (minimum, how many lawful bodies to keep)


def condition_for(variant: int, n_variants: Optional[int] = None,
                  level: Optional[str] = None) -> str:
    """Which condition this clip carries. See `CONDITION_CYCLE`.

    **SPREAD ACROSS WHATEVER A LEVEL WAS GIVEN, not indexed from zero.** A level
    high on the ladder gets few variants -- at ten, L3 gets two -- and the six
    `standard` slots come first, so indexing the cycle directly gave every
    upper level nothing but `standard`. Measured on a real ladder run: L1, L2
    and L3 were 100% standard, so the dataset would have shipped its three
    hardest levels with no camera motion, no clutter and no multi-culprit clips
    at all, and the conditions would have existed only at L0.

    So a level's `n` variants are spread over the cycle's `P` slots --
    `slot = v*P // n` -- which keeps each level's mix close to the declared
    shares whatever it was allocated. A per-level PHASE then rotates the result,
    so the levels do not all sample the same few slots and the ladder as a whole
    covers every condition:

        L0 (10 var)  standard x6, camera, distractors, multi, camera+multi
        L1 ( 5 var)  standard x3, distractors, camera+multi
        L2 ( 3 var)  standard x2, multi
        L3 ( 2 var)  standard, multi

    `n_variants` unknown or at least `P` means index directly, which is the
    L0 case and every single-level run -- so `--complexity L3 --variants 10`
    still walks the cycle in order, and a short run still starts on `standard`.
    """
    P = len(CONDITION_CYCLE)
    v = int(variant)
    n = int(n_variants or 0)
    if n <= 0 or n >= P:
        return CONDITION_CYCLE[v % P]
    phase = LEVEL_PHASE.get(str(level), 0)
    return CONDITION_CYCLE[((v % n) * P // n + phase) % P]


#: How far each level rotates the cycle. Its position in the ladder, so the
#: levels sample different slots and no condition is confined to one level.
LEVEL_PHASE = {name: i for i, name in enumerate(COMPLEXITY)}


def condition_share(name: str) -> float:
    """What fraction of clips carry `name`, straight off the cycle."""
    return CONDITION_CYCLE.count(name) / float(len(CONDITION_CYCLE))


def has_moving_camera(condition: str) -> bool:
    return "camera" in str(condition)


def has_distractors(condition: str) -> bool:
    return str(condition) == "distractors"


def has_multi(condition: str) -> bool:
    return "multi" in str(condition)


def variants_at(level: str, variants: int) -> int:
    """How many of a run's `variants` per cell this level gets.

    Rounds, and rounds DOWN TO ZERO: a level whose share does not buy a whole
    variant is skipped rather than promoted to one, so the declared ratios hold
    instead of the tail levels being over-represented in every small run.
    Naming a level explicitly is the override.
    """
    cx = COMPLEXITY.get(level)
    return int(variants) if cx is None else int(round(int(variants) * cx.share))


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
    """The `camera` and `camera+multi` clips get a moving camera, of one of
    three kinds.

    Deliberately NOT a level on the ladder. Viewpoint is not realism, and a level
    that fired on only some of its clips would stop the level above it from
    isolating its own axis. See `COMPLEXITY` and `CONDITION_CYCLE`.

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
    if not COMPLEXITY[spec.complexity].camera_moves:
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
    if not forced and not has_moving_camera(spec.condition):
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

    Skipped from L2 up, where an HDRI supplies the ground and the sky together
    and a chosen colour would be painted over.

    TWO SURFACES, TWO COLOURS. Below L2 the scene is a 6 m slab in front of a
    void, so the horizon is a real edge and something has to make it read. Both
    are drawn here and both are held clear of every actor; the void is drawn
    from its own, darker band so the ground does not dissolve into the sky.

    This was one colour for a while, and correctly so: the ground was a dome
    that curved up behind the scene and WAS the backdrop, so a second colour
    would have had nothing to paint. When the dome went back to being a
    render-only backdrop at the HDRI levels, that left `BACKDROP_VALUE` unused
    and every L0 clip with a floor and a void the same shade -- 0.795, 0.812,
    0.82 for both, on `drop` at seed 91731, which reads as fog.
    """
    import colorsys

    from ..residuals.laws import _srgb_to_lab

    if COMPLEXITY[spec.complexity].background == "hdri":
        return
    floors = [b for b in spec.bodies if b.role == "floor"]
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

    # THE VOID BEHIND THE SLAB, scanned the same way and against one more
    # constraint: it has to stand clear of the actors AND of the floor it meets
    # at the horizon, or there is no horizon. Its band is the darker one
    # (`BACKDROP_VALUE`), so the usual outcome is a light floor against a dim
    # surround; under a floor that is itself dark the scan takes the light end
    # of that band instead, which separates just as well and is the reason it
    # is a scan and not a fixed offset. Over the thirteen scenarios at three
    # seeds apiece the closest pair comes out 23 Lab apart, on `resting_table`
    # at L0, seed 7.
    lab_floor = _srgb_to_lab(np.asarray(best, np.float64))
    hue_b = float(rng.uniform(0.0, 1.0))
    void, void_gap = spec.background_color, -1.0
    for k in range(17):
        v = BACKDROP_VALUE[0] + (
            BACKDROP_VALUE[1] - BACKDROP_VALUE[0]) * k / 16.0
        for j in range(3):
            sat = BACKDROP_SATURATION[0] + (
                BACKDROP_SATURATION[1] - BACKDROP_SATURATION[0]) * j / 2.0
            cand = tuple(float(c) for c in colorsys.hsv_to_rgb(hue_b, sat, v))
            lab = _srgb_to_lab(np.asarray(cand, np.float64))
            gap = min(separation(cand),
                      float(np.linalg.norm(lab - lab_floor)))
            if gap > void_gap:
                void, void_gap = cand, gap
    spec.background_color = void


class Scenario:
    """Base class. Subclasses implement `_sample`."""

    name: str = "unnamed"

    def sample(self, seed: int, tier: Tier,
               complexity: str = DEFAULT_COMPLEXITY,
               variant: int = 0, n_variants: Optional[int] = None) -> SceneSpec:
        """Sample one instance, then vary how it looks. Do not override.

        Level 4 of the taxonomy is the *instance*, and two instances of one
        scenario should differ in more than their random seed's effect on a
        radius. Subclasses stage the physics in `_sample`; the appearance
        variation that is common to all of them -- where the camera stands, how
        the scene is lit -- is applied here so it cannot drift between thirteen
        files.
        """
        # THE UNBUILT-LEVEL GUARD LIVES HERE, not in a scenario. It was in
        # `drop` alone, referencing a `Complexity` field that no longer exists,
        # so twelve scenarios would have quietly sampled a level that cannot
        # render and one would have raised an AttributeError explaining
        # nothing.
        cx = COMPLEXITY.get(complexity)
        if cx is None:
            raise KeyError("unknown complexity %r; known: %s"
                           % (complexity, sorted(COMPLEXITY)))
        if not cx.implemented:
            raise NotImplementedError(
                "complexity %s is scaffolded but not built yet; built: %s"
                % (complexity, implemented_complexities()))
        spec = self._sample(seed, tier, complexity)
        spec.variant = int(variant)
        spec.n_variants = None if n_variants is None else int(n_variants)
        # RESOLVED ONCE, here, and carried on the spec. Three gates and the
        # metadata all need it, and deriving it separately in each was how the
        # scene and its label could have disagreed.
        spec.condition = condition_for(spec.variant, spec.n_variants, complexity)
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

    def rescript(self, spec: "SceneSpec", traj, plan) -> None:
        """Re-derive the scenario's own driven bodies on an INVALID trajectory.

        `script` runs once, on the valid rollout, before any injector exists.
        That is right for a body whose motion is a property of the scene -- and
        wrong the moment an injector changes the body that motion is derived
        FROM. `shadow_track` is the case: its cast shadow is a projection of the
        actor, so a clip that teleports the actor, grows it, pushes it or
        removes it must move, grow, push or remove the shadow with it.

        It did not. Measured on the review sweep: under `continuity` the ball
        jumped 1.4 m and its shadow carried on down its lawful track; under
        `phantom_impulse` the ball accelerated away and the shadow did not
        follow; under `immutability` the ball grew to 2.3x and its shadow
        stayed the original size; under `permanence` the ball vanished and its
        shadow stayed. Every one of those clips shipped a detached shadow --
        which is the `shadow` family -- inside a clip claiming something else,
        and annotated only the something else.

        Called on every invalid trajectory, after the intervention and before
        the prefix check. `plan` is passed so a scenario can leave alone
        whatever the intervention is actually about: when the culprit IS the
        shadow, the injector owns it and this must not overwrite its work.

        Prefix identity is untouched: the driven body is a function of a
        trajectory whose prefix is already identical, so its prefix is too.
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
