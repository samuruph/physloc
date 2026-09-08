"""`shadow_track` -- an object translates under a fixed light, casting a shadow.

**The shadow is a body, not a render effect.** Blender's own cast shadow has no
segmentation id, so it cannot carry a mask, and a violation nobody can localise
is not what this dataset ships. So the actor's Cycles shadow ray visibility is
switched off and a flat, dark `shadow` body is scripted onto the ground at the
position the light geometry puts it. It is a stand-in -- no penumbra, no shape
distortion over uneven ground -- and it is labelled `role="shadow"` so a
consumer is never misled about what it is looking at.

What that buys: the shadow now has pixels, an id, a footprint in both twins and
therefore an exact `violation_mask` under the same union rule as every other
family. Grounded in LikePhys *Moving Shadow*.
"""
from __future__ import annotations

import math

import numpy as np

from .. import camera as cam
from . import _common as C
from . import materials as M
from .base import (COMPLEXITY, DEFAULT_COMPLEXITY, BodySpec, LightSpec,
                   SceneSpec, Scenario, Tier, register)


CAMERA = (0.0, -6.2, 3.1)
LOOK_AT = (0.0, 0.0, 0.5)


class ShadowTrack(Scenario):
    name = "shadow_track"
    SEG_FLOOR, SEG_ACTOR, SEG_SHADOW = 1, 2, 4

    def _sample(self, seed: int, tier: Tier,
                complexity: str = DEFAULT_COMPLEXITY) -> SceneSpec:
        rng = self.rng(seed)
        cx = COMPLEXITY[complexity]
        arng = C.appearance_rng(seed, self.name)
        if not cx.implemented:
            raise NotImplementedError("complexity %s not built" % complexity)

        r = float(rng.uniform(0.30, 0.40))
        height = float(rng.uniform(1.0, 1.35))
        flight = tier.num_frames / float(tier.fps)

        lamp = (-2.6, -1.4, 4.2)
        light_dir = _unit(np.array([0.0, 0.0, 0.0]) - np.array(lamp))

        # Derived from the frame, and the shadow is what has to fit -- it is the
        # culprit of every family staged here, and it does not sit under the
        # actor. A low key light throws it `height/|Lz|` metres off to one side,
        # so the pair together span the actor's travel PLUS that offset, and
        # sizing the travel alone put the shadow past the edge on wide seeds.
        throw = height / max(abs(float(light_dir[2])), 1e-6)
        lead = throw * float(np.linalg.norm(light_dir[:2]))
        budget = 2.0 * cam.frame_extent(CAMERA, LOOK_AT) * float(
            rng.uniform(0.60, 0.70))
        span = max(1.0, budget - lead - 2.0 * r)
        speed = span / flight

        actor = BodySpec(
            name="body", kind="sphere", position=(-span / 2.0, 0.0, height),
            scale=(r,) * 3, velocity=(speed, 0.0, 0.0), mass=1.0,
            static=False, scripted=True, visible_shadow=False,
            color=C.hue_rgb(float(rng.uniform(0, 1))),
            segmentation_id=self.SEG_ACTOR, role="actor")

        psi = math.atan2(float(light_dir[1]), float(light_dir[0]))
        actor = C.with_material(actor, M.pick(arng), arng)
        shade = BodySpec(
            # A flattened SPHERE, not a flattened cube. A round caster does not
            # throw a square shadow, and shipping one meant every `shadow` clip
            # carried a second, unlabelled inconsistency for a viewer to notice
            # first. Whether the shadow's shape matches its caster is a
            # violation in its own right -- see the `shadow_shape` family --
            # which only works if the lawful clip gets it right.
            name="shadow", kind="sphere", position=(0.0, 0.0, 0.006),
            # Declared round for the simulator, drawn as a flattened ellipse.
            # Kubric's PyBullet wrapper asserts uniform scaling on spheres, so
            # the squash has to happen on the render side -- see
            # `BodySpec.render_scale`. Stretched along the light's ground-plane
            # bearing, which is the direction a low sun smears a round object's
            # shadow.
            scale=(r, r, r),
            render_scale=(r / max(abs(float(light_dir[2])), 0.35), r, 0.006),
            quaternion=(math.cos(psi / 2.0), 0.0, 0.0, math.sin(psi / 2.0)),
            mass=1.0, static=False, scripted=True, visible_shadow=False,
            color=(0.035, 0.035, 0.045), segmentation_id=self.SEG_SHADOW,
            role="shadow")

        return SceneSpec(
            scenario=self.name, seed=seed, tier=tier,
            bodies=[C.ground(cx, self.SEG_FLOOR), shade, actor],
            # A hard directional key regardless of complexity: the whole point
            # is a shadow whose direction the viewer can reason about, and an
            # HDRI on its own gives a soft one with no obvious source.
            lights=[LightSpec("key", position=lamp, look_at=(0.0, 0.0, 0.0),
                              intensity=3.2)],
            camera_position=CAMERA, camera_look_at=LOOK_AT,
            floor_level=0.0, complexity=complexity,
            camera_jitter_deg=(15.0, 8.0),
            notes={"radius": r, "height": height, "speed": speed,
                   "light_dir": [float(x) for x in light_dir],
                   "caster_id": self.SEG_ACTOR, "shadow_id": self.SEG_SHADOW,
                   "surface_top": 0.0,
                   "family_targets": {"shadow_shape": [self.SEG_SHADOW]}})

    # ------------------------------------------------------------------ #
    def script(self, spec, traj) -> None:
        """Translate the actor, and put its shadow where the light says it goes."""
        n = traj.num_frames
        t = np.arange(n, dtype=np.float64) * traj.dt
        actor = spec.body("body")
        ja = spec.index_of("body")

        p = np.asarray(actor.position, np.float64)[None, :] + \
            np.asarray(actor.velocity, np.float64)[None, :] * t[:, None]
        traj.pos[:, ja, :] = p.astype(np.float32)
        traj.lin_vel[:, ja, :] = np.tile(np.asarray(actor.velocity, np.float32),
                                         (n, 1))
        self._cast(spec, traj)

    def rescript(self, spec, traj, plan) -> None:
        """Re-cast the shadow from whatever the actor ENDED UP doing.

        A shadow is a projection, so it is a function of the caster and of the
        light -- never a track of its own. Every family that acts on the actor
        therefore changes the shadow too, and until this existed none of them
        did: the ball teleported and its shadow stayed on the lawful line, the
        ball grew and its shadow did not, the ball was removed and its shadow
        went on sliding across an empty floor. Each of those clips carried a
        detached shadow -- which is the `shadow` family, a violation in its own
        right -- while claiming and annotating something else entirely.

        The three optical families are exactly the exception: their culprit IS
        the shadow, and re-deriving it from a caster that never moved would
        simply undo them. So this stands aside whenever the shadow is named in
        the plan.
        """
        shade = next((b for b in spec.bodies if b.role == "shadow"), None)
        if shade is None:
            return
        if int(shade.segmentation_id) in {int(i) for i in plan.causal_body_ids}:
            return
        self._cast(spec, traj)

    def _cast(self, spec, traj) -> None:
        """Put the shadow under the caster, at the caster's size and presence.

        Four channels, because a shadow follows its object in all of them:

        * **position** -- the light's projection of the caster onto the ground.
        * **footprint** -- a body that doubles in size doubles its shadow, so
          the caster's horizontal `scale_mul` carries over. Not the vertical
          one: the shadow is a flat patch on the floor and scaling its
          thickness would only lift it off the ground.
        * **presence** -- what is not there casts nothing. This is what makes
          `permanence` and `dissolve` read correctly on this scenario instead
          of leaving an orphaned shadow behind.
        * **opacity** -- a half-transparent body casts a half-strength shadow,
          so a dissolve fades both together.
        """
        ja, js = spec.index_of("body"), spec.index_of("shadow")
        p = np.asarray(traj.pos[:, ja, :], np.float64)
        traj.pos[:, js, :] = project(p, spec.notes["light_dir"],
                                     float(spec.notes["surface_top"]),
                                     0.006).astype(np.float32)
        if traj.num_frames > 1:
            traj.lin_vel[1:, js, :] = ((traj.pos[1:, js, :]
                                        - traj.pos[:-1, js, :]) / traj.dt)
            traj.lin_vel[0, js, :] = traj.lin_vel[1, js, :]
        lin = np.asarray(traj.scale_mul[:, ja, :], np.float64)
        traj.scale_mul[:, js, 0] = lin[:, 0]
        traj.scale_mul[:, js, 1] = lin[:, 1]
        traj.present[:, js] = np.asarray(traj.present[:, ja], bool)
        traj.opacity[:, js] = np.asarray(traj.opacity[:, ja], np.float32)


def project(p: np.ndarray, light_dir, surface_top: float,
            lift: float = 0.0) -> np.ndarray:
    """Where a body at `p` casts its shadow on the plane z == surface_top."""
    L = _unit(np.asarray(light_dir, np.float64))
    denom = -L[2] if abs(L[2]) > 1e-6 else -1e-6
    t = (p[:, 2] - surface_top) / denom
    out = np.zeros_like(p)
    out[:, :2] = p[:, :2] + t[:, None] * L[None, :2]
    out[:, 2] = surface_top + lift
    return out


def _unit(v: np.ndarray) -> np.ndarray:
    return v / max(float(np.linalg.norm(v)), 1e-9)


register(ShadowTrack())
