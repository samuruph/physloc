"""`shadow_track` -- an object translates under a fixed light.

The visible shadow is a real Cycles cast shadow.  A duplicate of the actor is
hidden from camera rays and visible only to shadow rays; the visible actor has
its shadow ray disabled.  Optical interventions move or deform that internal
caster, so Cycles still performs the light transport while the RGB actor stays
unchanged.  The caster is renderer-only and is removed from public object,
segmentation, physics and energy annotations by the v3 writer.
"""
from __future__ import annotations

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

        r = float(rng.uniform(0.30, 0.40)) * C.size_scale(seed, self.name)
        height = float(rng.uniform(1.0, 1.35))
        flight = tier.num_frames / float(tier.fps)

        lamp = (-2.6, -1.4, 4.2)
        light_dir = _unit(np.array([0.0, 0.0, 0.0]) - np.array(lamp))

        # Derived from the frame, and the shadow is what has to fit -- it is the
        # violator of every family staged here, and it does not sit under the
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

        actor = C.with_material(actor, M.pick(arng), arng)
        shade = BodySpec(
            name="shadow_caster", kind="sphere", position=actor.position,
            scale=(r, r, r),
            mass=0.0, static=False, scripted=True, collides=False,
            visible_camera=False, visible_shadow=True,
            color=actor.color, segmentation_id=self.SEG_SHADOW,
            role="shadow_caster")

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
        """Translate the actor and its renderer-only Cycles shadow caster."""
        n = traj.num_frames
        t = np.arange(n, dtype=np.float64) * traj.dt
        ja = _index(spec, self.SEG_ACTOR)
        actor = spec.bodies[ja]

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

        The three optical families are exactly the exception: their violator IS
        the shadow, and re-deriving it from a caster that never moved would
        simply undo them. So this stands aside whenever the shadow is named in
        the plan.
        """
        shade = next((b for b in spec.bodies if b.role == "shadow_caster"), None)
        if shade is None:
            return
        if int(shade.segmentation_id) in {int(i) for i in plan.causal_body_ids}:
            return
        self._cast(spec, traj)

    def _cast(self, spec, traj) -> None:
        """Keep the hidden caster identical to the visible actor.

        Four channels, because a shadow follows its object in all of them:

        Cycles, not this trajectory, projects it onto receiving geometry.
        * **presence** -- what is not there casts nothing. This is what makes
          `permanence` and `dissolve` read correctly on this scenario instead
          of leaving an orphaned shadow behind.
        * **opacity** -- a half-transparent body casts a half-strength shadow,
          so a dissolve fades both together.
        """
        ja, js = _index(spec, self.SEG_ACTOR), _index(spec, self.SEG_SHADOW)
        p = np.asarray(traj.pos[:, ja, :], np.float64)
        traj.pos[:, js, :] = p.astype(np.float32)
        traj.quat[:, js, :] = np.asarray(traj.quat[:, ja, :], np.float32)
        if traj.num_frames > 1:
            traj.lin_vel[1:, js, :] = ((traj.pos[1:, js, :]
                                        - traj.pos[:-1, js, :]) / traj.dt)
            traj.lin_vel[0, js, :] = traj.lin_vel[1, js, :]
        traj.scale_mul[:, js, :] = np.asarray(traj.scale_mul[:, ja, :], np.float32)
        traj.present[:, js] = np.asarray(traj.present[:, ja], bool)
        traj.opacity[:, js] = np.asarray(traj.opacity[:, ja], np.float32)


def _unit(v: np.ndarray) -> np.ndarray:
    return v / max(float(np.linalg.norm(v)), 1e-9)


register(ShadowTrack())


def _index(spec, segmentation_id: int) -> int:
    """A body's position in `spec.bodies`, by id: L3 renames a body after the
    scan it became, so its name is not a stable handle."""
    for i, b in enumerate(spec.bodies):
        if int(b.segmentation_id) == int(segmentation_id):
            return i
    raise KeyError(segmentation_id)
