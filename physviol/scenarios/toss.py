"""`toss` -- a body thrown on a ballistic arc.

Pure free flight, no contact for the whole clip. This is the **clean control**
of docs/PLAN.md Part 2: nothing occludes the actor, so `t_event` and
`t_observable` coincide, and every occluded scenario's observability lag is
measured against it.

The camera is *derived*, not hand-tuned -- see `camera.frame_flight`. A body
airborne for the whole clip rises `g*T^2/8`, so the shot has to grow with the
clip length or the actor leaves frame; deriving it also makes the scenario
scale-invariant, so the same seed frames identically at every tier.
"""
from __future__ import annotations

from .. import camera as cam
from . import _common as C
from . import materials as M
from ._hdri import pick as pick_hdri
from .base import (COMPLEXITY, DEFAULT_COMPLEXITY, BodySpec, SceneSpec,
                   Scenario, Tier, register)


class Toss(Scenario):
    name = "toss"
    SEG_FLOOR, SEG_BALL, SEG_SPLIT = 1, 2, 4

    def _sample(self, seed: int, tier: Tier,
                complexity: str = DEFAULT_COMPLEXITY) -> SceneSpec:
        rng = self.rng(seed)
        cx = COMPLEXITY[complexity]
        # ONE appearance stream for the whole sample. `appearance_rng` builds a
        # fresh RandomState per call, so asking it once per draw handed every
        # draw the same first number -- material, colour and proportions came
        # out identical across every scenario on a given seed. Threading one
        # stream lets the draws advance.
        arng = C.appearance_rng(seed, self.name)
        if not cx.implemented:
            raise NotImplementedError("complexity %s not built" % complexity)

        # Airborne from the first frame to the last: the arc's duration is the
        # clip's, and everything else about the shot follows from it.
        flight = tier.num_frames / float(tier.fps)
        f = cam.frame_flight(flight, angular_radius=float(rng.uniform(0.100, 0.120)))
        camera_position, camera_look_at = cam.flight_camera(f)

        radius = f.radius
        vx = float(f.x_travel / flight * rng.choice([-1.0, 1.0]))
        # Launch and land symmetrically about the frame centre, so the actor is
        # never near an edge at either end of the clip.
        x0 = -vx * flight * 0.5

        hdri_id = pick_hdri(C.appearance_rng(seed, "hdri")) if cx.background == "hdri" else None
        # Any shape it likes: nothing here assumes a rolling contact, so a
        # cone that topples and a ring that rolls away are both fair
        # pictures of the same physics. Drawn off the appearance
        # stream -- see `C.pick_shape`.
        kind = C.pick_shape(arng)
        # THROWN WITH SPIN, which is what `tumble` used to be for. Retiring that
        # scenario left this one as the only free-flight throw, and a body
        # thrown without any rotation at all is the less interesting half of
        # what a throw looks like.
        #
        # A SPHERE gets none, and that is not an oversight: with the v0 asset
        # set the primitives are untextured, so a spinning ball is
        # pixel-identical to a still one. Giving it spin would put rotation in
        # the scene that no viewer and no residual can see.
        spin = ((float(rng.uniform(-1.5, 1.5)), float(rng.uniform(3.5, 5.5)),
                 float(rng.uniform(-1.5, 1.5))) if kind == "cube"
                else (0.0, 0.0, 0.0))
        ball = BodySpec(
            name="ball", kind=kind, position=(x0, 0.0, f.launch_z),
            scale=(radius,) * 3, velocity=(vx, 0.0, f.vz),
            angular_velocity=spin, mass=1.0,
            friction=0.4, restitution=0.6, color=C.hue_rgb(float(rng.uniform(0, 1))),
            segmentation_id=self.SEG_BALL, role="actor")
        ball = C.with_material(ball, M.pick(arng), arng)
        ball = C.vary_dims(ball, arng)

        return SceneSpec(
            scenario=self.name, seed=seed, tier=tier,
            bodies=[C.ground(cx, self.SEG_FLOOR,
                             size=max(6.0, 1.6 * f.half_extent)),
                    ball, C.understudy(ball, self.SEG_SPLIT)],
            lights=C.lights(cx, look_at=(0.0, 0.0, f.look_z * 0.4),
                            scale=f.scene_scale),
            camera_position=camera_position, camera_look_at=camera_look_at,
            floor_level=0.0, complexity=complexity, hdri_id=hdri_id,
            camera_jitter_deg=(15.0, 8.0),
            notes={"radius": radius, "v0": [vx, 0.0, f.vz],
                   "flight_seconds": flight, "actor_kind": kind,
                   "apex": f.apex, "scene_scale": f.scene_scale,
                   "frame_half_extent": f.half_extent})


register(Toss())
