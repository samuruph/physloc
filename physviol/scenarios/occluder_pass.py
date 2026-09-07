"""`occluder_pass` -- a body travels behind a screen and re-emerges.

The scenario that makes **observability lag** non-zero. Everything else in v0
has `t_event == t_observable`; here a violation injected while the actor is
hidden is not visible until it fails to re-emerge, which is the gap the whole
three-clocks design exists to measure (docs/PLAN.md 1.1).

The occluded frame interval is computed geometrically at sample time and stored
in `notes`, so an injector can choose to fire *while hidden* without needing a
rendered segmentation first.
"""
from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np

from .. import camera as cam
from . import _common as C
from . import materials as M
from ._hdri import pick as pick_hdri
from .base import (COMPLEXITY, DEFAULT_COMPLEXITY, BodySpec, SceneSpec,
                   Scenario, Tier, register)


CAMERA = (0.0, -8.5, 1.5)
LOOK_AT = (0.0, 0.4, 0.9)


class OccluderPass(Scenario):
    name = "occluder_pass"
    SEG_FLOOR, SEG_SCREEN, SEG_BALL, SEG_SPLIT = 1, 3, 2, 4

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

        radius = float(rng.uniform(0.26, 0.36))
        # Derived from the frame -- see `collision`. This also fixes the
        # occluded *fraction* of the clip, which used to shrink as the tiers got
        # longer: the screen is a fixed width, so a ball that travels further
        # spends proportionally less of the clip behind it, and the observability
        # lag this scenario exists to produce quietly shortened with tier.
        speed = float(cam.traverse_speed(
            CAMERA, LOOK_AT, tier.num_frames / float(tier.fps),
            fraction=float(rng.uniform(0.62, 0.74))))
        y_path = 0.55
        half_w = float(rng.uniform(0.78, 1.02))
        screen_h = float(rng.uniform(1.05, 1.35))
        eye = CAMERA

        x0 = -speed * (tier.num_frames / float(tier.fps)) * 0.5
        kind = "sphere" if rng.rand() < 0.6 else "cube"
        ball = BodySpec(
            name="ball", kind=kind, position=(x0, y_path, radius),
            scale=(radius,) * 3, velocity=(speed, 0.0, 0.0), mass=1.0,
            friction=0.02, restitution=0.2,
            color=C.hue_rgb(float(rng.uniform(0, 1))),
            segmentation_id=self.SEG_BALL, role="actor")
        screen = BodySpec(
            name="screen", kind="cube", position=(0.0, 0.0, screen_h),
            scale=(half_w, 0.06, screen_h), mass=0.0, static=True,
            color=(0.22, 0.24, 0.30), segmentation_id=self.SEG_SCREEN,
            role="occluder")

        # "Occluded" means fully occluded (CLAUDE.md) -- the silhouette margin
        # below is the actor's *projected* radius, and a sphere's is `radius`
        # from every angle. A cube's is not: a corner can swing out to
        # `radius * sqrt(3)` (half its space diagonal) depending on how it is
        # tumbling, so use that conservative bound for a cube rather than
        # silently under-covering the true silhouette and marking a frame
        # "occluded" while a corner still pokes out past the screen's edge.
        ball = C.with_material(ball, M.pick(arng), arng)
        silhouette_radius = radius * (1.0 if kind == "sphere" else math.sqrt(3.0))
        occ = _occluded_frames(eye, ball, screen, tier, radius, y_path,
                               silhouette_radius)
        hdri_id = pick_hdri(C.appearance_rng(seed, "hdri")) if cx.background == "hdri" else None

        return SceneSpec(
            scenario=self.name, seed=seed, tier=tier,
            bodies=[C.ground(cx, self.SEG_FLOOR), screen, ball,
                    C.understudy(ball, self.SEG_SPLIT)],
            lights=C.lights(cx, look_at=(0, 0, 0.8)),
            camera_position=CAMERA, camera_look_at=LOOK_AT,
            # NO CAMERA MOTION, until `_occluded_frames` can account for it.
            # That function intersects a camera->ball ray with the screen plane
            # once, at sample time, and the frame list it returns is where
            # every observability label in the dataset comes from -- this is
            # the only scenario that supplies an occlusion interval at all. A
            # camera that moves changes which frames are hidden, and the
            # precomputed list would quietly describe a different clip.
            camera_motion=False,
            floor_level=0.0, complexity=complexity, hdri_id=hdri_id,
            notes={"radius": radius, "speed": speed,
                   "occluded_frames": occ, "actor_kind": kind,
                   "occluder_id": self.SEG_SCREEN})


def _occluded_frames(cam, ball, screen, tier, radius, y_path,
                     silhouette_radius=None) -> List[int]:
    """Frames where the ball is **fully** hidden behind the screen box.

    Intersect the camera->ball ray with the screen's y plane and test the
    crossing against the screen's extents, shrunk by the ball's *projected*
    silhouette radius at that plane. Full occlusion -- not merely substantial --
    is what matters: an injector firing on a frame where even a few pixels of
    the actor still show makes the violation instantly observable, which
    collapses the observability lag this scenario exists to produce.

    `radius` is the body's true half-extent -- it fixes where its centre
    actually sits (`bz`), and must stay exact. `silhouette_radius` is a
    separate, possibly larger bound used only for the occlusion margin: a
    sphere's silhouette is `radius` from every angle, but a cube's is not (a
    corner can swing out to `radius * sqrt(3)`), so the caller passes the
    conservative bound for whatever shape this is. Defaults to `radius` for a
    sphere, where the two coincide.
    """
    if silhouette_radius is None:
        ball = C.with_material(ball, M.pick(arng), arng)
        silhouette_radius = radius
    cx, cy, cz = cam
    sx, sy, sz = screen.position
    hw, _, hh = screen.scale
    out = []
    dt = 1.0 / float(tier.fps)
    for f in range(tier.num_frames):
        bx = ball.position[0] + ball.velocity[0] * f * dt
        by, bz = y_path, radius
        if by <= sy:                       # ball in front of the screen
            continue
        s = (sy - cy) / (by - cy)          # ray parameter at the screen plane
        ix = cx + s * (bx - cx)
        iz = cz + s * (bz - cz)
        r_proj = silhouette_radius * s     # the ball's silhouette at that plane
        if (abs(ix - sx) <= hw - r_proj
                and (sz - hh) + r_proj <= iz <= (sz + hh) - r_proj):
            out.append(f)
    return out


register(OccluderPass())
