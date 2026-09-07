"""Optical-domain injectors: shadows that lie about their caster.

The only family whose culprit is not the object but its *shadow*. It relies on
`shadow_track` handing the cast shadow to a body of its own -- see that
scenario's docstring for why a real Blender shadow cannot carry a mask.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..sim.trajectory import Trajectory
from . import _geom
from .base import Injector, InterventionPlan, register


class Shadow(Injector):
    """The shadow detaches from what casts it.

    Displaced along the ground plane, perpendicular to the light's bearing, so
    the shadow is still plausibly *a* shadow -- right shape, right darkness,
    wrong place. Offsetting it along the light bearing instead would read as the
    object being at a different height, which is a lawful reading and therefore
    a much weaker violation.

    The culprit is the shadow body alone. Listing the caster too would put the
    object's own pixels in `violation_mask`, and the object is not what is
    wrong.
    """

    family = "shadow"
    # A shadow that has come loose stays loose: the window runs to the end of
    # the clip and `--window` is ignored. Truncating it left `apply` offsetting
    # the shadow for every frame after `t1` while `active` claimed those frames
    # were lawful -- a violation the clip showed and the annotation denied.
    persistent = True
    OFFSET_RADII = {"weak": 1.2, "medium": 3.0, "strong": 5.5}

    def strong_residual_reference(self, spec) -> float:
        return float(self.OFFSET_RADII["strong"])

    def plan(self, spec, traj, rng, severity_bin) -> Optional[InterventionPlan]:
        shade = next((b for b in spec.bodies if b.role == "shadow"), None)
        caster = self._primary(spec)
        if shade is None or caster is None:
            return None
        T = traj.num_frames
        t0 = max(1, T // 3)
        n_win = self._window_len(max(2, T - t0), t0, T)
        t1 = min(T - 1, t0 + n_win - 1)

        light = np.asarray(spec.notes["light_dir"], np.float64)
        bearing = light[:2]
        norm = float(np.linalg.norm(bearing))
        perp = (np.array([-bearing[1], bearing[0]]) / norm if norm > 1e-6
                else np.array([1.0, 0.0]))
        offset = (self.OFFSET_RADII[severity_bin]
                  * float(caster.bounding_radius)
                  * perp * float(self._instance_rng(spec).choice([-1.0, 1.0])))

        return InterventionPlan(
            family=self.family, kind="sustained", t_event=t0, windows=[(t0, t1)],
            causal_body_ids=[int(shade.segmentation_id)],
            params={"type": "shadow_offset",
                    "offset_m": [float(offset[0]), float(offset[1])],
                    "offset_radii": self.OFFSET_RADII[severity_bin]},
            magnitude=float(np.linalg.norm(offset)),
            magnitude_unit="shadow_caster_offset_radii",
            severity_bin=severity_bin,
            notes={"radius": float(caster.bounding_radius),
                   "surface_top": float(spec.notes.get("surface_top", 0.0)),
                   "caster_id": int(caster.segmentation_id),
                   "light_dir": [float(x) for x in light],
                   "shadow_id": int(shade.segmentation_id)})

    def _apply(self, spec, traj, plan) -> Trajectory:
        out = self._clone(traj)
        shade = next(b for b in spec.bodies if b.role == "shadow")
        bi = traj.index_of(int(shade.segmentation_id))
        t0, t1 = plan.windows[0]
        off = np.asarray(plan.params["offset_m"], np.float32)
        # Eased in over two frames: a shadow that jumps a body-width between
        # consecutive frames reads as a dropped frame rather than as physics.
        n = t1 - t0 + 1
        ramp = np.minimum(1.0, (np.arange(n, dtype=np.float32) + 1.0) / 2.0)
        out.pos[t0:t1 + 1, bi, 0] += off[0] * ramp
        out.pos[t0:t1 + 1, bi, 1] += off[1] * ramp
        if t1 + 1 < traj.num_frames:
            out.pos[t1 + 1:, bi, 0] += off[0]
            out.pos[t1 + 1:, bi, 1] += off[1]
        self._sync_velocity(traj, out, bi, t0)
        out.meta = dict(traj.meta)
        out.meta["intervention"] = plan.to_dict()
        out.meta["label"] = "invalid"
        return out


register(Shadow())


def _projection(spec, traj, plan_notes=None):
    """(caster index, shadow index, unit light dir, surface top) for a scene."""
    shade = next(b for b in spec.bodies if b.role == "shadow")
    L = np.asarray(spec.notes["light_dir"], np.float64)
    L = L / max(float(np.linalg.norm(L)), 1e-9)
    return (traj.index_of(int(spec.notes["caster_id"])),
            traj.index_of(int(shade.segmentation_id)), L,
            float(spec.notes.get("surface_top", 0.0)))


def _lead(spec) -> float:
    """How far the light throws the shadow sideways from under its caster."""
    L = np.asarray(spec.notes["light_dir"], np.float64)
    L = L / max(float(np.linalg.norm(L)), 1e-9)
    throw = float(spec.notes["height"]) / max(abs(float(L[2])), 1e-6)
    return throw * float(np.linalg.norm(L[:2]))


class ShadowInverted(Injector):
    """The shadow is in the wrong place for the *whole* clip, and never moves.

    The other two optical families stage a change: `shadow` detaches partway
    through and slides off, `shadow_shape` stops matching its caster. Both put
    a lawful prefix in front of the violation, which is what the twin structure
    is built around -- and both are therefore detectable by watching for the
    moment something alters, without ever reasoning about light at all.

    This one has no such moment. The shadow tracks its caster perfectly, at
    perfectly constant offset, from the first frame to the last; it is simply
    on the side of the object the light is coming *from*. Nothing changes, so
    nothing can be caught changing, and a model has to actually compare the
    shadow's bearing against the key light's to notice. That is LikePhys's
    "inverted shadow", and it is the one clip type this dataset had no way to
    express.

    **`t_event` is 0, and that is deliberate.** The identical prefix is empty,
    so this family alone ships a pair whose `divergence_map` is non-zero
    everywhere and whose "before" is nothing at all. `violation_mask` and
    `severity_map` are unaffected -- they are computed from the culprit's
    footprint and its residual, neither of which needs a lawful prefix -- and
    the valid twin is still exactly the counterfactual, still rendered from the
    same seed. What is lost is only the "spot the change" shortcut, which is
    the point.
    """

    family = "shadow_inverted"
    persistent = True
    #: Where the shadow is put, as a fraction of the offset the light actually
    #: implies. 1.0 would be correct. 0.0 parks it directly beneath the body --
    #: which is what an overhead sun would give, and the scene's key light is
    #: visibly not overhead. Negative mirrors it to the lit side, which no
    #: light can produce at all.
    FRACTION_BY_BIN = {"weak": 0.15, "medium": -0.40, "strong": -1.00}

    def strong_residual_reference(self, spec) -> float:
        r = max(float(spec.notes["radius"]), 1e-9)
        return abs(1.0 - self.FRACTION_BY_BIN["strong"]) * _lead(spec) / r

    def plan(self, spec, traj, rng, severity_bin) -> Optional[InterventionPlan]:
        shade = next((b for b in spec.bodies if b.role == "shadow"), None)
        caster = self._primary(spec)
        if shade is None or caster is None or "caster_id" not in spec.notes:
            return None
        k = float(self.FRACTION_BY_BIN[severity_bin])
        radius = max(float(caster.bounding_radius), 1e-9)
        return InterventionPlan(
            family=self.family, kind="sustained", t_event=0,
            windows=[(0, traj.num_frames - 1)],
            causal_body_ids=[int(shade.segmentation_id)],
            params={"type": "shadow_invert", "offset_fraction": k},
            magnitude=abs(1.0 - k) * _lead(spec) / radius,
            magnitude_unit="shadow_caster_offset_radii",
            severity_bin=severity_bin,
            notes={"radius": radius,
                   "surface_top": float(spec.notes.get("surface_top", 0.0)),
                   "caster_id": int(spec.notes["caster_id"]),
                   "light_dir": [float(x) for x in spec.notes["light_dir"]],
                   "shadow_id": int(shade.segmentation_id)})

    def _apply(self, spec, traj, plan) -> Trajectory:
        out = self._clone(traj)
        ci, si, L, top = _projection(spec, traj)
        k = float(plan.params["offset_fraction"])

        p = np.asarray(traj.pos[:, ci, :], np.float64)
        denom = -L[2] if abs(L[2]) > 1e-6 else -1e-6
        t = (p[:, 2] - top) / denom
        expected = p[:, :2] + t[:, None] * L[None, :2]
        # Scaled about the point directly beneath the caster, so the shadow
        # still tracks the body exactly -- it is the bearing that is wrong, not
        # the tracking. Anything that made the offset drift would be a second,
        # unlabelled violation for a viewer to notice first.
        out.pos[:, si, :2] = (p[:, :2]
                              + k * (expected - p[:, :2])).astype(np.float32)
        self._sync_velocity(traj, out, si, 0)
        out.meta = dict(traj.meta)
        out.meta["intervention"] = plan.to_dict()
        out.meta["label"] = "invalid"
        return out


register(ShadowInverted())
