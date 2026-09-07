"""THE SEAM -- docs/PLAN.md Part 4.

Everything upstream (scenario sampling, simulation, injection) writes one of these;
everything downstream (rendering, residuals, annotation) reads one. Swapping the
dynamics backend touches nothing below this line.

Imported by BOTH environments (py3.9 + numpy 1.21 in the container, py3.11 +
numpy 2.x on the host), so: numpy only, no 3.10+ syntax.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

SCHEMA_VERSION = 0


@dataclass
class Contacts:
    """Contact events, flattened. Frame f's contacts are the rows where frame == f."""

    frame: np.ndarray          # [N] int32
    body_a: np.ndarray         # [N] int32
    body_b: np.ndarray         # [N] int32
    point: np.ndarray          # [N, 3] float32   world-space contact point
    normal: np.ndarray         # [N, 3] float32   on body_a, pointing toward body_b
    impulse: np.ndarray        # [N] float32
    penetration: np.ndarray    # [N] float32      signed; negative == interpenetrating

    @staticmethod
    def empty() -> "Contacts":
        z_i = np.zeros((0,), np.int32)
        z_f = np.zeros((0,), np.float32)
        return Contacts(z_i, z_i.copy(), z_i.copy(), np.zeros((0, 3), np.float32),
                        np.zeros((0, 3), np.float32), z_f, z_f.copy())

    def at(self, frame: int) -> "Contacts":
        m = self.frame == frame
        return Contacts(self.frame[m], self.body_a[m], self.body_b[m], self.point[m],
                        self.normal[m], self.impulse[m], self.penetration[m])

    def __len__(self) -> int:
        return int(self.frame.shape[0])


@dataclass
class Trajectory:
    """Per-body kinematics over T frames, plus contacts, plus what was done to it."""

    # -- identity ----------------------------------------------------------
    body_ids: np.ndarray       # [B] int32  -- match seg.npz instance ids
    body_names: List[str]

    # -- kinematics, T frames x B bodies -----------------------------------
    pos: np.ndarray            # [T, B, 3] float32
    quat: np.ndarray           # [T, B, 4] float32  (w, x, y, z) -- Kubric order
    lin_vel: np.ndarray        # [T, B, 3] float32
    ang_vel: np.ndarray        # [T, B, 3] float32

    # -- static per-body properties ----------------------------------------
    mass: np.ndarray           # [B] float32
    radius: np.ndarray         # [B] float32  -- bounding radius, normalises residuals
    is_static: np.ndarray      # [B] bool

    # [T, B] bool -- is the body in the scene at all on this frame? False is how
    # a `permanence` violation is expressed at the seam: the body is not moved
    # somewhere odd, it is *gone*. Defaults to all-True.
    present: Optional[np.ndarray] = None

    # [T, B, 3] float32 -- per-frame multiplier on each body's declared scale.
    # This is how `immutability` is expressed at the seam: the body is not
    # replaced, its size changes. Defaults to all-ones, which the renderer reads
    # as "leave the declared scale alone".
    scale_mul: Optional[np.ndarray] = None

    # [T, B, 3] float32 -- per-frame RGB, absolute. This is how `colour_shift`
    # is expressed at the seam: not a new kind of edit, just another channel the
    # renderer keyframes, exactly like `scale_mul`. Defaults to each body's
    # declared colour, which the worker fills in.
    colour: Optional[np.ndarray] = None

    # [T, B] float32 -- 1 is solid, 0 is invisible. How `dissolve` is expressed
    # at the seam: the renderer mixes the shaded surface with a Transparent BSDF
    # and keyframes the blend, so a body can fade out without changing shape.
    opacity: Optional[np.ndarray] = None

    contacts: Contacts = None

    # -- scene constants ---------------------------------------------------
    fps: float = 12.0
    gravity: np.ndarray = None  # [3] float32

    # -- provenance / intervention ----------------------------------------
    meta: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    def __post_init__(self) -> None:
        if self.present is None:
            self.present = np.ones(self.pos.shape[:2], dtype=bool)
        if self.scale_mul is None:
            self.scale_mul = np.ones(self.pos.shape, dtype=np.float32)
        if self.colour is None:
            self.colour = np.zeros(self.pos.shape, dtype=np.float32)
        if self.opacity is None:
            self.opacity = np.ones(self.pos.shape[:2], dtype=np.float32)

    @property
    def num_frames(self) -> int:
        return int(self.pos.shape[0])

    @property
    def num_bodies(self) -> int:
        return int(self.pos.shape[1])

    @property
    def dt(self) -> float:
        return 1.0 / float(self.fps)

    def index_of(self, body_id: int) -> int:
        hits = np.flatnonzero(self.body_ids == body_id)
        if hits.size == 0:
            raise KeyError("body_id %d not in trajectory" % body_id)
        return int(hits[0])

    def acceleration(self, b: int) -> np.ndarray:
        """[T, 3] finite-difference acceleration of body index `b`.

        Central differences inside, one-sided at the ends. This is what the
        free-fall and momentum residuals are computed from.
        """
        v = self.lin_vel[:, b, :]
        a = np.zeros_like(v)
        if v.shape[0] >= 3:
            a[1:-1] = (v[2:] - v[:-2]) / (2.0 * self.dt)
        if v.shape[0] >= 2:
            a[0] = (v[1] - v[0]) / self.dt
            a[-1] = (v[-1] - v[-2]) / self.dt
        return a

    # ------------------------------------------------------------------ #
    def save(self, path) -> None:
        np.savez_compressed(
            str(path),
            schema_version=np.int32(SCHEMA_VERSION),
            body_ids=self.body_ids, body_names=np.array(self.body_names, dtype=object),
            pos=self.pos, quat=self.quat, lin_vel=self.lin_vel, ang_vel=self.ang_vel,
            mass=self.mass, radius=self.radius, is_static=self.is_static,
            present=self.present, scale_mul=self.scale_mul,
            colour=self.colour, opacity=self.opacity,
            contact_frame=self.contacts.frame, contact_a=self.contacts.body_a,
            contact_b=self.contacts.body_b, contact_point=self.contacts.point,
            contact_normal=self.contacts.normal, contact_impulse=self.contacts.impulse,
            contact_penetration=self.contacts.penetration,
            fps=np.float32(self.fps), gravity=self.gravity,
            meta_json=np.array(json.dumps(self.meta, sort_keys=True)),
        )

    @staticmethod
    def load(path) -> "Trajectory":
        z = np.load(str(path), allow_pickle=True)
        return Trajectory(
            body_ids=z["body_ids"], body_names=[str(n) for n in z["body_names"]],
            pos=z["pos"], quat=z["quat"], lin_vel=z["lin_vel"], ang_vel=z["ang_vel"],
            mass=z["mass"], radius=z["radius"], is_static=z["is_static"],
            present=z["present"] if "present" in z.files else None,
            scale_mul=z["scale_mul"] if "scale_mul" in z.files else None,
            colour=z["colour"] if "colour" in z.files else None,
            opacity=z["opacity"] if "opacity" in z.files else None,
            contacts=Contacts(z["contact_frame"], z["contact_a"], z["contact_b"],
                              z["contact_point"], z["contact_normal"],
                              z["contact_impulse"], z["contact_penetration"]),
            fps=float(z["fps"]), gravity=z["gravity"],
            meta=json.loads(str(z["meta_json"])),
        )


def prefix_identical(a: Trajectory, b: Trajectory, upto: int,
                     atol: float = 0.0) -> Tuple[bool, Optional[str]]:
    """Are two trajectories identical for every frame < `upto`?

    The physics-level counterpart of the pixel-level prefix-identity test. Cheap
    enough to assert on every generated pair, and it localises a determinism bug
    to the simulator rather than the renderer.
    """
    if a.num_bodies != b.num_bodies:
        return False, "body count %d != %d" % (a.num_bodies, b.num_bodies)
    for name in ("pos", "quat", "lin_vel", "ang_vel", "scale_mul", "colour",
                 "opacity"):
        x = getattr(a, name)[:upto]
        y = getattr(b, name)[:upto]
        if x.shape != y.shape:
            return False, "%s shape %s != %s" % (name, x.shape, y.shape)
        d = np.abs(x - y)
        if d.size and float(d.max()) > atol:
            f = int(np.unravel_index(int(np.argmax(d)), d.shape)[0])
            return False, "%s diverges at frame %d by %.3e" % (name, f, float(d.max()))
    return True, None
