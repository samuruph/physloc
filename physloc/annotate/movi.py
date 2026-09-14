"""The MOVi annotation fields, derived on the host from what a clip already has.

Kubric's MOVi datasets (`refs/kubric/challenges/movi/README.md`) describe a
clip as `metadata`, `camera`, `instances` and `events`, with dense passes named
`segmentations`, `depth`, `forward_flow`, `backward_flow`, `normal` and
`object_coordinates`. PhysLoc renders the same passes through the same
renderer, and its trajectory carries every pose, velocity and contact -- but it
never wrote the camera intrinsics, the camera's rotation, 2D or 3D boxes, image
positions, per-frame visibility, or collision events. Everything here computes
those from the scene spec, the trajectory and the segmentation, so no render or
simulation has to be redone.

Conventions are Kubric's, copied rather than approximated, so a MOVi loader
reads these arrays the way it reads MOVi's:

* `K` is `PerspectiveCamera.intrinsics` -- NORMALISED to a 1x1 image, with the
  -y / -1 signs of a camera looking down its -Z axis.
* camera quaternions are `look_at_quat(position, target, up="Y", front="-Z")`,
  (w, x, y, z), camera-to-world.
* `image_positions` are `Camera.project_point(p)[:2]`: (x, y) in [0, 1], y down.
* `bboxes` are `post_processing.compute_bboxes`: (ymin, xmin, ymax, xmax) in
  [0, 1], max edges exclusive.
* body quaternions are PyBullet's as the trajectory stores them, (w, x, y, z).

One deliberate difference: instances are keyed by the DECLARED segmentation id
and kept in declaration order, not re-indexed by visibility. The valid and
invalid twins must agree on who is who, and a visibility sort would order them
differently.

Pure numpy: imported by the host only.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

#: Kubric's default lens, which every PhysLoc camera uses -- see
#: `camera.TAN_HALF_FOV`, which is derived from the same two numbers.
FOCAL_LENGTH_MM = 50.0
SENSOR_WIDTH_MM = 36.0


# ------------------------------------------------------------------ camera --
def intrinsics(focal_length: float = FOCAL_LENGTH_MM,
               sensor_width: float = SENSOR_WIDTH_MM,
               resolution: Tuple[int, int] = (1, 1)) -> np.ndarray:
    """`PerspectiveCamera.intrinsics`: normalised K, for an (H, W) image."""
    h, w = (float(v) for v in resolution)
    sensor_height = sensor_width / w * h
    fx = focal_length / sensor_width
    fy = focal_length / sensor_height
    return np.array([[fx, 0.0, -0.5],
                     [0.0, -fy, -0.5],
                     [0.0, 0.0, -1.0]], np.float64)


def field_of_view_deg(focal_length: float = FOCAL_LENGTH_MM,
                      sensor_width: float = SENSOR_WIDTH_MM) -> float:
    """Horizontal field of view, in degrees, as MOVi reports it."""
    return float(np.degrees(2.0 * np.arctan2(sensor_width / 2.0, focal_length)))


def _normalise(v: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return fallback if n < 1e-12 else v / n


def quat_from_matrix(m: np.ndarray) -> np.ndarray:
    """(w, x, y, z) of a rotation matrix, with w >= 0."""
    m = np.asarray(m, np.float64)
    t = float(np.trace(m))
    if t > 0.0:
        s = 0.5 / np.sqrt(t + 1.0)
        q = np.array([0.25 / s, (m[2, 1] - m[1, 2]) * s,
                      (m[0, 2] - m[2, 0]) * s, (m[1, 0] - m[0, 1]) * s])
    else:
        i = int(np.argmax(np.diag(m)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = np.sqrt(max(m[i, i] - m[j, j] - m[k, k] + 1.0, 1e-12)) * 2.0
        q = np.zeros(4)
        q[i + 1] = 0.25 * s
        q[0] = (m[k, j] - m[j, k]) / s
        q[j + 1] = (m[j, i] + m[i, j]) / s
        q[k + 1] = (m[k, i] + m[i, k]) / s
    q /= np.linalg.norm(q)
    return q if q[0] >= 0.0 else -q


def rotation_matrix(q) -> np.ndarray:
    """3x3 rotation of a (w, x, y, z) quaternion (normalised first)."""
    w, x, y, z = np.asarray(q, np.float64) / max(float(np.linalg.norm(q)), 1e-12)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def look_at_quaternion(position, target) -> np.ndarray:
    """`kubric.core.objects.look_at_quat(position, target, "Y", "-Z")`."""
    world_up = np.array([0.0, 0.0, 1.0])
    world_right = np.array([1.0, 0.0, 0.0])
    up = np.array([0.0, 1.0, 0.0])
    front = np.array([0.0, 0.0, -1.0])
    right = np.cross(up, front)
    look_front = _normalise(np.asarray(target, np.float64)
                            - np.asarray(position, np.float64), -up)
    look_right = _normalise(np.cross(world_up, look_front), world_right)
    look_up = _normalise(np.cross(look_front, look_right), world_up)
    r1 = np.stack([look_right, look_up, look_front])
    r2 = np.stack([right, up, front])
    return quat_from_matrix(r1.T @ r2)


def camera_track(spec, num_frames: int) -> Dict[str, np.ndarray]:
    """Per-frame camera `positions` [s,3] and `quaternions` [s,4], from
    `SceneSpec.camera_at` -- the same poses the renderer keyframes."""
    pos = np.zeros((num_frames, 3), np.float64)
    quat = np.zeros((num_frames, 4), np.float64)
    for f in range(num_frames):
        eye, aim = spec.camera_at(f, num_frames)
        pos[f] = eye
        quat[f] = look_at_quaternion(eye, aim)
    return {"positions": pos, "quaternions": quat}


def project(points: np.ndarray, cam_positions: np.ndarray,
            cam_quaternions: np.ndarray, K: np.ndarray) -> np.ndarray:
    """`Camera.project_point` per frame: [s, ..., 3] world -> [s, ..., 3] of
    (x, y) in [0, 1] and the sign of the projective depth (+1 in front)."""
    pts = np.asarray(points, np.float64)
    s = pts.shape[0]
    flat = pts.reshape(s, -1, 3)
    out = np.zeros(flat.shape, np.float64)
    for f in range(s):
        R = rotation_matrix(cam_quaternions[f])
        cam = (flat[f] - cam_positions[f]) @ R           # world -> camera
        proj = cam @ np.asarray(K, np.float64).T
        z = proj[:, 2:3]
        with np.errstate(divide="ignore", invalid="ignore"):
            out[f, :, :2] = proj[:, :2] / z
        out[f, :, 2] = np.sign(z[:, 0])
    return out.reshape(pts.shape)


# ---------------------------------------------------------------- instances --
def body_bounds(body) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(lo, hi, scale) of a body's mesh in its own frame, as DRAWN."""
    lo, hi = body.mesh_bounds
    return (np.asarray(lo, np.float64), np.asarray(hi, np.float64),
            np.asarray(body.draw_scale, np.float64))


def bboxes_3d(pos: np.ndarray, quat: np.ndarray, lo: np.ndarray, hi: np.ndarray,
              scale: np.ndarray, scale_mul: Optional[np.ndarray] = None) -> np.ndarray:
    """[s, 8, 3] world corners -- `PhysicalObject.bbox_3d`, per frame.

    Corners are ordered as `itertools.product(x, y, z)` over (lo, hi), which is
    Kubric's order. `scale_mul` [s, 3] carries a violation that resizes the
    body (`immutability`, `deformation`).
    """
    s = pos.shape[0]
    xs, ys, zs = zip(lo, hi)
    corners = np.array([[x, y, z] for x in xs for y in ys for z in zs], np.float64)
    out = np.zeros((s, 8, 3), np.float64)
    for f in range(s):
        k = scale * (scale_mul[f] if scale_mul is not None else 1.0)
        out[f] = pos[f] + (corners * k) @ rotation_matrix(quat[f]).T
    return out


def seg_boxes(seg: np.ndarray, body_id: int) -> Dict[str, object]:
    """`compute_bboxes` and `compute_visibility` for one declared id.

    Returns `bboxes` [s, 4] (NaN where the body has no pixels), `bbox_frames`
    and `visibility` [s] pixel counts.
    """
    s = seg[..., 0] if seg.ndim == 4 else seg
    T, H, W = s.shape
    boxes = np.full((T, 4), np.nan, np.float32)
    vis = np.zeros((T,), np.int64)
    frames: List[int] = []
    for t in range(T):
        ys, xs = np.nonzero(s[t] == body_id)
        vis[t] = ys.size
        if ys.size:
            boxes[t] = (ys.min() / H, xs.min() / W, (ys.max() + 1) / H,
                        (xs.max() + 1) / W)
            frames.append(t)
    return {"bboxes": boxes, "bbox_frames": frames, "visibility": vis}


def instance_arrays(spec, traj, seg: np.ndarray, cam: Dict[str, np.ndarray],
                    K: np.ndarray) -> Dict[str, np.ndarray]:
    """The MOVi per-instance tensors, [k, s, ...], in `spec.bodies` order."""
    T = int(traj.num_frames)
    k = len(spec.bodies)
    out = {
        "ids": np.zeros((k,), np.int32),
        "positions": np.zeros((k, T, 3), np.float32),
        "quaternions": np.zeros((k, T, 4), np.float32),
        "velocities": np.zeros((k, T, 3), np.float32),
        "angular_velocities": np.zeros((k, T, 3), np.float32),
        "bboxes_3d": np.zeros((k, T, 8, 3), np.float32),
        "image_positions": np.zeros((k, T, 2), np.float32),
        "bboxes": np.full((k, T, 4), np.nan, np.float32),
        "visibility": np.zeros((k, T), np.int32),
    }
    scale_mul = getattr(traj, "scale_mul", None)
    for i, body in enumerate(spec.bodies):
        bid = int(body.segmentation_id)
        j = traj.index_of(bid)
        out["ids"][i] = bid
        pos = np.asarray(traj.pos[:, j, :], np.float64)
        quat = np.asarray(traj.quat[:, j, :], np.float64)
        out["positions"][i] = pos
        out["quaternions"][i] = quat
        out["velocities"][i] = traj.lin_vel[:, j, :]
        out["angular_velocities"][i] = traj.ang_vel[:, j, :]
        lo, hi, sc = body_bounds(body)
        mul = (np.asarray(scale_mul[:, j, :], np.float64)
               if scale_mul is not None else None)
        out["bboxes_3d"][i] = bboxes_3d(pos, quat, lo, hi, sc, mul)
        out["image_positions"][i] = project(pos[:, None, :], cam["positions"],
                                            cam["quaternions"], K)[:, 0, :2]
        boxes = seg_boxes(seg, bid)
        out["bboxes"][i] = boxes["bboxes"]
        out["visibility"][i] = boxes["visibility"]
    return out


# ------------------------------------------------------------------- events --
def collisions(contacts, cam: Dict[str, np.ndarray], K: np.ndarray,
               num_frames: int, min_force: float = 1e-6) -> List[Dict[str, object]]:
    """`events.collisions`: one record per (frame, pair of bodies).

    The trajectory stores a contact row per substep and per contact point, so
    a ball resting on the floor produces twenty rows a frame. Kubric's own
    event list is per solver report too, and MOVi consumers expect events, not
    solver rows -- so rows are grouped by frame and unordered pair: `force` is
    the largest, `position` the force-weighted mean, `contact_normal` that of
    the largest row. `instances` are declared segmentation ids, not indices.
    """
    frame = np.asarray(getattr(contacts, "frame", ()), np.int64)
    if frame.size == 0:
        return []
    a = np.asarray(contacts.body_a, np.int64)
    b = np.asarray(contacts.body_b, np.int64)
    point = np.asarray(contacts.point, np.float64).reshape(-1, 3)
    normal = np.asarray(contacts.normal, np.float64).reshape(-1, 3)
    force = np.asarray(contacts.impulse, np.float64)
    groups: Dict[Tuple[int, int, int], List[int]] = {}
    for r in range(frame.size):
        if force[r] <= min_force or not (0 <= frame[r] < num_frames):
            continue
        key = (int(frame[r]),) + tuple(sorted((int(a[r]), int(b[r]))))
        groups.setdefault(key, []).append(r)
    out: List[Dict[str, object]] = []
    for (f, lo_id, hi_id), rows in sorted(groups.items()):
        rows = np.asarray(rows)
        w = force[rows]
        top = int(rows[int(np.argmax(w))])
        where = (point[rows] * w[:, None]).sum(axis=0) / float(w.sum())
        img = project(where[None, None, :], cam["positions"][f:f + 1],
                      cam["quaternions"][f:f + 1], K)[0, 0, :2]
        out.append({
            "instances": [lo_id, hi_id],
            "frame": f,
            "force": float(w.max()),
            "position": [float(v) for v in where],
            "image_position": [float(v) for v in img],
            "contact_normal": [float(v) for v in normal[top]],
        })
    return out
