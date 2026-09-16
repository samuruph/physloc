"""The MOVi fields PhysLoc derives on the host (`physloc.annotate.movi`).

Conventions are Kubric's, so these pin them against the definitions in
`refs/kubric/kubric/core/{cameras,objects}.py` and `post_processing.py` rather
than against our own reading of them.
"""
from __future__ import annotations

import numpy as np
import pytest

import mockroll
from physloc import scenarios
from physloc.annotate import movi
from physloc.camera import TAN_HALF_FOV
from physloc.scenarios import TIERS
from physloc.sim.trajectory import Contacts

TIER = TIERS["debug"]


def _cam(eye, aim, n=1):
    q = movi.look_at_quaternion(eye, aim)
    return (np.tile(np.asarray(eye, np.float64), (n, 1)), np.tile(q, (n, 1)))


def test_intrinsics_match_kubrics_default_lens():
    K = movi.intrinsics()
    assert K[0, 0] == pytest.approx(50.0 / 36.0)
    assert K[1, 1] == pytest.approx(-50.0 / 36.0)
    assert K[2, 2] == -1.0
    # The same lens `camera.TAN_HALF_FOV` frames every scene with.
    half = np.radians(movi.field_of_view_deg() / 2.0)
    assert np.tan(half) == pytest.approx(TAN_HALF_FOV)


def test_projection_centre_right_and_up():
    eye, aim = (0.0, -8.0, 1.0), (0.0, 0.0, 1.0)
    pos, quat = _cam(eye, aim)
    K = movi.intrinsics()
    pts = np.array([[aim, (1.0, 0.0, 1.0), (0.0, 0.0, 2.0), (0.0, -20.0, 1.0)]])
    p = movi.project(pts, pos, quat, K)[0]
    assert p[0, :2] == pytest.approx([0.5, 0.5])
    assert p[1, 0] > 0.5                       # +x is to the right, seen from -y
    assert p[2, 1] < 0.5                       # up in the world is up in the image
    assert p[0, 2] > 0 and p[3, 2] < 0         # behind the camera flips the sign


def test_projection_agrees_with_the_framing_frustum():
    eye, aim = (0.0, -8.0, 1.0), (0.0, 0.0, 1.0)
    pos, quat = _cam(eye, aim)
    edge = (8.0 * TAN_HALF_FOV, 0.0, 1.0)      # on the right edge of the frustum
    x = movi.project(np.array([[edge]]), pos, quat, movi.intrinsics())[0, 0, 0]
    assert x == pytest.approx(1.0, abs=1e-6)


def test_camera_quaternion_is_a_rotation_that_looks_at_the_target():
    q = movi.look_at_quaternion((3.0, -4.0, 2.0), (0.0, 0.0, 0.5))
    R = movi.rotation_matrix(q)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)
    forward = R @ np.array([0.0, 0.0, -1.0])   # the camera looks down its -Z
    want = np.array([-3.0, 4.0, -1.5])
    assert np.allclose(forward, want / np.linalg.norm(want), atol=1e-9)


def test_bboxes_3d_follow_kubrics_corner_order_and_scale():
    lo, hi = np.array([-1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0])
    pos = np.array([[1.0, 2.0, 3.0]])
    quat = np.array([[1.0, 0.0, 0.0, 0.0]])
    box = movi.bboxes_3d(pos, quat, lo, hi, np.array([0.5, 0.25, 2.0]))[0]
    assert box.shape == (8, 3)
    assert box[0] == pytest.approx([0.5, 1.75, 1.0])   # (lo, lo, lo)
    assert box[7] == pytest.approx([1.5, 2.25, 5.0])   # (hi, hi, hi)
    assert box[1] == pytest.approx([0.5, 1.75, 5.0])   # z varies fastest


def test_seg_boxes_are_normalised_with_exclusive_max_edges():
    seg = np.zeros((3, 10, 20), np.uint16)
    seg[1, 2:5, 4:9] = 7
    got = movi.seg_boxes(seg, 7)
    assert got["bbox_frames"] == [1]
    assert np.isnan(got["bboxes"][0]).all()
    assert got["bboxes"][1] == pytest.approx([0.2, 0.2, 0.5, 0.45])
    assert list(got["visibility"]) == [0, 15, 0]


def test_collisions_are_one_event_per_frame_and_pair():
    pos, quat = _cam((0.0, -8.0, 1.0), (0.0, 0.0, 1.0), n=4)
    c = Contacts(
        frame=np.array([2, 2, 2, 3], np.int32),
        body_a=np.array([1, 2, 1, 2], np.int32),
        body_b=np.array([2, 1, 2, 5], np.int32),
        point=np.array([[0, 0, 0], [0, 0, 2], [9, 9, 9], [1, 1, 1]], np.float32),
        normal=np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 0, 1]], np.float32),
        impulse=np.array([1.0, 3.0, 0.0, 2.0], np.float32),
        penetration=np.zeros(4, np.float32))
    ev = movi.collisions(c, {"positions": pos, "quaternions": quat},
                         movi.intrinsics(), num_frames=4)
    assert [(e["frame"], e["instances"]) for e in ev] == [(2, [1, 2]), (3, [2, 5])]
    assert ev[0]["force"] == 3.0
    assert ev[0]["position"] == pytest.approx([0.0, 0.0, 1.5])  # force-weighted
    assert ev[0]["contact_normal"] == pytest.approx([0.0, 1.0, 0.0])


def test_instance_arrays_have_movi_shapes_on_a_real_scene():
    sc = scenarios.get("drop")
    spec = sc.sample(3, TIER, "L0")
    traj = mockroll.roll(spec, sc)
    T, k = traj.num_frames, len(spec.bodies)
    H = W = TIER.resolution
    seg = np.zeros((T, H, W), np.uint16)
    cam = movi.camera_track(spec, T)
    K = movi.intrinsics(resolution=(H, W))
    arr = movi.instance_arrays(spec, traj, seg, cam, K)
    assert arr["positions"].shape == (k, T, 3)
    assert arr["quaternions"].shape == (k, T, 4)
    assert arr["bboxes_3d"].shape == (k, T, 8, 3)
    assert arr["image_positions"].shape == (k, T, 2)
    assert arr["bboxes"].shape == (k, T, 4)
    assert arr["visibility"].shape == (k, T)
    # The actor, which the scene frames, projects inside the image at frame 0.
    i = [b.role for b in spec.bodies].index("actor")
    assert 0.0 <= arr["image_positions"][i, 0, 0] <= 1.0
    assert 0.0 <= arr["image_positions"][i, 0, 1] <= 1.0
