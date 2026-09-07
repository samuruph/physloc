"""Camera geometry, shared by scene sampling and by the injectors.

Lives at the top of the package rather than inside either, because both sides
need the same frustum and a duplicated `TAN_HALF_FOV` is a silent way for a
scenario to frame a shot the injectors then judge by a different rule.

Pure stdlib + numpy: imported by BOTH environments, like `taxonomy`.
"""
from __future__ import annotations

from typing import NamedTuple, Tuple

import numpy as np

#: Kubric's default camera is a 50 mm lens on a 36 mm sensor, so the half-angle
#: tangent is 18/50. The render is square, so this is the half-width *and* the
#: half-height.
TAN_HALF_FOV = 18.0 / 50.0

#: Reference frustum half-extent, in metres. The scenes that were hand-framed
#: before anything scaled -- camera ~10 m out, bodies ~0.35 m -- all sit near
#: this, so it is the unit `scene_scale` is expressed in: a scene with
#: `half_extent == REFERENCE_HALF_EXTENT` needs lights and a ground plane at
#: exactly their hand-tuned sizes.
REFERENCE_HALF_EXTENT = 4.35

GRAVITY = 9.81


def camera_basis(camera_position, camera_look_at):
    """(eye, forward, right, up) for a camera, as unit vectors."""
    eye = np.asarray(camera_position, np.float64)
    fwd = np.asarray(camera_look_at, np.float64) - eye
    fwd /= max(float(np.linalg.norm(fwd)), 1e-9)
    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(fwd, world_up)
    n = float(np.linalg.norm(right))
    right = (np.array([1.0, 0.0, 0.0]) if n < 1e-6 else right / n)
    up = np.cross(right, fwd)
    return eye, fwd, right, up


def visible(camera_position, camera_look_at, points: np.ndarray,
            margin: float = 0.04) -> np.ndarray:
    """[...] bool: are these world points inside the camera's view?"""
    eye, fwd, right, up = camera_basis(camera_position, camera_look_at)
    p = np.asarray(points, np.float64) - eye
    z = p @ fwd
    with np.errstate(divide="ignore", invalid="ignore"):
        x = (p @ right) / z
        y = (p @ up) / z
    lim = TAN_HALF_FOV * (1.0 - margin)
    return (z > 1e-3) & (np.abs(x) <= lim) & (np.abs(y) <= lim)


# --------------------------------------------------------------------------
# Framing a free-flight scenario
# --------------------------------------------------------------------------


class Flight(NamedTuple):
    """A ballistic arc and the camera that frames the whole of it."""

    apex: float               # rise above the launch point, metres
    launch_z: float           # height the body is thrown from
    vz: float                 # launch speed, straight up
    half_extent: float        # frustum half-height at the arc's depth
    distance: float           # how far back the camera sits
    look_z: float             # height the camera aims at
    radius: float             # actor radius that holds its on-screen size
    x_travel: float           # horizontal distance covered over the flight
    scene_scale: float        # multiplier for lights, ground, anything else


def frame_flight(flight_seconds: float, angular_radius: float = 0.11,
                 fill: float = 0.80, launch_fraction: float = 0.11,
                 x_fraction: float = 0.52) -> Flight:
    """Frame a body that is airborne for `flight_seconds`, whatever that costs.

    **The apex of a ballistic arc is not ours to choose.** A body airborne for
    T seconds rises `g*T^2/8` and no camera, mass or launch angle changes that.
    So when the clip length doubled (13 -> 25 frames at the debug tier, 25 -> 49 at
    tier v0) the arcs got *four* times taller while the cameras stayed where they
    had been hand-tuned for the short clips, and `toss` and `tumble` threw their
    actor clean out of shot -- measured at 42% and 16% of frames still in view
    at the debug tier, and 1% and 0% at tier v0. Every violation staged mid-flight was
    being injected off-screen.

    The fix is to stop hand-tuning the camera and derive it, which also makes
    the two scenarios scale-invariant: a longer clip is the *same shot* of a
    bigger scene from further away, so the actor covers the same pixels at
    every tier and a bug found at the debug tier looks the same at tier v0.

    `angular_radius` is the actor's radius as a fraction of the frustum
    half-extent -- the one number that fixes its on-screen size. `fill` is how
    much of the frame the arc is allowed to use; the rest is the margin that
    keeps `antigravity` and `phantom_impulse` from having to clamp themselves
    down to stay in shot.
    """
    apex = GRAVITY * flight_seconds * flight_seconds / 8.0
    # An arc `T` seconds long is roughly ten times as tall as the actor is
    # wide, and the frame is square, so most of the shot is necessarily empty
    # sky and the actor is necessarily small. `launch_fraction` is kept just
    # high enough to clear the body's own radius -- every metre of launch
    # height is a metre of arc the camera has to back off to include.
    launch_z = launch_fraction * apex
    top = launch_z + apex
    # fill*H = top/2 + radius, and radius = angular_radius*H. Solve for H.
    span = max(fill - angular_radius, 0.05)
    half = (top / 2.0) / span
    return Flight(
        apex=apex, launch_z=launch_z,
        vz=float(np.sqrt(2.0 * GRAVITY * apex)),
        half_extent=half, distance=half / TAN_HALF_FOV, look_z=top / 2.0,
        radius=angular_radius * half, x_travel=x_fraction * 2.0 * fill * half,
        scene_scale=half / REFERENCE_HALF_EXTENT)


def flight_camera(f: Flight, sway: float = 0.06,
                  tilt: float = 0.10) -> Tuple[Tuple[float, float, float],
                                               Tuple[float, float, float]]:
    """(position, look_at) for a `Flight`.

    Level-on would frame the arc exactly, but shows almost no ground and reads
    as a flat cut-out. `sway` slides the eye sideways for a little parallax and
    `tilt` drops it below the aim point so the floor plane is visible; both are
    inside the `fill` margin.
    """
    return ((sway * f.distance, -f.distance, f.look_z * (1.0 - tilt)),
            (0.0, 0.0, f.look_z))


# --------------------------------------------------------------------------
# Framing a scenario whose extent is set by how far something travels
# --------------------------------------------------------------------------


def frame_extent(camera_position, camera_look_at) -> float:
    """Frustum half-extent, in metres, at the depth of the aim point.

    The render is square, so this is the half-width and the half-height both.
    """
    d = float(np.linalg.norm(np.asarray(camera_look_at, np.float64)
                             - np.asarray(camera_position, np.float64)))
    return TAN_HALF_FOV * d


def traverse_speed(camera_position, camera_look_at, duration: float,
                   fraction: float = 0.55) -> float:
    """Speed that carries a body `fraction` of the frame width in `duration`.

    The dial that keeps a rolling or sliding scenario framed when the clip
    length changes. A body on a near-frictionless floor covers `v*T` whatever
    else is true, so a hand-picked `v` frames correctly for exactly one clip
    length -- which is how `collision`, `barrier_pass` and `occluder_pass` came
    to roll their actors out of shot when the tiers went from 13 frames to 25
    and 49. Deriving it makes the actor cross the same fraction of the frame in
    every tier, so the shot is the same and only the pace changes.
    """
    return fraction * 2.0 * frame_extent(camera_position, camera_look_at) / duration


def frame_box(x_range, z_range, fill: float = 0.80, sway: float = 0.06,
              tilt: float = 0.10) -> Tuple[Tuple[float, float, float],
                                           Tuple[float, float, float]]:
    """(position, look_at) for a camera that frames a world-space x/z box.

    For scenarios whose extent is geometry rather than a speed -- a ramp plus
    the run-out its block coasts through -- where the honest thing is to work
    out where the action happens and point the camera at *that*, rather than
    tune an eye position against one tier and hope.
    """
    xc = 0.5 * (x_range[0] + x_range[1])
    zc = 0.5 * (z_range[0] + z_range[1])
    half = max(x_range[1] - x_range[0], z_range[1] - z_range[0]) / (2.0 * fill)
    d = half / TAN_HALF_FOV
    return ((xc + sway * d, -d, zc - tilt * half), (xc, 0.0, zc))
