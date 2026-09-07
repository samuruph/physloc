"""A resized body's collider must be the shape that is drawn.

`ShapeSwap` is the only way a body changes size mid-run -- `immutability`,
`deformation` and `fission` all go through it -- and it builds a convex hull
from these vertices. Every number below was measured in the pinned image by
`physloc/render/probe_fission.py`, dropping the real KuBasic asset and the
proxy onto the same plane and reading where each came to rest.

The bug this exists for: the hull was a unit sphere spanning [-1, +1] for every
kind that is not a cube. At scale 0.5 the KuBasic cone rests at z = 0.156 and
that proxy rested at 0.395 -- so `drop x fission`'s two halves hung in the air,
higher than the cone they split from while being drawn smaller. Following the
asset's own bounds puts a half at 0.123 against a correct 0.124.
"""
import numpy as np
import pytest

from physloc.render.stepper import KIND_BOUNDS, hull_for, unit_hull

KINDS = sorted(KIND_BOUNDS)

#: Where each kind comes to rest, at scale 0.5, on a plane. `native` is the
#: KuBasic asset itself; anything built here has to agree with it.
CONE_NATIVE_REST = 0.1560
CONE_SCALE = 0.5


@pytest.mark.parametrize("kind", KINDS)
def test_the_hull_fills_the_meshs_own_bounds(kind):
    """Not a [-1, +1] cube. A KuBasic cylinder is half that on every axis and a
    torus a fifth as thick as it is wide; assuming the cube made a cylinder
    proxy twice the size of the cylinder."""
    lo, hi = np.asarray(KIND_BOUNDS[kind], np.float64)
    v = hull_for(kind)
    assert np.allclose(v.min(0), lo, atol=1e-6), kind
    assert np.allclose(v.max(0), hi, atol=1e-6), kind


def test_the_cone_is_not_centred_on_its_own_origin():
    """The one that floated. Its base is at z = -0.306 and its apex at +0.900,
    so a hull symmetric about the origin puts the collision surface 0.24 mesh
    units below the cone -- and the cone that far above the floor."""
    lo, hi = KIND_BOUNDS["cone"]
    assert lo[2] == pytest.approx(-0.3062)
    assert hi[2] == pytest.approx(0.8997)
    v = hull_for("cone")
    apex = v[np.argmax(v[:, 2])]
    assert apex[:2] == pytest.approx([0.0, 0.0]), "the apex is on the axis"
    base = v[v[:, 2] < 0]
    assert len(base) >= 12, "the base is a ring, not a point"
    assert np.allclose(np.linalg.norm(base[:, :2], axis=1), 0.6007, atol=1e-3)


def test_a_cone_proxy_rests_where_the_cone_rests():
    """The measurement, reproduced from geometry alone.

    A body resting on a plane sits with its lowest point on it, so its origin
    ends up `-lo_z * scale` above the floor. Measured in the image: 0.1560 for
    the native asset at scale 0.5 and 0.1541 for this hull -- within the
    solver's contact slop. The old sphere hull gave 0.3948.
    """
    lo, _ = KIND_BOUNDS["cone"]
    predicted = -lo[2] * CONE_SCALE
    assert predicted == pytest.approx(CONE_NATIVE_REST, abs=0.005)
    sphere_rest = -unit_hull().min(0)[2] * CONE_SCALE
    assert sphere_rest > CONE_NATIVE_REST * 3.0, (
        "the regression this test exists for should be glaring")


def test_scaling_the_hull_scales_where_it_rests():
    """`fission` cleaves to 0.5**(1/3). A half must rest lower than the whole
    it came from -- it is drawn smaller -- and the old hull had it rest
    higher."""
    from physloc.injectors.identity import Fission

    k = Fission.CLEAVE_SCALE
    lo, _ = KIND_BOUNDS["cone"]
    whole = -lo[2] * CONE_SCALE
    half = -lo[2] * CONE_SCALE * k
    assert half < whole
    assert half == pytest.approx(0.1238, abs=0.003)


def test_live_bounds_win_over_the_table():
    """The table is a fallback for the host rollout, which has no Kubric to
    ask. When there is an asset, its own bounds are the authority -- so a
    KuBasic revision cannot silently desync the collider from the mesh."""
    odd = ((-2.0, -3.0, -4.0), (2.0, 3.0, 5.0))
    v = hull_for("cone", odd)
    assert np.allclose(v.min(0), odd[0], atol=1e-6)
    assert np.allclose(v.max(0), odd[1], atol=1e-6)
    assert not np.allclose(v.min(0), hull_for("cone").min(0))


@pytest.mark.parametrize("kind", KINDS)
def test_every_hull_is_finite_and_three_dimensional(kind):
    v = hull_for(kind)
    assert v.ndim == 2 and v.shape[1] == 3
    assert np.isfinite(v).all()
    assert len(v) >= 12, "too few points to be a hull of anything"
    assert (v.max(0) - v.min(0) > 1e-6).all(), "%s is flat on some axis" % kind
