"""A clip becomes observable when one of its violators does -- not sooner.

`observable_frames` rejects anything under `min_pixels` changed pixels as
path-tracing noise. That floor is meant PER BODY, so measuring it once over a
whole granular medium divides it by however many grains share the clock: 96
grains a pixel or two apart from their twins sum to well over it while no grain
is anywhere near it.

The clip-level timeline is the union of what the violators each report, so the
two can never disagree. When they did, `schema.write` rejected the sample --
correctly, since the loader derives the clip's clocks from the violators' --
and an L2 `pour` took a 130-job sweep down with it.
"""
import numpy as np

from physloc.annotate.windows import observable_frames, to_windows

N_BODIES = 50
LEVEL = 8            # the intensity step `observable_frames` counts as changed
MIN_PIXELS = 4       # ...and the per-body floor it needs that many of


def _twins(changed_per_body):
    """Two renders of `N_BODIES`, each owning four pixels, differing on frame t
    by `changed_per_body[t]` pixels of each body and nothing else."""
    T, H, W = len(changed_per_body), 1, N_BODIES * 4
    seg = np.zeros((T, H, W), np.uint16)
    for b in range(N_BODIES):
        seg[:, 0, b * 4:(b + 1) * 4] = b + 1
    rgb_v = np.zeros((T, H, W, 3), np.uint8)
    rgb_i = rgb_v.copy()
    for t, n in enumerate(changed_per_body):
        for b in range(N_BODIES):
            rgb_i[t, 0, b * 4:b * 4 + n] = LEVEL + 2
    # The segmentation is identical in both twins: appearance is the only
    # evidence here, which is the `colour_shift` case that failed.
    return seg, seg.copy(), rgb_v, rgb_i


def test_the_aggregate_fires_before_any_single_body_does():
    """The bug, stated as a measurement: two changed pixels per body is noise
    to every body and evidence to the sum of them."""
    below, at = MIN_PIXELS - 2, MIN_PIXELS
    seg_v, seg_i, rgb_v, rgb_i = _twins([0, below, at])
    ids = list(range(1, N_BODIES + 1))

    aggregate = observable_frames(seg_v, seg_i, ids, rgb_valid=rgb_v,
                                  rgb_invalid=rgb_i)
    per_body = [observable_frames(seg_v, seg_i, [b], rgb_valid=rgb_v,
                                  rgb_invalid=rgb_i) for b in ids]

    assert aggregate[1], "%d bodies x %d px sums past the floor" % (N_BODIES, below)
    assert not any(o[1] for o in per_body), "no body is over the floor alone"
    assert all(o[2] for o in per_body), "every body is over it on frame 2"


def test_the_clip_takes_the_union_of_its_violators():
    """What the pipeline must compute, and what `schema.write` checks: the
    clip's first observable frame is the first of any violator's."""
    seg_v, seg_i, rgb_v, rgb_i = _twins([0, MIN_PIXELS - 2, MIN_PIXELS])
    ids = list(range(1, N_BODIES + 1))
    per_body = [observable_frames(seg_v, seg_i, [b], rgb_valid=rgb_v,
                                  rgb_invalid=rgb_i) for b in ids]

    union = np.zeros(seg_v.shape[0], bool)
    for own in per_body:
        union |= own

    first_clip = int(np.flatnonzero(union)[0])
    first_violator = min(int(np.flatnonzero(o)[0]) for o in per_body)
    assert first_clip == first_violator == 2
    # ...which is the comparison `_violation` makes before writing a sample.
    assert to_windows(union)[0][0] == first_violator


def test_one_violator_is_unaffected():
    """The union changes nothing where there is nothing to union: a clip with a
    single violator measures exactly what it always did."""
    seg_v, seg_i, rgb_v, rgb_i = _twins([0, MIN_PIXELS, MIN_PIXELS])
    one = observable_frames(seg_v, seg_i, [1], rgb_valid=rgb_v, rgb_invalid=rgb_i)
    assert list(np.flatnonzero(one)) == [1, 2]


def test_independent_clock_ignores_another_bodys_early_difference():
    from physloc.annotate.windows import violator_observable, rasterise, build
    from physloc.schema.write import document_from_generation
    from sample_fixture import make_meta

    # Body A is active but hidden from frame 1. B differs at frame 2 while
    # inactive and becomes observable at its own intervention on frame 3.
    T = 5
    clip_active = rasterise([(1, 4)], T)
    own = np.array([False, False, True, True, True])
    later_active = rasterise([(3, 4)], T)
    hidden = np.zeros(T, bool)
    evidence = violator_observable(own, later_active, clip_active,
                                   hidden, hidden, independent=True)
    assert to_windows(evidence) == [(3, 4)]
    seg = np.ones((T, 1, 1), np.uint16)
    timing = build([(1, 4)], T, seg, seg, [1], 1, observable=evidence)
    assert timing["t_observable_frame"] == 3
    meta = make_meta("test/p/invalid", frames=T, n_objects=3)
    v = meta["violation"]
    a = dict(v["violators"][0], instance_id=2, observable_windows=[])
    b = dict(v["violators"][0], instance_id=3, t_event_frame=3,
             t_observable_frame=3, observability_lag_frames=0,
             violation_windows=[[3, 4]], intervention_windows=[[3, 3]],
             consequence_windows=[[3, 4]],
             observable_windows=[list(w) for w in to_windows(evidence)])
    v.update(t_observable_frame=3, observability_lag_frames=2,
             t_intervention_end_frame=3, t_consequence_end_frame=4,
             violators=[a, b], violator_timing="independent")
    document_from_generation(meta)  # same consistency check that failed live


def test_carried_evidence_is_stored_in_the_object_clock():
    from physloc.annotate.windows import violator_observable, rasterise
    own = rasterise([(4, 4)], 6)
    visible = rasterise([(5, 5)], 6)
    active = rasterise([(1, 2)], 6)
    result = violator_observable(own, active, active, visible, visible, True)
    assert to_windows(result) == [(4, 5)]
    assert np.array_equal(violator_observable(
        own, active, active, visible, visible, False), own)
