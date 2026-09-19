"""Detection difficulty: the label, the nesting, and the seven factors.

The label is what a benchmark reports against, so the properties that make it
usable are worth pinning harder than the thresholds themselves -- thresholds
are a judgement call and are documented as one, while "the sets nest" and "the
worst factor wins" are what make three numbers comparable at all.
"""
import numpy as np
import pytest

from physloc.annotate import difficulty as D


#: Top-level inputs to difficulty analysis; any other keyword to `_meta` is an
#: identity field and lands in the `metadata` block.
_BLOCKS = ("camera", "violation", "difficulty", "instances")


def _meta(**over):
    """A clip that is easy on every factor, so a test can spoil exactly one."""
    meta = {
        "metadata": {
            "num_frames": 100,
            "resolution": [512, 512],
            "n_actors": 1,
            "n_distractors": 0,
            "n_violators": 1,
            "physics_medium": "rigid",
        },
        "camera": {"positions": [[0.0, -6.0, 2.0]] * 100,
                   "look_at": [0.0, 0.0, 0.0]},
        "violation": {
            "causal_body_ids": [2],
            "violation_windows": [[10, 89]],
            "observable_windows": [[10, 89]],
            "peak_residual": {"score": 1.0},
            "difficulty_inputs": {"violation_area": 0.2, "occlusion": 0.0},
        },
    }
    for key, value in over.items():
        if key in _BLOCKS:
            meta[key] = value
        else:
            meta["metadata"][key] = value
    return meta


def test_an_easy_clip_is_easy_on_every_factor():
    got = D.assess(_meta())
    assert got["level"] == "easy"
    assert got["rank"] == 0
    assert all(f["level"] == "easy" for f in got["factors"].values())


@pytest.mark.parametrize("name", [f.name for f in D.FACTORS])
def test_any_single_factor_can_make_a_clip_hard(name):
    """The worst-factor rule, one axis at a time.

    Also a completeness check: a factor that CANNOT bind is dead weight in the
    table, and this fails if one is ever added without a way to reach `hard`.
    """
    meta = _meta()
    f = D.BY_NAME[name]
    # A value comfortably past the hard boundary, on the correct side.
    bad = f.moderate * 0.5 if f.easier == "high" else f.moderate + 10.0
    if f.easier == "high" and f.moderate == 0:
        bad = -1.0

    if name == "violation_area":
        meta["violation"]["difficulty_inputs"]["violation_area"] = bad
    elif name == "occlusion":
        meta["violation"]["difficulty_inputs"]["occlusion"] = min(1.0, bad)
    elif name == "duration":
        meta["violation"]["observable_windows"] = [[10, 11]]
    elif name == "severity":
        meta["violation"]["peak_residual"] = {"score": bad}
    elif name == "object_count":
        meta["metadata"]["n_actors"] = int(bad)
    elif name == "violators":
        meta["metadata"]["n_actors"] = meta["metadata"]["n_violators"] = int(bad)
    elif name == "camera_motion":
        meta["camera"] = {"positions": [[0.0, -6.0 + 0.1 * i, 2.0]
                                        for i in range(100)],
                          "look_at": [0.0, 0.0, 0.0]}

    got = D.assess(meta)
    assert got["level"] == "hard", (name, got["factors"][name])
    assert name in got["binding_factors"], (name, got["binding_factors"])


def test_the_worst_factor_wins_not_the_average():
    """Six easy factors do not rescue one hard one.

    The reason this is a max and not a mean: averaging says a tiny footprint
    and a static camera cancel, and they do not -- the clip is still one nobody
    can see.
    """
    meta = _meta()
    meta["violation"]["difficulty_inputs"]["violation_area"] = 0.0001
    got = D.assess(meta)
    assert got["level"] == "hard"
    assert got["binding_factors"] == ["violation_area"]


def test_the_sets_nest():
    """easy is a subset of moderate is a subset of hard.

    What makes "AP at moderate" mean the same thing here as in every other
    benchmark that reports three numbers.
    """
    metas = []
    for foot in (0.3, 0.005, 0.001):    # easy, moderate, hard by area
        m = _meta()
        m["violation"]["difficulty_inputs"]["violation_area"] = foot
        m["difficulty"] = D.assess(m)
        metas.append(m)
    sets = [D.evaluation_set(metas, lv) for lv in D.LEVELS]
    assert [len(s) for s in sets] == [1, 2, 3]
    for a, b in zip(sets, sets[1:]):
        assert all(m in b for m in a)


def test_a_valid_twin_has_no_difficulty():
    """It has no violation to detect, so it belongs to no evaluation set."""
    meta = _meta()
    meta["violation"] = None
    assert D.assess(meta) is None


def test_an_unmeasurable_factor_is_moderate_never_easy():
    """A missing value must not flatter a clip.

    `violation_area` and `occlusion` come from arrays. If either measurement is
    unavailable, the honest answer is the middle rather than `easy`.
    """
    meta = _meta()
    meta["violation"].pop("difficulty_inputs")
    got = D.assess(meta)
    assert got["factors"]["violation_area"]["level"] == "moderate"
    assert got["factors"]["occlusion"]["level"] == "moderate"
    assert got["level"] == "moderate"


def test_every_factor_declares_a_direction_and_ordered_thresholds():
    for f in D.FACTORS:
        assert f.easier in ("high", "low"), f.name
        if f.easier == "high":
            assert f.easy > f.moderate, (f.name, "easy must be the higher bar")
        else:
            assert f.easy < f.moderate, (f.name, "easy must be the lower bar")
        assert f.question.endswith("?"), f.name


def test_a_granular_medium_counts_as_one_thing():
    """`pour` has 80 grains and nobody is asked which grain is wrong.

    Left uncollapsed this pinned every granular clip to `hard` on both counts
    -- 63 of the review corpus's 431 clips -- for a reason that has nothing to
    do with how hard the violation is to see.
    """
    meta = _meta(physics_medium="granular", n_actors=80, n_violators=80)
    got = D.assess(meta)
    assert got["factors"]["object_count"]["value"] == 1
    assert got["factors"]["violators"]["value"] == 1
    assert got["level"] == "easy"

    # A genuine SUBSET keeps its count: then the question really is "which".
    part = D.assess(_meta(physics_medium="granular", n_actors=80,
                          n_violators=48))
    assert part["factors"]["violators"]["value"] == 48
    assert part["factors"]["violators"]["level"] == "hard"


def test_camera_travel_is_path_length_over_standoff():
    """A ratio, so 0.15 means the same thing on `pour`'s 0.68 m box as on a
    `toss` several metres across. Path length, not displacement: an orbit that
    returns near its start still moved the whole way."""
    assert D.camera_travel(None) == 0.0
    assert D.camera_travel({"positions": []}) == 0.0
    # Ten metres of travel at a ten-metre standoff: a chord of a circle of
    # radius ten around the fixed aim point.
    half = 5.0, -np.sqrt(75.0)
    cam = {"positions": [[-half[0], half[1], 0.0], [half[0], half[1], 0.0]],
           "look_at": [0.0, 0.0, 0.0]}
    assert D.camera_travel(cam) == pytest.approx(1.0)


def test_camera_travel_takes_the_arrays_annotate_passes():
    """`annotate` hands over the renderer's track as numpy arrays, and an
    array's truth value is an error -- `positions or []` failed every clip."""
    half = 5.0, -np.sqrt(75.0)
    cam = {"positions": np.asarray([[-half[0], half[1], 0.0],
                                    [half[0], half[1], 0.0]]),
           "look_at": np.zeros(3)}
    assert D.camera_travel(cam) == pytest.approx(1.0)


def test_footprint_and_occlusion_come_from_the_arrays_when_given():
    """And that the stored inputs reproduce them exactly.

    The whole point of `difficulty_inputs`: a re-derivation from `meta.json`
    alone must return what the clip was labelled with.
    """
    T, H, W = 20, 8, 8
    vmask = np.zeros((T, H, W), bool)
    vmask[5:15, :4, :4] = True                       # 16 of 64 px = 0.25
    seg = np.full((T, H, W), 1, np.uint16)
    seg[5:10] = 2                                    # visible for half the window
    meta = _meta(num_frames=T, resolution=[W, H])
    meta["violation"]["violation_windows"] = [[5, 14]]
    meta["violation"]["observable_windows"] = [[5, 14]]

    got = D.measure(meta, vmask, seg)
    assert got["violation_area"] == pytest.approx(0.25)
    assert got["occlusion"] == pytest.approx(0.5)

    stored = D.inputs_for_meta(vmask, seg, meta)
    meta["violation"]["difficulty_inputs"] = stored
    again = D.measure(meta)
    assert again["violation_area"] == pytest.approx(got["violation_area"])
    assert again["occlusion"] == pytest.approx(got["occlusion"])


def test_a_shadow_is_occluded_only_when_the_shadow_is_out_of_view():
    """A shadow's causal id is the renderer-only caster, which has no
    segmentation, so reading occlusion from segmentation called every shadow
    clip fully hidden. `seen` -- frames the shadow has pixels -- replaces it."""
    T, H, W = 20, 8, 8
    seg = np.full((T, H, W), 1, np.uint16)           # caster id 9 never appears
    meta = _meta(num_frames=T, resolution=[W, H])
    meta["violation"]["causal_body_ids"] = [9]
    meta["violation"]["violation_windows"] = [[5, 14]]
    assert D.measure(meta, None, seg)["occlusion"] == pytest.approx(1.0)
    seen = np.ones(T, bool)
    seen[5:8] = False                                # shadow gone 3 of 10 frames
    assert D.measure(meta, None, seg, seen)["occlusion"] == pytest.approx(0.3)
    assert D.inputs_for_meta(None, seg, meta, seen)["occlusion"] == pytest.approx(0.3)


def test_relabel_measures_a_shadow_from_its_isolation_pass(tmp_path):
    from sample_fixture import make_sample
    from physloc import loader

    pair = "t/L0/shadow_track/0001_standard"
    make_sample(tmp_path, pair + "/valid", "valid", pair_uid=pair, component=2)
    d = make_sample(tmp_path, pair + "/invalid_shadow_strong", pair_uid=pair,
                    component=2)
    assert D.shadow_seen(loader.Sample(d)).all()
    document = D.relabel(d)
    assert document["violation"]["difficulty"]["factors"]["occlusion"]["value"] == 0.0


def test_the_cuts_are_reachable_from_a_config():
    """`configs/common.yaml` owns the thresholds, and the whole table moves.

    Rebuilt as a table rather than field by field, so a factor cannot end up
    with one threshold from the config and the other from the shipped default
    -- which would silently invert `easier` on that factor and put the easy
    band on the wrong side of the moderate one.
    """
    from physloc import params

    meta = _meta(n_actors=4)
    try:
        params.apply()
        assert D.assess(meta)["factors"]["object_count"]["level"] == "moderate"

        params.apply({"difficulty": {"object_count": [4, 12]}})
        assert D.assess(meta)["factors"]["object_count"]["level"] == "easy"
        assert D.BY_NAME["object_count"].easy == 4
        # Untouched factors keep BOTH of their shipped values.
        assert (D.BY_NAME["severity"].easy,
                D.BY_NAME["severity"].moderate) == (0.90, 0.40)
    finally:
        params.apply()

    with pytest.raises(KeyError):
        params.resolved({"difficulty": {"nonesuch": [1, 2]}})


# --------------------------------------------------------- per-violator label
def _own(**over):
    """One violator's own measurements, easy on every factor."""
    values = {"violation_area": 0.2, "occlusion": 0.0, "duration": 0.9,
              "severity": 1.0, "camera_motion": 0.0}
    values.update(over)
    return values


def test_a_violator_is_scored_on_its_own_evidence():
    """The clip label describes the clip; this describes one body in it."""
    assert D.assess_violator(_own())["level"] == "easy"
    hidden = D.assess_violator(_own(occlusion=0.9))
    assert hidden["level"] == "hard"
    assert hidden["binding_factors"] == ["occlusion"]
    assert hidden["factors"]["duration"]["level"] == "easy"


def test_a_violator_label_does_not_count_the_others():
    """`object_count` and `violators` describe the scene, not a body in it --
    a `multi` clip's easy violator is still easy however many peers it has."""
    assert "object_count" not in D.VIOLATOR_FACTORS
    assert "violators" not in D.VIOLATOR_FACTORS
    assert set(D.VIOLATOR_FACTORS) == set(D.violator_values(0.1, 0.0, 0.5, 0.8, None))


def test_a_violator_and_its_clip_read_the_same_way():
    """Same cuts, same worst-factor rule, so the two labels are comparable."""
    for name in D.VIOLATOR_FACTORS:
        f = D.BY_NAME[name]
        bad = f.moderate * 0.5 if f.easier == "high" else f.moderate + 10.0
        got = D.assess_violator(_own(**{name: bad}))
        assert got["level"] == "hard", (name, got)
        assert name in got["binding_factors"]


def test_an_unmeasured_violator_factor_is_moderate():
    got = D.assess_violator(_own(violation_area=None))
    assert got["factors"]["violation_area"]["level"] == "moderate"
    assert got["level"] == "moderate"
