"""Detection difficulty: the label, the nesting, and the seven factors.

The label is what a benchmark reports against, so the properties that make it
usable are worth pinning harder than the thresholds themselves -- thresholds
are a judgement call and are documented as one, while "the sets nest" and "the
worst factor wins" are what make three numbers comparable at all.
"""
import numpy as np
import pytest

from physloc.annotate import difficulty as D


def _meta(**over):
    """A clip that is easy on every factor, so a test can spoil exactly one."""
    meta = {
        "num_frames": 100,
        "resolution": [512, 512],
        "n_actors": 1,
        "n_distractors": 0,
        "n_culprits": 1,
        "physics_medium": "rigid",
        "camera": {"extrinsics_per_frame": [
            {"position": [0.0, -6.0, 2.0], "look_at": [0.0, 0.0, 0.0]}] * 100},
        "violation": {
            "causal_body_ids": [2],
            "violation_windows": [[10, 89]],
            "observable_windows": [[10, 89]],
            "peak_residual": {"score": 1.0},
            "difficulty_inputs": {"footprint": 0.2, "occlusion": 0.0},
        },
    }
    meta.update(over)
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

    if name == "footprint":
        meta["violation"]["difficulty_inputs"]["footprint"] = bad
    elif name == "occlusion":
        meta["violation"]["difficulty_inputs"]["occlusion"] = min(1.0, bad)
    elif name == "duration":
        meta["violation"]["observable_windows"] = [[10, 11]]
    elif name == "severity":
        meta["violation"]["peak_residual"] = {"score": bad}
    elif name == "clutter":
        meta["n_actors"] = int(bad)
    elif name == "culprits":
        meta["n_actors"] = meta["n_culprits"] = int(bad)
    elif name == "camera":
        ext = [{"position": [0.0, -6.0 + 0.1 * i, 2.0],
                "look_at": [0.0, 0.0, 0.0]} for i in range(100)]
        meta["camera"] = {"extrinsics_per_frame": ext}

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
    meta["violation"]["difficulty_inputs"]["footprint"] = 0.0001
    got = D.assess(meta)
    assert got["level"] == "hard"
    assert got["binding_factors"] == ["footprint"]


def test_the_sets_nest():
    """easy is a subset of moderate is a subset of hard.

    What makes "AP at moderate" mean the same thing here as in every other
    benchmark that reports three numbers.
    """
    metas = []
    for foot in (0.3, 0.02, 0.001):
        m = _meta()
        m["violation"]["difficulty_inputs"]["footprint"] = foot
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

    `footprint` and `occlusion` come from arrays. If a consumer hands us a
    `meta.json` written before those were stored, the honest answer is the
    middle -- not `easy`, which would quietly move clips into the strictest
    evaluation set.
    """
    meta = _meta()
    meta["violation"].pop("difficulty_inputs")
    got = D.assess(meta)
    assert got["factors"]["footprint"]["level"] == "moderate"
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
    meta = _meta(physics_medium="granular", n_actors=80, n_culprits=80)
    got = D.assess(meta)
    assert got["factors"]["clutter"]["value"] == 1
    assert got["factors"]["culprits"]["value"] == 1
    assert got["level"] == "easy"

    # A genuine SUBSET keeps its count: then the question really is "which".
    part = D.assess(_meta(physics_medium="granular", n_actors=80,
                          n_culprits=48))
    assert part["factors"]["culprits"]["value"] == 48
    assert part["factors"]["culprits"]["level"] == "hard"


def test_camera_travel_is_path_length_over_standoff():
    """A ratio, so 0.15 means the same thing on `pour`'s 0.68 m box as on a
    `toss` several metres across. Path length, not displacement: an orbit that
    returns near its start still moved the whole way."""
    assert D.camera_travel(None) == 0.0
    assert D.camera_travel({"extrinsics_per_frame": []}) == 0.0
    # Ten metres of travel at a ten-metre standoff.
    ext = [{"position": [0.0, -10.0, 0.0], "look_at": [0.0, 0.0, 0.0]},
           {"position": [10.0, -10.0, 0.0], "look_at": [10.0, 0.0, 0.0]}]
    assert D.camera_travel({"extrinsics_per_frame": ext}) == pytest.approx(1.0)


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
    assert got["footprint"] == pytest.approx(0.25)
    assert got["occlusion"] == pytest.approx(0.5)

    stored = D.inputs_for_meta(vmask, seg, meta)
    meta["violation"]["difficulty_inputs"] = stored
    again = D.measure(meta)
    assert again["footprint"] == pytest.approx(got["footprint"])
    assert again["occlusion"] == pytest.approx(got["occlusion"])

