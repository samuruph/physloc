"""A declined cell is retried as the SAME KIND OF CLIP, on a fresh seed.

A family may refuse a scene without the cell being wrong -- `angular_momentum`
declines a sphere -- so `generate` offers the cell another sample. This pins
what that retry is allowed to change.

The scheduler is driven with the container stubbed out: `_run_worker` returns
the shape a real one would, so the whole job-building path runs on the host in
milliseconds. It caught the retry job being built with four fields where
`run_one` unpacks six, which no unit test of the pieces could have seen.
"""
from __future__ import annotations

import contextlib
import io

from physloc import cli

BASE, VARIANTS = 777, 10
DECLINER = ("toss", "angular_momentum")


def _run(monkeypatch, argv, declines_while):
    """Drive `cmd_generate`, returning every worker call it made.

    `declines_while(offset)` says whether `DECLINER` refuses this sample, where
    the offset is the seed's position inside its level's block.
    """
    seen = []

    def fake_worker(scenario, seed, tier, family, severity, workdir,
                    complexity="L0", window=None, dials=None, variant=0,
                    n_variants=None, params_path=None):
        off = seed % cli.LEVEL_SEED_STRIDE - BASE
        seen.append({"level": complexity, "variant": variant, "off": off,
                     "scenario": scenario, "seed": seed, "n_v": n_variants,
                     "families": tuple(family.split(","))})
        out = []
        for f in family.split(","):
            if (scenario, f) == DECLINER and declines_while(off):
                out.append({"family": f, "ok": False, "error": cli.NO_PLAN})
            else:
                out.append({"family": f, "ok": True, "dir": "/dev/null"})
        return 0, {"outdir": workdir, "variants": out}

    monkeypatch.setattr(cli, "_run_worker", fake_worker)
    monkeypatch.setattr(cli, "_annotate", lambda *a, **k: iter(()))
    ap, _ = cli._build()
    a = ap.parse_args(["generate", "--scenario", "toss,drop",
                       "--variants", str(VARIANTS), "--seed", str(BASE),
                       "--workdir", "w", "--outdir", "r",
                       "--no-overlay", "--keep-going"] + argv)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        a.fn(a)
    return seen, buf.getvalue()


def test_retry_keeps_the_level_and_variant_it_declined_on(monkeypatch):
    """The retried clip differs from the declining one IN ITS SEED ONLY.

    The variant index picks the condition (`condition_for` reads it with the
    level's variant count), so renumbering it would hand the rebuilt cell a
    different camera treatment or a different amount of clutter than the clip
    it replaces -- and the level says which seed block, which materials and
    which backdrop.
    """
    seen, _ = _run(monkeypatch, ["--complexity", "all"],
                   lambda off: off < VARIANTS)
    main = [s for s in seen if s["off"] < VARIANTS]
    retries = [s for s in seen if s["off"] >= VARIANTS]
    assert retries, "the declined cell was never retried"
    # Every retry is a toss clip -- `drop` declined nothing -- carrying only
    # the family that declined, at a level and variant the main pass ran.
    ran = {(s["level"], s["variant"], s["scenario"]) for s in main}
    for s in retries:
        assert s["scenario"] == DECLINER[0]
        assert s["families"] == (DECLINER[1],)
        assert (s["level"], s["variant"], s["scenario"]) in ran
        assert s["n_v"] == next(m["n_v"] for m in main
                                if m["level"] == s["level"])
    # One retry per declining clip, not one per scenario: the same cell can
    # decline at several levels and several variants of one level.
    assert len(retries) == sum(1 for s in main if s["scenario"] == DECLINER[0])


def test_retry_seeds_never_collide(monkeypatch):
    """A clip is written to `<level>/<scenario>/<seed>_<condition>/`.

    So a retry seed must be distinct per variant as well as per level and per
    attempt -- several variants of one level carry the same condition, and one
    seed for the whole level would have had the `standard` retries overwrite
    each other in turn.
    """
    seen, text = _run(monkeypatch, ["--complexity", "all"],
                      lambda off: off < VARIANTS * 2)   # declines twice
    assert text.count("retrying") >= 2, "the second attempt never ran"
    keys = [(s["level"], s["scenario"], s["seed"]) for s in seen]
    assert len(keys) == len(set(keys)), "two clips share an output directory"


def test_retry_stops_once_the_cell_lands(monkeypatch):
    """A cell that builds on its retry is not offered a third seed."""
    _, text = _run(monkeypatch, ["--complexity", "L0"],
                   lambda off: off < VARIANTS)
    assert text.count("retrying") == 1
