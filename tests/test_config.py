"""Config files fill in settings; typed flags always win."""
from __future__ import annotations

import os

import pytest

from physloc import cli, config


def _valid(subs, cmd):
    return {a.dest for a in subs[cmd]._actions} - {"help", "config"}


def _resolve(argv):
    """What `main` would run with, without running it."""
    ap, subs = cli._build()
    a = ap.parse_args(argv)
    typed = set(vars(cli._build(suppress=True)[0].parse_args(argv)))
    for key, value in config.load(getattr(a, "config", None), a.cmd,
                                  _valid(subs, a.cmd)).items():
        if key not in typed:
            setattr(a, key, value)
    return a


def test_shipped_configs_load_for_every_command():
    ap, subs = cli._build()
    root = os.path.join(cli.REPO, "configs")
    for name in sorted(os.listdir(root)):
        if not name.endswith(".yaml"):
            continue
        for cmd in subs:
            config.load(name[:-5], cmd, _valid(subs, cmd))   # must not raise


def test_config_fills_in_and_flags_override():
    """The MECHANISM, read against the file rather than against a copy of it.

    This used to hardcode `severity == "strong"`, so editing `review.yaml` --
    a policy choice, made twice since -- broke a test about whether config
    loading works at all. Reading the expected value out of the same file keeps
    it a test of the mechanism.
    """
    import yaml

    want = yaml.safe_load(
        open(os.path.join(cli.REPO, "configs", "review.yaml")))["defaults"]

    a = _resolve(["generate", "--config", "review"])
    assert (a.tier, a.severity, a.seed) == (
        want["tier"], want["severity"], want["seed"])
    assert a.keep_going is True          # from the file's `generate:` block

    b = _resolve(["generate", "--config", "review", "--seed", "42", "--tier", "release"])
    assert (b.seed, b.tier) == (42, "release")
    assert b.severity == want["severity"]        # still from the file


def test_a_typed_flag_that_equals_the_default_still_wins():
    """The reason this uses a SUPPRESS pass rather than comparing to defaults.

    `--variants 1` is also the argparse default, so a "did it differ from the
    default" test would call it untyped and let the config's 3 through.
    """
    a = _resolve(["generate", "--config", "v0_release", "--variants", "1"])
    assert a.variants == 1


def test_unknown_key_in_a_command_block_is_an_error(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("generate:\n  tierr: D\n")
    ap, subs = cli._build()
    with pytest.raises(config.ConfigError) as exc:
        config.load(str(p), "generate", _valid(subs, "generate"))
    assert "tierr" in str(exc.value)


def test_defaults_block_is_filtered_not_rejected(tmp_path):
    """`seed` means something to generate and nothing to taxonomy, and a shared
    block that could not hold it would not be worth having."""
    p = tmp_path / "shared.yaml"
    p.write_text("defaults:\n  seed: 5\n  complexity: L0\n")
    ap, subs = cli._build()
    assert config.load(str(p), "taxonomy", _valid(subs, "taxonomy")) == {
        "complexity": "L0"}
    assert config.load(str(p), "generate", _valid(subs, "generate")) == {
        "seed": 5, "complexity": "L0"}


def test_missing_config_is_an_error():
    ap, subs = cli._build()
    with pytest.raises(config.ConfigError):
        config.load("no_such_config", "generate", _valid(subs, "generate"))
