"""Generated output lands under `out/`, never in the repository root.

`generate --outdir r --workdir w` used to create `r/` and `w/` beside the source
code. `_under_out` places a relative output path under `out/`; `generate` and
`config-path` both apply it, so `scripts/run.sh` validates and packages the same
directory it generated into.
"""
import os

from physloc import cli


def test_a_bare_name_goes_under_out():
    assert cli._under_out("r") == os.path.join("out", "r")
    assert cli._under_out("runs/v0") == os.path.join("out", "runs", "v0")


def test_a_path_already_under_out_is_unchanged():
    assert cli._under_out("out/physloc_v0") == os.path.join("out", "physloc_v0")
    assert cli._under_out("./out/work_v0") == os.path.join("out", "work_v0")
    assert cli._under_out("out") == "out"


def test_absolute_and_escaping_paths_are_left_as_given(tmp_path):
    assert cli._under_out(str(tmp_path)) == str(tmp_path)
    assert cli._under_out("../elsewhere") == os.path.join("..", "elsewhere")


def test_no_path_means_the_default():
    assert cli._under_out(None) is None
    assert cli._under_out("") == ""


def test_config_path_resolves_like_generate(capsys):
    assert cli.main(["config-path", "--outdir", "r"]) == 0
    assert capsys.readouterr().out.strip() == os.path.join("out", "r")
