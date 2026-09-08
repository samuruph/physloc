"""The README's tables must match the code they describe.

Counts live in `physloc/taxonomy.py` and `physloc/scenarios/base.py` and
nowhere else -- prose copies of them have drifted five separate ways, which is
the whole reason `physloc/reference.py` exists. This is the tripwire: add a
family, move a share, build a rung, and the README goes stale until someone
runs one command.

    python -m physloc.reference --write
"""
import os
import re

import pytest

from physloc import reference

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
README = os.path.join(HERE, "README.md")


def test_the_readme_tables_are_current():
    text = open(README).read()
    assert reference.splice(text) == text, (
        "README.md is stale -- run `python -m physloc.reference --write`")


@pytest.mark.parametrize("name", sorted(reference.BLOCKS))
def test_every_block_is_present_and_filled(name):
    """A marker that lost its content, or a block nobody spliced in, would let
    the test above pass while the README said nothing."""
    text = open(README).read()
    pat = re.compile(r"<!-- physloc:%s -->\n(.*?)\n<!-- /physloc:%s -->"
                     % (name, name), re.S)
    m = pat.search(text)
    assert m, "README has no `%s` block" % name
    body = m.group(1)
    assert body.count("\n") >= 3, "%s block looks empty" % name
    assert body.startswith("| "), "%s block is not a table" % name


@pytest.mark.parametrize("name", sorted(reference.BLOCKS))
def test_a_block_renders_without_hand_written_numbers(name):
    """Every row comes from a live import, so a block that raises is a block
    describing something that no longer exists."""
    out = reference.render(name)
    assert out.startswith("| ") and out.count("\n") >= 3, name


def test_the_ladder_block_reports_what_is_actually_built():
    from physloc.scenarios.base import COMPLEXITY

    out = reference.render("ladder")
    for name, cx in COMPLEXITY.items():
        row = next(l for l in out.splitlines() if l.startswith("| **%s**" % name))
        assert ("**not built**" in row) != bool(cx.implemented), (
            "%s: the table and `implemented` disagree" % name)


def test_the_condition_block_sums_to_one():
    from physloc.scenarios.base import CONDITION_CYCLE

    out = reference.render("conditions")
    shares = [int(x) for x in re.findall(r"\| (\d+)% \|", out)]
    assert shares, out
    assert sum(shares) == 100, "conditions do not partition the clips: %s" % shares
    assert len(shares) == len(set(CONDITION_CYCLE))
