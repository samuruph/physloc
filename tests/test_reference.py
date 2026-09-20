"""The generated tables must match the code they describe.

Counts live in `physloc/taxonomy.py` and `physloc/scenarios/base.py` and
nowhere else -- prose copies of them have drifted five separate ways, which is
the whole reason `physloc/reference.py` exists. This is the tripwire: add a
family, move a share, build a level, and the documents go stale until someone
runs one command.

    python -m physloc.reference --write

The blocks live in more than one document -- the README carries what a reader
of the dataset needs, `docs/generating.md` what a run costs -- so every check
here is parametrized over `reference.BLOCK_FILES` rather than one path.
"""
import os
import re

import pytest

from physloc import reference

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FILES = sorted(set(reference.BLOCK_FILES.values()))


def _read(name):
    return open(os.path.join(HERE, name)).read()


@pytest.mark.parametrize("path", FILES)
def test_the_generated_tables_are_current(path):
    text = _read(path)
    assert reference.splice(text, reference.blocks_in_file(path)) == text, (
        "%s is stale -- run `python -m physloc.reference --write`" % path)


@pytest.mark.parametrize("name", sorted(reference.BLOCKS))
def test_every_block_is_present_and_filled(name):
    """A marker that lost its content, or a block nobody spliced in, would let
    the test above pass while the document said nothing."""
    path = reference.BLOCK_FILES[name]
    text = _read(path)
    pat = re.compile(r"<!-- physloc:%s -->\n(.*?)\n<!-- /physloc:%s -->"
                     % (name, name), re.S)
    m = pat.search(text)
    assert m, "%s has no `%s` block" % (path, name)
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


def test_every_block_has_exactly_one_home():
    """A block missing from BLOCK_FILES would be spliced nowhere and checked
    nowhere, which is the silent-nothing failure this module exists to stop."""
    assert set(reference.BLOCK_FILES) == set(reference.BLOCKS)
    for path in FILES:
        assert os.path.exists(os.path.join(HERE, path)), path


def test_the_hub_card_carries_the_generated_tables_and_no_cost_blocks():
    """The card used to be a hand-written stub naming none of the taxonomy. It
    is built from `render()` now, and must not price a config on the way."""
    card = reference.hf_card("CC-BY-4.0", 4, (("main", 0.75), ("held_out", 0.25)))
    assert card.startswith("---\nlicense: cc-by-4.0\n")
    for name in reference.CARD_BLOCKS:
        assert reference.render(name) in card, name
    for name in ("costs", "costs_ladder", "tiers"):
        assert name not in reference.CARD_BLOCKS
    assert "`main` 75%" in card and "`held_out` 25%" in card
