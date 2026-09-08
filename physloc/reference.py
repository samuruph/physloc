"""The taxonomy, the ladder and the conditions, rendered as Markdown tables.

ONE SOURCE, TWO READERS. The README explains the dataset to whoever downloads
it and the HuggingFace card explains it to whoever finds it there, and both
need the same tables. Written by hand they would be two copies of numbers that
already live in `taxonomy.py` and `scenarios/base.py` -- and prose copies of
these tables have drifted five separate ways before, which is why CLAUDE.md
says counts live in one place and nowhere else.

So they are generated here, and `tests/test_reference.py` fails if the README's
copy is stale. Adding a family or moving a share updates both documents by
re-running one command:

    python -m physloc.reference --write
"""
from __future__ import annotations

import collections
from typing import List


def _table(head: List[str], rows: List[List[str]]) -> str:
    out = ["| " + " | ".join(head) + " |",
           "|" + "|".join("---" for _ in head) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def media() -> str:
    from .taxonomy import MEDIA, SCENARIOS
    live = collections.Counter(s.physics_medium for s in SCENARIOS.values())
    rows = [[("**%s**" % k), v, live.get(k, 0) or "*not in v0*"]
            for k, v in MEDIA.items()]
    return _table(["medium", "what it means", "scenarios"], rows)


def domains() -> str:
    from .taxonomy import DOMAINS, FAMILIES, build_cells
    by_dom = collections.defaultdict(list)
    for name, fam in FAMILIES.items():
        by_dom[fam.domain].append(name)
    cells = collections.Counter()
    for scen, fam in build_cells():
        cells[FAMILIES[fam].domain] += 1
    rows = [["**%s**" % d, q, len(by_dom[d]), cells[d],
             ", ".join("`%s`" % f for f in sorted(by_dom[d]))]
            for d, q in DOMAINS.items()]
    return _table(["domain", "the question it asks", "n", "cells",
                   "families"], rows)


def scenarios() -> str:
    from .taxonomy import SCENARIOS, UNBUILT, build_cells
    cells = collections.Counter(s for s, _ in build_cells())
    rows = []
    for name, s in sorted(SCENARIOS.items()):
        if name in UNBUILT:
            continue
        rows.append(["`%s`" % name, s.physics_medium, cells[name],
                     s.description])
    return _table(["scenario", "medium", "families", "what happens"], rows)


def ladder() -> str:
    from .scenarios.base import COMPLEXITY
    total = sum(c.share for c in COMPLEXITY.values())
    adds = {"L0": "*the baseline*", "L1": "**materials** — wood, steel, rubber",
            "L2": "**an HDRI environment**",
            "L3": "**GSO objects** — real 3D scans"}
    rows = []
    for k, c in COMPLEXITY.items():
        rows.append(["**%s**" % k, adds.get(k, ""),
                     c.background, c.actor_assets,
                     "yes" if c.materials else "one shared density",
                     "%d%%" % round(100 * c.share / total),
                     "yes" if c.implemented else "**not built**"])
    return _table(["level", "adds", "background", "objects", "materials",
                   "share", "built"], rows)


def conditions() -> str:
    from .scenarios.base import CONDITION_CYCLE, EXTRA_OBJECTS
    order, seen = [], set()
    for c in CONDITION_CYCLE:
        if c not in seen:
            order.append(c)
            seen.add(c)
    n_extra = "**%d–%d**" % EXTRA_OBJECTS
    what = {
        "standard": ("static", "—", "**1**"),
        "camera": ("**moves**", "—", "**1**"),
        "distractors": ("static", n_extra, "**1**"),
        "multi": ("static", n_extra, "**2 … N−1**"),
        "camera+multi": ("**moves**", n_extra, "**2 … N−1**"),
    }
    n = len(CONDITION_CYCLE)
    rows = []
    for c in order:
        cam, extra, culp = what.get(c, ("", "", ""))
        k = CONDITION_CYCLE.count(c)
        rows.append(["`%s`" % c, "%d%%" % round(100 * k / n), cam, extra, culp])
    return _table(["condition", "share", "camera", "extra objects",
                   "objects with invalid physics"], rows)


def materials() -> str:
    from .scenarios.materials import MATERIALS
    from .scenarios.materials import ACTOR_MATERIALS
    total = sum(MATERIALS[n].weight for n in ACTOR_MATERIALS)
    rows = []
    for k, v in sorted(MATERIALS.items(), key=lambda kv: kv[1].density):
        look = []
        if v.metallic:
            look.append("**metal**")
        if v.transmission:
            look.append("**transmissive**, ior %.2f" % v.ior)
        look.append("rough %.2f" % v.roughness)
        look.append("spec %.2f" % v.specular)
        share = ("%.0f%%" % (100 * v.weight / total)) if k in ACTOR_MATERIALS else "—"
        rows.append(["`%s`" % k, "%.0f" % v.density, share, ", ".join(look)])
    return _table(["material", "density kg/m³", "share", "surface"], rows)


def tiers() -> str:
    from .scenarios import TIERS
    rows = []
    for k, t in TIERS.items():
        rows.append(["`%s`" % k, "%d²" % t.resolution,
                     "%d @ %d fps" % (t.num_frames, t.fps),
                     "%.2f s" % (t.num_frames / t.fps),
                     t.samples_per_pixel,
                     "%d×%d×%d" % (t.latent_frames, t.latent_hw, t.latent_hw)])
    return _table(["tier", "resolution", "frames", "duration", "spp",
                   "latent grid"], rows)


#: Every block, by the marker that delimits it in a document. A generated
#: section is wrapped in `<!-- physloc:NAME -->` ... `<!-- /physloc:NAME -->`,
#: so the surrounding prose is written by hand and only the tables are
#: replaced.
BLOCKS = {"media": media, "domains": domains, "scenarios": scenarios,
          "ladder": ladder, "conditions": conditions,
          "materials": materials, "tiers": tiers}


def render(name: str) -> str:
    return BLOCKS[name]()


def splice(text: str) -> str:
    """Replace every generated block in `text`, leaving the prose alone."""
    import re

    for name, fn in BLOCKS.items():
        # `(.*?)` with DOTALL, so it matches an empty block and a filled one
        # alike. Both narrower forms have now failed, in opposite directions
        # and both silently:
        #
        #   `\n.*?\n`  needed content, so a freshly written README with bare
        #              markers spliced nothing and shipped empty tables;
        #   `\s*?`     matched only whitespace, so once a block was filled it
        #              could never be updated again -- `--write` did nothing
        #              and `--current` reported CURRENT because there was no
        #              match to compare.
        #
        # Neither was caught by comparing `splice(text)` to `text`: when
        # nothing matches, that comparison is trivially equal. The test now
        # checks each block's CONTENT against `render(name)` instead, which is
        # the question actually being asked.
        pat = re.compile(
            r"(<!-- physloc:%s -->)(.*?)(<!-- /physloc:%s -->)" % (name, name),
            re.S)
        if pat.search(text):
            text = pat.sub(
                lambda m, f=fn: "%s\n%s\n%s" % (m.group(1), f(), m.group(3)),
                text)
    return text


def block_in(text: str, name: str):
    """The content currently between `name`'s markers, or None if absent."""
    import re

    m = re.search(r"<!-- physloc:%s -->\n?(.*?)\n?<!-- /physloc:%s -->"
                  % (name, name), text, re.S)
    return None if m is None else m.group(1)


def main(argv=None) -> int:
    import argparse
    import os

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true",
                    help="rewrite the generated blocks in README.md in place")
    ap.add_argument("--block", choices=sorted(BLOCKS),
                    help="print one block and exit")
    a = ap.parse_args(argv)
    if a.block:
        print(render(a.block))
        return 0
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(here, "README.md")
    text = open(path).read()
    out = splice(text)
    if not a.write:
        print("README is %s" % ("CURRENT" if out == text else "STALE"))
        return 0 if out == text else 1
    open(path, "w").write(out)
    print("wrote %s" % path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
