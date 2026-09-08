"""Check `physloc/scenarios/_hdri.py` against the live HDRI Haven manifest.

The id list is baked into the repo so that sampling a `SceneSpec` works on the
host, with no container and no network -- but that means nothing verifies it,
and it drifted: the curated list carried ids the manifest does not have, and a
scenario that drew one failed at render time with a `KeyError` from deep inside
Kubric's asset source, hundreds of clips into a run.

    bash docker/kubric.sh scripts/refresh_hdri_ids.py            # report
    bash docker/kubric.sh scripts/refresh_hdri_ids.py --write    # fix the file

`--write` keeps every curated id that still exists and drops the rest, printing
what it removed. It never invents replacements: the list is curated for a
reason -- an indoor/outdoor spread -- and picking substitutes is a judgement
call, not a refresh.
"""
import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

HERE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "physloc", "scenarios", "_hdri.py")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true",
                    help="rewrite _hdri.py, dropping ids the manifest lacks")
    ap.add_argument("--all", action="store_true",
                    help="take EVERY id the manifest has, not just the curated "
                         "subset -- an environment map has no geometry to get "
                         "wrong, so breadth costs nothing")
    a = ap.parse_args()

    import kubric as kb

    from physloc.render.worker import HDRI
    from physloc.scenarios._hdri import HDRI_IDS

    src = kb.AssetSource.from_manifest(HDRI)
    live = set(getattr(src, "_assets", {}) or {})
    if not live:                       # older API shape
        live = set(src.db["id"]) if hasattr(src, "db") else set()
    print("manifest %s" % HDRI)
    print("live assets: %d" % len(live))
    print("curated ids: %d" % len(HDRI_IDS))

    bad = [i for i in HDRI_IDS if i not in live]
    good = [i for i in HDRI_IDS if i in live]
    if a.all:
        # EVERY environment the manifest has. The curated list was an
        # indoor/outdoor spread of 44, which is a reasonable shape and a poor
        # size: a release of thousands of clips reusing 36 backdrops teaches
        # the backdrops. There is no reason to curate at all here -- unlike a
        # GSO object, an environment map has no geometry to go wrong, so the
        # only question is whether it resolves.
        good = sorted(live)
        bad = []
        print("\n--all: taking every id the manifest has (%d)" % len(good))
    elif not bad:
        print("\nevery curated id resolves. Nothing to do.")
        return 0

    print("\n%d curated id(s) the manifest does NOT have:" % len(bad))
    for i in bad:
        near = sorted(x for x in live if i.split("_")[0] in x)[:3]
        print("   %-32s nearest: %s" % (i, ", ".join(near) or "-"))

    if not a.write:
        print("\nRe-run with --write to apply.")
        return 1

    body = ", ".join('"%s"' % i for i in good)
    wrapped, line = [], "    "
    for tok in body.split(", "):
        piece = tok + ", "
        if len(line) + len(piece) > 76:
            wrapped.append(line.rstrip())
            line = "    "
        line += piece
    wrapped.append(line.rstrip().rstrip(","))
    src_txt = open(HERE).read()
    new = re.sub(r"HDRI_IDS: List\[str\] = \[\n.*?\n\]",
                 "HDRI_IDS: List[str] = [\n%s\n]" % "\n".join(wrapped),
                 src_txt, flags=re.S)
    open(HERE, "w").write(new)
    print("\nwrote %s -- %d ids kept, %d dropped" % (HERE, len(good), len(bad)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
