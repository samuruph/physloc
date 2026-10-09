"""Swap one family's samples in a release for those of a refill run.

A refill (`slurm/refill/render_refill.slurm`) re-renders ONE family on jobs that
already ran, into its own folder, `$PHYSLOC_DATA/refill_<family>/<release>`.
This moves the result into the release, per scene slot -- a job's (level,
variant, scenario), which also covers the retry scenes it built on fresh seeds:

  1. the release's samples of that family on every scene of a refilled slot
     are REMOVED, so no old clip of the family survives beside new ones --
     a bin the fixed code declines must not keep its pre-fix clip;
  2. the refill's samples of that family are copied in (with their scene's
     valid twin, for a retry scene the release does not have yet);
  3. the refill's ledgers go to `.jobs/<name>__refill_<family>.json`, which
     `status.py` reads as overriding that family for their slot.

A slot whose refill job did not finish has no ledger and is left untouched.
Nothing is changed without --apply; then run `sbatch slurm/finish/finalize.slurm`.

    source slurm/env.sh
    python slurm/refill/merge_refill.py --family solidity            # what it would do
    python slurm/refill/merge_refill.py --family solidity --apply
"""
import argparse
import glob
import json
import os
import shutil
import sys


def ledgers(root):
    """{ledger name: request} under `root/.jobs`."""
    out = {}
    for path in sorted(glob.glob(os.path.join(root, ".jobs", "*.json"))):
        try:
            with open(path) as fh:
                out[os.path.basename(path)[:-len(".json")]] = json.load(fh)["request"]
        except (OSError, ValueError, KeyError):
            continue
    return out


def slot_of(req):
    return (str(req["level"]), int(req["variant"]), str(req["scenario"]))


def pair_dirs(root, release):
    """{(level, scenario, seed): pair dir} for every scene under `root`."""
    out = {}
    for path in glob.glob(os.path.join(root, "samples", release, "*", "*", "*")):
        level, scenario, name = path.split(os.sep)[-3:]
        seed = name.split("_", 1)[0]
        if seed.isdigit() and os.path.isdir(path):
            out[(level, scenario, int(seed))] = path
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--family", required=True)
    ap.add_argument("--release", default=os.environ.get("PHYSLOC_OUTDIR"),
                    help="the release folder (default $PHYSLOC_OUTDIR)")
    ap.add_argument("--refill", help="the refill folder (default "
                    "$PHYSLOC_DATA/refill_<family>/<release name>)")
    ap.add_argument("--apply", action="store_true", help="make the changes")
    a = ap.parse_args()
    if not a.release:
        sys.exit("no release folder: source slurm/env.sh, or pass --release")
    name = os.path.basename(os.path.normpath(a.release))
    refill = a.refill or os.path.join(os.environ.get("PHYSLOC_DATA", ""),
                                      "refill_" + a.family, name)
    prefix = "invalid_%s_" % a.family

    new = ledgers(refill)
    if not new:
        sys.exit("no finished refill jobs under %s/.jobs" % refill)
    slots = {slot_of(r) for r in new.values()}
    # Which slot each scene of either folder belongs to, from the requests.
    seed_slot = {(r["level"], r["scenario"], int(r["seed"])): slot_of(r)
                 for r in list(ledgers(a.release).values()) + list(new.values())}

    removed, added, new_scenes = [], [], []
    for key, pair in sorted(pair_dirs(a.release, name).items()):
        if seed_slot.get(key) in slots:
            removed += sorted(glob.glob(os.path.join(pair, prefix + "*")))
    have = pair_dirs(a.release, name)
    for key, pair in sorted(pair_dirs(refill, name).items()):
        samples = sorted(glob.glob(os.path.join(pair, prefix + "*")))
        if not samples:
            continue
        if key not in have:
            new_scenes.append(pair)
        else:
            added += samples

    print("refill   %s  (%d finished jobs, %d scene slots)" % (refill, len(new), len(slots)))
    print("release  %s" % a.release)
    print("  remove %4d old %s samples" % (len(removed), a.family))
    print("  copy   %4d new %s samples into existing scenes" % (len(added), a.family))
    print("  copy   %4d new scenes (retry seeds), with their valid twin" % len(new_scenes))
    if not a.apply:
        print("\nNothing changed. Re-run with --apply to do it.")
        return 0

    for path in removed:
        shutil.rmtree(path)
    for src in added:
        rel = os.path.relpath(src, refill)
        shutil.copytree(src, os.path.join(a.release, rel), dirs_exist_ok=True)
    for src in new_scenes:
        rel = os.path.relpath(src, refill)
        dst = os.path.join(a.release, rel)
        os.makedirs(dst, exist_ok=True)
        for item in sorted(os.listdir(src)):
            if item == "valid" or item.startswith(prefix):
                shutil.copytree(os.path.join(src, item), os.path.join(dst, item),
                                dirs_exist_ok=True)
    os.makedirs(os.path.join(a.release, ".jobs"), exist_ok=True)
    for led in sorted(new):
        shutil.copy2(os.path.join(refill, ".jobs", led + ".json"),
                     os.path.join(a.release, ".jobs",
                                  "%s__refill_%s.json" % (led, a.family)))
    print("\nDone. Check with  python slurm/check/status.py, then  sbatch slurm/finish/finalize.slurm")
    return 0


if __name__ == "__main__":
    sys.exit(main())
