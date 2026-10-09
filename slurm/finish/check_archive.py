"""Check that a set of tar files holds exactly one release folder, nothing
missing and nothing extra. Run by archive.slurm right after it writes the tars;
exits 1 on any difference, so a bad archive fails the job instead of being
found on restore.

    python slurm/finish/check_archive.py <release folder> <tar> [<tar> ...]

Compared, for every entry under the release folder: its path, its type (file,
folder, link), a file's size and a link's target. Empty folders count too.
Standard library only -- it runs with any Python 3. Tar headers are read
without reading the data, so it is quick even on a release of a few TB.
"""
import argparse
import os
import sys
import tarfile


def tree(root):
    """{path relative to root's parent: ("file", size) | ("dir",) | ("link", target)}."""
    root = os.path.abspath(root)
    base = os.path.dirname(root)
    out = {}
    for here, dirs, files in os.walk(root):
        rel = os.path.relpath(here, base)
        out[rel] = ("dir",)
        for name in dirs + files:
            path = os.path.join(here, name)
            key = os.path.join(rel, name)
            if os.path.islink(path):
                out[key] = ("link", os.readlink(path))
            elif os.path.isfile(path):
                out[key] = ("file", os.path.getsize(path))
        # links to folders are listed (above) but not walked into: tar does the same
        dirs[:] = [d for d in dirs if not os.path.islink(os.path.join(here, d))]
    return out


def archived(tars):
    """The same mapping, read from the tar headers. Also returns entries that
    appear in more than one tar (harmless for folders, a mistake for files)."""
    out, twice = {}, []
    for path in tars:
        with tarfile.open(path, "r:") as tf:
            for m in tf:
                key = os.path.normpath(m.name)
                if m.isdir():
                    entry = ("dir",)
                elif m.issym():
                    entry = ("link", m.linkname)
                elif m.isfile():
                    entry = ("file", m.size)
                else:
                    entry = ("other",)
                if key in out and entry[0] != "dir":
                    twice.append(key)
                out[key] = entry
    return out, twice


def compare(root, tars):
    """A list of problems, empty when the tars hold exactly the release."""
    want = tree(root)
    have, twice = archived(tars)
    # A folder that only appears as the parent of another tar's contents is
    # still restored by extracting that tar, so missing parents are fine
    # as long as something below them is present.
    problems = []
    for key, entry in want.items():
        got = have.get(key)
        if got is None:
            if entry == ("dir",) and any(k.startswith(key + os.sep) for k in have):
                continue
            problems.append("missing from the tars: %s" % key)
        elif got != entry:
            problems.append("differs: %s  (folder %s, tar %s)" % (key, entry, got))
    for key in have:
        if key not in want:
            problems.append("in a tar but not in the folder: %s" % key)
    problems += ["in more than one tar: %s" % k for k in twice]
    return problems, sum(1 for e in want.values() if e[0] == "file")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("root", help="the release folder that was archived")
    ap.add_argument("tars", nargs="+", help="every tar written for it")
    a = ap.parse_args()
    problems, n_files = compare(a.root, a.tars)
    if problems:
        print("ARCHIVE DOES NOT MATCH %s (%d problem(s)):" % (a.root, len(problems)))
        for p in problems[:50]:
            print("  " + p)
        return 1
    print("archive OK: %d files, every path, size and link identical to %s"
          % (n_files, a.root))
    return 0


if __name__ == "__main__":
    sys.exit(main())
