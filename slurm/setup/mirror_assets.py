"""Mirror the Kubric assets PhysLoc renders with, for a host with no internet.

The worker loads three asset sources -- KuBasic, GSO and HDRI Haven -- from
`gs://kubric-public/assets`. A cluster's compute nodes cannot reach it, so this
downloads what the code can actually ask for into a local folder and writes
each manifest with its `data_dir` pointed there. `PHYSLOC_ASSETS=<dest>` then
makes the worker read the mirror (`physloc/render/worker.py`).

    python slurm/setup/mirror_assets.py --dest $WORK/kubric_assets

Only the ids the code samples from are fetched: every KuBasic asset (a few,
small), the curated GSO list in `physloc/scenarios/_gso.py`, and the HDRI list
in `physloc/scenarios/_hdri.py`. Each mirrored manifest lists exactly those, so
an id outside the mirror fails as "unknown asset" rather than as a network
error on a node that has none.

Runs on the host (plain urllib, which honours the cluster's https_proxy) and is
resumable: a file is skipped when it is already there at its full size.
"""
import argparse
import concurrent.futures
import copy
import json
import os
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

BASE = "https://storage.googleapis.com/kubric-public/assets"
ATTEMPTS = 4


def wanted_ids():
    """{source name: the ids PhysLoc can ask for, or None for all of them}."""
    from physloc.scenarios._gso import GSO_IDS
    from physloc.scenarios._hdri import HDRI_IDS

    return {"KuBasic": None, "GSO": list(GSO_IDS), "HDRI_haven": list(HDRI_IDS)}


def asset_file(asset_id, entry):
    """The file an asset unpacks from, relative to `data_dir`, or None.

    Mirrors `kubric.AssetSource._resolve_asset_path` in the pinned image: no
    `path` key means `<id>.tar.gz`, and `path: null` means no file at all.
    """
    if "path" in entry and entry["path"] is None:
        return None
    return entry.get("path") or "%s.tar.gz" % asset_id


def rewrite(manifest, data_dir, keep):
    """The manifest a mirror ships: `data_dir` local, assets limited to `keep`.

    `keep` None keeps every asset. Entries themselves are untouched -- their
    `{asset_dir}` placeholders are resolved by Kubric against the unpacked
    file, wherever `data_dir` is.
    """
    out = copy.deepcopy(manifest)
    out["data_dir"] = str(data_dir)
    if keep is not None:
        missing = sorted(set(keep) - set(out["assets"]))
        if missing:
            raise KeyError("ids not in the %s manifest: %s"
                           % (out.get("name", "?"), ", ".join(missing[:10])))
        out["assets"] = {k: out["assets"][k] for k in keep}
    return out


def _fetch_json(url):
    with urllib.request.urlopen(url, timeout=60) as resp:
        return json.load(resp)


def _download(url, dest):
    """Fetch `url` to `dest` unless it is already there whole. Returns bytes."""
    for attempt in range(1, ATTEMPTS + 1):
        try:
            with urllib.request.urlopen(url, timeout=120) as resp:
                size = int(resp.headers.get("Content-Length") or -1)
                if size >= 0 and os.path.exists(dest) \
                        and os.path.getsize(dest) == size:
                    return 0
                tmp = dest + ".part"
                with open(tmp, "wb") as fh:
                    while True:
                        chunk = resp.read(1 << 20)
                        if not chunk:
                            break
                        fh.write(chunk)
            if size >= 0 and os.path.getsize(tmp) != size:
                raise IOError("short read: %d of %d bytes"
                              % (os.path.getsize(tmp), size))
            os.replace(tmp, dest)
            return os.path.getsize(dest)
        except Exception as exc:                           # noqa: BLE001
            if attempt == ATTEMPTS:
                raise RuntimeError("%s: %s" % (url, exc))
            time.sleep(2 ** attempt)
    return 0


def mirror(dest, jobs=8, only=None):
    dest = os.path.abspath(dest)
    total, failed = 0, []
    for name, keep in wanted_ids().items():
        if only and name not in only:
            continue
        folder = os.path.join(dest, name)
        os.makedirs(folder, exist_ok=True)
        manifest = _fetch_json("%s/%s/%s.json" % (BASE, name, name))
        local = rewrite(manifest, folder, keep)
        files = [f for f in (asset_file(k, v) for k, v in local["assets"].items())
                 if f is not None]
        print("%-11s %4d asset(s) -> %s" % (name, len(files), folder), flush=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
            futures = {pool.submit(_download, "%s/%s/%s" % (BASE, name, f),
                                   os.path.join(folder, f)): f for f in files}
            for fut in concurrent.futures.as_completed(futures):
                try:
                    total += fut.result()
                except Exception as exc:                   # noqa: BLE001
                    failed.append(str(exc))
        # The manifest last, and atomically: a mirror whose manifest exists is
        # one whose downloads were all attempted.
        path = os.path.join(folder, name + ".json")
        with open(path + ".tmp", "w") as fh:
            json.dump(local, fh)
        os.replace(path + ".tmp", path)
    print("downloaded %.1f GB" % (total / 1e9))
    if failed:
        print("%d download(s) FAILED -- re-run to retry:" % len(failed),
              file=sys.stderr)
        for line in failed[:20]:
            print("  " + line, file=sys.stderr)
        return 1
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dest", required=True, help="mirror root (PHYSLOC_ASSETS)")
    ap.add_argument("--jobs", type=int, default=8, help="parallel downloads")
    ap.add_argument("--only", help="comma list of sources (KuBasic,GSO,HDRI_haven)")
    a = ap.parse_args(argv)
    only = {x.strip() for x in a.only.split(",")} if a.only else None
    return mirror(a.dest, a.jobs, only)


if __name__ == "__main__":
    sys.exit(main())
