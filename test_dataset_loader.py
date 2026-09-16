"""Load a PhysLoc dataset THE WAY A USER GETS IT, print what is in it, look at it.

    python test_dataset_loader.py                                  # download samueleruf/physloc-mini
    python test_dataset_loader.py --repo samueleruf/physloc-review_L0
    python test_dataset_loader.py path/to/downloaded/physloc-mini  # a release already downloaded
    python test_dataset_loader.py --render 3 \
        --layers violation,reference,bbox3d --panels rgb,valid,segmentation,camera
    python test_dataset_loader.py --gui                            # browser viewer, port 8765

**What a user gets is the exported release**: `README.md`, `loader.py`,
`index.parquet`, `splits/` and `clips/`, the same clip folders a generator run
writes. So the default is to download it from the Hub, and the summary reads it
through the `loader.py`
SHIPPED INSIDE the download -- not `physloc/loader.py` from this repository --
so a stale or broken shipped loader fails here instead of on someone else's
machine. `data/` is gitignored, and a repeated download only fetches what
changed.

A generated run under `out/` is not what anyone downloads; `--generated` reads
one anyway, through this repository's loader.

`--render` and `--gui` use this repository's viewer (`physloc/viz`), which does
not ship with the data -- they read the same downloaded files.
"""
import argparse
import importlib.util
import os
from collections import Counter

DEFAULT_REPO = "samueleruf/physloc-mini"
CACHE = os.path.join("data", "hub")


def download(repo: str, cache: str = CACHE) -> str:
    """The release as the Hub serves it, in `<cache>/<owner>__<name>`."""
    from huggingface_hub import snapshot_download

    local = os.path.join(cache, repo.replace("/", "__"))
    return snapshot_download(repo_id=repo, repo_type="dataset", local_dir=local)


def shipped_loader(root: str):
    """Import the `loader.py` that came WITH the data."""
    path = os.path.join(root, "loader.py")
    if not os.path.exists(path):
        raise SystemExit(
            "%s has no loader.py, so it is not a downloaded PhysLoc release. "
            "Download one (the default, or --repo), export a run with "
            "`physloc export`, or pass --generated to read a generator run."
            % root)
    spec = importlib.util.spec_from_file_location("physloc_release_loader", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def summarize(loader, ds) -> None:
    print("%d clips in %d pairs under %s" % (len(ds), len(ds.pairs()), ds.root))
    print("loader: %s (schema %s)" % (loader.__file__, loader.SCHEMA_VERSION))
    for field in ("level", "scenario", "family", "condition", "label"):
        counts = Counter(ds.info(i)[field] or "-" for i in range(len(ds)))
        print("  %-9s %s" % (field, ", ".join("%s %d" % kv for kv in sorted(counts.items()))))
    splits = os.path.join(ds.root, "splits")
    if os.path.isdir(splits):
        sizes = {name[:-4]: sum(1 for line in open(os.path.join(splits, name)) if line.strip())
                 for name in sorted(os.listdir(splits)) if name.endswith(".txt")}
        print("  %-9s %s" % ("split", ", ".join("%s %d" % kv for kv in sizes.items())))

    print("\nclips (index for --render):")
    for i, clip in enumerate(ds.clips[:30]):
        print("  %3d  %s" % (i, clip.uid))

    pair = next(p for p in ds.pairs() if p.invalids)
    clip = pair.invalids[0]
    print("\none invalid clip: %s   violators %s   valid twin found: %s"
          % (clip.uid, clip.violator_ids, clip.twin is not None))
    arrays = {"video": clip.video, "segmentations": clip.segmentations,
              "violation": clip.violation, "causal": clip.causal,
              "causal_source": clip.causal_source,
              "violation_mask": clip.violation_mask,
              "visible_violation": clip.visible_violation,
              "severity_map": clip.severity_map}
    if clip.twin is not None:
        arrays["reference_mask"] = clip.reference_mask
    arrays.update({"objects." + k: v for k, v in clip.objects.items()})
    arrays.update({"timeline." + k: v for k, v in clip.timeline.items()})
    arrays.update({"latent_grid." + k: v for k, v in clip.latent_grid().items()})
    arrays.update({"instances." + k: v for k, v in clip.instances.items()})
    for name, a in arrays.items():
        print("  %-28s %-8s %s" % (name, a.dtype, list(a.shape)))

    batch = loader.collate([ds[i] for i in range(min(4, len(ds)))])
    print("\na collated batch of %d:" % len(batch["uid"]))
    for key, value in batch.items():
        for name, a in (value.items() if isinstance(value, dict) else [(key, value)]):
            if hasattr(a, "shape"):
                print("  %-28s %s" % (name if name == key else key + "." + name, list(a.shape)))

    print("\nfields= picks what an item carries; any of:\n  %s"
          % ", ".join(sorted(loader.FIELDS)))
    scenes = loader.PhysLocDataset(ds.root, unit="pair", fields=("video_path",))
    if len(scenes):
        item = scenes[0]
        print("\nunit=\"pair\": %d scenes; the first, %s, has its valid clip and "
              "%d invalid:" % (len(scenes), item["pair_uid"], len(item["invalid"])))
        for c in [item["valid"]] + item["invalid"][:3]:
            print("  %-8s %s" % (c["label"], c["video_path"]))


def render(root: str, which: str, layers, panels, out: str) -> None:
    from physloc.loader import PhysLocDataset
    from physloc.viz import overlay, video

    ds = PhysLocDataset(root)
    clip = (ds.clips[int(which)] if which.isdigit()
            else next(c for c in ds.clips if c.uid.endswith(which)))
    frames = overlay.Renderer(clip, layers, panels).render()
    video.write(frames, out, fps=clip.fps)
    print("wrote %s  (%s)" % (out, clip.uid))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", nargs="?",
                    help="a downloaded release folder; omit to download --repo")
    ap.add_argument("--repo", default=DEFAULT_REPO,
                    help="Hub dataset to download when no root is given")
    ap.add_argument("--cache", default=CACHE, help="where downloads go")
    ap.add_argument("--generated", action="store_true",
                    help="root is a generator run (has clips/), not a download")
    ap.add_argument("--gui", action="store_true", help="serve the browser viewer")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--render", metavar="CLIP", help="clip index or uid suffix to render to mp4")
    ap.add_argument("--layers", default="violation,reference",
                    help="drawn on RGB (see physloc/viz/overlay.py LAYERS)")
    ap.add_argument("--panels", default=None,
                    help="side by side (see physloc/viz/overlay.py PANELS)")
    ap.add_argument("--out", default="render.mp4")
    a = ap.parse_args()

    root = a.root or download(a.repo, a.cache)
    if (not a.generated and os.path.isdir(os.path.join(root, "clips"))
            and not os.path.exists(os.path.join(root, "loader.py"))):
        # A generator run: it has clips/ and no shipped loader. Say so and read
        # it, rather than refuse a command whose intent is unambiguous.
        print("%s is a generator run (clips/, no loader.py): reading it with "
              "this repository's loader, as --generated would." % root)
        a.generated = True
    if a.generated:
        from physloc import loader
    else:
        loader = shipped_loader(root)

    if a.gui:
        from physloc.viz import gui
        gui.serve(root, a.port)
    elif a.render is not None:
        from physloc.viz import overlay
        split = lambda s: [x for x in s.split(",") if x]
        panels = split(a.panels) if a.panels else list(overlay.DEFAULT_PANELS)
        render(root, a.render, split(a.layers), panels, a.out)
    else:
        summarize(loader, loader.PhysLocDataset(root))


if __name__ == "__main__":
    main()
