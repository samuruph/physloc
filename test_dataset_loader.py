"""Load a PhysLoc dataset, print what is in it, and look at it.

    python test_dataset_loader.py                                  # download samueleruf/physloc-mini
    python test_dataset_loader.py --repo samueleruf/physloc-review_L0
    python test_dataset_loader.py path/to/physloc-dataset          # generated or exported
    python test_dataset_loader.py --render 3 \
        --layers violation,reference,bbox3d --panels rgb,valid,segmentation,camera
    python test_dataset_loader.py --gui                            # browser viewer, port 8765

Generated runs and exported releases have the same schema-v4 sample tree, so
this script uses one code path for both: `physloc.loader.PhysLocDataset(root)`.
When no root is given it downloads the default Hub dataset into `data/hub`.
"""
import argparse
import os
from collections import Counter

from physloc import loader

DEFAULT_REPO = "samueleruf/physloc-mini"
CACHE = os.path.join("data", "hub")


def download(repo: str, cache: str = CACHE) -> str:
    """The release as the Hub serves it, in `<cache>/<owner>__<name>`."""
    from huggingface_hub import snapshot_download

    local = os.path.join(cache, repo.replace("/", "__"))
    return snapshot_download(repo_id=repo, repo_type="dataset", local_dir=local)


def _shape(value) -> str:
    if hasattr(value, "shape"):
        return "%s %s" % (value.dtype, list(value.shape))
    if isinstance(value, dict):
        return "dict(%d)" % len(value)
    if isinstance(value, (list, tuple)):
        return "list(%d)" % len(value)
    return repr(value) if len(repr(value)) < 70 else repr(value)[:67] + "..."


def show(namespace, title: str, keys=None) -> None:
    """One namespace of a sample, a line per key: its value or its shape."""
    print("  %s" % title)
    for key in keys or namespace.keys():
        print("    %-22s %s" % (key, _shape(namespace[key])))


def summarize(ds) -> None:
    print("%d samples in %d pairs under %s" % (len(ds), len(ds.pairs()), ds.root))
    print("loader: %s (schema %s)" % (loader.__file__, loader.SCHEMA_VERSION))
    accessors = {
        "level": lambda s: s.scene.level,
        "scenario": lambda s: s.scene.scenario,
        "family": lambda s: s.scene.family,
        "condition": lambda s: s.scene.condition,
        "label": lambda s: s.info.label,
    }
    for field, get_value in accessors.items():
        counts = Counter(get_value(sample) or "-" for sample in ds.samples)
        print("  %-9s %s" % (field, ", ".join("%s %d" % kv for kv in sorted(counts.items()))))
    splits = os.path.join(ds.root, "splits")
    if os.path.isdir(splits):
        sizes = {name[:-4]: sum(1 for line in open(os.path.join(splits, name)) if line.strip())
                 for name in sorted(os.listdir(splits)) if name.endswith(".txt")}
        print("  %-9s %s" % ("split", ", ".join("%s %d" % kv for kv in sizes.items())))

    print("\nsamples (index for --render):")
    for i, sample in enumerate(ds.samples[:30]):
        print("  %3d  %s" % (i, sample.uid))

    pair = next(p for p in ds.pairs() if p.invalids)
    s = pair.invalids[0]
    print("\none invalid sample, namespace by namespace: %s  (valid twin found: %s)"
          % (s.uid, s.twin is not None))
    show(s.info, "s.info")
    show(s.video, "s.video", [k for k in s.video.keys() if k != "rgb"])
    show(s.scene, "s.scene")
    show(s.scene.camera, "s.scene.camera")
    show(s.observations, "s.observations")
    show(s.objects, "s.objects")
    show(s.violation, "s.violation")
    show(s.energy, "s.energy")
    show(s.events, "s.events")
    print("  s.violation.violators[0]")
    for key, value in s.violation.violators[0].items():
        print("    %-22s %s" % (key, _shape(value)))
    print("  s.violation.latent_grid()")
    for key, value in s.violation.latent_grid().items():
        print("    %-22s %s" % (key, _shape(value)))

    fields = ("video.rgb", "observations.segmentation", "violation.mask",
              "violation.severity_map", "violation.severity", "objects.ids")
    sub = loader.PhysLocDataset(ds.root, fields=fields)
    batch = loader.collate([sub[i] for i in range(min(4, len(sub)))])
    print("\na collated batch of %d, fields=%s:" % (len(batch["info"]["uid"]), list(fields)))

    def walk(value, path):
        if isinstance(value, dict):
            for key, item in value.items():
                walk(item, path + [key])
        elif hasattr(value, "shape"):
            print("  %-34s %s" % (".".join(path), _shape(value)))
    walk(batch, [])

    print("\nfields= picks what to_dict/collate carry; any of:\n  %s"
          % ", ".join(sorted(loader.FIELDS)))
    scenes = loader.PhysLocDataset(ds.root, unit="pair")
    if len(scenes):
        item = scenes[0]
        print("\nunit=\"pair\": %d scenes; the first, %s, has its valid sample and "
              "%d invalid:" % (len(scenes), item["pair_uid"], len(item["invalid"])))
        for c in [item["valid"]] + item["invalid"][:3]:
            print("  %-8s %s" % (c.info.label, c.video.path))


def render(root: str, which: str, layers, panels, out: str) -> None:
    from physloc.loader import PhysLocDataset
    from physloc.viz import overlay, video

    ds = PhysLocDataset(root)
    sample = (ds.samples[int(which)] if which.isdigit()
              else next(value for value in ds.samples if value.uid.endswith(which)))
    frames = overlay.Renderer(sample, layers, panels).render()
    video.write(frames, out, fps=sample.video.fps)
    print("wrote %s  (%s)" % (out, sample.uid))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", nargs="?",
                    help="dataset root; omit to download --repo")
    ap.add_argument("--repo", default=DEFAULT_REPO,
                    help="Hub dataset to download when no root is given")
    ap.add_argument("--cache", default=CACHE, help="where downloads go")
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

    if a.gui:
        from physloc.viz import gui
        gui.serve(root, a.port)
    elif a.render is not None:
        from physloc.viz import overlay
        split = lambda s: [x for x in s.split(",") if x]
        panels = split(a.panels) if a.panels else list(overlay.DEFAULT_PANELS)
        render(root, a.render, split(a.layers), panels, a.out)
    else:
        summarize(loader.PhysLocDataset(root))


if __name__ == "__main__":
    main()
