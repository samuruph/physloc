"""Load a PhysLoc dataset, print what is in it, and look at it.

    python test_dataset_loader.py out/physloc_mini                 # structure and shapes
    python test_dataset_loader.py out/physloc_mini --gui           # browser viewer, port 8765
    python test_dataset_loader.py out/physloc_mini --render 3 \
        --layers violation,reference,bbox3d --panels rgb,valid,segmentation,camera

The loader (`physloc/loader.py`) and the renderer (`physloc/viz/overlay.py`) do
the work; this file only shows how to use them.
"""
import argparse
from collections import Counter

from physloc.loader import PhysLocDataset, collate
from physloc.viz import overlay, video


def summarize(ds: PhysLocDataset) -> None:
    print("%d clips in %d pairs under %s" % (len(ds), len(ds.pairs()), ds.root))
    for field in ("level", "scenario", "family", "condition", "label"):
        counts = Counter(ds.fields(i)[field] or "-" for i in range(len(ds)))
        print("  %-9s %s" % (field, ", ".join("%s %d" % kv for kv in sorted(counts.items()))))

    print("\nclips (index for --render):")
    for i, clip in enumerate(ds.clips[:30]):
        print("  %3d  %s" % (i, clip.uid))

    clip = ds.pairs()[0].invalids[0]
    print("\none invalid clip: %s   violators %s" % (clip.uid, clip.violator_ids))
    arrays = {"video": clip.video, "segmentations": clip.segmentations,
              "violation": clip.violation, "causal": clip.causal,
              "causal_source": clip.causal_source,
              "violation_mask": clip.violation_mask,
              "visible_violation": clip.visible_violation,
              "severity_map": clip.severity_map, "reference_mask": clip.reference_mask}
    arrays.update({"objects." + k: v for k, v in clip.objects.items()})
    arrays.update({"timeline." + k: v for k, v in clip.timeline.items()})
    arrays.update({"latent_grid." + k: v for k, v in clip.latent_grid().items()})
    arrays.update({"instances." + k: v for k, v in clip.instances.items()})
    for name, a in arrays.items():
        print("  %-28s %-8s %s" % (name, a.dtype, list(a.shape)))

    batch = collate([ds[i] for i in range(min(4, len(ds)))])
    print("\na collated batch of %d:" % len(batch["uid"]))
    for key, value in batch.items():
        for name, a in (value.items() if isinstance(value, dict) else [(key, value)]):
            if hasattr(a, "shape"):
                print("  %-28s %s" % (name if name == key else key + "." + name, list(a.shape)))


def render(ds: PhysLocDataset, which: str, layers, panels, out: str) -> None:
    clip = (ds.clips[int(which)] if which.isdigit()
            else next(c for c in ds.clips if c.uid.endswith(which)))
    frames = overlay.Renderer(clip, layers, panels).render()
    video.write(frames, out, fps=clip.fps)
    print("wrote %s  (%s)" % (out, clip.uid))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="a generated release (has clips/) or an exported one (has shards/)")
    ap.add_argument("--gui", action="store_true", help="serve the browser viewer")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--render", metavar="CLIP", help="clip index or uid suffix to render to mp4")
    ap.add_argument("--layers", default="violation,reference",
                    help="drawn on RGB: " + ",".join(overlay.LAYERS))
    ap.add_argument("--panels", default=",".join(overlay.DEFAULT_PANELS),
                    help="side by side: " + ",".join(overlay.PANELS))
    ap.add_argument("--out", default="render.mp4")
    a = ap.parse_args()

    ds = PhysLocDataset(a.root)
    if a.gui:
        from physloc.viz import gui
        gui.serve(a.root, a.port)
    elif a.render is not None:
        split = lambda s: [x for x in s.split(",") if x]
        render(ds, a.render, split(a.layers), split(a.panels), a.out)
    else:
        summarize(ds)


if __name__ == "__main__":
    main()
