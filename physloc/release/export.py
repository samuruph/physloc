"""Package a generated release for distribution -- WebDataset shards + a card.

The generator writes one directory per clip, twenty-odd files each. That is the
right shape for producing and inspecting clips and the wrong shape for handing
to anyone else: a thousand directories is slow to download, impossible to
stream, and tells a reader nothing about what is in it.

This turns that tree into what a dataset host expects:

    physloc_v0/
      README.md                     the dataset card, with YAML front-matter
      LICENSE
      index.parquet                 one row per clip, for the dataset viewer
      splits/{train,val,test}.txt   grouped by pair_uid
      shards/core-{000..NNN}.tar    rgb + annotations, the default download
      shards/passes-{000..NNN}.tar  depth/flow/coords, optional and much larger

**Two shard sets, and the split is the whole point.** Measured on a debug
sweep: `flow_fwd`, `depth` and `object_coords` are 86% of the bytes, at
~3.2 MB per clip against ~1 KB for each annotation. At v0 geometry the raw
passes are ~57x that. Someone training on `violation_mask` and `severity_map`
should not download a hundred gigabytes of optical flow to get them, and
someone who wants the flow should not have to guess whether it exists.

Splits group by `pair_uid`, never by clip. A valid twin and its invalid
siblings share a scene, a seed and a bit-identical prefix, so putting them on
opposite sides of a split leaks the answer: a model that saw the valid clip has
seen every frame before `t_event` of the invalid one.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import tarfile
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

#: Files that go in the core shards -- what a model trains on.
CORE_FILES = ("meta.json", "rgb.mp4", "violation_mask.npz", "mask_invalid.npz",
              "reference_mask.npz", "causal_mask.npz", "severity_map.npz",
              "timelines.npz", "residuals.npz", "seg.npz", "energy.npz",
              "bodies.npz", "traj.npz", "grids.npz")

#: Files that go in the optional shards -- the raw geometry passes.
PASS_FILES = ("depth.npz", "flow_fwd.npz", "flow_bwd.npz", "normals.npz",
              "object_coords.npz", "energy_map.npz", "divergence_map.npz")

#: Roughly how large a shard should get before starting another. 400 MB is the
#: usual WebDataset advice: big enough that sequential reads dominate, small
#: enough to retry cheaply on a bad connection.
SHARD_BYTES = 400 * 1024 * 1024

#: Fractions of PAIRS, not clips.
#:
#: Modelled on IntPhys 2 (arXiv:2506.09849), which this project already takes
#: its debug/artifact split from. It reports 1416 videos over three splits --
#: Debug (5 scenes, 60 videos, for calibration), Main (253 scenes, 1012 videos,
#: released WITH metadata) and Held-Out (86 scenes, 344 videos, released
#: WITHOUT metadata "to avoid training data contamination"). Counted in SCENES,
#: which is why the unit here is the pair.
#:
#: Not a random train/val/test, which is what this had first and what neither
#: prior art does. LikePhys (arXiv:2510.11512) does not split at all -- it is a
#: training-free evaluator doing pairwise valid-versus-invalid comparison -- and
#: PhysLoc's primary use is the same: evaluation, not fitting. A `train` split
#: would imply the opposite.
SPLIT_FRACTIONS = (("main", 0.75), ("held_out", 0.20), ("debug", 0.05))

#: EVERY SPLIT SHIPS EVERY ANNOTATION, and the split is a label rather than a
#: filter. IntPhys 2 withholds its held-out metadata to stop training
#: contamination, and that is the right call for a leaderboard someone else
#: submits to; it is the wrong one here, where the release has to stay
#: re-splittable and every clip has to be scoreable. A held-out set whose masks
#: are missing cannot be measured, only guessed at, and the boundary can never
#: be moved afterwards without regenerating.
#:
#: The leakage protection that remains is the part that cannot be undone later:
#: pairs never straddle a split, and splits are stratified per scenario. Anyone
#: publishing a leaderboard from this can strip the annotations at that point,
#: from `splits/held_out.txt`, without regenerating anything.


def _clip_dirs(root: str) -> List[str]:
    return sorted(os.path.dirname(p) for p in
                  glob.glob(os.path.join(root, "clips", "**", "meta.json"),
                            recursive=True))


def _hash_unit(pair_uid: str) -> float:
    """A stable number in [0, 1) from a pair uid."""
    h = hashlib.sha256(pair_uid.encode()).digest()
    return int.from_bytes(h[:8], "big") / float(1 << 64)


def assign_splits(pair_uids: Iterable[str]) -> Dict[str, str]:
    """{pair_uid: split}, in the declared proportions exactly.

    Pairs are ORDERED by a hash of their uid and then cut at the quantiles,
    rather than each pair independently falling into a hash bucket. Independent
    bucketing is the tidier rule and it does not hold its proportions on a small
    release: thirteen pairs came out 11/2/0, and a benchmark whose test split is
    empty is not a benchmark.

    Deterministic for a given set of pairs -- no rng, and the hash fixes the
    order -- so regenerating a release reproduces its splits exactly. ADDING
    pairs does reshuffle, because the quantile boundaries move; a release that
    grows should be re-split and re-reported, not appended to.
    """
    uids = sorted(set(str(p) for p in pair_uids))
    names = [name for name, _ in SPLIT_FRACTIONS]
    if not uids:
        return {}

    # STRATIFIED BY SCENARIO. A pair uid is `<release>/<scenario>/<seed>`, and
    # cutting the whole population at once lets a scenario land entirely in one
    # split -- which for a 13-scenario release is likely, and makes the held-out
    # set measure "have you seen `pour` before" rather than "do you understand
    # pouring". Splitting within each scenario keeps every split a picture of
    # the same benchmark.
    by_scenario: Dict[str, List[str]] = {}
    for uid in uids:
        parts = uid.split("/")
        by_scenario.setdefault(parts[1] if len(parts) > 2 else "", []).append(uid)
    out: Dict[str, str] = {}
    for group in by_scenario.values():
        out.update(_cut(sorted(group, key=_hash_unit), names))
    return out


def _cut(ordered: List[str], names: List[str]) -> Dict[str, str]:
    """Assign one ordered group of pairs to splits, in the declared shares."""
    n = len(ordered)
    if n == 0:
        return {}

    sizes = [int(round(frac * n)) for _, frac in SPLIT_FRACTIONS]
    # EVERY SPLIT GETS AT LEAST ONE, whenever there are enough pairs to go
    # round. Rounding alone starves the small splits on a small release -- six
    # pairs at 80/10/10 rounds to 5/1/0 -- and a test split of zero is not a
    # test split. Borrowed from the largest, which can afford it.
    if n >= len(names):
        for i in range(len(sizes)):
            if sizes[i] == 0:
                sizes[sizes.index(max(sizes))] -= 1
                sizes[i] = 1
    # Rounding can also over- or under-shoot the total by one or two.
    sizes[0] += n - sum(sizes)

    out: Dict[str, str] = {}
    start = 0
    for name, size in zip(names, sizes):
        for uid in ordered[start:start + size]:
            out[uid] = name
        start += size
    return out



def _row(meta: Dict, splits: Dict[str, str]) -> Dict:
    """One flat record per clip, for the index."""
    v = meta.get("violation") or {}
    cam = meta.get("camera") or {}
    # `instances`, not `assets`: the asset list is the licence record and
    # carries only name/source/licence, while the per-body physics and
    # appearance live in `instances`.
    bodies = meta.get("instances") or []
    actor = next((b for b in bodies
                  if b.get("role") == "actor" and not b.get("dormant")), {})
    windows = v.get("violation_windows") or []

    def _name(x):
        """`tier` and `complexity` are a bare string in some releases and a
        block in others; the index wants the name either way."""
        return x.get("name") if isinstance(x, dict) else x
    return {
        "clip_uid": meta.get("clip_uid"),
        "pair_uid": meta.get("pair_uid"),
        "twin_uid": meta.get("twin_uid"),
        "label": meta.get("label"),
        "scenario": meta.get("scenario"),
        "family": meta.get("family"),
        "domain": meta.get("domain"),
        "medium": meta.get("physics_medium"),
        "severity_bin": (v.get("intervention") or {}).get("severity_bin"),
        "magnitude": (v.get("intervention") or {}).get("magnitude"),
        "peak_severity": (v.get("peak_residual") or {}).get("score"),
        "t_event_frame": v.get("t_event_frame"),
        "violation_windows": json.dumps(windows),
        "observability_lag": v.get("observability_lag_frames"),
        "seed": meta.get("seed"),
        "tier": _name(meta.get("tier")),
        "complexity": _name(meta.get("complexity")),
        "num_frames": meta.get("num_frames"),
        "fps": meta.get("fps"),
        "camera_motion": cam.get("motion"),
        "actor_shape": actor.get("category"),
        "actor_material": actor.get("material"),
        "actor_mass": actor.get("mass_kg"),
        "prompt": meta.get("prompt"),
        "split": splits.get(str(meta.get("pair_uid")), "train"),
    }


def _write_shards(clips: Sequence[Tuple[str, Dict]], outdir: str, prefix: str,
                  members: Sequence[str], max_bytes: int) -> List[str]:
    """Pack each clip into a tar as one WebDataset sample.

    A sample's key is its `clip_uid` with slashes replaced, so every file of one
    clip shares a stem and `webdataset` groups them without being told how.
    """
    os.makedirs(outdir, exist_ok=True)
    written: List[str] = []
    tar = None
    size = 0
    try:
        for cdir, meta in clips:
            key = str(meta["clip_uid"]).replace("/", "__")
            present = [(m, os.path.join(cdir, m)) for m in members
                       if os.path.exists(os.path.join(cdir, m))]
            if not present:
                continue
            need = sum(os.path.getsize(p) for _, p in present)
            if tar is None or (size and size + need > max_bytes):
                if tar is not None:
                    tar.close()
                path = os.path.join(outdir, "%s-%03d.tar" % (prefix, len(written)))
                written.append(path)
                tar = tarfile.open(path, "w")
                size = 0
            for name, path in present:
                tar.add(path, arcname="%s.%s" % (key, name))
            size += need
    finally:
        if tar is not None:
            tar.close()
    return written


def export(root: str, outdir: str, with_passes: bool = False,
           shard_bytes: int = SHARD_BYTES,
           license_name: str = "CC-BY-4.0") -> Dict[str, object]:
    """Package the release at `root` into `outdir`. Returns a summary."""
    clip_dirs = _clip_dirs(root)
    if not clip_dirs:
        raise FileNotFoundError("no clips under %s" % root)

    clips: List[Tuple[str, Dict]] = []
    for cdir in clip_dirs:
        with open(os.path.join(cdir, "meta.json")) as fh:
            clips.append((cdir, json.load(fh)))

    os.makedirs(outdir, exist_ok=True)
    splits = assign_splits(str(m.get("pair_uid")) for _, m in clips)
    rows = [_row(m, splits) for _, m in clips]

    # Splits, by PAIR. Writing the clip uids out per split rather than the pair
    # uids means a consumer never has to know the grouping rule to respect it.
    os.makedirs(os.path.join(outdir, "splits"), exist_ok=True)
    for name, _ in SPLIT_FRACTIONS:
        members = sorted(r["clip_uid"] for r in rows if r["split"] == name)
        with open(os.path.join(outdir, "splits", "%s.txt" % name), "w") as fh:
            fh.write("\n".join(members) + ("\n" if members else ""))

    # SHARDED BY SPLIT, so a consumer can fetch one without the others -- but
    # every split gets the same files, so the boundary can be moved later and
    # any clip can be scored.
    shard_dir = os.path.join(outdir, "shards")
    shards: Dict[str, List[str]] = {}
    for name, _ in SPLIT_FRACTIONS:
        part = [(d, m) for d, m in clips
                if splits.get(str(m.get("pair_uid"))) == name]
        if not part:
            continue
        shards[name] = _write_shards(part, shard_dir, name, CORE_FILES,
                                     shard_bytes)
        if with_passes:
            shards["%s_passes" % name] = _write_shards(
                part, shard_dir, "%s-passes" % name, PASS_FILES, shard_bytes)

    index_path = _write_index(rows, outdir)
    _write_taxonomy(outdir, rows)
    _write_card(rows, outdir, license_name, shards)
    _write_license(outdir, license_name)
    counts = {n: sum(1 for r in rows if r["split"] == n)
              for n, _ in SPLIT_FRACTIONS}
    notes = []
    empty = [n for n, c in counts.items() if not c]
    if empty:
        # Said out loud, because an empty held-out split is the difference
        # between a benchmark and a training set, and it is not obvious from
        # the numbers why it happened. Splits are stratified per scenario, so a
        # release with one pair per scenario cannot fill three splits -- it
        # needs several variants per scenario before there is anything to hold
        # out.
        per = len({r["pair_uid"] for r in rows}) / max(
            len({r["scenario"] for r in rows}), 1)
        notes.append(
            "splits %s are empty: %.1f pairs per scenario is too few to "
            "stratify three ways. Generate more variants per scenario."
            % (", ".join(sorted(empty)), per))
    return {
        "clips": len(clips),
        "pairs": len({r["pair_uid"] for r in rows}),
        "notes": notes,
        "shards": {k: len(v) for k, v in shards.items()},
        "index": index_path,
        "splits": counts,
        "outdir": outdir,
    }


def _write_taxonomy(outdir: str, rows: List[Dict]) -> str:
    """`taxonomy.json` -- what every scenario and family in the release IS.

    The index says which clip is which; this says what the labels mean. Without
    it a consumer has `family: "newton1_inertia"` and a string, and has to come
    back to the repository to learn that it is a kinematics violation, that its
    law is `momentum_conservation`, and that a scenario called `pour` is
    granular rather than rigid.

    Generated from `physloc.taxonomy`, never hand-written, so it cannot drift
    from the code the way the prose tables in docs/ did -- five hand-copies of
    the same table disagreed with each other and with the code.
    """
    from ..taxonomy import DOMAINS, FAMILIES, MEDIA, SCENARIOS

    present_scen = {r["scenario"] for r in rows}
    present_fam = {r["family"] for r in rows if r["family"]}
    blob = {
        "media": {k: v for k, v in MEDIA.items()},
        "domains": {k: v for k, v in DOMAINS.items()},
        "scenarios": {
            name: {
                "description": sc.description,
                "event_structure": sc.event_structure,
                "physics_medium": sc.physics_medium,
                "has_occluder": bool(sc.has_occluder),
                "provides": list(sc.provides),
                "families": sorted(f for f in present_fam
                                   if any(r["scenario"] == name
                                          and r["family"] == f for r in rows)),
                "clips_in_release": sum(1 for r in rows
                                        if r["scenario"] == name),
            }
            for name, sc in SCENARIOS.items() if name in present_scen
        },
        "families": {
            name: {
                "domain": fam.domain,
                "law": fam.law,
                "injection": fam.injection,
                "kind": fam.kind,
                "magnitude_unit": fam.magnitude_unit,
                "detectable": fam.detectable,
                "graded": bool(fam.graded),
                "requires": list(fam.requires),
                "clips_in_release": sum(1 for r in rows if r["family"] == name),
            }
            for name, fam in FAMILIES.items() if name in present_fam
        },
    }
    path = os.path.join(outdir, "taxonomy.json")
    with open(path, "w") as fh:
        json.dump(blob, fh, indent=2, sort_keys=True)
    return path


def _write_index(rows: List[Dict], outdir: str) -> str:
    """The per-clip table. Parquet when pyarrow is here, JSONL when it is not.

    Falling back rather than failing: the index is what makes the dataset
    viewer work, and a release that packaged everything except the viewer table
    is still a release. The card records which format shipped.
    """
    path = os.path.join(outdir, "index.parquet")
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        path = os.path.join(outdir, "index.jsonl")
        with open(path, "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        return path
    pq.write_table(pa.Table.from_pylist(rows), path)
    return path


def _write_license(outdir: str, license_name: str) -> None:
    path = os.path.join(outdir, "LICENSE")
    if os.path.exists(path):
        return
    with open(path, "w") as fh:
        fh.write(
            "%s\n\n"
            "The clips in this dataset are rendered from primitive geometry\n"
            "with Kubric (Apache-2.0) and Blender. Per-asset licences are\n"
            "recorded in every clip's meta.json under `assets[].license`.\n"
            % license_name)


def _write_card(rows: List[Dict], outdir: str, license_name: str,
                shards: Dict[str, List[str]]) -> None:
    """The dataset card. YAML front-matter first, because the hub parses it."""
    from collections import Counter

    scenarios = Counter(r["scenario"] for r in rows)
    families = Counter(r["family"] for r in rows if r["family"])
    tiers = sorted({r["tier"] for r in rows if r["tier"]})
    invalid = sum(1 for r in rows if r["label"] == "invalid")
    moving = sum(1 for r in rows if r["camera_motion"] not in (None, "static"))
    index_kind = ("parquet" if os.path.exists(os.path.join(outdir, "index.parquet"))
                  else "jsonl")

    # THE VIEWER NEEDS `configs`. Without it the hub shows a repository of
    # opaque tar and parquet files and nothing renders. Two configs, because
    # they answer different questions:
    #
    #   `index`  parquet, one row per clip -- sortable, filterable, instant.
    #            Listed FIRST so it is the default: it is guaranteed to render,
    #            where a webdataset config can fail on an unfamiliar member and
    #            take the whole viewer down with it.
    #   `clips`  the shards themselves -- mp4 previews and the annotations.
    splits_present = [n for n, _ in SPLIT_FRACTIONS
                      if any(r["split"] == n for r in rows)]
    cfg = ["configs:",
           "- config_name: index",
           "  data_files:"]
    for n in splits_present:
        cfg.append("  - split: %s" % n)
        cfg.append("    path: index.parquet")
        break                      # the index carries every split in one file
    if any(shards.get(n) for n in splits_present):
        cfg += ["- config_name: clips", "  data_files:"]
        for n in splits_present:
            if shards.get(n):
                cfg.append("  - split: %s" % n)
                cfg.append("    path: shards/%s-*.tar" % n)

    lines = [
        "---",
        "license: %s" % license_name.lower(),
        *cfg,
        "task_categories:",
        "- video-classification",
        "tags:",
        "- physics",
        "- video",
        "- intuitive-physics",
        "- counterfactual",
        "size_categories:",
        "- %s" % ("n<1K" if len(rows) < 1000 else "1K<n<10K"),
        "---",
        "",
        "# PhysLoc",
        "",
        "Physics-violation video clips with **spatio-temporal annotations**: "
        "every invalid clip ships where the violation is, when it happens, and "
        "how badly -- all derived from the simulator rather than annotated by "
        "hand.",
        "",
        "Clips come in **twins**. A valid clip and its invalid partner share a "
        "scene, a seed, and a bit-identical prefix up to `t_event`; only after "
        "that do they differ. That is what makes the difference between them "
        "attributable to the intervention and nothing else.",
        "",
        "## What is here",
        "",
        "| | |",
        "|---|---|",
        "| clips | %d (%d invalid, %d valid) |" % (len(rows), invalid,
                                                   len(rows) - invalid),
        "| pairs | %d |" % len({r["pair_uid"] for r in rows}),
        "| scenarios | %d |" % len(scenarios),
        "| violation families | %d |" % len(families),
        "| tier | %s |" % ", ".join(tiers),
        "| moving camera | %d clips (%.0f%%) |" % (
            moving, 100.0 * moving / max(len(rows), 1)),
        "",
        "## Files",
        "",
        "- `shards/<split>-*.tar` -- RGB video and every annotation, one set "
        "per split. **Start here.**",
        ("- `shards/<split>-passes-*.tar` -- depth, optical flow, normals, "
         "object coordinates. Far larger: about 86% of the bytes, so they ship "
         "separately rather than inside the download everyone needs."
         if any(k.endswith("_passes") for k in shards) else
         "- The raw geometry passes (depth, optical flow, normals, object "
         "coordinates) are **not** in this release. They are about 86% of the "
         "bytes and are packaged only on request."),
        "- `index.%s` -- one row per clip: uids, scenario, family, severity, "
        "windows, camera motion, actor shape and material." % index_kind,
        "- `splits/*.txt` -- clip uids per split.",
        "",
        "## Splits leak nothing",
        "",
        "Splits are grouped by `pair_uid`, never by clip. A valid twin and its "
        "invalid siblings share every frame before `t_event`, so splitting them "
        "apart would put the answer on the other side.",
        "",
        "Pairs are ordered by a hash of their uid and cut at the quantiles "
        "WITHIN each scenario, so every split sees every scenario and the "
        "proportions are exact. There is no rng, so reproducing this release "
        "reproduces its splits. Adding clips moves the boundaries.",
        "",
        "**Every split ships every annotation.** The split is a label, not a "
        "filter: you can re-cut it, and you can score any clip in the release. "
        "If you need a genuinely blind held-out set -- for a leaderboard others "
        "submit to -- strip the annotations from `splits/held_out.txt` at that "
        "point; nothing here has to be regenerated to do it.",
        "",
        "## Reading a clip",
        "",
        "```python",
        "import webdataset as wds",
        "",
        'ds = wds.WebDataset("shards/core-000.tar")',
        "for sample in ds:",
        '    meta = sample["meta.json"]      # bytes -> json.loads',
        '    video = sample["rgb.mp4"]',
        '    mask = sample["violation_mask.npz"]',
        "```",
        "",
        "## The annotations, and what they are not",
        "",
        "- `violation_mask` -- where the violation is, unioned over BOTH twins. "
        "A vanished body has no pixels in the invalid render, which is exactly "
        "why the union is needed.",
        "- `mask_invalid` -- the same footprint, invalid side only.",
        "- `severity_map` -- continuous, how badly, invalid side only.",
        "- `causal_mask` -- level 1 the culprit, level 2 a body it disturbed.",
        "- `reference_mask` -- where the culprit lawfully should have been.",
        "- `divergence_map` -- `|valid - invalid|` in pixels. **Ships for "
        "inspection, not for training**: it diverges everywhere downstream of "
        "the event, so a model trained on it learns to find the edit rather "
        "than the physics.",
        "",
        "## Scenarios",
        "",
        "".join("- `%s` (%d clips)\n" % (k, v)
                for k, v in sorted(scenarios.items())),
        "## Violation families",
        "",
        "".join("- `%s` (%d clips)\n" % (k, v)
                for k, v in sorted(families.items())),
    ]
    with open(os.path.join(outdir, "README.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")


def upload(outdir: str, repo_id: str, private: bool = False,
           token: Optional[str] = None) -> str:
    """Push a packaged release to the HuggingFace Hub. Returns its URL.

    Separate from `export` on purpose: packaging is local and repeatable,
    uploading is neither. Publishing puts the clips somewhere they can be
    fetched, indexed and cached by people who are not you, so it is worth being
    a deliberate second step rather than a flag on the first.
    """
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private,
                    exist_ok=True)
    api.upload_folder(repo_id=repo_id, repo_type="dataset", folder_path=outdir)
    return "https://huggingface.co/datasets/%s" % repo_id
