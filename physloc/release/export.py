"""Package a generated schema-v3 tree for local or Hugging Face use."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections import Counter
from typing import Dict, Iterable, List, Optional

from .. import loader
from ..annotate import layout
from ..residuals.energy import ENERGY_EXCLUDED_ROLES

SPLIT_FRACTIONS = (("main", 0.75), ("held_out", 0.20), ("debug", 0.05))


def _hash_unit(value: str) -> float:
    digest = hashlib.sha256(value.encode()).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def _cut(values: List[str], names: List[str]) -> Dict[str, str]:
    n = len(values)
    if not n:
        return {}
    weights = dict(SPLIT_FRACTIONS)
    raw = [weights[name] * n for name in names]
    counts = [int(value) for value in raw]
    for index in sorted(range(len(names)), key=lambda i: raw[i] - counts[i], reverse=True)[:n - sum(counts)]:
        counts[index] += 1
    if n >= len(names):
        for index, count in enumerate(counts):
            if count:
                continue
            donor = max(range(len(counts)), key=lambda i: counts[i])
            counts[donor] -= 1
            counts[index] = 1
    out = {}
    start = 0
    for name, count in zip(names, counts):
        for value in values[start:start + count]:
            out[value] = name
        start += count
    return out


def assign_splits(pair_uids: Iterable[str]) -> Dict[str, str]:
    """Deterministic, pair-grouped and scenario-stratified splits."""
    groups: Dict[str, List[str]] = {}
    for uid in sorted(set(str(value) for value in pair_uids)):
        parts = uid.replace(os.sep, "/").split("/")
        scenario = parts[-2] if len(parts) >= 2 else "dataset"
        groups.setdefault(scenario, []).append(uid)
    names = [name for name, _ in SPLIT_FRACTIONS]
    out = {}
    for values in groups.values():
        ordered = sorted(values, key=lambda value: (_hash_unit(value), value))
        out.update(_cut(ordered, names))
    return out


def _row(document: Dict, sample_path: str) -> Dict:
    info = document["metadata"]["sample_info"]
    scene = document["metadata"]["scene"]["info"]
    world = document["metadata"]["scene"]["world"]
    counts = world.get("objects_summary") or {}
    return {
        "sample_uid": info["sample_uid"], "pair_uid": info["pair_uid"],
        "valid_sample_uid": info["valid_sample_uid"], "label": info["label"],
        "split": info["split"], "scenario": scene.get("type"),
        "family": scene.get("family"), "domain": scene.get("domain"),
        "physics_medium": scene.get("physics_medium"),
        "severity": scene.get("severity"), "complexity": scene.get("complexity"),
        "difficulty": scene.get("difficulty"), "condition": scene.get("condition"),
        "seed": info.get("seed"), "variant": info.get("variant"),
        "num_frames": info.get("num_frames"), "fps": info.get("fps"),
        "n_actors": counts.get("n_actors"), "n_distractors": counts.get("n_distractors"),
        "n_violators": counts.get("n_violators"), "prompt": scene.get("prompt"),
        "sample_path": sample_path,
        "rgb_path": sample_path + "/" + loader.RGB,
        "data_path": sample_path + "/" + loader.DATA,
        "dataset_version": info.get("dataset_version"),
        "generation_config_id": info.get("generation_config_id"),
        "resolution": info.get("resolution"),
    }


def _write_index(rows: List[Dict], root: str) -> str:
    for name in ("index.parquet", "index.jsonl"):
        existing = os.path.join(root, name)
        if os.path.exists(existing):
            os.remove(existing)
    path = os.path.join(root, "index.parquet")
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        path = os.path.join(root, "index.jsonl")
        with open(path, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        return path
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path)
    return path


def _schema() -> Dict:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://physloc/schema/v3/sample.json",
        "title": "PhysLoc sample schema v3", "type": "object",
        "required": ["schema_version", "metadata", "observations", "annotations", "storage"],
        "properties": {
            "schema_version": {"const": loader.SCHEMA_VERSION},
            "metadata": {"type": "object", "required": ["sample_info", "scene"]},
            "observations": {"type": "object", "required": ["rgb", "dense"]},
            "annotations": {"type": "object", "required": ["objects", "maps", "events"]},
            "storage": {"type": "object", "required": ["dense_store", "format"]},
        }, "additionalProperties": False,
    }


def _difficulty_analysis(root: str) -> Dict[str, object]:
    """Compact release-wide difficulty measurements for ``dataset.json``."""
    from . import stats
    from ..annotate import difficulty as difficulty_module

    records = stats.load(root)
    summary = stats.summarise(records)

    def quantile(values, fraction):
        ordered = sorted(float(value) for value in values)
        if not ordered:
            return None
        position = (len(ordered) - 1) * float(fraction)
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        weight = position - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    zone_counts = {factor.name: Counter() for factor in difficulty_module.FACTORS}
    for record in records:
        for name, value in ((record.get("difficulty") or {}).get("factors") or {}).items():
            if name in zone_counts and value.get("level"):
                zone_counts[name][str(value["level"])] += 1
    factors = {}
    for name, values in summary["factor_values"].items():
        numbers = [float(value) for value in values]
        factors[name] = {
            "count": len(numbers), "zone_counts": dict(zone_counts[name]),
            "min": min(numbers) if numbers else None,
            "p05": quantile(numbers, 0.05), "median": quantile(numbers, 0.50),
            "p95": quantile(numbers, 0.95), "max": max(numbers) if numbers else None,
            "mean": (sum(numbers) / len(numbers)) if numbers else None,
        }
    return {
        "number_of_invalid_samples": summary["invalid"],
        "label_distribution": summary["difficulty"],
        "labels_by_complexity": summary["difficulty_by_level"],
        "label_setting_factors": summary["binding_factors"],
        "thresholds": summary["thresholds"],
        "factors": factors,
    }


def _write_global_files(root: str, rows: List[Dict], license_name: str) -> None:
    splits = Counter(row["split"] for row in rows)
    complexity = Counter(row.get("complexity") for row in rows)
    versions = sorted({str(row.get("dataset_version") or "") for row in rows})
    dataset = {
        "schema_version": loader.SCHEMA_VERSION,
        "dataset_metadata": {
            "name": "PhysLoc", "version": versions[-1] if versions else "",
            "number_of_samples": len(rows),
            "number_of_pairs": len({row["pair_uid"] for row in rows}),
            "splits": dict(sorted(splits.items())),
            "split_definitions": {
                name: {"fraction": fraction, "unit": "pair",
                       "assignment": "deterministic scenario-stratified SHA-256"}
                for name, fraction in SPLIT_FRACTIONS
            },
            "complexity_splits": {str(key): value for key, value in sorted(complexity.items())},
            "taxonomy": {
                "scenarios": sorted({str(row["scenario"]) for row in rows
                                     if row.get("scenario") is not None}),
                "families": sorted({str(row["family"]) for row in rows
                                    if row.get("family") is not None}),
                "domains": sorted({str(row["domain"]) for row in rows
                                   if row.get("domain") is not None}),
            },
            "generation_config_ids": sorted({str(row["generation_config_id"])
                                             for row in rows
                                             if row.get("generation_config_id")}),
            "difficulty_analysis": _difficulty_analysis(root),
            "energy_accounting": {
                "world_total": {
                    "hdf5_path": "/energy/scene/total",
                    "scope": "all energy-eligible physical objects, whether visible or occluded",
                    "purpose": "physics conservation, residual, and anomaly analysis",
                },
                "visible_scene_total": {
                    "hdf5_path": "/energy/scene/energy_in_frame",
                    "scope": "energy-eligible objects visible in rendered instance segmentation",
                    "purpose": "video-grounded analysis and visualization",
                },
                "per_object": {
                    "hdf5_path": "/energy/objects/by_body",
                    "object_axis": "/objects/ids",
                },
                "included": [
                    "dynamic subjects", "dynamic context and distractors",
                    "dynamic affected objects and peers",
                ],
                "excluded_roles": sorted(ENERGY_EXCLUDED_ROLES),
            },
            "license": license_name,
            "units": {"length": "m", "time": "s", "mass": "kg", "energy": "J"},
            "coordinate_convention": "right-handed; +Z up; quaternions w,x,y,z",
            "shadow_strength_threshold": 1.0 / 255.0,
            "violation_components": {"0": "none", "1": "body", "2": "shadow",
                                     "3": "trajectory", "4": "interaction", "5": "energy"},
            "analysis_groups": ["subject", "context", "support", "background"],
            "checksum_policy": "HDF5 Fletcher32 per chunk",
        }
    }
    with open(os.path.join(root, "dataset.json"), "w", encoding="utf-8") as handle:
        json.dump(dataset, handle, indent=2, sort_keys=True)
    with open(os.path.join(root, "schema.json"), "w", encoding="utf-8") as handle:
        json.dump(_schema(), handle, indent=2, sort_keys=True)
    with open(os.path.join(root, "LICENSE"), "w", encoding="utf-8") as handle:
        handle.write(license_name + "\n")
    card = """---
license: {license}
task_categories:
- video-classification
tags:
- physics
- video
---

# PhysLoc schema v3

PhysLoc stores each sample as `sample.json`, a standalone `rgb.mp4`, and one
chunked `data.h5`. The Parquet index contains relative paths and never embeds
duplicate video bytes.

```python
from loader import PhysLocDataset
dataset = PhysLocDataset(".")
sample = dataset.samples[0]
sample.video_path
sample.objects("violators")
```

Download with `hf download <repo> --repo-type dataset --local-dir physloc`.
Publish with `hf upload <repo> <directory> --type dataset`.
""".format(license=license_name.lower())
    with open(os.path.join(root, "README.md"), "w", encoding="utf-8") as handle:
        handle.write(card)
    shutil.copyfile(loader.__file__, os.path.join(root, "loader.py"))


def finalize(root: str, license_name: str = "CC-BY-4.0") -> Dict[str, object]:
    """Write the global index, splits, schema, card, and dataset metadata."""
    metadata_paths = layout.find(root)
    if not metadata_paths:
        raise FileNotFoundError("no schema-v3 samples under %s" % root)
    documents = [(os.path.dirname(path), layout.read(os.path.dirname(path)))
                 for path in metadata_paths]
    splits = assign_splits(layout.identity(document)["pair_uid"]
                           for _, document in documents)
    rows = []
    for sample_dir, document in documents:
        info = layout.identity(document)
        uid = str(info["sample_uid"])
        document["metadata"]["sample_info"]["split"] = splits[str(info["pair_uid"])]
        with open(os.path.join(sample_dir, loader.SAMPLE_METADATA), "w", encoding="utf-8") as handle:
            json.dump(document, handle, indent=2, sort_keys=True)
        rows.append(_row(document, "samples/" + uid))
    split_dir = os.path.join(root, "splits")
    if os.path.isdir(split_dir):
        shutil.rmtree(split_dir)
    os.makedirs(split_dir)
    for name, _ in SPLIT_FRACTIONS:
        members = sorted(row["sample_uid"] for row in rows if row["split"] == name)
        with open(os.path.join(split_dir, name + ".txt"), "w", encoding="utf-8") as handle:
            handle.write("\n".join(members) + ("\n" if members else ""))
    index = _write_index(rows, root)
    _write_global_files(root, rows, license_name)
    return {"samples": len(rows), "pairs": len({row["pair_uid"] for row in rows}),
            "schema_version": loader.SCHEMA_VERSION, "index": index,
            "splits": {name: sum(row["split"] == name for row in rows)
                       for name, _ in SPLIT_FRACTIONS}, "outdir": root}


def export(root: str, outdir: str, license_name: str = "CC-BY-4.0") -> Dict[str, object]:
    """Copy an already-v3 sample tree and finalize it for publication."""
    if os.path.realpath(root) == os.path.realpath(outdir):
        raise ValueError("export source and destination must be different directories")
    metadata_paths = layout.find(root)
    if not metadata_paths:
        raise FileNotFoundError("no schema-v3 samples under %s" % root)
    target_samples = os.path.join(outdir, "samples")
    if os.path.isdir(target_samples):
        shutil.rmtree(target_samples)
    os.makedirs(target_samples, exist_ok=True)
    for path in metadata_paths:
        source = os.path.dirname(path)
        document = layout.read(source)
        info = layout.identity(document)
        uid = str(info["sample_uid"])
        destination = os.path.join(outdir, "samples", *uid.split("/"))
        shutil.copytree(source, destination)
    return finalize(outdir, license_name)


def upload(outdir: str, repo_id: str, private: bool = False,
           token: Optional[str] = None) -> str:
    """Publish through the current Hugging Face `hf` CLI."""
    import subprocess
    command = ["hf", "upload", repo_id, outdir, ".", "--type", "dataset",
               "--commit-message", "Publish PhysLoc schema v3"]
    if private:
        command.append("--private")
    environment = os.environ.copy()
    if token:
        environment["HF_TOKEN"] = token
    subprocess.run(command, check=True, env=environment)
    return "https://huggingface.co/datasets/%s" % repo_id
