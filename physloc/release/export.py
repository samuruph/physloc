"""Package a generated schema-v4 tree for local or Hugging Face use."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections import Counter
from typing import Dict, Iterable, List, Optional, Tuple

from .. import loader, reference
from ..annotate import layout
from ..schema import write
from ..residuals.energy import ENERGY_EXCLUDED_ROLES

#: The share of PAIRS held out for evaluation. A pair -- one scene, its valid
#: twin and every invalid sibling -- is the unit, because siblings share their
#: whole prefix: 283 scenes are 283 independent observations, not 9,579.
HELD_OUT_SHARE = 0.25
SPLITS = ("main", "held_out")
SPLIT_FRACTIONS = (("main", 1.0 - HELD_OUT_SHARE), ("held_out", HELD_OUT_SHARE))
#: `debug` is a SUBSET of `main`, one pair per scenario -- a smoke test that
#: spends none of the evaluation pairs. It used to be a third split taking 5%.
DEBUG_PER_SCENARIO = 1
#: How many candidate splits are drawn before the most balanced is kept.
REDRAWS = 1000
#: Written beside the per-sample lists; pass it back with `--splits` to keep a
#: published assignment fixed when scenes are added.
PAIRS_FILE = "pairs.json"


def _hash_unit(value: str) -> float:
    digest = hashlib.sha256(value.encode()).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def strata(pair_uid: str) -> Tuple[str, str, str]:
    """(level, scenario, condition) of a ``release/level/scenario/seed_condition``
    pair uid -- the layout `annotate.pipeline` writes."""
    parts = str(pair_uid).replace(os.sep, "/").split("/")
    scenario = parts[-2] if len(parts) >= 2 else "dataset"
    level = parts[-3] if len(parts) >= 3 else "?"
    condition = parts[-1].split("_", 1)[1] if "_" in parts[-1] else "?"
    return level, scenario, condition


def _held_counts(cells: Dict[Tuple[str, str], List[str]], salt: str
                 ) -> Dict[Tuple[str, str], int]:
    """How many pairs each (level, scenario) cell holds out.

    Every cell of two or more pairs holds out at least one, so each level's
    held_out covers every scenario -- at L2 and L3 a scenario has two to five
    scenes, and a plain 25% leaves some with none, which makes "L3 against L0"
    compare different scenario mixes. Above that floor a level's total is the
    share of its pairs, its rounding remainders going to the cells with the
    largest fractional part (ties broken by `salt`, so the redraws can pick
    which scenarios take them).
    """
    out: Dict[Tuple[str, str], int] = {}
    levels: Dict[str, List[Tuple[str, str]]] = {}
    for key in cells:
        levels.setdefault(key[0], []).append(key)
    for keys in levels.values():
        exact = {key: HELD_OUT_SHARE * len(cells[key]) for key in keys}
        for key in keys:
            floor = int(exact[key])
            out[key] = max(1, floor) if len(cells[key]) >= 2 else floor
        target = int(HELD_OUT_SHARE * sum(len(cells[key]) for key in keys) + 0.5)
        spare = sorted((key for key in keys if out[key] < len(cells[key]) - 1
                        and out[key] < exact[key]),
                       key=lambda k: (-(exact[k] - out[k]), _hash_unit(salt + "/".join(k))))
        for key in spare[:max(0, target - sum(out[key] for key in keys))]:
            out[key] += 1
    return out


def _draw(cells: Dict[Tuple[str, str], List[str]], salt: str) -> Dict[str, str]:
    """One candidate: within each cell, pairs ordered by condition and taken at a
    regular stride from a hashed offset -- systematic sampling, so every
    condition in the cell is held out in proportion without a cut of its own."""
    out = {}
    counts = _held_counts(cells, salt)
    for key, uids in cells.items():
        ordered = sorted(uids, key=lambda u: (strata(u)[2], _hash_unit(salt + u), u))
        n, k = len(ordered), counts[key]
        offset = _hash_unit(salt + "/".join(key))
        picks = {int((offset + j) * n / k) % n for j in range(k)} if k else set()
        for index, uid in enumerate(ordered):
            out[uid] = "held_out" if index in picks else "main"
    return out


def imbalance(assignment: Dict[str, str]) -> float:
    """Squared distance of every margin from `HELD_OUT_SHARE`, per pair --
    scenario, level, condition and their pairwise crossings."""
    margins: Dict[Tuple, List[int]] = {}
    for uid, split in assignment.items():
        level, scenario, condition = strata(uid)
        for key in (("s", scenario), ("l", level), ("c", condition),
                    ("ls", level, scenario), ("lc", level, condition),
                    ("cs", condition, scenario)):
            got = margins.setdefault(key, [0, 0])
            got[0] += split == "held_out"
            got[1] += 1
    return sum((held - HELD_OUT_SHARE * n) ** 2 / n for held, n in margins.values())


def assign_splits(pair_uids: Iterable[str],
                  frozen: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """``pair_uid -> "main" | "held_out"``: deterministic, pair-grouped, and
    stratified by level x scenario with condition balanced inside each cell.

    `REDRAWS` candidates are drawn and the most balanced kept (rerandomisation:
    each candidate is a fair draw, so the one kept is too). The result depends
    on the SET of pairs, not their order -- and therefore changes when pairs are
    added. `frozen` pins a published assignment: those pairs keep their split
    and only the others are drawn, among themselves.
    """
    frozen = {str(k): v for k, v in (frozen or {}).items() if v in SPLITS}
    uids = sorted(set(str(value) for value in pair_uids))
    out = {uid: frozen[uid] for uid in uids if uid in frozen}
    cells: Dict[Tuple[str, str], List[str]] = {}
    for uid in uids:
        if uid not in out:
            level, scenario, _ = strata(uid)
            cells.setdefault((level, scenario), []).append(uid)
    if cells:
        best = min((_draw(cells, "redraw%d:" % i) for i in range(REDRAWS)),
                   key=imbalance)
        out.update(best)
    return out


def debug_subset(assignment: Dict[str, str]) -> List[str]:
    """`DEBUG_PER_SCENARIO` main pairs per scenario, by hash."""
    by_scenario: Dict[str, List[str]] = {}
    for uid, split in assignment.items():
        if split == "main":
            by_scenario.setdefault(strata(uid)[1], []).append(uid)
    return sorted(uid for uids in by_scenario.values()
                  for uid in sorted(uids, key=lambda u: (_hash_unit("debug:" + u), u))
                  [:DEBUG_PER_SCENARIO])


def read_pairs(path: str) -> Dict[str, str]:
    """A `PAIRS_FILE` written by `finalize`, as ``pair_uid -> split``."""
    with open(path, encoding="utf-8") as handle:
        return dict(json.load(handle)["pairs"])


def _row(sample: "loader.Sample", sample_path: str) -> Dict:
    """One index.parquet row: the columns a user filters a release on."""
    info, scene, v = sample.info, sample.scene, sample.violation
    roles = sample.objects.roles
    return {
        "sample_uid": info.uid, "pair_uid": info.pair_uid,
        "valid_uid": info.valid_uid, "label": info.label, "split": info.split,
        "scenario": scene.scenario, "family": scene.family, "domain": scene.domain,
        "physics_medium": scene.physics_medium, "level": scene.level,
        "condition": scene.condition, "severity": v.severity_bin,
        "difficulty": (v.difficulty or {}).get("level"),
        "seed": info.seed, "variant": info.variant,
        "num_frames": sample.video.num_frames, "fps": sample.video.fps,
        "resolution": list(sample.video.resolution),
        "n_objects": len(roles), "n_actors": roles.count("actor"),
        "n_distractors": roles.count("distractor"), "n_violators": len(v.violators),
        "prompt": scene.prompt, "release": info.release, "tier": info.tier,
        "sample_path": sample_path,
        "rgb_path": sample_path + "/" + loader.RGB,
        "data_path": sample_path + "/" + loader.DATA,
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
    obj = {"type": "object"}
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://physloc/schema/v%d/sample.json" % loader.SCHEMA_VERSION,
        "title": "PhysLoc sample schema v%d" % loader.SCHEMA_VERSION,
        "type": "object",
        "required": ["schema_version", "sample", "video", "scene", "objects"],
        "properties": {
            "schema_version": {"const": loader.SCHEMA_VERSION},
            "sample": dict(obj, required=["uid", "pair_uid", "valid_uid", "label",
                                          "split", "release", "tier", "seed"]),
            "video": dict(obj, required=["num_frames", "fps", "resolution"]),
            "scene": dict(obj, required=["scenario", "level", "condition", "camera",
                                         "environment", "physics"]),
            "objects": {"type": "array", "items": dict(
                obj, required=["id", "name", "role", "analysis_group", "asset",
                               "physics", "render"])},
            "violation": dict(obj, required=["severity_bin", "kind", "component",
                                             "intervention", "violators"]),
            "provenance": obj,
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
    complexity = Counter(row.get("level") for row in rows)
    versions = sorted({str(row.get("release") or "") for row in rows})
    dataset = {
        "schema_version": loader.SCHEMA_VERSION,
        "dataset_metadata": {
            "name": "PhysLoc", "version": versions[-1] if versions else "",
            "number_of_samples": len(rows),
            "number_of_pairs": len({row["pair_uid"] for row in rows}),
            "splits": dict(sorted(splits.items())),
            "split_ratios": dict(SPLIT_FRACTIONS),
            "split_design": {
                "unit": "pair_uid", "held_out_share_of_pairs": HELD_OUT_SHARE,
                "strata": "level x scenario, at least one held_out pair per "
                          "cell of two or more; condition balanced within",
                "debug": "subset of main, %d pair(s) per scenario" % DEBUG_PER_SCENARIO,
                "pairs_file": "splits/" + PAIRS_FILE,
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
            "generation_config_ids": sorted({"%s:%s" % (row["release"], row["tier"])
                                             for row in rows if row.get("release")}),
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
            "floor_datum": {"z": 0.0, "units": "m"},
            "storage": {"metadata": loader.SAMPLE_METADATA, "rgb": loader.RGB,
                        "dense": loader.DATA, "format": "HDF5",
                        "compression": "gzip-4", "checksum": "fletcher32"},
            "shadow_strength_threshold": loader.SHADOW_STRENGTH_THRESHOLD,
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
    card = reference.hf_card(license_name, loader.SCHEMA_VERSION, SPLIT_FRACTIONS)
    with open(os.path.join(root, "README.md"), "w", encoding="utf-8") as handle:
        handle.write(card)
    shutil.copyfile(loader.__file__, os.path.join(root, "loader.py"))


def finalize(root: str, license_name: str = "CC-BY-4.0",
             splits_from: Optional[str] = None) -> Dict[str, object]:
    """Write the global index, splits, schema, card, and dataset metadata.

    `splits_from` is a `PAIRS_FILE` whose assignments are kept as they are;
    without it the split is drawn afresh from the pairs present.
    """
    metadata_paths = layout.find(root)
    if not metadata_paths:
        raise FileNotFoundError("no schema-v%d samples under %s" % (loader.SCHEMA_VERSION, root))
    documents = [(os.path.dirname(path), layout.read(os.path.dirname(path)))
                 for path in metadata_paths]
    frozen = read_pairs(splits_from) if splits_from else None
    splits = assign_splits((layout.identity(document)["pair_uid"]
                            for _, document in documents), frozen=frozen)
    debug = set(debug_subset(splits))
    rows = []
    for sample_dir, document in documents:
        info = layout.identity(document)
        info["split"] = splits[str(info["pair_uid"])]
        write.dump(document, os.path.join(sample_dir, loader.SAMPLE_METADATA))
        sample = loader.Sample(sample_dir)
        rows.append(_row(sample, "samples/" + str(info["uid"])))
        sample.release()
    split_dir = os.path.join(root, "splits")
    if os.path.isdir(split_dir):
        shutil.rmtree(split_dir)
    os.makedirs(split_dir)
    lists = {name: [row for row in rows if row["split"] == name] for name in SPLITS}
    lists["debug"] = [row for row in rows if row["pair_uid"] in debug]
    for name, members in lists.items():
        uids = sorted(row["sample_uid"] for row in members)
        with open(os.path.join(split_dir, name + ".txt"), "w", encoding="utf-8") as handle:
            handle.write("\n".join(uids) + ("\n" if uids else ""))
    with open(os.path.join(split_dir, PAIRS_FILE), "w", encoding="utf-8") as handle:
        json.dump({"held_out_share": HELD_OUT_SHARE, "pairs": dict(sorted(splits.items())),
                   "debug": sorted(debug), "frozen_from": splits_from},
                  handle, indent=1, sort_keys=True)
    index = _write_index(rows, root)
    _write_global_files(root, rows, license_name)
    return {"samples": len(rows), "pairs": len({row["pair_uid"] for row in rows}),
            "schema_version": loader.SCHEMA_VERSION, "index": index,
            "splits": {name: len(members) for name, members in lists.items()},
            "outdir": root}


def export(root: str, outdir: str, license_name: str = "CC-BY-4.0",
           splits_from: Optional[str] = None) -> Dict[str, object]:
    """Copy an already-v4 sample tree and finalize it for publication."""
    if os.path.realpath(root) == os.path.realpath(outdir):
        raise ValueError("export source and destination must be different directories")
    metadata_paths = layout.find(root)
    if not metadata_paths:
        raise FileNotFoundError("no schema-v%d samples under %s" % (loader.SCHEMA_VERSION, root))
    target_samples = os.path.join(outdir, "samples")
    if os.path.isdir(target_samples):
        shutil.rmtree(target_samples)
    os.makedirs(target_samples, exist_ok=True)
    for path in metadata_paths:
        source = os.path.dirname(path)
        document = layout.read(source)
        uid = str(layout.identity(document)["uid"])
        destination = os.path.join(outdir, "samples", *uid.split("/"))
        shutil.copytree(source, destination)
    return finalize(outdir, license_name, splits_from)


def upload(outdir: str, repo_id: str, private: bool = False,
           token: Optional[str] = None) -> str:
    """Publish through the current Hugging Face `hf` CLI."""
    import subprocess
    command = ["hf", "upload", repo_id, outdir, ".", "--type", "dataset",
               "--commit-message", "Publish PhysLoc schema v%d" % loader.SCHEMA_VERSION]
    if private:
        command.append("--private")
    environment = os.environ.copy()
    if token:
        environment["HF_TOKEN"] = token
    subprocess.run(command, check=True, env=environment)
    return "https://huggingface.co/datasets/%s" % repo_id
