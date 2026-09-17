"""The single PhysLoc schema-v3 writer used by generation and export."""
from __future__ import annotations

import json
import os
from typing import Dict, Optional, Tuple

import numpy as np

from .. import loader


COMPONENTS = {"none": 0, "body": 1, "shadow": 2, "trajectory": 3,
              "interaction": 4, "energy": 5}
SHADOW_FAMILIES = {"shadow", "shadow_inverted", "shadow_shape"}


def analysis_group(instance: Dict) -> str:
    # The experiment contract is stronger than a scenario author's semantic
    # role name: anything selected as a violator is always a subject.
    if instance.get("is_violator"):
        return "subject"
    declared = instance.get("analysis_group")
    if declared in {"subject", "context", "support", "background"}:
        return str(declared)
    role = str(instance.get("role") or "").lower()
    if role in {"actor", "violator", "target"}:
        return "subject"
    if role in {"floor", "support", "table", "ramp", "barrier", "wall"}:
        return "support"
    if role in {"background", "backdrop", "dome", "sky"}:
        return "background"
    return "context"


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def public_objects(instances) -> Tuple[list, Dict[int, int], Dict[int, int]]:
    """Return public objects, id-to-row and renderer-caster-to-actor remapping."""
    caster = next((item for item in instances
                   if item.get("role") == "shadow_caster"), None)
    actor = next((item for item in instances
                  if item.get("role") == "actor" and not item.get("dormant")), None)
    remap = ({int(caster["id"]): int(actor["id"])}
             if caster is not None and actor is not None else {})
    objects = []
    for item in instances:
        if item.get("role") == "shadow_caster":
            continue
        object_id = int(item["id"])
        value = {
            "id": object_id,
            "name": item.get("name", str(object_id)),
            "category": item.get("category", "unknown"),
            "role": item.get("role", "unknown"),
            "analysis_group": analysis_group(item),
            "asset": {
                "id": item.get("asset_id"), "source": item.get("source"),
                "license": item.get("license"), "held_out": item.get("held_out", False),
            },
            "static_fields": {
                "mass": item.get("mass"), "dimensions": item.get("scale"),
                "draw_scale": item.get("scale"), "material": item.get("material"),
                "base_color": item.get("color"), "friction": item.get("friction"),
                "rolling_friction": item.get("rolling_friction"),
                "restitution": item.get("restitution"),
                "static": bool(item.get("static", False)),
                "energy_eligible": bool(item.get("energy_eligible", False)),
                "scripted": bool(item.get("scripted", False)),
                "dormant": bool(item.get("dormant", False)),
                "collidable": bool(item.get("collides", True)),
                "camera_visible": True, "shadow_visible": True,
            },
            "temporal_info": {"store": loader.DATA, "group": "/objects",
                              "row": len(objects)},
            "energy": {"store": loader.DATA, "group": "/energy/objects",
                       "row": len(objects)},
            "violation": {"store": loader.DATA, "group": "/violations/objects",
                          "row": len(objects)},
        }
        objects.append(jsonable(value))
    return objects, {int(item["id"]): row for row, item in enumerate(objects)}, remap


def _remap_summary(summary, remap: Dict[int, int]):
    value = json.loads(json.dumps(jsonable(summary))) if summary else None
    if value is None or not remap:
        return value
    value["causal_body_ids"] = [remap.get(int(i), int(i))
                                for i in value.get("causal_body_ids") or []]
    for record in value.get("violators") or []:
        object_id = int(record.get("instance_id", 0))
        record["instance_id"] = remap.get(object_id, object_id)
        record["affected_instance_ids"] = [remap.get(int(i), int(i))
                                             for i in record.get("affected_instance_ids") or []]
    return value


def document_from_generation(meta: Dict, split: str = "unassigned"):
    """Build the v3 JSON document from generator state held in memory."""
    old = meta["metadata"]
    objects, rows, remap = public_objects(meta.get("instances") or [])
    valid_uid = str(old.get("valid_sample_uid") or (
        old["sample_uid"] if old["label"] == "valid"
        else old["pair_uid"] + "/valid"))
    complexity = old.get("complexity")
    level = complexity.get("name") if isinstance(complexity, dict) else complexity
    summary = _remap_summary(meta.get("violation"), remap)
    if summary is not None and str(old.get("family")) in SHADOW_FAMILIES:
        actor_ids = [int(item["id"]) for item in objects
                     if item.get("analysis_group") == "subject"]
        receiver_ids = [int(item["id"]) for item in objects
                        if item.get("analysis_group") == "support"]
        summary["target_component"] = "shadow"
        for record in summary.get("violators") or []:
            record["target_component"] = "shadow"
        summary["shadow"] = {
            "caster_object_id": actor_ids[0] if actor_ids else None,
            "light_id": "key",
            "receiver_object_ids": receiver_ids,
            "rendering_method": "Cycles matched shadow-isolation render",
            "strength_threshold": 1.0 / 255.0,
        }
    violators = {int(record.get("instance_id", -1)): record
                 for record in (summary or {}).get("violators") or []}
    for item in objects:
        record = violators.get(int(item["id"]))
        if record is not None:
            item["violation"].update({
                "type": (summary or {}).get("kind"),
                "component": record.get("target_component",
                                        (summary or {}).get("target_component", "body")),
                "severity": (summary or {}).get("intervention", {}).get("severity_bin"),
                "active_intervals": record.get("violation_windows") or [],
                "intervention_intervals": record.get("intervention_windows") or [],
                "consequence_intervals": record.get("consequence_windows") or [],
                "observable_intervals": record.get("observable_windows") or [],
                "affected_object_ids": record.get("affected_instance_ids") or [],
            })
    affected_ids = {int(value) for record in (summary or {}).get("violators") or []
                    for value in record.get("affected_instance_ids") or []}
    counts = {
        "n_objects": len(objects),
        "n_subjects": sum(o["analysis_group"] == "subject" for o in objects),
        "n_context": sum(o["analysis_group"] == "context" for o in objects),
        "n_support": sum(o["analysis_group"] == "support" for o in objects),
        "n_background": sum(o["analysis_group"] == "background" for o in objects),
        "n_actors": int(old.get("n_actors") or 0),
        "n_distractors": int(old.get("n_distractors") or 0),
        "n_violators": int(old.get("n_violators") or 0),
        "n_affected": len(affected_ids),
    }
    sample_info = {
        "schema_version": loader.SCHEMA_VERSION,
        "dataset_version": str(old.get("release") or ""),
        "sample_uid": str(old["sample_uid"]),
        "pair_uid": str(old["pair_uid"]),
        "valid_sample_uid": valid_uid,
        "label": str(old["label"]),
        "split": split,
        "seed": old.get("seed"),
        "render_seed": old.get("render_seed", old.get("seed")),
        "variant": old.get("variant", 0),
        "generation_config_id": "%s:%s" % (old.get("release", "dataset"),
                                               old.get("tier", "default")),
        "num_frames": int(old.get("num_frames") or 0),
        "fps": float(old.get("frame_rate") or 0),
        "duration_seconds": (float(old.get("num_frames") or 0)
                             / max(float(old.get("frame_rate") or 1), 1.0)),
        "resolution": old.get("resolution"),
        "latent_frames": old.get("latent_frames"),
        "latent_hw": old.get("latent_hw"),
        "framing_attempt": old.get("framing_attempt", 0),
        "size_scale": old.get("size_scale", 1.0),
        "provenance": meta.get("provenance") or {},
    }
    scene_info = {
        "level": level, "type": old.get("scenario"),
        "family": old.get("family"), "domain": old.get("domain"),
        "physics_medium": old.get("physics_medium"),
        "condition": old.get("condition"), "complexity": level,
        "difficulty": (meta.get("difficulty") or {}).get("level"),
        "difficulty_analysis": meta.get("difficulty"),
        "severity": (summary or {}).get("intervention", {}).get("severity_bin"),
        "label": old.get("label"), "variant": old.get("variant", 0),
        "prompt": old.get("prompt"),
    }
    background = old.get("background") or {}
    world = {
        "camera": meta.get("camera") or {},
        "environment": {
            "background": background.get("color"), "hdri": background.get("hdri_id"),
            "lighting": (meta.get("scene") or {}).get("lights") or old.get("lights") or [],
            "floor_datum": {"z": 0.0, "units": "m"},
        },
        "physics": {
            "gravity": old.get("gravity"), "step_rate": old.get("step_rate"),
            "timestep_seconds": (1.0 / float(old["step_rate"])
                                 if old.get("step_rate") else None),
            "substeps_per_frame": (int(round(float(old.get("step_rate"))
                                              / max(float(old.get("frame_rate") or 1), 1)))
                                   if old.get("step_rate") else None),
            "solver": {"engine": "PyBullet", "settings": old.get("params") or {}},
            "units": {"length": "m", "time": "s", "mass": "kg", "energy": "J"},
        },
        "objects_summary": counts,
    }
    relations = []
    for record in (summary or {}).get("violators") or []:
        source_id = int(record.get("instance_id", 0))
        for target_id in record.get("affected_instance_ids") or []:
            relations.append({
                "source_object_id": source_id,
                "target_object_id": int(target_id),
                "relation_type": "physical_consequence",
                "intervals": record.get("consequence_windows") or [],
                "confidence": 1.0,
            })
    annotations = {
        "objects": objects,
        "scene_energy": {"store": loader.DATA, "group": "/energy/scene"},
        "maps": {"store": loader.DATA, "group": "/violations/maps"},
        "events": meta.get("events") or [],
        "causal_relations": ((summary or {}).get("causal_relations") or relations),
    }
    if summary is not None:
        annotations["violation_summary"] = summary
    document = {
        "schema_version": loader.SCHEMA_VERSION,
        "metadata": {"sample_info": sample_info,
                     "scene": {"info": scene_info, "world": world}},
        "observations": {
            "rgb": {"path": loader.RGB, "format": "mp4", "decode": "explicit"},
            "dense": {"store": loader.DATA, "group": "/observations"},
        },
        "annotations": annotations,
        "storage": {"dense_store": loader.DATA, "format": "HDF5",
                    "compression": "gzip-4", "checksum": "fletcher32"},
    }
    return document, rows, remap


def _dataset(group, name: str, value, axes: Optional[str] = None,
             units: Optional[str] = None, encoding: Optional[str] = None):
    array = np.asarray(value)
    if array.dtype.kind == "O":
        return None
    options = {}
    if array.ndim and array.size:
        options.update(compression="gzip", compression_opts=4, shuffle=True,
                       fletcher32=True)
        if axes and axes.startswith("N,T") and array.ndim >= 2:
            options["chunks"] = (1, min(32, array.shape[1])) + tuple(array.shape[2:])
        else:
            options["chunks"] = (1,) + tuple(array.shape[1:])
    target = group.create_dataset(name, data=array, **options)
    if axes:
        target.attrs["axes"] = axes
    if units:
        target.attrs["units"] = units
    if encoding:
        target.attrs["encoding"] = encoding
    target.attrs["fill_value"] = "NaN" if array.dtype.kind == "f" else 0
    return target


def _align_first(values: Dict[str, np.ndarray], source_ids, public_ids, rows):
    aligned = {}
    source_ids = [int(value) for value in source_ids]
    lookup = {value: i for i, value in enumerate(source_ids)}
    for key, value in values.items():
        array = np.asarray(value)
        if key == "ids" or not array.ndim or array.shape[0] != len(source_ids):
            continue
        fill = np.nan if array.dtype.kind == "f" else 0
        target = np.full((len(public_ids),) + array.shape[1:], fill, array.dtype)
        for object_id in public_ids:
            if object_id in lookup:
                target[rows[object_id]] = array[lookup[object_id]]
        aligned[key] = target
    return aligned


def write_sample(root: str, document: Dict, observations: Dict[str, np.ndarray],
                 instance_arrays: Dict[str, np.ndarray],
                 energy: Dict[str, np.ndarray], energy_map: np.ndarray,
                 violation_maps: Dict[str, np.ndarray],
                 violation_objects: Dict[str, np.ndarray]) -> str:
    """Atomically replace one generated sample's JSON and HDF5 payload."""
    try:
        import h5py
    except ImportError as exc:
        raise ImportError("writing PhysLoc schema v3 requires h5py") from exc
    info = document["metadata"]["sample_info"]
    uid = str(info["sample_uid"])
    sample_dir = os.path.join(root, "samples", *uid.split("/"))
    os.makedirs(sample_dir, exist_ok=True)
    objects = document["annotations"]["objects"]
    public_ids = [int(item["id"]) for item in objects]
    rows = {value: row for row, value in enumerate(public_ids)}
    T = int(info["num_frames"])

    temp = os.path.join(sample_dir, loader.DATA + ".tmp")
    with h5py.File(temp, "w") as h5:
        h5.attrs["schema_version"] = loader.SCHEMA_VERSION
        h5.attrs["coordinate_system"] = "right-handed; +Z up; metres"
        obs_group = h5.create_group("observations")
        axes = {"segmentation": "T,H,W", "depth": "T,H,W,C",
                "forward_flow": "T,H,W,2", "backward_flow": "T,H,W,2",
                "normal": "T,H,W,3", "object_coordinates": "T,H,W,3",
                "shadow_strength": "T,H,W", "shadow_source_id": "T,H,W"}
        for key, value in observations.items():
            _dataset(obs_group, key, value, axes.get(key),
                     "m" if key == "depth" else ("pixels" if "flow" in key else None))

        obj_group = h5.create_group("objects")
        _dataset(obj_group, "ids", np.asarray(public_ids, np.int32), "N")
        source_ids = np.asarray(instance_arrays.get("ids", []), int)
        aligned = _align_first(instance_arrays, source_ids, public_ids, rows)
        for key, value in aligned.items():
            _dataset(obj_group, key, value, "N,T" if value.ndim >= 2 else "N")

        scene_energy = h5.create_group("energy/scene")
        object_energy = h5.create_group("energy/objects")
        energy_ids = np.asarray(energy.get("body_ids", []), int)
        for key, value in energy.items():
            array = np.asarray(value)
            if key == "body_ids":
                continue
            if array.ndim >= 2 and array.shape[1] == len(energy_ids):
                target = np.zeros((len(public_ids), array.shape[0]) + array.shape[2:],
                                  array.dtype)
                for column, object_id in enumerate(energy_ids):
                    if int(object_id) in rows:
                        target[rows[int(object_id)]] = array[:, column]
                _dataset(object_energy, key, target,
                         "N,T" + (",D" if target.ndim > 2 else ""), "J")
            else:
                _dataset(scene_energy, key, array,
                         "T" if array.ndim == 1 and len(array) == T else None, "J")
        _dataset(h5["energy"], "map", energy_map, "T,H,W", "J")

        maps = h5.create_group("violations/maps")
        for key, value in violation_maps.items():
            _dataset(maps, key, value, "T,H,W")
        object_violations = h5.create_group("violations/objects")
        source = np.asarray(violation_objects.get("ids", []), int)
        for key, value in violation_objects.items():
            if key == "ids":
                continue
            array = np.asarray(value)
            dtype = bool if key in loader.CLOCKS + ("affected", "is_violator") else array.dtype
            target = np.zeros((len(public_ids),) + array.shape[1:], dtype)
            for row, object_id in enumerate(source):
                if int(object_id) in rows and row < len(array):
                    target[rows[int(object_id)]] = array[row]
            _dataset(object_violations, key, target,
                     "N" if target.ndim == 1 else "N,T")
        if "is_violator" not in object_violations:
            _dataset(object_violations, "is_violator",
                     np.zeros(len(public_ids), bool), "N")
    os.replace(temp, os.path.join(sample_dir, loader.DATA))
    with open(os.path.join(sample_dir, loader.SAMPLE_METADATA), "w", encoding="utf-8") as handle:
        json.dump(jsonable(document), handle, indent=2, sort_keys=True)
    return sample_dir
