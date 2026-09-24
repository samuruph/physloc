"""The single PhysLoc schema-v4 writer used by generation and export.

**Every fact is stored once**, at the most specific level where it varies, and
anything `physloc/loader.py` can compute is not stored at all. Before v4 the
same clock windows lived in three places, the severity bin in three, and a
count called `n_violators` disagreed with the violator list it summarised.
What was dropped, and the derivation that replaces it, is in docs/schema.md;
each derivation was checked against every v3 sample before its field went.

`document_from_generation` turns the generator's in-memory `meta` into the
`sample.json` document; `write_sample` writes it beside `data.h5`.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from .. import loader


COMPONENTS = {"none": 0, "body": 1, "shadow": 2, "trajectory": 3,
              "interaction": 4, "energy": 5}
SHADOW_FAMILIES = {"shadow", "shadow_inverted", "shadow_shape"}

#: Summary fields `loader.Violation` derives from the violator records.
DERIVED_SUMMARY = {"t_event_frame", "t_observable_frame", "t_end_frame",
                   "t_intervention_end_frame", "t_consequence_end_frame",
                   "observability_lag_frames", "violation_windows",
                   "intervention_windows", "consequence_windows",
                   "observable_windows", "causal_relations"}
#: Violator fields the loader derives from the violator's own windows.
DERIVED_VIOLATOR = {"t_event_frame", "t_observable_frame", "observability_lag_frames"}
#: Generator window name -> the clock it records.
WINDOW_CLOCKS = {"violation_windows": "active", "intervention_windows": "intervening",
                 "consequence_windows": "consequence", "observable_windows": "observable"}

#: The unit of each energy dataset. Everything used to be tagged "J".
ENERGY_UNITS = {"mass": "kg", "height": "m", "inertia": "kg*m^2",
                "momentum": "kg*m/s", "angular_momentum": "kg*m^2/s",
                "in_frame": None}


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


def dumps(document) -> str:
    """``sample.json`` text: two-space indent, keys sorted, and a list of
    numbers (a window, a vector, a matrix row) kept on one line."""
    def short(value) -> bool:
        return isinstance(value, list) and all(
            isinstance(x, (int, float, bool)) or x is None
            or (isinstance(x, list) and all(isinstance(y, (int, float)) for y in x))
            for x in value)

    def render(value, depth: int) -> str:
        pad, inner = "  " * depth, "  " * (depth + 1)
        if isinstance(value, dict):
            if not value:
                return "{}"
            items = [inner + json.dumps(str(key)) + ": " + render(value[key], depth + 1)
                     for key in sorted(value)]
            return "{\n" + ",\n".join(items) + "\n" + pad + "}"
        if isinstance(value, list) and value and not short(value):
            return ("[\n" + ",\n".join(inner + render(x, depth + 1) for x in value)
                    + "\n" + pad + "]")
        return json.dumps(value, separators=(", ", ": "))
    return render(jsonable(document), 0) + "\n"


def dump(document, path: str) -> None:
    """Atomically write ``sample.json``."""
    temp = path + ".tmp"
    with open(temp, "w", encoding="utf-8") as handle:
        handle.write(dumps(document))
    os.replace(temp, path)


def public_objects(instances) -> Tuple[list, Dict[int, int], Dict[int, int]]:
    """Return public objects, id-to-row and renderer-caster-to-actor remapping.

    The list order is the row order of every ``[N,...]`` array in ``data.h5``.
    """
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
        objects.append(jsonable({
            "id": object_id,
            "name": item.get("name", str(object_id)),
            "category": item.get("category", "unknown"),
            "role": item.get("role", "unknown"),
            "analysis_group": analysis_group(item),
            "asset": {
                "id": item.get("asset_id"), "source": item.get("source"),
                "license": item.get("license"), "held_out": item.get("held_out", False),
            },
            "physics": {
                "mass": item.get("mass"), "friction": item.get("friction"),
                "rolling_friction": item.get("rolling_friction"),
                "restitution": item.get("restitution"),
                "static": bool(item.get("static", False)),
                "collidable": bool(item.get("collides", True)),
                "scripted": bool(item.get("scripted", False)),
                "dormant": bool(item.get("dormant", False)),
                "energy_eligible": bool(item.get("energy_eligible", False)),
            },
            "render": {"material": item.get("material"), "color": item.get("color"),
                       "scale": item.get("scale")},
        }))
    return objects, {int(item["id"]): row for row, item in enumerate(objects)}, remap


def _first(windows) -> Optional[int]:
    return int(windows[0][0]) if windows else None


def _last(windows) -> Optional[int]:
    return int(windows[-1][1]) if windows else None


def _violator(record: Dict, remap: Dict[int, int], magnitude, T: int) -> Dict:
    """One generator violator record -> its v4 form, checked against the
    derivations the loader will apply to it."""
    object_id = remap.get(int(record["instance_id"]), int(record["instance_id"]))
    windows = {clock: [[int(a), int(b)] for a, b in record.get(name) or []]
               for name, clock in WINDOW_CLOCKS.items()}
    t_event = _first(windows["active"])
    later = [f for a, b in windows["observable"] for f in range(a, b + 1)
             if t_event is not None and f >= t_event]
    t_observable = later[0] if later else t_event
    for key, derived in (("t_event_frame", t_event),
                         ("t_observable_frame", t_observable),
                         ("observability_lag_frames",
                          None if t_event is None else t_observable - t_event)):
        if key in record and record[key] != derived:
            raise ValueError("violator %d: %s=%r but its windows give %r"
                             % (object_id, key, record[key], derived))
    # A remapped shadow caster is the actor itself, and a violator is not a
    # body its own violation disturbed.
    affected = sorted({remap.get(int(i), int(i))
                       for i in record.get("affected_instance_ids") or []} - {object_id})
    out = {"id": object_id, "affected_ids": affected, "windows": windows}
    for key, value in record.items():
        if key in DERIVED_VIOLATOR or key in WINDOW_CLOCKS or key in (
                "instance_id", "affected_instance_ids"):
            continue
        if key == "magnitude" and value == magnitude:
            continue            # the plan's magnitude; kept only where it differs
        out[key] = value
    return out


def _violation(meta: Dict, objects: List[Dict], remap: Dict[int, int],
               T: int) -> Optional[Dict]:
    """The ``violation`` block of an invalid sample, or None."""
    summary = jsonable(meta.get("violation"))
    if not summary:
        return None
    family = str((meta.get("metadata") or {}).get("family"))
    intervention = dict(summary.get("intervention") or {})
    severity_bin = intervention.pop("severity_bin", None)
    if "magnitude_unit" in intervention:
        intervention["unit"] = intervention.pop("magnitude_unit")
    violators = [_violator(record, remap, intervention.get("magnitude"), T)
                 for record in summary.get("violators") or []]
    # The clip-level times the loader derives must be the ones the generator
    # measured -- otherwise dropping them changes the data.
    union = {clock: loader.union_intervals([v["windows"][clock] for v in violators], T)
             for clock in ("active", "intervening", "consequence", "observable")}
    for key, derived in (("t_event_frame", _first(union["active"])),
                         ("t_end_frame", _last(union["active"])),
                         ("t_observable_frame", _first(union["observable"])),
                         ("t_intervention_end_frame", _last(union["intervening"])),
                         ("t_consequence_end_frame", _last(union["consequence"]))):
        if key in summary and violators and summary[key] != derived:
            raise ValueError("%s=%r but the violator windows give %r"
                             % (key, summary[key], derived))
    difficulty = meta.get("difficulty")
    if difficulty is not None:
        difficulty = dict(jsonable(difficulty))
        if summary.get("difficulty_inputs") is not None:
            difficulty["inputs"] = summary["difficulty_inputs"]
    out = {
        "severity_bin": severity_bin,
        "kind": summary.get("kind"),
        "component": "shadow" if family in SHADOW_FAMILIES else "body",
        "spatial_extent": summary.get("spatial_extent"),
        "timing": summary.get("violator_timing"),
        "intervention": intervention,
        "difficulty": difficulty,
        "causal_ids": sorted({remap.get(int(i), int(i))
                              for i in summary.get("causal_body_ids") or []}),
        "violators": violators,
    }
    renamed = {"kind", "spatial_extent", "violator_timing", "intervention",
               "difficulty_inputs", "causal_body_ids", "violators"}
    for key, value in summary.items():
        if key not in renamed and key not in DERIVED_SUMMARY:
            out[key] = value
    if out["component"] == "shadow":
        actors = [o["id"] for o in objects if o["analysis_group"] == "subject"]
        out["shadow"] = {
            "caster_object_id": actors[0] if actors else None,
            "light_id": "key",
            "receiver_object_ids": [o["id"] for o in objects
                                    if o["analysis_group"] == "support"],
            "rendering_method": "Cycles matched shadow-isolation render",
            "strength_threshold": loader.SHADOW_STRENGTH_THRESHOLD,
        }
    return out


def document_from_generation(meta: Dict, split: str = "unassigned"):
    """Build the v4 document from generator state held in memory.

    Returns ``(document, rows, remap, arrays)``: ``rows`` maps an object id to
    its row, ``remap`` the renderer's shadow caster to its actor, and
    ``arrays`` the per-frame blocks (camera track, collisions) that belong in
    ``data.h5`` rather than JSON -- pass it to `write_sample`.
    """
    old = meta["metadata"]
    objects, rows, remap = public_objects(meta.get("instances") or [])
    T = int(old.get("num_frames") or 0)
    complexity = old.get("complexity")
    level = complexity.get("name") if isinstance(complexity, dict) else complexity
    label = str(old["label"])
    camera = dict(meta.get("camera") or {})
    arrays = {"camera": {key: np.asarray(camera.pop(key), np.float64)
                         for key in ("positions", "quaternions") if key in camera},
              "collisions": _collision_columns(
                  (meta.get("events") or {}).get("collisions") or [], remap)}
    step_rate = old.get("step_rate")
    fps = float(old.get("frame_rate") or 0)
    provenance = dict(meta.get("provenance") or {})
    if provenance.get("render_seed") == old.get("seed"):
        provenance.pop("render_seed")
    background = old.get("background") or {}
    document = {
        "schema_version": loader.SCHEMA_VERSION,
        "sample": {
            "uid": str(old["sample_uid"]),
            "pair_uid": str(old["pair_uid"]),
            "valid_uid": str(old.get("valid_sample_uid") or (
                old["sample_uid"] if label == "valid" else old["pair_uid"] + "/valid")),
            "label": label,
            "split": split,
            "release": str(old.get("release") or ""),
            "tier": old.get("tier"),
            "seed": old.get("seed"),
            "variant": int(old.get("variant") or 0),
            "framing_attempt": int(old.get("framing_attempt") or 0),
        },
        "video": {
            "num_frames": T, "fps": fps, "resolution": old.get("resolution"),
            "latent": {"frames": old.get("latent_frames"), "hw": old.get("latent_hw")},
        },
        "scene": {
            "scenario": old.get("scenario"), "family": old.get("family"),
            "domain": old.get("domain"), "physics_medium": old.get("physics_medium"),
            "level": level, "condition": old.get("condition"),
            "prompt": old.get("prompt"), "size_scale": old.get("size_scale", 1.0),
            "camera": camera,
            "environment": {
                "hdri": background.get("hdri_id"), "background": background.get("color"),
                "lighting": (meta.get("scene") or {}).get("lights") or old.get("lights") or [],
            },
            "physics": {
                "engine": "PyBullet", "gravity": old.get("gravity"),
                "step_rate": step_rate,
                "substeps_per_frame": (int(round(float(step_rate) / max(fps, 1.0)))
                                       if step_rate else None),
                "params": old.get("params") or {},
            },
        },
        "objects": objects,
        "provenance": provenance,
    }
    violation = _violation(meta, objects, remap, T)
    if violation is not None:
        document["violation"] = violation
    return jsonable(document), rows, remap, arrays


def _collision_columns(records: List[Dict], remap: Dict[int, int]) -> Dict[str, np.ndarray]:
    """MOVi's collision list as one array per field, ids made public."""
    def column(key, width, dtype):
        values = [record.get(key) for record in records]
        shape = (len(values),) + ((width,) if width else ())
        out = np.full(shape, np.nan if np.dtype(dtype).kind == "f" else 0, dtype)
        for i, value in enumerate(values):
            if value is not None:
                out[i] = value
        return out
    columns = {"frame": column("frame", 0, np.int32),
               "instances": column("instances", 2, np.int32),
               "force": column("force", 0, np.float32),
               "position": column("position", 3, np.float32),
               "contact_normal": column("contact_normal", 3, np.float32),
               "image_position": column("image_position", 2, np.float32)}
    for old, new in remap.items():
        columns["instances"][columns["instances"] == old] = new
    return columns


def _axes(prefix: str, array: np.ndarray) -> str:
    """``"N,T"`` for an ``[N,T,8,3]`` array -> ``"N,T,8,3"``: named leading
    axes, then the size of each trailing one."""
    named = prefix.split(",")[:array.ndim]
    return ",".join(named + [str(n) for n in array.shape[len(named):]])


def _dataset(group, name: str, value, axes: Optional[str] = None,
             units: Optional[str] = None):
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


def _record_clocks(document: Dict, violation_objects: Dict[str, np.ndarray],
                   rows: Dict[int, int], T: int) -> None:
    """Put each violator's `occluded` clock into its windows, and check that
    the dense clocks the pipeline computed are exactly what the windows say --
    the loader rebuilds them from the windows alone."""
    violation = document.get("violation")
    if not violation:
        return
    source = [int(i) for i in violation_objects.get("ids", [])]
    source_row = {object_id: row for row, object_id in enumerate(source)}
    violator_ids = set()
    for record in violation["violators"]:
        row = source_row.get(record["id"])
        if row is None:
            raise ValueError("violator %d has no row in the violation arrays" % record["id"])
        violator_ids.add(record["id"])
        record["windows"]["occluded"] = loader.mask_to_intervals(
            np.asarray(violation_objects["occluded"][row], bool)[:T])
        for clock in ("active", "intervening", "consequence", "observable"):
            dense = np.asarray(violation_objects[clock][row], bool)[:T]
            if not np.array_equal(dense, loader.intervals_to_mask(
                    record["windows"][clock], T)):
                raise ValueError("violator %d: %s clock disagrees with its windows"
                                 % (record["id"], clock))
    flagged = {object_id for object_id, row in source_row.items()
               if bool(np.asarray(violation_objects.get("is_violator"))[row])}
    if flagged != violator_ids:
        raise ValueError("is_violator %s disagrees with the violator records %s"
                         % (sorted(flagged), sorted(violator_ids)))
    if "affected" in violation_objects:
        derived = np.zeros((len(source), T), bool)
        for record in violation["violators"]:
            frames = loader.intervals_to_mask(record["windows"]["consequence"], T)
            for target in record["affected_ids"]:
                if target in source_row:
                    derived[source_row[target]] |= frames
        if not np.array_equal(derived, np.asarray(violation_objects["affected"], bool)[:, :T]):
            raise ValueError("affected clock disagrees with the violator records")


def write_sample(root: str, document: Dict, observations: Dict[str, np.ndarray],
                 instance_arrays: Dict[str, np.ndarray],
                 energy: Dict[str, np.ndarray], energy_map: np.ndarray,
                 violation_maps: Dict[str, np.ndarray],
                 violation_objects: Dict[str, np.ndarray],
                 arrays: Optional[Dict[str, Dict[str, np.ndarray]]] = None) -> str:
    """Atomically replace one generated sample's JSON and HDF5 payload.

    ``violation_maps`` uses the generator's names (``violation_object_id``,
    ``violation_component``, ``causal_level``, ``causal_source_id``,
    ``severity``); ``violation_objects`` holds ``ids``, ``is_violator``, the
    five clocks, ``affected``, ``severity``, ``residual`` and ``score``. Only
    what the loader cannot rebuild is stored, and nothing at all under
    ``/violation`` for a valid sample.
    """
    try:
        import h5py
    except ImportError as exc:
        raise ImportError("writing PhysLoc schema v4 requires h5py") from exc
    arrays = arrays or {}
    info = document["sample"]
    uid = str(info["uid"])
    sample_dir = os.path.join(root, "samples", *uid.split("/"))
    os.makedirs(sample_dir, exist_ok=True)
    public_ids = [int(item["id"]) for item in document["objects"]]
    rows = {value: row for row, value in enumerate(public_ids)}
    T = int(document["video"]["num_frames"])
    invalid = "violation" in document
    if invalid:
        _record_clocks(document, violation_objects, rows, T)

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

        camera = h5.create_group("camera")
        for key, value in (arrays.get("camera") or {}).items():
            _dataset(camera, key, value, "T,3" if key == "positions" else "T,4",
                     "m" if key == "positions" else None)

        obj_group = h5.create_group("objects")
        _dataset(obj_group, "ids", np.asarray(public_ids, np.int32), "N")
        source_ids = np.asarray(instance_arrays.get("ids", []), int)
        for key, value in _align_first(instance_arrays, source_ids,
                                       public_ids, rows).items():
            _dataset(obj_group, key, value, _axes("N,T", value),
                     "m" if key == "positions" else (
                         "m/s" if key == "velocities" else (
                             "rad/s" if key == "angular_velocities" else None)))

        scene_energy = h5.create_group("energy/scene")
        object_energy = h5.create_group("energy/objects")
        energy_ids = np.asarray(energy.get("body_ids", []), int)
        for key, value in energy.items():
            array = np.asarray(value)
            if key in ("body_ids", "momentum_magnitude"):
                continue            # the object axis is /objects/ids; |momentum| derives
            units = ENERGY_UNITS.get(key, "J")
            if array.ndim >= 2 and array.shape[1] == len(energy_ids):
                target = np.zeros((len(public_ids), array.shape[0]) + array.shape[2:],
                                  array.dtype)
                for column, object_id in enumerate(energy_ids):
                    if int(object_id) in rows:
                        target[rows[int(object_id)]] = array[:, column]
                _dataset(object_energy, key, target, _axes("N,T", target), units)
            else:
                _dataset(scene_energy, key, array,
                         "T" if array.ndim == 1 and len(array) == T else None, units)
        _dataset(h5["energy"], "map", energy_map, "T,H,W", "J")

        events = h5.create_group("events/collisions")
        for key, value in (arrays.get("collisions") or {}).items():
            _dataset(events, key, value, _axes("E", value),
                     {"force": "N", "position": "m"}.get(key))

        if invalid:
            maps = h5.create_group("violation/maps")
            component = np.asarray(violation_maps["violation_component"])
            for source, target in (("violation_object_id", "object_id"),
                                   ("violation_component", "component"),
                                   ("causal_level", "causal_level"),
                                   ("causal_source_id", "causal_source_id")):
                _dataset(maps, target, violation_maps[source], "T,H,W")
            # Painting reproduces the severity map exactly for a body; for a
            # shadow component it does not, so only there is the map stored.
            if "severity" in violation_maps and np.any(component == COMPONENTS["shadow"]):
                _dataset(maps, "severity", violation_maps["severity"], "T,H,W")
            per_object = h5.create_group("violation/objects")
            source = np.asarray(violation_objects.get("ids", []), int)
            for key in ("severity", "residual", "score"):
                if key not in violation_objects:
                    continue
                array = np.asarray(violation_objects[key], np.float32)
                target = np.zeros((len(public_ids),) + array.shape[1:], np.float32)
                for row, object_id in enumerate(source):
                    if int(object_id) in rows and row < len(array):
                        target[rows[int(object_id)]] = array[row]
                _dataset(per_object, key, target, "N,T")
    os.replace(temp, os.path.join(sample_dir, loader.DATA))
    dump(document, os.path.join(sample_dir, loader.SAMPLE_METADATA))
    return sample_dir
