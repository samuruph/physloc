"""Standalone reader for the PhysLoc schema-v3 sample format.

A release has one public representation::

    dataset.json  schema.json  index.parquet  splits/
    samples/<sample_uid>/sample.json  rgb.mp4  data.h5

RGB is returned as a path unless decoding is explicitly requested. Dense
arrays are read lazily from the sample's chunked HDF5 store. This module has no
imports from :mod:`physloc`, so the exact file can ship with a Hub dataset.
"""
from __future__ import annotations

import glob
import json
import os
import sys
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence

import numpy as np

SCHEMA_VERSION = 3
SAMPLE_METADATA = "sample.json"
DATA = "data.h5"
RGB = "rgb.mp4"
PASSES = ("depth", "forward_flow", "backward_flow", "normal",
          "object_coordinates", "shadow_strength", "shadow_source_id")
CLOCKS = ("active", "intervening", "consequence", "observable", "occluded")
DEPTH_BACKGROUND = 1e6

METADATA_FIELDS = ("metadata",)
LOCALIZATION_FIELDS = (
    "observations.segmentation",
    "annotations.maps.violation_object_id",
    "annotations.maps.violation_component",
    "annotations.maps.causal_level",
    "annotations.maps.causal_source_id",
)
OBJECT_LOCALIZATION_FIELDS = LOCALIZATION_FIELDS + (
    "annotations.objects.ids", "annotations.objects.analysis_group",
    "annotations.objects.is_violator", "annotations.objects.active",
    "annotations.objects.observable", "annotations.objects.severity",
)
ENERGY_FIELDS = ("annotations.scene_energy", "annotations.objects.energy")
VISUALIZATION_FIELDS = (
    "observations.rgb_path", "observations.segmentation",
) + LOCALIZATION_FIELDS[1:] + ENERGY_FIELDS
FIELDS: Dict[str, str] = {
    "metadata": "sample metadata and scene description",
    "observations.rgb_path": "path to the standalone MP4",
    "observations.rgb": "uint8 [T,H,W,3], explicitly decoded",
    "observations.segmentation": "uint16 [T,H,W], 0 is background",
    "annotations.maps.violation_object_id": "uint16 [T,H,W]",
    "annotations.maps.violation_component": "uint8 [T,H,W]",
    "annotations.maps.causal_level": "uint8 [T,H,W]",
    "annotations.maps.causal_source_id": "uint16 [T,H,W]",
}


def visible_violation(violation: np.ndarray, segmentation: np.ndarray,
                      component: Optional[np.ndarray] = None,
                      shadow_source_id: Optional[np.ndarray] = None) -> np.ndarray:
    if component is None:
        return (violation > 0) & (segmentation == violation)
    body = (component != 2) & (segmentation == violation)
    shadow = component == 2
    if shadow_source_id is not None:
        shadow &= shadow_source_id == violation
    return (violation > 0) & (body | shadow)


def paint_severity(violation: np.ndarray, segmentation: np.ndarray,
                   ids: Sequence[int], severity: np.ndarray,
                   component: Optional[np.ndarray] = None) -> np.ndarray:
    out = np.zeros(violation.shape, np.float32)
    for row, object_id in enumerate(ids):
        owned = violation == int(object_id)
        visible = owned & ((segmentation == int(object_id)) if component is None
                           else ((segmentation == int(object_id)) | (component == 2)))
        has_pixels = visible.reshape(len(visible), -1).any(axis=1)
        where = np.where(has_pixels[:, None, None], visible, owned)
        out = np.where(where, np.asarray(severity[row], np.float32)[:, None, None], out)
    return out


def reference_mask(valid_segmentation: np.ndarray, ids: Sequence[int]) -> np.ndarray:
    return np.isin(valid_segmentation,
                   np.asarray(list(ids), dtype=valid_segmentation.dtype))


def sample_timeline(objects: Dict[str, np.ndarray], num_frames: int) -> Dict[str, np.ndarray]:
    n = len(objects.get("ids", ()))
    is_violator = np.asarray(objects.get("is_violator", np.ones(n, bool)), bool)
    out: Dict[str, np.ndarray] = {}
    for key in CLOCKS:
        rows = np.asarray(objects.get(key, np.zeros((n, num_frames), bool)), bool)
        selected = rows[is_violator]
        out[key] = (selected.any(axis=0) if len(selected)
                    else np.zeros(num_frames, bool))
    severity = np.asarray(objects.get(
        "severity", np.zeros((n, num_frames), np.float32)), np.float32)[is_violator]
    out["severity"] = (severity.max(axis=0) if len(severity)
                       else np.zeros(num_frames, np.float32))
    return out


def temporal_bins(num_frames: int, latent_frames: int) -> List[np.ndarray]:
    if (num_frames - 1) % 4 or (num_frames - 1) // 4 + 1 != latent_frames:
        raise ValueError("num_frames=%d does not bin to %d latent frames (need 4k+1)"
                         % (num_frames, latent_frames))
    return [np.array([0])] + [np.arange(4 * i - 3, 4 * i + 1)
                              for i in range(1, latent_frames)]


def _block(value: np.ndarray, side: int, reduction: str) -> np.ndarray:
    height, width = value.shape
    if height % side or width % side:
        raise ValueError("%dx%d does not divide into %d" % (height, width, side))
    blocks = value.reshape(side, height // side, side, width // side)
    return (blocks.max(axis=(1, 3)) if reduction == "max"
            else blocks.mean(axis=(1, 3)))


def latent_grid(mask: np.ndarray, severity: np.ndarray, latent_frames: int,
                latent_hw: int) -> Dict[str, np.ndarray]:
    bins = temporal_bins(mask.shape[0], latent_frames)
    shape = (latent_frames, latent_hw, latent_hw)
    mask_grid = np.zeros(shape, bool)
    severity_max = np.zeros(shape, np.float32)
    severity_mean = np.zeros(shape, np.float32)
    for i, source in enumerate(bins):
        mask_grid[i] = _block(mask[source].any(axis=0).astype(np.float32),
                              latent_hw, "max") > 0
        severity_max[i] = _block(severity[source].max(axis=0), latent_hw, "max")
        severity_mean[i] = _block(severity[source].mean(axis=0), latent_hw, "mean")
    return {"mask": mask_grid, "severity_max": severity_max,
            "severity_mean": severity_mean}


def divergence(video_valid: np.ndarray, video_invalid: np.ndarray) -> np.ndarray:
    return (np.abs(np.asarray(video_valid, np.float32)
                   - np.asarray(video_invalid, np.float32)).mean(axis=-1) / 255.0)


def _decode_video(path: str) -> np.ndarray:
    try:
        import imageio.v3 as iio
    except ImportError:
        iio = None
    if iio is not None:
        return np.asarray(iio.imread(path)[..., :3], np.uint8)
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("decoding RGB needs imageio or opencv-python") from exc
    capture = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    if not frames:
        raise IOError("could not decode %s" % path)
    return np.stack(frames).astype(np.uint8)


class _GroupView:
    def __init__(self, sample: "Sample", prefix: str, static: Optional[Dict] = None):
        self.sample = sample
        self.prefix = prefix.strip("/")
        self.static = dict(static or {})

    def _path(self, key: str) -> str:
        return "/" + "/".join(x for x in (self.prefix, key) if x)

    def __getitem__(self, key: str) -> Any:
        return self.static[key] if key in self.static else self.sample._h5_read(self._path(key))

    def get(self, key: str, default=None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def __contains__(self, key: str) -> bool:
        return key in self.static or self.sample._h5_has(self._path(key))

    def keys(self) -> List[str]:
        return sorted(set(self.static) | set(self.sample._h5_keys("/" + self.prefix)))

    def items(self):
        return [(key, self[key]) for key in self.keys()]

    def as_dict(self) -> Dict[str, Any]:
        return dict(self.items())


class Sample:
    """One sample with lazy, process-local HDF5 access."""

    def __init__(self, path: str, twin: Optional["Sample"] = None,
                 selected_fields: Optional[Sequence[str]] = None):
        self.path = os.path.abspath(path)
        json_path = os.path.join(self.path, SAMPLE_METADATA)
        if not os.path.exists(json_path):
            raise FileNotFoundError("missing %s in %s" % (SAMPLE_METADATA, self.path))
        with open(json_path, encoding="utf-8") as handle:
            self._document = json.load(handle)
        version = int(self._document.get("schema_version", 0))
        if version != SCHEMA_VERSION:
            raise ValueError("sample %s uses schema v%d; expected v%d"
                             % (self.path, version, SCHEMA_VERSION))
        metadata = self._document.get("metadata") or {}
        if "sample_info" not in metadata or "clip_properties" in metadata:
            raise ValueError("%s must contain metadata.sample_info" % json_path)
        self._cache: Dict[str, Any] = {}
        self._h5 = None
        self._h5_pid: Optional[int] = None
        self._twin = twin
        self.selected_fields = tuple(selected_fields or ())

    @classmethod
    def from_dir(cls, path: str) -> "Sample":
        sample = cls(path)
        if sample.is_valid:
            return sample
        # Canonical layout is <root>/samples/<sample_uid>. Resolve the valid
        # twin here as well as in PhysLocDataset so direct visualizer/CLI use
        # has the same reference masks and pair-aware behaviour.
        uid_parts = [part for part in sample.uid.replace("\\", "/").split("/")
                     if part]
        samples_root = os.path.abspath(path)
        for _ in uid_parts:
            samples_root = os.path.dirname(samples_root)
        twin_path = os.path.join(
            samples_root, *sample.valid_sample_uid.replace("\\", "/").split("/"))
        if os.path.realpath(twin_path) != os.path.realpath(path) \
                and os.path.exists(os.path.join(twin_path, SAMPLE_METADATA)):
            sample._twin = cls(twin_path)
        return sample

    @property
    def metadata(self) -> Dict[str, Any]:
        return self._document["metadata"]

    @property
    def sample_info(self) -> Dict[str, Any]:
        return self.metadata["sample_info"]

    info = property(lambda self: self.sample_info)
    # Newer samples keep taxonomy fields in scene.info; exported clip metadata
    # may keep the same fields at the metadata root.
    scene_info = property(lambda self: ((self.metadata.get("scene") or {}).get("info")
                                        or self.metadata))
    uid = property(lambda self: str(self.sample_info["sample_uid"]))
    pair_uid = property(lambda self: str(self.sample_info["pair_uid"]))
    valid_sample_uid = property(lambda self: str(
        self.sample_info.get("valid_sample_uid") or self.uid))
    label = property(lambda self: str(self.sample_info["label"]))
    is_valid = property(lambda self: self.label == "valid")
    num_frames = property(lambda self: int(self.sample_info["num_frames"]))
    fps = property(lambda self: float(self.sample_info["fps"]))
    prompt = property(lambda self: self.scene_info.get("prompt"))
    family = property(lambda self: self.scene_info.get("family"))
    # Schema-v3 metadata uses `scenario`; `type` is retained for older files.
    scenario = property(lambda self: self.scene_info.get("scenario")
                        or self.scene_info.get("type"))
    condition = property(lambda self: self.scene_info.get("condition"))
    level = property(lambda self: self.scene_info.get("level"))
    severity_bin = property(lambda self: self.scene_info.get("severity"))
    twin = property(lambda self: self._twin)

    @property
    def video_path(self) -> str:
        relative = ((self._document.get("observations") or {}).get("rgb") or {}).get(
            "path", RGB)
        path = os.path.join(self.path, relative)
        if not os.path.exists(path):
            raise FileNotFoundError("sample %s is missing %s" % (self.uid, path))
        return path

    rgb_path = video_path
    video = property(lambda self: self._cached("rgb", lambda: _decode_video(self.video_path)))

    def decode_rgb(self) -> np.ndarray:
        return self.video

    def _cached(self, key: str, make: Callable[[], Any]) -> Any:
        if key not in self._cache:
            self._cache[key] = make()
        return self._cache[key]

    def _h5_file(self):
        try:
            import h5py
        except ImportError as exc:
            raise ImportError(
                "PhysLoc schema v3 requires h5py. This interpreter is %s; "
                "activate the physloc conda env or select "
                "/home/ec2-user/miniconda3/envs/physloc/bin/python in VS Code."
                % sys.executable) from exc
        process = os.getpid()
        if self._h5 is None or self._h5_pid != process:
            if self._h5 is not None:
                try:
                    self._h5.close()
                except Exception:
                    pass
            path = os.path.join(self.path, DATA)
            if not os.path.exists(path):
                raise FileNotFoundError("sample %s is missing %s" % (self.uid, DATA))
            self._h5 = h5py.File(path, "r")
            self._h5_pid = process
        return self._h5

    def _h5_has(self, path: str) -> bool:
        return path in self._h5_file()

    def _h5_keys(self, path: str) -> List[str]:
        h5 = self._h5_file()
        return list(h5[path].keys()) if path in h5 else []

    def _h5_read(self, path: str) -> Any:
        def read():
            h5 = self._h5_file()
            if path not in h5:
                raise KeyError(path)
            node = h5[path]
            if hasattr(node, "shape"):
                return node[()]
            return {key: self._h5_read(path.rstrip("/") + "/" + key)
                    for key in node.keys()}
        return self._cached("h5:" + path, read)

    observations = property(lambda self: _GroupView(
        self, "observations", {"rgb_path": self.video_path}))
    maps = property(lambda self: _GroupView(self, "violations/maps"))
    violation_arrays = property(lambda self: _GroupView(self, "violations/objects"))
    scene_energy = property(lambda self: _GroupView(self, "energy/scene"))
    object_energy = property(lambda self: _GroupView(self, "energy/objects"))

    @property
    def object_definitions(self) -> List[Dict[str, Any]]:
        return list((self._document.get("annotations") or {}).get("objects") or [])

    @property
    def object_arrays(self) -> _GroupView:
        groups = [obj.get("analysis_group", "context") for obj in self.object_definitions]
        return _GroupView(self, "objects", {"analysis_group": groups})

    object_ids = property(lambda self: np.asarray(self.object_arrays["ids"], np.int32))
    violator_ids = property(lambda self: self.object_ids[np.asarray(
        self.violation_arrays.get("is_violator", np.zeros(len(self.object_ids), bool)), bool)])
    segmentations = property(lambda self: self.observations["segmentation"])
    violation = property(lambda self: self.maps.get(
        "violation_object_id", np.zeros(self.segmentations.shape, np.uint16)))
    violation_component = property(lambda self: self.maps.get(
        "violation_component", np.zeros(self.segmentations.shape, np.uint8)))
    causal = property(lambda self: self.maps.get(
        "causal_level", np.zeros(self.segmentations.shape, np.uint8)))
    causal_source = property(lambda self: self.maps.get(
        "causal_source_id", np.zeros(self.segmentations.shape, np.uint16)))
    violation_mask = property(lambda self: self.violation > 0)

    @property
    def visible_violation(self) -> np.ndarray:
        shadow = self.observations.get("shadow_source_id")
        return self._cached("visible_violation", lambda: visible_violation(
            self.violation, self.segmentations, self.violation_component, shadow))

    @property
    def object_table(self) -> Dict[str, np.ndarray]:
        n = len(self.object_ids)
        out = {"ids": self.object_ids,
               "is_violator": self.violation_arrays.get("is_violator", np.zeros(n, bool))}
        for key in ("severity", "residual", "score") + CLOCKS + ("affected",):
            out[key] = self.violation_arrays.get(
                key, np.zeros((n, self.num_frames),
                              np.float32 if key in ("severity", "residual", "score") else bool))
        return out

    @property
    def severity_map(self) -> np.ndarray:
        stored = self.maps.get("severity")
        if stored is not None:
            return np.asarray(stored, np.float32)
        table = self.object_table
        return paint_severity(self.violation, self.segmentations, table["ids"],
                              table["severity"], self.violation_component)

    @property
    def reference_mask(self) -> np.ndarray:
        if self.is_valid or self.twin is None:
            return np.zeros(self.segmentations.shape, bool)
        table = self.object_table
        return reference_mask(self.twin.segmentations,
                              table["ids"][np.asarray(table["is_violator"], bool)])

    timeline = property(lambda self: sample_timeline(self.object_table, self.num_frames))
    energy = property(lambda self: self.scene_energy.as_dict())
    energy_map = property(lambda self: self._h5_read("/energy/map"))
    instances = property(lambda self: self.object_arrays.as_dict())

    @property
    def camera(self) -> Dict[str, Any]:
        camera = self.metadata["scene"]["world"].get("camera") or {}
        return {key: np.asarray(value) if isinstance(value, list) else value
                for key, value in camera.items()}

    @property
    def display_metadata(self) -> Dict[str, Any]:
        """Compact metadata view used by the frame compositor."""
        world = self.metadata["scene"]["world"]
        identity = {**self.sample_info,
                    "scenario": self.scenario, "family": self.family,
                    "domain": self.scene_info.get("domain"),
                    "physics_medium": self.scene_info.get("physics_medium"),
                    "condition": self.condition, "complexity": {"name": self.level},
                    "frame_rate": self.fps,
                    "n_violators": world.get("objects_summary", {}).get("n_violators", 0)}
        annotations = self._document.get("annotations") or {}
        events = annotations.get("events") or []
        if isinstance(events, list):
            events = {"events": events,
                      "collisions": [event for event in events
                                     if event.get("type") == "collision"]}
        return {"metadata": identity, "camera": world.get("camera") or {},
                "instances": self.object_definitions, "events": events,
                "violation": annotations.get("violation_summary"),
                "difficulty": (self.scene_info.get("difficulty_analysis")
                               or ({"level": self.scene_info.get("difficulty")}
                                   if self.scene_info.get("difficulty") else None))}

    def object(self, object_id: int) -> Dict[str, Any]:
        matches = np.flatnonzero(self.object_ids == int(object_id))
        if not matches.size:
            raise KeyError("sample %s has no object id %s" % (self.uid, object_id))
        row = int(matches[0])

        def row_values(group: str) -> Dict[str, Any]:
            values = {}
            for key in self._h5_keys(group):
                value = self._h5_read(group + "/" + key)
                if isinstance(value, np.ndarray) and value.ndim \
                        and value.shape[0] == len(self.object_ids):
                    values[key] = value[row]
            return values
        return {**dict(self.object_definitions[row]),
                "temporal": row_values("/objects"),
                "violation": row_values("/violations/objects"),
                "energy": row_values("/energy/objects")}

    def objects(self, selection: str = "subjects") -> List[Dict[str, Any]]:
        allowed = {"subjects", "violators", "affected", "context", "support", "all"}
        if selection not in allowed:
            raise ValueError("selection must be one of %s" % sorted(allowed))
        table = self.object_table
        affected = np.asarray(table["affected"], bool).any(axis=1)
        out = []
        for row, definition in enumerate(self.object_definitions):
            group = definition.get("analysis_group", "context")
            keep = (selection == "all"
                    or selection == "subjects" and group == "subject"
                    or selection == group
                    or selection == "violators" and bool(table["is_violator"][row])
                    or selection == "affected" and bool(affected[row]))
            if keep:
                out.append(self.object(int(definition["id"])))
        return out

    def object_mask(self, selection: str = "subjects") -> np.ndarray:
        """Boolean ``[N]`` mask for a named analysis selection."""
        selected = {int(item["id"]) for item in self.objects(selection)}
        return np.asarray([int(object_id) in selected for object_id in self.object_ids],
                          dtype=bool)

    def spatial_mask(self, selection: str = "subjects") -> np.ndarray:
        """Boolean ``[T,H,W]`` segmentation mask for a named selection."""
        return np.isin(self.segmentations, self.object_ids[self.object_mask(selection)])

    subject_mask = property(lambda self: self.spatial_mask("subjects"))
    context_mask = property(lambda self: self.spatial_mask("context"))
    support_mask = property(lambda self: self.spatial_mask("support"))
    violator_mask = property(lambda self: self.spatial_mask("violators"))
    affected_mask = property(lambda self: self.spatial_mask("affected"))

    violation_summary = property(lambda self:
        (self._document.get("annotations") or {}).get("violation_summary"))
    causal_relations = property(lambda self:
        (self._document.get("annotations") or {}).get("causal_relations") or [])

    def pass_(self, name: str) -> np.ndarray:
        if name not in PASSES:
            raise KeyError("unknown observation %r" % name)
        return self.observations[name]

    def has(self, name: str) -> bool:
        aliases = {"video": "rgb", "video_path": "rgb", RGB: "rgb",
                   "energy_map": "/energy/map", "energy": "/energy/scene"}
        key = aliases.get(name, name)
        if key == "rgb":
            return os.path.exists(self.video_path)
        if str(key).startswith("/"):
            return self._h5_has(str(key))
        if key == "segmentations":
            key = "segmentation"
        return self._h5_has("/observations/" + str(key))

    def latent_grid(self, latent_frames: Optional[int] = None,
                    latent_hw: Optional[int] = None) -> Dict[str, np.ndarray]:
        frames = latent_frames or int(self.sample_info.get(
            "latent_frames") or (self.num_frames - 1) // 4 + 1)
        side = latent_hw or int(self.sample_info.get(
            "latent_hw") or self.segmentations.shape[-1] // 16)
        return latent_grid(self.violation_mask, self.severity_map, frames, side)

    @property
    def divergence(self) -> np.ndarray:
        if self.is_valid or self.twin is None:
            return np.zeros(self.segmentations.shape, np.float32)
        return self._cached("divergence", lambda: divergence(self.twin.video, self.video))

    path_info = property(lambda self: {**self.sample_info, **self.scene_info})

    def get(self, field: str) -> Any:
        if field == "metadata":
            return self.metadata
        if field == "observations.rgb_path":
            return self.video_path
        if field == "observations.rgb":
            return self.video
        if field.startswith("observations."):
            return self.observations[field.split(".", 1)[1]]
        if field.startswith("annotations.maps."):
            return self.maps[field.rsplit(".", 1)[1]]
        if field == "annotations.scene_energy":
            return self.scene_energy.as_dict()
        if field == "annotations.objects.energy":
            return self.object_energy.as_dict()
        if field.startswith("annotations.objects."):
            key = field.rsplit(".", 1)[1]
            if key == "analysis_group":
                return [obj.get("analysis_group") for obj in self.object_definitions]
            return (self.violation_arrays[key] if key in self.violation_arrays
                    else self.object_arrays[key])
        raise KeyError("unknown schema-v3 field %r" % field)

    def to_dict(self, fields: Optional[Sequence[str]] = None) -> Dict[str, Any]:
        selected = tuple(fields if fields is not None else self.selected_fields)
        out = {"uid": self.uid, "pair_uid": self.pair_uid, "label": self.label}
        out.update({field: self.get(field) for field in selected})
        return out

    def __getitem__(self, key: str) -> Any:
        if key == "metadata":
            return self.metadata
        if key == "observations":
            return self.observations
        if key == "annotations":
            annotations = self._document.get("annotations") or {}
            return {"objects": self.object_definitions, "scene_energy": self.scene_energy,
                    "maps": self.maps, "events": annotations.get("events", []),
                    "violation_summary": annotations.get("violation_summary")}
        if key in ("uid", "sample_uid"):
            return self.uid
        if key in ("pair_uid", "label"):
            return getattr(self, key)
        return self.get(key)

    def release(self) -> None:
        self._cache.clear()
        if self._h5 is not None:
            self._h5.close()
        self._h5 = None
        self._h5_pid = None

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_h5"] = None
        state["_h5_pid"] = None
        return state

    def __repr__(self) -> str:
        return "Sample(%s)" % self.uid


class Pair:
    def __init__(self, pair_uid: str, valid: Optional[Sample] = None,
                 invalids: Optional[List[Sample]] = None):
        self.pair_uid = pair_uid
        self.valid = valid
        self.invalids = list(invalids or [])

    @property
    def prompt(self) -> Optional[str]:
        sample = self.valid or (self.invalids[0] if self.invalids else None)
        return sample.prompt if sample else None


def _as_set(value: Any) -> set:
    if isinstance(value, (str, int, float)) or value is None:
        return {None if value is None else str(value)}
    return {None if item is None else str(item) for item in value}


class PhysLocDataset:
    """All schema-v3 samples under ``root``."""

    FILTERS = ("label", "scenario", "family", "level", "condition",
               "severity", "seed", "split")
    UNITS = ("sample", "pair")

    def __init__(self, root: str, fields: Optional[Sequence[str]] = None,
                 split: Optional[str] = None, unit: str = "sample", **filters: Any):
        if unit not in self.UNITS:
            raise ValueError("unit must be one of %s, not %r" % (self.UNITS, unit))
        unknown = set(filters) - set(self.FILTERS)
        if unknown:
            raise TypeError("unknown filter(s) %s; known: %s"
                            % (sorted(unknown), self.FILTERS))
        selected = tuple(fields or ())
        known = set(FIELDS) | set(METADATA_FIELDS) | set(LOCALIZATION_FIELDS) \
            | set(OBJECT_LOCALIZATION_FIELDS) | set(ENERGY_FIELDS) | set(VISUALIZATION_FIELDS)
        bad = [field for field in selected if field not in known]
        if bad:
            raise KeyError("unknown field(s) %s" % bad)
        if split is not None:
            filters["split"] = split
        self.root = os.path.abspath(root)
        self.selected = selected
        self.unit = unit
        paths = sorted(glob.glob(os.path.join(
            self.root, "samples", "**", SAMPLE_METADATA), recursive=True))
        if not paths:
            raise FileNotFoundError(
                "no schema-v3 samples under %s; expected samples/<sample_uid>/%s"
                % (self.root, SAMPLE_METADATA))
        wanted = {key: _as_set(value) for key, value in filters.items()}
        self.samples: List[Sample] = []
        for json_path in paths:
            sample = Sample(os.path.dirname(json_path), selected_fields=selected)
            values = {"label": sample.label, "family": sample.family,
                      "scenario": sample.scenario, "level": sample.level,
                      "condition": sample.condition, "severity": sample.severity_bin,
                      "seed": sample.sample_info.get("seed"),
                      "split": sample.sample_info.get("split")}
            if all(str(values.get(key)) in choices for key, choices in wanted.items()):
                self.samples.append(sample)
        self._by_uid = {sample.uid: sample for sample in self.samples}
        valid = {sample.pair_uid: sample for sample in self.samples if sample.is_valid}
        for sample in self.samples:
            if not sample.is_valid:
                sample._twin = (self._by_uid.get(sample.valid_sample_uid)
                                or valid.get(sample.pair_uid))
        grouped: Dict[str, Pair] = {}
        for sample in self.samples:
            pair = grouped.setdefault(sample.pair_uid, Pair(sample.pair_uid))
            if sample.is_valid:
                pair.valid = sample
            else:
                pair.invalids.append(sample)
        self._pairs = [grouped[key] for key in sorted(grouped)]
        if unit == "pair":
            self._pairs = [pair for pair in self._pairs
                           if pair.valid is not None and pair.invalids]

    def get(self, sample_uid: str) -> Sample:
        if str(sample_uid) not in self._by_uid:
            raise KeyError("unknown sample %r" % sample_uid)
        return self._by_uid[str(sample_uid)]

    def pairs(self) -> List[Pair]:
        return list(self._pairs)

    def info(self, index: int) -> Dict[str, Any]:
        return self.samples[index].path_info

    def __len__(self) -> int:
        return len(self._pairs) if self.unit == "pair" else len(self.samples)

    def __getitem__(self, index: int):
        if self.unit == "sample":
            return self.samples[index]
        pair = self._pairs[index]
        return {"pair_uid": pair.pair_uid, "prompt": pair.prompt,
                "valid": pair.valid, "invalid": list(pair.invalids)}

    def __iter__(self) -> Iterator[Any]:
        for index in range(len(self)):
            yield self[index]

    def release(self) -> None:
        for sample in self.samples:
            sample.release()

    def __repr__(self) -> str:
        return "PhysLocDataset(%s, %d %ss)" % (self.root, len(self), self.unit)


def _collate_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key in rows[0]:
        values = [row[key] for row in rows]
        if isinstance(values[0], np.ndarray) and all(
                isinstance(value, np.ndarray) and value.shape == values[0].shape
                for value in values):
            out[key] = np.stack(values)
        elif isinstance(values[0], dict) and all(isinstance(value, dict) for value in values):
            out[key] = _collate_rows(values)
        else:
            out[key] = values
    return out


def collate(samples: Sequence[Any]) -> Dict[str, Any]:
    if not samples:
        return {}
    if not isinstance(samples[0], Sample):
        return _collate_rows(samples)
    rows = [sample.to_dict(sample.selected_fields) for sample in samples]
    out = _collate_rows(rows)
    ids = [sample.object_ids for sample in samples]
    width = max((len(value) for value in ids), default=0)
    out["object_ids"] = np.stack([
        np.pad(value, (0, width - len(value)), constant_values=0) for value in ids])
    out["object_valid"] = np.stack([
        np.arange(width) < len(value) for value in ids])
    for field in samples[0].selected_fields:
        if not field.startswith("annotations.objects."):
            continue
        values = [sample.get(field) for sample in samples]
        if not all(isinstance(value, np.ndarray) and value.ndim
                   and value.shape[0] == len(object_ids)
                   for value, object_ids in zip(values, ids)):
            continue
        shape = (len(samples), width) + values[0].shape[1:]
        dtype = np.result_type(*[value.dtype for value in values])
        fill = np.nan if np.issubdtype(dtype, np.floating) else 0
        padded = np.full(shape, fill, dtype=dtype)
        for row, value in enumerate(values):
            padded[row, :len(value)] = value
        out[field] = padded
    return out


def torch_dataset(dataset: PhysLocDataset):
    from torch.utils.data import Dataset

    class _TorchDataset(Dataset):
        def __len__(self):
            return len(dataset)

        def __getitem__(self, index):
            return dataset[index]

    return _TorchDataset()
