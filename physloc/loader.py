"""Standalone reader for the PhysLoc schema-v4 sample format.

A release has one public representation::

    dataset.json  schema.json  index.parquet  splits/
    samples/<sample_uid>/sample.json  rgb.mp4  data.h5

A `Sample` is read the way the files are laid out, one namespace per block::

    s = PhysLocDataset(root)[0]
    s.info          who it is: uid, pair_uid, label, split, release, tier, seed ...
    s.video         num_frames, fps, resolution, path, rgb (decoded on access)
    s.scene         scenario, family, level, condition, prompt, camera, physics ...
    s.observations  segmentation, depth, forward_flow, ...        [T,H,W,...]
    s.objects       ids, names, roles, records, positions, bboxes ... [N,T,...]
    s.violation     what is wrong, where, when and how badly (empty on a valid sample)
    s.energy        scene [T], objects [N,T], map [T,H,W]
    s.events        collisions, one column per field
    s.twin          the valid sample of the same pair

**Every fact is stored once.** Whatever can be computed from what is stored --
the clip-level windows and times, the [N,T] clocks, causal relations, object
counts, the severity map -- is computed here, and this module is the one
implementation of each. Dense arrays are read lazily from the sample's chunked
HDF5 store and cached per sample. This module has no imports from
:mod:`physloc`, so the exact file ships with a Hub dataset.
"""
from __future__ import annotations

import glob
import json
import os
import sys
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence

import numpy as np

SCHEMA_VERSION = 4
SAMPLE_METADATA = "sample.json"
DATA = "data.h5"
RGB = "rgb.mp4"
PASSES = ("segmentation", "depth", "forward_flow", "backward_flow", "normal",
          "object_coordinates", "shadow_strength", "shadow_source_id")
CLOCKS = ("active", "intervening", "consequence", "observable", "occluded")
COMPONENTS = {0: "none", 1: "body", 2: "shadow", 3: "trajectory",
              4: "interaction", 5: "energy"}
GROUPS = ("subject", "context", "support", "background")
SELECTIONS = ("subjects", "violators", "affected", "context", "support",
              "background", "all")
DEPTH_BACKGROUND = 1e6
SHADOW_CASTER_GUARD_PX = 2

#: Every selector `PhysLocDataset(fields=...)` and `Sample.to_dict` accept, with
#: its axes. A selector names a namespace attribute; `to_dict` returns it nested
#: under its namespace. Axes starting with N are per object, and `collate` pads
#: them to the batch's largest N.
FIELDS: Dict[str, str] = {
    "info": "identity block (dict)",
    "video": "clip geometry (dict)",
    "scene": "scene description (dict)",
    "video.rgb": "uint8 T,H,W,3 (decoded mp4)",
    "video.path": "path to rgb.mp4",
    **{"observations." + name: "T,H,W[,C]" for name in PASSES},
    "objects.ids": "N",
    "objects.groups": "N",
    "objects.is_violator": "N",
    "objects.positions": "N,T,3",
    "objects.quaternions": "N,T,4",
    "objects.velocities": "N,T,3",
    "objects.angular_velocities": "N,T,3",
    "objects.bboxes": "N,T,4",
    "objects.bboxes_3d": "N,T,8,3",
    "objects.image_positions": "N,T,2",
    "objects.visibility": "N,T",
    "violation.object_id": "T,H,W",
    "violation.component_map": "T,H,W",
    "violation.causal": "T,H,W",
    "violation.causal_source": "T,H,W",
    "violation.mask": "T,H,W",
    "violation.visible": "T,H,W",
    "violation.severity_map": "T,H,W",
    "violation.reference_mask": "T,H,W",
    "violation.timeline": "dict of T",
    **{"violation." + clock: "N,T" for clock in CLOCKS},
    "violation.affected": "N,T",
    "violation.severity": "N,T",
    "violation.residual": "N,T",
    "violation.score": "N,T",
    "energy.scene": "dict of T",
    "energy.objects": "dict of N,T[,3]",
    "energy.map": "T,H,W",
    "events.collisions": "dict of E[,...]",
    "divergence": "T,H,W (needs the valid twin's video)",
}


# ---------------------------------------------------------------- functions
def intervals_to_mask(windows: Sequence[Sequence[int]], num_frames: int) -> np.ndarray:
    """``[[a, b], ...]`` (inclusive) -> ``[T]`` bool."""
    out = np.zeros(int(num_frames), bool)
    for start, end in windows or ():
        out[int(start):int(end) + 1] = True
    return out


def mask_to_intervals(mask: np.ndarray) -> List[List[int]]:
    """``[T]`` bool -> inclusive ``[[a, b], ...]``."""
    idx = np.flatnonzero(np.asarray(mask, bool))
    if not idx.size:
        return []
    breaks = np.flatnonzero(np.diff(idx) > 1)
    starts = np.concatenate([[idx[0]], idx[breaks + 1]])
    ends = np.concatenate([idx[breaks], [idx[-1]]])
    return [[int(a), int(b)] for a, b in zip(starts, ends)]


def union_intervals(lists: Sequence[Sequence[Sequence[int]]],
                    num_frames: int) -> List[List[int]]:
    mask = np.zeros(int(num_frames), bool)
    for windows in lists:
        mask |= intervals_to_mask(windows, num_frames)
    return mask_to_intervals(mask)


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


def shadow_reference_mask(shadow_strength: np.ndarray,
                          shadow_source_id: np.ndarray,
                          ids: Sequence[int],
                          threshold: float = 1.0 / 255.0) -> np.ndarray:
    """Return the lawful cast-shadow footprint for public caster IDs.

    A Cycles shadow has no segmentation pixels of its own.  The source pass
    therefore carries the public actor ID while ``shadow_strength`` carries
    the receiver footprint.  Keeping this separate from ``reference_mask`` is
    important: falling back to segmentation would outline the caster itself,
    exactly the object the shadow component is meant to exclude.
    """
    strength = np.asarray(shadow_strength, np.float32)
    source = np.asarray(shadow_source_id)
    return (strength > float(threshold)) & np.isin(
        source, np.asarray(list(ids), dtype=source.dtype))


def shadow_receiver_mask(shadow_strength: np.ndarray,
                         shadow_source_id: np.ndarray,
                         segmentation: np.ndarray,
                         ids: Sequence[int],
                         guard_px: int = SHADOW_CASTER_GUARD_PX) -> np.ndarray:
    """Projected shadow pixels, excluding the caster and its local halo."""
    projected = shadow_reference_mask(shadow_strength, shadow_source_id, ids)
    caster = np.isin(np.asarray(segmentation),
                     np.asarray(list(ids), dtype=np.asarray(segmentation).dtype))
    return projected & ~_dilate_mask(caster, int(guard_px))


def _dilate_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    """Square per-frame dilation without an optional image-processing dependency."""
    out = np.asarray(mask, bool).copy()
    if radius <= 0:
        return out
    source = out.copy()
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy == 0 and dx == 0:
                continue
            out |= np.roll(np.roll(source, dy, axis=1), dx, axis=2)
    return out


def sample_timeline(clocks: Dict[str, np.ndarray], severity: np.ndarray,
                    is_violator: np.ndarray, num_frames: int) -> Dict[str, np.ndarray]:
    """Clip-level ``[T]`` clocks and severity: any / max over the violators."""
    rows = np.asarray(is_violator, bool)
    out: Dict[str, np.ndarray] = {}
    for key in CLOCKS:
        selected = np.asarray(clocks[key], bool)[rows]
        out[key] = (selected.any(axis=0) if len(selected)
                    else np.zeros(num_frames, bool))
    selected = np.asarray(severity, np.float32)[rows]
    out["severity"] = (selected.max(axis=0) if len(selected)
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


# --------------------------------------------------------------- namespaces
class _Namespace:
    """A read-only block of a sample: ``ns.key`` and ``ns["key"]`` alike.

    ``KEYS`` lists what the block holds; values are properties, so an array is
    read from HDF5 only when asked for. ``as_dict()`` materialises the block.
    """

    KEYS: Sequence[str] = ()

    def __init__(self, sample: "Sample"):
        self._sample = sample

    def keys(self) -> List[str]:
        return list(self.KEYS)

    def __getitem__(self, key: str) -> Any:
        if key not in self.keys():
            raise KeyError("%s has no %r; keys: %s"
                           % (type(self).__name__.lower(), key, ", ".join(self.keys())))
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            value = self[key]
        except KeyError:
            return default
        return default if value is None else value

    def __contains__(self, key: str) -> bool:
        return key in self.keys()

    def as_dict(self, keys: Optional[Sequence[str]] = None) -> Dict[str, Any]:
        return {key: self[key] for key in (keys or self.keys())}

    def __repr__(self) -> str:
        scalars = []
        for key in self.keys():
            if _is_lazy(type(self), key):
                continue
            item = self[key]
            if isinstance(item, (str, int, float, bool)) or item is None:
                scalars.append("%s=%r" % (key, item))
        return "<%s %s>" % (type(self).__name__, " ".join(scalars))


class _lazy(property):
    """A property backed by HDF5 or a derivation: cached on the sample, and
    skipped by ``repr`` and by whole-namespace ``to_dict`` selectors."""

    def __init__(self, fn: Callable):
        super().__init__(self._get)
        self.fn = fn
        self.name = fn.__name__
        self.__doc__ = fn.__doc__

    def __set_name__(self, owner, name: str) -> None:
        self.name = name

    def _get(self, namespace):
        return namespace._sample._cached(
            "%s.%s" % (type(namespace).__name__, self.name), lambda: self.fn(namespace))


def _is_lazy(kind, key: str) -> bool:
    return isinstance(getattr(kind, key, None), _lazy)


class Info(_Namespace):
    """Who the sample is. Straight from ``sample.json["sample"]``."""

    KEYS = ("uid", "pair_uid", "valid_uid", "label", "is_valid", "split", "release",
            "tier", "seed", "variant", "framing_attempt", "provenance")

    def __init__(self, sample: "Sample"):
        super().__init__(sample)
        self._d = sample._document["sample"]

    uid = property(lambda self: str(self._d["uid"]))
    pair_uid = property(lambda self: str(self._d["pair_uid"]))
    valid_uid = property(lambda self: str(self._d.get("valid_uid") or self._d["uid"]))
    label = property(lambda self: str(self._d["label"]))
    is_valid = property(lambda self: self.label == "valid")
    split = property(lambda self: self._d.get("split"))
    release = property(lambda self: self._d.get("release"))
    tier = property(lambda self: self._d.get("tier"))
    seed = property(lambda self: self._d.get("seed"))
    variant = property(lambda self: int(self._d.get("variant") or 0))
    framing_attempt = property(lambda self: int(self._d.get("framing_attempt") or 0))
    provenance = property(lambda self: self._sample._document.get("provenance") or {})


class Video(_Namespace):
    """Clip geometry, and the RGB video itself."""

    KEYS = ("num_frames", "fps", "resolution", "duration", "latent_frames",
            "latent_hw", "path", "rgb")

    def __init__(self, sample: "Sample"):
        super().__init__(sample)
        self._d = sample._document["video"]

    num_frames = property(lambda self: int(self._d["num_frames"]))
    fps = property(lambda self: float(self._d["fps"]))
    resolution = property(lambda self: tuple(int(v) for v in self._d["resolution"]))
    duration = property(lambda self: self.num_frames / self.fps if self.fps else 0.0)
    latent_frames = property(lambda self: int(
        (self._d.get("latent") or {}).get("frames") or (self.num_frames - 1) // 4 + 1))
    latent_hw = property(lambda self: int(
        (self._d.get("latent") or {}).get("hw") or self.resolution[-1] // 16))

    @property
    def path(self) -> str:
        path = os.path.join(self._sample.path, RGB)
        if not os.path.exists(path):
            raise FileNotFoundError("sample %s is missing %s" % (self._sample.uid, path))
        return path

    @_lazy
    def rgb(self) -> np.ndarray:
        """uint8 [T,H,W,3], decoded from the mp4 on first access."""
        return _decode_video(self.path)


class Camera(_Namespace):
    """Lens and aim from ``sample.json``; the per-frame pose from ``/camera``."""

    KEYS = ("motion", "K", "field_of_view", "focal_length", "sensor_width",
            "look_at", "position", "end_position", "positions", "quaternions")

    def __init__(self, sample: "Sample"):
        super().__init__(sample)
        self._d = sample._document["scene"].get("camera") or {}

    motion = property(lambda self: self._d.get("motion", "static"))
    K = property(lambda self: None if self._d.get("K") is None
                 else np.asarray(self._d["K"], np.float64))
    field_of_view = property(lambda self: self._d.get("field_of_view"))
    focal_length = property(lambda self: self._d.get("focal_length"))
    sensor_width = property(lambda self: self._d.get("sensor_width"))
    look_at = property(lambda self: self._d.get("look_at"))
    position = property(lambda self: self._d.get("position"))
    end_position = property(lambda self: self._d.get("end_position"))

    @_lazy
    def positions(self) -> np.ndarray:
        """float [T,3] camera-to-world translation, Kubric convention."""
        return self._sample._h5_read("/camera/positions")

    @_lazy
    def quaternions(self) -> np.ndarray:
        """float [T,4] camera-to-world rotation (w,x,y,z), looking down -Z."""
        return self._sample._h5_read("/camera/quaternions")


class Scene(_Namespace):
    """What was simulated and how it was shot."""

    KEYS = ("scenario", "family", "domain", "physics_medium", "level", "condition",
            "prompt", "size_scale", "camera", "environment", "physics")

    def __init__(self, sample: "Sample"):
        super().__init__(sample)
        self._d = sample._document["scene"]

    scenario = property(lambda self: self._d.get("scenario"))
    family = property(lambda self: self._d.get("family"))
    domain = property(lambda self: self._d.get("domain"))
    physics_medium = property(lambda self: self._d.get("physics_medium"))
    level = property(lambda self: self._d.get("level"))
    condition = property(lambda self: self._d.get("condition"))
    prompt = property(lambda self: self._d.get("prompt"))
    size_scale = property(lambda self: self._d.get("size_scale"))
    environment = property(lambda self: self._d.get("environment") or {})
    physics = property(lambda self: self._d.get("physics") or {})

    @property
    def camera(self) -> Camera:
        return self._sample._namespace(Camera)


class _Group(_Namespace):
    """Every dataset under one HDF5 group, by name."""

    GROUP = ""

    def keys(self) -> List[str]:
        return sorted(self._sample._h5_keys(self.GROUP))

    def __getitem__(self, key: str) -> Any:
        if key not in self.keys():
            raise KeyError("%s has no %r; keys: %s"
                           % (self.GROUP, key, ", ".join(self.keys())))
        return self._sample._h5_read(self.GROUP + "/" + key)

    def __getattr__(self, key: str) -> Any:
        if key.startswith("_"):
            raise AttributeError(key)
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(str(exc)) from None

    def __repr__(self) -> str:
        return "<%s %s>" % (type(self).__name__, ", ".join(self.keys()))


class Observations(_Group):
    """Dense render passes, ``[T,H,W,...]``: ``segmentation`` always, the rest
    when the tier rendered them (``"depth" in s.observations``)."""

    GROUP = "/observations"


class Objects(_Namespace):
    """Every public object, in one order: the ``sample.json`` list, which is
    also the row order of every ``[N,...]`` array in ``data.h5``."""

    STATIC = ("ids", "names", "roles", "groups", "is_violator", "records")

    def __init__(self, sample: "Sample"):
        super().__init__(sample)
        self.records: List[Dict[str, Any]] = list(sample._document.get("objects") or [])
        self.ids = np.asarray([int(o["id"]) for o in self.records], np.int32)
        self.names = [str(o.get("name", o["id"])) for o in self.records]
        self.roles = [o.get("role") for o in self.records]
        self.groups = [o.get("analysis_group", "context") for o in self.records]
        self._rows = {int(i): row for row, i in enumerate(self.ids)}

    def keys(self) -> List[str]:
        return list(self.STATIC) + [key for key in self._sample._h5_keys("/objects")
                                    if key not in self.STATIC]

    def __getitem__(self, key: str) -> Any:
        if key in self.STATIC:
            return getattr(self, key)
        if key in self._sample._h5_keys("/objects"):
            return self._sample._h5_read("/objects/" + key)
        raise KeyError("objects has no %r; keys: %s" % (key, ", ".join(self.keys())))

    def __getattr__(self, key: str) -> Any:
        if key.startswith("_"):
            raise AttributeError(key)
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(str(exc)) from None

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        return iter(self.records)

    @property
    def is_violator(self) -> np.ndarray:
        return np.isin(self.ids, self._sample.violation.ids)

    def row(self, object_id: int) -> int:
        if int(object_id) not in self._rows:
            raise KeyError("sample %s has no object id %s" % (self._sample.uid, object_id))
        return self._rows[int(object_id)]

    def record(self, object_id: int) -> Dict[str, Any]:
        """One object: its ``sample.json`` record plus its row of every array."""
        row = self.row(object_id)
        s = self._sample
        n = len(self)

        def rows(group: Dict[str, Any]) -> Dict[str, Any]:
            return {key: value[row] for key, value in group.items()
                    if isinstance(value, np.ndarray) and value.ndim and value.shape[0] == n}
        violation = s.violation
        per_object = {key: violation[key] for key in CLOCKS + (
            "affected", "severity", "residual", "score")} if violation.present else {}
        own = next((v for v in violation.violators if v["id"] == int(object_id)), None)
        return {**self.records[row],
                "arrays": rows({key: self[key] for key in s._h5_keys("/objects")}),
                "energy": rows(s.energy.objects),
                "violation": dict(rows(per_object), record=own) if own else None}

    def __repr__(self) -> str:
        return "<Objects %d: %s>" % (len(self), ", ".join(
            "%s#%d" % (name, i) for name, i in zip(self.names, self.ids)))

    def select(self, selection: str = "subjects") -> np.ndarray:
        """Ids of a named selection: subjects, violators, affected, context,
        support, background or all."""
        return self.ids[self.mask(selection)]

    def mask(self, selection: str = "subjects") -> np.ndarray:
        """Boolean ``[N]`` for a named selection."""
        if selection not in SELECTIONS:
            raise ValueError("selection must be one of %s" % (SELECTIONS,))
        if selection == "all":
            return np.ones(len(self), bool)
        if selection == "violators":
            return self.is_violator
        if selection == "affected":
            return self._sample.violation.affected.any(axis=1)
        group = "subject" if selection == "subjects" else selection
        return np.asarray([g == group for g in self.groups], bool)

    def spatial_mask(self, selection: str = "subjects") -> np.ndarray:
        """Boolean ``[T,H,W]``: the segmentation pixels of a named selection."""
        return np.isin(self._sample.observations.segmentation, self.select(selection))


class Violation(_Namespace):
    """What is wrong, where, when and how badly.

    ``present`` is False on a valid sample, and every array is then zeros of the
    right shape, so a batch mixes valid and invalid samples without special
    cases. Per-violator records (``violators``) carry the five clocks as
    inclusive frame intervals; the clip-level windows and times, the ``[N,T]``
    clock arrays and the causal relations are derived from them here.
    """

    KEYS = ("present", "severity_bin", "kind", "component", "spatial_extent",
            "timing", "intervention", "difficulty", "peak_residual",
            "occluded_at_event", "consequences", "shadow", "causal_ids",
            "violators", "ids", "windows", "t_event", "t_observable", "t_end",
            "t_intervention_end", "t_consequence_end", "observability_lag",
            "causal_relations") + CLOCKS + (
            "affected", "severity", "residual", "score",
            "object_id", "component_map", "causal", "causal_source",
            "mask", "visible", "severity_map", "reference_mask", "timeline")

    def __init__(self, sample: "Sample"):
        super().__init__(sample)
        self._d = sample._document.get("violation") or {}
        self._T = sample.video.num_frames

    present = property(lambda self: bool(self._d))
    severity_bin = property(lambda self: self._d.get("severity_bin"))
    kind = property(lambda self: self._d.get("kind"))
    component = property(lambda self: self._d.get("component"))
    spatial_extent = property(lambda self: self._d.get("spatial_extent"))
    timing = property(lambda self: self._d.get("timing"))
    intervention = property(lambda self: self._d.get("intervention") or {})
    difficulty = property(lambda self: self._d.get("difficulty"))
    peak_residual = property(lambda self: self._d.get("peak_residual") or {})
    occluded_at_event = property(lambda self: self._d.get("occluded_at_event"))
    consequences = property(lambda self: self._d.get("consequences") or [])
    shadow = property(lambda self: self._d.get("shadow"))
    causal_ids = property(lambda self: [int(i) for i in self._d.get("causal_ids") or []])

    @property
    def violators(self) -> List[Dict[str, Any]]:
        """Per-violator records, completed with the fields derived from them."""
        def complete(record: Dict[str, Any]) -> Dict[str, Any]:
            out = dict(record)
            windows = {clock: [list(w) for w in (record.get("windows") or {}).get(clock) or []]
                       for clock in CLOCKS}
            out["windows"] = windows
            active = windows["active"]
            t_event = int(active[0][0]) if active else None
            out["t_event"] = t_event
            # The first observable frame AT OR AFTER the event; the event itself
            # when the violator is never seen afterwards.
            later = [f for a, b in windows["observable"] for f in range(a, b + 1)
                     if t_event is not None and f >= t_event]
            out["t_observable"] = later[0] if later else t_event
            out["observability_lag"] = (None if t_event is None
                                        else out["t_observable"] - t_event)
            out.setdefault("magnitude", self.intervention.get("magnitude"))
            out["affected_ids"] = [int(i) for i in record.get("affected_ids") or []]
            return out
        return self._sample._cached("violators", lambda: [
            complete(v) for v in self._d.get("violators") or []])

    ids = property(lambda self: np.asarray([v["id"] for v in self.violators], np.int32))

    @property
    def windows(self) -> Dict[str, List[List[int]]]:
        """Clip-level windows per clock: the union over violators."""
        return {clock: union_intervals([v["windows"][clock] for v in self.violators],
                                       self._T) for clock in CLOCKS}

    def _first(self, clock: str) -> Optional[int]:
        windows = self.windows[clock]
        return int(windows[0][0]) if windows else None

    def _last(self, clock: str) -> Optional[int]:
        windows = self.windows[clock]
        return int(windows[-1][1]) if windows else None

    t_event = property(lambda self: self._first("active"))
    t_observable = property(lambda self: self._first("observable"))
    t_end = property(lambda self: self._last("active"))
    t_intervention_end = property(lambda self: self._last("intervening"))
    t_consequence_end = property(lambda self: self._last("consequence"))

    @property
    def observability_lag(self) -> Optional[int]:
        if self.t_event is None or self.t_observable is None:
            return None
        return self.t_observable - self.t_event

    @property
    def causal_relations(self) -> List[Dict[str, Any]]:
        """violator -> affected body, over the violator's consequence windows."""
        return [{"source_object_id": v["id"], "target_object_id": target,
                 "relation_type": "physical_consequence",
                 "intervals": v["windows"]["consequence"]}
                for v in self.violators for target in v["affected_ids"]]

    # ---- [N,T] per object --------------------------------------------------
    def _clock(self, clock: str) -> np.ndarray:
        objects = self._sample.objects
        out = np.zeros((len(objects), self._T), bool)
        for v in self.violators:
            out[objects.row(v["id"])] = intervals_to_mask(v["windows"][clock], self._T)
        return out

    active = _lazy(lambda self: self._clock("active"))
    intervening = _lazy(lambda self: self._clock("intervening"))
    consequence = _lazy(lambda self: self._clock("consequence"))
    observable = _lazy(lambda self: self._clock("observable"))
    occluded = _lazy(lambda self: self._clock("occluded"))

    @_lazy
    def affected(self) -> np.ndarray:
        """bool [N,T]: a body a violator disturbed, over that violator's
        consequence windows."""
        objects = self._sample.objects
        out = np.zeros((len(objects), self._T), bool)
        for v in self.violators:
            frames = intervals_to_mask(v["windows"]["consequence"], self._T)
            for target in v["affected_ids"]:
                out[objects.row(target)] |= frames
        return out

    def _per_object(self, key: str) -> np.ndarray:
        path = "/violation/objects/" + key
        if self._sample._h5_has(path):
            return np.asarray(self._sample._h5_read(path), np.float32)
        return np.zeros((len(self._sample.objects), self._T), np.float32)

    severity = _lazy(lambda self: self._per_object("severity"))
    residual = _lazy(lambda self: self._per_object("residual"))
    score = _lazy(lambda self: self._per_object("score"))

    # ---- [T,H,W] maps ------------------------------------------------------
    def _map(self, key: str, dtype) -> np.ndarray:
        path = "/violation/maps/" + key
        if self._sample._h5_has(path):
            return self._sample._h5_read(path)
        return np.zeros(self._sample.observations.segmentation.shape, dtype)

    object_id = _lazy(lambda self: self._map("object_id", np.uint16))
    component_map = _lazy(lambda self: self._map("component", np.uint8))
    causal = _lazy(lambda self: self._map("causal_level", np.uint8))
    causal_source = _lazy(lambda self: self._map("causal_source_id", np.uint16))
    mask = _lazy(lambda self: self.object_id > 0)

    @_lazy
    def visible(self) -> np.ndarray:
        """The invalid-side footprint of ``mask``: what a viewer can see."""
        observations = self._sample.observations
        shadow = (observations["shadow_source_id"]
                  if "shadow_source_id" in observations else None)
        return visible_violation(self.object_id, observations.segmentation,
                                 self.component_map, shadow)

    @_lazy
    def severity_map(self) -> np.ndarray:
        """float [T,H,W]. Painted from the per-object severity; stored only
        where painting cannot reproduce it (a shadow component)."""
        if self._sample._h5_has("/violation/maps/severity"):
            return np.asarray(self._sample._h5_read("/violation/maps/severity"), np.float32)
        return paint_severity(self.object_id, self._sample.observations.segmentation,
                              self._sample.objects.ids, self.severity, self.component_map)

    @_lazy
    def reference_mask(self) -> np.ndarray:
        """Where the violators lawfully are: their footprint in the valid twin."""
        twin = self._sample.twin
        shape = self._sample.observations.segmentation.shape
        if not self.present or twin is None:
            return np.zeros(shape, bool)
        ids = self.ids
        # Shadow components are annotated from the matched Cycles isolation
        # pass.  The valid segmentation contains the visible caster, not its
        # projected shadow, so using the generic body footprint here paints
        # the object instead of the lawful shadow in the MASK panel.
        if np.any(self.component_map == 2):
            observations = twin.observations
            if "shadow_strength" in observations and "shadow_source_id" in observations:
                return shadow_receiver_mask(observations.shadow_strength,
                                            observations.shadow_source_id,
                                            observations.segmentation, ids)
            return np.zeros(shape, bool)
        return reference_mask(twin.observations.segmentation, ids)

    @_lazy
    def timeline(self) -> Dict[str, np.ndarray]:
        """Clip-level ``[T]`` clocks (any violator) and severity (max)."""
        clocks = {clock: self[clock] for clock in CLOCKS}
        return sample_timeline(clocks, self.severity,
                               self._sample.objects.is_violator, self._T)

    def latent_grid(self, latent_frames: Optional[int] = None,
                    latent_hw: Optional[int] = None) -> Dict[str, np.ndarray]:
        video = self._sample.video
        return latent_grid(self.mask, self.severity_map,
                           latent_frames or video.latent_frames,
                           latent_hw or video.latent_hw)


class Energy(_Namespace):
    """Mechanical energy: scene totals ``[T]``, per object ``[N,T,...]``, and
    the per-pixel ``map`` ``[T,H,W]``. Units are on each HDF5 dataset."""

    KEYS = ("scene", "objects", "map")

    @_lazy
    def scene(self) -> Dict[str, np.ndarray]:
        return {key: self._sample._h5_read("/energy/scene/" + key)
                for key in self._sample._h5_keys("/energy/scene")}

    @_lazy
    def objects(self) -> Dict[str, np.ndarray]:
        out = {key: self._sample._h5_read("/energy/objects/" + key)
               for key in self._sample._h5_keys("/energy/objects")}
        if "momentum" in out:
            out["momentum_magnitude"] = np.linalg.norm(out["momentum"], axis=-1)
        return out

    @_lazy
    def map(self) -> Optional[np.ndarray]:
        return (self._sample._h5_read("/energy/map")
                if self._sample._h5_has("/energy/map") else None)


class Events(_Namespace):
    """Discrete events, one array per field (MOVi's ``collisions``)."""

    KEYS = ("collisions",)

    @_lazy
    def collisions(self) -> Dict[str, np.ndarray]:
        """frame [E], instances [E,2], force [E], position [E,3],
        contact_normal [E,3], image_position [E,2]."""
        return {key: self._sample._h5_read("/events/collisions/" + key)
                for key in self._sample._h5_keys("/events/collisions")}


def _plain(value: Any) -> Any:
    """A namespace as a nested dict. Blocks backed by ``sample.json`` give their
    metadata only (``video`` without ``rgb``, ``violation`` without its maps --
    name those explicitly); blocks that are all arrays give every array."""
    if not isinstance(value, _Namespace):
        return value
    if isinstance(value, (_Group, Energy, Events)):
        return {key: _plain(value[key]) for key in value.keys()}
    if isinstance(value, Objects):
        return {key: value[key] for key in value.STATIC}
    kind = type(value)
    return {key: _plain(value[key]) for key in value.keys() if not _is_lazy(kind, key)}


# ------------------------------------------------------------------- sample
class Sample:
    """One sample, read lazily. See the module docstring for the namespaces."""

    NAMESPACES = ("info", "video", "scene", "observations", "objects",
                  "violation", "energy", "events")

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
            raise ValueError("sample %s uses schema v%d; expected v%d -- regenerate it"
                             % (self.path, version, SCHEMA_VERSION))
        self._cache: Dict[str, Any] = {}
        self._h5 = None
        self._h5_pid: Optional[int] = None
        self._twin = twin
        self.selected_fields = tuple(selected_fields or ())

    @classmethod
    def from_dir(cls, path: str) -> "Sample":
        """A sample with its valid twin resolved from the canonical layout
        ``<root>/samples/<sample_uid>``."""
        sample = cls(path)
        if sample.info.is_valid:
            return sample
        samples_root = os.path.abspath(path)
        for _ in [part for part in sample.uid.split("/") if part]:
            samples_root = os.path.dirname(samples_root)
        twin_path = os.path.join(samples_root, *sample.info.valid_uid.split("/"))
        if os.path.realpath(twin_path) != os.path.realpath(path) \
                and os.path.exists(os.path.join(twin_path, SAMPLE_METADATA)):
            sample._twin = cls(twin_path)
        return sample

    # ---- namespaces ----------------------------------------------------------
    def _namespace(self, kind):
        return self._cached("ns:" + kind.__name__, lambda: kind(self))

    info = property(lambda self: self._namespace(Info))
    video = property(lambda self: self._namespace(Video))
    scene = property(lambda self: self._namespace(Scene))
    observations = property(lambda self: self._namespace(Observations))
    objects = property(lambda self: self._namespace(Objects))
    violation = property(lambda self: self._namespace(Violation))
    energy = property(lambda self: self._namespace(Energy))
    events = property(lambda self: self._namespace(Events))
    twin = property(lambda self: self._twin)
    #: The two most-used identifiers, for logs and dictionary keys.
    uid = property(lambda self: self.info.uid)
    pair_uid = property(lambda self: self.info.pair_uid)

    @property
    def divergence(self) -> np.ndarray:
        """``|valid - invalid|`` in pixel space, for inspection only.

        Deliberately NOT under ``violation``: it diverges everywhere downstream
        of the event, and it is never a training target."""
        if self.info.is_valid or self.twin is None:
            return np.zeros(self.observations.segmentation.shape, np.float32)
        return self._cached("divergence", lambda: divergence(
            self.twin.video.rgb, self.video.rgb))

    def resolve(self, field: str) -> Any:
        """The value a dotted selector names, e.g. ``"violation.mask"``."""
        head, _, rest = field.partition(".")
        if head == "divergence" and not rest:
            return self.divergence
        if head not in self.NAMESPACES:
            raise KeyError("unknown field %r; namespaces: %s"
                           % (field, ", ".join(self.NAMESPACES)))
        value = getattr(self, head)
        for part in rest.split(".") if rest else ():
            value = value[part]
        return _plain(value)

    def to_dict(self, fields: Optional[Sequence[str]] = None) -> Dict[str, Any]:
        """A nested dict of the selected fields, always with ``info``.

        ``to_dict(["violation.mask", "video.rgb"])`` ->
        ``{"info": {...}, "violation": {"mask": ...}, "video": {"rgb": ...}}``.
        """
        selected = tuple(fields if fields is not None else self.selected_fields)
        out: Dict[str, Any] = {"info": self.resolve("info")}
        for field in selected:
            parts = field.split(".")
            target = out
            for part in parts[:-1]:
                target = target.setdefault(part, {})
            value = self.resolve(field)
            if isinstance(value, dict) and isinstance(target.get(parts[-1]), dict):
                target[parts[-1]].update(value)
            else:
                target[parts[-1]] = value
        return out

    # ---- storage -------------------------------------------------------------
    def _cached(self, key: str, make: Callable[[], Any]) -> Any:
        if key not in self._cache:
            self._cache[key] = make()
        return self._cache[key]

    def _h5_file(self):
        try:
            import h5py
        except ImportError as exc:
            raise ImportError(
                "PhysLoc schema v%d requires h5py (interpreter: %s)"
                % (SCHEMA_VERSION, sys.executable)) from exc
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
        def read():
            h5 = self._h5_file()
            return sorted(h5[path].keys()) if path in h5 else []
        return self._cached("keys:" + path, read)

    def _h5_read(self, path: str) -> Any:
        def read():
            h5 = self._h5_file()
            if path not in h5:
                raise KeyError(path)
            return h5[path][()]
        return self._cached("h5:" + path, read)

    def release(self) -> None:
        """Drop cached arrays and close the HDF5 handle."""
        self._cache.clear()
        if self._h5 is not None:
            self._h5.close()
        self._h5 = None
        self._h5_pid = None

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_h5"] = None
        state["_h5_pid"] = None
        state["_cache"] = {}
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
        return sample.scene.prompt if sample else None


def _as_set(value: Any) -> set:
    if isinstance(value, (str, int, float)) or value is None:
        return {None if value is None else str(value)}
    return {None if item is None else str(item) for item in value}


def _check_field(field: str) -> bool:
    head = field.split(".", 1)[0]
    return field in FIELDS or (head in Sample.NAMESPACES and "." not in field)


class PhysLocDataset:
    """All schema-v4 samples under ``root``.

    ``ds[i]`` is a lazy `Sample` (``unit="sample"``) or a pair dict
    (``unit="pair"``); ``fields=`` picks what ``to_dict``/``collate`` carry.
    """

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
        bad = [field for field in selected if not _check_field(field)]
        if bad:
            raise KeyError("unknown field(s) %s; see loader.FIELDS" % bad)
        if split is not None:
            filters["split"] = split
        self.root = os.path.abspath(root)
        self.selected = selected
        self.unit = unit
        paths = sorted(glob.glob(os.path.join(
            self.root, "samples", "**", SAMPLE_METADATA), recursive=True))
        if not paths:
            raise FileNotFoundError(
                "no schema-v%d samples under %s; expected samples/<sample_uid>/%s"
                % (SCHEMA_VERSION, self.root, SAMPLE_METADATA))
        wanted = {key: _as_set(value) for key, value in filters.items()}
        self.samples: List[Sample] = []
        for json_path in paths:
            sample = Sample(os.path.dirname(json_path), selected_fields=selected)
            info, scene = sample.info, sample.scene
            values = {"label": info.label, "family": scene.family,
                      "scenario": scene.scenario, "level": scene.level,
                      "condition": scene.condition,
                      "severity": sample.violation.severity_bin,
                      "seed": info.seed, "split": info.split}
            if all(str(values.get(key)) in choices for key, choices in wanted.items()):
                self.samples.append(sample)
        self._by_uid = {sample.uid: sample for sample in self.samples}
        valid = {sample.pair_uid: sample for sample in self.samples
                 if sample.info.is_valid}
        for sample in self.samples:
            if not sample.info.is_valid:
                sample._twin = (self._by_uid.get(sample.info.valid_uid)
                                or valid.get(sample.pair_uid))
        grouped: Dict[str, Pair] = {}
        for sample in self.samples:
            pair = grouped.setdefault(sample.pair_uid, Pair(sample.pair_uid))
            if sample.info.is_valid:
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


# ------------------------------------------------------------------ batching
def _object_axis(path: str) -> bool:
    return FIELDS.get(path, "").startswith("N")


def _collate(rows: Sequence[Any], path: str = "") -> Any:
    first = rows[0]
    if isinstance(first, dict) and all(isinstance(row, dict) for row in rows):
        keys = [key for key in first if all(key in row for row in rows)]
        return {key: _collate([row[key] for row in rows],
                              (path + "." + key) if path else key) for key in keys}
    if isinstance(first, np.ndarray) and all(isinstance(row, np.ndarray) for row in rows):
        if _object_axis(path) and first.ndim:
            width = max(row.shape[0] for row in rows)
            dtype = np.result_type(*[row.dtype for row in rows])
            fill = np.nan if np.issubdtype(dtype, np.floating) else 0
            out = np.full((len(rows), width) + first.shape[1:], fill, dtype)
            for i, row in enumerate(rows):
                out[i, :row.shape[0]] = row
            return out
        if all(row.shape == first.shape for row in rows):
            return np.stack(rows)
    return list(rows)


def collate(samples: Sequence[Any]) -> Dict[str, Any]:
    """Batch samples (via their ``to_dict``) or plain nested dicts.

    Nested dicts collate key by key; equal-shape arrays stack; per-object
    arrays (axes ``N,...`` in `FIELDS`) are padded to the batch's largest N,
    and ``objects.valid`` [B,N] says which rows are real.
    """
    if not samples:
        return {}
    rows = [sample.to_dict() if isinstance(sample, Sample) else sample
            for sample in samples]
    counts = [_object_count(row) for row in rows]
    out = _collate(rows)
    if all(count is not None for count in counts):
        width = max(counts)
        out.setdefault("objects", {})["valid"] = np.stack(
            [np.arange(width) < count for count in counts])
    return out


def _object_count(row: Dict[str, Any], path: str = "") -> Optional[int]:
    """N of a ``to_dict`` row: the leading axis of its first per-object array,
    or None when it carries none -- so batches of plain dicts (a DataLoader's
    worker output) get ``objects.valid`` exactly as batches of Samples do."""
    for key, value in row.items():
        here = (path + "." + key) if path else key
        if isinstance(value, dict):
            found = _object_count(value, here)
            if found is not None:
                return found
        elif isinstance(value, np.ndarray) and value.ndim and _object_axis(here):
            return int(value.shape[0])
    return None


def torch_dataset(dataset: PhysLocDataset):
    """A ``torch.utils.data.Dataset`` returning ``to_dict()`` of each sample;
    pass ``collate_fn=loader.collate``."""
    from torch.utils.data import Dataset

    class _TorchDataset(Dataset):
        def __len__(self):
            return len(dataset)

        def __getitem__(self, index):
            item = dataset[index]
            return item.to_dict() if isinstance(item, Sample) else item

    return _TorchDataset()
