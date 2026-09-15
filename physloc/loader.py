"""Load a PhysLoc dataset -- one file, numpy and the standard library only.

Deliberately imports nothing from `physloc`, so a consumer in another
environment (LikePhys-PhysLoc, a training repo) can import this file by path or
copy it, and read a release without installing the generator.

Two on-disk forms are read with the same `Clip`:

    <root>/clips/<release>/<level>/<scenario>/<seed>_<condition>/<valid|invalid_<family>_<bin>>/
    <root>/shards/*.tar          the `physloc export` form, read in place

**Schema v2 stores only what cannot be derived.** An invalid clip ships two
annotation files, `masks.npz` and `objects.npz`; everything else a model trains
or is scored on is a function of them and of `segmentations.npz`, and this file
holds the one implementation of each function:

    violation_mask     = violation > 0
    visible_violation  = (violation > 0) & (segmentations == violation)
    severity_map       = objects.severity[k, t] painted on violator k's visible
                         pixels, or on its `violation` pixels on a frame where
                         it has none (a body that vanished)
    reference_mask     = isin(valid twin's segmentations, violator ids)
    timeline           = any over violators (max for severity; `occluded` is
                         the primary violator's row)
    latent_grid        = masks and severity reduced to the VAE token grid

    >>> ds = PhysLocDataset("out/physloc_v0_mini", family="permanence")
    >>> clip = ds.clips[0]
    >>> clip.video.shape, clip.violation_mask.shape, clip.objects["severity"].shape
    ((25, 128, 128, 3), (25, 128, 128), (1, 25))
"""
from __future__ import annotations

import glob
import io
import json
import os
import tarfile
import tempfile
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np

SCHEMA_VERSION = 2

METADATA = "metadata.json"
VIDEO = "video.mp4"
MASKS = "masks.npz"
OBJECTS = "objects.npz"
SEGMENTATIONS = "segmentations.npz"
INSTANCES = "instances.npz"
TRAJECTORY = "traj.npz"
BODIES = "bodies.npz"
ENERGY = "energy.npz"
ENERGY_MAP = "energy_map.npz"

#: Dense renderer passes: file stem == array key.
PASSES = ("depth", "forward_flow", "backward_flow", "normal", "object_coordinates")

#: Per-violator boolean timelines in `objects.npz`, each [K, T].
CLOCKS = ("active", "intervening", "consequence", "observable", "occluded")

#: Depth marks the background with a huge sentinel (~1.1e10), not inf.
DEPTH_BACKGROUND = 1e6

#: What `Clip.sample()` returns when not told otherwise: what a model trains and
#: is scored on. Dense passes are never decoded unless asked for.
DEFAULT_KEYS = ("video", "violation_mask", "severity_map", "causal", "timeline",
                "objects")


# ---------------------------------------------------------------------------
# Derivations -- pure functions, the only implementation of each
# ---------------------------------------------------------------------------

def visible_violation(violation: np.ndarray, segmentations: np.ndarray) -> np.ndarray:
    """[T,H,W] bool: violation pixels where the violator itself is rendered in
    THIS video. The part of `violation_mask` a model can actually see."""
    return (violation > 0) & (segmentations == violation)


def paint_severity(violation: np.ndarray, segmentations: np.ndarray,
                   ids: Sequence[int], severity: np.ndarray) -> np.ndarray:
    """[T,H,W] float32 from the per-violator, per-frame table `severity` [K,T].

    Severity is one value per (violator, frame), painted over the violator's
    visible pixels. On a frame where a violator has none -- it vanished -- the
    value goes on its `violation` pixels instead, which on such a frame are its
    lawful footprint.
    """
    out = np.zeros(violation.shape, np.float32)
    for k, vid in enumerate(ids):
        own = violation == vid
        seen = own & (segmentations == vid)
        has = seen.reshape(len(seen), -1).any(axis=1)
        where = np.where(has[:, None, None], seen, own)
        value = np.asarray(severity[k], np.float32)[:, None, None]
        out = np.where(where, value, out)
    return out


def reference_mask(valid_segmentations: np.ndarray, ids: Sequence[int]) -> np.ndarray:
    """[T,H,W] bool: where the violators lawfully are, from the valid twin."""
    return np.isin(valid_segmentations, np.asarray(list(ids), valid_segmentations.dtype))


def clip_timeline(objects: Dict[str, np.ndarray], num_frames: int) -> Dict[str, np.ndarray]:
    """Clip-level [T] timelines from the per-violator [K,T] rows."""
    K = len(objects.get("ids", ()))
    out = {}
    for key in CLOCKS:
        rows = np.asarray(objects[key], bool) if K else np.zeros((0, num_frames), bool)
        if key == "occluded":
            out[key] = rows[0] if K else np.zeros(num_frames, bool)
        else:
            out[key] = rows.any(axis=0) if K else np.zeros(num_frames, bool)
    sev = np.asarray(objects["severity"], np.float32) if K else None
    out["severity"] = sev.max(axis=0) if K else np.zeros(num_frames, np.float32)
    return out


def temporal_bins(num_frames: int, latent_frames: int) -> List[np.ndarray]:
    """Source frames per latent frame under 4x VAE binning (latent 0 <- frame 0,
    latent i>0 <- frames 4i-3..4i). Exact only for a 4k+1 clip."""
    if (num_frames - 1) % 4 or (num_frames - 1) // 4 + 1 != latent_frames:
        raise ValueError("num_frames=%d does not bin to %d latent frames (need 4k+1)"
                         % (num_frames, latent_frames))
    return [np.array([0])] + [np.arange(4 * i - 3, 4 * i + 1)
                              for i in range(1, latent_frames)]


def _block(x: np.ndarray, hw: int, how: str) -> np.ndarray:
    H, W = x.shape
    if H % hw or W % hw:
        raise ValueError("%dx%d does not divide into %d" % (H, W, hw))
    b = x.reshape(hw, H // hw, hw, W // hw)
    return b.max(axis=(1, 3)) if how == "max" else b.mean(axis=(1, 3))


def latent_grid(mask: np.ndarray, severity: np.ndarray, latent_frames: int,
                latent_hw: int) -> Dict[str, np.ndarray]:
    """Mask and severity on the token grid, time-major [F, h, w].

    A mask cell is set if any source pixel in any source frame is; severity is
    reduced by both max and mean, since peak and average are different questions.
    """
    bins = temporal_bins(mask.shape[0], latent_frames)
    shape = (latent_frames, latent_hw, latent_hw)
    m = np.zeros(shape, bool)
    smax = np.zeros(shape, np.float32)
    smean = np.zeros(shape, np.float32)
    sev = np.asarray(severity, np.float32)
    for i, src in enumerate(bins):
        m[i] = _block(mask[src].any(axis=0).astype(np.float32), latent_hw, "max") > 0
        smax[i] = _block(sev[src].max(axis=0), latent_hw, "max")
        smean[i] = _block(sev[src].mean(axis=0), latent_hw, "mean")
    return {"mask": m, "severity_max": smax, "severity_mean": smean}


def divergence(video_valid: np.ndarray, video_invalid: np.ndarray) -> np.ndarray:
    """[T,H,W] float32 in [0,1]: |valid - invalid| per pixel, mean over RGB.

    Diverges everywhere downstream of the event. For inspection only -- it is
    NOT the violation region and never a training target.
    """
    a = np.asarray(video_valid, np.float32)
    b = np.asarray(video_invalid, np.float32)
    return np.abs(a - b).mean(axis=-1) / 255.0


# ---------------------------------------------------------------------------
# Where a clip's files come from: a directory, or members of a tar shard
# ---------------------------------------------------------------------------

class _DirSource:
    def __init__(self, path: str):
        self.path = path

    def has(self, name: str) -> bool:
        return os.path.exists(os.path.join(self.path, name))

    def read(self, name: str) -> bytes:
        with open(os.path.join(self.path, name), "rb") as fh:
            return fh.read()

    def npz(self, name: str, allow_pickle: bool = False):
        return np.load(os.path.join(self.path, name), allow_pickle=allow_pickle)

    def video_path(self) -> str:
        return os.path.join(self.path, VIDEO)


class _TarSource:
    """One clip's members inside tar shards, read by offset -- nothing is extracted.

    A clip can span two tars, its core shard and the optional passes shard, so
    every member remembers which tar it is in."""

    def __init__(self, members: Dict[str, tuple]):
        self.members = members           # file name -> (tar path, data offset, size)

    def has(self, name: str) -> bool:
        return name in self.members

    def read(self, name: str) -> bytes:
        path, offset, size = self.members[name]
        with open(path, "rb") as fh:
            fh.seek(offset)
            return fh.read(size)

    def npz(self, name: str, allow_pickle: bool = False):
        return np.load(io.BytesIO(self.read(name)), allow_pickle=allow_pickle)

    def video_path(self) -> str:
        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        tmp.write(self.read(VIDEO))
        tmp.close()
        return tmp.name


def _decode_video(path: str) -> np.ndarray:
    import imageio.v3 as iio
    frames = iio.imread(path)
    return np.asarray(frames[..., :3], np.uint8)


# ---------------------------------------------------------------------------
# Clip
# ---------------------------------------------------------------------------

class Clip:
    """One valid or invalid clip. Every array is loaded on first access and cached."""

    def __init__(self, source, twin: Optional["Clip"] = None):
        self._src = source
        self._cache: Dict[str, Any] = {}
        self._twin = twin

    @classmethod
    def from_dir(cls, path: str) -> "Clip":
        return cls(_DirSource(path))

    # ---- identity --------------------------------------------------------
    def _cached(self, key: str, make: Callable[[], Any]) -> Any:
        if key not in self._cache:
            self._cache[key] = make()
        return self._cache[key]

    @property
    def metadata(self) -> Dict[str, Any]:
        return self._cached("metadata", lambda: json.loads(self._src.read(METADATA)))

    @property
    def info(self) -> Dict[str, Any]:
        return self.metadata["metadata"]

    uid = property(lambda self: self.info["clip_uid"])
    pair_uid = property(lambda self: self.info["pair_uid"])
    label = property(lambda self: self.info["label"])
    is_valid = property(lambda self: self.info["label"] == "valid")
    family = property(lambda self: self.info.get("family"))
    scenario = property(lambda self: self.info["scenario"])
    condition = property(lambda self: self.info.get("condition"))
    prompt = property(lambda self: self.info.get("prompt"))
    num_frames = property(lambda self: int(self.info["num_frames"]))
    fps = property(lambda self: int(self.info["frame_rate"]))

    @property
    def level(self) -> str:
        c = self.info.get("complexity")
        return c.get("name") if isinstance(c, dict) else c

    @property
    def severity_bin(self) -> Optional[str]:
        v = self.metadata.get("violation") or {}
        return (v.get("intervention") or {}).get("severity_bin")

    @property
    def violator_ids(self) -> List[int]:
        v = self.metadata.get("violation") or {}
        return [int(o["instance_id"]) for o in v.get("violators", [])]

    @property
    def twin(self) -> Optional["Clip"]:
        """The valid clip of this pair (None for a valid clip or a lone clip)."""
        if self._twin is None and not self.is_valid and isinstance(self._src, _DirSource):
            path = os.path.join(os.path.dirname(self._src.path), "valid")
            if os.path.exists(os.path.join(path, METADATA)):
                self._twin = Clip.from_dir(path)
        return self._twin

    def _npz(self, name: str, key: Optional[str] = None, allow_pickle: bool = False):
        def load():
            with self._src.npz(name, allow_pickle=allow_pickle) as z:
                return {k: z[k] for k in z.files}
        arrays = self._cached("npz:" + name, load)
        return arrays if key is None else arrays[key]

    # ---- what the renderer made ------------------------------------------
    @property
    def video_path(self) -> str:
        """A path to this clip's `video.mp4`, for tools that open the file
        themselves -- a temporary copy when the clip is read from a shard."""
        return self._src.video_path()

    @property
    def video(self) -> np.ndarray:
        """uint8 [T,H,W,3]."""
        return self._cached("video", lambda: _decode_video(self._src.video_path()))

    @property
    def segmentations(self) -> np.ndarray:
        """uint16 [T,H,W], instance ids, 0 = background."""
        return self._npz(SEGMENTATIONS, "segmentations")

    def pass_(self, name: str) -> np.ndarray:
        """A dense pass: depth [T,H,W,1] (metres), forward_flow / backward_flow
        [T,H,W,2] ((row, col) pixels), normal / object_coordinates uint16 [T,H,W,3]."""
        if name not in PASSES:
            raise KeyError("unknown pass %r; one of %s" % (name, PASSES))
        return self._npz(name + ".npz", name)

    def has(self, name: str) -> bool:
        return self._src.has(name)

    @property
    def instances(self) -> Dict[str, np.ndarray]:
        """MOVi's per-instance tensors, instance-major: positions [k,T,3], bboxes
        [k,T,4] (ymin,xmin,ymax,xmax in [0,1], NaN when unseen), bboxes_3d [k,T,8,3],
        image_positions [k,T,2] ((x,y) in [0,1]), visibility [k,T], ids [k]."""
        return self._npz(INSTANCES)

    @property
    def camera(self) -> Dict[str, np.ndarray]:
        """`K` [3,3] (normalised: multiply the first two rows by W, H for pixels),
        `positions` [T,3], `quaternions` [T,4] (w,x,y,z), plus the lens scalars."""
        cam = self.metadata["camera"]
        return {k: (np.asarray(v, np.float64) if isinstance(v, list) else v)
                for k, v in cam.items()}

    @property
    def trajectory(self) -> Dict[str, np.ndarray]:
        """The simulator rollout, time-major: pos [T,B,3], quat, lin_vel, present, ..."""
        return self._npz(TRAJECTORY, allow_pickle=True)

    @property
    def bodies(self) -> Dict[str, np.ndarray]:
        return self._npz(BODIES)

    @property
    def energy(self) -> Dict[str, np.ndarray]:
        return self._npz(ENERGY)

    @property
    def energy_map(self) -> np.ndarray:
        return self._npz(ENERGY_MAP, "energy")

    # ---- stored annotations ----------------------------------------------
    def _zeros(self, dtype) -> np.ndarray:
        return np.zeros(self.segmentations.shape, dtype)

    @property
    def violation(self) -> np.ndarray:
        """uint16 [T,H,W]: 0 = no violation, else the violator's instance id.
        The union over BOTH twins, so a vanished body still has its pixels."""
        return self._zeros(np.uint16) if self.is_valid else self._npz(MASKS, "violation")

    @property
    def causal(self) -> np.ndarray:
        """uint8 [T,H,W]: 0 nothing, 1 a violator, 2 a body it affected."""
        return self._zeros(np.uint8) if self.is_valid else self._npz(MASKS, "causal")

    @property
    def causal_source(self) -> np.ndarray:
        """uint16 [T,H,W]: the violator each causal pixel belongs to."""
        return self._zeros(np.uint16) if self.is_valid else self._npz(MASKS, "causal_source")

    @property
    def objects(self) -> Dict[str, np.ndarray]:
        """Per violator, per frame: ids [K], severity [K,T], the clocks
        (active, intervening, consequence, observable, occluded) [K,T] bool,
        residual [K,T] (raw law residual) and score [K,T] (residual on 0..1,
        before being restricted to visible evidence). K = 0 on a valid clip."""
        if not self.is_valid:
            return self._npz(OBJECTS)
        T = self.num_frames
        out = {"ids": np.zeros((0,), np.int32), "severity": np.zeros((0, T), np.float32),
               "residual": np.zeros((0, T), np.float32), "score": np.zeros((0, T), np.float32)}
        out.update({k: np.zeros((0, T), bool) for k in CLOCKS})
        return out

    # ---- derived annotations ---------------------------------------------
    @property
    def violation_mask(self) -> np.ndarray:
        """bool [T,H,W]: THE localisation target."""
        return self.violation > 0

    @property
    def visible_violation(self) -> np.ndarray:
        """bool [T,H,W]: violation pixels visible in this video."""
        return self._cached("visible_violation",
                            lambda: visible_violation(self.violation, self.segmentations))

    @property
    def severity_map(self) -> np.ndarray:
        """float32 [T,H,W] in [0,1]: how badly, per pixel."""
        return self._cached("severity_map", lambda: paint_severity(
            self.violation, self.segmentations, self.objects["ids"],
            self.objects["severity"]))

    @property
    def reference_mask(self) -> np.ndarray:
        """bool [T,H,W]: where the violators lawfully are (from the valid twin)."""
        if self.is_valid or self.twin is None:
            return self._zeros(bool)
        return self._cached("reference_mask", lambda: reference_mask(
            self.twin.segmentations, self.objects["ids"]))

    @property
    def timeline(self) -> Dict[str, np.ndarray]:
        """Clip-level [T]: the five clocks and `severity` (peak per frame)."""
        return clip_timeline(self.objects, self.num_frames)

    def latent_grid(self, latent_frames: Optional[int] = None,
                    latent_hw: Optional[int] = None) -> Dict[str, np.ndarray]:
        """Mask and severity reduced to the video-VAE token grid [F,h,w]."""
        F = latent_frames or int(self.info.get("latent_frames")
                                 or (self.num_frames - 1) // 4 + 1)
        hw = latent_hw or int(self.info.get("latent_hw")
                              or self.segmentations.shape[-1] // 16)
        return latent_grid(self.violation_mask, self.severity_map, F, hw)

    @property
    def divergence(self) -> np.ndarray:
        """float32 [T,H,W]: |valid - invalid|. Inspection only, never a target."""
        if self.is_valid or self.twin is None:
            return np.zeros(self.segmentations.shape, np.float32)
        return self._cached("divergence", lambda: divergence(self.twin.video, self.video))

    # ---- batching --------------------------------------------------------
    def sample(self, keys: Sequence[str] = DEFAULT_KEYS) -> Dict[str, Any]:
        """A dict of the requested arrays plus `uid`, `label` and `metadata`.

        A key is any array property above (`video`, `violation_mask`, ...), a
        pass name (`depth`, ...) or `latent_grid`.
        """
        out: Dict[str, Any] = {"uid": self.uid, "label": self.label,
                               "metadata": self.metadata}
        for key in keys:
            if key in PASSES:
                out[key] = self.pass_(key)
            elif key == "latent_grid":
                out[key] = self.latent_grid()
            else:
                out[key] = getattr(self, key)
        return out

    def release(self) -> None:
        """Drop every cached array (keeps nothing but the file handles' paths)."""
        self._cache.clear()

    def __repr__(self) -> str:
        return "Clip(%s)" % self.uid


class Pair:
    """One scene: its valid clip and every invalid clip made from it.

    A plain class, not a dataclass: `@dataclass` looks its module up in
    `sys.modules`, and a loader imported by path is not always registered there.
    """

    def __init__(self, pair_uid: str, valid: Optional[Clip] = None,
                 invalids: Optional[List[Clip]] = None):
        self.pair_uid = pair_uid
        self.valid = valid
        self.invalids = list(invalids or [])

    @property
    def prompt(self) -> Optional[str]:
        return (self.valid or self.invalids[0]).prompt

    def __repr__(self) -> str:
        return "Pair(%s, %d invalid)" % (self.pair_uid, len(self.invalids))


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

#: Filters readable from a clip's path alone, so indexing a release never opens
#: a metadata file: clips/<release>/<level>/<scenario>/<seed>_<condition>/<leaf>.
def _path_fields(uid: str) -> Dict[str, Optional[str]]:
    parts = uid.split("/")
    release, level, scenario, seedcond, leaf = parts[-5:]
    seed, _, condition = seedcond.partition("_")
    if leaf == "valid":
        label, family, sev = "valid", None, None
    else:
        body = leaf[len("invalid_"):]
        family, _, sev = body.rpartition("_")
        label = "invalid"
    return {"release": release, "level": level, "scenario": scenario, "seed": seed,
            "condition": condition.replace("-", "+"), "label": label,
            "family": family, "severity_bin": sev,
            "pair_uid": "/".join(parts[-5:-1])}


def _index_dirs(root: str) -> List[tuple]:
    base = os.path.join(root, "clips") if os.path.isdir(os.path.join(root, "clips")) else root
    rows = []
    for mp in sorted(glob.glob(os.path.join(base, "**", METADATA), recursive=True)):
        cdir = os.path.dirname(mp)
        uid = os.path.relpath(cdir, base).replace(os.sep, "/")
        rows.append((uid, _DirSource(cdir)))
    return rows


def _index_tars(paths: Iterable[str]) -> List[tuple]:
    grouped: Dict[str, Dict[str, tuple]] = {}
    for path in sorted(paths):
        with tarfile.open(path) as tf:
            for m in tf.getmembers():
                if not m.isfile():
                    continue
                key, _, name = m.name.partition(".")
                grouped.setdefault(key, {})[name] = (path, m.offset_data, m.size)
    return [(key.replace("__", "/"), _TarSource(files))
            for key, files in sorted(grouped.items())]


class PhysLocDataset:
    """Every clip under `root`, filtered, indexable, and grouped into pairs.

    `root` is a generated release (it has `clips/`) or an exported one (it has
    `shards/`, read in place; the optional `passes-*` shards are merged when
    present). Filters take a value or a collection of values:

        PhysLocDataset(root, label="invalid", family=("permanence", "solidity"),
                       level="L0", condition="standard", split="main")

    `split` reads `splits/<name>.txt` (clip or pair uids) and so needs an
    exported root.
    """

    FILTERS = ("label", "family", "scenario", "level", "condition", "severity_bin")

    def __init__(self, root: str, keys: Sequence[str] = DEFAULT_KEYS,
                 split: Optional[str] = None, **filters: Any):
        unknown = set(filters) - set(self.FILTERS)
        if unknown:
            raise TypeError("unknown filter(s) %s; known: %s"
                            % (sorted(unknown), self.FILTERS))
        self.root = root
        self.keys = tuple(keys)
        shards = sorted(glob.glob(os.path.join(root, "shards", "*.tar")))
        rows = _index_tars(shards) if shards else _index_dirs(root)
        want = {k: ({v} if isinstance(v, str) or v is None else set(v))
                for k, v in filters.items()}
        in_split = None
        if split is not None:
            # `physloc export` lists clip uids; a pair uid names every clip of
            # its pair. Pairs never straddle a split, so either selects the same.
            with open(os.path.join(root, "splits", "%s.txt" % split)) as fh:
                in_split = {line.strip() for line in fh if line.strip()}

        self.clips: List[Clip] = []
        self._fields: List[Dict[str, Optional[str]]] = []
        for uid, source in rows:
            f = _path_fields(uid)
            if any(f.get(k) not in v for k, v in want.items()):
                continue
            if in_split is not None and uid not in in_split and f["pair_uid"] not in in_split:
                continue
            self.clips.append(Clip(source))
            self._fields.append(f)
        self._link_twins()

    def _link_twins(self) -> None:
        valid = {f["pair_uid"]: c for c, f in zip(self.clips, self._fields)
                 if f["label"] == "valid"}
        for c, f in zip(self.clips, self._fields):
            if f["label"] == "invalid" and f["pair_uid"] in valid:
                c._twin = valid[f["pair_uid"]]

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        return self.clips[i].sample(self.keys)

    def fields(self, i: int) -> Dict[str, Optional[str]]:
        """Level, scenario, seed, condition, label, family and bin of clip `i`,
        read from its path."""
        return dict(self._fields[i])

    def pairs(self) -> List[Pair]:
        out: Dict[str, Pair] = {}
        for c, f in zip(self.clips, self._fields):
            p = out.setdefault(f["pair_uid"], Pair(f["pair_uid"], None))
            if f["label"] == "valid":
                p.valid = c
            else:
                p.invalids.append(c)
        return list(out.values())


def collate(samples: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Batch samples from `Clip.sample`. Arrays of one shape are stacked;
    `objects` is padded to the batch's largest K with `objects["valid"]` [B,K]
    marking real rows; anything else is kept as a list."""
    out: Dict[str, Any] = {}
    for key in samples[0]:
        values = [s[key] for s in samples]
        if key == "objects":
            out[key] = _pad_objects(values)
        elif isinstance(values[0], dict):
            if all(isinstance(v, np.ndarray) for v in values[0].values()):
                out[key] = collate(values)
            else:
                out[key] = values
        elif (isinstance(values[0], np.ndarray)
              and all(v.shape == values[0].shape for v in values)):
            out[key] = np.stack(values)
        else:
            out[key] = values
    return out


def _pad_objects(tables: Sequence[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    K = max(len(t["ids"]) for t in tables)
    out: Dict[str, np.ndarray] = {}
    for key in tables[0]:
        rows = []
        for t in tables:
            a = np.asarray(t[key])
            pad = [(0, K - a.shape[0])] + [(0, 0)] * (a.ndim - 1)
            rows.append(np.pad(a, pad))
        out[key] = np.stack(rows)
    out["valid"] = np.stack([np.arange(K) < len(t["ids"]) for t in tables])
    return out


def torch_dataset(dataset: PhysLocDataset):
    """Wrap a `PhysLocDataset` as a `torch.utils.data.Dataset` (needs torch).
    Use with `DataLoader(..., collate_fn=collate)`."""
    from torch.utils.data import Dataset

    class _Torch(Dataset):
        def __len__(self):
            return len(dataset)

        def __getitem__(self, i):
            return dataset[i]

    return _Torch()
