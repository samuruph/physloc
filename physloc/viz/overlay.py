"""Annotation visualiser -- every annotation a clip ships, drawn on its video.

One renderer behind three front-ends: `overlay.mp4` (written by `generate`, the
nine default panels), `test_dataset_loader.py --render` (any layers and panels)
and the browser viewer in `viz/gui.py`, which asks for one frame at a time.

Two kinds of thing are drawn, and they are never mixed:

* **layers** go ON the RGB panel and combine freely -- the violation, its
  visible part, the reference outline, severity, causal, 2D and 3D boxes,
  centres, velocities, labels and collision events. They are outlines or
  translucent, so several read at once.
* **panels** go BESIDE it -- the valid twin, segmentation, depth, both flows,
  normals, object coordinates, energy, the mask / severity / causal views,
  divergence and a top-down camera map. Each one covers the whole frame, so
  drawing it over the video would hide the thing it annotates.

    +---------------------------------------------------------------------+
    | domain / family / scenario   L0 strong standard  [o] ACTIVE  f 7/24  |
    +-------------+-------------+-------------+-------------+-------------+
    | RGB+layers  | ENERGY      | SEGMENTATION| DEPTH       | FLOW        |
    +-------------+-------------+-------------+-------------+-------------+
    | MASK        | SEVERITY    | CAUSAL      | DIVERGENCE  |             |
    +-------------+-------------+-------------+-------------+-------------+
    | legend: one colour per violator, then the layers drawn               |
    | clocks: intervening / consequence / observable                       |
    | one row per violator (active, observable, occluded) + severity lane  |
    | frame ticks, t_event / t_applied / t_obs / t_end, playhead           |
    +---------------------------------------------------------------------+

Reads ONLY through `physloc/loader.py`, so every picture is also a check that
the loader derives what the pipeline meant. mp4 only -- no image files.
"""
from __future__ import annotations

import math
import os
from typing import Dict, List, Optional, Sequence

import numpy as np

from .. import loader
from ..annotate import layout
from ..annotate import movi
from . import video as vid

PANEL = 288                      # each panel is rendered at this size
HEADER = 34
LEGEND = 24
PAD = 6
COLUMNS = 5
CLOCK_ROW = 9
VIOLATOR_ROW = 11
SEVERITY_LANE = 34
#: One colour per violator, used identically by every layer, the legend and the
#: timeline, so "which object is this" never needs a second lookup.
C_VIOLATORS = ((255, 120, 200), (120, 230, 255), (200, 255, 120),
               (255, 200, 90), (180, 150, 255), (255, 150, 120))

C_BG = (18, 18, 22)
C_TEXT = (232, 232, 238)
C_DIM = (150, 150, 160)
C_MASK = (255, 70, 70)
C_ACTIVE = (255, 60, 60)
C_OBS = (250, 185, 60)
C_CAUSAL1 = (255, 70, 70)
C_CAUSAL2 = (90, 160, 255)
C_REF = (90, 235, 120)      # "where it should be", from the valid twin
C_EVENT = (255, 230, 90)
C_BLANK = (26, 26, 32)

#: Layers drawn on the RGB panel, in drawing order, with what each one shows.
LAYERS: Dict[str, str] = {
    "violation": "violation mask (both twins), one colour per violator",
    "visible": "the part of the violation visible in this video (white outline)",
    "severity": "severity heat over the violators",
    "causal": "causal mask: violators (red outline), affected bodies (blue)",
    "reference": "where the violators lawfully are, from the valid twin (green)",
    "bbox2d": "2D boxes from the segmentation",
    "bbox3d": "3D boxes projected through the camera",
    "centers": "projected object centres (hollow when not visible)",
    "velocity": "velocity, projected 1/3 s ahead",
    "labels": "instance name and id (violators marked !)",
    "events": "collisions on this frame",
}

#: Panels, in the order a caller lists them, with their titles.
PANELS: Dict[str, str] = {
    "rgb": "RGB",
    "valid": "VALID TWIN",
    "energy": "ENERGY  visible scene + per-object map",
    "segmentation": "SEGMENTATION  instance ids",
    "depth": "DEPTH  metres, near->far",
    "flow": "FORWARD FLOW  hue=direction val=speed",
    "backward_flow": "BACKWARD FLOW",
    "normal": "NORMALS",
    "object_coordinates": "OBJECT COORDINATES",
    "mask": "MASK  colour=violator  green=should-be",
    "severity": "SEVERITY MAP",
    "causal": "CAUSAL MASK",
    "divergence": "DIVERGENCE (not GT)",
    "camera": "CAMERA  top-down, metres",
}

#: What `generate` writes as `overlay.mp4`: evidence first, then the annotation
#: derived from it -- the same nine, in the same order, as the grid and sheet.
DEFAULT_PANELS = ("rgb", "energy", "segmentation", "depth", "flow", "mask",
                  "severity", "causal", "divergence")
# Keep generated overlays unobstructed: optional geometry layers (especially
# bbox2d) can cross the cast-shadow evidence or make a growing actor look like
# a detection box. They remain available when explicitly requested in the GUI.
DEFAULT_LAYERS: Sequence[str] = ()

#: Bodies too big to box: the floor and the HDRI dome would frame the whole shot.
UNBOXED_ROLES = ("floor", "backdrop")
#: The file a panel needs, so a clip without it gets a captioned blank.
PANEL_FILES = {"energy": "energy_map", "depth": "depth",
               "flow": "forward_flow", "backward_flow": "backward_flow",
               "normal": "normal", "object_coordinates": "object_coordinates"}
#: The 12 edges of a box whose 8 corners are `itertools.product(x, y, z)`:
#: corners joined by an edge differ in exactly one of the three bits.
BOX_EDGES = tuple((a, a | bit) for bit in (1, 2, 4) for a in range(8) if not a & bit)
VELOCITY_SECONDS = 1.0 / 3.0
#: Panels that say nothing about a valid clip, which has no violation.
VIOLATION_PANELS = ("mask", "severity", "causal", "divergence")


def build(sample_dir: str, out_path: Optional[str] = None, panel: int = PANEL,
          layers: Sequence[str] = DEFAULT_LAYERS,
          panels: Sequence[str] = DEFAULT_PANELS, **_ignored) -> Dict[str, object]:
    """Render a sample to mp4 (default: `<sample_dir>/overlay.mp4`)."""
    sample = loader.Sample.from_dir(sample_dir)
    r = Renderer(sample, layers, panels, panel)
    out_path = out_path or os.path.join(sample_dir, layout.OVERLAY)
    vid.write(r.render(), out_path, fps=sample.video.fps)
    return {"path": out_path, "frames": sample.video.num_frames,
            "panels": list(r.panels), "layers": list(r.layers)}


class Renderer:
    """Composes one frame at a time from a schema-v4 `loader.Sample`."""

    def __init__(self, clip: "loader.Sample", layers: Sequence[str] = DEFAULT_LAYERS,
                 panels: Sequence[str] = DEFAULT_PANELS, panel: int = PANEL,
                 columns: int = COLUMNS):
        unknown = ([x for x in layers if x not in LAYERS]
                   + [x for x in panels if x not in PANELS])
        if unknown:
            raise KeyError("unknown %s; layers: %s; panels: %s"
                           % (unknown, ", ".join(LAYERS), ", ".join(PANELS)))
        self.clip = clip
        self.layers = tuple(layers)
        self.panels = tuple(panels) or ("rgb",)
        self.size = int(panel)
        self.columns = max(1, min(int(columns), len(self.panels)))
        self.rows = math.ceil(len(self.panels) / self.columns)
        self.v = clip.violation
        self.T = clip.video.num_frames
        self.ids = [int(i) for i in self.v.ids]
        objects = clip.objects
        self.object_rows = {int(iid): row for row, iid in enumerate(objects.ids)}
        self.granular = (clip.scene.physics_medium == "granular" and len(self.ids) > 1)
        self.names = dict(zip((int(i) for i in objects.ids), objects.names))
        self.roles = dict(zip((int(i) for i in objects.ids), objects.roles))
        self.width = self.columns * self.size + (self.columns + 1) * PAD
        self.panels_bottom = HEADER + PAD + self.rows * (self.size + PAD)
        timeline_rows = 1 if self.granular and self.ids else len(self.ids)
        clocks = (0 if clip.info.is_valid else
                  30 + 3 * CLOCK_ROW + timeline_rows * VIOLATOR_ROW + SEVERITY_LANE)
        self.height = self.panels_bottom + LEGEND + clocks + 40
        self._cache: Dict[str, object] = {}

    # ---- whole frames ----------------------------------------------------
    def render(self) -> np.ndarray:
        return np.stack([self.frame(t) for t in range(self.T)])

    def frame(self, t: int) -> np.ndarray:
        import cv2
        S = self.size
        f = np.full((self.height, self.width, 3), C_BG, np.uint8)
        for n, name in enumerate(self.panels):
            x = PAD + (n % self.columns) * (S + PAD)
            y = HEADER + PAD + (n // self.columns) * (S + PAD)
            f[y:y + S, x:x + S] = self._panel(name, t)
            cv2.rectangle(f, (x, y), (x + S - 1, y + S - 1), (60, 60, 70), 1)
            title = PANELS[name]
            if name == "rgb" and self.layers:
                title = "RGB + " + ", ".join(self.layers)
            _label(f, _fit(title, S - 10, 0.44), (x + 5, y + 16))
            self._notes(f, name, t, x, y)
        tl = self.v.timeline
        _header(f, self.width, self.clip, t, self.T, bool(tl["active"][t]),
                bool(tl["observable"][t]), bool(tl["occluded"][t]))
        self._legend(f, self.panels_bottom)
        self._timeline(f, t, self.panels_bottom + LEGEND)
        return f

    def panel(self, name: str, t: int) -> np.ndarray:
        """One panel on its own, S x S, with its scale bars and read-outs but no
        title, header or timeline -- for a viewer that lays those out itself.
        Layers are drawn when `name` is "rgb"."""
        if name not in PANELS:
            raise KeyError("unknown panel %r; panels: %s" % (name, ", ".join(PANELS)))
        img = np.array(self._panel(name, t), np.uint8, copy=True)
        self._notes(img, name, t, 0, 0)
        return img

    def colour(self, instance_id: int):
        if instance_id in self.ids:
            if self.granular:
                return C_VIOLATORS[0]
            return C_VIOLATORS[self.ids.index(instance_id) % len(C_VIOLATORS)]
        return SEG_PALETTE[int(instance_id) % len(SEG_PALETTE)]

    # ---- panels ----------------------------------------------------------
    def _blank(self, text: str) -> np.ndarray:
        img = np.full((self.size, self.size, 3), C_BLANK, np.uint8)
        _text(img, _fit(text, self.size - 16, 0.42), (8, self.size // 2), C_DIM, 0.42, 1)
        return img

    def _panel(self, name: str, t: int) -> np.ndarray:
        c, v, S = self.clip, self.v, self.size
        obs = c.observations
        if name in PANEL_FILES and not _has(c, PANEL_FILES[name]):
            return self._blank("no %s in this clip" % name)
        if name in VIOLATION_PANELS and c.info.is_valid:
            return self._blank("valid clip: no %s" % name)
        if name in ("valid", "divergence") and not c.info.is_valid and c.twin is None:
            return self._blank("valid twin not found")
        if name == "rgb":
            img = _resize(c.video.rgb[t], S)
            for layer in self.layers:
                getattr(self, "_layer_" + layer)(img, t)
            return img
        if name == "valid":
            return _resize((c if c.info.is_valid else c.twin).video.rgb[t], S)
        if name == "camera":
            return self._camera(t)
        kind, arrays = {
            "energy": ("energy", lambda: {"energy": c.energy.map}),
            "segmentation": ("seg", lambda: {"seg": obs.segmentation}),
            "depth": ("depth", lambda: {"depth": obs.depth}),
            "flow": ("flow", lambda: {"flow": obs.forward_flow}),
            "backward_flow": ("flow", lambda: {"flow": obs.backward_flow}),
            "normal": ("normals", lambda: {"normals": obs.normal}),
            "object_coordinates": ("normals",
                                   lambda: {"normals": obs.object_coordinates}),
            "mask": ("mask", lambda: {
                "mask": v.object_id, "ref": v.reference_mask,
                "mask_colours": {iid: self.colour(iid) for iid in self.ids}}),
            "severity": ("sev", lambda: {"sev": v.severity_map}),
            "causal": ("causal", lambda: {"causal": v.causal}),
            "divergence": ("div", lambda: {"diverg": c.divergence}),
        }[name]
        args = dict(mask=None, sev=None, causal=None, diverg=None, ref=None,
                    mask_colours=None,
                    energy=None, seg=None, depth=None, flow=None, normals=None)
        args.update(arrays())
        return _panel(kind, t, c.video.rgb, size=S, **args)

    def _notes(self, f, name: str, t: int, x: int, y: int) -> None:
        """The per-panel scale bars and read-outs, drawn on the composed frame."""
        c, v, S = self.clip, self.v, self.size
        if name in PANEL_FILES and not _has(c, PANEL_FILES[name]):
            return
        if name in VIOLATION_PANELS and c.info.is_valid:
            return
        if name == "mask":
            _text(f, "%d px" % int(v.mask[t].sum()), (x + 5, y + S - 8),
                  C_DIM, 0.40, 1)
        elif name == "severity":
            _sev_scale(f, x, y, S, float(v.timeline["severity"][t]))
        elif name == "causal":
            _causal_key(f, x, y, S, v.causal[t])
        elif name == "segmentation":
            ids = [int(u) for u in np.unique(c.observations.segmentation[t]) if u]
            _text(f, _fit("ids %s" % ",".join(map(str, ids)), S - 10, 0.36),
                  (x + 5, y + S - 8), C_DIM, 0.36, 1)
        elif name in ("flow", "backward_flow"):
            last = t >= self.T - 1 if name == "flow" else t == 0
            if last:
                _text(f, "undefined on this frame", (x + 5, y + S - 8), C_DIM, 0.36, 1)
        elif name == "depth":
            d = np.asarray(c.observations.depth[t], np.float32)
            d = d[..., 0] if d.ndim == 3 else d
            here = d < DEPTH_SENTINEL
            if here.any():
                _text(f, "%.1f-%.1f m" % (d[here].min(), d[here].max()),
                      (x + 5, y + S - 8), C_DIM, 0.36, 1)
        elif name == "energy":
            _energy_scale(f, x, y, S, float(np.abs(c.energy.map[t]).max()))
            if _has(c, "energy"):
                twin = c.twin if not c.info.is_valid else None
                _energy_curve(f, x, y, S, t, dict(c.energy.scene),
                              dict(twin.energy.scene) if twin is not None
                              and _has(twin, "energy") else None)

    def _camera(self, t: int) -> np.ndarray:
        """Top-down (x, y) map: the camera's path and view wedge, object tracks."""
        import cv2
        S, cam, inst = self.size, self.clip.scene.camera, self.clip.objects
        img = np.full((S, S, 3), C_BLANK, np.uint8)
        eye = np.asarray(cam["positions"], np.float64)[:, :2]
        keep = [i for i, iid in enumerate(inst["ids"])
                if self.roles.get(int(iid)) not in UNBOXED_ROLES]
        tracks = np.asarray(inst["positions"], np.float64)[keep][:, :, :2]
        pts = np.concatenate([eye, tracks.reshape(-1, 2)])
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        span = max(float((hi - lo).max()), 1e-6) * 1.15
        mid = (lo + hi) / 2.0

        def px(p):
            return (int(S / 2 + (p[0] - mid[0]) / span * (S - 24)),
                    int(S / 2 + 10 - (p[1] - mid[1]) / span * (S - 24)))

        step = _nice_step(span / 4.0)
        for g in np.arange(math.floor((mid[0] - span) / step) * step, mid[0] + span, step):
            cv2.line(img, px((g, mid[1] - span)), px((g, mid[1] + span)), (36, 36, 44), 1)
        for g in np.arange(math.floor((mid[1] - span) / step) * step, mid[1] + span, step):
            cv2.line(img, px((mid[0] - span, g)), px((mid[0] + span, g)), (36, 36, 44), 1)
        _text(img, "grid %.2g m" % step, (6, S - 8), C_DIM, 0.34, 1)

        for j, i in enumerate(keep):
            iid = int(inst["ids"][i])
            col = self.colour(iid)
            track = [px(p) for p in tracks[j]]
            for a, b in zip(track, track[1:]):
                cv2.line(img, a, b, tuple(int(v * 0.45) for v in col), 1, cv2.LINE_AA)
            cv2.circle(img, track[t], 4, col, -1, cv2.LINE_AA)
            if iid in self.ids:
                cv2.circle(img, track[t], 7, col, 1, cv2.LINE_AA)

        path = [px(p) for p in eye]
        for a, b in zip(path, path[1:]):
            cv2.line(img, a, b, (120, 120, 135), 1, cv2.LINE_AA)
        R = movi.rotation_matrix(cam["quaternions"][t])
        forward = (R @ np.array([0.0, 0.0, -1.0]))[:2]
        if np.linalg.norm(forward) > 1e-6:
            half = math.radians(float(cam.get("field_of_view", 40.0)) / 2.0)
            base = math.atan2(forward[1], forward[0])
            reach = span * 0.35
            for side in (-half, half):
                tip = eye[t] + reach * np.array([math.cos(base + side), math.sin(base + side)])
                cv2.line(img, path[t], px(tip), (220, 220, 230), 1, cv2.LINE_AA)
        cv2.circle(img, path[t], 5, (255, 255, 255), -1, cv2.LINE_AA)
        _text(img, "camera", (path[t][0] + 8, path[t][1] + 4), C_DIM, 0.34, 1)
        return img

    # ---- layers (drawn at panel resolution, on the RGB panel) -------------
    def _up(self, m: np.ndarray) -> np.ndarray:
        import cv2
        return cv2.resize(np.asarray(m, np.uint8), (self.size, self.size),
                          interpolation=cv2.INTER_NEAREST).astype(bool)

    def _layer_violation(self, img, t):
        violation = self.v.object_id[t]
        for iid in self.ids:
            m = self._up(violation == iid)
            if m.any():
                col = np.array(self.colour(iid), np.float32)
                img[m] = (img[m] * 0.5 + col * 0.5).astype(np.uint8)
                img[m ^ _erode(m)] = col.astype(np.uint8)

    def _layer_visible(self, img, t):
        m = self._up(self.v.visible[t])
        img[m ^ _erode(m)] = (255, 255, 255)

    def _layer_severity(self, img, t):
        import cv2
        s = cv2.resize(self.v.severity_map[t], (self.size, self.size),
                       interpolation=cv2.INTER_NEAREST)
        on = s > 0
        if on.any():
            heat = cv2.applyColorMap((np.clip(s, 0, 1) ** SEV_GAMMA * 255).astype(np.uint8),
                                     cv2.COLORMAP_INFERNO)[..., ::-1]
            img[on] = (img[on] * 0.25 + heat[on] * 0.75).astype(np.uint8)

    def _layer_causal(self, img, t):
        causal = self.v.causal[t]
        affected = self._up(causal == 2)
        img[affected] = (img[affected] * 0.5 + np.array(C_CAUSAL2) * 0.5).astype(np.uint8)
        img[affected ^ _erode(affected)] = C_CAUSAL2
        violators = self._up(causal == 1)
        img[violators ^ _erode(violators)] = C_CAUSAL1

    def _layer_reference(self, img, t):
        m = self._up(self.v.reference_mask[t])
        img[m ^ _erode(m)] = C_REF

    def _boxed(self, t: int):
        """(row, id) of every body worth boxing on frame `t`.

        Skips the floor and the dome, and any body that does not exist on this
        frame: a vanished body keeps a pose in the trajectory, and a box drawn
        around nothing reads as a detection."""
        if "present" not in self._cache:
            table = {}
            inst = self.clip.objects
            if "present" in inst.keys():
                for j, bid in enumerate(inst["ids"]):
                    table[int(bid)] = np.asarray(inst["present"][j], bool)
            self._cache["present"] = table
        present = self._cache["present"]
        inst = self.clip.objects
        return [(i, int(iid)) for i, iid in enumerate(inst["ids"])
                if self.roles.get(int(iid)) not in UNBOXED_ROLES
                and (int(iid) not in present or present[int(iid)][t])]

    def _layer_bbox2d(self, img, t):
        import cv2
        boxes, S = self.clip.objects["bboxes"], self.size
        for i, iid in self._boxed(t):
            b = boxes[i, t]
            if np.isnan(b).any():
                continue
            y0, x0, y1, x1 = (int(round(v * S)) for v in b)
            cv2.rectangle(img, (x0, y0), (x1 - 1, y1 - 1), self.colour(iid), 1)

    def _projected(self, key: str, points: np.ndarray) -> np.ndarray:
        """Project instance-major world points [k,T,...,3] -> [k,T,...,3]
        of (x, y) in [0, 1] and the depth sign, once per clip."""
        if key not in self._cache:
            cam = self.clip.scene.camera
            p = movi.project(np.swapaxes(points, 0, 1), cam["positions"],
                             cam["quaternions"], cam["K"])
            self._cache[key] = np.swapaxes(p, 0, 1)
        return self._cache[key]

    def _layer_bbox3d(self, img, t):
        import cv2
        S = self.size
        proj = self._projected("bbox3d", self.clip.objects["bboxes_3d"])
        for i, iid in self._boxed(t):
            p = proj[i, t]
            if (p[:, 2] <= 0).any() or not np.isfinite(p[:, :2]).all():
                continue
            xy = np.round(p[:, :2] * S).astype(int)
            if np.abs(xy).max() > 8 * S:
                continue
            for a, b in BOX_EDGES:
                cv2.line(img, tuple(xy[a]), tuple(xy[b]), self.colour(iid), 1, cv2.LINE_AA)

    def _layer_centers(self, img, t):
        import cv2
        inst, S = self.clip.objects, self.size
        for i, iid in self._boxed(t):
            x, y = inst["image_positions"][i, t]
            if not (0 <= x <= 1 and 0 <= y <= 1):
                continue
            seen = inst["visibility"][i, t] > 0
            cv2.circle(img, (int(x * S), int(y * S)), 3, self.colour(iid),
                       -1 if seen else 1, cv2.LINE_AA)

    def _layer_velocity(self, img, t):
        import cv2
        inst, S = self.clip.objects, self.size
        ahead = inst["positions"] + inst["velocities"] * VELOCITY_SECONDS
        tips = self._projected("velocity", ahead[:, :, None, :])[:, :, 0]
        for i, iid in self._boxed(t):
            if float(np.linalg.norm(inst["velocities"][i, t])) < 0.05 or tips[i, t, 2] <= 0:
                continue
            x0, y0 = inst["image_positions"][i, t]
            x1, y1 = tips[i, t, :2]
            if not all(np.isfinite([x0, y0, x1, y1])):
                continue
            cv2.arrowedLine(img, (int(x0 * S), int(y0 * S)), (int(x1 * S), int(y1 * S)),
                            self.colour(iid), 1, cv2.LINE_AA, tipLength=0.25)

    def _layer_labels(self, img, t):
        inst, S = self.clip.objects, self.size
        for i, iid in self._boxed(t):
            b = inst["bboxes"][i, t]
            if np.isnan(b).any():
                continue
            mark = "! " if iid in self.ids else ""
            _text(img, "%s%s #%d" % (mark, self.names.get(iid, iid), iid),
                  (int(b[1] * S) + 2, max(int(b[0] * S) - 3, 10)),
                  self.colour(iid), 0.34, 1)

    def _layer_events(self, img, t):
        import cv2
        S = self.size
        events = self.clip.events.collisions
        if "frame" not in events:
            return
        for k in np.flatnonzero(events["frame"] == t):
            x, y = events["image_position"][k]
            if not (np.isfinite(x) and np.isfinite(y)):
                continue
            if 0 <= x <= 1 and 0 <= y <= 1:
                r = int(3 + 2 * math.log10(1 + 10 * max(float(events["force"][k]), 0)))
                cv2.circle(img, (int(x * S), int(y * S)), r, C_EVENT, 1, cv2.LINE_AA)

    # ---- legend and timeline ---------------------------------------------
    def _legend(self, f, y: int) -> None:
        import cv2
        x, yb = PAD + 2, y + 15
        if self.clip.info.is_valid:
            _text(f, "valid clip -- nothing here is violated", (x, yb), C_DIM, 0.40, 1)
            x += _w("valid clip -- nothing here is violated", 0.40) + 18
        if self.granular and self.ids:
            cv2.rectangle(f, (x, yb - 9), (x + 10, yb), C_VIOLATORS[0], -1)
            s = "violating medium (%d grains)" % len(self.ids)
            _text(f, s, (x + 14, yb), C_TEXT, 0.40, 1)
            x += 14 + _w(s, 0.40) + 16
        compact = not self.granular and len(self.ids) > 4
        for iid in (() if self.granular else self.ids):
            cv2.rectangle(f, (x, yb - 9), (x + 10, yb), self.colour(iid), -1)
            # Names remain on the matching timeline rows. With many violators,
            # repeating them here made the legend run beyond the canvas and
            # hid the colour key that the legend exists to provide.
            s = "#%d" % iid if compact else "violator %s #%d" % (self.names.get(iid, iid), iid)
            _text(f, s, (x + 14, yb), C_TEXT, 0.40, 1)
            x += 14 + _w(s, 0.40) + 16
        if self.layers:
            s = "layers: " + ", ".join(self.layers)
            _text(f, _fit(s, max(self.width - x - PAD, 40), 0.36), (x, yb), C_DIM, 0.36, 1)

    def _timeline(self, f, t: int, y0: int) -> None:
        import cv2
        T, v = self.T, self.v
        x0, x1 = PAD + 2, self.width - PAD - 2
        span = x1 - x0

        def fx(frame):
            return int(x0 + (frame / max(T, 1)) * span)

        def display_windows(windows):
            wins = [tuple(w) for w in windows]
            if self.granular and wins:
                return [(min(s for s, _ in wins), max(e for _, e in wins))]
            return wins

        y = y0
        if not self.clip.info.is_valid:
            windows = v.windows
            vwin = display_windows(windows["active"])
            iwin = display_windows(windows["intervening"]) or vwin
            cwin = display_windows(windows["consequence"]) or vwin
            owin = display_windows(windows["observable"])
            rows = [("intervening", "being changed", iwin, C_INTERVENE),
                    ("consequence", "still wrong", cwin, C_MASK),
                    ("observable", "a viewer could tell", owin, C_OBS)]
            lx = x0
            for label, gloss, _wins, colour in rows:
                cv2.rectangle(f, (lx, y + 3), (lx + 10, y + 12), colour, -1)
                lx += 14
                _text(f, label, (lx, y + 11), C_TEXT, 0.40, 1)
                lx += _w(label, 0.40) + 4
                _text(f, "(%s)" % gloss, (lx, y + 11), C_DIM, 0.34, 1)
                lx += _w("(%s)" % gloss, 0.34) + 16
            # The legend line, then a row of its own for the clock markers, so a
            # label like `t_event=t_applied=t_obs=t_end` never lands on the legend.
            y += 30
            for _label_, _gloss, wins, colour in rows:
                cv2.rectangle(f, (x0, y), (x1, y + 6), (40, 40, 48), -1)
                for s, e in wins:
                    cv2.rectangle(f, (fx(s), y), (max(fx(e + 1) - 1, fx(s) + 2), y + 6),
                                  colour, -1)
                y += CLOCK_ROW

            # One row per violator. A granular medium is intentionally one
            # aggregate row: dozens of grain rows and fragmented windows hide
            # the event instead of explaining it.
            obj = {key: v[key] for key in ("active", "observable", "occluded", "severity")}
            if self.granular and self.ids:
                rows = [self.object_rows[iid] for iid in self.ids]
                series = [("medium", C_VIOLATORS[0],
                           {"active": np.any(obj["active"][rows], axis=0),
                            "observable": np.any(obj["observable"][rows], axis=0),
                            "occluded": np.all(obj["occluded"][rows], axis=0),
                            "severity": np.max(obj["severity"][rows], axis=0)})]
            else:
                series = [("#%d %s" % (iid, self.names.get(iid, "")),
                           self.colour(iid),
                           {key: obj[key][self.object_rows[iid]]
                            for key in ("active", "observable", "occluded", "severity")})
                          for iid in self.ids]
            for name, col, values in series:
                cv2.rectangle(f, (x0, y), (x1, y + 8), (34, 34, 40), -1)
                for key, top, colour, h in (("active", y, col, 9),
                                            ("observable", y + 7, C_OBS, 2),
                                            ("occluded", y, (130, 130, 140), 2)):
                    wins = _runs(values[key])
                    if self.granular and wins:
                        wins = [(wins[0][0], wins[-1][1])]
                    for s, e in wins:
                        cv2.rectangle(f, (fx(s), top),
                                      (max(fx(e + 1) - 1, fx(s) + 2), top + h - 1),
                                      colour, -1)
                _text(f, name, (x1 - _w(name, 0.32) - 2, y + 8), col, 0.32, 1)
                y += VIOLATOR_ROW

            # Severity per violator on one 0..1 lane: the value `severity_map`
            # paints over each body on that frame.
            top, h = y + 2, SEVERITY_LANE - 6
            cv2.rectangle(f, (x0, top), (x1, top + h), (26, 26, 32), -1)
            for _name, col, values in series:
                sev = np.clip(values["severity"], 0, 1)
                pts = [(fx(i + 0.5), int(top + h - sev[i] * (h - 2) - 1)) for i in range(T)]
                for a, b in zip(pts, pts[1:]):
                    cv2.line(f, a, b, col, 1, cv2.LINE_AA)
            _text(f, "severity [0, 1]", (x0 + 3, top + 11), C_DIM, 0.32, 1)
            y += SEVERITY_LANE

        # frame ticks, labelled every 4 (every 8 on a long clip)
        every = 4 if T <= 40 else 8
        for i in range(T):
            cv2.line(f, (fx(i), y), (fx(i), y + 4), (90, 90, 100), 1)
            if i % every == 0:
                _text(f, str(i), (fx(i) - 3, y + 15), C_DIM, 0.34, 1)

        t_event = t_iend = t_obs = t_end = -1
        if not self.clip.info.is_valid:
            as_frame = lambda value: -1 if value is None else int(value)
            t_event, t_obs, t_end = (as_frame(v.t_event), as_frame(v.t_observable),
                                     as_frame(v.t_end))
            t_iend = as_frame(v.t_intervention_end if v.t_intervention_end is not None
                              else v.t_end)
            marks = [(t_event, "t_event", C_INTERVENE), (t_iend, "t_applied", C_INTERVENE),
                     (t_obs, "t_obs", C_OBS), (t_end, "t_end", (170, 170, 210))]
            by_frame: Dict[int, List] = {}
            for frame, tag, col in marks:
                if 0 <= frame < T:
                    by_frame.setdefault(frame, []).append((tag, col))
            for frame in sorted(by_frame):
                tags = by_frame[frame]
                label = "=".join(tag for tag, _ in tags)
                cv2.line(f, (fx(frame), y0 + 17), (fx(frame), y + 4), tags[0][1], 1)
                tx = fx(frame) + 3
                if tx + _w(label, 0.36) > x1:
                    tx = fx(frame) - _w(label, 0.36) - 3
                _text(f, label, (tx, y0 + 27), tags[0][1], 0.36, 1)

        px = fx(t) + max(1, span // (2 * max(T, 1)))
        cv2.line(f, (px, y0 + 17), (px, y + 5), (255, 255, 255), 1)

        if self.clip.info.is_valid:
            info = "valid twin of %s" % self.clip.pair_uid
        else:
            peak = v.peak_residual
            info = ("t_event=%d  applied_to=%d  t_obs=%d  t_end=%d  lag=%d  |  "
                    "severity(t)=%.3f  peak=%.3f  r=%.3f (%s)"
                    % (t_event, t_iend, t_obs, t_end, v.observability_lag or 0,
                       float(v.timeline["severity"][t]),
                       float(peak.get("score", 0.0)), float(peak.get("value", 0.0)),
                       peak.get("law", "-")))
        _text(f, _fit(info, span, 0.44), (x0, self.height - 8), C_TEXT, 0.44, 1)


# ------------------------------------------------------------------ panels
SEV_GAMMA = 0.55   # display-only; see the note in _panel


#: Stable per-instance colours for the segmentation panel. Index by
#: `id % len`, so a body keeps its colour across every clip of a scenario.
SEG_PALETTE = [
    (231, 76, 60), (52, 152, 219), (46, 204, 113), (241, 196, 15),
    (155, 89, 182), (26, 188, 156), (230, 126, 34), (149, 165, 166),
    (255, 121, 198), (139, 233, 253), (189, 147, 249), (80, 250, 123),
]
#: Depth beyond this is the renderer's "nothing here" sentinel (~1e10), not a
#: distance. Anything past it is background and must not enter the normalisation.
DEPTH_SENTINEL = loader.DEPTH_BACKGROUND


_FLOW_SCALE_CACHE: Dict[int, float] = {}

#: Below this, a flow vector is render-pass jitter on the static background,
#: not real motion -- observed noise floor is ~1e-5 px/frame, observed real
#: motion (including antialiased silhouette edges) starts ~0.05, so this sits
#: in the gap with margin either side. Below it we zero the vector outright,
#: both so it can't skew the scale in _flow_scale and so it can't paint the
#: background with speckled colour at display time.
FLOW_NOISE_FLOOR = 0.02


def _flow_scale(flow) -> float:
    """One magnitude scale for the whole clip, from a high percentile.

    The 99th rather than the max, so a couple of edge pixels at a silhouette --
    where flow is undefined and can be enormous -- do not black out the body
    they belong to. Computed over moving pixels only: the background is the
    overwhelming majority of pixel-frames, so pooling everything drags the
    99th percentile down into the noise floor instead of the real motion
    scale, which is what was blowing up background jitter into visible
    colour in the panel.
    """
    key = id(flow)
    if key not in _FLOW_SCALE_CACHE:
        mag = np.linalg.norm(np.asarray(flow, np.float32), axis=-1)
        moving = mag[mag > FLOW_NOISE_FLOOR]
        pool = moving if moving.size else mag
        _FLOW_SCALE_CACHE[key] = max(float(np.percentile(pool, 99.0)), 1e-6)
    return _FLOW_SCALE_CACHE[key]


def _panel(kind, t, rgb, mask, sev, causal, diverg, size, ref=None,
           mask_colours=None, energy=None, seg=None, depth=None, flow=None,
           normals=None):
    import cv2
    base = rgb[t].astype(np.float32)

    if kind == "rgb":
        img = base
    elif kind == "mask":
        img = base.copy()
        # Paint the lawful/reference location first. It remains visible for a
        # vanished object, while current violation locations are then drawn in
        # their stable per-object colours. A bright green outline is repeated
        # last so overlap never erases the "where it should be" cue.
        if ref is not None:
            r = ref[t].astype(bool)
            img[r] = img[r] * 0.60 + np.array(C_REF, np.float32) * 0.40
        if mask is not None:
            labels = np.asarray(mask[t])
            if labels.dtype == bool or not mask_colours:
                labels = labels.astype(bool)
                colours = {1: C_MASK}
                labels = labels.astype(np.uint8)
            else:
                colours = mask_colours
            for iid, colour in colours.items():
                m = labels == int(iid)
                if not m.any():
                    continue
                col = np.array(colour, np.float32)
                img[m] = img[m] * 0.50 + col * 0.50
                img[m ^ _erode(m)] = col
        if ref is not None:
            r = ref[t].astype(bool)
            img[r ^ _erode(r)] = np.array(C_REF, np.float32)
    elif kind == "sev":
        s = np.clip(sev[t], 0.0, 1.0)
        # Gamma for display only. A severity of 0.13 lands in the near-black
        # end of inferno, so an honest linear ramp renders a real violation as
        # an invisible smudge -- and "is the severity field sensible" is the
        # question this panel exists to answer. The transform is monotone, so
        # brighter still means worse and columns of a grid stay comparable, and
        # the exact value is printed beside the scale bar either way.
        heat = cv2.applyColorMap((s ** SEV_GAMMA * 255).astype(np.uint8),
                                 cv2.COLORMAP_INFERNO)[..., ::-1].astype(np.float32)
        a = (s > 0)[..., None].astype(np.float32) * 0.88
        img = base * 0.35 * (1 - a) + base * (1 - a) * 0.65 + heat * a
    elif kind == "seg":
        # Instance ids in a fixed palette, so the same body is the same colour
        # in every panel and every clip of a scenario. Background stays dark.
        lab = seg[t]
        img = np.zeros(base.shape, np.float32)
        for uid in np.unique(lab):
            if uid == 0:
                continue
            img[lab == uid] = np.array(
                SEG_PALETTE[int(uid) % len(SEG_PALETTE)], np.float32)
        img = img * 0.85 + base * 0.15
    elif kind == "depth":
        d = np.asarray(depth[t], np.float32)
        if d.ndim == 3:
            d = d[..., 0]
        here = d < DEPTH_SENTINEL
        img = np.zeros(base.shape, np.float32)
        if here.any():
            lo, hi = float(d[here].min()), float(d[here].max())
            norm = np.clip((d - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
            heat = cv2.applyColorMap((norm * 255).astype(np.uint8),
                                     cv2.COLORMAP_TURBO)[..., ::-1]
            img[here] = heat[here].astype(np.float32)
    elif kind == "flow":
        # The standard flow wheel: hue is direction, value is magnitude. Our
        # flow is (row, col), so the angle is atan2(col, row) to read as screen
        # direction rather than transposed.
        #
        # Scaled over the WHOLE clip, not per frame. Per-frame normalisation
        # makes the colour scale jump between frames, so a body at constant
        # speed changes brightness for no reason -- and it degenerates entirely
        # on the terminal frame, where forward flow is all zeros because there
        # is no frame T to flow to.
        fl = np.asarray(flow[t], np.float32)
        mag = np.linalg.norm(fl, axis=-1)
        scale = _flow_scale(flow)
        still = mag <= FLOW_NOISE_FLOOR
        ang = np.arctan2(fl[..., 1], fl[..., 0])
        hsv = np.zeros(base.shape, np.uint8)
        hsv[..., 0] = ((ang + np.pi) / (2 * np.pi) * 179).astype(np.uint8)
        hsv[..., 1] = 255
        hsv[..., 2] = np.clip(mag / scale * 255, 0, 255).astype(np.uint8)
        hsv[still, 2] = 0
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB).astype(np.float32)
        img = img * 0.9 + base * 0.1
    elif kind == "normals":
        img = (np.asarray(normals[t], np.float32) / 65535.0 * 255.0)
    elif kind == "energy":
        # VIRIDIS, deliberately not severity's INFERNO: the two panels sit side
        # by side and answer different questions, so they must not be mistakable
        # for one another at a glance.
        e = np.abs(np.clip(energy[t], -ENERGY_CLIP, ENERGY_CLIP)) / ENERGY_CLIP
        heat = cv2.applyColorMap((e ** ENERGY_GAMMA * 255).astype(np.uint8),
                                 cv2.COLORMAP_VIRIDIS)[..., ::-1].astype(np.float32)
        a = (e > 1e-6)[..., None].astype(np.float32) * 0.9
        img = base * (1 - a) * 0.5 + heat * a
    elif kind == "causal":
        img = base * 0.45
        c = causal[t]
        img[c == 1] = np.array(C_CAUSAL1, np.float32)
        img[c == 2] = np.array(C_CAUSAL2, np.float32)
    else:  # divergence
        d = np.clip(diverg[t], 0.0, 1.0)
        img = (cv2.applyColorMap((d * 255).astype(np.uint8),
                                 cv2.COLORMAP_BONE)[..., ::-1]).astype(np.float32)

    img = np.clip(img, 0, 255).astype(np.uint8)
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_NEAREST)


# ------------------------------------------------------------------ chrome
def _w(s, scale, thick=1):
    import cv2
    return cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)[0][0]


def _fit(s: str, width: int, scale: float) -> str:
    """`s`, cut with an ellipsis until it fits `width` pixels."""
    if _w(s, scale) <= width:
        return s
    while s and _w(s + "...", scale) > width:
        s = s[:-1]
    return s + "..."


def _header(f, W, clip, t, T, active, observable, occluded):
    """Laid out right-to-left from measured widths so nothing ever collides."""
    import cv2
    info, scene, v = clip.info, clip.scene, clip.violation

    # --- right edge: frame counter, then the state pill ---
    frame_s = "f %d/%d" % (t, T - 1)
    x = W - PAD - _w(frame_s, 0.46)
    _text(f, frame_s, (x, 22), C_TEXT, 0.46, 1)

    state = "VIOLATION ACTIVE" if active else (
        "observable only" if observable else ("occluded" if occluded else "clean"))
    col = C_ACTIVE if active else C_DIM
    x -= 14 + _w(state, 0.50 if active else 0.46)
    _text(f, state, (x, 22), col, 0.50 if active else 0.46, 1)
    x -= 16
    cv2.circle(f, (x, 16), 7, C_ACTIVE if active else (70, 70, 78),
               -1 if active else 1)
    if active:
        cv2.circle(f, (x, 16), 7, (255, 255, 255), 1)
    right_limit = x - 14

    # --- left edge: identity, then metadata if it still fits ---
    left = ("%s / %s / %s" % (scene.domain, scene.family, scene.scenario)
            if not info.is_valid else "valid / %s" % scene.scenario)
    _text(f, left, (PAD + 2, 22), C_TEXT, 0.52, 1)
    lx = PAD + 2 + _w(left, 0.52) + 22
    sev_bin = v.severity_bin or "-"
    # THE CONDITION AND THE LEVEL, on the frame. A clip's difficulty is not
    # readable from the picture -- a static camera looks like a moving one that
    # has not moved yet, and five lawful peers look like five distractors --
    # so watching a run meant remembering which seed was which. It is written
    # here, in the video, and it degrades gracefully: the pieces are dropped
    # right to left as the width runs out, identity first.
    cond = scene.condition
    cam = scene.camera.motion
    if cam and cam != "static":
        cond = "%s:%s" % (cond or "?", cam)
    bits = [str(scene.level or ""), sev_bin]
    if cond:
        bits.append(cond)
    # HOW MANY BODIES ARE WRONG, when it is more than one. `multi` is the
    # condition that asks "which of these is wrong", but it is not the only way
    # a clip ends up with several violators: `superelastic` boosts BOTH bodies of
    # a two-body collision, because boosting one would add a momentum
    # violation the clip does not annotate; `fission` and `fusion` name both
    # halves. So a `standard` clip can carry two violators, and nothing on the
    # frame said so -- you found it on L3 `stack_topple x superelastic`, where
    # the severity landed on two blocks under a label that did not mention it.
    nc = len(v.violators)
    if nc > 1:
        timing = v.timing
        bits.append("%d violators%s" % (nc, " (%s)" % timing
                                        if timing in ("independent", "sync")
                                        else ""))
    bits += ["seed %s" % info.seed, "tier %s" % info.tier]
    for k in range(len(bits), 0, -1):
        mid = "   ".join(x for x in bits[:k] if x)
        if lx + _w(mid, 0.46) < right_limit:
            _text(f, mid, (lx, 22), C_DIM, 0.46, 1)
            break


#: The intervention bar. Deliberately a different hue from the consequence bar:
#: they overlap at the start and the eye needs to separate "we are changing
#: something" from "the scene is wrong as a result".
C_INTERVENE = (120, 200, 255)


def _sev_scale(f, x, y, size, value):
    """A 0..1 severity axis in INFERNO with the live value marked.

    The axis stays linear in severity; only the colour lookup is gamma'd, so
    the tick position reads off directly and a dark purple blob is still
    distinguishable from an empty map."""
    import cv2
    bw, bh = size - 20, 9
    bx, by = x + 10, y + size - 26
    # The bar is a legend for the panel, so it has to carry the panel's gamma.
    # A linear ramp under a gamma'd map would put a colour on the scale that
    # never appears in the image beside it.
    axis = np.linspace(0.0, 1.0, bw) ** SEV_GAMMA
    ramp = (axis * 255).astype(np.uint8)[None, :].repeat(bh, 0)
    f[by:by + bh, bx:bx + bw] = cv2.applyColorMap(ramp,
                                                  cv2.COLORMAP_INFERNO)[..., ::-1]
    mx = bx + int(np.clip(value, 0, 1) * (bw - 1))
    cv2.line(f, (mx, by - 3), (mx, by + bh + 3), (255, 255, 255), 1)
    _text(f, "s=%.3f" % value, (bx, by - 6), (255, 255, 255), 0.42, 1)
    _text(f, "0", (bx - 7, by + bh + 1), C_DIM, 0.34, 1)
    _text(f, "1", (bx + bw + 2, by + bh + 1), C_DIM, 0.34, 1)


def _label(f, s, org):
    _text(f, s, org, (255, 255, 255), 0.44, 1)


def _causal_key(f, x, y, panel, layer):
    """Name the two levels, in their own colours, in the panel that uses them.

    The array is uint8 with three values and no key anywhere on the frame, so
    "what is red and what is blue" was a question the overlay made a reader ask
    and then did not answer. Red is the violator -- the body the plan names --
    and blue is a body it affected, which is measured rather than declared.
    Counts beside each, because a level that is present in the legend and absent
    from the image is worth being able to tell apart from one that is simply
    hard to see.
    """
    import cv2

    rows = [("violator", C_CAUSAL1, int((layer == 1).sum())),
            ("affected", C_CAUSAL2, int((layer == 2).sum()))]
    ly = y + panel - 8 - 11 * (len(rows) - 1)
    for label, colour, count in rows:
        cv2.rectangle(f, (x + 5, ly - 7), (x + 13, ly - 1), colour, -1)
        _text(f, "%s %d px" % (label, count), (x + 17, ly), C_DIM, 0.36, 1)
        ly += 11


def _text(img, s, org, color, scale, thick=1, backing=True):
    """Draw legible text over arbitrary imagery.

    Deliberately a dark backing box rather than a thick black outline: in
    OpenCV the Hershey glyph *advance width* grows with stroke thickness, so an
    outline pass drawn at thick+2 is wider than the fill pass and its tail
    glyphs poke out past the text as dark ghosts.
    """
    import cv2
    (tw, th), base = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    x, y = int(org[0]), int(org[1])
    if backing:
        x0, y0 = max(0, x - 2), max(0, y - th - 2)
        x1, y1 = min(img.shape[1], x + tw + 2), min(img.shape[0], y + base + 1)
        if x1 > x0 and y1 > y0:
            roi = img[y0:y1, x0:x1].astype(np.float32)
            img[y0:y1, x0:x1] = (roi * 0.25).astype(np.uint8)
    cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick,
                cv2.LINE_AA)


# ------------------------------------------------------------------ helpers
def _resize(img: np.ndarray, size: int) -> np.ndarray:
    import cv2
    return cv2.resize(np.ascontiguousarray(img[..., :3]), (size, size),
                      interpolation=cv2.INTER_NEAREST)


def _erode(m):
    e = m.copy()
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        e &= np.roll(np.roll(m, dy, axis=0), dx, axis=1)
    return e


def _runs(flags) -> List[tuple]:
    """Inclusive (start, end) of each run of True."""
    out, start = [], None
    for i, on in enumerate(np.asarray(flags, bool)):
        if on and start is None:
            start = i
        elif not on and start is not None:
            out.append((start, i - 1))
            start = None
    if start is not None:
        out.append((start, len(flags) - 1))
    return out


def _nice_step(x: float) -> float:
    """1, 2 or 5 times a power of ten, at least `x`."""
    p = 10 ** math.floor(math.log10(max(x, 1e-9)))
    return next(m * p for m in (1, 2, 5, 10) if m * p >= x)


def _has(clip, name: str) -> bool:
    """Whether a sample carries a panel's source: a render pass, the energy
    map (``energy_map``) or the scene energy trace (``energy``)."""
    if name == "energy_map":
        return clip.energy.map is not None
    if name == "energy":
        return bool(clip.energy.scene)
    return name in clip.observations


# ------------------------------------------------------------------ energy
def _energy_trace(sample_dir):
    if not sample_dir:
        return None
    try:
        sample = loader.Sample.from_dir(sample_dir)
    except (FileNotFoundError, ValueError):
        return None
    try:
        return dict(sample.energy.scene)
    finally:
        sample.release()


#: Energy map display range, as a multiple of E0, and the gamma the panel uses.
#: A body typically holds well under E0 on its own, so the useful range is the
#: low end -- hence the sqrt, which is monotone so brighter still means more.
ENERGY_CLIP = 4.0
ENERGY_GAMMA = 0.5


def _energy_scale(f, x, y, size, value=None):
    """VIRIDIS legend for the energy map, carrying the panel's own gamma.

    Without it the map is a field of colours with no key -- you can see that two
    bodies differ but not by how much, which is most of what the panel is for.
    A linear ramp under a gamma'd map would also put colours on the scale that
    never appear in the image beside it.
    """
    import cv2
    bw, bh = size - 20, 9
    bx, by = x + 10, y + size - 26 - max(34, size // 5)
    axis = np.linspace(0.0, 1.0, bw) ** ENERGY_GAMMA
    ramp = (axis * 255).astype(np.uint8)[None, :].repeat(bh, 0)
    f[by:by + bh, bx:bx + bw] = cv2.applyColorMap(
        ramp, cv2.COLORMAP_VIRIDIS)[..., ::-1]
    _text(f, "0", (bx, by - 4), C_DIM, 0.34, 1)
    hi = "%.0fx E0" % ENERGY_CLIP
    _text(f, hi, (bx + bw - _w(hi, 0.34), by - 4), C_DIM, 0.34, 1)
    if value is not None:
        mx = bx + int(np.clip(value / ENERGY_CLIP, 0, 1) ** ENERGY_GAMMA * (bw - 1))
        cv2.line(f, (mx, by - 3), (mx, by + bh + 3), (255, 255, 255), 1)


def _energy_curve(f, x, y, size, t, trace, twin=None):
    """E(t) for this clip and, where it exists, its twin -- with the playhead.

    The map answers "where is the energy"; this answers "what did it do", which
    is the question the annotation actually exists for. A violation that creates
    energy is a step in a line, and no amount of colouring pixels shows a step.
    """
    import cv2
    h = max(34, size // 5)
    bx, by, bw = x + 6, y + size - h - 6, size - 12
    cv2.rectangle(f, (bx, by), (bx + bw, by + h), (12, 12, 16), -1)
    cv2.rectangle(f, (bx, by), (bx + bw, by + h), (60, 60, 70), 1)

    # WHAT THE CAMERA CAN STILL ACCOUNT FOR. The curve is `energy_in_frame`,
    # not `total`, because the panel sits beside a video and has to describe
    # the same clip the video does. A super-elastic bounce that throws the ball
    # out of the top of the shot leaves `total` sitting at 12 J for the rest of
    # the clip while every pixel of evidence for it is gone -- you reported
    # exactly that on `drop x superelastic`, `solidity` and `antigravity`.
    # `total` is still drawn, dimmed, so the difference between the two IS the
    # reading: where they separate, energy has left the frame rather than the
    # scene, and `/energy/scene` ships both.
    key = "energy_in_frame" if "energy_in_frame" in trace else "total"
    series = [(trace[key], (255, 120, 120))]
    if trace.get("total") is not None and key != "total":
        series.insert(0, (trace["total"], (110, 60, 60)))
    if twin is not None and key in twin:
        series.insert(0, (twin[key], C_REF))
    lo = min(float(np.min(s)) for s, _ in series)
    hi = max(float(np.max(s)) for s, _ in series)
    span = max(hi - lo, 1e-9)
    T = len(trace[key])
    for s, col in series:
        pts = [(bx + int(i / max(T - 1, 1) * (bw - 1)),
                by + h - 1 - int((float(s[i]) - lo) / span * (h - 3)))
               for i in range(T)]
        for i in range(1, len(pts)):
            cv2.line(f, pts[i - 1], pts[i], col, 1, cv2.LINE_AA)
    px = bx + int(t / max(T - 1, 1) * (bw - 1))
    cv2.line(f, (px, by), (px, by + h), (255, 255, 255), 1)
    seen = float(trace[key][t])
    lab = "E %.2fJ" % seen
    _text(f, lab, (bx + 3, by + 11), (255, 255, 255), 0.36, 1)
    # Which line is which. Without it the panel shows two curves crossing and
    # leaves the reader to guess which one is the violation.
    lx = bx + 6 + _w(lab, 0.36)
    if twin is not None and key in twin:
        _text(f, "valid", (lx, by + 11), C_REF, 0.34, 1)
        lx += 6 + _w("valid", 0.34)
    _text(f, "this", (lx, by + 11), (255, 120, 120), 0.34, 1)
    # Named only when it says something. A body inside the frustum makes the
    # two curves identical, and a legend for a line nobody can see is noise.
    if key != "total":
        gone = float(trace["total"][t]) - seen
        if abs(gone) > 0.01 * max(abs(float(trace["total"][0])), 1e-9):
            _text(f, "off-frame %.2fJ" % gone,
                  (lx + 6 + _w("this", 0.34), by + 11), (110, 60, 60), 0.34, 1)
    # Name the channel instead of signing the number. All three are
    # non-negative by construction, so "%+.0f%%" printed a plus on every frame
    # and told the reader nothing -- and which channel fired is the part that
    # distinguishes energy appearing from energy vanishing.
    zero = np.zeros(T, np.float32)
    channels = (("gain", float(np.asarray(trace.get("contact_anomaly", zero))[t])),
                ("uncaused", float(np.asarray(trace.get("free_anomaly", zero))[t])),
                ("loss", float(np.asarray(trace.get("excess_loss", zero))[t])))
    name, peak = max(channels, key=lambda kv: kv[1])
    if peak > 0.01:
        _text(f, "%s %.0f%%" % (name, 100 * peak), (bx + 3, by + h - 4),
              (255, 120, 120), 0.36, 1)
