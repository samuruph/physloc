"""Annotation visualiser -- docs/PLAN.md Verification.

Produces ONE mp4 per invalid clip showing every annotation side by side, so a
human can tell at a glance whether the labels are right. Never trust a metric
built on an unviewed label.

Layout:

    +---------------------------------------------------------------+
    | domain / family / scenario   seed   severity      [O] ACTIVE   |
    +------------+------------+------------+------------+------------+
    | RGB        | + MASK     | SEVERITY   | CAUSAL     | DIVERGENCE |
    +------------+------------+------------+------------+------------+
    | timeline: violation window (red) / observable (amber) + playhead|
    | t_event  t_obs  t_end  lag   sev(t)   peak                     |
    +---------------------------------------------------------------+

No image files are written -- mp4 only.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import video as vid

PANEL = 288                      # each panel is rendered at this size
HEADER = 34
TIMELINE = 84
PAD = 6

C_BG = (18, 18, 22)
C_TEXT = (232, 232, 238)
C_DIM = (150, 150, 160)
C_MASK = (255, 70, 70)
C_ACTIVE = (255, 60, 60)
C_OBS = (250, 185, 60)
C_CAUSAL1 = (255, 70, 70)
C_CAUSAL2 = (90, 160, 255)
C_REF = (90, 235, 120)      # "where it should be", from the valid twin


def build(clip_dir: str, out_path: Optional[str] = None,
          panel: int = PANEL, **_ignored) -> Dict[str, object]:
    import cv2

    meta = _json(os.path.join(clip_dir, "meta.json"))
    v = meta.get("violation") or {}
    T = int(meta["num_frames"])

    rgb = _rgb(clip_dir, T)
    mask = _npz(clip_dir, "violation_mask.npz", "mask")
    sev = _npz(clip_dir, "severity_map.npz", "severity")
    causal = _npz(clip_dir, "causal_mask.npz", "mask")
    ref = _npz(clip_dir, "reference_mask.npz", "mask")
    diverg = _npz(clip_dir, "divergence_map.npz", "divergence")
    energy = _npz(clip_dir, "energy_map.npz", "energy")
    seg = _npz(clip_dir, "seg.npz", "seg")
    depth = _npz(clip_dir, "depth.npz", "depth")
    flow = _npz(clip_dir, "flow_fwd.npz", "flow_fwd")
    normals = _npz(clip_dir, "normals.npz", "normals")
    etrace = _energy_trace(clip_dir)
    etwin = _energy_trace(_twin_dir(clip_dir, meta))
    tl = np.load(os.path.join(clip_dir, "timelines.npz"))

    sev = None if sev is None else sev.astype(np.float32)
    diverg = None if diverg is None else diverg.astype(np.float32)

    vwin = [tuple(w) for w in v.get("violation_windows", [])]
    owin = [tuple(w) for w in v.get("observable_windows", [])]
    iwin = [tuple(w) for w in v.get("intervention_windows", [])]
    cwin = [tuple(w) for w in v.get("consequence_windows", [])]
    t_event = int(v.get("t_event_frame", -1))
    t_obs = int(v.get("t_observable_frame", -1))
    t_end = int(v.get("t_end_frame", -1))
    peak = (v.get("peak_residual") or {})

    # Order: what the renderer saw, then what we derived from it. RGB, energy
    # and the three geometry passes describe the scene; mask, severity, causal
    # and divergence are the annotation built on top. Reading left to right
    # therefore goes from evidence to label, and the same order is used in the
    # grid and the sheet so a panel is a panel wherever you meet it.
    panels: List[Tuple[str, str]] = [("RGB", "rgb")]
    if energy is not None:
        panels.append(("ENERGY  per body, fraction of E0", "energy"))
    if seg is not None:
        panels.append(("SEGMENTATION  per-instance tracks", "seg"))
    if depth is not None:
        panels.append(("DEPTH  metres, near->far", "depth"))
    if flow is not None:
        panels.append(("OPTICAL FLOW  hue=direction val=speed", "flow"))
    panels.append(("MASK red=violation  green=should-be", "mask"))
    if sev is not None:
        panels.append(("SEVERITY MAP", "sev"))
    if causal is not None:
        panels.append(("CAUSAL MASK", "causal"))
    if diverg is not None:
        panels.append(("DIVERGENCE (not GT)", "div"))

    n = len(panels)
    W = n * panel + (n + 1) * PAD
    H = HEADER + panel + 2 * PAD + TIMELINE
    out = np.zeros((T, H, W, 3), np.uint8)

    for t in range(T):
        f = np.full((H, W, 3), C_BG, np.uint8)
        active = bool(tl["active"][t])
        observable = bool(tl["observable"][t])
        occluded = bool(tl["occluded"][t]) if "occluded" in tl.files else False

        for i, (label, kind) in enumerate(panels):
            img = _panel(kind, t, rgb, mask, sev, causal, diverg, panel, ref,
                         energy, seg, depth, flow, normals)
            x = PAD + i * (panel + PAD)
            y = HEADER + PAD
            f[y:y + panel, x:x + panel] = img
            cv2.rectangle(f, (x, y), (x + panel - 1, y + panel - 1), (60, 60, 70), 1)
            _label(f, label, (x + 5, y + 16))
            if kind == "sev":
                _sev_scale(f, x, y, panel, float(tl["severity_t"][t]))
            if kind == "mask":
                px = int(mask[t].sum()) if mask is not None else 0
                _text(f, "%d px" % px, (x + 5, y + panel - 8), C_DIM, 0.40, 1)
            if kind == "causal" and causal is not None:
                _causal_key(f, x, y, panel, causal[t])
            if kind == "seg" and seg is not None:
                ids = [int(u) for u in np.unique(seg[t]) if u]
                _text(f, "ids %s" % ",".join(map(str, ids[:8])),
                      (x + 5, y + panel - 8), C_DIM, 0.36, 1)
            if kind == "flow" and flow is not None and t >= len(flow) - 1:
                # Kubric writes zeros here: forward flow at the last frame has
                # no next frame to point at. Saying so beats rendering a black
                # square that looks like a broken pass.
                _text(f, "undefined at last frame", (x + 5, y + panel - 8),
                      C_DIM, 0.36, 1)
            if kind == "depth" and depth is not None:
                d = np.asarray(depth[t], np.float32)
                d = d[..., 0] if d.ndim == 3 else d
                here = d < DEPTH_SENTINEL
                if here.any():
                    _text(f, "%.1f-%.1f m" % (d[here].min(), d[here].max()),
                          (x + 5, y + panel - 8), C_DIM, 0.36, 1)
            if kind == "energy":
                _energy_scale(f, x, y, panel,
                              float(np.abs(energy[t]).max()) if energy is not None
                              else None)
                if etrace is not None:
                    _energy_curve(f, x, y, panel, t, etrace, etwin)

        _header(f, W, meta, t, T, active, observable, occluded)
        _timeline(f, W, H, T, t, vwin, owin, t_event, t_obs, t_end,
                  float(tl["severity_t"][t]), peak, v, iwin, cwin)
        out[t] = f

    out_path = out_path or os.path.join(clip_dir, "overlay.mp4")
    vid.write(out, out_path, fps=int(meta.get("fps", 12)))
    return {"path": out_path, "frames": T, "panels": [p[0] for p in panels],
            "t_event": t_event, "t_observable": t_obs, "t_end": t_end,
            "violation_windows": vwin, "observable_windows": owin}


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
DEPTH_SENTINEL = 1e6


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


def _panel(kind, t, rgb, mask, sev, causal, diverg, size, ref=None, energy=None,
           seg=None, depth=None, flow=None, normals=None):
    import cv2
    base = rgb[t].astype(np.float32)

    if kind == "rgb":
        img = base
    elif kind == "mask":
        img = base.copy()
        if mask is not None:
            m = mask[t].astype(bool)
            img[m] = img[m] * 0.55 + np.array(C_MASK, np.float32) * 0.45
            edge = m ^ _erode(m)
            img[edge] = np.array(C_MASK, np.float32)
        if ref is not None:
            # Where the culprit should be, per the valid twin -- drawn *last*
            # and outline-only. The violation mask is a superset of it for
            # vanish-type families, so drawing the reference first would let the
            # red fill hide it exactly when it matters most.
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


def _header(f, W, meta, t, T, active, observable, occluded):
    """Laid out right-to-left from measured widths so nothing ever collides."""
    import cv2
    v = meta.get("violation") or {}

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
    left = "%s / %s / %s" % (meta.get("domain"), meta.get("family"),
                             meta.get("scenario"))
    _text(f, left, (PAD + 2, 22), C_TEXT, 0.52, 1)
    lx = PAD + 2 + _w(left, 0.52) + 22
    sev_bin = (v.get("intervention") or {}).get("severity_bin", "-")
    mid = "seed %s   tier %s   %s" % (meta.get("seed"), meta.get("tier"), sev_bin)
    if lx + _w(mid, 0.46) < right_limit:
        _text(f, mid, (lx, 22), C_DIM, 0.46, 1)


#: The intervention bar. Deliberately a different hue from the consequence bar:
#: they overlap at the start and the eye needs to separate "we are changing
#: something" from "the scene is wrong as a result".
C_INTERVENE = (120, 200, 255)


def _timeline(f, W, H, T, t, vwin, owin, t_event, t_obs, t_end, sev_t, peak, v,
              iwin=None, cwin=None):
    import cv2
    y0 = H - TIMELINE + 4
    x0, x1 = PAD + 2, W - PAD - 2
    span = x1 - x0

    def fx(frame):
        return int(x0 + (frame / max(T, 1)) * span)

    # THREE clocks, not two. The split is in the data but was not drawn, so a
    # `colour_shift` whose intervention ends at frame 14 still looked like it
    # ran to frame 24 -- the bar being shown was the union, which does.
    iwin = iwin if iwin else vwin
    cwin = cwin if cwin else vwin
    rows = [("intervening", "being changed", iwin, C_INTERVENE),
            ("consequence", "still wrong", cwin, C_MASK),
            ("observable", "a viewer could tell", owin, C_OBS)]

    # A filled swatch in the bar's own colour, then the name in white. Colouring
    # the text instead left the reader matching a thin coloured glyph against a
    # thick coloured bar, which is exactly the guessing this legend exists to
    # remove -- and the three colours have no conventional meaning to fall back
    # on. The gloss says what the window means, not just what it is called.
    lx = x0
    for label, gloss, _wins, colour in rows:
        cv2.rectangle(f, (lx, y0 + 3), (lx + 10, y0 + 12), colour, -1)
        cv2.rectangle(f, (lx, y0 + 3), (lx + 10, y0 + 12), (20, 20, 26), 1)
        lx += 14
        _text(f, label, (lx, y0 + 11), C_TEXT, 0.40, 1)
        lx += _w(label, 0.40) + 4
        gl = "(%s)" % gloss
        _text(f, gl, (lx, y0 + 11), C_DIM, 0.34, 1)
        lx += _w(gl, 0.34) + 16

    for row, (_label, _gloss, wins, color) in enumerate(rows):
        ry = y0 + 16 + row * 9
        cv2.rectangle(f, (x0, ry), (x1, ry + 7), (40, 40, 48), -1)
        # The row's own colour repeated at the left edge, so a bar that happens
        # to start late is still identifiable without counting rows.
        cv2.rectangle(f, (x0 - 1, ry), (x0 + 1, ry + 7), color, -1)
        for s, e in wins:
            cv2.rectangle(f, (fx(s), ry), (max(fx(e + 1) - 1, fx(s) + 2), ry + 7),
                          color, -1)

    # frame ticks every frame, labelled every 4
    ty = y0 + 45
    for i in range(T):
        cv2.line(f, (fx(i), ty), (fx(i), ty + 4), (90, 90, 100), 1)
        if i % 4 == 0:
            _text(f, str(i), (fx(i) - 3, ty + 15), C_DIM, 0.34, 1)

    # Markers. Clocks that land on the same frame -- t_event == t_obs is the
    # common case whenever nothing occludes the culprit -- are merged into one
    # label rather than overprinted.
    t_iend = max((e for _, e in iwin), default=t_event)
    marks = [(t_event, "t_event", C_INTERVENE),
             (t_iend, "t_applied", C_INTERVENE),
             (t_obs, "t_obs", C_OBS),
             (t_end, "t_end", (170, 170, 210))]
    by_frame = {}
    for frame, tag, col in marks:
        if 0 <= frame < T:
            by_frame.setdefault(frame, []).append((tag, col))
    for frame in sorted(by_frame):
        tags = by_frame[frame]
        col = tags[0][1]
        label = "=".join(t for t, _ in tags)
        cv2.line(f, (fx(frame), y0 + 14), (fx(frame), ty + 4), col, 1)
        tw = _w(label, 0.36)
        tx = fx(frame) + 3
        if tx + tw > x1:
            tx = fx(frame) - tw - 3
        _text(f, label, (tx, y0 + 13), col, 0.36, 1)

    # playhead
    px = fx(t) + max(1, span // (2 * max(T, 1)))
    cv2.line(f, (px, y0 + 14), (px, ty + 5), (255, 255, 255), 1)

    lag = v.get("observability_lag_frames", 0)
    info = ("t_event=%d  applied_to=%d  t_obs=%d  t_end=%d  lag=%d  |  "
            "severity(t)=%.3f  peak=%.3f  r=%.3f (%s)"
            % (t_event, t_iend, t_obs, t_end, lag, sev_t,
               float(peak.get("score", 0.0)), float(peak.get("value", 0.0)),
               peak.get("law", "-")))
    _text(f, info, (x0, H - 6), C_TEXT, 0.44, 1)


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
    and then did not answer. Red is the culprit -- the body the plan names --
    and blue is a body it disturbed, which is measured rather than declared.
    Counts beside each, because a level that is present in the legend and absent
    from the image is worth being able to tell apart from one that is simply
    hard to see.
    """
    import cv2

    rows = [("culprit", C_CAUSAL1, int((layer == 1).sum())),
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


# ------------------------------------------------------------------ io
def _json(p):
    with open(p) as fh:
        return json.load(fh)


def _npz(clip_dir, fname, key):
    p = os.path.join(clip_dir, fname)
    if not os.path.exists(p):
        return None
    z = np.load(p)
    return z[key] if key in z.files else z[z.files[0]]


def _rgb(clip_dir, T):
    import imageio.v2 as imageio
    p = os.path.join(clip_dir, "rgb.mp4")
    if not os.path.exists(p):
        raise FileNotFoundError("no rgb.mp4 in %s" % clip_dir)
    r = imageio.get_reader(p)
    frames = [np.asarray(x)[..., :3] for x in r]
    r.close()
    return np.stack(frames[:T]).astype(np.uint8)


def _erode(m):
    e = m.copy()
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        e &= np.roll(np.roll(m, dy, axis=0), dx, axis=1)
    return e


# ------------------------------------------------------------------ energy
def _twin_dir(clip_dir, meta):
    """Sibling clip directory named by `twin_uid`, or None."""
    twin = (meta or {}).get("twin_uid")
    if not twin:
        return None
    root = os.path.dirname(os.path.dirname(os.path.abspath(clip_dir)))
    cand = os.path.join(root, os.path.basename(os.path.dirname(twin)),
                        os.path.basename(twin))
    if os.path.isdir(cand):
        return cand
    cand = os.path.join(os.path.dirname(os.path.abspath(clip_dir)),
                        os.path.basename(twin))
    return cand if os.path.isdir(cand) else None


def _energy_trace(clip_dir):
    if not clip_dir:
        return None
    path = os.path.join(clip_dir, "energy.npz")
    if not os.path.exists(path):
        return None
    z = np.load(path)
    return {k: z[k] for k in z.files}


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

    series = [(trace["total"], (255, 120, 120))]
    if twin is not None and "total" in twin:
        series.insert(0, (twin["total"], C_REF))
    lo = min(float(np.min(s)) for s, _ in series)
    hi = max(float(np.max(s)) for s, _ in series)
    span = max(hi - lo, 1e-9)
    T = len(trace["total"])
    for s, col in series:
        pts = [(bx + int(i / max(T - 1, 1) * (bw - 1)),
                by + h - 1 - int((float(s[i]) - lo) / span * (h - 3)))
               for i in range(T)]
        for i in range(1, len(pts)):
            cv2.line(f, pts[i - 1], pts[i], col, 1, cv2.LINE_AA)
    px = bx + int(t / max(T - 1, 1) * (bw - 1))
    cv2.line(f, (px, by), (px, by + h), (255, 255, 255), 1)
    lab = "E %.2fJ" % float(trace["total"][t])
    _text(f, lab, (bx + 3, by + 11), (255, 255, 255), 0.36, 1)
    # Which line is which. Without it the panel shows two curves crossing and
    # leaves the reader to guess which one is the violation.
    lx = bx + 6 + _w(lab, 0.36)
    if len(series) > 1:
        _text(f, "valid", (lx, by + 11), C_REF, 0.34, 1)
        _text(f, "this", (lx + 6 + _w("valid", 0.34), by + 11),
              (255, 120, 120), 0.34, 1)
    # Name the channel instead of signing the number. All three are
    # non-negative by construction, so "%+.0f%%" printed a plus on every frame
    # and told the reader nothing -- and which channel fired is the part that
    # distinguishes energy appearing from energy vanishing.
    channels = (("gain", float(trace["contact_anomaly"][t])),
                ("uncaused", float(trace["free_anomaly"][t])),
                ("loss", float(trace["excess_loss"][t])))
    name, peak = max(channels, key=lambda kv: kv[1])
    if peak > 0.01:
        _text(f, "%s %.0f%%" % (name, 100 * peak), (bx + 3, by + h - 4),
              (255, 120, 120), 0.36, 1)
