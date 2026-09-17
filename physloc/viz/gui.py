"""A browser viewer over the overlay renderer: browse samples, play them at their
real frame rate, lay annotation panels beside the video, hover an object to
read it.

    python test_dataset_loader.py out/physloc_mini --gui     # http://localhost:8765

Standard library only on the server. Every panel is rendered on its own by
`overlay.Renderer.panel` -- the same drawing code as `overlay.mp4` -- and sent
as JPEG from memory, so nothing is written to disk. The page lays panels out
itself, which is what lets the RGB view be large and the rest sit beside it.

**Why playback is smooth.** The page does not ask for a frame and wait: it
queues every frame of every visible panel, nearest the playhead first, six at a
time, and plays from what has arrived on a clock at the clip's own fps. A frame
that has not arrived holds the playhead rather than stuttering. URLs carry a
per-launch token so the browser may cache them for the life of the server.

Hovering reads the segmentation (`/api/seg`, one frame, at most `SEG_MAX` px a
side) in the page, so the outline and the read-out cost no render at all.

On a remote machine, forward the port (VS Code does it when the URL is printed).
"""
from __future__ import annotations

import json
import math
import os
import threading
import time
import urllib.parse
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, List, Tuple

import numpy as np

from .. import loader
from . import overlay

#: Renderers kept warm. Each holds its clip's decoded arrays, so this bounds memory.
KEEP = 8
#: Encoded panels kept on the server, for toggling back to a view already seen.
JPEG_CACHE = 2000
#: The largest side the hover segmentation is sent at.
SEG_MAX = 256
SIZES = (128, 1024)


def serve(root: str, port: int = 8765, host: str = "127.0.0.1") -> None:
    ds = loader.PhysLocDataset(root)
    if not len(ds):
        raise SystemExit("no samples under %s" % root)
    app = _App(ds, root)

    class Handler(BaseHTTPRequestHandler):
        # Keep-alive: a page loading hundreds of panels should not pay a TCP
        # handshake for each one.
        protocol_version = "HTTP/1.1"

        def log_message(self, *_args):
            pass

        def do_GET(self):
            url = urllib.parse.urlparse(self.path)
            q = {k: v[-1] for k, v in urllib.parse.parse_qs(url.query).items()}
            try:
                if url.path == "/":
                    self._send(200, "text/html; charset=utf-8", PAGE.encode())
                elif url.path == "/api/index":
                    self._json(app.index())
                elif url.path == "/api/sample":
                    self._json(app.sample(int(q["i"])))
                elif url.path == "/api/panel":
                    body = app.panel(int(q["i"]), int(q.get("t", 0)), q.get("p", "rgb"),
                                     _split(q.get("l")), int(q.get("s", overlay.PANEL)))
                    self._send(200, "image/jpeg", body, cache=True)
                elif url.path == "/api/seg":
                    body, w, h, nbytes = app.seg(int(q["i"]), int(q.get("t", 0)))
                    self._send(200, "application/octet-stream", body, cache=True,
                               extra={"X-W": w, "X-H": h, "X-Bytes": nbytes})
                else:
                    self._send(404, "text/plain", b"not found")
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as exc:                          # noqa: BLE001
                try:
                    self._send(500, "text/plain", repr(exc).encode())
                except (BrokenPipeError, ConnectionResetError):
                    pass

        def _json(self, obj):
            self._send(200, "application/json",
                       json.dumps(_clean(obj), default=str, allow_nan=False).encode())

        def _send(self, code, ctype, body, cache=False, extra=None):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            # Panel URLs carry the launch token, so they are immutable for as
            # long as this server runs; everything else is always fresh.
            self.send_header("Cache-Control",
                             "private, max-age=86400" if cache else "no-store")
            for k, v in (extra or {}).items():
                self.send_header(k, str(v))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer((host, port), Handler)
    print("PhysLoc viewer: %d samples from %s -> http://%s:%d  (Ctrl+C to stop)"
          % (len(ds), root, "localhost" if host == "127.0.0.1" else host, port),
          flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


def _split(s) -> List[str]:
    return [x for x in (s or "").split(",") if x]


def _clean(obj):
    """JSON-safe: NaN and infinities become null -- `JSON.parse` rejects them."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    return obj


def _rounded(a, digits: int = 3) -> list:
    return np.round(np.nan_to_num(np.asarray(a, np.float64)), digits).tolist()


def _hex(rgb) -> str:
    return "#%02x%02x%02x" % tuple(int(c) for c in rgb)


class _App:
    def __init__(self, ds: "loader.PhysLocDataset", root: str):
        self.ds = ds
        self.root = root
        self.token = "%x" % int(time.time())
        self.lock = threading.Lock()
        self.renderers: "OrderedDict[tuple, overlay.Renderer]" = OrderedDict()
        self.jpegs: "OrderedDict[tuple, bytes]" = OrderedDict()

    # ---- JSON --------------------------------------------------------------
    def index(self) -> Dict[str, object]:
        samples = []
        for i, s in enumerate(self.ds.samples):
            info = dict(self.ds.info(i), i=i, uid=s.uid)
            # Keep the browser index on the same normalized vocabulary as the
            # sample endpoint: `scenario` is the scene type, `family` the
            # violation taxonomy.
            info.update(scenario=s.scenario, family=s.family,
                        condition=s.condition, level=s.level,
                        severity_bin=s.severity_bin)
            samples.append(info)
        return {"samples": samples, "layers": overlay.LAYERS, "panels": overlay.PANELS,
                "token": self.token,
                "root": os.path.basename(os.path.normpath(self.root))}

    def sample(self, i: int) -> Dict[str, object]:
        c = self.ds.samples[i]
        r = self._renderer(i, (), 256)
        meta, md = c.display_metadata, c.info
        v = meta.get("violation") or {}
        obj = c.object_table
        row = {int(iid): k for k, iid in enumerate(obj["ids"])
               if bool(obj["is_violator"][k])}
        # A `multi` clip's violators differ -- one obvious, one behind a screen
        # -- so each carries its own easy / moderate / hard beside the clip's.
        own = {int(o.get("instance_id", -1)): (o.get("difficulty") or {})
               for o in (v.get("violators") or [])}
        static = {int(x["id"]): x for x in meta.get("instances", [])}

        # Per-object energy is stored on the stable [N,T] object axis.
        energy: Dict[int, list] = {}
        if c.has("energy"):
            e = c.object_energy
            if "by_body" in e:
                by = np.asarray(e["by_body"], np.float64)
                energy = {int(object_id): _rounded(by[j], 4)
                          for j, object_id in enumerate(c.object_ids)
                          if j < by.shape[0]}

        objects = []
        if c.object_definitions:
            inst = c.instances
            for j, iid in enumerate(int(x) for x in inst["ids"]):
                m = static.get(iid, {})
                fields = m.get("static_fields") or {}
                asset = m.get("asset") or {}
                entry = {"id": iid, "name": m.get("name", str(iid)), "role": m.get("role"),
                         "analysis_group": m.get("analysis_group", "context"),
                         "category": m.get("category"), "material": fields.get("material"),
                         "mass": fields.get("mass"), "friction": fields.get("friction"),
                         "restitution": fields.get("restitution"), "static": fields.get("static"),
                         "asset": asset.get("id"), "colour": _hex(r.colour(iid)),
                         "violator": iid in row,
                         "pos": _rounded(inst["positions"][j]),
                         "vel": _rounded(inst["velocities"][j]),
                         "vis": np.asarray(inst["visibility"][j]).astype(int).tolist(),
                         "energy": energy.get(iid)}
                entry["difficulty"] = own.get(iid) or None
                if iid in row:
                    k = row[iid]
                    entry["severity"] = _rounded(obj["severity"][k])
                    for clock in ("active", "observable", "occluded"):
                        entry[clock] = np.asarray(obj[clock][k]).astype(int).tolist()
                objects.append(entry)

        tl = c.timeline
        return {
            "i": i, "uid": c.uid, "frames": c.num_frames, "fps": c.fps, "label": c.label,
            "prompt": c.prompt, "resolution": md.get("resolution"),
            "scenario": c.scenario, "family": c.family, "level": c.level,
            "condition": c.condition, "severity_bin": c.severity_bin,
            "medium": md.get("physics_medium"), "domain": md.get("domain"),
            "difficulty": meta.get("difficulty"),
            "violation": {k: v.get(k) for k in (
                "t_event_frame", "t_observable_frame", "t_end_frame",
                "observability_lag_frames", "violation_windows", "observable_windows",
                "violator_timing", "peak_residual")} if v else None,
            "magnitude": (v.get("intervention") or {}).get("magnitude") if v else None,
            "timeline": {k: _rounded(a) for k, a in tl.items()},
            "objects": objects,
        }

    # ---- images ------------------------------------------------------------
    def _renderer(self, i: int, layers, size: int) -> "overlay.Renderer":
        key = (i, tuple(layers), size)
        with self.lock:
            r = self.renderers.get(key)
            if r is None:
                r = overlay.Renderer(self.ds.samples[i], layers, ("rgb",), size)
                self.renderers[key] = r
                while len(self.renderers) > KEEP:
                    _, old = self.renderers.popitem(last=False)
                    if all(x.clip is not old.clip for x in self.renderers.values()):
                        old.clip.release()
            self.renderers.move_to_end(key)
            return r

    def panel(self, i: int, t: int, name: str, layers, size: int) -> bytes:
        import cv2

        size = max(SIZES[0], min(SIZES[1], int(size)))
        layers = tuple(layers) if name == "rgb" else ()
        key = (i, t, name, layers, size)
        with self.lock:
            hit = self.jpegs.get(key)
        if hit is not None:
            return hit
        r = self._renderer(i, layers, size)
        img = r.panel(name, max(0, min(t, r.T - 1)))
        _ok, buf = cv2.imencode(".jpg", img[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 90])
        body = buf.tobytes()
        with self.lock:
            self.jpegs[key] = body
            while len(self.jpegs) > JPEG_CACHE:
                self.jpegs.popitem(last=False)
        return body

    def seg(self, i: int, t: int) -> Tuple[bytes, int, int, int]:
        seg = self.ds.samples[i].segmentations
        t = max(0, min(t, len(seg) - 1))
        step = max(1, math.ceil(max(seg.shape[1:]) / SEG_MAX))
        a = np.asarray(seg[t, ::step, ::step])
        if a.size and int(a.max()) < 256:
            return a.astype(np.uint8).tobytes(), a.shape[1], a.shape[0], 1
        return a.astype("<u2").tobytes(), a.shape[1], a.shape[0], 2


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PhysLoc Viewer</title>
<style>
:root{
  --bg:#0d0e12; --panel:#14161c; --panel2:#1b1e26; --line:#252933; --line2:#313643;
  --text:#e9ebf1; --dim:#a0a6b3; --faint:#6c7281; --accent:#6aa6ff; --accent2:#2f7ae0;
  --bad:#ff6b6b; --good:#4fd18b; --amber:#f5b83d; --r:12px;
  --left:276px; --right:312px;
}
*{box-sizing:border-box}
html,body{height:100%}
body{margin:0;font:13px/1.45 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
  background:var(--bg);color:var(--text);display:grid;overflow:hidden;height:100vh;
  grid-template-columns:var(--left) minmax(0,1fr) var(--right);transition:grid-template-columns .18s}
body{position:relative}
body.no-left{--left:0px} body.no-right{--right:0px}
.resize-handle{position:absolute;top:0;bottom:0;width:9px;z-index:12;cursor:col-resize}
#resizeLeft{left:calc(var(--left) - 4px)} #resizeRight{right:calc(var(--right) - 4px)}
.resize-handle:hover,.resize-handle.dragging{background:rgba(106,166,255,.16)}
button{font:inherit;color:inherit}
::-webkit-scrollbar{width:8px;height:8px} ::-webkit-scrollbar-thumb{background:#2a2e38;border-radius:8px}
.icon{background:transparent;border:1px solid transparent;border-radius:8px;height:30px;min-width:30px;
  padding:0 8px;cursor:pointer;color:var(--dim);display:inline-flex;align-items:center;justify-content:center;gap:6px}
.icon:hover{background:var(--panel2);color:var(--text)}
.icon.on{color:var(--text);background:var(--panel2);border-color:var(--line2)}
.icon svg{width:16px;height:16px}
aside{background:var(--panel);min-height:0;min-width:0;display:flex;flex-direction:column;overflow:hidden}
#left{border-right:1px solid var(--line)} #right{border-left:1px solid var(--line)}
.brand{display:flex;align-items:center;gap:9px;padding:14px 14px 10px;font-weight:650;font-size:14px;white-space:nowrap}
.logo{width:18px;height:18px;border-radius:5px;background:conic-gradient(from 210deg,#ff6b6b,#f5b83d,#6aa6ff,#ff6b6b)}
.brand small{color:var(--faint);font-weight:450;margin-left:auto;overflow:hidden;text-overflow:ellipsis}
.search{margin:0 12px 8px}
.search input{width:100%;background:var(--bg);border:1px solid var(--line);border-radius:9px;color:var(--text);
  padding:7px 10px;font:inherit;outline:none}
.search input:focus{border-color:var(--accent2)}
.filters{display:flex;flex-wrap:wrap;gap:6px;padding:0 12px 10px}
.filters select{appearance:none;-webkit-appearance:none;background:var(--panel2);border:1px solid var(--line);
  color:var(--dim);border-radius:999px;padding:3px 11px;font-size:12px;cursor:pointer;max-width:100%;outline:none}
.filters select:hover{color:var(--text)}
.filters select.set{color:#dce8ff;border-color:#35507f;background:#1b2740}
.count{padding:6px 14px 4px;color:var(--faint);font-size:11.5px;display:flex;justify-content:space-between;
  align-items:center;border-top:1px solid var(--line)}
.link{background:none;border:none;color:var(--accent);cursor:pointer;font-size:11.5px;padding:0}
#clips{list-style:none;margin:0;padding:4px 8px 14px;overflow:auto;flex:1}
#clips li{padding:7px 9px;border-radius:9px;cursor:pointer;display:grid;grid-template-columns:8px minmax(0,1fr);
  gap:1px 10px;align-items:center}
#clips li:hover{background:var(--panel2)}
#clips li.on{background:#1c2638;box-shadow:inset 0 0 0 1px #2d4670}
.dot{width:8px;height:8px;border-radius:50%;grid-row:span 2}
.dot.invalid{background:var(--bad)} .dot.valid{background:var(--good)}
#clips .t{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
#clips .s{color:var(--faint);font-size:11.5px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
#clips .more{color:var(--faint);font-size:11.5px;text-align:center;display:block;cursor:default}
main{display:flex;flex-direction:column;min-width:0;min-height:0}
.top{display:flex;align-items:flex-start;gap:10px;padding:10px 14px 8px}
.title{min-width:0;flex:1}
.title h2{margin:0;font-size:16px;font-weight:650;display:flex;gap:8px;align-items:center;flex-wrap:wrap}
.title p{margin:3px 0 0;color:var(--dim);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.badge{display:inline-flex;align-items:center;gap:5px;font-size:11.5px;font-weight:500;padding:1px 8px;
  border-radius:999px;background:var(--panel2);border:1px solid var(--line2);color:var(--dim);white-space:nowrap}
.badge.invalid{color:#ffb8b8;border-color:#5a2b2f;background:#26161a}
.badge.valid{color:#a9f0c8;border-color:#24503a;background:#12221a}
.badge.easy{color:#cfe2ff;border-color:#2a3d5c} .badge.moderate{color:#dce9ff;border-color:#35598f;background:#1a2a45}
.badge.hard{color:#fff;background:#23478a;border-color:#3764b3}
.siblings{display:flex;gap:5px;flex-wrap:wrap;margin-top:7px;align-items:center}
.siblings .lab{color:var(--faint);font-size:11.5px;margin-right:2px}
.sib{font-size:11.5px;padding:1px 9px;border-radius:7px;border:1px solid var(--line2);background:transparent;
  color:var(--dim);cursor:pointer}
.sib:hover{color:var(--text);border-color:#465068} .sib.on{background:#1c2638;color:var(--text);border-color:var(--accent2)}
.stage{flex:1;min-height:0;display:flex;gap:12px;padding:4px 14px 10px}
.view{flex:1;min-width:0;position:relative;display:flex;align-items:center;justify-content:center;
  background:radial-gradient(ellipse at center,#171a21 0,#0a0b0e 75%);border-radius:var(--r);border:1px solid var(--line);overflow:hidden}
.wrap{position:relative;line-height:0}
#main{display:block;width:100%;height:100%;border-radius:8px;background:#000;box-shadow:0 18px 50px rgba(0,0,0,.5);cursor:crosshair}
#hover{position:absolute;inset:0;width:100%;height:100%;pointer-events:none;border-radius:8px}
.pill{position:absolute;font-size:11.5px;padding:3px 9px;border-radius:999px;background:rgba(13,14,18,.72);
  color:var(--dim);pointer-events:none;backdrop-filter:blur(6px);border:1px solid rgba(255,255,255,.06)}
#viewlabel{top:10px;left:12px}
#live{top:10px;right:12px;color:#ffc2c2;background:rgba(120,20,24,.55);display:none;align-items:center;gap:6px;font-weight:600}
#live::before{content:"";width:7px;height:7px;border-radius:50%;background:#ff5a5a;box-shadow:0 0 8px #ff5a5a}
#live.on{display:inline-flex} #live.occ{color:#e6e6ee;background:rgba(70,74,88,.7)} #live.occ::before{background:#b4b8c4;box-shadow:none}
#buffer{bottom:10px;left:12px;display:none} #buffer.on{display:block}
#hint{bottom:10px;right:12px}
.rail{width:var(--rail,230px);flex:none;display:flex;flex-direction:column;gap:10px;overflow:auto;padding:0 2px 2px 0}
.rail.hidden{display:none}
.thumb{position:relative;border-radius:10px;overflow:hidden;border:1px solid var(--line);background:#000;
  cursor:pointer;line-height:0;flex:none;transition:border-color .12s,transform .12s}
.thumb:hover{border-color:#4a6fae}
.thumb canvas{width:100%;aspect-ratio:1/1;display:block}
.thumb .cap{position:absolute;left:0;right:0;top:0;line-height:1.3;font-size:11.5px;padding:5px 8px 12px;
  background:linear-gradient(rgba(0,0,0,.78),rgba(0,0,0,0));color:#e4e7ee;display:flex;justify-content:space-between;gap:6px}
.thumb .cap small{color:#a9afbd;font-size:10.5px;display:block}
.thumb .x{opacity:0;border:none;background:rgba(0,0,0,.5);border-radius:6px;cursor:pointer;color:#fff;height:20px;width:20px;line-height:1;flex:none}
.thumb:hover .x{opacity:1}
.addpanel{border:1px dashed var(--line2);border-radius:10px;color:var(--faint);background:transparent;padding:9px;cursor:pointer;flex:none}
.addpanel:hover{color:var(--text);border-color:#4a6fae}
.transport{padding:0 14px 12px}
.controls{display:flex;align-items:center;gap:6px;margin-bottom:7px}
.play{width:36px;height:36px;border-radius:50%;border:none;background:var(--text);color:#0d0e12;cursor:pointer;
  display:grid;place-items:center;flex:none;transition:transform .08s}
.play:hover{background:#fff} .play:active{transform:scale(.94)} .play svg{width:15px;height:15px}
.clock{font-variant-numeric:tabular-nums;color:var(--dim);font-size:12.5px;margin-left:6px;white-space:nowrap}
.clock b{color:var(--text);font-weight:600}
.spacer{flex:1}
.seg{display:inline-flex;background:var(--panel2);border:1px solid var(--line);border-radius:9px;padding:2px}
.seg button{background:transparent;border:none;color:var(--dim);padding:3px 9px;border-radius:7px;cursor:pointer;font-size:12px}
.seg button:hover{color:var(--text)} .seg button.on{background:#2b3242;color:var(--text)}
#timeline{width:100%;display:block;cursor:pointer;border-radius:10px;background:var(--panel);border:1px solid var(--line);touch-action:none}
.legend{display:flex;gap:16px;color:var(--faint);font-size:11.5px;margin-top:7px;flex-wrap:wrap}
.legend i{display:inline-block;width:12px;height:4px;border-radius:2px;margin-right:6px;vertical-align:middle}
.legend b{color:var(--dim);font-weight:500}
.tabs{display:flex;gap:2px;padding:8px 10px 0;border-bottom:1px solid var(--line)}
.tabs button{flex:1;background:transparent;border:none;border-bottom:2px solid transparent;padding:8px 4px 9px;
  color:var(--dim);cursor:pointer;font-weight:550}
.tabs button:hover{color:var(--text)} .tabs button.on{color:var(--text);border-bottom-color:var(--accent)}
.pane{overflow:auto;flex:1;padding:4px 12px 18px;display:none} .pane.on{display:block}
.group{margin-top:14px} .group h5{margin:0 0 6px 2px;font-size:10.5px;font-weight:650;text-transform:uppercase;letter-spacing:.7px;color:var(--faint)}
.toggle{display:flex;align-items:center;gap:10px;padding:6px 8px;border-radius:9px;cursor:pointer;user-select:none}
.toggle:hover{background:var(--panel2)}
.toggle .name{flex:1;min-width:0} .toggle .desc{display:block;color:var(--faint);font-size:11.5px;line-height:1.3}
.swatch{width:10px;height:10px;border-radius:3px;display:inline-block;flex:none}
.switch{width:30px;height:18px;border-radius:999px;background:#2a2f3a;position:relative;flex:none;transition:background .15s}
.switch::after{content:"";position:absolute;top:2px;left:2px;width:14px;height:14px;border-radius:50%;background:#8d93a1;transition:transform .15s,background .15s}
.toggle.on .switch{background:var(--accent2)} .toggle.on .switch::after{transform:translateX(12px);background:#fff}
.chips{display:flex;flex-wrap:wrap;gap:6px}
.chip{border:1px solid var(--line2);background:transparent;color:var(--dim);border-radius:999px;padding:3px 11px;cursor:pointer;font-size:12px}
.chip:hover{color:var(--text);border-color:#465068} .chip.on{background:#1c2638;border-color:#35507f;color:var(--text)}
.chip.main{background:var(--accent2);border-color:var(--accent2);color:#fff}
.hint{color:var(--faint);font-size:11.5px;margin:6px 2px 0}
.card{background:var(--panel2);border:1px solid var(--line);border-radius:12px;padding:11px 12px;margin-top:12px}
.card h4{margin:0 0 8px;font-size:12.5px;display:flex;align-items:center;gap:7px}
.kv{display:grid;grid-template-columns:auto 1fr;gap:3px 12px;font-size:12px}
.kv span:nth-child(odd){color:var(--faint)} .kv span:nth-child(even){text-align:right;font-variant-numeric:tabular-nums;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.uid{font:11px ui-monospace,SFMono-Regular,Menlo,monospace;color:var(--faint);word-break:break-all;margin-top:6px}
table.factors{width:100%;border-collapse:collapse;font-size:12px}
table.factors td{padding:4px 0;border-top:1px solid var(--line)}
table.factors td:nth-child(2){text-align:right;font-variant-numeric:tabular-nums;color:var(--dim);padding-right:8px}
table.factors td:last-child{text-align:right;width:70px}
table.factors tr.bind td:first-child{font-weight:650}
.lvl{font-size:11px;padding:0 7px;border-radius:999px;display:inline-block}
.lvl.easy{background:#1d2c44;color:#cfe2ff} .lvl.moderate{background:#23406f;color:#e0ebff} .lvl.hard{background:#3764b3;color:#fff}
.objlist{display:flex;flex-direction:column;gap:2px;margin-top:4px}
.obj{display:flex;align-items:center;gap:8px;padding:5px 8px;border-radius:8px;cursor:pointer}
.obj:hover{background:var(--panel2)} .obj.on{background:#1c2638}
.obj .role{margin-left:auto;color:var(--faint);font-size:11.5px}
.spark{width:100%;height:42px;display:block;margin-top:4px}
.sparklab{display:flex;justify-content:space-between;color:var(--faint);font-size:11px;margin-top:8px}
#tip{position:fixed;z-index:20;pointer-events:none;background:rgba(17,19,25,.96);border:1px solid var(--line2);border-radius:12px;
  padding:10px 12px;min-width:196px;max-width:270px;box-shadow:0 14px 34px rgba(0,0,0,.55);display:none;font-size:12px;backdrop-filter:blur(8px)}
#tip h4{margin:0 0 7px;font-size:13px;display:flex;align-items:center;gap:7px}
#tip .foot{color:var(--faint);font-size:11px;margin-top:7px}
kbd{font:11px ui-monospace,monospace;background:var(--panel2);border:1px solid var(--line2);border-bottom-width:2px;border-radius:5px;padding:0 6px;color:var(--text)}
#help{position:fixed;inset:0;background:rgba(0,0,0,.55);display:none;place-items:center;z-index:30}
#help.on{display:grid}
#help .box{background:var(--panel);border:1px solid var(--line2);border-radius:16px;padding:18px 22px 14px;min-width:360px;box-shadow:0 20px 60px rgba(0,0,0,.6)}
#help h3{margin:0 0 10px;font-size:14px}
#help .row{display:flex;justify-content:space-between;gap:30px;padding:4px 0;color:var(--dim)}
.empty{color:var(--faint);padding:24px 8px;text-align:center}
@media (max-width:1180px){body{--right:0px} body.show-right{--right:312px}}
@media (max-width:860px){body{--left:0px} body.show-left{--left:276px} .rail{--rail:150px}}
</style></head><body>
<aside id="left">
  <div class="brand"><span class="logo"></span>PhysLoc <small id="rootname"></small></div>
  <div class="search"><input id="q" placeholder="Search samples   /" autocomplete="off" spellcheck="false"></div>
  <div class="filters" id="filters"></div>
  <div class="count"><span id="count"></span><button class="link" id="clear">Reset filters</button></div>
  <ul id="clips"></ul>
</aside>
<div class="resize-handle" id="resizeLeft" title="Resize sample list"></div>
<main>
  <div class="top">
    <button class="icon" id="toggleLeft" title="Sample list  ( [ )"><svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5"><rect x="1.5" y="2.5" width="13" height="11" rx="2"/><path d="M6 2.5v11"/></svg></button>
    <div class="title"><h2 id="title">Loading...</h2><p id="prompt"></p><div class="siblings" id="siblings"></div></div>
    <button class="icon" id="helpBtn" title="Keyboard shortcuts  ( ? )">?</button>
    <button class="icon" id="toggleRight" title="Inspector  ( ] )"><svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5"><rect x="1.5" y="2.5" width="13" height="11" rx="2"/><path d="M10 2.5v11"/></svg></button>
  </div>
  <div class="stage">
    <div class="view" id="view">
      <div class="wrap" id="wrap"><canvas id="main"></canvas><canvas id="hover"></canvas></div>
      <span class="pill" id="viewlabel">RGB</span>
      <span class="pill" id="live">VIOLATION</span>
      <span class="pill" id="buffer"></span>
      <span class="pill" id="hint">hover an object &middot; click to pin</span>
    </div>
    <div class="rail" id="rail"></div>
  </div>
  <div class="transport">
    <div class="controls">
      <button class="play" id="play" title="Play / pause  (space)"></button>
      <button class="icon" id="prev" title="Previous frame  ( &larr; )"><svg viewBox="0 0 16 16" fill="currentColor"><path d="M10.5 3.5v9L4.5 8z"/></svg></button>
      <button class="icon" id="next" title="Next frame  ( &rarr; )"><svg viewBox="0 0 16 16" fill="currentColor"><path d="M5.5 3.5v9l6-4.5z"/></svg></button>
      <span class="clock" id="clock"></span>
      <span class="spacer"></span>
      <div class="seg" id="speed" title="Playback speed  ( - / + )"></div>
      <button class="icon on" id="loop" title="Loop  ( L )"><svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M3 7.5V6a2 2 0 0 1 2-2h7.5M10.5 2l2 2-2 2M13 8.5V10a2 2 0 0 1-2 2H3.5M5.5 14l-2-2 2-2"/></svg></button>
    </div>
    <canvas id="timeline"></canvas>
    <div class="legend" id="legend"></div>
  </div>
</main>
<div class="resize-handle" id="resizeRight" title="Resize inspector"></div>
<aside id="right">
  <div class="tabs"><button data-tab="view" class="on">View</button><button data-tab="clip">Sample</button><button data-tab="object">Objects</button></div>
  <div class="pane on" id="pane-view"></div>
  <div class="pane" id="pane-clip"></div>
  <div class="pane" id="pane-object"></div>
</aside>
<div id="tip"></div>
<div id="help"><div class="box"><h3>Keyboard</h3>
  <div class="row"><span>Play / pause</span><kbd>space</kbd></div>
  <div class="row"><span>Previous / next frame</span><span><kbd>&larr;</kbd> <kbd>&rarr;</kbd></span></div>
  <div class="row"><span>Jump 10 frames</span><span><kbd>shift</kbd> + <kbd>&larr;</kbd> <kbd>&rarr;</kbd></span></div>
  <div class="row"><span>First / last frame</span><span><kbd>home</kbd> <kbd>end</kbd></span></div>
  <div class="row"><span>Jump to the violation</span><kbd>e</kbd></div>
  <div class="row"><span>Previous / next sample</span><span><kbd>&uarr;</kbd> <kbd>&darr;</kbd></span></div>
  <div class="row"><span>Valid twin &harr; last violated sample</span><kbd>t</kbd></div>
  <div class="row"><span>Slower / faster</span><span><kbd>-</kbd> <kbd>+</kbd></span></div>
  <div class="row"><span>Loop</span><kbd>l</kbd></div>
  <div class="row"><span>Toggle violation mask</span><kbd>m</kbd></div>
  <div class="row"><span>Sample list / inspector</span><span><kbd>[</kbd> <kbd>]</kbd></span></div>
  <div class="row"><span>Search</span><kbd>/</kbd></div>
  <div class="row"><span>Unpin object, close</span><kbd>esc</kbd></div>
</div></div>
<script>
"use strict";
const $ = s => document.querySelector(s);
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]));
const pretty = s => String(s ?? "").replace(/_/g, " ");
const fmt = (x, d = 2) => x == null || Number.isNaN(+x) ? "-" : (+x).toFixed(d);

const FILTERS = [["scenario", "Scenario"], ["family", "Violation family"], ["level", "Level"],
                 ["label", "Result"], ["condition", "Condition"], ["severity_bin", "Severity"]];
const LAYER_GROUPS = [
  ["Annotation", ["violation", "visible", "severity", "causal", "reference"]],
  ["Geometry", ["bbox2d", "bbox3d", "centers", "velocity", "labels", "events"]]];
const LAYER_NAME = {violation: "Violation mask", visible: "Visible part", severity: "Severity heat",
  causal: "Causal bodies", reference: "Lawful position", bbox2d: "2D boxes", bbox3d: "3D boxes",
  centers: "Centres", velocity: "Velocity", labels: "Labels", events: "Collisions"};
const LAYER_SWATCH = {violation: "#ff4646", visible: "#ffffff", severity: "#f7931e", causal: "#5aa0ff",
  reference: "#5aeb78", bbox2d: "#c3c8d4", bbox3d: "#c3c8d4", centers: "#c3c8d4", velocity: "#c3c8d4",
  labels: "#c3c8d4", events: "#ffe65a"};
const PRESETS = [["Clean", []], ["Mask", ["violation", "reference"]], ["Causal", ["causal", "reference"]],
                 ["Severity", ["severity"]], ["Geometry", ["bbox3d", "velocity", "labels"]]];
const SPEEDS = [0.25, 0.5, 1, 2];
const QUALITY = [["Auto", 0], ["384", 384], ["512", 512], ["768", 768]];
const THUMB = 256;
const VIOLATION_ONLY = ["mask", "severity", "causal", "divergence"];
const ICON_PLAY = '<svg viewBox="0 0 16 16" fill="currentColor"><path d="M4.5 2.8v10.4L13 8z"/></svg>';
const ICON_PAUSE = '<svg viewBox="0 0 16 16" fill="currentColor"><rect x="3.5" y="2.8" width="3" height="10.4" rx="1"/><rect x="9.5" y="2.8" width="3" height="10.4" rx="1"/></svg>';

const S = {
  index: null, byUid: new Map(), shown: [], cur: -1, clip: null, sampleCache: new Map(),
  t: 0, playing: false, speed: 1, loop: true, acc: 0, last: 0, lastInvalid: -1,
  layers: ["violation", "reference"], panels: ["segmentation", "depth", "mask", "severity"], main: "rgb",
  quality: 0, mainSize: 512, tab: "view",
  objectGroup: "subjects",
  mouse: null, hoverId: 0, pinned: 0, seg: new Map(), segDone: new Map(),
};

// ------------------------------------------------------------ image queue
// Every visible panel of every frame, nearest the playhead first, a few at a
// time. Loaded images live in `imgs`; the playhead only moves onto frames
// that are there, so playback holds instead of stuttering.
const imgs = new Map();
const Q = {list: [], queued: new Set(), inflight: 0, max: 6};

function panelUrl(name, size, t) {
  const l = name === "rgb" ? S.layers.join(",") : "";
  return `api/panel?v=${S.index.token}&i=${S.cur}&t=${t}&p=${name}&s=${size}&l=${l}`;
}
function request(url, urgent) {
  if (imgs.has(url)) return;
  if (Q.queued.has(url)) {
    if (urgent) { Q.list.splice(Q.list.indexOf(url), 1); Q.list.unshift(url); }
    return;
  }
  Q.queued.add(url);
  urgent ? Q.list.unshift(url) : Q.list.push(url);
  pump();
}
function pump() {
  while (Q.inflight < Q.max && Q.list.length) {
    const url = Q.list.shift();
    Q.queued.delete(url);
    const im = new Image();
    Q.inflight++;
    im.onload = () => {
      Q.inflight--; imgs.set(url, im); trim();
      if (url.includes(`&i=${S.cur}&t=${S.t}&`)) draw();
      bufferState(); pump();
    };
    im.onerror = () => { Q.inflight--; pump(); };
    im.src = url;
  }
}
function trim() {
  while (imgs.size > 2500) imgs.delete(imgs.keys().next().value);
}
function views() {
  const out = [[S.main, S.mainSize]];
  for (const p of railPanels()) out.push([p, THUMB]);
  return out;
}
function plan() {
  if (!S.clip) return;
  Q.list = []; Q.queued.clear();
  const T = S.clip.frames, vs = views();
  for (let k = 0; k < T; k++) {
    const t = (S.t + k) % T;
    for (const [name, size] of vs) request(panelUrl(name, size, t));
  }
  bufferState();
}
function bufferState() {
  if (!S.clip) return;
  let n = 0;
  for (let t = 0; t < S.clip.frames; t++) if (imgs.has(panelUrl(S.main, S.mainSize, t))) n++;
  const b = $("#buffer");
  b.classList.toggle("on", n < S.clip.frames);
  b.textContent = `loading ${n} / ${S.clip.frames} frames`;
  if (!S.playing) drawTimeline();
}

// ------------------------------------------------------------ boot
async function boot() {
  S.index = await (await fetch("api/index")).json();
  S.index.samples.forEach(c => S.byUid.set(c.uid, c.i));
  $("#rootname").textContent = S.index.root;
  document.title = `${S.index.root} - PhysLoc Viewer`;
  buildFilters();
  buildSpeed();
  wire();
  const h = readHash();
  if (h.layers !== undefined) S.layers = h.layers.filter(x => x in S.index.layers);
  if (h.panels !== undefined) S.panels = h.panels.filter(x => x in S.index.panels);
  if (h.main && h.main in S.index.panels) S.main = h.main;
  list();
  const start = h.uid && S.byUid.has(h.uid) ? S.byUid.get(h.uid) : (S.shown[0] ?? 0);
  await openClip(start, h.t ?? 0);
}

function readHash() {
  const p = new URLSearchParams(location.hash.slice(1)), out = {};
  if (p.has("clip")) out.uid = p.get("clip");
  if (p.has("layers")) out.layers = p.get("layers").split(",").filter(Boolean);
  if (p.has("panels")) out.panels = p.get("panels").split(",").filter(Boolean);
  if (p.has("main")) out.main = p.get("main");
  if (p.has("t")) out.t = +p.get("t");
  return out;
}
let hashTimer = 0;
function saveHash() {
  clearTimeout(hashTimer);
  hashTimer = setTimeout(() => {
    if (!S.clip) return;
    const p = new URLSearchParams({clip: S.clip.uid, main: S.main, layers: S.layers.join(","),
                                   panels: S.panels.join(","), t: S.t});
    history.replaceState(null, "", "#" + p.toString());
  }, 250);
}

// ------------------------------------------------------------ clip list
function buildFilters() {
  const box = $("#filters");
  for (const [f, label] of FILTERS) {
    const counts = {};
    for (const c of S.index.samples) { const v = c[f] ?? "-"; counts[v] = (counts[v] || 0) + 1; }
    const values = Object.keys(counts).sort();
    if (values.length < 2 && f !== "label") continue;
    const sel = document.createElement("select");
    sel.id = "f_" + f;
    sel.innerHTML = `<option value="">${label}: all</option>` +
      values.map(v => `<option value="${esc(v)}">${esc(pretty(v))}  (${counts[v]})</option>`).join("");
    sel.onchange = () => { sel.classList.toggle("set", !!sel.value); list(); };
    box.append(sel);
  }
  $("#q").oninput = list;
  $("#clear").onclick = () => {
    document.querySelectorAll("#filters select").forEach(s => { s.value = ""; s.classList.remove("set"); });
    $("#q").value = ""; list();
  };
}
function list() {
  const want = FILTERS.map(([f]) => [f, $("#f_" + f)?.value || ""]).filter(([, v]) => v);
  const words = $("#q").value.toLowerCase().split(/\s+/).filter(Boolean);
  S.shown = S.index.samples.filter(c =>
    want.every(([f, v]) => String(c[f] ?? "-") === v) &&
    words.every(w => c.uid.toLowerCase().includes(w))).map(c => c.i);
  $("#count").textContent = `${S.shown.length} of ${S.index.samples.length} samples`;
  const ul = $("#clips");
  ul.innerHTML = "";
  const LIMIT = 400;
  for (const i of S.shown.slice(0, LIMIT)) {
    const c = S.index.samples[i];
    const li = document.createElement("li");
    li.dataset.i = i;
    if (i === S.cur) li.className = "on";
    const what = c.label === "valid" ? "valid twin" : pretty(c.family);
    li.innerHTML = `<span class="dot ${c.label}"></span><span class="t">Scenario: ${esc(pretty(c.scenario))}</span>` +
      `<span class="s">Violation family: ${esc(what)} &middot; ${esc(c.level)} &middot; seed ${esc(c.seed)} &middot; ${esc(c.condition)}${c.severity_bin ? " &middot; " + esc(c.severity_bin) : ""}</span>`;
    // A clip selected from the list starts at its first frame.
    li.onclick = () => openClip(i);
    ul.append(li);
  }
  if (S.shown.length > LIMIT) {
    const li = document.createElement("li");
    li.innerHTML = `<span></span><span class="more">+${S.shown.length - LIMIT} more &mdash; narrow the filters</span>`;
    ul.append(li);
  }
  if (!S.shown.length) ul.innerHTML = `<div class="empty">No sample matches.</div>`;
}
function markList() {
  document.querySelectorAll("#clips li").forEach(li => {
    const on = +li.dataset.i === S.cur;
    li.classList.toggle("on", on);
    if (on) li.scrollIntoView({block: "nearest"});
  });
}

// ------------------------------------------------------------ opening a clip
async function openClip(i, t = 0) {
  if (i == null || i < 0) return;
  let clip = S.sampleCache.get(i);
  if (!clip) {
    $("#title").textContent = "Loading...";
    const r = await fetch("api/sample?i=" + i);
    if (!r.ok) { $("#title").textContent = "Could not load sample: " + (await r.text()); return; }
    clip = await r.json();
    S.sampleCache.set(i, clip);
    if (S.sampleCache.size > 12) S.sampleCache.delete(S.sampleCache.keys().next().value);
  }
  if (S.clip && S.clip.label === "invalid") S.lastInvalid = S.cur;
  S.cur = i; S.clip = clip;
  S.t = Math.max(0, Math.min(t, clip.frames - 1));
  S.hoverId = 0; S.pinned = 0; S.acc = 0;
  // A valid twin has nothing to show in a violation panel: keep the picture.
  if (!clip.violation && VIOLATION_ONLY.includes(S.main)) setMain("rgb", false);
  clip.byId = new Map(clip.objects.map(o => [o.id, o]));
  markList();
  header(); rail(); layout(); plan(); inspector(); legend(); draw(); saveHash();
}

function header() {
  const c = S.clip, d = c.difficulty;
  const label = c.label === "valid" ? `<span class="badge valid">valid twin</span>`
                                    : `<span class="badge invalid">${esc(pretty(c.family))}</span>`;
  $("#title").innerHTML = `Scenario: ${esc(pretty(c.scenario))} ${label}` +
    `<span class="badge">${esc(c.level)}</span><span class="badge">${esc(c.condition)}</span>` +
    (c.severity_bin ? `<span class="badge">${esc(c.severity_bin)}</span>` : "") +
    (d ? `<span class="badge ${d.level}" title="detection difficulty; set by ${esc((d.binding_factors || []).join(", "))}">${d.level}</span>` : "");
  $("#prompt").textContent = c.prompt || c.uid;
  const pair = S.index.samples.filter(x => x.pair_uid === S.index.samples[S.cur].pair_uid);
  const box = $("#siblings");
  box.innerHTML = pair.length > 1 ? `<span class="lab">same scene</span>` : "";
  for (const x of pair) {
    if (pair.length < 2) break;
    const b = document.createElement("button");
    b.className = "sib" + (x.i === S.cur ? " on" : "");
    b.textContent = x.label === "valid" ? "valid" : pretty(x.family) + (x.severity_bin && x.severity_bin !== "strong" ? " " + x.severity_bin : "");
    b.onclick = () => openClip(x.i);
    box.append(b);
  }
}

// ------------------------------------------------------------ layout
function railPanels() {
  const out = S.panels.filter(p => p !== S.main);
  if (S.main !== "rgb" && !out.includes("rgb")) out.unshift("rgb");
  return out;
}
function titleOf(name) {
  const full = S.index.panels[name] || name;
  const [head, ...rest] = full.split(/\s{2,}/);
  const h = head.toLowerCase();
  return [h.charAt(0).toUpperCase() + h.slice(1), rest.join(" ")];
}
function rail() {
  const box = $("#rail");
  box.innerHTML = "";
  for (const name of railPanels()) {
    const [title, sub] = titleOf(name);
    const d = document.createElement("div");
    d.className = "thumb";
    d.title = "Click to show this panel large";
    d.innerHTML = `<canvas data-panel="${name}"></canvas><div class="cap"><span>${esc(title)}${sub ? `<small>${esc(sub)}</small>` : ""}</span>` +
      (name === "rgb" ? "" : `<button class="x" title="Remove">&times;</button>`) + `</div>`;
    d.onclick = e => {
      if (e.target.classList.contains("x")) { e.stopPropagation(); S.panels = S.panels.filter(p => p !== name); rail(); plan(); inspector(); draw(); saveHash(); return; }
      setMain(name);
    };
    box.append(d);
  }
  const add = document.createElement("button");
  add.className = "addpanel";
  add.textContent = "+ add panel";
  add.onclick = () => { showTab("view"); setRight(true); $("#pane-view .sidechips")?.scrollIntoView({block: "center"}); };
  box.append(add);
}
function setMain(name, redraw = true) {
  if (name === S.main) return;
  const old = S.main;
  const k = S.panels.indexOf(name);
  if (k >= 0) S.panels[k] = old; else if (!S.panels.includes(old)) S.panels.unshift(old);
  S.panels = [...new Set(S.panels.filter(p => p !== name))];
  S.main = name;
  S.hoverId = 0;
  if (redraw) { rail(); plan(); inspector(); draw(); saveHash(); }
}
function layout() {
  if (!S.clip) return;
  const v = $("#view").getBoundingClientRect();
  const res = S.clip.resolution || [1, 1];
  const aspect = S.main === "rgb" || S.main === "valid" ? res[0] / res[1] : 1;
  const bw = v.width - 28, bh = v.height - 28;
  let w = Math.min(bw, bh * aspect), h = w / aspect;
  w = Math.max(64, Math.floor(w)); h = Math.max(64, Math.floor(h));
  const wrap = $("#wrap");
  wrap.style.width = w + "px"; wrap.style.height = h + "px";
  const dpr = window.devicePixelRatio || 1;
  const hover = $("#hover");
  hover.width = Math.round(w * dpr); hover.height = Math.round(h * dpr);
  const want = S.quality || [256, 384, 512, 640, 768, 1024].find(s => s >= Math.max(w, h) * dpr * 0.85) || 1024;
  if (want !== S.mainSize) { S.mainSize = want; plan(); }
  drawHover();
}

// ------------------------------------------------------------ drawing
function paint(canvas, url) {
  const im = imgs.get(url);
  if (!im) { request(url, true); return false; }
  if (canvas.width !== im.naturalWidth || canvas.height !== im.naturalHeight) {
    canvas.width = im.naturalWidth; canvas.height = im.naturalHeight;
  }
  canvas.getContext("2d").drawImage(im, 0, 0);
  return true;
}
function draw() {
  if (!S.clip) return;
  paint($("#main"), panelUrl(S.main, S.mainSize, S.t));
  document.querySelectorAll("#rail canvas").forEach(c => paint(c, panelUrl(c.dataset.panel, THUMB, S.t)));
  const [title] = titleOf(S.main);
  $("#viewlabel").textContent = S.main === "rgb" && S.layers.length
    ? "RGB + " + S.layers.map(l => LAYER_NAME[l].toLowerCase()).join(", ") : title;
  const tl = S.clip.timeline, live = $("#live");
  const active = S.clip.violation && tl.active[S.t], occ = S.clip.violation && tl.occluded[S.t];
  live.classList.toggle("on", !!active);
  live.classList.toggle("occ", !!(active && occ));
  live.textContent = active ? (occ ? "violation, hidden" : (tl.observable[S.t] ? "violation, visible" : "violation")) : "";
  $("#hint").style.display = S.mouse || S.pinned ? "none" : "";
  clock(); drawTimeline(); drawHover();
  if (S.mouse) hover(); else if (S.pinned) segFor(S.t).then(drawHover);
  if (S.tab === "object") objectCard(false);
}
function clock() {
  const c = S.clip, T = c.frames;
  $("#clock").innerHTML = `<b>${S.t}</b> / ${T - 1} &nbsp;&middot;&nbsp; ${(S.t / c.fps).toFixed(2)} s / ${((T - 1) / c.fps).toFixed(2)} s &nbsp;&middot;&nbsp; ${c.fps} fps`;
}

const TL = {x0: 84, pad: 12};
function drawTimeline() {
  const c = $("#timeline"), cl = S.clip;
  if (!cl) return;
  const invalid = !!cl.violation;
  const violators = cl.objects.filter(o => o.violator);
  const H = invalid ? 30 + 3 * 14 + 30 : 52;
  if (c.style.height !== H + "px") c.style.height = H + "px";
  const dpr = window.devicePixelRatio || 1, W = c.clientWidth;
  if (c.width !== Math.round(W * dpr) || c.height !== Math.round(H * dpr)) { c.width = Math.round(W * dpr); c.height = Math.round(H * dpr); }
  const g = c.getContext("2d");
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, W, H);
  const T = cl.frames, x0 = TL.x0, x1 = W - TL.pad, fw = (x1 - x0) / T, fx = f => x0 + f * fw;
  g.font = "11px ui-sans-serif, system-ui, sans-serif";
  g.textBaseline = "middle";
  const every = [1, 2, 4, 5, 10, 20, 50].find(e => e * fw >= 26) || 100;
  g.fillStyle = "#6c7281";
  for (let f = 0; f < T; f += every) { g.fillText(String(f), fx(f) + 2, 10); g.fillRect(fx(f), 16, 1, 3); }
  let y = 22;
  if (invalid) {
    const rows = [["violation", cl.timeline.active, "#ff5a5a"], ["observable", cl.timeline.observable, "#f5b83d"],
                  ["occluded", cl.timeline.occluded, "#8a90a0"]];
    for (const [name, arr, col] of rows) {
      g.fillStyle = "#8d93a1"; g.fillText(name, 10, y + 5);
      g.fillStyle = "#1f232c"; g.fillRect(x0, y, x1 - x0, 10);
      g.fillStyle = col;
      for (let f = 0; f < T; f++) if (arr[f]) g.fillRect(fx(f), y, Math.ceil(fw) + 0.5, 10);
      y += 14;
    }
    const h = 24;
    g.fillStyle = "#8d93a1"; g.fillText("severity", 10, y + h / 2);
    g.fillStyle = "#181b22"; g.fillRect(x0, y, x1 - x0, h);
    for (const o of violators) {
      g.strokeStyle = o.colour; g.lineWidth = 1.6; g.beginPath();
      o.severity.forEach((s, f) => { const X = fx(f + 0.5), Y = y + h - 2 - Math.max(0, Math.min(1, s)) * (h - 4); f ? g.lineTo(X, Y) : g.moveTo(X, Y); });
      g.stroke();
    }
    const v = cl.violation;
    for (const [f, col] of [[v.t_event_frame, "#ff8080"], [v.t_observable_frame, "#f5b83d"], [v.t_end_frame, "#b4b4e6"]]) {
      if (f == null || f < 0 || f >= T) continue;
      g.strokeStyle = col; g.lineWidth = 1; g.setLineDash([3, 3]);
      g.beginPath(); g.moveTo(fx(f) + 0.5, 18); g.lineTo(fx(f) + 0.5, H - 6); g.stroke();
      g.setLineDash([]);
    }
  } else {
    g.fillStyle = "#6c7281"; g.fillText("valid twin: nothing is violated in this clip", x0, y + 10);
  }
  // Frames of the main view already loaded.
  g.fillStyle = "#232730"; g.fillRect(x0, H - 3, x1 - x0, 2);
  g.fillStyle = "#4a5366";
  for (let f = 0; f < T; f++) if (imgs.has(panelUrl(S.main, S.mainSize, f))) g.fillRect(fx(f), H - 3, Math.ceil(fw), 2);
  // Playhead.
  g.fillStyle = "rgba(255,255,255,.07)"; g.fillRect(fx(S.t), 16, Math.max(fw, 2), H - 20);
  g.fillStyle = "#ffffff"; g.fillRect(fx(S.t + 0.5) - 1, 14, 2, H - 16);
}
function legend() {
  const v = S.clip.violation, fps = S.clip.fps, box = $("#legend");
  if (!v) { box.innerHTML = ""; return; }
  const at = f => f == null || f < 0 ? "-" : `frame ${f} (${(f / fps).toFixed(2)} s)`;
  const vs = S.clip.objects.filter(o => o.violator);
  box.innerHTML =
    `<span><i style="background:#ff8080"></i>event <b>${at(v.t_event_frame)}</b></span>` +
    `<span><i style="background:#f5b83d"></i>observable <b>${at(v.t_observable_frame)}</b></span>` +
    `<span><i style="background:#b4b4e6"></i>end <b>${at(v.t_end_frame)}</b></span>` +
    vs.slice(0, 6).map(o => `<span><i style="background:${o.colour}"></i>${esc(o.name)} <b>#${o.id}</b></span>`).join("") +
    (vs.length > 6 ? `<span>+${vs.length - 6} violators</span>` : "");
}

// ------------------------------------------------------------ playback
function setT(t) {
  const T = S.clip.frames;
  S.t = Math.max(0, Math.min(T - 1, t));
  draw(); saveHash();
}
function play(on = !S.playing) {
  S.playing = on;
  $("#play").innerHTML = on ? ICON_PAUSE : ICON_PLAY;
  if (on) {
    if (S.t >= S.clip.frames - 1 && !S.loop) S.t = 0;
    S.last = performance.now(); S.acc = 0;
    requestAnimationFrame(tick);
  }
}
function tick(now) {
  if (!S.playing || !S.clip) return;
  const dt = Math.min(0.25, (now - S.last) / 1000);
  S.last = now;
  S.acc += dt * S.clip.fps * S.speed;
  if (S.acc >= 1) {
    const T = S.clip.frames;
    let next = S.t + Math.floor(S.acc);
    if (next >= T) {
      if (!S.loop) { setT(T - 1); play(false); return; }
      next %= T;
    }
    if (imgs.has(panelUrl(S.main, S.mainSize, next))) { S.acc -= Math.floor(S.acc); setT(next); }
    else { S.acc = 1; request(panelUrl(S.main, S.mainSize, next), true); }
  }
  requestAnimationFrame(tick);
}
function buildSpeed() {
  const box = $("#speed");
  for (const s of SPEEDS) {
    const b = document.createElement("button");
    b.textContent = s + "×";
    b.onclick = () => setSpeed(s);
    box.append(b);
  }
  setSpeed(1);
}
function setSpeed(s) {
  S.speed = s;
  [...$("#speed").children].forEach((b, k) => b.classList.toggle("on", SPEEDS[k] === s));
}

// ------------------------------------------------------------ hover
function segFor(t) {
  const key = `${S.cur}:${t}`;
  if (S.seg.has(key)) return S.seg.get(key);
  const p = fetch(`api/seg?v=${S.index.token}&i=${S.cur}&t=${t}`).then(async r => {
    const W = +r.headers.get("X-W"), H = +r.headers.get("X-H"), B = +r.headers.get("X-Bytes");
    const buf = await r.arrayBuffer();
    const seg = {W, H, ids: B === 1 ? new Uint8Array(buf) : new Uint16Array(buf)};
    S.segDone.set(key, seg);
    return seg;
  });
  S.seg.set(key, p);
  if (S.seg.size > 600) { const k = S.seg.keys().next().value; S.seg.delete(k); S.segDone.delete(k); }
  return p;
}
async function hover() {
  if (!S.mouse || !S.clip || S.main === "camera") { tip(0); return; }
  const t = S.t, i = S.cur;
  const seg = await segFor(t);
  if (!S.mouse || t !== S.t || i !== S.cur) return;
  const x = Math.min(seg.W - 1, Math.max(0, Math.floor(S.mouse.u * seg.W)));
  const y = Math.min(seg.H - 1, Math.max(0, Math.floor(S.mouse.v * seg.H)));
  const id = seg.ids[y * seg.W + x];
  if (id !== S.hoverId) { S.hoverId = id; drawHover(); }
  tip(id);
}
function objectRows(o, t) {
  const sp = o.vel ? Math.hypot(...o.vel[t]) : null;
  const rows = [["role", o.role], ["category", o.category], ["material", o.material],
    ["mass", o.mass != null ? fmt(o.mass, 3) + " kg" : null], ["speed", sp != null ? fmt(sp) + " m/s" : null],
    ["energy", o.energy ? fmt(o.energy[t], 2) + " J" : null],
    ["height", o.pos ? fmt(o.pos[t][2]) + " m" : null], ["visible", o.vis ? (o.vis[t] ? o.vis[t] + " px" : "hidden") : null]];
  if (o.violator) rows.push(["severity", fmt(o.severity[t])], ["clock", o.active[t] ? (o.occluded[t] ? "active, hidden" : o.observable[t] ? "active, visible" : "active") : "inactive"]);
  if (o.difficulty) rows.push(["difficulty", o.difficulty.level + " (" + (o.difficulty.binding_factors || []).map(pretty).join(", ") + ")"]);
  return rows.filter(([, v]) => v != null && v !== "");
}
function tip(id) {
  const box = $("#tip");
  const o = id && S.clip.byId.get(id);
  if (!o || !S.mouse) { box.style.display = "none"; return; }
  box.innerHTML = `<h4><span class="swatch" style="background:${o.colour}"></span>${esc(pretty(o.name))}` +
    `<span class="badge">#${o.id}</span>${o.violator ? '<span class="badge invalid">violator</span>' : ""}</h4>` +
    `<div class="kv">${objectRows(o, S.t).map(([k, v]) => `<span>${k}</span><span>${esc(v)}</span>`).join("")}</div>` +
    `<div class="foot">${S.pinned === id ? "pinned &middot; esc to release" : "click to pin"}</div>`;
  box.style.display = "block";
  const r = box.getBoundingClientRect();
  let x = S.mouse.x + 16, y = S.mouse.y + 16;
  if (x + r.width > innerWidth - 8) x = S.mouse.x - r.width - 16;
  if (y + r.height > innerHeight - 8) y = S.mouse.y - r.height - 16;
  box.style.left = x + "px"; box.style.top = y + "px";
}
const OUT = document.createElement("canvas");
function drawHover() {
  const c = $("#hover"), g = c.getContext("2d");
  g.clearRect(0, 0, c.width, c.height);
  if (!S.clip || S.main === "camera") return;
  const seg = S.segDone.get(`${S.cur}:${S.t}`);
  if (!seg) return;
  const ids = [...new Set([S.pinned, S.hoverId])].filter(Boolean);
  if (!ids.length) return;
  OUT.width = seg.W; OUT.height = seg.H;
  const o = OUT.getContext("2d"), im = o.createImageData(seg.W, seg.H), px = im.data, a = seg.ids, W = seg.W, H = seg.H;
  for (const id of ids) {
    const col = (S.clip.byId.get(id)?.colour || "#ffffff").match(/\w\w/g).map(h => parseInt(h, 16));
    const fill = id === S.pinned ? 70 : 45;
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      const p = y * W + x;
      if (a[p] !== id) continue;
      const edge = x === 0 || y === 0 || x === W - 1 || y === H - 1 || a[p - 1] !== id || a[p + 1] !== id || a[p - W] !== id || a[p + W] !== id;
      const q = p * 4;
      px[q] = edge ? 255 : col[0]; px[q + 1] = edge ? 255 : col[1]; px[q + 2] = edge ? 255 : col[2]; px[q + 3] = edge ? 235 : fill;
    }
  }
  o.putImageData(im, 0, 0);
  g.imageSmoothingEnabled = false;
  g.drawImage(OUT, 0, 0, c.width, c.height);
}

// ------------------------------------------------------------ inspector
function showTab(name) {
  S.tab = name;
  document.querySelectorAll(".tabs button").forEach(b => b.classList.toggle("on", b.dataset.tab === name));
  document.querySelectorAll(".pane").forEach(p => p.classList.toggle("on", p.id === "pane-" + name));
  if (name === "object") objectCard(true);
}
function inspector() { viewPane(); clipPane(); objectCard(true); }

function viewPane() {
  const P = $("#pane-view");
  const panels = Object.keys(S.index.panels);
  let h = `<div class="group"><h5>Overlays on the video</h5><div class="chips presets">` +
    PRESETS.map(([n, ls]) => `<button class="chip${same(ls, S.layers) ? " on" : ""}" data-preset="${n}">${n}</button>`).join("") + `</div>`;
  if (S.main !== "rgb") h += `<div class="hint">Overlays draw on the RGB view, which is in the side rail now.</div>`;
  h += `</div>`;
  for (const [group, names] of LAYER_GROUPS) {
    h += `<div class="group"><h5>${group}</h5>` + names.filter(n => n in S.index.layers).map(n =>
      `<div class="toggle${S.layers.includes(n) ? " on" : ""}" data-layer="${n}"><span class="swatch" style="background:${LAYER_SWATCH[n]}"></span>` +
      `<span class="name">${LAYER_NAME[n] || n}<span class="desc">${esc(S.index.layers[n])}</span></span><span class="switch"></span></div>`).join("") + `</div>`;
  }
  h += `<div class="group sidechips"><h5>Panels beside the video</h5><div class="chips">` +
    panels.filter(p => p !== S.main).map(p => `<button class="chip${S.panels.includes(p) ? " on" : ""}" data-side="${p}" title="${esc(S.index.panels[p])}">${esc(titleOf(p)[0])}</button>`).join("") +
    `</div><div class="hint">Click a panel in the rail to show it large.</div></div>`;
  h += `<div class="group"><h5>Large view</h5><div class="chips">` +
    panels.map(p => `<button class="chip${p === S.main ? " main" : ""}" data-main="${p}">${esc(titleOf(p)[0])}</button>`).join("") + `</div></div>`;
  h += `<div class="group"><h5>Resolution</h5><div class="seg">` +
    QUALITY.map(([n, q]) => `<button class="${S.quality === q ? "on" : ""}" data-quality="${q}">${n}</button>`).join("") +
    `</div><div class="hint">Auto matches the screen; now ${S.mainSize}px.</div></div>`;
  P.innerHTML = h;
  P.querySelectorAll("[data-preset]").forEach(b => b.onclick = () => setLayers(PRESETS.find(p => p[0] === b.dataset.preset)[1]));
  P.querySelectorAll("[data-layer]").forEach(b => b.onclick = () => {
    const n = b.dataset.layer;
    setLayers(S.layers.includes(n) ? S.layers.filter(x => x !== n) : orderLayers([...S.layers, n]));
  });
  P.querySelectorAll("[data-side]").forEach(b => b.onclick = () => {
    const n = b.dataset.side;
    S.panels = S.panels.includes(n) ? S.panels.filter(x => x !== n) : [...S.panels, n];
    rail(); plan(); viewPane(); draw(); saveHash();
  });
  P.querySelectorAll("[data-main]").forEach(b => b.onclick = () => setMain(b.dataset.main));
  P.querySelectorAll("[data-quality]").forEach(b => b.onclick = () => { S.quality = +b.dataset.quality; S.mainSize = 0; layout(); viewPane(); });
}
const same = (a, b) => a.length === b.length && a.every(x => b.includes(x));
const orderLayers = ls => Object.keys(S.index.layers).filter(n => ls.includes(n));
function setLayers(ls) {
  S.layers = orderLayers(ls);
  if (S.main !== "rgb" && ls.length) setMain("rgb");
  plan(); viewPane(); draw(); saveHash();
}

function clipPane() {
  const c = S.clip, v = c.violation, d = c.difficulty, fps = c.fps, P = $("#pane-clip");
  const at = f => f == null || f < 0 ? "-" : `${f}  (${(f / fps).toFixed(2)} s)`;
  let h = `<div class="card"><h4>Sample <span class="badge ${c.label}">${c.label}</span></h4><div class="kv">` +
    [["scenario", pretty(c.scenario)], ["violation family", pretty(c.family)], ["level", c.level], ["condition", c.condition], ["medium", c.medium], ["domain", c.domain],
     ["severity bin", c.severity_bin], ["magnitude", c.magnitude != null ? fmt(c.magnitude, 3) : null],
     ["frames", `${c.frames} at ${fps} fps`], ["resolution", (c.resolution || []).join(" x ")]]
      .filter(([, x]) => x != null && x !== "").map(([k, x]) => `<span>${k}</span><span>${esc(x)}</span>`).join("") +
    `</div>${c.prompt ? `<div class="hint" style="margin-top:8px">&ldquo;${esc(c.prompt)}&rdquo;</div>` : ""}<div class="uid">${esc(c.uid)}</div></div>`;
  if (v) {
    const pr = v.peak_residual || {};
    h += `<div class="card"><h4>Violation</h4><div class="kv">` +
      [["event", at(v.t_event_frame)], ["observable", at(v.t_observable_frame)], ["end", at(v.t_end_frame)],
       ["lag", v.observability_lag_frames != null ? `${v.observability_lag_frames} frames` : null],
       ["timing", v.violator_timing], ["law", pr.law], ["peak score", pr.score != null ? fmt(pr.score, 3) : null]]
        .filter(([, x]) => x != null).map(([k, x]) => `<span>${k}</span><span>${esc(x)}</span>`).join("") +
      `</div><div class="objlist" style="margin-top:8px">` +
      c.objects.filter(o => o.violator).map(o =>
        `<div class="obj" data-obj="${o.id}"><span class="swatch" style="background:${o.colour}"></span>${esc(pretty(o.name))} <span class="badge">#${o.id}</span>` +
        `<span class="role">peak ${fmt(Math.max(...o.severity))}</span></div>`).join("") + `</div></div>`;
  }
  if (d) {
    const bind = new Set(d.binding_factors || []);
    h += `<div class="card"><h4>Detection difficulty <span class="lvl ${d.level}">${d.level}</span></h4>` +
      `<table class="factors">` + Object.entries(d.factors || {}).map(([k, f]) =>
        `<tr class="${bind.has(k) ? "bind" : ""}"><td>${esc(pretty(k))}</td><td>${f.value == null ? "-" : fmt(f.value, 3)}</td><td><span class="lvl ${f.level}">${f.level}</span></td></tr>`).join("") +
      `</table><div class="hint">The label is the worst factor; bold factors set it.</div></div>`;
  }
  P.innerHTML = h;
  P.querySelectorAll("[data-obj]").forEach(b => b.onclick = () => pin(+b.dataset.obj));
}

function pin(id) {
  S.pinned = S.pinned === id ? 0 : id;
  if (S.pinned) { showTab("object"); setRight(true); }
  segFor(S.t).then(drawHover);
  objectCard(true); draw();
}
function objectCard(rebuild) {
  const P = $("#pane-object"), c = S.clip;
  if (!c) return;
  const id = S.pinned || S.hoverId, o = id && c.byId.get(id);
  if (rebuild || P.dataset.id !== String(id || 0)) {
    P.dataset.id = String(id || 0);
    let h = "";
    if (o) {
      h += `<div class="card"><h4><span class="swatch" style="background:${o.colour}"></span>${esc(pretty(o.name))} <span class="badge">#${o.id}</span>` +
        (o.violator ? `<span class="badge invalid">violator</span>` : "") +
        (o.difficulty ? `<span class="badge ${o.difficulty.level}" title="this object's own detection difficulty">${o.difficulty.level}</span>` : "") +
        `</h4><div class="kv" id="objkv"></div>` +
        `<div class="sparklab"><span>speed, m/s</span><span id="spmax"></span></div><canvas class="spark" id="spSpeed"></canvas>` +
        (o.energy ? `<div class="sparklab"><span>energy, J</span><span id="enmax"></span></div><canvas class="spark" id="spEnergy"></canvas>` : "") +
        (o.violator ? `<div class="sparklab"><span>severity</span><span>0 - 1</span></div><canvas class="spark" id="spSev"></canvas>` : "") +
        `<div class="kv" style="margin-top:10px">` +
        [["asset", o.asset], ["friction", o.friction != null ? fmt(o.friction) : null], ["restitution", o.restitution != null ? fmt(o.restitution) : null],
         ["static", o.static != null ? String(o.static) : null]].filter(([, x]) => x != null)
          .map(([k, x]) => `<span>${k}</span><span>${esc(x)}</span>`).join("") + `</div>` +
        (S.pinned ? `<div class="hint"><button class="link" id="unpin">Unpin</button> &middot; or press esc</div>` : `<div class="hint">Click the object to pin it here.</div>`) + `</div>`;
    } else {
      h += `<div class="hint" style="margin-top:12px">Hover the video to read an object; click it to pin it here.</div>`;
    }
    const selected = c.objects.filter(x => S.objectGroup === "all" ||
      (S.objectGroup === "subjects" && (x.analysis_group === "subject" || x.violator)) ||
      x.analysis_group === S.objectGroup);
    const order = [...selected].sort((a, b) => (b.violator - a.violator) || a.id - b.id);
    h += `<div class="group"><h5>Objects (${order.length}) <select id="objectGroup">` +
      [["subjects","Subjects + violators"],["context","Context"],["support","Support"],["background","Background"],["all","All"]]
        .map(([v,n]) => `<option value="${v}"${S.objectGroup === v ? " selected" : ""}>${n}</option>`).join("") +
      `</select></h5><div class="objlist">` +
      order.slice(0, 300).map(x => `<div class="obj${x.id === S.pinned ? " on" : ""}" data-obj="${x.id}"><span class="swatch" style="background:${x.colour}"></span>` +
        `${esc(pretty(x.name))} <span class="badge">#${x.id}</span>` +
        (x.difficulty ? `<span class="badge ${x.difficulty.level}">${x.difficulty.level}</span>` : "") +
        `<span class="role">${x.violator ? "violator" : esc(x.role || "")}</span></div>`).join("") + `</div></div>`;
    P.innerHTML = h;
    P.querySelectorAll("[data-obj]").forEach(b => b.onclick = () => pin(+b.dataset.obj));
    $("#objectGroup")?.addEventListener("change", e => { S.objectGroup = e.target.value; objectCard(true); });
    $("#unpin")?.addEventListener("click", () => pin(S.pinned));
    P.querySelectorAll("[data-obj]").forEach(b => {
      b.onmouseenter = () => { if (!S.pinned) { S.hoverId = +b.dataset.obj; segFor(S.t).then(drawHover); } };
    });
  }
  if (!o) return;
  const kv = $("#objkv");
  if (kv) kv.innerHTML = objectRows(o, S.t).map(([k, v]) => `<span>${k}</span><span>${esc(v)}</span>`).join("");
  const speed = o.vel.map(v => Math.hypot(...v)), top = Math.max(1e-6, ...speed);
  $("#spmax").textContent = "max " + fmt(top);
  spark($("#spSpeed"), speed.map(s => s / top), o.colour);
  if (o.energy) {
    const hi = Math.max(1e-9, ...o.energy.map(Math.abs));
    $("#enmax").textContent = "max " + fmt(hi);
    spark($("#spEnergy"), o.energy.map(e => e / hi), "#8fe3a0");
  }
  if (o.violator) spark($("#spSev"), o.severity, "#f7931e", o.active);
}
function spark(c, ys, colour, band) {
  if (!c) return;
  const dpr = window.devicePixelRatio || 1, W = c.clientWidth, H = c.clientHeight;
  c.width = Math.round(W * dpr); c.height = Math.round(H * dpr);
  const g = c.getContext("2d"), T = ys.length, fx = f => (f + 0.5) / T * W;
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.fillStyle = "#15181f"; g.fillRect(0, 0, W, H);
  if (band) { g.fillStyle = "rgba(255,90,90,.16)"; band.forEach((b, f) => b && g.fillRect(f / T * W, 0, W / T + 0.5, H)); }
  g.strokeStyle = colour; g.lineWidth = 1.6; g.beginPath();
  ys.forEach((y, f) => { const X = fx(f), Y = H - 3 - Math.max(0, Math.min(1, y)) * (H - 6); f ? g.lineTo(X, Y) : g.moveTo(X, Y); });
  g.stroke();
  g.fillStyle = "#fff"; g.fillRect(fx(S.t) - 0.75, 0, 1.5, H);
}

// ------------------------------------------------------------ wiring
function setLeft(on) { const small = matchMedia("(max-width:860px)").matches; document.body.classList.toggle(small ? "show-left" : "no-left", small ? on : !on); setTimeout(layout, 200); }
function setRight(on) { const small = matchMedia("(max-width:1180px)").matches; document.body.classList.toggle(small ? "show-right" : "no-right", small ? on : !on); setTimeout(layout, 200); }
const leftOn = () => matchMedia("(max-width:860px)").matches ? document.body.classList.contains("show-left") : !document.body.classList.contains("no-left");
const rightOn = () => matchMedia("(max-width:1180px)").matches ? document.body.classList.contains("show-right") : !document.body.classList.contains("no-right");

function wireResizer(id, side, min, max) {
  const handle = $(id);
  handle.onpointerdown = e => {
    if ((side === "left" && !leftOn()) || (side === "right" && !rightOn())) return;
    handle.setPointerCapture(e.pointerId);
    handle.classList.add("dragging");
    const move = ev => {
      const width = side === "left" ? ev.clientX : innerWidth - ev.clientX;
      document.body.style.setProperty("--" + side, Math.max(min, Math.min(max, width)) + "px");
      layout();
    };
    handle.onpointermove = move;
    handle.onpointerup = () => {
      handle.releasePointerCapture(e.pointerId);
      handle.onpointermove = null;
      handle.classList.remove("dragging");
    };
  };
}

function wire() {
  $("#play").innerHTML = ICON_PLAY;
  $("#play").onclick = () => play();
  $("#prev").onclick = () => { play(false); setT(S.t - 1); };
  $("#next").onclick = () => { play(false); setT(S.t + 1); };
  $("#loop").onclick = () => { S.loop = !S.loop; $("#loop").classList.toggle("on", S.loop); };
  $("#toggleLeft").onclick = () => setLeft(!leftOn());
  $("#toggleRight").onclick = () => setRight(!rightOn());
  wireResizer("#resizeLeft", "left", 190, 480);
  wireResizer("#resizeRight", "right", 240, 480);
  $("#helpBtn").onclick = () => $("#help").classList.add("on");
  $("#help").onclick = () => $("#help").classList.remove("on");
  document.querySelectorAll(".tabs button").forEach(b => b.onclick = () => showTab(b.dataset.tab));

  const tl = $("#timeline");
  const scrub = e => {
    const r = tl.getBoundingClientRect(), fw = (r.width - TL.x0 - TL.pad) / S.clip.frames;
    setT(Math.floor((e.clientX - r.left - TL.x0) / fw));
  };
  tl.onpointerdown = e => { if (!S.clip) return; tl.setPointerCapture(e.pointerId); play(false); scrub(e); tl.onpointermove = scrub; };
  tl.onpointerup = e => { tl.releasePointerCapture(e.pointerId); tl.onpointermove = null; };

  const main = $("#main");
  main.onmousemove = e => {
    const r = main.getBoundingClientRect();
    S.mouse = {u: (e.clientX - r.left) / r.width, v: (e.clientY - r.top) / r.height, x: e.clientX, y: e.clientY};
    $("#hint").style.display = "none";
    hover();
  };
  main.onmouseleave = () => { S.mouse = null; S.hoverId = 0; tip(0); drawHover(); if (S.tab === "object") objectCard(false); };
  main.onclick = () => { if (S.hoverId) pin(S.hoverId); else if (S.pinned) pin(S.pinned); };

  new ResizeObserver(() => { layout(); drawTimeline(); }).observe($("#view"));

  document.addEventListener("keydown", e => {
    if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") {
      if (e.key === "Escape") e.target.blur();
      return;
    }
    if (e.metaKey || e.ctrlKey || e.altKey || !S.clip) return;
    const k = e.key;
    const moves = {ArrowRight: 1, ArrowLeft: -1};
    if (k === " ") { e.preventDefault(); play(); }
    else if (k in moves) { e.preventDefault(); play(false); setT(S.t + moves[k] * (e.shiftKey ? 10 : 1)); }
    else if (k === "Home") setT(0);
    else if (k === "End") setT(S.clip.frames - 1);
    else if (k === "ArrowDown" || k === "ArrowUp" || k === "j" || k === "k") {
      e.preventDefault();
      const at = S.shown.indexOf(S.cur), d = k === "ArrowDown" || k === "j" ? 1 : -1;
      const next = S.shown[Math.max(0, Math.min(S.shown.length - 1, (at < 0 ? 0 : at + d)))];
      if (next != null && next !== S.cur) openClip(next);
    }
    else if (k === "e" && S.clip.violation) { play(false); setT(Math.max(0, S.clip.violation.t_event_frame)); }
    else if (k === "t") {
      const me = S.index.samples[S.cur];
      if (me.label === "invalid") { const v = S.index.samples.find(x => x.pair_uid === me.pair_uid && x.label === "valid"); if (v) openClip(v.i, S.t); }
      else if (S.lastInvalid >= 0 && S.index.samples[S.lastInvalid].pair_uid === me.pair_uid) openClip(S.lastInvalid, S.t);
      else { const v = S.index.samples.find(x => x.pair_uid === me.pair_uid && x.label === "invalid"); if (v) openClip(v.i, S.t); }
    }
    else if (k === "l") $("#loop").click();
    else if (k === "m") setLayers(S.layers.includes("violation") ? S.layers.filter(x => x !== "violation") : [...S.layers, "violation"]);
    else if (k === "-" || k === "_") setSpeed(SPEEDS[Math.max(0, SPEEDS.indexOf(S.speed) - 1)]);
    else if (k === "+" || k === "=") setSpeed(SPEEDS[Math.min(SPEEDS.length - 1, SPEEDS.indexOf(S.speed) + 1)]);
    else if (k === "[") setLeft(!leftOn());
    else if (k === "]") setRight(!rightOn());
    else if (k === "/") { e.preventDefault(); setLeft(true); $("#q").focus(); }
    else if (k === "?") $("#help").classList.toggle("on");
    else if (k === "Escape") { $("#help").classList.remove("on"); if (S.pinned) pin(S.pinned); }
  });
}

boot().catch(err => { $("#title").textContent = "Viewer failed to start: " + err; });
</script></body></html>
"""
