"""A browser viewer over the overlay renderer: pick a clip, tick layers and
panels, scrub or play.

    python test_dataset_loader.py out/physloc_mini --gui     # http://localhost:8765

Standard library only. Frames are rendered on request by `overlay.Renderer` --
the same code that writes `overlay.mp4` -- and sent as JPEG from memory, so
nothing is written to disk. On a remote machine, forward the port (VS Code does
it when the URL is printed).
"""
from __future__ import annotations

import json
import threading
import urllib.parse
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, List

from .. import loader
from . import overlay

#: Renderers kept warm. Each holds its clip's decoded arrays, so this bounds memory.
KEEP = 6


def serve(root: str, port: int = 8765, host: str = "127.0.0.1") -> None:
    ds = loader.PhysLocDataset(root)
    if not len(ds):
        raise SystemExit("no clips under %s" % root)
    app = _App(ds)

    class Handler(BaseHTTPRequestHandler):
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
                elif url.path == "/api/clip":
                    self._json(app.clip(int(q["i"])))
                elif url.path == "/api/frame":
                    self._send(200, "image/jpeg", app.frame(
                        int(q["i"]), int(q.get("t", 0)), _split(q.get("layers")),
                        _split(q.get("panels")), int(q.get("size", overlay.PANEL))))
                else:
                    self._send(404, "text/plain", b"not found")
            except Exception as exc:                          # noqa: BLE001
                self._send(500, "text/plain", repr(exc).encode())

        def _json(self, obj):
            self._send(200, "application/json", json.dumps(obj, default=str).encode())

        def _send(self, code, ctype, body):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer((host, port), Handler)
    print("PhysLoc viewer: %d clips from %s -> http://%s:%d  (Ctrl+C to stop)"
          % (len(ds), root, "localhost" if host == "127.0.0.1" else host, port),
          flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


def _split(s) -> List[str]:
    return [x for x in (s or "").split(",") if x]


class _App:
    def __init__(self, ds: "loader.PhysLocDataset"):
        self.ds = ds
        self.lock = threading.Lock()
        self.renderers: "OrderedDict[tuple, overlay.Renderer]" = OrderedDict()

    def index(self) -> Dict[str, object]:
        clips = [dict(self.ds.fields(i), i=i, uid=c.uid) for i, c in enumerate(self.ds.clips)]
        return {"clips": clips, "layers": overlay.LAYERS, "panels": overlay.PANELS,
                "default_panels": list(overlay.DEFAULT_PANELS)}

    def clip(self, i: int) -> Dict[str, object]:
        c = self.ds.clips[i]
        v = c.metadata.get("violation") or {}
        names = {int(x["id"]): x.get("name") for x in c.metadata.get("instances", [])}
        return {
            "uid": c.uid, "frames": c.num_frames, "fps": c.fps, "label": c.label,
            "prompt": c.prompt,
            "violation": {k: v.get(k) for k in (
                "t_event_frame", "t_observable_frame", "t_end_frame",
                "violation_windows", "violator_timing")} if v else None,
            "violators": [{"id": o["instance_id"], "name": names.get(int(o["instance_id"])),
                           "t_event": o.get("t_event_frame"),
                           "peak_severity": o.get("peak_severity")}
                          for o in v.get("violators", [])],
        }

    def frame(self, i: int, t: int, layers, panels, size: int) -> bytes:
        import cv2
        key = (i, tuple(layers), tuple(panels), size)
        with self.lock:
            r = self.renderers.get(key)
            if r is None:
                r = overlay.Renderer(self.ds.clips[i], layers, panels or ("rgb",), size)
                self.renderers[key] = r
                while len(self.renderers) > KEEP:
                    _, old = self.renderers.popitem(last=False)
                    if all(x.clip is not old.clip for x in self.renderers.values()):
                        old.clip.release()
            self.renderers.move_to_end(key)
            img = r.frame(max(0, min(t, r.T - 1)))
        ok, buf = cv2.imencode(".jpg", img[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 92])
        return buf.tobytes()


PAGE = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>PhysLoc viewer</title>
<style>
:root{--bg:#121216;--panel:#1b1b21;--line:#2c2c35;--text:#e8e8ee;--dim:#9696a0}
*{box-sizing:border-box}
body{margin:0;font:13px/1.4 system-ui,sans-serif;background:var(--bg);color:var(--text);
     display:grid;grid-template-columns:300px 1fr;height:100vh}
aside{border-right:1px solid var(--line);display:flex;flex-direction:column;min-height:0}
aside header{padding:12px 14px;border-bottom:1px solid var(--line)}
h1{font-size:15px;margin:0 0 10px}
.filters{display:grid;grid-template-columns:1fr 1fr;gap:6px}
select,button{background:var(--panel);color:var(--text);border:1px solid var(--line);
              border-radius:6px;padding:4px 6px;font:inherit}
button{cursor:pointer;min-width:38px}
#clips{overflow:auto;flex:1;margin:0;padding:6px;list-style:none}
#clips li{padding:6px 8px;border-radius:6px;cursor:pointer}
#clips li:hover{background:var(--panel)} #clips li.on{background:#2a2a36}
.tag{display:block;font-size:11px;color:var(--dim)} .inv{color:#ff8a8a} .val{color:#8fe3a0}
main{display:flex;flex-direction:column;min-width:0;min-height:0}
.bar{display:flex;flex-wrap:wrap;gap:10px;align-items:flex-start;padding:10px 14px;
     border-bottom:1px solid var(--line)}
fieldset{border:1px solid var(--line);border-radius:8px;padding:4px 10px 6px;margin:0;
         display:flex;flex-wrap:wrap;gap:3px 12px;max-width:640px}
legend{color:var(--dim);font-size:11px;padding:0 4px}
label{white-space:nowrap;cursor:pointer}
.stage{flex:1;overflow:auto;padding:14px;display:flex;justify-content:center;align-items:flex-start}
#frame{max-width:100%;border-radius:6px;box-shadow:0 0 0 1px var(--line)}
.player{display:flex;gap:10px;align-items:center;padding:8px 14px;border-top:1px solid var(--line)}
#t{flex:1} #tl{color:var(--dim);min-width:70px;text-align:right}
#info{padding:8px 14px;border-top:1px solid var(--line);color:var(--dim);font-size:12px;
      max-height:110px;overflow:auto;white-space:pre-wrap}
</style></head><body>
<aside><header><h1>PhysLoc viewer</h1><div class="filters" id="filters"></div></header>
<ul id="clips"></ul></aside>
<main>
  <div class="bar">
    <fieldset id="layers"><legend>layers, on the RGB panel</legend></fieldset>
    <fieldset id="panels"><legend>panels, side by side</legend></fieldset>
    <label>size <select id="size"><option>192</option><option>256</option>
      <option selected>288</option><option>384</option><option>512</option></select></label>
  </div>
  <div class="stage"><img id="frame" alt=""></div>
  <div class="player"><button id="play" title="space">&#9654;</button>
    <input id="t" type="range" min="0" value="0"><span id="tl"></span></div>
  <div id="info"></div>
</main>
<script>
const $ = s => document.querySelector(s);
const FILTERS = ["level", "scenario", "family", "label", "condition", "severity_bin"];
let index, clip = null, cur = -1, playing = false, busy = false, again = false;

async function boot() {
  index = await (await fetch("api/index")).json();
  for (const f of FILTERS) {
    const sel = document.createElement("select");
    sel.id = "f_" + f;
    const values = [...new Set(index.clips.map(c => c[f] ?? "-"))].sort();
    sel.innerHTML = `<option value="">all ${f.replace("_", " ")}</option>` +
                    values.map(v => `<option>${v}</option>`).join("");
    sel.onchange = list;
    $("#filters").append(sel);
  }
  checkboxes("#layers", index.layers, ["violation", "reference"]);
  checkboxes("#panels", index.panels, index.default_panels);
  $("#size").onchange = draw;
  $("#t").oninput = draw;
  $("#play").onclick = toggle;
  document.addEventListener("keydown", e => {
    if (e.key === " ") { e.preventDefault(); toggle(); }
    if (e.key === "ArrowRight") step(1);
    if (e.key === "ArrowLeft") step(-1);
  });
  list();
}

function checkboxes(sel, entries, on) {
  for (const [key, tip] of Object.entries(entries)) {
    const l = document.createElement("label");
    l.title = tip;
    l.innerHTML = `<input type="checkbox" value="${key}" ${on.includes(key) ? "checked" : ""}> ${key}`;
    l.querySelector("input").onchange = draw;
    $(sel).append(l);
  }
}

const picked = sel => [...document.querySelectorAll(sel + " input:checked")].map(i => i.value).join(",");

function list() {
  const ul = $("#clips");
  ul.innerHTML = "";
  const want = Object.fromEntries(FILTERS.map(f => [f, $("#f_" + f).value]));
  for (const c of index.clips) {
    if (FILTERS.some(f => want[f] && String(c[f] ?? "-") !== want[f])) continue;
    const li = document.createElement("li");
    if (c.i === cur) li.className = "on";
    const what = c.label === "valid" ? `<span class="val">valid</span>` : `<span class="inv">${c.family}</span>`;
    li.innerHTML = `${c.scenario} &middot; ${what}<span class="tag">${c.level} &middot; seed ${c.seed} &middot; ${c.condition}${c.severity_bin ? " &middot; " + c.severity_bin : ""}</span>`;
    li.onclick = () => openClip(c.i);
    ul.append(li);
  }
  if (cur < 0 && index.clips.length) openClip(index.clips[0].i);
}

async function openClip(i) {
  cur = i;
  clip = await (await fetch("api/clip?i=" + i)).json();
  $("#t").max = clip.frames - 1;
  $("#t").value = Math.min(+$("#t").value, clip.frames - 1);
  list();
  const v = clip.violation;
  $("#info").textContent = clip.uid + (clip.prompt ? "\n" + clip.prompt : "") + "\n" + (v
    ? `t_event ${v.t_event_frame}   t_obs ${v.t_observable_frame}   t_end ${v.t_end_frame}   windows ${JSON.stringify(v.violation_windows)}   timing ${v.violator_timing}\n` +
      "violators: " + clip.violators.map(o => `#${o.id} ${o.name} (t_event ${o.t_event}, peak severity ${(o.peak_severity ?? 0).toFixed(2)})`).join(", ")
    : "valid clip: nothing is violated");
  draw();
}

function draw() {
  if (cur < 0) return;
  $("#tl").textContent = `${$("#t").value} / ${clip.frames - 1}`;
  if (busy) { again = true; return; }
  busy = true;
  const img = $("#frame");
  img.onload = img.onerror = () => {
    busy = false;
    if (again) { again = false; draw(); }
    else if (playing) setTimeout(() => step(1, true), 1000 / clip.fps);
  };
  img.src = `api/frame?i=${cur}&t=${$("#t").value}&layers=${picked("#layers")}` +
            `&panels=${picked("#panels")}&size=${$("#size").value}`;
}

function step(d, loop) {
  const t = $("#t");
  let n = +t.value + d;
  if (n > +t.max) n = loop ? 0 : +t.max;
  if (n < 0) n = 0;
  t.value = n;
  draw();
}

function toggle() {
  playing = !playing;
  $("#play").innerHTML = playing ? "&#10074;&#10074;" : "&#9654;";
  if (playing) step(1, true);
}

boot();
</script></body></html>
"""
