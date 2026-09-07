"""Phase 0 smoke worker: prove the container renders, and measure throughput.

Deliberately uses only built-in primitives -- no AssetSource, so nothing is
downloaded from GCS and the timing measures render cost alone.

Run:  bash docker/kubric.sh physloc/render/worker_smoke.py --frames 4

Reports seconds per frame at the master clip settings (512 sq, Cycles) so the
CPU-vs-OptiX decision in docs/PLAN.md Part 6 rests on a number, not a guess.
"""
import argparse
import json
import logging
import time

import kubric as kb
from kubric.simulator import PyBullet
from kubric.renderer import Blender

logging.basicConfig(level="INFO")

p = argparse.ArgumentParser()
p.add_argument("--resolution", type=int, default=512)
p.add_argument("--frames", type=int, default=4, help="frames to render for timing")
p.add_argument("--spp", type=int, default=64, help="samples per pixel (MOVi uses 64)")
p.add_argument("--outdir", type=str, default="out/smoke")
args = p.parse_args()

scene = kb.Scene(
    resolution=(args.resolution, args.resolution),
    frame_start=1,
    frame_end=args.frames,
    frame_rate=24,
)
scratch = kb.as_path(args.outdir) / "scratch"
scratch.mkdir(parents=True, exist_ok=True)

simulator = PyBullet(scene, scratch)
renderer = Blender(scene, scratch, use_denoising=True, samples_per_pixel=args.spp)

# A falling ball on a floor: enough dynamics to exercise PyBullet, enough
# geometry to exercise Cycles shadows.
scene += kb.Cube(name="floor", scale=(10, 10, 0.1), position=(0, 0, -0.1),
                 static=True, restitution=0.5)
scene += kb.Sphere(name="ball", scale=1.0, position=(0, 0, 4.0), restitution=0.8)
scene += kb.DirectionalLight(name="sun", position=(-1, -0.5, 3),
                             look_at=(0, 0, 0), intensity=1.5)
scene += kb.PerspectiveCamera(name="camera", position=(6, -4, 4), look_at=(0, 0, 1))

t0 = time.perf_counter()
animation, collisions = simulator.run(frame_start=0, frame_end=args.frames + 1)
t_sim = time.perf_counter() - t0

t0 = time.perf_counter()
data_stack = renderer.render()
t_render = time.perf_counter() - t0

outdir = kb.as_path(args.outdir)
outdir.mkdir(parents=True, exist_ok=True)
kb.write_image_dict(data_stack, outdir)

per_frame = t_render / args.frames
report = {
    "resolution": args.resolution,
    "samples_per_pixel": args.spp,
    "frames_rendered": args.frames,
    "sim_seconds": round(t_sim, 2),
    "render_seconds": round(t_render, 2),
    "seconds_per_frame": round(per_frame, 2),
    "passes": sorted(data_stack.keys()),
    "collisions": len(collisions),
    # The number that actually decides the stack: 600 clips x 96 frames.
    "projected_hours_600_clips_96_frames": round(per_frame * 96 * 600 / 3600, 1),
}
print("SMOKE_REPORT " + json.dumps(report))
kb.write_json(filename=outdir / "smoke_report.json", data=report)
