"""Feasibility probe: can we bake a Mantaflow liquid inside the pinned image?

OUTCOME (recorded 2026-08-21): NO, not headlessly. Kept as evidence and as the
starting point for a future attempt.

  - Blender 2.93.4 in the image DOES ship Mantaflow: FluidModifier,
    FluidDomainSettings, FluidFlowSettings, bpy.ops.fluid.bake_* all exist.
  - But bpy.ops.fluid.bake_data() fails with
        NameError: name 'liquid_save_data_N' is not defined
    followed by a Manta::Error crash in fluidsolver.cpp. Elapsed time is ~0.02 s,
    i.e. nothing simulated.
  - The documented workaround (cache_type='REPLAY' + scene.frame_set() stepping,
    which is what this file now does) fails identically.
  - Root cause is the known Blender-in-background limitation: bake_data is a modal
    operator that expects the job system, absent under `blender -b`.

A future attempt should drive Mantaflow through its own Python bindings, or use a
newer Blender. Until then fluid is Phase 3 and `pour` is the v0 stand-in.
See docs/PLAN.md Part 2, "On fluid and deformables".

Run: bash docker/kubric.sh physloc/render/probe_fluid.py --res 32 --frames 25
"""
import argparse, json, time
import bpy

ap = argparse.ArgumentParser()
ap.add_argument("--res", type=int, default=32, help="domain resolution_max")
ap.add_argument("--frames", type=int, default=25)
ap.add_argument("--cache", type=str, default="/kubric/out/fluid_cache")
a = ap.parse_args()

scene = bpy.context.scene
scene.frame_start, scene.frame_end = 1, a.frames

# --- domain
bpy.ops.mesh.primitive_cube_add(size=2, location=(0, 0, 1))
dom = bpy.context.object
bpy.ops.object.modifier_add(type='FLUID')
dom.modifiers["Fluid"].fluid_type = 'DOMAIN'
ds = dom.modifiers["Fluid"].domain_settings
ds.domain_type = 'LIQUID'
ds.resolution_max = a.res
ds.cache_directory = a.cache
ds.use_mesh = True
ds.cache_type = 'REPLAY'      # REPLAY simulates on frame change; no job system needed

# --- inflow: a falling blob
bpy.ops.mesh.primitive_uv_sphere_add(radius=0.25, location=(0, 0, 1.5))
flow = bpy.context.object
bpy.ops.object.modifier_add(type='FLUID')
flow.modifiers["Fluid"].fluid_type = 'FLOW'
fs = flow.modifiers["Fluid"].flow_settings
fs.flow_type = 'LIQUID'
fs.flow_behavior = 'GEOMETRY'

t0 = time.perf_counter()
ok, err, stepped = True, None, 0
try:
    # REPLAY mode: stepping the frame forces Mantaflow to advance the solver.
    for f in range(scene.frame_start, scene.frame_end + 1):
        scene.frame_set(f)
        dg = bpy.context.evaluated_depsgraph_get()
        dom.evaluated_get(dg)
        stepped += 1
except Exception as e:                      # noqa: BLE001
    ok, err = False, repr(e)
t_bake = time.perf_counter() - t0

print("FLUID_PROBE " + json.dumps({
    "resolution_max": a.res, "frames": a.frames,
    "baked": ok, "error": err, "frames_stepped": stepped,
    "bake_seconds": round(t_bake, 2),
    "seconds_per_frame": round(t_bake / a.frames, 3),
}))
