"""Container-side worker: sample -> simulate -> inject -> render both twins.

Runs INSIDE the pinned Kubric image (python 3.9). Everything it writes is on the
host side of the seam: traj.npz, the raw render passes, and a plan.json.

    bash docker/kubric.sh physloc/render/worker.py \
        --scenario drop --seed 91731 --tier debug \
        --family solidity --severity medium --outdir out/phase0

Renders the valid and invalid twins through a *bit-identical* path: same scene
object, same renderer, same seed -- only the replayed keyframes differ.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zlib

sys.path.insert(0, "/kubric")

import numpy as np
import bpy
import kubric as kb
from kubric.renderer import Blender
from kubric.simulator import PyBullet

from physloc import injectors
from physloc import taxonomy
from physloc import scenarios
from physloc.render import stepper
from physloc.scenarios.base import SceneSpec, Tier
from physloc.sim.trajectory import Contacts, Trajectory, prefix_identical

PASSES = ("rgba", "segmentation", "depth", "forward_flow", "backward_flow",
          "normal", "object_coordinates")


# --------------------------------------------------------------------------
KUBASIC = "gs://kubric-public/assets/KuBasic/KuBasic.json"

#: KuBasic primitives usable as actors, beyond the two Kubric builds natively.
#: The full set also has `gear`, `torus_knot`, `sponge`, `spot`, `teapot` and
#: `suzanne`; these three are the ones whose physics is legible -- a viewer can
#: predict how a cylinder rolls or a cone topples, which is the premise of
#: asking whether the clip obeyed physics. A suzanne head tumbling is a fine
#: picture and a poor test.
KUBASIC_SHAPES = ("cylinder", "cone", "torus")
HDRI = "gs://kubric-public/assets/HDRI_haven/HDRI_haven.json"


def build_scene(spec: SceneSpec, scratch):
    scene = kb.Scene(
        resolution=(spec.tier.resolution, spec.tier.resolution),
        frame_start=0, frame_end=spec.tier.num_frames - 1,
        frame_rate=spec.tier.fps, step_rate=spec.tier.fps * 20,
        gravity=spec.gravity,
    )
    simulator = PyBullet(scene, scratch)
    renderer = Blender(scene, scratch, use_denoising=True,
                       samples_per_pixel=spec.tier.samples_per_pixel,
                       background_transparency=False)
    scene.background = kb.Color(*spec.background_color)
    scene.ambient_illumination = kb.Color(0.04, 0.04, 0.05)

    # --- complexity L1+: photographic environment lighting + a dome backdrop
    hdri_tex = None
    if spec.hdri_id:
        hdri_src = kb.AssetSource.from_manifest(HDRI)
        hdri_tex = hdri_src.create(asset_id=spec.hdri_id)
        renderer._set_ambient_light_hdri(hdri_tex.filename)

    objs = {}
    for b in spec.bodies:
        # Roughness and metallic come from the body, which gets them from its
        # material -- steel is smooth and metallic, rubber is rough and dark.
        # Scenery leaves the defaults.
        material = kb.PrincipledBSDFMaterial(color=kb.Color(*b.color),
                                             roughness=b.roughness,
                                             metallic=b.metallic)
        # `sim_static`, not `static`: a scripted body is pinned in the
        # simulator so it neither falls nor generates contacts, and its motion
        # arrives from the trajectory instead. Downstream it is still dynamic.
        common = dict(name=b.name, position=b.position, quaternion=b.quaternion,
                      static=b.sim_static, mass=b.mass, friction=b.friction,
                      restitution=b.restitution, material=material,
                      segmentation_id=b.segmentation_id)
        if b.kind == "sphere":
            obj = kb.Sphere(scale=b.scale, **common)
        elif b.kind == "cube":
            obj = kb.Cube(scale=b.scale, **common)
        elif b.kind in KUBASIC_SHAPES:
            # A KuBasic mesh, with a collision mesh Kubric bakes for it, so
            # these simulate as themselves rather than as a bounding box.
            #
            # UNIFORM SCALE ONLY: `kubric/simulator/pybullet.py` asserts
            # `scale[1] == scale[2] == scale[0]` for every file-based object,
            # with the message "Pybullet does not support non-uniform scaling".
            # `_common.vary_dims` therefore leaves everything but a box alone,
            # and this takes `scale[0]` rather than quietly using a mean.
            kubasic = kb.AssetSource.from_manifest(KUBASIC)
            obj = kubasic.create(asset_id=b.kind, name=b.name,
                                 scale=float(b.scale[0]),
                                 position=b.position, quaternion=b.quaternion,
                                 static=b.sim_static, mass=b.mass,
                                 friction=b.friction, restitution=b.restitution)
            obj.material = material
            obj.segmentation_id = b.segmentation_id
        elif b.kind == "dome":
            # KuBasic's dome is both the ground plane and the backdrop the HDRI
            # is projected onto -- the idiom movi_def_worker.py uses.
            kubasic = kb.AssetSource.from_manifest(KUBASIC)
            obj = kubasic.create(asset_id="dome", name=b.name, friction=b.friction,
                                 restitution=b.restitution, static=True,
                                 background=True)
            obj.segmentation_id = b.segmentation_id
            scene += obj
            if hdri_tex is not None:
                dome_blender = obj.linked_objects[renderer]
                node = dome_blender.data.materials[0].node_tree.nodes["Image Texture"]
                node.image = bpy.data.images.load(hdri_tex.filename)
            _set_visibility(renderer, obj, b)
            objs[b.name] = obj
            continue
        else:
            raise ValueError("unknown body kind %r" % b.kind)
        if not b.sim_static:
            obj.velocity = b.velocity
            obj.angular_velocity = b.angular_velocity
        scene += obj
        if b.render_scale is not None:
            # After joining the scene, so PyBullet has already validated the
            # uniform collision scale it was given. Only the renderer observes
            # `scale`, so this changes what is drawn and nothing else.
            obj.scale = tuple(float(x) for x in b.render_scale)
        _set_visibility(renderer, obj, b)
        objs[b.name] = obj

    for l in spec.lights:
        scene += kb.DirectionalLight(name=l.name, position=l.position,
                                     look_at=l.look_at, intensity=l.intensity)
    scene += kb.PerspectiveCamera(name="camera", position=spec.camera_position,
                                  look_at=spec.camera_look_at)
    return scene, simulator, renderer, objs


def _fade_control(renderer, obj):
    """Mix the body's shaded surface with a Transparent BSDF; return the blend.

    Cycles renders the Principled BSDF's own `Alpha` input as opaque here --
    measured, see `probe_opacity.py` -- so fading has to be done by mixing in a
    transparent shader. The returned socket is 1 for solid and 0 for invisible,
    and it keyframes like anything else.

    Built once per body and reused -- the node is looked up by name on repeat
    calls, so re-rendering the same scene for the next family does not stack up
    mix shaders.
    """
    bmat = obj.material.linked_objects.get(renderer)
    if bmat is None or getattr(bmat, "node_tree", None) is None:
        return None
    nt = bmat.node_tree
    existing = nt.nodes.get("physloc_fade")
    if existing is not None:
        return existing.inputs[0]
    principled = nt.nodes.get("Principled BSDF")
    out_node = next((n for n in nt.nodes if n.type == "OUTPUT_MATERIAL"), None)
    if principled is None or out_node is None:
        return None
    transp = nt.nodes.new("ShaderNodeBsdfTransparent")
    mix = nt.nodes.new("ShaderNodeMixShader")
    mix.name = "physloc_fade"
    nt.links.new(transp.outputs[0], mix.inputs[1])
    nt.links.new(principled.outputs["BSDF"], mix.inputs[2])
    nt.links.new(mix.outputs[0], out_node.inputs["Surface"])
    return mix.inputs[0]


def _clear_animation(renderer, obj) -> None:
    """Drop every existing keyframe on a body before re-keying it.

    **`replay()` is not idempotent without this**, and that was a prefix-identity
    bug in 29 of 176 clips in the review sweep. The valid clip is the first
    replay a scene ever sees, so its curves are built from empty; every invalid
    clip is a replay laid on top of a fully populated curve. Blender re-solves
    each key's Bezier handles against whatever keys exist at insert time, so the
    two orders do not converge on the same F-curve even when every keyed value
    is identical -- and the render then differs by a few levels on scattered
    frames *before* `t_event`, on families that touch no material channel at
    all. `probe_replay.py` renders the same trajectory twice and measures it:
    53 levels on `pendulum_swing` before this, 0 after.

    Cheap: clearing and re-keying costs a fraction of one frame's render.
    """
    bo = obj.linked_objects.get(renderer)
    if bo is not None and getattr(bo, "animation_data", None) is not None:
        bo.animation_data_clear()
    mat = getattr(obj, "material", None)
    bmat = None if mat is None else mat.linked_objects.get(renderer)
    if bmat is not None:
        if getattr(bmat, "animation_data", None) is not None:
            bmat.animation_data_clear()
        nt = getattr(bmat, "node_tree", None)
        if nt is not None and getattr(nt, "animation_data", None) is not None:
            nt.animation_data_clear()
    # Kubric's own record of what it has keyed, so its bookkeeping agrees with
    # Blender's rather than silently describing curves that no longer exist.
    for member in list(getattr(obj, "keyframes", {})):
        obj.keyframes[member].clear()


def _set_visibility(renderer, obj, body) -> None:
    """Cycles ray visibility for one body.

    `shadow_track` needs it: the actor's own cast shadow is switched off so the
    scripted shadow body is the only one in frame. Blender moved these flags
    from `cycles_visibility` onto the object in 2.92, and the pinned image is
    2.93.4, but the fallback costs two lines and removes a version dependency
    from a path that silently produces two shadows if it breaks.
    """
    if body.visible_camera and body.visible_shadow:
        return
    bobj = obj.linked_objects.get(renderer)
    if bobj is None:
        return
    for attr, want in (("visible_camera", body.visible_camera),
                       ("visible_shadow", body.visible_shadow)):
        if hasattr(bobj, attr):
            setattr(bobj, attr, bool(want))
        elif hasattr(bobj, "cycles_visibility"):
            setattr(bobj.cycles_visibility, attr.replace("visible_", ""),
                    bool(want))


# --------------------------------------------------------------------------
def simulate(spec, scene, simulator, objs, hooks=()) -> Trajectory:
    """The valid rollout.

    `hooks` are the scenario's own per-substep constraints -- see
    `Scenario.sim_hooks`. When there are any the rollout goes through
    `stepper.run_from`, because `simulator.run` offers no per-step callback and
    a constraint that only holds after `t_event` is not a constraint. It records
    the same poses and the same contacts, so nothing downstream can tell which
    path produced the trajectory.
    """
    T = spec.tier.num_frames
    steps_per_frame = max(int(scene.step_rate) // max(int(scene.frame_rate), 1), 1)
    if hooks:
        return _simulate_with_hooks(spec, scene, simulator, objs, hooks)
    animation, collisions = simulator.run(frame_start=0, frame_end=T - 1)

    order = [b.name for b in spec.bodies]
    B = len(order)
    pos = np.zeros((T, B, 3), np.float32)
    quat = np.zeros((T, B, 4), np.float32)
    lvel = np.zeros((T, B, 3), np.float32)
    avel = np.zeros((T, B, 3), np.float32)
    for j, name in enumerate(order):
        a = animation[objs[name]]
        pos[:, j] = np.asarray(a["position"][:T], np.float32)
        quat[:, j] = np.asarray(a["quaternion"][:T], np.float32)
        lvel[:, j] = np.asarray(a["velocity"][:T], np.float32)
        avel[:, j] = np.asarray(a["angular_velocity"][:T], np.float32)

    seg_of = {b.name: b.segmentation_id for b in spec.bodies}
    name_of_obj = {objs[n]: n for n in order}

    # Kubric reports collisions with a float frame and *no* penetration depth
    # (it discards pybullet's contact_distance), so depth is left at 0 here and
    # computed geometrically by the residual module.
    cf, ca, cb, cp, cn, ci = [], [], [], [], [], []
    for c in collisions:
        a_obj, b_obj = c["instances"]
        if a_obj not in name_of_obj or b_obj not in name_of_obj:
            continue
        f = int(round(float(c["frame"])))
        if not (0 <= f < T):
            continue
        cf.append(f)
        ca.append(seg_of[name_of_obj[a_obj]])
        cb.append(seg_of[name_of_obj[b_obj]])
        cp.append(c["position"])
        cn.append(c["contact_normal"])
        # FRAME-AVERAGED normal force, not the substep sum.
        #
        # Kubric records one contact row per *substep* -- 20 per frame at the
        # 240 Hz step rate -- and rounding `frame` to an integer collapses all
        # twenty onto the same frame, where `laws.linear_momentum` sums them.
        # That inflated a resting body's contact force twentyfold: 1 kg at rest
        # is 9.81 N, summed to 196 N, which through F*dt / (m*|g|*dt) reads as a
        # momentum residual of ~20. That number is exactly the noise floor the
        # `linear_momentum` families were being scored against, and it is why
        # newton1_inertia, phantom_impulse and both newton2/3 shipped at
        # severity 0.000 -- their real signal of 1-3 sat far underneath it.
        #
        # Dividing by the substep count makes the stored value a force again, so
        # F*dt is the impulse actually delivered over the frame.
        ci.append(float(c["force"]) / max(steps_per_frame, 1))
    n = len(cf)
    contacts = Contacts(
        np.asarray(cf, np.int32) if n else np.zeros((0,), np.int32),
        np.asarray(ca, np.int32) if n else np.zeros((0,), np.int32),
        np.asarray(cb, np.int32) if n else np.zeros((0,), np.int32),
        np.asarray(cp, np.float32).reshape(n, 3) if n else np.zeros((0, 3), np.float32),
        np.asarray(cn, np.float32).reshape(n, 3) if n else np.zeros((0, 3), np.float32),
        np.asarray(ci, np.float32) if n else np.zeros((0,), np.float32),
        np.zeros((n,), np.float32),
    )

    # A dormant understudy is in the scene graph but not in the scene: absent
    # from frame 0, so it renders as nothing at all until `fission` switches it
    # on. `is_static` follows `static`, not `sim_static`, so a scripted actor
    # still gets residuals computed for it.
    present = np.ones((T, B), bool)
    for j, b in enumerate(spec.bodies):
        if b.dormant:
            present[:, j] = False

    colour = np.zeros((T, B, 3), np.float32)
    for j, b in enumerate(spec.bodies):
        colour[:, j, :] = np.asarray(b.color, np.float32)

    return Trajectory(
        body_ids=np.asarray([seg_of[n_] for n_ in order], np.int32),
        body_names=list(order),
        pos=pos, quat=quat, lin_vel=lvel, ang_vel=avel, present=present,
        colour=colour,
        mass=np.asarray([b.mass for b in spec.bodies], np.float32),
        radius=np.asarray([b.bounding_radius for b in spec.bodies], np.float32),
        is_static=np.asarray([b.static for b in spec.bodies], bool),
        contacts=contacts, fps=float(spec.tier.fps),
        gravity=np.asarray(spec.gravity, np.float32),
        meta={"scenario": spec.scenario, "seed": spec.seed,
              "tier": spec.tier.name, "label": "valid",
              "spec": spec.to_dict()},
    )


def _simulate_with_hooks(spec, scene, simulator, objs, hooks) -> Trajectory:
    """A valid rollout stepped by hand, so the scenario's constraints can run.

    Shares `stepper.run_from` with the staged path, and then borrows the shape
    of the trajectory from an empty `simulate` so every field -- masses, radii,
    colours, the dormant bodies' `present` flags -- is filled in exactly as it
    would have been.
    """
    T = spec.tier.num_frames
    tail = stepper.run_from(simulator, scene, spec, objs, 0, T - 1, hooks)
    blank = _empty_trajectory(spec)
    return stepper.splice(blank, tail, 0)


def _empty_trajectory(spec) -> Trajectory:
    """Every per-body constant, with no motion in it yet."""
    T, B = spec.tier.num_frames, len(spec.bodies)
    present = np.ones((T, B), bool)
    for j, b in enumerate(spec.bodies):
        if b.dormant:
            present[:, j] = False
    colour = np.zeros((T, B, 3), np.float32)
    for j, b in enumerate(spec.bodies):
        colour[:, j, :] = np.asarray(b.color, np.float32)
    empty = np.zeros((0,), np.int32)
    return Trajectory(
        body_ids=np.asarray([b.segmentation_id for b in spec.bodies], np.int32),
        body_names=[b.name for b in spec.bodies],
        pos=np.zeros((T, B, 3), np.float32), quat=np.zeros((T, B, 4), np.float32),
        lin_vel=np.zeros((T, B, 3), np.float32),
        ang_vel=np.zeros((T, B, 3), np.float32),
        present=present, colour=colour,
        mass=np.asarray([b.mass for b in spec.bodies], np.float32),
        radius=np.asarray([b.bounding_radius for b in spec.bodies], np.float32),
        is_static=np.asarray([b.static for b in spec.bodies], bool),
        contacts=Contacts(empty, empty, empty,
                          np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32),
                          np.zeros((0,), np.float32), np.zeros((0,), np.float32)),
        fps=float(spec.tier.fps), gravity=np.asarray(spec.gravity, np.float32),
        meta={"scenario": spec.scenario, "seed": spec.seed,
              "tier": spec.tier.name, "label": "valid", "spec": spec.to_dict()})


# Where a removed body is parked. Far enough outside any scenario's camera
# frustum that it contributes no pixels, which is how `permanence` renders as
# genuine absence rather than as a teleport.
GONE_Z = -1000.0


def _keyframe_camera(scene, spec, num_frames: int) -> None:
    """Lay the camera track down as keyframes, every variant, unconditionally.

    UNCONDITIONALLY is the whole point, and it is the rule `_clear_animation`
    already states for body channels. One scene is built per (scenario, seed)
    and reused for the valid render and for every family and severity after it
    (see `render_variants`), so a track keyframed once and not rewritten would
    leak into every clip that follows -- and a camera that differs between the
    twins shows up as a pre-`t_event` pixel diff, which is the one thing the
    whole trajectory seam exists to prevent.

    Static cameras are keyframed too, at a constant pose, so that a moving
    camera in one variant cannot survive into a still one in the next.

    The loop runs one frame either side of the clip, as
    `movi_def_worker.py:165-201` does: the forward and backward flow passes are
    differences between neighbouring frames, and without the extra keyframes
    the first and last of them are computed against a camera that was still.
    """
    cam = scene.camera
    if cam is None:
        return
    for frame in range(-1, num_frames + 1):
        position, look_at = spec.camera_at(frame, num_frames)
        cam.position = position
        # `look_at` sets the quaternion NOW and animates nothing by itself, so
        # it has to be re-called per frame and the quaternion keyframed beside
        # the position.
        cam.look_at(look_at)
        cam.keyframe_insert("position", frame)
        cam.keyframe_insert("quaternion", frame)


def replay(spec, objs, traj: Trajectory, renderer=None, scene=None) -> None:
    """Write a trajectory back onto the scene as keyframes -- the seam's read side.

    Same mechanism Kubric's own simulator.run uses to hand animation to Blender
    (see refs/kubric/kubric/simulator/pybullet.py), so this is a supported path.
    """
    if scene is not None:
        _keyframe_camera(scene, spec, int(traj.num_frames))
    present = traj.present
    scale_mul = traj.scale_mul
    colour = traj.colour
    opacity = traj.opacity
    for j, b in enumerate(spec.bodies):
        obj = objs[b.name]
        _clear_animation(renderer, obj)
        # Scale is keyframed on EVERY body, every frame, whether or not this
        # trajectory changes it. One worker run renders several families through
        # the same scene object, and a keyframe nobody overwrites simply stays:
        # without this, `immutability`'s resize leaked into every family
        # rendered after it in the same run and silently enlarged the actor in
        # clips that had nothing to do with size. It surfaced as a *negative*
        # observability lag -- the twins differing before the event -- which is
        # the one thing that cannot happen honestly.
        #
        # The dome is the exception: a background asset with its own scale,
        # which resizing would turn into resizing the world.
        resize = b.kind != "dome"
        # Material animation is keyframed the same way pose and scale are.
        # Verified against the pinned image: `material.keyframe_insert("color",
        # f)` writes one fcurve per RGBA channel on the Principled BSDF's Base
        # Colour and the render follows it. The dome is excluded because its
        # material carries the HDRI, not a colour.
        # EVERY animated channel is keyframed on EVERY body on EVERY frame,
        # whether or not this trajectory touches it. This is not defensive
        # coding, it is the fix for a bug that has now happened three times.
        #
        # One worker run renders several families through the same scene object,
        # and a keyframe nobody overwrites simply stays. Each time a new channel
        # was added it was keyframed "only when it changes", and each time the
        # family that used it contaminated every family rendered after it in the
        # same run: `immutability` enlarged the actor in solidity clips,
        # `colour_shift` turned it green in everything downstream, and
        # `dissolve` faded it out of barrier_pass and newton1. All three
        # produced clips depicting two violations and labelled with one.
        #
        # The rule for any channel added later: keyframe it unconditionally.
        # The cost is a few thousand redundant keyframes; the alternative is a
        # silent cross-family leak that only shows up by eye.
        recolour = b.kind != "dome"
        fade_socket = _fade_control(renderer, obj) if b.kind != "dome" else None
        for f in range(traj.num_frames):
            if present is not None and not bool(present[f, j]):
                obj.position = (0.0, 0.0, GONE_Z)
            else:
                obj.position = tuple(float(x) for x in traj.pos[f, j])
            obj.quaternion = tuple(float(x) for x in traj.quat[f, j])
            obj.keyframe_insert("position", f)
            obj.keyframe_insert("quaternion", f)
            if resize:
                obj.scale = tuple(float(b.draw_scale[k] * scale_mul[f, j, k])
                                  for k in range(3))
                obj.keyframe_insert("scale", f)
            if recolour:
                obj.material.color = kb.Color(*(float(x) for x in colour[f, j]))
                obj.material.keyframe_insert("color", f)
            if fade_socket is not None:
                fade_socket.default_value = float(opacity[f, j])
                fade_socket.keyframe_insert("default_value", frame=f)


def render_and_save(renderer, scene, spec, objs, outdir, tag: str):
    t0 = time.perf_counter()
    stack = renderer.render()
    dt = time.perf_counter() - t0

    # Kubric numbers the raw segmentation by scene-asset order and only honours
    # each asset's declared `segmentation_id` if you run this post-process --
    # exactly what movi_def_worker.py does. Skipping it silently mislabels every
    # instance whenever declaration order differs from insertion order, which
    # then quietly corrupts every mask and residual downstream.
    ordered = [objs[b.name] for b in spec.bodies if b.name in objs]
    stack["segmentation"] = kb.adjust_segmentation_idxs(
        stack["segmentation"], scene.assets, ordered)

    declared = sorted({int(b.segmentation_id) for b in spec.bodies})
    got = sorted(int(x) for x in np.unique(np.asarray(stack["segmentation"])) if x != 0)
    if not set(got).issubset(set(declared)):
        raise RuntimeError("segmentation ids %s are not the declared %s"
                           % (got, declared))

    arrays = {}
    for k in PASSES:
        if k in stack:
            arrays[k] = np.asarray(stack[k])
    np.savez_compressed(os.path.join(outdir, "passes_%s.npz" % tag), **arrays)

    # No image files here: the container has no ffmpeg. All mp4s -- previews,
    # released rgb.mp4 and the annotation overlays -- are written host-side by
    # physloc.viz, which keeps a single encoder and one set of settings.
    return dt, {k: list(v.shape) for k, v in arrays.items()}


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="drop")
    ap.add_argument("--seed", type=int, default=91731)
    ap.add_argument("--tier", default="debug", choices=sorted(scenarios.TIERS))
    ap.add_argument("--family", default="solidity",
                    help="one family, or a comma list -- every family named "
                         "shares this run's single valid render")
    ap.add_argument("--severity", default="all",
                    help="one of weak|medium|strong, 'all', or a comma list")
    ap.add_argument("--complexity", default="L0")
    ap.add_argument("--window", type=int, default=None,
                    help="uniform violation duration in frames; sustained "
                         "families honour it, instant ones ignore it")
    ap.add_argument("--resolution", type=int)
    ap.add_argument("--fps", type=int)
    ap.add_argument("--frames", type=int)
    ap.add_argument("--spp", type=int)
    ap.add_argument("--outdir", default="out/phase0")
    a = ap.parse_args()

    tier = scenarios.TIERS[a.tier].override(
        resolution=a.resolution, fps=a.fps, num_frames=a.frames,
        samples_per_pixel=a.spp)
    spec = scenarios.get(a.scenario).sample(a.seed, tier, a.complexity)

    pair_uid = "%s/%04d" % (a.scenario, a.seed)
    outdir = os.path.join(a.outdir, a.scenario, "%04d" % a.seed)
    scratch = os.path.join(outdir, "_scratch")
    os.makedirs(scratch, exist_ok=True)

    severities = (["weak", "medium", "strong"] if a.severity == "all"
                  else [x.strip() for x in a.severity.split(",") if x.strip()])

    # One scene build, one HDRI load, one valid rollout -- shared by every
    # variant. At complexity L1 the environment costs ~4.6x an L0 render, so
    # re-paying it per severity was most of the wall clock.
    scene, simulator, renderer, objs = build_scene(spec, scratch)
    # The scenario's own constraints, which must hold on the valid rollout and
    # on every invalid one -- a rope that is only inextensible after `t_event`
    # is not a rope.
    scen_hooks = tuple(scenarios.get(a.scenario).sim_hooks(spec, simulator, objs)
                       or ())
    traj_valid = simulate(spec, scene, simulator, objs, scen_hooks)
    # Bodies whose pose is drawn rather than solved -- `shadow_track`'s cast
    # shadow, which is not an object and has no dynamics to get right. A
    # CONSTRAINED scenario no longer comes through here: a pendulum is a real
    # body held by a real constraint (`sim_hooks`), so the simulator produces
    # its arc like any other.
    scenarios.get(a.scenario).script(spec, traj_valid)
    traj_valid.save(os.path.join(outdir, "traj_valid.npz"))

    replay(spec, objs, traj_valid, renderer, scene)
    t_valid, shapes = render_and_save(renderer, scene, spec, objs, outdir, "valid")

    families = [f.strip() for f in a.family.split(",") if f.strip()]
    variants, timings = [], {"valid": round(t_valid, 2)}
    for family in families:
        inj = injectors.get(family)
        inj.window_frames = a.window
        # An ungraded family has no magnitude axis, so its three bins would be
        # the same clip three times. Generate the strongest only.
        meta = taxonomy.FAMILIES.get(family)
        bins = severities
        if meta is not None and not getattr(meta, "graded", True):
            bins = [severities[-1]] if severities else []
        for sev in bins:
            tag = "%s/%s" % (family, sev)
            # The rng is seeded per (family, severity), not per run, so adding
            # a family to the list cannot change the clips the others produce.
            #
            # crc32, never `hash()`. Python randomises string hashing per
            # process unless PYTHONHASHSEED is set, and it is set nowhere here,
            # so `hash(tag)` made the same (scenario, seed, family, severity)
            # render a *different clip on every run*. Nothing in the test suite
            # noticed, because every check ran against a single generation.
            rng = np.random.RandomState(
                (a.seed + 7919 + zlib.crc32(tag.encode())) % (2 ** 31 - 1))
            try:
                plan = inj.plan(spec, traj_valid, rng, sev)
            except Exception as exc:                       # noqa: BLE001
                variants.append({"family": family, "severity": sev, "ok": False,
                                 "error": "plan raised: %r" % (exc,)})
                continue
            if plan is None:
                variants.append({"family": family, "severity": sev, "ok": False,
                                 "error": "injector produced no plan"})
                continue
            # A SCRIPTED body is pinned in the simulator -- `sim_static`, mass
            # zero -- and its motion arrives from the trajectory instead. So a
            # staged intervention on one cannot move it: the world is reset to
            # `t_event`, PyBullet is run forward, and the body simply sits
            # there. Every `simulated` family on `pendulum_swing` shipped that
            # way -- continuity, deformation, global_gravity, immutability and
            # phantom_impulse all froze the bob mid-swing, which is five
            # different labels on one picture of a stopped pendulum.
            #
            # The rule is the one CLAUDE.md already states: a family stages
            # itself only where something in the simulator corresponds to what
            # it changes. Nothing does, for a body the simulator does not move.
            # Checked here rather than in each injector because it is a fact
            # about the SCENE, and no injector should have to learn it.
            scripted = {int(b.segmentation_id) for b in spec.bodies if b.scripted}
            staged = (inj.simulates(plan)
                      and not scripted.intersection(plan.causal_body_ids))
            if staged:
                # Real physics from t_event: reset the world to the valid state,
                # stage the intervention as something PyBullet honours, run
                # forward, then undo the staging so the next variant starts
                # clean. Frames before t_event come from the valid rollout
                # verbatim, so prefix identity holds by construction.
                try:
                    stepper.reset_to(spec, objs, traj_valid, plan.t_event)
                    hooks = tuple(inj.stage(spec, simulator, objs, plan)
                                  or ()) + scen_hooks
                    tail = stepper.run_from(simulator, scene, spec, objs,
                                            plan.t_event, spec.tier.num_frames - 1,
                                            hooks)
                    traj_invalid = stepper.splice(traj_valid, tail, plan.t_event)
                finally:
                    inj.unstage(spec, simulator, objs, plan)
                    stepper.reset_to(spec, objs, traj_valid,
                                     spec.tier.num_frames - 1)
                traj_invalid = inj.post_simulate(spec, traj_valid, traj_invalid,
                                                 plan)
            else:
                traj_invalid = inj.apply(spec, traj_valid, plan)

            ok, why = prefix_identical(traj_valid, traj_invalid, plan.t_event)
            if not ok:
                variants.append({"family": family, "severity": sev, "ok": False,
                                 "error": "trajectory prefix differs: %s" % why})
                continue

            # Windows that can only be known from the finished trajectory --
            # `solidity` ends when the bodies stop overlapping, not after a
            # fixed number of frames. Runs before plan.json is written.
            inj.refine_windows(spec, traj_valid, traj_invalid, plan)

            vdir = os.path.join(outdir, "variants", "%s_%s" % (family, sev))
            os.makedirs(vdir, exist_ok=True)
            traj_invalid.save(os.path.join(vdir, "traj_invalid.npz"))
            with open(os.path.join(vdir, "plan.json"), "w") as fh:
                json.dump({"pair_uid": pair_uid, "spec": spec.to_dict(),
                           "plan": plan.to_dict()}, fh, indent=2, sort_keys=True)

            replay(spec, objs, traj_invalid, renderer, scene)
            dt, _ = render_and_save(renderer, scene, spec, objs, vdir, "invalid")
            timings[tag] = round(dt, 2)
            variants.append({"family": family, "severity": sev, "ok": True,
                             "dir": vdir, "kind": plan.kind,
                             "t_event": plan.t_event, "windows": plan.windows,
                             "magnitude": plan.magnitude,
                             "magnitude_unit": plan.magnitude_unit})

    print("PHASE0 " + json.dumps({
        "ok": any(v.get("ok") for v in variants),
        "pair_uid": pair_uid, "outdir": outdir, "family": a.family,
        "tier": tier.name, "complexity": a.complexity,
        "frames": tier.num_frames, "variants": variants,
        "render_seconds": timings, "passes": shapes,
    }))
    return 0 if any(v.get("ok") for v in variants) else 2


if __name__ == "__main__":
    raise SystemExit(main())
