"""Can a body be faded out rather than switched off? Measured, and yes.

    bash docker/kubric.sh physloc/render/probe_opacity2.py

Two mechanisms were tried against the pinned image (Kubric 2022.4.1 / Blender
2.93.4). The Principled BSDF's own `Alpha` input does nothing here. Mixing the
shaded surface with a **Transparent BSDF** works exactly as it should:

    solid        ball mean RGB  152.6  80.5  82.3   (a red ball)
    green        ball mean RGB   52.4 155.1  53.8   (base colour animates)
    transparent  ball mean RGB   88.8  88.8  97.8   (the floor, seen through it)

**The reason this took three attempts is worth recording.** The first two probes
never called `kb.adjust_segmentation_idxs`, so the segmentation was numbered by
scene-asset order rather than by the declared ids -- id 2 was the *floor*. Every
measurement was of the wrong object, and both mechanisms looked broken because
the floor does not change when you recolour a ball. CLAUDE.md warns about
exactly this trap and it still cost two rounds; the conclusion "transparency is
impossible here" was written down and acted on before the probe was correct.

One real constraint remains. Segmentation is unaffected by transparency -- the
ball still reports 1264 pixels when fully invisible, because cryptomatte tracks
geometry. That is why `observable_frames` compares RGB as well as occupancy: a
seg-only test would call an invisible body present and report a dissolve as
having no visible evidence at all.
"""
import sys, os
sys.path.insert(0, "/kubric")
import numpy as np, kubric as kb
from kubric.renderer import Blender

def build(mode):
    scratch = "/kubric/out/_probe_%s" % mode; os.makedirs(scratch, exist_ok=True)
    scene = kb.Scene(resolution=(96, 96), frame_start=0, frame_end=0, frame_rate=12)
    r = Blender(scene, scratch, samples_per_pixel=16, background_transparency=False)
    scene.background = kb.Color(0.02, 0.02, 0.03)
    mat = kb.PrincipledBSDFMaterial(color=kb.Color(0.85, 0.15, 0.15), roughness=0.5)
    ball = kb.Sphere(name="ball", scale=(0.6,)*3, position=(0, 0, 0.6),
                     static=True, material=mat, segmentation_id=2)
    scene += ball
    scene += kb.Cube(name="floor", scale=(4, 4, 0.1), position=(0, 0, -0.1),
                     static=True, segmentation_id=1,
                     material=kb.PrincipledBSDFMaterial(color=kb.Color(.35,.35,.4)))
    scene += kb.DirectionalLight(name="sun", position=(-2,-2,4), look_at=(0,0,0),
                                 intensity=2.2)
    scene += kb.PerspectiveCamera(name="cam", position=(0,-4,1.2), look_at=(0,0,0.6))

    if mode == "transparent":
        bmat = ball.material.linked_objects[r]
        nt = bmat.node_tree
        principled = nt.nodes["Principled BSDF"]
        out_node = next(n for n in nt.nodes if n.type == "OUTPUT_MATERIAL")
        transp = nt.nodes.new("ShaderNodeBsdfTransparent")
        mix = nt.nodes.new("ShaderNodeMixShader")
        nt.links.new(transp.outputs[0], mix.inputs[1])
        nt.links.new(principled.outputs["BSDF"], mix.inputs[2])
        nt.links.new(mix.outputs[0], out_node.inputs["Surface"])
        mix.inputs[0].default_value = 0.0          # fully transparent
        print("  linked:", out_node.inputs["Surface"].links[0].from_node.name)
    elif mode == "green":
        ball.material.color = kb.Color(0.05, 0.9, 0.05)

    out = r.render(return_layers=["rgba", "segmentation"])
    # Without this the segmentation is numbered by scene-asset order, not by the
    # declared ids -- so "id 2" was the floor and every measurement above was of
    # the wrong object. CLAUDE.md warns about exactly this.
    out["segmentation"] = kb.adjust_segmentation_idxs(
        out["segmentation"], scene.assets, [ball])
    rgba, seg = np.asarray(out["rgba"]), np.asarray(out["segmentation"])
    m = seg[0, ..., 0] == 2
    return int(m.sum()), np.round(rgba[0][m][:, :3].mean(0), 1)

for mode in ("solid", "green", "transparent"):
    px, rgb = build(mode)
    print("%-12s segpx %4d  ball mean rgb %s" % (mode, px, rgb))
