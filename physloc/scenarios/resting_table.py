"""`resting_table` -- several bodies sitting still on a table top.

Nothing happens, which is exactly what it is for. Static equilibrium is the
cleanest possible baseline for `support`, `newton1_inertia` and
`phantom_impulse`: any motion at all is the violation, so the observability lag
is zero by construction and the mask has no ambiguity. It is also the scenario
where a `permanence` removal is most obvious. Grounded in IntPhys 2 permanence.

Two static surfaces on purpose -- the floor and the table -- because that is the
case where "what is holding this up?" has a wrong answer available. An injector
that picks the first static body in the list annotates a mug as hovering a
table's height above the ground; `_geom.support_under` picks by height instead.
"""
from __future__ import annotations

from . import _common as C
from . import materials as M
from .base import (COMPLEXITY, DEFAULT_COMPLEXITY, BodySpec, SceneSpec,
                   Scenario, Tier, register)


class RestingTable(Scenario):
    name = "resting_table"
    SEG_FLOOR, SEG_ACTOR, SEG_TABLE, SEG_POST, SEG_SPLIT = 1, 2, 3, 8, 9
    SEG_PROPS = (4, 5)

    def _sample(self, seed: int, tier: Tier,
                complexity: str = DEFAULT_COMPLEXITY) -> SceneSpec:
        rng = self.rng(seed)
        cx = COMPLEXITY[complexity]
        arng = C.appearance_rng(seed, self.name)
        if not cx.implemented:
            raise NotImplementedError("complexity %s not built" % complexity)

        top_z = 0.76
        table = BodySpec(name="table", kind="cube", position=(0.0, 0.0, top_z - 0.06),
                         scale=(1.5, 0.95, 0.06), mass=0.0, static=True,
                         friction=0.7, restitution=0.05, color=(0.42, 0.33, 0.26),
                         segmentation_id=self.SEG_TABLE, role="prop")
        post = BodySpec(name="post", kind="cube", position=(0.0, 0.0, 0.35),
                        scale=(0.16, 0.16, 0.35), mass=0.0, static=True,
                        friction=0.7, restitution=0.05, color=(0.35, 0.28, 0.22),
                        segmentation_id=self.SEG_POST, role="prop")

        hue = float(rng.uniform(0, 1))
        xs = [-0.85, 0.0, 0.85]
        rng.shuffle(xs)
        r0 = float(rng.uniform(0.20, 0.26))
        actor_kind = "sphere" if rng.rand() < 0.6 else "cube"
        actor = BodySpec(name="mug", kind=actor_kind,
                         position=(xs[0], float(rng.uniform(-0.2, 0.2)), top_z + r0),
                         scale=(r0,) * 3, mass=0.9, friction=0.7, restitution=0.05,
                         color=C.hue_rgb(hue), segmentation_id=self.SEG_ACTOR,
                         role="actor")
        props = []
        for i in range(2):
            h = float(rng.uniform(0.17, 0.23))
            kind = "cube" if i == 0 else "sphere"
            props.append(BodySpec(
                name="prop_%d" % i, kind=kind,
                position=(xs[i + 1], float(rng.uniform(-0.2, 0.2)), top_z + h),
                scale=(h,) * 3, mass=0.9, friction=0.7, restitution=0.05,
                color=C.hue_rgb((hue + 0.2 + 0.25 * i) % 1.0),
                segmentation_id=self.SEG_PROPS[i], role="prop"))

        # The actor and the props are separate objects on a table, so they are
        # free to be made of different things -- unlike `collision`'s pair or
        # `pour`'s medium, whose sameness is load-bearing.
        actor = C.with_material(actor, M.pick(arng), arng)
        actor = C.vary_dims(actor, arng)
        props = [C.vary_dims(C.with_material(pr, M.pick(arng), arng), arng)
                 for pr in props]

        return SceneSpec(
            scenario=self.name, seed=seed, tier=tier,
            bodies=([C.ground(cx, self.SEG_FLOOR), post, table, actor]
                    + props + [C.understudy(actor, self.SEG_SPLIT)]),
            lights=C.lights(cx, look_at=(0, 0, 0.9)),
            camera_position=(2.6, -4.2, 1.85), camera_look_at=(0.0, 0.0, 0.85),
            floor_level=0.0, complexity=complexity,
            notes={"table_top": top_z, "actor_radius": r0,
                   "actor_kind": actor_kind})


register(RestingTable())
