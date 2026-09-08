"""What a body is made of -- appearance and density, kept consistent.

Mass used to be the literal `1.0` on almost every body in the project, and a
benchmark whose objects all weigh the same teaches a model that mass is not a
variable. The obvious fix -- draw a random mass -- makes a different problem:
a viewer has no way to see that one ball is four times heavier than another,
so a clip where the heavy one barely moves reads as a violation when it is
lawful physics. The label would say "valid" and the picture would say
otherwise.

So mass is not drawn. It is DERIVED from a material, and the material is
visible: a steel cube looks like steel and is dense, a cork cube looks like
cork and is light. The appearance and the number agree by construction, which
is what makes the resulting motion legible rather than arbitrary.

`density x volume` also keeps size and mass coherent. Two balls of the same
material differ in mass exactly as much as they differ in size, and a big one
is never lighter than a small one -- which is the other way a viewer's
expectation gets violated without anything being labelled.

Densities are the real ones in kg/m^3, rounded. They are used as ratios, so the
absolute scale only matters in that PyBullet's solver behaves best when masses
sit within a couple of orders of magnitude of each other -- hence `MASS_SCALE`,
which anchors a typical wooden actor at the 1 kg every body in the project used
to be hard-coded to. The spread that results is about 32x, cork to steel.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class Material:
    """A material's physics and its look, defined together.

    `roughness` and `metallic` are the two PrincipledBSDF dials that carry most
    of "what is this made of" at the v0 asset set's fidelity -- untextured
    primitives under one sun. Metal is smooth and metallic; rubber is rough and
    dark; ceramic is smooth and bright.

    `value` and `saturation` bound the HSV the colour draw is allowed to use, so
    a body reads as its material whatever hue it is given. Steel cannot come out
    a saturated pink, and wood stays within a believable brown-orange band via
    `hue_range`; a plastic toy can be any colour at all.
    """

    name: str
    density: float                     # kg/m^3, real
    roughness: float
    metallic: float
    saturation: Tuple[float, float]
    value: Tuple[float, float]
    #: `None` means "any hue"; otherwise the (lo, hi) band in turns.
    hue_range: Tuple[float, float] = None
    #: The three dials that were left at their defaults, and the reason L1
    #: looked like L0. `roughness` and `metallic` alone cannot separate seven
    #: matte dielectrics under one sun -- six of the seven materials rendered
    #: as "flat colour, slightly different sheen", which is what L0 already is.
    #:
    #: `specular` is what makes ceramic read as glazed and rubber as dead.
    #: `transmission` with an `ior` is glass, and it is the single most visible
    #: material available. Both matter far more from L2 up: a metal or a glass
    #: body REFLECTS AND REFRACTS the environment, which is precisely what
    #: makes an object look like it belongs in the scene rather than composited
    #: onto it -- your L2 note.
    specular: float = 0.5
    transmission: float = 0.0
    ior: float = 1.45
    #: How often this material is drawn, relative to the others. NOT uniform,
    #: because the palette is not uniform in density: adding three dense metals
    #: to make L1 legible also pushed the mean density from 2313 to 3458 and
    #: doubled the median, so a uniform draw would have quietly made the whole
    #: dataset heavier than the one the scenarios were tuned against. Weighting
    #: the light end back up restores the old centre of mass while keeping the
    #: wider palette -- see `pick`.
    weight: float = 1.0


#: Divides every density. Chosen so a MEDIAN actor -- the scenarios draw radii
#: around 0.45 -- comes out near 1 kg in wood, which is the mass nearly every
#: body in the project carried as a hard-coded literal before materials
#: existed. Anchoring there keeps a scenario in the contact regime its
#: restitution and friction were tuned in; only the spread around it is new.
#:
#: Anchoring on a 0.3 m body instead put a large ceramic block at 37 kg, since
#: mass goes as the cube of the size and the actors are bigger than that.
#:
#: Ratios between materials are what the physics actually reads, and those are
#: the real ones, untouched.
MASS_SCALE = 250.0

MATERIALS: Dict[str, Material] = {
    # ---- light dielectrics -------------------------------------------------
    # Cork rather than foam on purpose: real foam is around 60 kg/m^3, which
    # next to a dense metal is a mass ratio of 150 to 1, and a body that light
    # skitters under a solver whose contact parameters were tuned near 1 kg.
    "cork": Material("cork", 240.0, 0.94, 0.0, (0.15, 0.40), (0.55, 0.80),
                     hue_range=(0.05, 0.12), specular=0.18, weight=1.6),
    # Warm, matte, mid-density. Hue is bounded because a blue plank does not
    # read as wood, and the whole point is that the material is recognisable.
    "wood": Material("wood", 650.0, 0.82, 0.0, (0.35, 0.62), (0.45, 0.75),
                     hue_range=(0.03, 0.11), specular=0.30, weight=1.6),
    # The free hue: a plastic object is credibly any colour, which keeps the
    # dataset's colour variety from collapsing once materials bound the rest.
    "plastic": Material("plastic", 1100.0, 0.40, 0.0, (0.45, 0.85),
                        (0.60, 0.92), specular=0.55, weight=1.6),
    # Dark and very rough, and now with almost no specular -- rubber is the one
    # material that should have no highlight at all, and at the 0.5 default it
    # had the same sheen as plastic.
    "rubber": Material("rubber", 1300.0, 0.96, 0.0, (0.05, 0.30), (0.12, 0.30),
                       specular=0.12, weight=1.6),

    # Matte, pale, and flatter than wood -- the light end needed a second
    # ordinary material, not another exotic one. Cardboard reads instantly as
    # light, which is the point: a viewer should be able to guess the mass.
    "cardboard": Material("cardboard", 700.0, 0.95, 0.0, (0.18, 0.38),
                          (0.50, 0.72), hue_range=(0.06, 0.11), specular=0.10,
                          weight=1.6),

    # ---- glazed and transmissive ------------------------------------------
    # Porcelain rather than a toy: smooth, bright, and a strong highlight.
    "ceramic": Material("ceramic", 2400.0, 0.15, 0.0, (0.02, 0.20),
                        (0.75, 0.95), specular=0.85),
    # FROSTED, not clear. Clear glass is the most visible material there is and
    # also the least localisable -- a transparent culprit is hard to point at,
    # and pointing at it is the task. At 0.92 transmission with roughness 0.22
    # it is unmistakably glass and still has a silhouette.
    "glass": Material("glass", 2500.0, 0.22, 0.0, (0.02, 0.18), (0.80, 0.98),
                      specular=0.9, transmission=0.92, ior=1.46),
    # The LIGHT transmissive one. Glass at 2500 was the only way to be
    # see-through, which tied "transparent" to "middling heavy"; ice is
    # transparent at 920 and breaks that correlation, so transmission stops
    # being a cue for mass.
    "ice": Material("ice", 920.0, 0.10, 0.0, (0.02, 0.12), (0.85, 0.98),
                    hue_range=(0.50, 0.58), specular=0.9, transmission=0.88,
                    ior=1.31, weight=1.6),
    # Polished stone. Same density as `stone`, opposite finish, so the pair
    # separates surface from substance.
    "marble": Material("marble", 2700.0, 0.16, 0.0, (0.02, 0.14),
                       (0.62, 0.88), specular=0.75),
    "stone": Material("stone", 2700.0, 0.92, 0.0, (0.03, 0.18), (0.30, 0.55),
                      specular=0.20),

    # ---- metals ------------------------------------------------------------
    # FOUR of them, where there was one. Metal is the fastest-reading material
    # a viewer has -- and under an HDRI it mirrors the environment, so it is
    # also what ties an object to its background. One metal in seven meant six
    # clips in seven looked like painted primitives.
    #
    # Tinted metals need saturation: a copper that is grey is a steel. The
    # colour of a metal is in its REFLECTION, so the base colour is what tints
    # everything it mirrors.
    "aluminium": Material("aluminium", 2700.0, 0.38, 1.0, (0.00, 0.06),
                          (0.72, 0.92), specular=0.6, weight=0.8),
    "steel": Material("steel", 7800.0, 0.22, 1.0, (0.00, 0.08), (0.55, 0.80),
                      specular=0.6, weight=0.8),
    "brass": Material("brass", 8500.0, 0.28, 1.0, (0.35, 0.60),
                      (0.62, 0.85), hue_range=(0.10, 0.14), specular=0.6, weight=0.6),
    "copper": Material("copper", 8900.0, 0.26, 1.0, (0.45, 0.70),
                       (0.55, 0.78), hue_range=(0.02, 0.06), specular=0.6, weight=0.6),
}

#: The default draw for an ACTOR: everything. Twelve materials, four of them
#: metallic and one transmissive, against the seven near-matte dielectrics that
#: made L1 hard to tell from L0.
ACTOR_MATERIALS: Tuple[str, ...] = tuple(MATERIALS)

#: What SCENERY may be made of -- a ramp, a barrier, a table, a pendulum post.
#: A narrower set on purpose: a mirror-finish floor throws the actor's
#: reflection across the scene and a glass ramp is a puzzle rather than a
#: surface, and neither is the difficulty this dataset is measuring. What is
#: wanted is that the staging stops looking like untextured grey blocks.
SCENERY_MATERIALS: Tuple[str, ...] = ("wood", "stone", "marble", "ceramic",
                                      "plastic", "steel")


def get(name: str) -> Material:
    return MATERIALS[name]


def pick(rng, choices: Tuple[str, ...] = ACTOR_MATERIALS) -> str:
    """One material name, drawn off whatever stream is handed in.

    WEIGHTED, not uniform. The palette is not uniform in density -- the three
    dense metals that make L1 legible are also 7800 to 8900 kg/m^3 -- so a
    uniform draw over fourteen materials put the mean density at 3458 against
    the seven-material set's 2313, and doubled the median. That is a change to
    the PHYSICS smuggled in by a change to the appearance: every scenario's
    contact tuning assumes a mass regime, and the dataset would have drifted
    out of it while the labels said nothing had changed.

    Weighting the light end up puts the mean back at 2256, close to where the
    scenarios were tuned, while keeping the wider palette. Weights live on the
    material so the two facts sit together.

    Draw this from `appearance_rng`, never from the physics stream: it decides
    a mass, but it decides it *deterministically* from the scene's look, and
    consuming a physics draw here would shift every value after it.
    """
    names = list(choices)
    w = [float(MATERIALS[n].weight) for n in names]
    total = sum(w)
    if total <= 0:                          # a caller passed all-zero weights
        return str(names[int(rng.randint(0, len(names)))])
    u = float(rng.uniform(0.0, total))
    acc = 0.0
    for name, weight in zip(names, w):
        acc += weight
        if u < acc:
            return str(name)
    return str(names[-1])


def mass_for(name: str, scale, kind: str = "sphere") -> float:
    """`density x volume`, in the project's mass units.

    `scale` is Kubric's half-extent triple, so a "unit" sphere of scale r has
    radius r and a cube of scale h is 2h on a side. Volume is computed from the
    actual shape rather than a bounding box, because a sphere is only 52% of
    its box and treating the two alike would make every ball implausibly heavy.
    """
    sx, sy, sz = (float(s) for s in scale)
    if kind == "sphere":
        volume = 4.0 / 3.0 * 3.141592653589793 * sx * sy * sz
    elif kind in ("cylinder", "cone"):
        # Kubric's cylinder is radius sx, half-height sz; a cone is a third of
        # the cylinder that contains it.
        volume = 3.141592653589793 * sx * sy * (2.0 * sz)
        if kind == "cone":
            volume /= 3.0
    elif kind == "torus":
        # KuBasic's torus is an outer radius of `scale` with a tube about a
        # quarter of it: V = 2*pi^2*R*a^2.
        R, a = sx * 0.75, sx * 0.25
        volume = 2.0 * 3.141592653589793 ** 2 * R * a * a
    else:                                    # cube, and anything box-shaped
        volume = 8.0 * sx * sy * sz
    return float(MATERIALS[name].density * volume / MASS_SCALE)


def appearance(name: str, rng):
    """`(rgb, roughness, metallic, specular, transmission, ior)` for a material.

    The hue is free unless the material bounds it; saturation and value always
    come from the material's band. That is what stops `prompts.color_name` --
    which buckets on hue and documents its assumption that s and v are fixed --
    from calling a near-black rubber puck "green": the caption path is given the
    material name instead, and the hue only names a tint within it.
    """
    import colorsys

    m = MATERIALS[name]
    if m.hue_range is None:
        hue = float(rng.uniform(0.0, 1.0))
    else:
        hue = float(rng.uniform(*m.hue_range))
    sat = float(rng.uniform(*m.saturation))
    val = float(rng.uniform(*m.value))
    rgb = tuple(float(c) for c in colorsys.hsv_to_rgb(hue, sat, val))
    return (rgb, float(m.roughness), float(m.metallic), float(m.specular),
            float(m.transmission), float(m.ior))


#: The material a hand-tuned mass is implicitly expressed in. `ratio` masses
#: are scaled against this, so a scenario that was tuned at 0.12 kg per grain
#: keeps 0.12 kg when it draws wood and moves proportionally either side.
REFERENCE_MATERIAL = "wood"


def density_ratio(name: str) -> float:
    """How much denser than wood this material is."""
    return float(MATERIALS[name].density / MATERIALS[REFERENCE_MATERIAL].density)
