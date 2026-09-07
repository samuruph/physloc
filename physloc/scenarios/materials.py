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
    # The light end, and it is cork rather than foam on purpose: real foam is
    # around 60 kg/m^3, which next to steel is a mass ratio of 130 to 1, and a
    # body that light skitters under a solver whose contact parameters were
    # tuned near 1 kg. Cork is genuinely light, still obviously light on
    # camera, and keeps the spread to a solver-friendly ~30x.
    "cork": Material("cork", 240.0, 0.90, 0.0, (0.15, 0.40), (0.55, 0.80),
                     hue_range=(0.05, 0.12)),
    # Warm, matte, mid-density. Hue is bounded because a blue plank does not
    # read as wood, and the whole point is that the material is recognisable.
    "wood": Material("wood", 650.0, 0.80, 0.0, (0.35, 0.62), (0.45, 0.75),
                     hue_range=(0.03, 0.11)),
    # The free hue: a plastic object is credibly any colour, which keeps the
    # dataset's colour variety from collapsing once materials bound the rest.
    "plastic": Material("plastic", 1100.0, 0.45, 0.0, (0.45, 0.85), (0.60, 0.92)),
    # Dark and very rough. Restitution is a body property rather than a
    # material one here, so `rubber` is about how it LOOKS and weighs; the
    # scenarios that care about bounce still set their own.
    "rubber": Material("rubber", 1300.0, 0.92, 0.0, (0.05, 0.30), (0.12, 0.30)),
    # Smooth and bright, low saturation -- porcelain rather than a toy.
    "ceramic": Material("ceramic", 2400.0, 0.25, 0.0, (0.02, 0.20), (0.75, 0.95)),
    "stone": Material("stone", 2700.0, 0.88, 0.0, (0.03, 0.18), (0.30, 0.55)),
    # The dense end, and the only metallic one. Near-zero saturation so it
    # reads as metal and not as a grey-painted object.
    "steel": Material("steel", 7800.0, 0.20, 1.0, (0.00, 0.08), (0.55, 0.80)),
}

#: The default draw. `dome`, floors and ramps are not made of anything in
#: particular, and giving them materials would put metallic backdrops in the
#: dataset for no gain.
ACTOR_MATERIALS: Tuple[str, ...] = ("cork", "wood", "plastic", "rubber",
                                    "ceramic", "stone", "steel")


def get(name: str) -> Material:
    return MATERIALS[name]


def pick(rng, choices: Tuple[str, ...] = ACTOR_MATERIALS) -> str:
    """One material name, drawn off whatever stream is handed in.

    Draw this from `appearance_rng`, never from the physics stream: it decides
    a mass, but it decides it *deterministically* from the scene's look, and
    consuming a physics draw here would shift every value after it.
    """
    return str(choices[int(rng.randint(0, len(choices)))])


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


def appearance(name: str, rng) -> Tuple[Tuple[float, float, float], float, float]:
    """`(rgb, roughness, metallic)` consistent with the material.

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
    return rgb, float(m.roughness), float(m.metallic)


#: The material a hand-tuned mass is implicitly expressed in. `ratio` masses
#: are scaled against this, so a scenario that was tuned at 0.12 kg per grain
#: keeps 0.12 kg when it draws wood and moves proportionally either side.
REFERENCE_MATERIAL = "wood"


def density_ratio(name: str) -> float:
    """How much denser than wood this material is."""
    return float(MATERIALS[name].density / MATERIALS[REFERENCE_MATERIAL].density)
