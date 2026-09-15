"""How hard is this violation to SEE? One label per invalid clip.

**`difficulty` is to `condition` what `peak_severity` is to `magnitude`.**
`condition` is the knob -- we asked for a moving camera, or for clutter, and
the sampler delivered it. `difficulty` is what came out: a clip built with
eight distractors whose violator still fills a quarter of the frame is not hard,
and a `standard` clip whose two-frame violation happens behind a screen is.
The project already refuses to conflate the knob with the measurement on the
severity axis; this is the same refusal on the detection axis.

## The shape, and why it is KITTI's

Seven factors, each with two published thresholds, each mapping a clip to
easy(0) / moderate(1) / hard(2). **The clip takes its WORST factor.** So a clip
is `easy` only when it is easy on every axis, and one small violation area is enough
to make it hard however clean the rest of it is.

That is KITTI's Easy/Moderate/Hard construction and it is chosen over a
weighted score for three reasons:

* **It says WHY.** `binding_factors` names the axes that set the label, so
  "hard" is never a number nobody can act on -- a model that fails on
  occlusion-bound clips and passes on area-bound ones has told you
  something.
* **The sets NEST.** easy is a subset of moderate is a subset of hard, so
  evaluating "at moderate" means every clip of rank <= 1 and the three numbers
  are comparable to each other. A weighted score gives three disjoint buckets
  whose union is the dataset and whose members share nothing.
* **A weighted sum hides trade-offs.** Averaging a tiny violation area against a
  static camera says the two cancel. They do not.

## What is NOT a factor, deliberately

**The complexity level.** L3 is harder to parse than L0 -- and it is already
its own axis, with its own share of the release and its own directory. Folding
it in here would correlate the two axes and destroy the ablation both exist
for. Report `difficulty x complexity` as a grid; that grid is the interesting
result, and it only exists if the two are measured apart.

**The family and the scenario.** Same argument: those are the taxonomy, and a
benchmark wants to know which families are hard, not to have that baked into
the label it is scoring against.

**`magnitude`.** It is the knob. `severity` below is its measured counterpart
and is the one that belongs here.

## The factors

Five of the seven read straight off `metadata.json`; `violation_area` and `occlusion`
need the rendered masks, so `annotate` measures them where the arrays are in
hand and writes the raw values into `metadata.json` beside the label. A consumer
re-deriving a difficulty never has to open an `.npz`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

#: In order. The index is the rank, and `rank <= k` is the evaluation set at
#: level k -- see the nesting argument above.
LEVELS = ("easy", "moderate", "hard")


@dataclass(frozen=True)
class Factor:
    """One axis of detection difficulty, with its two published thresholds."""

    name: str
    #: What it asks, in one line. This is what the README table prints.
    question: str
    unit: str
    #: Which end is EASIER -- "high" for footprint (a big mask is easy to see),
    #: "low" for occlusion (being hidden is not). Without this the thresholds
    #: read backwards on half the table.
    easier: str
    #: The easy/moderate boundary and the moderate/hard boundary, both on the
    #: value's own scale and both stated in the easier-is-better direction.
    easy: float
    moderate: float
    #: "fitted" (cut at the review corpus's quantiles) or "chosen" (argued from
    #: what the quantity means) -- see the note above `FACTORS`.
    basis: str = ""

    def level(self, value: Optional[float]) -> int:
        """0, 1 or 2. A missing value is `moderate` -- never silently easy.

        A factor that could not be measured must not make a clip look easier
        than it is, and must not make an otherwise clean clip hard on the
        strength of an absence. The middle is the only honest answer.
        """
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return 1
        v = float(value)
        if self.easier == "high":
            return 0 if v >= self.easy else (1 if v >= self.moderate else 2)
        return 0 if v <= self.easy else (1 if v <= self.moderate else 2)

    def describe(self) -> str:
        """The threshold pair as a reader would say it."""
        op = ">=" if self.easier == "high" else "<="
        flip = "<" if self.easier == "high" else ">"
        return "easy %s %g, moderate %s %g, hard %s %g" % (
            op, self.easy, op, self.moderate, flip, self.moderate)


#: THE TABLE. `scripts/fit_difficulty.py` reproduces every number below, and
#: they are FROZEN once published: a benchmark whose difficulty labels move
#: between releases cannot be compared with itself, so a refit is a new
#: release rather than a bug fix.
#:
#: Where each pair came from, because they were not all set the same way and a
#: reader deserves to know which. **fitted** means the tertiles of the observed
#: distribution over the 431 invalid clips of the review corpus
#: (`scripts/fit_difficulty.py`, 2026-09-09). **chosen** means the corpus could
#: not answer -- the review configs hold one window setting, so `duration`
#: barely varies in them -- and the boundary is argued from what the quantity
#: means instead. A fitted threshold is a description of this dataset; a chosen
#: one is a claim about detection, and the two age differently.
FACTORS: Sequence[Factor] = (
    # FITTED. p25 = 0.012, p50 = 0.029, p75 = 0.066 of the frame.
    Factor("violation_area",
           "how much of the frame does the violation cover, at its biggest?",
           "fraction of frame", "high", 0.05, 0.012, "fitted"),
    # FITTED, on the 15% of clips that have any occlusion at all: their median
    # is 0.67. `easy` is 0.05 rather than 0 to allow a frame of slop at the
    # edge of a window -- exact zero would make one clipped frame a demotion.
    Factor("occlusion",
           "how much of the violation happens while the violator is hidden?",
           "fraction of the violation window", "low", 0.05, 0.5, "fitted"),
    # CHOSEN. The corpus is concentrated at 0.68 -- one window setting across
    # every review config -- so its tertiles would encode the config rather
    # than the difficulty. Argued instead: on a 2.97 s release clip, under 15%
    # observable is under half a second, which is hard by any standard; under
    # 35% is a third of the evidence a full-length violation gives you. This is
    # the factor that will bite hardest at release, where `instant` families
    # and re-occlusion shorten the window.
    Factor("duration",
           "how long is the violation observable?",
           "fraction of the clip", "high", 0.35, 0.15, "chosen"),
    # FITTED, rounded. p25 = 0.57, p50 = 0.94, and a third of the corpus sits
    # at exactly 1.0, so the easy boundary is 0.90 rather than the p67 of 1.0 --
    # a threshold AT the mode puts half the mode on each side of it.
    Factor("severity",
           "how far from lawful does the physics actually get?",
           "bounded residual, 0-1", "high", 0.90, 0.40, "fitted"),
    # CHOSEN, against `EXTRA_OBJECTS` = 3-10 rather than against the corpus,
    # which is 60% `standard` and so mostly reports 1. A crowded clip draws
    # 3-10 extras, so these boundaries put the small draws in `moderate` and
    # the big ones in `hard`, which is the distinction the condition exists to
    # create.
    Factor("object_count",
           "how many objects must a model consider?",
           "count", "low", 2, 6, "chosen"),
    # CHOSEN, likewise: `multi` violates 2..N-1 of N actors.
    Factor("violators",
           "how many of them are violating?",
           "count", "low", 1, 3, "chosen"),
    # FITTED on the 24 clips that move: p10 = 0.091, median 0.136, max 0.201.
    # `easy` is 0.02 rather than 0 so that a camera which is static in intent
    # is not demoted by floating-point drift in its own keyframes.
    Factor("camera_motion",
           "how far does the camera move?",
           "path length / standoff", "low", 0.02, 0.12, "fitted"),
)

BY_NAME = {f.name: f for f in FACTORS}

#: The names three factors shipped under before they were renamed to say what
#: they measure. Clips generated earlier carry them in `difficulty.factors`,
#: `difficulty.binding_factors` and `violation.difficulty_inputs`, and a config
#: may still set their cuts under them; every reader maps through here.
RENAMED = {"footprint": "violation_area", "clutter": "object_count",
           "camera": "camera_motion",
           # Before culprits were called violators.
           "culprits": "violators"}


def canonical(name: str) -> str:
    """A factor's current name, given a current or a pre-rename one."""
    return RENAMED.get(name, name)


# --------------------------------------------------------------------- values
def _windows_frames(windows, num_frames: int) -> np.ndarray:
    """[T] bool over a list of inclusive `[lo, hi]` intervals."""
    out = np.zeros((max(0, int(num_frames)),), bool)
    for w in windows or ():
        lo, hi = int(w[0]), int(w[1])
        out[max(0, lo):min(len(out), hi + 1)] = True
    return out


def camera_travel(camera: Optional[Dict[str, object]]) -> float:
    """Path length of the camera, in units of its own standoff.

    A ratio rather than metres, because scenarios are built at different
    scales: `pour`'s box is 0.68 m across and `toss` throws several metres, and
    a metre of dolly means something different in each. Dividing by the
    distance to what the camera is aimed at makes 0.15 mean the same thing
    everywhere -- roughly "the view shifted by a seventh".

    Path length, not endpoint displacement: an orbit that returns near where it
    started still moved the whole way, and every frame of that is apparent
    motion a model has to explain away.
    """
    # MOVi's per-frame `positions`, and the one aim point every PhysLoc camera
    # holds for the whole clip -- see `SceneSpec.camera_end_position`.
    cam = camera or {}
    pos = np.asarray(cam.get("positions") or [], np.float64)
    if len(pos) < 2:
        return 0.0
    aim = np.asarray(cam.get("look_at") or [0.0, 0.0, 0.0], np.float64)
    standoff = float(np.linalg.norm(pos - aim[None, :], axis=1).mean())
    if standoff < 1e-9:
        return 0.0
    return float(np.linalg.norm(np.diff(pos, axis=0), axis=1).sum() / standoff)


def measure(meta: Dict[str, object],
            vmask: Optional[np.ndarray] = None,
            seg_invalid: Optional[np.ndarray] = None) -> Dict[str, float]:
    """The seven raw values for one INVALID clip.

    `vmask` and `seg_invalid` are optional: pass them from `annotate`, where
    they are already in memory, and omit them when re-deriving from a shipped
    `metadata.json`, which carries the two values they produce.
    """
    violation = meta.get("violation") or {}
    md = meta.get("metadata") or {}
    T = int(md.get("num_frames") or 0)
    res = md.get("resolution") or [0, 0]
    pixels = float(int(res[0]) * int(res[1])) or 1.0
    stored = (violation.get("difficulty_inputs") or {})

    # 1. VIOLATION AREA -- the peak, not the mean. A violation that is briefly
    # large is findable; one that is never large is not, and averaging over a
    # window that includes frames before it becomes visible penalises a slow
    # onset twice (the `duration` factor already prices that).
    if vmask is not None and vmask.size:
        violation_area = float(vmask.reshape(len(vmask), -1).sum(axis=1).max()
                               / pixels)
    else:
        violation_area = stored.get("violation_area", stored.get("footprint"))

    # 2. OCCLUSION -- FULLY hidden, per this project's rule that a few visible
    # actor pixels make a violation instantly observable. Measured over the
    # violation window rather than the whole clip: a violator hidden for the
    # first two seconds and then in plain sight while it misbehaves is not an
    # occluded clip.
    if seg_invalid is not None and seg_invalid.size:
        ids = [int(i) for i in (violation.get("causal_body_ids") or [])]
        active = _windows_frames(violation.get("violation_windows"), T)
        if active.any() and ids:
            present = np.isin(seg_invalid.reshape(len(seg_invalid), -1),
                              ids).any(axis=1)
            n = int(active.sum())
            occlusion = float((~present[:len(active)][active]).sum()) / n
        else:
            occlusion = 0.0
    else:
        occlusion = stored.get("occlusion")

    # 3. DURATION -- observable frames, which is the window a model actually
    # has evidence in. `violation_windows` would count frames behind an
    # occluder as time on screen.
    obs = _windows_frames(violation.get("observable_windows"), T)
    duration = float(obs.sum()) / float(T) if T else 0.0

    # 4. SEVERITY -- the bounded, measured residual. `peak_residual.score` and
    # not `.value`: the raw residual is in the law's own units and a metre of
    # free-fall error does not compare with a joule of energy anomaly.
    severity = float(((violation.get("peak_residual") or {}).get("score")) or 0.0)

    # 5/6. WHAT IS IN SHOT, and how much of it is wrong. Distractors count:
    # they are there precisely to be considered and rejected.
    #
    # A GRANULAR MEDIUM COUNTS ONCE. `pour` reports 80 actors and, for the
    # families that act on the whole medium, 80 violators -- which pinned every
    # granular clip to the top of both scales and made 63 of the corpus's 431
    # clips `hard` for a reason that has nothing to do with detection. Eighty
    # grains are one thing to attend to, not eighty candidates: nobody is asked
    # which grain is wrong. A family that picks a genuine SUBSET of the medium
    # keeps its count, because then the question really is "which ones".
    n_actors = int(md.get("n_actors") or 0)
    n_violators = int(md.get("n_violators") or 0)
    if str(md.get("physics_medium") or "") == "granular":
        bodies = 1
        if n_violators >= max(1, n_actors):
            n_violators = 1
    else:
        bodies = n_actors
    object_count = float(bodies + int(md.get("n_distractors") or 0))
    violators = float(n_violators)

    # 7. CAMERA MOTION.
    camera_motion = camera_travel(meta.get("camera"))

    return {"violation_area": violation_area, "occlusion": occlusion,
            "duration": duration, "severity": severity,
            "object_count": object_count, "violators": violators,
            "camera_motion": camera_motion}


def assess(meta: Dict[str, object],
           vmask: Optional[np.ndarray] = None,
           seg_invalid: Optional[np.ndarray] = None
           ) -> Optional[Dict[str, object]]:
    """The block that ships in `metadata.json`, or `None` for a valid clip.

    A valid twin has no violation to detect. Giving it a difficulty would put
    it in an evaluation set it does not belong to, exactly as giving it a
    `violation` block would.
    """
    if not meta.get("violation"):
        return None
    values = measure(meta, vmask, seg_invalid)
    levels = {f.name: f.level(values[f.name]) for f in FACTORS}
    rank = max(levels.values())
    return {
        "level": LEVELS[rank],
        "rank": rank,
        # WHICH AXES SET THE LABEL. The whole reason for a worst-factor rule
        # rather than a weighted score.
        "binding_factors": sorted(n for n, v in levels.items() if v == rank),
        "factors": {n: {"value": (None if values[n] is None
                                  else round(float(values[n]), 6)),
                        "level": LEVELS[v]}
                    for n, v in sorted(levels.items())},
    }


def inputs_for_meta(vmask: Optional[np.ndarray],
                    seg_invalid: Optional[np.ndarray],
                    meta: Dict[str, object]) -> Dict[str, float]:
    """The two array-derived values, to be stored so nobody needs the arrays.

    Written under `violation.difficulty_inputs` before `assess` runs, so that
    `measure` finds them on a re-derivation and returns the same numbers the
    clip was labelled with.
    """
    got = measure(dict(meta), vmask, seg_invalid)
    return {"violation_area": got["violation_area"], "occlusion": got["occlusion"]}


def evaluation_set(metas, level: str) -> List[Dict[str, object]]:
    """Every clip at or below `level` -- the nesting, made usable.

    `evaluation_set(metas, "moderate")` is the easy clips AND the moderate
    ones, which is what "AP at moderate" means in every benchmark that reports
    three numbers.
    """
    cap = LEVELS.index(level)
    out = []
    for m in metas:
        d = m.get("difficulty")
        if d and int(d["rank"]) <= cap:
            out.append(m)
    return out
