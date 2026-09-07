"""Severity fields -- docs/PLAN.md 3.4.

Six steps from simulator state to a trainable [T,H,W] field:

  1. per-body, per-frame residual r(b,t)      (residuals/laws.py)
  2. calibrate the noise floor mu, sigma      (over the VALID arm)
  3. bound it:  s = clip((r - mu) / (r_strong - mu), 0, 1)
  4. paint into pixels via seg
  5. reduce to the severity_t curve
  6. special cases (global extent, shadow, two-body families)

Note on step 3: writing the bound as (r - mu)/(r_strong - mu) is algebraically the
z-score ratio z/z_ref with sigma cancelling out. That matters in practice --
geometric residuals like penetration have sigma == 0 on valid clips, so the
explicit z form divides by zero while this form stays well defined.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np


@dataclass
class NoiseFloor:
    """Per (scenario, law) residual distribution measured on valid clips."""

    mu: float
    sigma: float
    n_samples: int
    sigma_min: float = 1e-3

    @property
    def sigma_eff(self) -> float:
        return max(self.sigma, self.sigma_min)

    def z(self, r: np.ndarray) -> np.ndarray:
        return (np.asarray(r, np.float64) - self.mu) / self.sigma_eff

    def to_dict(self, r_strong: Optional[float] = None) -> Dict[str, float]:
        out = {"mu": float(self.mu), "sigma": float(self.sigma),
               "sigma_eff": float(self.sigma_eff),
               "n_samples": int(self.n_samples)}
        # The family's residual at its `strong` bin -- the scale every severity
        # is divided by. It was reaching meta.json as null, which left a reader
        # unable to tell whether a score of 0.4 meant a weak violation or a
        # strong one measured against a large reference.
        if r_strong is not None:
            out["r_strong"] = float(r_strong)
        return out

    @staticmethod
    def calibrate(valid_residuals: Sequence[np.ndarray]) -> "NoiseFloor":
        """Median and a MAD-derived sigma, not mean and standard deviation.

        The floor is meant to capture the *typical* residual a lawful clip
        carries -- solver error, finite-difference noise -- so it has to survive
        rare enormous spikes, and lawful clips are full of them. Kubric reports
        a contact force rather than an impulse, so the momentum balance never
        closes on a real landing: a perfectly legal `drop` reads 208 on the
        frame it hits the floor. Averaged in, that put mu at 20.9 against a
        family reference of 5.5 -- a negative denominator -- and a clean
        phantom-impulse signal of 5.4 scored as exactly zero.

        Gating those frames out instead was the first attempt and it was worse:
        it fixed the airborne scenarios and silently zeroed every family staged
        on a body that rests on a table or slides down a ramp, which is in
        contact on every frame it exists. A robust estimator fixes both without
        anyone having to declare which frames to trust.

        1.4826 is the constant that makes the MAD a consistent estimator of the
        standard deviation for normally distributed data.
        """
        if not len(valid_residuals):
            return NoiseFloor(0.0, 0.0, 0)
        flat = np.concatenate([np.asarray(r, np.float64).ravel()
                               for r in valid_residuals])
        mu = float(np.median(flat))
        mad = float(np.median(np.abs(flat - mu)))
        return NoiseFloor(mu, 1.4826 * mad, int(flat.size))


def bounded_score(r: np.ndarray, floor: NoiseFloor, r_strong: float,
                  baseline: Optional[np.ndarray] = None) -> np.ndarray:
    """Step 3. How far past its lawful twin this clip's residual sits.

    `baseline` is the *same body's residual in the valid twin, frame by frame*,
    and passing it is strongly preferred over relying on the scalar noise floor.
    The twin is the control this whole dataset is built around; pooling it into
    one number throws away the part that matters whenever a lawful clip's
    residual is not stationary -- which is most of them.

    Three separate families shipped with a severity of exactly zero before this
    was per-frame. The clearest: a `drop` twin is airborne for half the
    clip and resting on the floor for the rest, and Kubric reports a contact
    *force* rather than an impulse, so its momentum residual is 0 while falling
    and about 20 while resting. No single scalar describes that. The median came
    out at 19.9 against a family reference of 5.5 -- a negative denominator --
    and a clean phantom-impulse signal of 3.9 scored as nothing at all.

    Frame by frame there is no such problem: at the frame in question the twin
    reads 0 and the clip reads 3.9, and the difference is the whole answer.

    `r_strong` is the family's residual at its `strong` bin, which is what keeps
    the score comparable across families whose units are not.
    """
    r = np.asarray(r, np.float64)
    if baseline is None:
        denom = max(float(r_strong) - floor.mu, 1e-9)
        return np.clip((r - floor.mu) / denom, 0.0, 1.0)

    # ABSOLUTE departure from the twin, not a one-sided excess. A violation can
    # move a residual either way, and half the taxonomy moves it *down*:
    # `newton1_inertia` stops a body dead, so its momentum residual falls below
    # the twin's and `max(0, r - base)` scored it as exactly nothing. Four
    # families shipped at severity 0.000 for this reason -- newton1_inertia,
    # phantom_impulse, newton2_mass and newton3_reaction, all of them on
    # `linear_momentum`.
    base = np.asarray(baseline, np.float64)

    # The floor does NOT belong in the denominator here. Subtracting the twin
    # frame by frame has already removed it, and subtracting it a second time
    # is what made this negative: `linear_momentum` has r_strong 5.5 against a
    # floor mu of 22 -- because Kubric reports contact force rather than impulse
    # and a resting body reads ~20 -- so the denominator clamped to 1e-9 and
    # every nonzero difference saturated at 1.0, flattening the severity ladder.
    denom = max(float(r_strong), 1e-9)
    return np.clip(np.abs(r - base) / denom, 0.0, 1.0)


def paint(seg: np.ndarray, score_by_body: Dict[int, np.ndarray],
          active: Optional[np.ndarray] = None,
          global_value: Optional[np.ndarray] = None) -> np.ndarray:
    """Step 4. Each pixel takes the score of the body occupying it.

    `seg` already resolves occlusion, so no depth reasoning is needed. Overlaps
    cannot occur (a pixel has one instance id), so `max` is implicit.
    `global_value` handles the no-culprit case (`global_gravity`) by filling the
    whole frame instead.
    """
    s = seg[..., 0] if seg.ndim == 4 else seg
    T = s.shape[0]
    out = np.zeros(s.shape, np.float32)

    if global_value is not None:
        out[:] = np.asarray(global_value, np.float32)[:, None, None]
    else:
        for body_id, score in score_by_body.items():
            m = s == body_id
            sc = np.asarray(score, np.float32)
            out = np.where(m, sc[:, None, None], out)

    if active is not None:
        out *= np.asarray(active, bool)[:, None, None]
    return out.astype(np.float16)


def attribute_to_evidence(score: np.ndarray, active: np.ndarray,
                          observable: np.ndarray):
    """Place each window's severity on the frames where its evidence shows.

    Returns `(gate, attributed)`: the frames the spatial annotations cover, and
    the score to paint on them.

    Three cases, and the same rule handles all of them.

    * **Nothing hidden** -- the common case. Every active frame is observable,
      so this is the identity and a shaped intervention like `antigravity`
      keeps its rise and fall rather than being flattened into a running
      maximum.
    * **Hidden frames inside the window.** A super-elastic bounce changes the
      velocity while the body is still in exactly the same place in both twins;
      a pendulum's angular momentum reverses a frame before the arc turns.
      Severity accumulated on those frames is carried to the next observable
      one, so the magnitude lands where it can be seen instead of reading 0.00
      on every frame that shows anything.
    * **No observable frame in the window at all.** The evidence arrives after
      the intervention has stopped acting -- a two-frame bounce whose
      displacement first registers a frame later. The severity spills to the
      first observable frame after the window, because the alternative is an
      invalid clip that ships with an empty mask and a zero severity field:
      technically consistent, and useless to train on.

    A window whose culprit is never observable at all -- removed while fully
    occluded -- correctly yields nothing. That is the case the observability lag
    exists to describe, and inventing a region for it would be a lie.
    """
    s = np.asarray(score, np.float64)
    act = np.asarray(active, bool)
    obs = np.asarray(observable, bool)
    T = int(s.shape[0])
    gate = np.zeros((T,), bool)
    out = np.zeros((T,), np.float64)

    t = 0
    while t < T:
        if not act[t]:
            t += 1
            continue
        end = t
        while end + 1 < T and act[end + 1]:
            end += 1

        carried, seen = 0.0, False
        for f in range(t, end + 1):
            carried = max(carried, float(s[f]))
            if obs[f]:
                out[f], gate[f], carried, seen = carried, True, 0.0, True
        if not seen:
            for f in range(end + 1, T):
                if act[f]:
                    break            # the next window owns everything from here
                if obs[f]:
                    out[f], gate[f] = carried, True
                    break
        t = end + 1
    return gate, out


def temporal_profile(severity_map: np.ndarray) -> np.ndarray:
    """Step 5. severity_t[t] == severity_map[t].max() -- a schema guarantee."""
    T = severity_map.shape[0]
    return severity_map.reshape(T, -1).max(axis=1).astype(np.float32)


def peak(residual: np.ndarray, score: np.ndarray, floor: NoiseFloor,
         law: str) -> Dict[str, object]:
    """The `peak_residual` block of meta.json."""
    f = int(np.argmax(residual))
    return {"law": law, "value": float(residual[f]),
            "z_vs_valid": float(floor.z(np.asarray([residual[f]]))[0]),
            "score": float(score[f]), "frame": f}
