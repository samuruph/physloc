"""`physviol validate` -- docs/schema.md "Cross-checks".

Structural checks that catch the failure modes the design actually has, not
just JSON well-formedness.
"""
from __future__ import annotations

import glob
import json
import os
from typing import Dict, List

import numpy as np

from ..annotate.windows import rasterise as win_rasterise

from ..annotate import windows as win
from ..taxonomy import FAMILIES, SCENARIOS, domain_of, is_compatible


#: Largest anomaly a *valid* clip may show, as a fraction of E0. The measured
#: worst case across all 14 scenarios is `pendulum_swing` at 0.31%, which is a
#: scripted scenario driving a body kinematically; every unscripted one is under
#: 0.1%. 1% leaves room for a harder impact without admitting a real violation,
#: the smallest of which is 7%.
VALID_ENERGY_TOL = 0.01


def validate_clip(cdir: str) -> List[str]:
    errs: List[str] = []

    def bad(msg):
        errs.append("%s: %s" % (os.path.basename(cdir), msg))

    mp = os.path.join(cdir, "meta.json")
    if not os.path.exists(mp):
        return ["%s: no meta.json" % cdir]
    with open(mp) as fh:
        m = json.load(fh)

    fam = m.get("family")
    if fam not in FAMILIES:
        bad("unknown family %r" % fam)
        return errs
    if m.get("scenario") not in SCENARIOS:
        bad("unknown scenario %r" % m.get("scenario"))
    # 9. domain matches family; (scenario, family) is in the matrix
    if m.get("domain") != domain_of(fam):
        bad("domain %r != %r for family %s" % (m.get("domain"),
                                               domain_of(fam), fam))
    if not is_compatible(m.get("scenario"), fam):
        bad("(%s, %s) not in the compatibility matrix"
            % (m.get("scenario"), fam))
    # 9b. physics_medium
    pm = m.get("physics_medium")
    if pm == "fluid":
        bad("physics_medium 'fluid' is not permitted at schema v0")
    if (pm == "granular") != (m.get("scenario") == "pour"):
        bad("physics_medium %r inconsistent with scenario %r"
            % (pm, m.get("scenario")))
    # 10. every asset carries a licence
    for asset in m.get("assets", []):
        if not asset.get("license"):
            bad("asset %r has no license" % asset.get("name"))
    # 11. prefix identity verified
    if not m.get("provenance", {}).get("prefix_identical_verified"):
        bad("prefix_identical_verified is not true")

    # 11b. the energy annotation is finite, self-consistent, and -- on a valid
    # clip -- lawful. A passive scene under gravity, contacts and friction can
    # only lose mechanical energy, so a valid twin that gains any is either a
    # simulator problem or an annotation bug, and either way every severity
    # scored against that twin is suspect.
    epath = os.path.join(cdir, "energy.npz")
    if os.path.exists(epath):
        try:
            ez = np.load(epath)
            total = np.asarray(ez["total"], np.float64)
            if not np.all(np.isfinite(total)):
                bad("energy.npz total is not finite")
            by_body = np.asarray(ez["by_body"], np.float64)
            drift = np.abs(by_body.sum(axis=1) - total).max()
            scale = max(abs(float(total[0])), 1e-9)
            if drift > 1e-3 * scale:
                bad("energy by_body does not sum to total (off by %.3g J)" % drift)
            if m.get("label") == "valid":
                for key, limit in (("contact_anomaly", VALID_ENERGY_TOL),
                                   ("free_anomaly", VALID_ENERGY_TOL),
                                   ("excess_loss", VALID_ENERGY_TOL)):
                    peak = float(np.asarray(ez[key], np.float64).max())
                    if peak > limit:
                        bad("valid clip %s peaks at %.2f%% of E0 (limit %.2f%%)"
                            % (key, 100 * peak, 100 * limit))
        except Exception as exc:                              # noqa: BLE001
            bad("energy.npz unreadable: %s" % exc)

    v = m.get("violation")
    # 12. violation is null iff label == valid
    if (v is None) != (m.get("label") == "valid"):
        bad("violation/label mismatch (label=%r)" % m.get("label"))

    if v is None:
        return errs

    T = int(m["num_frames"])

    # 10b. the three window families are consistent with each other.
    #
    # `violation_windows` is where the EVIDENCE is -- the frames the mask and
    # the severity map actually cover. `intervention` is when we are actively
    # changing something; `consequence` is when the scene differs as a result,
    # and it legitimately runs on past the evidence: a teleported ball is
    # somewhere else for the rest of the clip, but only the jump is visible as
    # a violation. So consequence is NOT required to sit inside the violation
    # window -- requiring that was what forced `violation_windows` to be the
    # union, and the union is what made the mask check below unsatisfiable for
    # every `event` family.
    #
    # The intervention must still be inside it: we cannot be changing something
    # on a frame we are not calling violating.
    if v is not None:
        wins_iv = v.get("intervention_windows")
        if wins_iv:
            union = win_rasterise([tuple(w) for w in v["violation_windows"]], T)
            part = win_rasterise([tuple(w) for w in wins_iv], T)
            if bool((part & ~union).any()):
                bad("intervention_windows falls outside violation_windows")

    # 10c. the invalid-side mask is a subset of the union it was carved from.
    ipath = os.path.join(cdir, "mask_invalid.npz")
    vpath = os.path.join(cdir, "violation_mask.npz")
    if os.path.exists(ipath) and os.path.exists(vpath):
        try:
            im = np.load(ipath)["mask"].astype(bool)
            vm = np.load(vpath)["mask"].astype(bool)
            if im.shape != vm.shape:
                bad("mask_invalid and violation_mask disagree on shape")
            elif bool((im & ~vm).any()):
                bad("mask_invalid has pixels outside violation_mask")
        except Exception as exc:                              # noqa: BLE001
            bad("mask_invalid unreadable: %s" % exc)
    te, to, tend = (int(v["t_event_frame"]), int(v["t_observable_frame"]),
                    int(v["t_end_frame"]))
    # 2. ordering. Evidence cannot precede its cause, and a window cannot end
    # before it starts.
    #
    # Deliberately NOT `te <= to <= tend`. Observability arriving *after* the
    # window closes is ordinary, not a bug: a super-elastic bounce adds its
    # energy on one frame and the body has not visibly moved by the time the
    # window shuts, while a one-frame teleport can take another frame to shift
    # a pixel at 128 squared. `t_end` is when the intervention stops acting;
    # when it becomes visible is a separate question, which is the entire
    # reason there are three clocks and not one.
    if to < te:
        bad("t_observable=%d precedes t_event=%d" % (to, te))
    if tend < te:
        bad("t_end=%d precedes t_event=%d" % (tend, te))
    # 3. windows sorted, disjoint, in range; endpoints agree
    wins = [tuple(w) for w in v["violation_windows"]]
    if not wins:
        bad("no violation_windows")
    prev_end = -2
    for s, e in wins:
        if s > e:
            bad("window (%d,%d) is inverted" % (s, e))
        if s <= prev_end:
            bad("windows overlap or are unsorted at (%d,%d)" % (s, e))
        if not (0 <= s < T and 0 <= e < T):
            bad("window (%d,%d) out of range T=%d" % (s, e, T))
        prev_end = e
    if wins and (min(s for s, _ in wins) != te or max(e for _, e in wins) != tend):
        bad("t_event/t_end do not match window extremes")
    # 8. spatial_extent
    if (v.get("spatial_extent") == "global") != (fam == "global_gravity"):
        bad("spatial_extent %r inconsistent with family %s"
            % (v.get("spatial_extent"), fam))
    # 7. newton3 needs both bodies
    if fam == "newton3_reaction" and len(v.get("causal_body_ids", [])) < 2:
        bad("newton3_reaction needs >= 2 causal_body_ids")

    tl_p = os.path.join(cdir, "timelines.npz")
    vm_p = os.path.join(cdir, "violation_mask.npz")
    sm_p = os.path.join(cdir, "severity_map.npz")
    if not (os.path.exists(tl_p) and os.path.exists(vm_p) and os.path.exists(sm_p)):
        bad("missing timelines/violation_mask/severity_map")
        return errs
    tl = np.load(tl_p)
    mask = np.load(vm_p)["mask"]
    smap = np.load(sm_p)["severity"].astype(np.float32)

    # 4. active is exactly the rasterisation of the windows
    if not np.array_equal(tl["active"], win.rasterise(wins, T)):
        bad("timelines.active != rasterise(violation_windows)")
    ow = [tuple(w) for w in v["observable_windows"]]
    if not np.array_equal(tl["observable"], win.rasterise(ow, T)):
        bad("timelines.observable != rasterise(observable_windows)")
    # 5. The mask must be non-empty on every frame that is active AND
    # observable -- that is the union-rule check.
    #
    # Deliberately NOT "every active frame". When the culprit is fully occluded
    # the violation is active but has no visible extent anywhere: it is absent
    # from the invalid render *and* hidden in the valid twin, so the union of
    # both footprints is legitimately empty. An empty mask is the truthful
    # annotation there, and it is exactly the case the observability lag exists
    # to describe. Requiring pixels would force us to invent them.
    active = np.asarray(tl["active"], bool)
    observable = np.asarray(tl["observable"], bool)
    per = mask.reshape(T, -1).any(axis=1)
    empty = np.flatnonzero(active & observable & ~per)
    if empty.size:
        bad("violation_mask empty on active+observable frames %s" % empty.tolist())
    hidden = np.flatnonzero(active & ~observable)
    if hidden.size and per[hidden].any():
        bad("violation_mask non-empty on frames %s where nothing is observable"
            % np.flatnonzero(active & ~observable & per).tolist())
    # 6. severity_t == severity_map.max()
    st = np.asarray(tl["severity_t"], np.float32)
    mx = smap.reshape(T, -1).max(axis=1)
    if not np.allclose(st, mx, atol=1e-3):
        bad("severity_t != severity_map.max() (max diff %.4g)"
            % float(np.abs(st - mx).max()))
    # 7. every causal id is a body that exists in the scene.
    #
    # Deliberately checked against the *declared* assets rather than against
    # the pixels in seg.npz. A causal body can be invisible for the whole clip
    # and still be a genuine participant: seven of `pour`'s forty
    # grains never show a pixel from that camera, and a vanished body has no
    # pixels precisely because it vanished. Requiring each one to be rendered
    # would fail exactly the cases the union rule and the observability lag
    # exist to handle. Whether the violation is *visible* is check 5's job.
    declared = {int(a["segmentation_id"]) for a in m.get("assets", [])
                if "segmentation_id" in a}
    if declared:
        for cid in v.get("causal_body_ids", []):
            if int(cid) not in declared:
                bad("causal_body_id %d is not a body in this scene" % cid)
    return errs


def validate_release(root: str) -> Dict[str, object]:
    metas = sorted(glob.glob(os.path.join(root, "clips", "**", "meta.json"),
                             recursive=True))
    errs: List[str] = []
    pairs: Dict[str, List[str]] = {}
    for mp in metas:
        cdir = os.path.dirname(mp)
        errs.extend(validate_clip(cdir))
        with open(mp) as fh:
            m = json.load(fh)
        pairs.setdefault(m.get("pair_uid", "?"), []).append(m.get("label"))
    # 13. every pair has exactly one valid twin and at least one invalid.
    # Several invalid variants legitimately share a single valid clip: they come
    # from the same scenario+seed, so the valid render is bit-identical and
    # storing it once is both correct and cheaper. IntPhys 2 and LikePhys ship
    # the same one-to-many shape.
    for uid, labels in pairs.items():
        n_valid = labels.count("valid")
        n_invalid = labels.count("invalid")
        if n_valid != 1 or n_invalid < 1:
            errs.append("%s: expected 1 valid and >=1 invalid, got %d/%d"
                        % (uid, n_valid, n_invalid))
    return {"ok": not errs, "clips": len(metas), "pairs": len(pairs),
            "errors": errs}
