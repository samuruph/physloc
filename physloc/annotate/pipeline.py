"""Annotation pipeline -- runs on the HOST side of the seam.

Reads what the container worker produced (traj_*.npz, passes_*.npz, plan.json)
and writes schema-v4 samples.

    conda activate physloc
    python -m physloc.cli annotate out/phase0/drop/0173
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Sequence

import numpy as np

from .. import injectors
from ..residuals import laws
from ..scenarios import TIERS
from ..scenarios.base import Tier
from ..residuals import energy as energy_mod
from ..prompts import compose_prompt
from ..sim.trajectory import Trajectory
from ..taxonomy import FAMILIES, SCENARIOS, domain_of
from . import difficulty as diff_mod
from . import masks as masks_mod
from . import severity as sev_mod
from . import layout
from . import movi as movi_mod
from . import windows as win_mod

SCHEMA_VERSION = layout.SCHEMA_VERSION
PASS_FILES = layout.PASSES


def _seg(passes) -> np.ndarray:
    s = passes["segmentation"]
    return s[..., 0] if s.ndim == 4 else s


def annotate_work(workdir: str, outroot: str, release: str = "physloc_v0",
                  write_video: bool = True,
                  only: Optional[Sequence[str]] = None) -> List[Dict[str, object]]:
    """Annotate the variants produced by one batched worker run.

    The worker emits one valid rollout plus N invalid variants (one per
    severity). They share a bit-identical valid render, so it is annotated once
    and every variant is scored against it.

    Pass `only` -- the variant directories this run actually produced. Several
    families share a workdir (same scenario and seed), so without it every call
    re-annotates each earlier family's variants as well: wasted work, and log
    lines attributed to the wrong family.
    """
    vroot = os.path.join(workdir, "variants")
    if not os.path.isdir(vroot):
        raise FileNotFoundError("no variants/ in %s" % workdir)
    if only:
        dirs = [d for d in only if os.path.exists(os.path.join(d, "plan.json"))]
    else:
        dirs = [os.path.join(vroot, n) for n in sorted(os.listdir(vroot))
                if os.path.exists(os.path.join(vroot, n, "plan.json"))]
    return [annotate_pair(workdir, d, outroot, release, write_video)
            for d in dirs]


def _asset_block(spec, spec_d):
    """Per-body source and licence, by what the body actually IS.

    A GSO body's licence comes from the baked manifest, which is where it is
    recorded when the source is enabled -- see `scenarios/_gso.py`. Anything
    else is a KuBasic primitive and carries Kubric's own licence.
    """
    from ..scenarios._gso import GSO_ASSETS

    by_id = {int(b.segmentation_id): b for b in spec.bodies}
    out = []
    for b in spec_d["bodies"]:
        sid = int(b["segmentation_id"])
        live = by_id.get(sid)
        aid = getattr(live, "asset_id", None) if live is not None else None
        if aid and getattr(live, "kind", None) == "gso":
            entry = GSO_ASSETS.get(aid) or {}
            out.append({"name": b["name"], "source": "gso",
                        "asset_id": aid,
                        "license": entry.get("license", "unknown"),
                        "held_out": bool(entry.get("held_out", False)),
                        "segmentation_id": sid})
        else:
            out.append({"name": b["name"], "source": "kubric_primitive",
                        "license": "Apache-2.0", "segmentation_id": sid})
    return out


def annotate_pair(workdir: str, vdir: str, outroot: str,
                  release: str = "physloc_v0",
                  write_video: bool = True) -> Dict[str, object]:
    """Turn one worker variant into a released valid/invalid sample pair."""
    with open(os.path.join(vdir, "plan.json")) as fh:
        blob = json.load(fh)
    spec_d, plan_d = blob["spec"], blob["plan"]

    scenario = spec_d["scenario"]
    seed = int(spec_d["seed"])
    tier = (Tier.from_dict(spec_d["tier_spec"]) if "tier_spec" in spec_d
            else TIERS[spec_d["tier"]])
    family = plan_d["family"]
    law_name = FAMILIES[family].law

    traj_v = Trajectory.load(os.path.join(workdir, "traj_valid.npz"))
    traj_i = Trajectory.load(os.path.join(vdir, "traj_invalid.npz"))
    pv = np.load(os.path.join(workdir, "passes_valid.npz"))
    pi = np.load(os.path.join(vdir, "passes_invalid.npz"))
    seg_v, seg_i = _seg(pv), _seg(pi)
    T = tier.num_frames
    shadow_component = family in {"shadow", "shadow_inverted", "shadow_shape"}
    shadow_v = shadow_i = None
    if shadow_component:
        required = ("shadow_strength", "shadow_source_id")
        if any(key not in pv or key not in pi for key in required):
            raise ValueError(
                "%s requires the true-Cycles shadow isolation pass; regenerate "
                "this sample with shadow_strength and shadow_source_id" % family)
        # ``shadow_source_id`` names the public actor that casts the shadow
        # (the renderer-only caster is removed from the released object table).
        # Restrict the footprint to that source instead of treating every
        # dark patch in the image as the violation.  This also makes the
        # annotation safe if a scene later gains a second shadow caster.
        source_id = int(spec_d.get("notes", {}).get("caster_id", 0))
        source_v = np.asarray(pv["shadow_source_id"], np.uint16)
        source_i = np.asarray(pi["shadow_source_id"], np.uint16)
        if source_id > 0:
            source_v = source_v == source_id
            source_i = source_i == source_id
        else:
            source_v = source_v > 0
            source_i = source_i > 0
        shadow_v = ((np.asarray(pv["shadow_strength"], np.float32)
                     > (1.0 / 255.0)) & source_v)
        shadow_i = ((np.asarray(pi["shadow_strength"], np.float32)
                     > (1.0 / 255.0)) & source_i)
        # The source object can contribute a dark contact/occlusion region to
        # the isolation difference when its projected shadow reaches its own
        # footprint.  A shadow violation is localized on the receiver, not on
        # the caster, so keep only receiver pixels in both twins.
        if source_id > 0:
            # Keep a small receiver-only moat as well as the caster footprint.
            # Anti-aliasing and contact shadows otherwise leave a one- or
            # two-pixel halo attached to the object, which reads as labelling
            # the object rather than its projected shadow at review scale.
            guard_px = 2
            shadow_v &= ~masks_mod.dilate(
                masks_mod.footprint(seg_v, [source_id]), guard_px)
            shadow_i &= ~masks_mod.dilate(
                masks_mod.footprint(seg_i, [source_id]), guard_px)

    # Pixel-level prefix identity, measured here because this is the only place
    # both renders are in memory at once. The trajectory-level check runs in the
    # worker and is exact; it passed on all 176 cells of the review sweep while
    # 29 of them rendered differently before `t_event` anyway, because the
    # divergence was in the replay, downstream of the trajectory.
    _te = int((plan_d or {}).get("t_event_frame", 0))
    prefix_diff = 0
    if _te > 0:
        prefix_diff = int((pv["rgba"][:_te] != pi["rgba"][:_te]).sum())

    # ---- rebuild the scene spec so the residual context is available -----
    from .. import scenarios as scen_mod
    # The variant index matters: camera motion is spread across a scenario's
    # variants, so re-sampling without it can reconstruct a static scene for a
    # clip that was rendered with a moving camera -- and every framing guard
    # and every camera field in `sample.json` would then describe the wrong shot.
    # And the framing attempt: a scene the worker resampled because its actors
    # left the frame is a different scene, and re-sampling attempt 0 here would
    # annotate the one that was rejected.
    spec = scen_mod.get(scenario).sample(
        seed, tier, spec_d.get("complexity", {}).get("name", "L0"),
        variant=int(spec_d.get("variant", 0)),
        attempt=int((spec_d.get("notes") or {}).get("framing_attempt", 0)))
    inj = injectors.get(family)

    causal_ids: List[int] = [int(i) for i in plan_d["causal_body_ids"]]
    # Order is the injector's, not the scene's. `causal_body_ids[0]` is the
    # violator the plan is *about* -- the body that failed to react, the half
    # that split off, the actor whose bounce gained energy -- and the residual,
    # the noise floor and `r_strong` are all measured on it. Re-deriving the
    # order by walking `spec.bodies` silently picked whichever participant the
    # scenario happened to declare first: in `pyramid_impact` that is a pyramid
    # sphere rather than the falling cube, so `superelastic` scored the wrong
    # body against the right reference and read 0.18 on a maximal violation.
    dynamic = {int(b.segmentation_id) for b in spec.bodies if not b.static}
    dynamic_ids = [i for i in causal_ids if i in dynamic]
    static_ids = [i for i in causal_ids if i not in dynamic]
    primary_id = dynamic_ids[0] if dynamic_ids else causal_ids[0]

    # The plan's notes ARE the residual context. Each injector knows what its
    # law needs -- the surface a body sank through, the frames a parabola should
    # be fitted over, which bodies count as siblings, where the light is -- and
    # passing the whole block through means adding a law never means touching
    # this file.
    ctx = dict(plan_d.get("notes", {}))
    # `energy_balance` needs mass and inertia, which live on the spec rather
    # than the trajectory. Laws that do not want it simply never read the key.
    ctx["spec"] = spec
    surface_top = ctx.get("surface_top", spec.floor_level)
    ctx["surface_top"] = surface_top
    # An explicit None means "the plan says nothing bounds this" -- `solidity`
    # sets it when every surface beneath the body has been suppressed.
    if ctx.get("support_bounds", False) is None:
        pass
    elif "support_bounds" not in ctx:
        # A raised surface is finite, and the penetration law needs to know
        # that or a body knocked off a table reads as having sunk through it.
        from ..injectors import _geom as _g
        primary_body = next((b for b in spec.bodies
                             if int(b.segmentation_id) == int(primary_id)), None)
        if primary_body is not None:
            _, _, bounds = _g.support_plane(spec, primary_body)
            if bounds is not None:
                ctx["support_bounds"] = list(bounds)

    # The free-fall law asks "is this body accelerating like gravity?", which is
    # only a fair question while nothing is holding it up. Without this gate a
    # perfectly legal bounce registers as a violation, and the noise floor
    # measured on the valid arm inherits the same spikes -- which then swamps
    # the real signal on the weak bin.
    #
    # Resting on the floor is not enough of a test. A bounce completes *between*
    # sampled frames, so the actor can be airborne on both neighbours while its
    # velocity reversed in between; the finite-difference acceleration then
    # straddles the contact and reports a huge bogus residual. So also exclude
    # frames where the vertical velocity flips upward, and their neighbours,
    # since the acceleration estimate is a central difference.
    def _unsupported(tr, body_index):
        r = float(tr.radius[body_index])
        z = tr.pos[:, body_index, 2] - r
        vz = tr.lin_vel[:, body_index, 2]
        ok = z > surface_top + 1e-3

        # A bounce is a velocity reversal *at the floor*. Testing the reversal
        # alone is wrong in an instructive way: an antigravity violation also
        # flips the actor's vertical velocity, so a reversal-only gate deletes
        # exactly the frames where the violation is strongest and reports zero
        # severity at the peak. The proximity term is what distinguishes "the
        # ground pushed back" from "the law was broken".
        #
        # The clearance allowance matters because the contact completes between
        # samples: the actor can be up to about |vz| * dt above the floor on the
        # neighbouring frames.
        clear = np.maximum(0.05, np.abs(vz) * tr.dt * 1.5)
        at_floor = (z - surface_top) < clear
        bounce = np.zeros_like(ok)
        bounce[1:] = ((vz[:-1] < -1e-3) & (vz[1:] > 1e-3)
                      & (at_floor[:-1] | at_floor[1:]))
        near = np.zeros_like(ok)
        for shift in (-1, 0, 1):
            near |= np.roll(bounce, shift)
        near[0] = bool(bounce[0])
        return (ok & ~near).astype(np.float64)

    # ---- 3.4 steps 1-3: residual, noise floor, bounded score -------------
    law = laws.get(law_name)

    def _score(body_id: int, notes: Dict) -> Dict[str, object]:
        """Residual, noise floor and bounded score for one violator body.

        Scored against the twin frame by frame, not against a pooled floor --
        see the note in `bounded_score`. The valid arm is the control; using it
        at each frame rather than averaging it into one number is what makes
        the score survive a scene whose lawful residual is not stationary.

        Families whose effect depends on the rollout measure their own
        strong-bin reference at plan time and record it in the notes; the rest
        answer from the spec. Either way it is the *strong* bin's value even on
        a weak clip, or the three bins would not be comparable. A violator
        planned on its own clock brings its own notes, which is where its own
        reference lives.
        """
        here = dict(ctx, **(notes or {}))
        bi_v, bi_i = traj_v.index_of(int(body_id)), traj_i.index_of(int(body_id))
        r_v = law(traj_v, bi_v, dict(here, unsupported=_unsupported(traj_v, bi_v)))
        r_i = law(traj_i, bi_i, dict(here, unsupported=_unsupported(traj_i, bi_i)))
        fl = sev_mod.NoiseFloor.calibrate([r_v])
        strong = here.get("r_strong") or inj.strong_residual_reference(spec)
        return {"r_valid": r_v, "r_invalid": r_i, "floor": fl, "r_strong": strong,
                "s_invalid": sev_mod.bounded_score(r_i, fl, strong, baseline=r_v),
                "s_valid": sev_mod.bounded_score(r_v, fl, strong, baseline=r_v)}

    primary = _score(primary_id, {})
    # A granular medium is the whole mass, not one representative bead.  The
    # masks already union every causal grain; scoring only ``primary_id`` made
    # the scalar severity and every painted grain inherit whichever collision
    # happened to that one bead.  Aggregate per-body scores by physical mass so
    # a large affected share raises the value and an isolated outlier cannot
    # speak for the pour.
    if (getattr(spec, "physics_medium", "rigid") == "granular"
            and not plan_d.get("violators") and len(dynamic_ids) > 1):
        medium_scores = [_score(int(bid), {}) for bid in dynamic_ids]
        weights = np.asarray([
            float(traj_i.mass[traj_i.index_of(int(bid))])
            for bid in dynamic_ids], np.float64)
        if float(weights.sum()) <= 0.0:
            weights[:] = 1.0
        weights /= float(weights.sum())

        # np.sum has no ``weights`` argument; keep the weighted reduction
        # explicit and shape-stable for both timeline arrays and scalars.
        def _weighted(key):
            values = np.stack([np.asarray(s[key], np.float64)
                               for s in medium_scores])
            return np.tensordot(weights, values, axes=(0, 0))

        rv, ri = _weighted("r_valid"), _weighted("r_invalid")
        primary = {
            "r_valid": rv, "r_invalid": ri,
            "floor": sev_mod.NoiseFloor.calibrate([rv]),
            "r_strong": float(_weighted("r_strong")),
            "s_invalid": _weighted("s_invalid"),
            "s_valid": _weighted("s_valid"),
        }
    r_valid, r_invalid = primary["r_valid"], primary["r_invalid"]
    floor, r_strong = primary["floor"], primary["r_strong"]
    s_invalid, s_valid = primary["s_invalid"], primary["s_valid"]

    # ---- 3.2 windows and timelines ---------------------------------------
    plan_windows = [tuple(w) for w in plan_d["violation_windows"]]
    active = win_mod.rasterise(plan_windows, T)
    # The two halves of `active`, shipped beside it rather than instead of it.
    # `intervening` is when we are changing something -- the colour ramping, the
    # body shrinking; `consequence` is when the scene differs as a result.
    intervening = win_mod.rasterise(
        [tuple(w) for w in plan_d.get("intervention_windows", plan_windows)], T)
    consequence = win_mod.rasterise(
        [tuple(w) for w in plan_d.get("consequence_windows", plan_windows)], T)
    # WHICH window severity is gated on depends on where the violation can be
    # seen. An `event` family is detectable only across the change -- a
    # recoloured cube is a perfectly normal cube -- so its severity lives in
    # the intervention window. A `state` family is wrong in any frame while it
    # lasts, so its severity lives in the consequence window.
    detectable = FAMILIES[family].detectable
    t_ev = int((plan_d or {}).get("t_event_frame", 0))
    frames = np.arange(T)

    # EACH VIOLATOR ON ITS OWN CLOCK. A `multi` clip whose violators were planned
    # separately carries `violators` in its plan -- each body's own moment,
    # windows and plan notes -- and everything per body below is gated on that
    # body's timeline and scored on that body's residual. Any other plan has
    # one clock: every dynamic violator shares the plan's windows and, as it
    # always has, the primary violator's score. `global_gravity` acts on the
    # whole scene and `fission` on both halves, and scoring each half alone
    # would describe a fraction of one violation.
    independent = bool(plan_d.get("violators"))

    # Observability is measured on the dynamic causal bodies only -- see the
    # note in windows.observable_frames -- and it gates the *spatial*
    # annotations, which answer "where can this be seen" rather than "when is
    # it happening". `active` remains the ground-truth timeline.
    observable_all = ((shadow_v ^ shadow_i).reshape(T, -1).any(axis=1)
                      if shadow_component else
                      win_mod.observable_frames(
                          seg_v, seg_i, dynamic_ids or causal_ids,
                          rgb_valid=pv["rgba"], rgb_invalid=pi["rgba"]))
    # A SECOND gate, for the annotations that are invalid-side only.
    # `observable` is a disagreement between the twins, so it is true on frames
    # where the violator is seen in the VALID render and not in the invalid one
    # -- whenever an intervention leaves the body where the camera cannot see
    # it. `mask_invalid` and `severity_map` have no pixels to put anywhere on
    # such a frame, so their severity waits for a frame the body is seen on.
    seen_all = (shadow_i.reshape(T, -1).any(axis=1) if shadow_component else
                masks_mod.footprint(seg_i, dynamic_ids or causal_ids).any(axis=(1, 2)))

    clocks = ([(int(c["body_id"]), c) for c in plan_d["violators"]] if independent
              else [(int(b), plan_d) for b in (dynamic_ids or [primary_id])])
    violators: List[Dict[str, object]] = []
    for bid, clock in clocks:
        wins = [tuple(w) for w in clock["violation_windows"]]
        c_active = win_mod.rasterise(wins, T)
        c_iv = win_mod.rasterise(
            [tuple(w) for w in clock.get("intervention_windows", wins)], T)
        c_cq = win_mod.rasterise(
            [tuple(w) for w in clock.get("consequence_windows", wins)], T)
        own_obs = (observable_all if shadow_component else
                   win_mod.observable_frames(seg_v, seg_i, [bid],
                                             rgb_valid=pv["rgba"],
                                             rgb_invalid=pi["rgba"]))
        if independent:
            scored_on = _score(bid, clock.get("notes") or {})
            obs = own_obs
            seen = (seen_all if shadow_component else
                    masks_mod.footprint(seg_i, [bid]).any(axis=(1, 2)))
        else:
            scored_on, obs, seen = primary, observable_all, seen_all
        scored = c_iv if detectable == "event" else c_cq
        visible, s_visible = sev_mod.attribute_to_evidence(
            scored_on["s_invalid"], scored, obs)
        visible_inv, s_inv_visible = sev_mod.attribute_to_evidence(
            scored_on["s_invalid"], scored, obs & seen)
        violators.append({
            "id": int(bid), "t_event": int(clock.get("t_event_frame", t_ev)),
            "windows": wins,
            "intervention_windows": [tuple(w) for w in
                                     clock.get("intervention_windows", wins)],
            "consequence_windows": [tuple(w) for w in
                                    clock.get("consequence_windows", wins)],
            "magnitude": float(clock.get("magnitude",
                                         plan_d["intervention"]["magnitude"])),
            "active": c_active, "intervening": c_iv, "consequence": c_cq,
            "own_obs": own_obs, "obs": obs, "score": scored_on,
            "visible": visible, "s_visible": s_visible,
            "visible_inv": visible_inv, "s_inv_visible": s_inv_visible,
        })

    # ---- 3.3 masks (the union rule) --------------------------------------
    #
    # Per violator, then combined. The union mask is gated on BOTH gates, not
    # just the two-twin one: the invalid-side gate can spill severity to a
    # later frame than the two-twin gate does, and `mask_invalid` must stay a
    # subset of the union it is carved from.
    vmask = np.zeros(seg_i.shape, bool)
    imask = np.zeros(seg_i.shape, bool)
    vids = np.zeros(seg_i.shape, np.uint16)
    for c in violators:
        if shadow_component:
            gate = np.asarray(c["visible"] | c["visible_inv"], bool)[:, None, None]
            c["vmask"] = (shadow_v | shadow_i) & gate
            c["imask"] = shadow_i & np.asarray(c["visible_inv"], bool)[:, None, None]
        else:
            c["vmask"] = masks_mod.violation_mask(seg_v, seg_i, [c["id"]],
                                                  c["visible"] | c["visible_inv"])
            c["imask"] = masks_mod.invalid_mask(seg_i, [c["id"]], c["visible_inv"])
        vmask |= c["vmask"]
        imask |= c["imask"]
        vids[c["vmask"]] = c["id"]
    # Where two violators' footprints overlap, the body actually rendered there
    # in the invalid clip owns the pixel.
    for c in violators:
        vids[c["imask"]] = c["id"]

    # Level 2 is MEASURED, not declared. `static_ids` are the participants the
    # plan named -- the floor a ball sinks through -- and to those we add every
    # body whose trajectory provably departs from the valid twin AND that
    # something carrying the violation actually reached, at or after the moment
    # it began (`_causal_touch_frames`). EVERY static body in the scene is
    # passed as a non-relay, so the floor cannot pass causality on.
    scene_static = [int(b.segmentation_id) for b in spec.bodies if b.static]
    driven = [int(b.segmentation_id) for b in spec.bodies
              if getattr(b, "scripted", False)]
    touched = _causal_touch_frames((traj_i, traj_v), causal_ids, t_ev,
                                   scene_static, driven)
    onset = _divergence_onset(traj_v, traj_i)
    affected = list(static_ids) + [
        b for b in _disturbed_bodies(traj_v, traj_i, causal_ids)
        if int(b) in touched]
    moving_affected = [int(b) for b in affected if int(b) not in static_ids]
    # A disturbed body belongs to the violator that reached it FIRST, from the
    # frame it was reached and began to behave differently.
    reach: Dict[int, tuple] = {}
    for k, c in enumerate(violators):
        own = (_causal_touch_frames((traj_i, traj_v), [c["id"]], c["t_event"],
                                    scene_static, driven)
               if independent else touched)
        for b in moving_affected:
            if b in own:
                start = max(int(own[b]), int(onset.get(b, c["t_event"])))
                if b not in reach or start < reach[b][0]:
                    reach[b] = (start, k)
    # CONSEQUENCES, so gated on the consequence window rather than the scored
    # one -- and MEASURED as well as declared: if something is still behaving
    # differently because of the violation, the causal mask still says so.
    cmask = np.zeros(seg_i.shape, np.uint8)
    cids = np.zeros(seg_i.shape, np.uint16)
    for k, c in enumerate(violators):
        owned = [b for b in moving_affected if b in reach and reach[b][1] == k]
        c["disturbed"] = owned
        diverged = _diverged_frames(
            traj_v, traj_i, ([c["id"]] if independent else list(causal_ids))
            + affected, c["t_event"], T)
        gate = (c["consequence"] | diverged) & c["obs"]
        if shadow_component:
            part = np.zeros(seg_i.shape, np.uint8)
            part[c["imask"] & gate[:, None, None]] = 1
        else:
            part = masks_mod.causal_mask(
                seg_v, seg_i, [c["id"]], list(static_ids) + owned, gate,
                static_ids=static_ids,
                secondary_active={b: (frames >= reach[b][0]) & gate for b in owned})
        second = (part == 2) & (cmask != 1)
        cmask[second], cids[second] = 2, c["id"]
        first = part == 1
        cmask[first], cids[first] = 1, c["id"]

    # ---- 3.4 steps 4-5: paint, then the temporal profile ------------------
    # Every dynamic violator is painted, each with its own gated score. INVALID
    # SIDE ONLY: `paint` reads `seg_i`, because at inference a model only has
    # the invalid video and there is nothing wrong to see at a body's lawful
    # footprint. The cost, accepted deliberately: `permanence` and `dissolve`
    # get an all-zero map once the body is gone -- except below.
    if shadow_component:
        smap = np.zeros(seg_i.shape, np.float32)
        for c in violators:
            smap = np.where(c["imask"],
                            np.asarray(c["s_inv_visible"], np.float32)[:, None, None],
                            smap)
    else:
        smap = sev_mod.paint(seg_i, {c["id"]: c["s_inv_visible"] for c in violators})

    # ONE exception, and only where the invalid side has nothing at all for
    # that violator. A body that VANISHED has no invalid footprint anywhere, so
    # its lawful footprint is the only localisation there is. ABSENT, not
    # merely hidden -- a body behind a screen also has no pixels, and painting
    # its lawful footprint would put severity where the object is not. The
    # trajectory knows which case it is, so ask it rather than the pixels.
    smap = smap.astype(np.float32)
    for c in violators:
        c["fallback"] = np.zeros(seg_i.shape, bool)
        try:
            j = traj_i.index_of(c["id"])
        except Exception:                                     # noqa: BLE001
            continue
        present = np.asarray(traj_i.present[:, j], bool)
        absent = np.zeros((T,), bool)
        n = min(T, present.shape[0])
        absent[:n] = ~present[:n]
        gone = c["visible"] & absent & ~c["imask"].any(axis=(1, 2))
        if gone.any():
            c["fallback"] = masks_mod.footprint(seg_v, [c["id"]]) & gone[:, None, None]
            smap = np.where(c["fallback"] & (smap == 0),
                            np.asarray(c["s_visible"], np.float32)[:, None, None],
                            smap)
    smap = smap.astype(np.float16)
    sev_t = sev_mod.temporal_profile(smap)

    # The clip's observable timeline. With one clock it is the twins'
    # disagreement about the violators, as it always was. With several, a frame
    # where violator A is active but hidden and violator B -- not active -- is
    # merely moving is NOT evidence of anything, so on active frames only a
    # violator's own active, observable frames count, plus wherever a violator's
    # severity was carried to.
    if independent:
        observable = np.zeros((T,), bool)
        for c in violators:
            observable |= c["own_obs"] & (c["active"] | ~active)
            observable |= c["visible"] | c["visible_inv"]
    else:
        observable = observable_all

    tinfo = win_mod.build(plan_windows, T, seg_v, seg_i, dynamic_ids or causal_ids,
                          primary_id, severity_t=sev_t, observable=observable)
    arrays = tinfo.pop("_arrays")
    arrays["severity_t"] = sev_t
    arrays["intervening"] = intervening
    arrays["consequence"] = consequence
    tinfo["intervention_windows"] = [list(w) for w in
                                     plan_d.get("intervention_windows",
                                                plan_windows)]
    tinfo["consequence_windows"] = [list(w) for w in
                                    plan_d.get("consequence_windows",
                                               plan_windows)]
    tinfo["t_intervention_end_frame"] = int(
        max(e for _, e in tinfo["intervention_windows"]))
    tinfo["t_consequence_end_frame"] = int(
        max(e for _, e in tinfo["consequence_windows"]))

    # ---- per-violator records: sample metadata and dense object tensors -------
    smap32 = smap.astype(np.float32)
    # The camera factor is the same for every body in the clip, so measure the
    # track once. `movi.camera_track` is the poses the renderer keyframed.
    pixels = float(tier.resolution * tier.resolution) or 1.0
    cam_block = {"positions": movi_mod.camera_track(spec, T)["positions"],
                 "look_at": list(spec.camera_look_at)}
    violator_meta = []
    for c in violators:
        t = int(np.clip(c["t_event"], 0, T - 1))
        after = np.flatnonzero(c["own_obs"] & (frames >= t))
        t_obs = int(after[0]) if after.size else t
        rendered = (shadow_i.reshape(T, -1).any(axis=1) if shadow_component else
                    masks_mod.footprint(seg_i, [c["id"]]).any(axis=(1, 2)))
        where = (c["imask"] if shadow_component else
                 masks_mod.footprint(seg_i, [c["id"]]) | c["fallback"])
        c["severity_t"] = np.where(where, smap32, 0.0).reshape(T, -1).max(axis=1)
        # THIS VIOLATOR'S OWN DIFFICULTY, on its own evidence: its mask at its
        # biggest, how much of its own window it spends fully hidden, how long
        # it is observable, and its own residual. A `multi` clip's violators
        # differ -- one large and obvious, one behind a screen -- and one label
        # for the clip cannot say that. The clip keeps its own label; this is
        # the per-object one an object detector is scored against.
        own_peak = sev_mod.peak(c["score"]["r_invalid"], c["score"]["s_invalid"],
                                c["score"]["floor"], law_name)
        own_hidden = (~seen_all if shadow_component else
                      win_mod.occluded_frames(seg_i, c["id"]))
        n_active = int(c["active"].sum())
        own_values = diff_mod.violator_values(
            area=float(where.reshape(T, -1).sum(axis=1).max()) / pixels,
            occlusion=(float((own_hidden & c["active"]).sum()) / n_active
                       if n_active else 0.0),
            duration=float(c["own_obs"].sum()) / float(T) if T else 0.0,
            severity=float(own_peak.get("score") or 0.0),
            camera=cam_block)
        violator_meta.append({
            "difficulty": diff_mod.assess_violator(own_values),
            "instance_id": c["id"],
            "t_event_frame": t,
            "t_observable_frame": t_obs,
            "observability_lag_frames": int(t_obs - t),
            "violation_windows": [[int(s), int(e)] for s, e in c["windows"]],
            "intervention_windows": [[int(s), int(e)]
                                     for s, e in c["intervention_windows"]],
            "consequence_windows": [[int(s), int(e)]
                                    for s, e in c["consequence_windows"]],
            "observable_windows": [[int(s), int(e)]
                                   for s, e in win_mod.to_windows(c["own_obs"])],
            "occluded_at_event": bool(~rendered[t]),
            "frames_visible_after_event": int(rendered[t:].sum()),
            "magnitude": c["magnitude"],
            "peak_residual": sev_mod.peak(c["score"]["r_invalid"],
                                          c["score"]["s_invalid"],
                                          c["score"]["floor"], law_name),
            "peak_severity": float(c["severity_t"].max()) if T else 0.0,
            "affected_instance_ids": sorted(int(b) for b in c["disturbed"]),
        })
    tinfo["violators"] = violator_meta
    tinfo["violator_timing"] = (plan_d.get("violator_timing")
                               or ("independent" if independent else "shared"))
    arrays["violator_ids"] = np.asarray([c["id"] for c in violators], np.int32)
    for key in ("active", "intervening", "consequence"):
        arrays["violator_" + key] = np.stack([c[key] for c in violators])
    arrays["violator_observable"] = np.stack([c["own_obs"] for c in violators])
    arrays["violator_occluded"] = np.stack([
        (~seen_all if shadow_component else win_mod.occluded_frames(seg_i, c["id"]))
        for c in violators])
    arrays["violator_severity_t"] = np.stack(
        [c["severity_t"] for c in violators]).astype(np.float32)
    violator_residuals = {
        "violator_ids": arrays["violator_ids"],
        "violator_r": np.stack([c["score"]["r_invalid"] for c in violators]
                              ).astype(np.float32),
        "violator_s": np.stack([c["score"]["s_invalid"] for c in violators]
                              ).astype(np.float32),
    }


    # ---- write both schema-v4 samples ------------------------------------
    from ..schema import write as schema_write

    sev_bin = plan_d["intervention"]["severity_bin"]
    # THE COMPLEXITY LEVEL IS PART OF A CLIP'S IDENTITY, so a level is a
    # directory you can hold up on its own -- copy one out, delete one, point a
    # loader at one -- without filtering a flat tree. It also cannot collide:
    # each level draws its own seed block.
    level = (spec_d.get("complexity") or {}).get("name") or "L0"
    # THE CONDITION IS IN THE PATH, because a seed is opaque. Browsing a run,
    # `0783_distractors` says what the clip is and `0783` says nothing -- and
    # the condition is the axis you most often want to compare along, so it
    # should not require opening `sample.json` to find. It sorts after the seed
    # so a scenario's variants stay in generation order.
    pair_uid = "%s/%s/%s/%04d_%s" % (release, level, scenario, seed,
                                     _condition_of(spec_d).replace("+", "-"))
    written = {}
    # The camera on every frame, as MOVi describes it: one pose track and one
    # intrinsics matrix, shared by both twins because the render path is.
    cam_track = movi_mod.camera_track(spec, T)
    K = movi_mod.intrinsics(resolution=(tier.resolution, tier.resolution))
    for label in ("valid", "invalid"):
        uid = "%s/%s" % (pair_uid, "valid" if label == "valid"
                         else "invalid_%s_%s" % (family, sev_bin))
        p = pv if label == "valid" else pi
        traj = traj_v if label == "valid" else traj_i

        seg_here = seg_v if label == "valid" else seg_i
        etrace = energy_mod.compute(traj, spec, seg=seg_here)
        energy_summary = etrace.summary()
        meta = _build_meta(release, uid, pair_uid, label, spec_d, plan_d, tier,
                           tinfo, floor, law_name, r_invalid, s_invalid, family,
                           scenario, seed, primary_id, sev_bin, prefix_diff,
                           energy_summary,
                           _instance_table(spec_d, plan_d, seg_here, spec),
                           r_strong, spec=spec,
                           camera=_camera_block(spec, spec_d, T, cam_track, K),
                           collisions=movi_mod.collisions(traj.contacts,
                                                          cam_track, K, T))
        # HOW HARD IS THIS ONE TO SEE -- measured, after the fact, from the
        # masks that were just written. Last, because it reads the finished
        # `meta` rather than any single ingredient: the footprint comes from
        # the mask, the occlusion from the invalid segmentation, and the other
        # five from fields `_build_meta` has only now assembled.
        #
        # The two array-derived values are stored beside the label so that a
        # consumer re-deriving a difficulty from `sample.json` alone gets the
        # numbers the clip was labelled with, rather than a `None` and a
        # silently different answer. `assess` returns None on a valid twin,
        # which has no violation to detect.
        if meta.get("violation"):
            meta["violation"]["difficulty_inputs"] = diff_mod.inputs_for_meta(
                vmask, seg_i, meta)
        meta["difficulty"] = diff_mod.assess(
            meta, vmask if label == "invalid" else None,
            seg_i if label == "invalid" else None)
        document, _rows, remap, side_arrays = schema_write.document_from_generation(meta)
        public_ids = [int(obj["id"]) for obj in document["objects"]]
        row = {object_id: index for index, object_id in enumerate(public_ids)}

        observations = {"segmentation": seg_here.astype(np.uint16)}
        for source, target in PASS_FILES.items():
            if source in p.files:
                observations[target] = np.asarray(p[source])

        violation_ids = np.zeros(seg_here.shape, np.uint16)
        causal_level = np.zeros(seg_here.shape, np.uint8)
        causal_ids = np.zeros(seg_here.shape, np.uint16)
        component = np.zeros(seg_here.shape, np.uint8)
        severity_pixels = np.zeros(seg_here.shape, np.float16)
        if label == "invalid":
            violation_ids = vids.astype(np.uint16)
            causal_level = cmask.astype(np.uint8)
            causal_ids = cids.astype(np.uint16)
            for old_id, new_id in remap.items():
                violation_ids = np.where(violation_ids == old_id, new_id,
                                         violation_ids).astype(np.uint16)
                causal_ids = np.where(causal_ids == old_id, new_id,
                                      causal_ids).astype(np.uint16)
            code = schema_write.COMPONENTS["shadow" if shadow_component else "body"]
            component = np.where(violation_ids > 0, code, 0).astype(np.uint8)
            severity_pixels = smap.astype(np.float16)
        violation_maps = {
            "violation_object_id": violation_ids,
            "violation_component": component,
            "causal_level": causal_level,
            "causal_source_id": causal_ids,
            "severity": severity_pixels,
        }

        N = len(public_ids)
        violation_objects = {
            "ids": np.asarray(public_ids, np.int32),
            "is_violator": np.zeros(N, bool),
            "severity": np.zeros((N, T), np.float32),
            "residual": np.zeros((N, T), np.float32),
            "score": np.zeros((N, T), np.float32),
            "affected": np.zeros((N, T), bool),
        }
        violation_objects.update({key: np.zeros((N, T), bool)
                                  for key in layout.CLOCKS})
        if label == "invalid":
            for source_row, raw_id in enumerate(arrays["violator_ids"]):
                object_id = remap.get(int(raw_id), int(raw_id))
                if object_id not in row:
                    continue
                target_row = row[object_id]
                violation_objects["is_violator"][target_row] = True
                violation_objects["severity"][target_row] = arrays[
                    "violator_severity_t"][source_row]
                violation_objects["residual"][target_row] = violator_residuals[
                    "violator_r"][source_row]
                violation_objects["score"][target_row] = violator_residuals[
                    "violator_s"][source_row]
                for key in layout.CLOCKS:
                    violation_objects[key][target_row] = arrays[
                        "violator_" + key][source_row]
                record = violator_meta[source_row]
                for affected_id in record.get("affected_instance_ids") or []:
                    if int(affected_id) not in row:
                        continue
                    for start, end in record.get("consequence_windows") or []:
                        violation_objects["affected"][row[int(affected_id)],
                                                       int(start):int(end) + 1] = True

        instance_arrays = movi_mod.instance_arrays(spec, traj, seg_here, cam_track, K)
        energy = etrace.to_npz()
        state = energy_mod.body_state(traj, spec)
        for key in ("mass", "inertia", "kinetic", "potential", "height",
                    "momentum", "momentum_magnitude", "angular_momentum"):
            if key in state:
                energy[key] = state[key]
        cdir = schema_write.write_sample(
            outroot, document, observations, instance_arrays, energy,
            energy_mod.energy_map(etrace, seg_here), violation_maps,
            violation_objects, side_arrays)
        if write_video:
            _write_mp4(p["rgba"], os.path.join(cdir, layout.RGB), tier.fps)
        written[label] = cdir

    return {"pair_uid": pair_uid, "family": family, "severity": sev_bin,
            "samples": written,
            "t_event": tinfo["t_event_frame"],
            "t_observable": tinfo["t_observable_frame"],
            "observability_lag": tinfo["observability_lag_frames"],
            "violation_windows": tinfo["violation_windows"],
            "observable_windows": tinfo["observable_windows"],
            "peak_residual": float(r_invalid.max()),
            # The raw bounded score, and the one that actually ships. They
            # differ whenever the residual peaks on a frame no camera can see,
            # and reporting only the raw one hid a clip whose severity field
            # was entirely zero behind a cheerful "peak_s=1.00".
            "peak_score": float(s_invalid.max()),
            "peak_severity": float(sev_t.max()),
            "mask": masks_mod.summarise(vmask),
            "noise_floor": floor.to_dict(r_strong)}


def _diverged_frames(traj_v, traj_i, body_ids, t_event: int, T: int,
                     tol: float = 1e-3) -> np.ndarray:
    """[T] bool: frames at or after `t_event` where any of these bodies has
    provably departed from its lawful path.

    The measured half of the causal gate. A violation whose consequences outlive
    its declared window still has consequences, and this is how the annotation
    finds out: two trajectories, one comparison, no family-specific rule.
    """
    out = np.zeros((int(T),), bool)
    wanted = {int(i) for i in body_ids}
    for j, bid in enumerate(np.asarray(traj_v.body_ids, int)):
        if int(bid) not in wanted:
            continue
        a = np.asarray(traj_v.pos[:, j, :], np.float64)
        b = np.asarray(traj_i.pos[:, j, :], np.float64)
        if a.shape != b.shape:
            continue
        n = min(out.shape[0], a.shape[0])
        out[:n] |= (np.abs(a[:n] - b[:n]).max(axis=1) > tol)
    out[:max(0, int(t_event))] = False
    return out


def _disturbed_bodies(traj_v, traj_i, causal_ids, tol: float = 1e-3) -> List[int]:
    """Bodies that moved differently from the valid twin without being violators.

    A collision the intervention prevented leaves the body that was going to be
    struck on a different path; a body shoved by a fissioned half likewise. Both
    are consequences of the violation and belong in `causal_mask` level 2, and
    neither is something a plan can enumerate in advance.
    """
    violators = {int(i) for i in causal_ids}
    out: List[int] = []
    for j, bid in enumerate(np.asarray(traj_v.body_ids, int)):
        if int(bid) in violators:
            continue
        a = np.asarray(traj_v.pos[:, j, :], np.float64)
        b = np.asarray(traj_i.pos[:, j, :], np.float64)
        if a.shape == b.shape and float(np.abs(a - b).max()) > tol:
            out.append(int(bid))
    return out


def _causal_touch_frames(trajs, violator_ids, t_event: int,
                         static_ids, driven_ids=()) -> Dict[int, int]:
    """{body_id: the frame it first became causally connected to a violator}.

    **Touched, and touched AFTER the violation started.** `_disturbed_bodies`
    answers "did this body move differently", which is necessary and nowhere
    near sufficient: in a `multi` or `distractors` scene the solver's own
    divergence moves bodies the violator never came near, and they were painted
    blue. Your rule, and it is the right one -- a body is a consequence only
    once something carrying the violation has actually reached it.

    It also settles the ordering question: contacts BEFORE `t_event` are
    skipped, because a collision that already happened is a lawful collision
    and the body it struck is not a consequence of anything.

    Causality travels through MOVING bodies only. A violator resting on the
    floor makes the floor a contact of a violator, and if the floor could pass
    causality on then every object standing on it would be a consequence --
    which is the whole scene. A static body can BE reached (that is how a
    surface being passed through gets its mask) and cannot relay.

    **BOTH TWINS**, because a PREVENTED collision is a consequence too. On
    `collision x permanence` the striker is removed and the target is never
    touched in the invalid clip -- yet it is affected precisely BY not being
    struck, and it is the clearest consequence in the scene. Reading only the
    invalid contacts silently dropped it, and with it every family whose whole
    content is a collision that did not happen. The valid twin says when the
    contact would have been; the earlier of the two is when the body's fate was
    decided.

    One forward pass over contacts in frame order is enough for the transitive
    case, since a relay can only happen at or after the frame its relayer was
    itself reached.
    """
    stat = {int(i) for i in (static_ids or ())}
    reached = {int(i): int(t_event) for i in violator_ids}
    # A DRIVEN BODY IS A CONSEQUENCE BY CONSTRUCTION, not by contact.
    # `pendulum_swing`'s rod and `shadow_track`'s shadow have their pose
    # computed from the actor by `Scenario.rescript`, and both carry
    # `collides=False` -- they touch nothing, ever, so a contact test drops
    # them however plainly they move with the body they are drawn from.
    # Contact is simply the wrong question for a body the scene drives.
    reached.update({int(i): int(t_event) for i in (driven_ids or ())})
    for traj in trajs:
        c = getattr(traj, "contacts", None)
        if c is None or not len(getattr(c, "frame", ())):
            continue
        seen = dict(reached)
        frames = np.asarray(c.frame, int)
        for k in np.argsort(frames, kind="stable"):
            f = int(frames[k])
            if f < int(t_event):
                continue
            a, b = int(c.body_a[k]), int(c.body_b[k])
            for src, dst in ((a, b), (b, a)):
                got = seen.get(src)
                if got is None or got > f or src in stat:
                    continue
                if dst not in seen or seen[dst] > f:
                    seen[dst] = f
        for k, v in seen.items():
            if k not in reached or reached[k] > v:
                reached[k] = v
    return reached


def _divergence_onset(traj_v, traj_i, tol: float = 1e-3) -> Dict[int, int]:
    """{body_id: the first frame its path departs from the valid twin}."""
    out: Dict[int, int] = {}
    for j, bid in enumerate(np.asarray(traj_v.body_ids, int)):
        a = np.asarray(traj_v.pos[:, j, :], np.float64)
        b = np.asarray(traj_i.pos[:, j, :], np.float64)
        if a.shape != b.shape:
            continue
        moved = np.flatnonzero(np.abs(a - b).max(axis=1) > tol)
        if moved.size:
            out[int(bid)] = int(moved[0])
    return out


def _instance_table(spec_d, plan_d, seg, spec=None) -> List[Dict[str, object]]:
    """One record per body, in `spec.bodies` order -- MOVi's `instances`.

    Standard practice for a tracking dataset and the thing a segmentation map is
    useless without. `id` is the pixel value in `/observations/segmentation`; it is
    stable for the whole clip, so it doubles as the track id and there is no
    separate association step. Unlike MOVi the list is not re-sorted by
    visibility: the valid and invalid twins must agree on who is who, and a
    visibility sort orders them differently. Row `i` here is row `i` of every
    array in the `/objects` HDF5 group.

    Visibility is measured from the rendered map rather than assumed, so a body
    parked out of frame reports zero rather than looking present. Every body
    carries its own asset source and licence (non-negotiable 7).
    """
    causal = [int(i) for i in (plan_d or {}).get("causal_body_ids", [])]
    assets = ({int(a["segmentation_id"]): a for a in _asset_block(spec, spec_d)}
              if spec is not None else {})
    live = ({int(b.segmentation_id): b for b in spec.bodies}
            if spec is not None else {})
    out: List[Dict[str, object]] = []
    for b in spec_d.get("bodies", []):
        bid = int(b["segmentation_id"])
        seen = (seg == bid)
        frames = np.flatnonzero(seen.any(axis=(1, 2))) if seen.size else []
        asset = assets.get(bid, {})
        body = live.get(bid)
        out.append({
            "id": bid,
            "track_id": bid,          # ids are stable, so id == track
            "name": b["name"],
            "category": b.get("kind", "unknown"),
            "asset_id": asset.get("asset_id", b.get("kind")),
            "source": asset.get("source"),
            "license": asset.get("license"),
            "held_out": asset.get("held_out", False),
            "material": b.get("material"),
            "color": b.get("color"),
            "role": b.get("role", "unknown"),
            "scale": (list(body.draw_scale) if body is not None
                      else b.get("render_scale")),
            "static": bool(b.get("static", False)),
            "energy_eligible": bool(energy_mod.is_energy_object(body)),
            "dormant": bool(b.get("dormant", False)),
            "collides": bool(getattr(body, "collides", True)),
            "is_violator": bid in causal,
            "violator_index": causal.index(bid) if bid in causal else None,
            "mass": b.get("mass"),
            "friction": b.get("friction"),
            "restitution": b.get("restitution"),
            "bbox_frames": [int(f) for f in frames],
            "first_frame": int(frames[0]) if len(frames) else None,
            "last_frame": int(frames[-1]) if len(frames) else None,
            "frames_visible": int(len(frames)),
            "pixels_peak": int(seen.sum(axis=(1, 2)).max()) if seen.size else 0,
        })
    return out


def _camera_block(spec, spec_d: Dict, num_frames: int, track=None,
                  K=None) -> Dict[str, Any]:
    """MOVi's `camera`: lens, intrinsics, and the pose on every frame.

    Consumers need this the moment a fifth of clips move: `forward_flow`,
    `backward_flow` and `depth` stop being pure object motion under a moving
    camera, and without the track there is no way to tell the two apart.

    The poses come from `spec.camera_at` -- the same method the renderer
    keyframes from -- via `movi.camera_track`. Deriving them any other way
    would be a second implementation of the track, and an orbit interpolates
    its ANGLE rather than its endpoints, so the two would disagree by a chord.
    `positions` and `quaternions` follow Kubric's conventions exactly
    (camera-to-world, looking down -Z); `K` is Kubric's normalised intrinsics.
    PhysLoc adds `motion`, the one `look_at` every camera holds for the whole
    clip, and the `end_position` a moving camera travels to.
    """
    base = {"focal_length": movi_mod.FOCAL_LENGTH_MM,
            "sensor_width": movi_mod.SENSOR_WIDTH_MM,
            "field_of_view": movi_mod.field_of_view_deg()}
    if spec is None:
        return dict(base, motion=spec_d.get("camera_motion_kind", "static"),
                    position=spec_d.get("camera_position"),
                    look_at=spec_d.get("camera_look_at"),
                    end_position=spec_d.get("camera_end_position"),
                    K=None, positions=[], quaternions=[])
    track = track or movi_mod.camera_track(spec, num_frames)
    if K is None:
        K = movi_mod.intrinsics(resolution=(spec.tier.resolution,) * 2)
    return dict(
        base,
        K=np.asarray(K, np.float64).tolist(),
        positions=np.round(track["positions"], 6).tolist(),
        quaternions=np.round(track["quaternions"], 6).tolist(),
        motion=spec.camera_motion_kind,
        position=list(spec.camera_position),
        look_at=list(spec.camera_look_at),
        end_position=(list(spec.camera_end_position)
                      if spec.camera_moves else None))


def _params_block():
    """The generation knobs recorded in each schema-v4 sample."""
    from .. import params

    return params.CURRENT


def _condition_of(spec_d) -> str:
    """The condition a clip carries.

    Read off the SPEC, which resolved it once. It used to be recomputed here
    from the variant index alone, and that stopped being enough the moment the
    condition began depending on how many variants the level was given -- the
    metadata would have said `standard` for clips that were built with
    distractors. One decider, recorded, and everything else reads it.
    """
    got = spec_d.get("condition")
    if got:
        return str(got)
    from ..scenarios.base import condition_for      # pre-condition releases

    return condition_for(int(spec_d.get("variant") or 0))


def _build_meta(release, uid, pair_uid, label, spec_d, plan_d, tier, tinfo,
                floor, law_name, r_inv, s_inv, family, scenario, seed,
                primary_id, sev_bin, prefix_diff: int = 0,
                energy_summary: Optional[Dict[str, float]] = None,
                instances: Optional[List[Dict[str, object]]] = None,
                r_strong: Optional[float] = None,
                spec=None, camera: Optional[Dict[str, object]] = None,
                collisions: Optional[List[Dict[str, object]]] = None
                ) -> Dict[str, object]:
    """Build the generator record consumed by the schema-v4 writer.

    MOVi's four: `metadata` (who the clip is and its geometry), `camera`,
    `instances` and `events`. PhysLoc's beside them: `segmentation`,
    `violation`, `difficulty`, `energy`, `noise_floor`, `provenance`,
    `real2sim`, and `files`. See docs/schema.md.
    """
    instances = instances or []
    seg_names = [("0", "background")] + [
        (str(i["id"]), i["name"]) for i in instances]
    fam = FAMILIES[family]
    notes = spec_d.get("notes") or {}
    # A VALID CLIP HAS NO FAMILY, and no single twin.
    #
    # It is shared by every family staged on this scene -- that is the whole
    # point of it, and why a release renders one valid twin per (scenario,
    # seed) rather than one per cell. Labelling it with a family recorded
    # whichever family happened to be annotated LAST: `barrier_pass/0777/valid`
    # shipped as `family: time_slip` while equally belonging to the other
    # thirteen, so filtering the index on a family returned a valid clip that
    # was not specially its own, and any per-family count was off by one.
    #
    # `pair_uid` is the join key. `valid_sample_uid` on an invalid
    # clip still points at its valid partner, which is a real one-to-one edge;
    # only the reverse direction was a fiction.
    is_valid = label == "valid"
    valid_sample_uid = uid if is_valid else "%s/valid" % pair_uid
    meta = {
        # ---- MOVi's `metadata`: identity, taxonomy, and clip geometry -------
        "metadata": {
            "schema_version": SCHEMA_VERSION,
            "sample_uid": uid, "pair_uid": pair_uid,
            "valid_sample_uid": valid_sample_uid,
            "label": label, "tier": tier.name, "release": release,
            # The token grid `loader.Sample.latent_grid` reduces to by default.
            # Not derivable from the resolution alone: 128 bins to 8, 512 to 16.
            "latent_frames": tier.latent_frames, "latent_hw": tier.latent_hw,
            "domain": None if is_valid else domain_of(family),
            "family": None if is_valid else family,
            "scenario": scenario, "seed": seed,
            # THE VARIANT INDEX IS DATA, not bookkeeping. Both orthogonal axes
            # -- camera motion and distractors -- are stratified by it, so it is
            # the only field that says why this clip got clutter and its
            # neighbour did not.
            "variant": int(spec_d.get("variant") or 0),
            # WHICH CONDITION THIS CLIP CARRIES -- standard, camera,
            # distractors, multi or camera+multi. The label a benchmark reports
            # accuracy against. Read off the spec, which resolved it once.
            "condition": _condition_of(spec_d),
            # What LANDED, not what was asked for: placement can fall short,
            # and only the crowded conditions ask for any at all.
            "n_distractors": int(notes.get("n_distractors_placed") or 0),
            # How many actors are in shot, and how many of them the plan names
            # as violators. Under `multi` a lawful MAJORITY is the point -- the
            # clip asks which objects are wrong, not whether something is.
            "n_actors": int(notes.get("n_actors")
                            or len([b for b in (spec_d.get("bodies") or [])
                                    if b.get("role") == "actor"
                                    and not b.get("dormant")])),
            "n_violators": (0 if is_valid
                           else len(plan_d.get("causal_body_ids") or [])),
            "physics_medium": SCENARIOS[scenario].physics_medium,
            "complexity": spec_d.get("complexity", {}),
            # THE KNOBS THIS CLIP WAS MADE UNDER. `configs/common.yaml` is
            # editable, which is the point -- and a tunable nobody can
            # reproduce is worse than a constant nobody can change, so the
            # resolved values ride along. See `physloc/params.py`.
            "params": _params_block(),
            # The per-scene size multiplier and, when the first scene drawn let
            # its actors leave the frame, which resampled scene this is.
            "size_scale": notes.get("size_scale"),
            "framing_attempt": int(notes.get("framing_attempt") or 0),
            "background": {
                "hdri_id": spec_d.get("hdri_id"),
                "color": (list(spec.background_color) if spec is not None
                          else None),
            },
            "intphys2_category": fam.intphys2, "likephys_domain": fam.likephys,
            "frame_rate": tier.fps, "num_frames": tier.num_frames,
            "resolution": [tier.resolution, tier.resolution],
            # Kubric's physics substeps: `worker.build_scene` sets 20 per frame.
            "step_rate": tier.fps * 20,
            "gravity": spec_d.get("gravity"),
            "prompt": compose_prompt(scenario, spec_d),
            "controls": {"is_surprising_but_valid": False,
                         "is_artifact_probe": False},
        },
        # ---- MOVi's `camera`, `instances` and `events` ----------------------
        # `motion` is what the scene actually does, and the per-frame poses
        # ship beside it so a consumer can undo the camera motion rather than
        # infer it. A static clip still gets a full-length track.
        "camera": camera if camera is not None else _camera_block(
            spec, spec_d, tier.num_frames),
        # EVERY BODY CARRIES ITS OWN LICENCE, which for two thirds of the
        # dataset is Kubric's and for L3's actors is a GSO scan's CC BY-SA --
        # a share-alike asset attributed as permissive is the one way
        # non-negotiable 7 can fail while the field is still populated.
        "instances": instances,
        "events": {"collisions": collisions or []},
        # ---- PhysLoc's blocks ----------------------------------------------
        # The label space, spelled out. A segmentation map is unusable without
        # the id -> name table beside it.
        "segmentation": {
            "encoding": "instance",
            "dtype": "uint16",
            "background_id": 0,
            "ids_are_declared": True,
            "id_to_name": dict(seg_names),
        },
        "provenance": {
            "generator_commit": os.environ.get("PHYSLOC_COMMIT", "uncommitted"),
            "kubric_image_digest": _digest(), "blender_version": "2.93.4",
            "render_seed": seed,
            # MEASURED, not asserted. This was a hardcoded `True` and the
            # validator has been checking it ever since, which is how 29 of 176
            # clips shipped with renders that differed *before* t_event
            # (`replay()` was not idempotent -- see worker._clear_animation).
            # A provenance field that cannot be false is not provenance.
            "prefix_identical_verified": bool(prefix_diff == 0),
            "prefix_differing_pixels": int(prefix_diff),
            "prefix_identical_upto_frame": tinfo["t_event_frame"],
        },
        "energy": energy_summary or {},
        "noise_floor": {law_name: floor.to_dict(r_strong)},
        "real2sim": None,
    }
    if label == "invalid":
        meta["violation"] = {
            "kind": plan_d["kind"],
            "t_event_frame": tinfo["t_event_frame"],
            "t_observable_frame": tinfo["t_observable_frame"],
            "t_end_frame": tinfo["t_end_frame"],
            "t_intervention_end_frame": tinfo["t_intervention_end_frame"],
            "t_consequence_end_frame": tinfo["t_consequence_end_frame"],
            "intervention_windows": tinfo["intervention_windows"],
            "consequence_windows": tinfo["consequence_windows"],
            "observability_lag_frames": tinfo["observability_lag_frames"],
            "violation_windows": tinfo["violation_windows"],
            "observable_windows": tinfo["observable_windows"],
            "occluded_at_event": tinfo["occluded_at_event"],
            "causal_body_ids": plan_d["causal_body_ids"],
            "spatial_extent": plan_d["spatial_extent"],
            "intervention": plan_d["intervention"],
            "consequences": [],
            # EACH VIOLATOR'S OWN CLOCK. `independent` when a `multi` clip's
            # violators were planned on moments of their own, `sync` when such a
            # clip drew one moment for all on purpose, `shared` for every plan
            # whose violators act together. The clip-level fields above are the
            # union; these are per body, in `causal_body_ids` order, and the
            # per-pixel attribution is in `/violations/maps`.
            "violator_timing": tinfo.get("violator_timing"),
            "violators": tinfo.get("violators", []),
            "peak_residual": sev_mod.peak(r_inv, s_inv, floor, law_name),
        }
    else:
        meta["violation"] = None
    return meta


def _digest() -> str:
    for p in ("docker/IMAGE_DIGEST", os.path.join(os.path.dirname(__file__),
                                                  "..", "..", "docker",
                                                  "IMAGE_DIGEST")):
        if os.path.exists(p):
            with open(p) as fh:
                return fh.read().strip().split("@")[-1]
    return "unknown"


def _write_mp4(rgba: np.ndarray, path: str, fps: int) -> Optional[str]:
    from ..viz import video
    return video.write(rgba, path, fps=fps)
