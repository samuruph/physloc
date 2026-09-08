"""Annotation pipeline -- runs on the HOST side of the seam.

Reads what the container worker produced (traj_*.npz, passes_*.npz, plan.json)
and writes the released clip layout of docs/PLAN.md Part 4.

    conda activate physloc
    python -m physloc.cli annotate out/phase0/drop/0173
"""
from __future__ import annotations

import json
import os
import shutil
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
from . import grids as grids_mod
from . import masks as masks_mod
from . import severity as sev_mod
from . import windows as win_mod

SCHEMA_VERSION = 0
PASS_FILES = {"depth": "depth", "forward_flow": "flow_fwd",
              "backward_flow": "flow_bwd", "normal": "normals",
              "object_coordinates": "object_coords"}


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


def annotate_pair(workdir: str, vdir: str, outroot: str,
                  release: str = "physloc_v0",
                  write_video: bool = True) -> Dict[str, object]:
    """Turn one worker variant into a released valid/invalid clip pair."""
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
    # and every camera field in `meta.json` would then describe the wrong shot.
    spec = scen_mod.get(scenario).sample(
        seed, tier, spec_d.get("complexity", {}).get("name", "L0"),
        variant=int(spec_d.get("variant", 0)))
    inj = injectors.get(family)

    causal_ids: List[int] = [int(i) for i in plan_d["causal_body_ids"]]
    # Order is the injector's, not the scene's. `causal_body_ids[0]` is the
    # culprit the plan is *about* -- the body that failed to react, the half
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
    bi_v = traj_v.index_of(primary_id)
    bi_i = traj_i.index_of(primary_id)
    law = laws.get(law_name)
    r_valid = law(traj_v, bi_v, dict(ctx, unsupported=_unsupported(traj_v, bi_v)))
    r_invalid = law(traj_i, bi_i, dict(ctx, unsupported=_unsupported(traj_i, bi_i)))

    floor = sev_mod.NoiseFloor.calibrate([r_valid])
    # Families whose effect depends on the rollout measure their own strong-bin
    # reference at plan time and record it; the rest can answer from the spec.
    # Either way it is the *strong* bin's value even on a weak clip, or the
    # three bins would not be comparable.
    r_strong = ctx.get("r_strong")
    if not r_strong:
        r_strong = inj.strong_residual_reference(spec)
    # Scored against the twin frame by frame, not against a pooled floor -- see
    # the note in `bounded_score`. The valid arm is the control; using it at
    # each frame rather than averaging it into one number is what makes the
    # score survive a scene whose lawful residual is not stationary.
    s_invalid = sev_mod.bounded_score(r_invalid, floor, r_strong,
                                      baseline=r_valid)
    s_valid = sev_mod.bounded_score(r_valid, floor, r_strong,
                                    baseline=r_valid)

    # ---- 3.2 windows and timelines ---------------------------------------
    plan_windows = [tuple(w) for w in plan_d["violation_windows"]]
    active = win_mod.rasterise(plan_windows, T)
    # The two halves of `active`, shipped beside it rather than instead of it.
    # `intervening` is when we are changing something -- the colour ramping, the
    # body shrinking; `consequence` is when the scene differs as a result. They
    # were the same array, so a colour that finished turning green at frame 14
    # was still reported as "being changed" at frame 24.
    intervening = win_mod.rasterise(
        [tuple(w) for w in plan_d.get("intervention_windows", plan_windows)], T)
    consequence = win_mod.rasterise(
        [tuple(w) for w in plan_d.get("consequence_windows", plan_windows)], T)

    # Observability is measured on the dynamic causal bodies only -- see the
    # note in windows.observable_frames -- and it gates the *spatial*
    # annotations, which answer "where can this be seen" rather than "when is
    # it happening". `active` remains the ground-truth timeline.
    observable = win_mod.observable_frames(
        seg_v, seg_i, dynamic_ids or causal_ids,
        rgb_valid=pv["rgba"], rgb_invalid=pi["rgba"])
    # WHICH window severity is gated on depends on where the violation can be
    # seen. An `event` family is detectable only across the change -- a
    # recoloured cube is a perfectly normal cube, and no frame after the ramp
    # contains the violation -- so its severity lives in the intervention
    # window. A `state` family is wrong in any frame while it lasts: a hovering
    # body is wrong whether or not you saw it rise.
    #
    # Gating everything on the union marked `colour_shift` as violating to the
    # end of the clip, which asks a model to report a violation from a frame
    # that does not contain one.
    scored = (intervening if FAMILIES[family].detectable == "event"
              else consequence)
    visible, s_visible = sev_mod.attribute_to_evidence(s_invalid, scored,
                                                       observable)

    # A SECOND gate, for the annotations that are invalid-side only.
    #
    # `observable` is a *disagreement between the twins*, so it is true on
    # frames where the culprit can be seen in the VALID render and not in the
    # invalid one -- which is exactly the case whenever an intervention leaves
    # the body somewhere the camera cannot see it. `mask_invalid` and
    # `severity_map` have no pixels to put anywhere on such a frame, so
    # `attribute_to_evidence` spends the severity on a frame that shows nothing
    # and there is none left for the frame the body re-emerges on.
    #
    # Two cells in the review sweep shipped that way: `occluder_pass` x
    # `friction` and x `phantom_impulse`, both with a healthy residual (peaking
    # at 1.00 and 0.18) and 8 to 14 observable frames, both with `mask_invalid`
    # empty on all 25 frames while the segmentation carried 89 px of culprit at
    # frames 13-16. A picture with a severity of zero is the more dangerous of
    # the two audit failures: it teaches a model that a clearly wrong clip is
    # fine.
    #
    # `observable` still gates the union mask, the clocks and the absent-body
    # fallback, because those are claims about both twins.
    seen_invalid = masks_mod.footprint(
        seg_i, dynamic_ids or causal_ids).any(axis=(1, 2))
    visible_inv, s_inv_visible = sev_mod.attribute_to_evidence(
        s_invalid, scored, observable & seen_invalid)

    # ---- 3.3 masks (the union rule) --------------------------------------
    #
    # The union mask is gated on BOTH gates, not just the two-twin one. The
    # invalid-side gate can spill the severity to a later frame than the
    # two-twin gate does -- that is the whole point of `seen_invalid`, which
    # skips frames the culprit is missing from the invalid render -- and when
    # it did, `mask_invalid` ended up with pixels on a frame `violation_mask`
    # had none on. `phantom_impulse` shipped exactly that, and it breaks the
    # subset invariant the union rule exists to guarantee.
    seen = visible | visible_inv
    vmask = masks_mod.violation_mask(seg_v, seg_i, dynamic_ids, seen)
    imask = masks_mod.invalid_mask(seg_i, dynamic_ids, visible_inv)
    rmask = masks_mod.reference_mask(seg_v, dynamic_ids)
    # Level 2 is MEASURED, not declared. `static_ids` are the participants the
    # plan named -- the floor a ball sinks through -- and to those we add every
    # body whose trajectory provably departs from the valid twin without being
    # a culprit itself. That turns "affected" from a hand-maintained list into a
    # comparison, and it is the same comparison the bystander guard runs.
    affected = list(static_ids) + _disturbed_bodies(traj_v, traj_i, causal_ids)
    # CONSEQUENCES, so gated on the consequence window rather than the scored
    # one. For an event family the scored window is the handful of frames the
    # change takes, and the causal mask was disappearing with it -- but "what
    # did this violation affect" outlives the moment of intervention. A
    # phantom impulse is delivered in two frames and the ball it launched is a
    # consequence for the rest of the clip.
    #
    # And MEASURED, not just declared. Your rule, and it is the right one: if
    # something is still behaving differently because of the violation, the
    # causal mask should still be saying so. A declared window cannot know
    # that -- `newton2_mass` declares two frames because the collision resolves
    # in two frames, and both balls then travel wrong for the rest of the clip;
    # `solidity`'s window closes when the bodies stop overlapping, and the ball
    # is out the far side of a wall for every frame after it. So the gate is the
    # union of what the plan declared and what the two trajectories actually
    # do, which is the same comparison level 2 already uses to decide *who* was
    # affected, now deciding *when* as well.
    diverged = _diverged_frames(traj_v, traj_i, causal_ids + affected,
                                int((plan_d or {}).get("t_event_frame", 0)), T)
    causal_gate = (consequence | diverged) & observable
    cmask = masks_mod.causal_mask(seg_v, seg_i, dynamic_ids, affected,
                                  causal_gate, static_ids=static_ids)
    dmap = masks_mod.divergence_map(pv["rgba"], pi["rgba"])

    # ---- 3.4 steps 4-5: paint, then the temporal profile ------------------
    # Every dynamic culprit is painted, not just the primary. `global_gravity`
    # acts on the whole scene and `fission` on both halves; painting one body
    # would leave the severity field describing a fraction of the violation.
    # INVALID SIDE ONLY. `paint` reads `seg_i`, and the union's extra pixels --
    # where the culprit sits in the valid twin but not in the invalid render --
    # used to be filled in with the same score here. That put severity on the
    # object's *lawful* footprint, which at inference is a place the model
    # cannot see anything wrong: it only ever has the invalid video, and there
    # is nothing there.
    #
    # The cost, accepted deliberately: `permanence` and `dissolve` get an
    # all-zero severity map once the body is gone. That is honest -- there is no
    # pixel evidence of severity where nothing is rendered -- and
    # `reference_mask` still says where it should have been.
    smap = sev_mod.paint(seg_i,
                         {int(b): s_inv_visible
                          for b in dynamic_ids or [primary_id]},
                         active=visible_inv)

    # ONE exception, and only where the invalid side has nothing at all. A body
    # that MOVED has pixels in the invalid render, so its lawful footprint stays
    # unpainted -- that is the case you asked about and it behaves as you asked.
    # A body that VANISHED has no invalid footprint anywhere, so invalid-only
    # leaves the whole family unscoreable: `severity_t[t] == severity_map[t].max()`
    # is a schema guarantee, so an empty map is a zero timeline, and `permanence`
    # came out at severity 0.00 on all fourteen of its cells.
    #
    # Where there is no wrong place to confuse it with, "where it should have
    # been" is the only honest localisation available, and it is exactly what
    # reference_mask carries.
    #
    # ABSENT, not merely hidden. "No pixels in the invalid render" is true of a
    # body that was removed AND of one that is behind a screen, and the two
    # deserve opposite answers: for the removed body the lawful footprint is the
    # only localisation there is, while for the hidden one it is a lie -- it
    # paints severity where the object would have been in the *valid* clip,
    # which is not where the object is. You caught this on
    # `occluder_pass x friction`: the ball had fallen behind the screen and the
    # severity field was sitting out in the open where its twin had gone.
    #
    # The trajectory knows which case it is, so ask it rather than the pixels.
    absent = np.zeros((T,), bool)
    for bid in (dynamic_ids or [primary_id]):
        try:
            j = traj_i.index_of(int(bid))
        except Exception:                                     # noqa: BLE001
            continue
        present = np.asarray(traj_i.present[:, j], bool)
        n = min(T, present.shape[0])
        absent[:n] |= ~present[:n]
    gone = visible & absent & ~imask.any(axis=(1, 2))
    if gone.any():
        fallback = rmask & gone[:, None, None]
        if fallback.any():
            smap = smap.astype(np.float32)
            broadcast = np.broadcast_to(
                s_visible.astype(np.float32)[:, None, None], smap.shape)
            smap = np.where(fallback & (smap == 0), broadcast, smap)
        smap = smap.astype(np.float16)
    sev_t = sev_mod.temporal_profile(smap)

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

    # ---- 3.6 token grids --------------------------------------------------
    g = grids_mod.reduce_all(vmask, smap.astype(np.float32),
                             tier.latent_frames, tier.latent_hw)

    # ---- write both clips -------------------------------------------------
    sev_bin = plan_d["intervention"]["severity_bin"]
    # THE COMPLEXITY LEVEL IS PART OF A CLIP'S IDENTITY, so a rung is a
    # directory you can hold up on its own -- copy one out, delete one, point a
    # loader at one -- without filtering a flat tree. It also cannot collide:
    # each rung draws its own seed block.
    level = (spec_d.get("complexity") or {}).get("name") or "L0"
    pair_uid = "%s/%s/%s/%04d" % (release, level, scenario, seed)
    written = {}
    for label in ("valid", "invalid"):
        uid = "%s/%s" % (pair_uid, "valid" if label == "valid"
                         else "invalid_%s_%s" % (family, sev_bin))
        cdir = os.path.join(outroot, "clips", uid)
        os.makedirs(cdir, exist_ok=True)
        p = pv if label == "valid" else pi
        traj = traj_v if label == "valid" else traj_i

        seg_here = seg_v if label == "valid" else seg_i
        np.savez_compressed(os.path.join(cdir, "seg.npz"),
                            seg=seg_here.astype(np.uint16))

        # Mechanical energy, on BOTH twins -- the valid clip's trace is the
        # baseline every anomaly is judged against, and shipping it means a
        # consumer never has to load the twin to know what lawful looked like.
        etrace = energy_mod.compute(traj, spec)
        np.savez_compressed(os.path.join(cdir, "energy.npz"), **etrace.to_npz())
        np.savez_compressed(
            os.path.join(cdir, "energy_map.npz"),
            energy=energy_mod.energy_map(etrace, seg_here))
        # The physical quantities the energy was computed from, DERIVED from
        # the trajectory rather than additional to it -- `traj.npz` ships beside
        # this and carries pos, quat, velocities, radius, gravity and contacts.
        #
        # Worth its own file for two reasons. `traj.mass` is the declared mass,
        # a per-body constant; `bodies.mass` is [T,B] and follows volume, which
        # is what the energy is actually computed against. And every derived
        # column here comes off the same code path as the energy trace, so the
        # two can never disagree about what a body weighed or how it was
        # spinning.
        np.savez_compressed(os.path.join(cdir, "bodies.npz"),
                            **energy_mod.body_state(traj, spec))
        energy_summary = etrace.summary()
        # Shipped on BOTH twins: the lawful footprint of the bodies the
        # violation acts on, so "where it should be" is always available
        # without having to load the other clip.
        np.savez_compressed(os.path.join(cdir, "reference_mask.npz"), mask=rmask)
        for src, dst in PASS_FILES.items():
            if src in p.files:
                np.savez_compressed(os.path.join(cdir, "%s.npz" % dst), **{dst: p[src]})
        traj.save(os.path.join(cdir, "traj.npz"))

        if label == "invalid":
            np.savez_compressed(os.path.join(cdir, "violation_mask.npz"), mask=vmask)
            np.savez_compressed(os.path.join(cdir, "mask_invalid.npz"), mask=imask)
            np.savez_compressed(os.path.join(cdir, "causal_mask.npz"), mask=cmask)
            np.savez_compressed(os.path.join(cdir, "severity_map.npz"), severity=smap)
            np.savez_compressed(os.path.join(cdir, "divergence_map.npz"),
                                divergence=dmap)
            np.savez_compressed(os.path.join(cdir, "grids.npz"), **g)
            np.savez_compressed(os.path.join(cdir, "timelines.npz"), **arrays)
            np.savez_compressed(os.path.join(cdir, "residuals.npz"),
                                r=r_invalid.astype(np.float32),
                                z=floor.z(r_invalid).astype(np.float32),
                                s=s_invalid.astype(np.float32),
                                law=np.array(law_name))
        else:
            zeros_t = np.zeros((T,), np.float32)
            np.savez_compressed(
                os.path.join(cdir, "timelines.npz"),
                active=np.zeros((T,), bool), observable=np.zeros((T,), bool),
                intervening=np.zeros((T,), bool),
                consequence=np.zeros((T,), bool),
                occluded=win_mod.occluded_frames(seg_v, primary_id),
                severity_t=zeros_t)
            np.savez_compressed(os.path.join(cdir, "residuals.npz"),
                                r=r_valid.astype(np.float32),
                                z=floor.z(r_valid).astype(np.float32),
                                s=s_valid.astype(np.float32),
                                law=np.array(law_name))

        if write_video:
            _write_mp4(p["rgba"], os.path.join(cdir, "rgb.mp4"), tier.fps)

        meta = _build_meta(release, uid, pair_uid, label, spec_d, plan_d, tier,
                           tinfo, floor, law_name, r_invalid, s_invalid, family,
                           scenario, seed, primary_id, sev_bin, prefix_diff,
                           energy_summary,
                           _instance_table(spec_d, plan_d, seg_here), r_strong,
                           spec=spec)
        with open(os.path.join(cdir, "meta.json"), "w") as fh:
            json.dump(meta, fh, indent=2, sort_keys=True)
        written[label] = cdir

    return {"pair_uid": pair_uid, "family": family, "severity": sev_bin,
            "clips": written,
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
    """Bodies that moved differently from the valid twin without being culprits.

    A collision the intervention prevented leaves the body that was going to be
    struck on a different path; a body shoved by a fissioned half likewise. Both
    are consequences of the violation and belong in `causal_mask` level 2, and
    neither is something a plan can enumerate in advance.
    """
    culprits = {int(i) for i in causal_ids}
    out: List[int] = []
    for j, bid in enumerate(np.asarray(traj_v.body_ids, int)):
        if int(bid) in culprits:
            continue
        a = np.asarray(traj_v.pos[:, j, :], np.float64)
        b = np.asarray(traj_i.pos[:, j, :], np.float64)
        if a.shape == b.shape and float(np.abs(a - b).max()) > tol:
            out.append(int(bid))
    return out


def _instance_table(spec_d, plan_d, seg) -> List[Dict[str, object]]:
    """One record per body: id, name, semantic label, role, and visibility.

    Standard practice for a tracking dataset and the thing a segmentation map is
    useless without. `id` is the pixel value in `seg.npz`; it is stable for the
    whole clip, so it doubles as the track id and there is no separate
    association step. Visibility is measured from the rendered map rather than
    assumed, so a body parked out of frame reports zero rather than looking
    present.
    """
    culprits = {int(i) for i in (plan_d or {}).get("causal_body_ids", [])}
    out: List[Dict[str, object]] = []
    for b in spec_d.get("bodies", []):
        bid = int(b["segmentation_id"])
        seen = (seg == bid)
        frames = np.flatnonzero(seen.any(axis=(1, 2))) if seen.size else []
        out.append({
            "id": bid,
            "track_id": bid,          # ids are stable, so id == track
            "name": b["name"],
            "category": b.get("kind", "unknown"),
            "material": b.get("material"),
            "color": b.get("color"),
            "role": b.get("role", "unknown"),
            "static": bool(b.get("static", False)),
            "dormant": bool(b.get("dormant", False)),
            "is_culprit": bid in culprits,
            "mass_kg": b.get("mass"),
            "friction": b.get("friction"),
            "restitution": b.get("restitution"),
            "first_frame": int(frames[0]) if len(frames) else None,
            "last_frame": int(frames[-1]) if len(frames) else None,
            "frames_visible": int(len(frames)),
            "pixels_peak": int(seen.sum(axis=(1, 2)).max()) if seen.size else 0,
        })
    return out


def _camera_block(spec, spec_d: Dict, num_frames: int) -> Dict[str, Any]:
    """Where the camera was on every frame, and whether it moved.

    Consumers need this the moment a fifth of clips move: `flow_fwd`,
    `flow_bwd` and `depth` stop being pure object motion under a moving
    camera, and without the track there is no way to tell the two apart.

    Takes the reconstructed `SceneSpec` so the per-frame poses come from
    `spec.camera_at` -- the same method the renderer keyframes from. Deriving
    them here instead would be a second implementation of the track, and an
    orbit interpolates its ANGLE rather than its endpoints, so the two would
    disagree by a chord and the shipped extrinsics would describe a camera the
    clip never had.

    Falls back to the serialised spec when no object is available, which is the
    case for an older workdir being re-annotated.
    """
    if spec is None:
        return {"motion": spec_d.get("camera_motion_kind", "static"),
                "position": spec_d.get("camera_position"),
                "look_at": spec_d.get("camera_look_at"),
                "end_position": spec_d.get("camera_end_position"),
                "intrinsics": [], "extrinsics_per_frame": []}
    poses = [spec.camera_at(f, num_frames) for f in range(num_frames)]
    return {
        "motion": spec.camera_motion_kind,
        "position": list(spec.camera_position),
        "look_at": list(spec.camera_look_at),
        "end_position": (list(spec.camera_end_position)
                         if spec.camera_moves else None),
        "intrinsics": [],
        "extrinsics_per_frame": [{"position": list(p), "look_at": list(la)}
                                 for p, la in poses],
    }


def _condition_of(spec_d) -> str:
    """The condition a clip carries, from its variant index.

    Read back rather than stored on the spec, so a clip cannot claim a
    condition the sampler did not build: the same function decides what the
    scene gets and what the metadata says it got.
    """
    from ..scenarios.base import condition_for

    return condition_for(int(spec_d.get("variant") or 0))


def _build_meta(release, uid, pair_uid, label, spec_d, plan_d, tier, tinfo,
                floor, law_name, r_inv, s_inv, family, scenario, seed,
                primary_id, sev_bin, prefix_diff: int = 0,
                energy_summary: Optional[Dict[str, float]] = None,
                instances: Optional[List[Dict[str, object]]] = None,
                r_strong: Optional[float] = None,
                spec=None) -> Dict[str, object]:
    instances = instances or []
    seg_names = [("0", "background")] + [
        (str(i["id"]), i["name"]) for i in instances]
    fam = FAMILIES[family]
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
    # `pair_uid` is the join key, and it always was. `twin_uid` on an INVALID
    # clip still points at its valid partner, which is a real one-to-one edge;
    # only the reverse direction was a fiction.
    is_valid = label == "valid"
    twin = None if is_valid else "%s/valid" % pair_uid
    meta = {
        "schema_version": SCHEMA_VERSION,
        "clip_uid": uid, "pair_uid": pair_uid, "twin_uid": twin,
        "label": label, "tier": tier.name, "release": release,
        "domain": None if is_valid else domain_of(family),
        "family": None if is_valid else family,
        "scenario": scenario, "seed": seed,
        # THE VARIANT INDEX IS DATA, not bookkeeping. Both orthogonal axes --
        # camera motion and distractors -- are stratified by it, so it is the
        # only field that says why this clip got clutter and its neighbour did
        # not.
        "variant": int(spec_d.get("variant") or 0),
        # WHICH CONDITION THIS CLIP CARRIES -- standard, camera, distractors,
        # multi or camera+multi. The label a benchmark reports accuracy
        # against, and the one field that says why this clip is harder than a
        # plain one. Derived from the variant index, so it cannot disagree with
        # what the sampler actually built.
        "condition": _condition_of(spec_d),
        # What this clip ACTUALLY has, not what its level allows. The level's
        # What LANDED, not what was asked for: placement can fall short, and
        # only the crowded conditions ask for any at all.
        "n_distractors": int((spec_d.get("notes") or {}).get(
            "n_distractors_placed") or 0),
        # How many actors are in shot, and how many of them the plan names as
        # culprits. Under `multi` a lawful MAJORITY is the point -- the clip
        # asks which objects are wrong, not whether something is.
        "n_actors": int((spec_d.get("notes") or {}).get("n_actors")
                        or len([b for b in (spec_d.get("bodies") or [])
                                if b.get("role") == "actor"
                                and not b.get("dormant")])),
        "n_culprits": (0 if is_valid
                       else len(plan_d.get("causal_body_ids") or [])),
        "physics_medium": SCENARIOS[scenario].physics_medium,
        "medium": SCENARIOS[scenario].physics_medium,
        "complexity": spec_d.get("complexity", {}),
        "hdri_id": spec_d.get("hdri_id"),
        "intphys2_category": fam.intphys2, "likephys_domain": fam.likephys,
        "fps": tier.fps, "num_frames": tier.num_frames,
        "resolution": [tier.resolution, tier.resolution],
        "prompt": compose_prompt(scenario, spec_d),
        # `motion` was hardcoded "static" while `Complexity.camera_motion`
        # separately claimed "linear" from L3 up -- one clip asserting two
        # different things. It is now what the scene actually does, and the
        # per-frame poses ship beside it so a consumer can undo the camera
        # motion rather than having to infer it. A static clip still gets a
        # full-length track, so the field never needs a special case.
        "camera": _camera_block(spec, spec_d, tier.num_frames),
        "controls": {"is_surprising_but_valid": False, "is_artifact_probe": False},
        "assets": [{"name": b["name"], "source": "kubric_primitive",
                    "license": "Apache-2.0",
                    "segmentation_id": b["segmentation_id"]}
                   for b in spec_d["bodies"]],
        # The label space, spelled out. A segmentation map is unusable without
        # the id -> name table beside it, and burying that in `assets` (which
        # exists to carry licences) made consumers reconstruct it.
        "segmentation": {
            "encoding": "instance",
            "dtype": "uint16",
            "background_id": 0,
            "ids_are_declared": True,
            "id_to_name": dict(seg_names),
        },
        "instances": instances,
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
