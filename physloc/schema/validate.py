"""Structural and relational validation for PhysLoc schema v4."""
from __future__ import annotations

import glob
import os
from typing import Dict, List

import numpy as np

from .. import loader

GROUPS = set(loader.GROUPS)


def validate_sample(sample_dir: str) -> List[str]:
    errors: List[str] = []

    def bad(message):
        errors.append("%s: %s" % (os.path.basename(sample_dir), message))

    json_path = os.path.join(sample_dir, loader.SAMPLE_METADATA)
    if not os.path.exists(json_path):
        return ["%s: missing %s" % (sample_dir, loader.SAMPLE_METADATA)]
    try:
        sample = loader.Sample(sample_dir)
    except Exception as exc:                                  # noqa: BLE001
        return ["%s: unreadable sample: %s" % (sample_dir, exc)]
    try:
        document = sample._document
        for block in ("sample", "video", "scene", "objects"):
            if block not in document:
                bad("sample.json missing %s" % block)
        for key in ("uid", "pair_uid", "valid_uid", "label"):
            if key not in document.get("sample", {}):
                bad("sample missing %s" % key)
        for key in ("num_frames", "fps", "resolution"):
            if key not in document.get("video", {}):
                bad("video missing %s" % key)
        info = sample.info
        uid = info.uid
        if (not uid or os.path.isabs(uid) or ".." in uid.replace("\\", "/").split("/")):
            bad("sample uid must be a safe relative path")
        if info.label not in {"valid", "invalid"}:
            bad("label must be valid or invalid")
        if not os.path.exists(os.path.join(sample_dir, loader.RGB)):
            bad("missing %s" % loader.RGB)
        if not os.path.exists(os.path.join(sample_dir, loader.DATA)):
            bad("missing %s" % loader.DATA)
            return errors

        objects = sample.objects
        ids = [int(i) for i in objects.ids]
        if 0 in ids or len(ids) != len(set(ids)):
            bad("object ids must be unique and positive; 0 is background")
        for record in objects.records:
            if record.get("analysis_group") not in GROUPS:
                bad("object %s has invalid analysis_group %r"
                    % (record.get("id"), record.get("analysis_group")))
            if record.get("role") in {"shadow", "shadow_caster"}:
                bad("renderer-only shadow caster leaked into public objects")
            for key in ("asset", "physics", "render"):
                if key not in record:
                    bad("object %s missing %s" % (record.get("id"), key))
        stored_ids = (sample._h5_read("/objects/ids") if sample._h5_has("/objects/ids")
                      else np.zeros(0, np.int32))
        if not np.array_equal(np.asarray(stored_ids, np.int32), objects.ids):
            bad("/objects/ids differs from sample.json object order")

        segmentation = sample.observations.segmentation
        frames = sample.video.num_frames
        if segmentation.ndim != 3 or segmentation.shape[0] != frames:
            bad("observations/segmentation must be [T,H,W]")
        if segmentation.dtype != np.uint16:
            bad("observations/segmentation must use uint16")
        if tuple(segmentation.shape[1:]) != sample.video.resolution:
            bad("segmentation resolution differs from video.resolution")
        stray = set(np.unique(segmentation).tolist()) - {0} - set(ids)
        if stray:
            bad("segmentation names undeclared objects %s" % sorted(stray))
        camera = sample.scene.camera
        for key in ("positions", "quaternions"):
            if len(camera[key]) != frames:
                bad("/camera/%s must have one row per frame" % key)
        instances = sample.events.collisions.get("instances")
        if instances is not None and instances.size:
            unknown = set(np.unique(instances).tolist()) - set(ids)
            if unknown:
                bad("collisions name undeclared objects %s" % sorted(unknown))

        violation = sample.violation
        has_group = sample._h5_has("/violation")
        if info.is_valid:
            if violation.present:
                bad("valid sample must omit the violation block")
            if has_group:
                bad("valid sample must not store /violation")
            return errors
        if not violation.present:
            bad("invalid sample is missing the violation block")
            return errors
        if not violation.violators:
            bad("invalid sample has no violators")
        for record in violation.violators:
            if record["id"] not in ids:
                bad("violator %s is not a declared object" % record["id"])
                continue
            if objects.records[objects.row(record["id"])].get("analysis_group") != "subject":
                bad("violator %s is not analysis_group=subject" % record["id"])
            for clock in loader.CLOCKS:
                for start, end in record["windows"][clock]:
                    if not 0 <= start <= end < frames:
                        bad("violator %s %s window [%s, %s] outside [0, %d)"
                            % (record["id"], clock, start, end, frames))
            for target in record["affected_ids"]:
                if target not in ids:
                    bad("violator %s affects undeclared object %s" % (record["id"], target))
        for key in ("object_id", "component", "causal_level", "causal_source_id"):
            path = "/violation/maps/" + key
            if not sample._h5_has(path) or sample._h5_read(path).shape != segmentation.shape:
                bad("%s must match segmentation shape" % path)
        for key in ("severity", "residual", "score"):
            value = violation[key]
            if value.shape != (len(ids), frames):
                bad("/violation/objects/%s has shape %s; expected %s"
                    % (key, value.shape, (len(ids), frames)))
        named = set(np.unique(violation.object_id).tolist()) - {0}
        allowed = set(violation.ids.tolist())
        if not named <= allowed:
            bad("violation object_id names non-violators %s" % sorted(named - allowed))
        components = set(np.unique(violation.component_map).tolist())
        if not components <= set(loader.COMPONENTS):
            bad("violation component contains unknown codes %s" % sorted(components))
        if (violation.component_map == 2).any():
            observations = sample.observations
            if "shadow_strength" not in observations \
                    or "shadow_source_id" not in observations:
                bad("shadow component requires shadow_strength and shadow_source_id")
            else:
                stray_shadow = set(np.unique(observations.shadow_source_id).tolist()) \
                    - {0} - set(ids)
                if stray_shadow:
                    bad("shadow_source_id names undeclared objects %s"
                        % sorted(stray_shadow))
    except Exception as exc:                                  # noqa: BLE001
        bad("dense annotations unreadable: %s" % exc)
    finally:
        sample.release()
    return errors


def validate_release(root: str) -> Dict[str, object]:
    paths = sorted(glob.glob(os.path.join(
        root, "samples", "**", loader.SAMPLE_METADATA), recursive=True))
    if not paths:
        return {"ok": False, "samples": 0, "pairs": 0,
                "errors": ["no schema-v%d samples under %s"
                           % (loader.SCHEMA_VERSION, root)]}
    errors: List[str] = []
    pairs: Dict[str, List[str]] = {}
    samples = []
    for path in paths:
        sample_dir = os.path.dirname(path)
        errors.extend(validate_sample(sample_dir))
        try:
            sample = loader.Sample(sample_dir)
            samples.append(sample)
            pairs.setdefault(sample.pair_uid, []).append(sample.info.label)
        except Exception:
            pass
    valid_uids = {sample.uid for sample in samples if sample.info.is_valid}
    for uid, labels in pairs.items():
        if labels.count("valid") != 1 or labels.count("invalid") < 1:
            errors.append("%s: expected 1 valid and >=1 invalid" % uid)
    for sample in samples:
        if sample.info.valid_uid not in valid_uids:
            errors.append("%s: valid_uid %s does not resolve"
                          % (sample.uid, sample.info.valid_uid))
        sample.release()
    return {"ok": not errors, "samples": len(paths), "pairs": len(pairs),
            "errors": errors}
