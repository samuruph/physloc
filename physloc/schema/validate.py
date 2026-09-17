"""Structural and relational validation for PhysLoc schema v3."""
from __future__ import annotations

import glob
import os
from typing import Dict, List

import numpy as np

from .. import loader

GROUPS = {"subject", "context", "support", "background"}


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
        if "clip_properties" in sample.metadata:
            bad("metadata uses removed clip_properties; expected sample_info")
        for key in ("sample_uid", "pair_uid", "valid_sample_uid", "label",
                    "num_frames", "fps", "resolution"):
            if key not in sample.sample_info:
                bad("sample_info missing %s" % key)
        uid = str(sample.sample_info.get("sample_uid") or "")
        if (not uid or os.path.isabs(uid) or ".." in uid.replace("\\", "/").split("/")):
            bad("sample_uid must be a safe relative path")
        if sample.label not in {"valid", "invalid"}:
            bad("label must be valid or invalid")
        if not os.path.exists(os.path.join(sample_dir, loader.RGB)):
            bad("missing %s" % loader.RGB)
        if not os.path.exists(os.path.join(sample_dir, loader.DATA)):
            bad("missing %s" % loader.DATA)
            return errors

        objects = sample.object_definitions
        ids = [int(obj.get("id", 0)) for obj in objects]
        if 0 in ids or len(ids) != len(set(ids)):
            bad("object ids must be unique and positive; 0 is background")
        for obj in objects:
            if obj.get("analysis_group") not in GROUPS:
                bad("object %s has invalid analysis_group %r"
                    % (obj.get("id"), obj.get("analysis_group")))
            if obj.get("role") in {"shadow", "shadow_caster"}:
                bad("renderer-only shadow caster leaked into public objects")
            for key in ("static_fields", "temporal_info", "energy", "violation"):
                if key not in obj:
                    bad("object %s missing %s" % (obj.get("id"), key))

        segmentation = sample.segmentations
        frames = sample.num_frames
        if segmentation.ndim != 3 or segmentation.shape[0] != frames:
            bad("observations/segmentation must be [T,H,W]")
        if segmentation.dtype != np.uint16:
            bad("observations/segmentation must use uint16")
        resolution = tuple(int(value) for value in sample.sample_info.get("resolution") or ())
        if len(resolution) == 2 and tuple(segmentation.shape[1:]) != resolution:
            bad("segmentation resolution differs from sample_info.resolution")
        stray = set(np.unique(segmentation).tolist()) - {0} - set(ids)
        if stray:
            bad("segmentation names undeclared objects %s" % sorted(stray))
        if not np.array_equal(sample.object_ids, np.asarray(ids, np.int32)):
            bad("objects/ids differs from sample.json object order")

        for key in ("violation_object_id", "violation_component",
                    "causal_level", "causal_source_id", "severity"):
            value = sample.maps.get(key)
            if value is None or value.shape != segmentation.shape:
                bad("violations/maps/%s must match segmentation shape" % key)
        count = len(ids)
        for key in ("is_violator", "severity", "active", "intervening",
                    "consequence", "observable", "occluded", "affected"):
            value = sample.violation_arrays.get(key)
            expected = (count,) if key == "is_violator" else (count, frames)
            if value is None or value.shape != expected:
                bad("violations/objects/%s has shape %s; expected %s"
                    % (key, None if value is None else value.shape, expected))
        is_violator = np.asarray(
            sample.violation_arrays.get("is_violator", np.zeros(count, bool)), bool)
        for row in np.flatnonzero(is_violator):
            if objects[int(row)].get("analysis_group") != "subject":
                bad("violator %s is not analysis_group=subject" % ids[int(row)])
        named = set(np.unique(sample.violation).tolist()) - {0}
        allowed = set(np.asarray(ids)[is_violator].tolist())
        if not named <= allowed:
            bad("violation_object_id names non-violators %s" % sorted(named - allowed))
        components = set(np.unique(sample.violation_component).tolist())
        if not components <= set(range(6)):
            bad("violation_component contains unknown codes %s" % sorted(components))
        if (sample.violation_component == 2).any():
            if "shadow_strength" not in sample.observations \
                    or "shadow_source_id" not in sample.observations:
                bad("shadow component requires shadow_strength and shadow_source_id")
            else:
                source = sample.observations["shadow_source_id"]
                stray_shadow = set(np.unique(source).tolist()) - {0} - set(ids)
                if stray_shadow:
                    bad("shadow_source_id names undeclared objects %s"
                        % sorted(stray_shadow))
        for relation in sample.causal_relations:
            for key in ("source_object_id", "target_object_id"):
                if int(relation.get(key, 0)) not in set(ids):
                    bad("causal relation %s is not a declared object" % key)
        summary = (sample._document.get("annotations") or {}).get("violation_summary")
        if sample.is_valid:
            if summary is not None:
                bad("valid sample must omit violation_summary")
            if sample.violation_mask.any() or is_violator.any():
                bad("valid sample carries non-zero violation annotations")
        elif summary is None:
            bad("invalid sample is missing violation_summary")
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
                "errors": ["no schema-v3 samples under %s" % root]}
    errors: List[str] = []
    pairs: Dict[str, List[str]] = {}
    samples = []
    for path in paths:
        sample_dir = os.path.dirname(path)
        errors.extend(validate_sample(sample_dir))
        try:
            sample = loader.Sample(sample_dir)
            samples.append(sample)
            pairs.setdefault(sample.pair_uid, []).append(sample.label)
        except Exception:
            pass
    valid_uids = {sample.uid for sample in samples if sample.is_valid}
    for uid, labels in pairs.items():
        if labels.count("valid") != 1 or labels.count("invalid") < 1:
            errors.append("%s: expected 1 valid and >=1 invalid" % uid)
    for sample in samples:
        if sample.valid_sample_uid not in valid_uids:
            errors.append("%s: valid_sample_uid %s does not resolve"
                          % (sample.uid, sample.valid_sample_uid))
        sample.release()
    return {"ok": not errors, "samples": len(paths), "pairs": len(pairs),
            "errors": errors}
