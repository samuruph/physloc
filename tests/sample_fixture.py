"""Small schema-v4 samples for storage, loader, export, and visual tests.

Built from a generator-shaped `meta` through `document_from_generation` and
`write_sample` -- the path generation takes -- so the fixture cannot drift from
the writer.
"""
from __future__ import annotations

import os

import numpy as np

from physloc import loader
from physloc.schema.write import document_from_generation, write_sample
from physloc.viz.video import write as write_video


def make_meta(uid, label="invalid", family="solidity", pair_uid=None,
              valid_uid=None, n_objects=2, frames=5, side=16, component=1,
              level="L0", scenario="drop", condition="standard", seed=7,
              variant=0, severity="strong", affected=()):
    """The generator record `pipeline._build_meta` would hand the writer."""
    pair_uid = pair_uid or uid.rsplit("/", 1)[0]
    ids = list(range(1, n_objects + 1))
    instances = [{"id": object_id,
                  "name": "floor" if object_id == 1 else "actor_%d" % object_id,
                  "category": "cube" if object_id == 1 else "sphere",
                  "role": "floor" if object_id == 1 else "actor",
                  "asset_id": "primitive", "source": "test", "license": "CC0",
                  "mass": 0.0 if object_id == 1 else 1.0, "scale": [1.0, 1.0, 1.0],
                  "static": object_id == 1, "energy_eligible": object_id != 1}
                 for object_id in ids]
    violation = None
    if label == "invalid":
        violation = {
            "kind": "instant", "t_event_frame": 1, "t_observable_frame": 1,
            "t_end_frame": frames - 1, "observability_lag_frames": 0,
            "violation_windows": [[1, frames - 1]],
            "observable_windows": [[1, frames - 1]],
            "intervention_windows": [[1, 1]],
            "consequence_windows": [[1, frames - 1]],
            "causal_body_ids": [ids[-1]] + list(affected),
            "spatial_extent": "local",
            "intervention": {"severity_bin": severity, "magnitude": 1.0,
                             "magnitude_unit": "m", "type": "position_set"},
            "consequences": [], "violator_timing": "shared",
            "peak_residual": {"score": 0.8, "value": 2.0, "law": "position_continuity"},
            "violators": [{"instance_id": ids[-1], "t_event_frame": 1,
                           "t_observable_frame": 1, "observability_lag_frames": 0,
                           "violation_windows": [[1, frames - 1]],
                           "intervention_windows": [[1, 1]],
                           "consequence_windows": [[1, frames - 1]],
                           "observable_windows": [[1, frames - 1]],
                           "magnitude": 1.0, "peak_severity": 0.8,
                           "affected_instance_ids": list(affected)}]}
    return {
        "metadata": {
            "sample_uid": uid, "pair_uid": pair_uid,
            "valid_sample_uid": valid_uid or (uid if label == "valid"
                                              else pair_uid + "/valid"),
            "label": label, "tier": "debug", "release": "test",
            "latent_frames": 2, "latent_hw": 1,
            "domain": None if label == "valid" else "dynamics",
            "frame_rate": 5, "num_frames": frames, "resolution": [side, side],
            "step_rate": 100, "gravity": [0, 0, -9.81], "prompt": "A test actor falls.",
            "family": None if label == "valid" else family, "scenario": scenario,
            "seed": seed, "variant": variant, "condition": condition,
            "physics_medium": "rigid", "complexity": {"name": level},
            "params": {"objects": {"extra_min": 3}}, "size_scale": 1.0,
            "framing_attempt": 0, "background": {"hdri_id": None, "color": [0, 0, 0]},
        },
        "camera": {"motion": "static", "K": np.eye(3).tolist(),
                   "positions": [[0, -5, 3]] * frames,
                   "quaternions": [[1, 0, 0, 0]] * frames,
                   "look_at": [0, 0, 0], "position": [0, -5, 3], "end_position": None},
        "instances": instances,
        "events": {"collisions": [{"instances": [1, ids[-1]], "frame": 2, "force": 3.0,
                                   "position": [0, 0, 0], "image_position": [0.5, 0.5],
                                   "contact_normal": [0, 0, 1]}]},
        "provenance": {"generator_commit": "test", "render_seed": seed},
        "violation": violation,
        "difficulty": None,
    }


def make_sample(root, uid, label="invalid", family="solidity", pair_uid=None,
                valid_uid=None, n_objects=2, frames=5, side=16, write_rgb=True,
                component=1, split="unassigned", level="L0", scenario="drop",
                condition="standard", seed=7, variant=0, severity="strong",
                affected=()):
    meta = make_meta(uid, label, family, pair_uid, valid_uid, n_objects, frames,
                     side, component, level, scenario, condition, seed, variant,
                     severity, affected)
    if component == 2:
        meta["metadata"]["family"] = "shadow" if label == "invalid" else None
    document, _rows, _remap, arrays = document_from_generation(meta, split)
    ids = np.arange(1, n_objects + 1, dtype=np.int32)
    segmentation = np.ones((frames, side, side), np.uint16)
    segmentation[:, 4:12, 4:12] = ids[-1]
    observations = {"segmentation": segmentation,
                    "depth": np.ones((frames, side, side), np.float32),
                    "forward_flow": np.zeros((frames, side, side, 2), np.float32),
                    "backward_flow": np.zeros((frames, side, side, 2), np.float32),
                    "normal": np.zeros((frames, side, side, 3), np.float32),
                    "object_coordinates": np.zeros((frames, side, side, 3), np.float32)}
    if component == 2:
        observations["shadow_strength"] = np.where(segmentation == 1, 0.5, 0).astype(np.float16)
        observations["shadow_source_id"] = np.where(segmentation == 1, ids[-1], 0).astype(np.uint16)
    instance = {"ids": ids,
                "positions": np.zeros((n_objects, frames, 3), np.float32),
                "quaternions": np.tile(np.array([1, 0, 0, 0], np.float32),
                                        (n_objects, frames, 1)),
                "velocities": np.zeros((n_objects, frames, 3), np.float32),
                "angular_velocities": np.zeros((n_objects, frames, 3), np.float32),
                "bboxes_3d": np.zeros((n_objects, frames, 8, 3), np.float32),
                "image_positions": np.zeros((n_objects, frames, 2), np.float32),
                "bboxes": np.zeros((n_objects, frames, 4), np.float32),
                "visibility": np.ones((n_objects, frames), np.int32)}
    energy = {"body_ids": ids, "by_body": np.zeros((frames, n_objects), np.float32),
              "momentum": np.zeros((frames, n_objects, 3), np.float32),
              "momentum_magnitude": np.zeros((frames, n_objects), np.float32),
              "total": np.zeros(frames, np.float32)}
    violation_id = np.zeros_like(segmentation)
    if label == "invalid":
        violation_id[1:] = np.where(segmentation[1:] == (1 if component == 2 else ids[-1]),
                                    ids[-1], 0)
    violation_maps = {"violation_object_id": violation_id,
                      "violation_component": np.where(violation_id > 0, component, 0).astype(np.uint8),
                      "causal_level": np.where(violation_id > 0, 1, 0).astype(np.uint8),
                      "causal_source_id": violation_id.copy(),
                      "severity": np.where(violation_id > 0, 0.8, 0).astype(np.float16)}
    per_object = {"ids": ids, "is_violator": np.zeros(n_objects, bool),
                  "severity": np.zeros((n_objects, frames), np.float32),
                  "residual": np.zeros((n_objects, frames), np.float32),
                  "score": np.zeros((n_objects, frames), np.float32),
                  "affected": np.zeros((n_objects, frames), bool)}
    for key in loader.CLOCKS:
        per_object[key] = np.zeros((n_objects, frames), bool)
    if label == "invalid":
        per_object["is_violator"][-1] = True
        per_object["severity"][-1, 1:] = 0.8
        per_object["score"][-1, 1:] = 0.8
        for key in ("active", "consequence", "observable"):
            per_object[key][-1, 1:] = True
        per_object["intervening"][-1, 1] = True
        for target in affected:
            per_object["affected"][int(target) - 1, 1:] = True
    sample_dir = write_sample(str(root), document, observations, instance, energy,
                              np.zeros((frames, side, side), np.float32),
                              violation_maps, per_object, arrays)
    if write_rgb:
        rgb = np.zeros((frames, side, side, 3), np.uint8)
        rgb[..., 1] = np.arange(frames, dtype=np.uint8)[:, None, None] * 20
        write_video(rgb, os.path.join(sample_dir, loader.RGB), fps=5)
    else:
        open(os.path.join(sample_dir, loader.RGB), "wb").close()
    return sample_dir


def make_pair(root, pair_uid="test/L0/drop/0007_standard", **kwargs):
    valid = make_sample(root, pair_uid + "/valid", "valid", pair_uid=pair_uid,
                        valid_uid=pair_uid + "/valid", **kwargs)
    invalid = make_sample(root, pair_uid + "/invalid_solidity_strong", "invalid",
                          pair_uid=pair_uid, valid_uid=pair_uid + "/valid", **kwargs)
    return valid, invalid
