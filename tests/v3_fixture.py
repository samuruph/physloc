"""Small schema-v3 samples for storage, loader, export, and visual tests."""
from __future__ import annotations

import os

import numpy as np

from physloc import loader
from physloc.schema.write import write_sample
from physloc.viz.video import write as write_video


def make_sample(root, uid, label="invalid", family="solidity", pair_uid=None,
                valid_uid=None, n_objects=2, frames=5, side=16, write_rgb=True,
                component=1, split="unassigned", level="L0", scenario="drop",
                condition="standard", seed=7, variant=0, severity="strong"):
    pair_uid = pair_uid or uid.rsplit("/", 1)[0]
    valid_uid = valid_uid or (uid if label == "valid" else pair_uid + "/valid")
    ids = np.arange(1, n_objects + 1, dtype=np.int32)
    objects = []
    for row, object_id in enumerate(ids):
        group = "support" if row == 0 else "subject"
        objects.append({
            "id": int(object_id), "name": "floor" if row == 0 else "actor_%d" % object_id,
            "category": "cube" if row == 0 else "sphere",
            "role": "floor" if row == 0 else "actor", "analysis_group": group,
            "asset": {"id": "primitive", "source": "test", "license": "CC0"},
            "static_fields": {"mass": 0.0 if row == 0 else 1.0,
                              "dimensions": [1.0, 1.0, 1.0], "static": row == 0,
                              "collidable": True, "camera_visible": True,
                              "shadow_visible": True},
            "temporal_info": {"store": loader.DATA, "group": "/objects", "row": row},
            "energy": {"store": loader.DATA, "group": "/energy/objects", "row": row},
            "violation": {"store": loader.DATA, "group": "/violations/objects", "row": row},
        })
    scene_info = {"level": level, "type": scenario, "family": None if label == "valid" else family,
                  "domain": None if label == "valid" else "dynamics",
                  "physics_medium": "rigid", "condition": condition,
                  "complexity": level, "difficulty": None if label == "valid" else "easy",
                  "severity": None if label == "valid" else severity, "label": label,
                  "variant": variant, "prompt": "A test actor falls."}
    summary = None
    if label == "invalid":
        summary = {"kind": family, "target_component": "shadow" if component == 2 else "body",
                   "t_event_frame": 1, "t_observable_frame": 1,
                   "violation_windows": [[1, frames - 1]],
                   "observable_windows": [[1, frames - 1]],
                   "intervention": {"severity_bin": severity, "magnitude": 1.0},
                   "violators": [{"instance_id": int(ids[-1]),
                                  "target_component": "shadow" if component == 2 else "body",
                                  "violation_windows": [[1, frames - 1]],
                                  "intervention_windows": [[1, 1]],
                                  "consequence_windows": [[1, frames - 1]],
                                  "observable_windows": [[1, frames - 1]],
                                  "affected_instance_ids": []}]}
    annotations = {"objects": objects,
                   "scene_energy": {"store": loader.DATA, "group": "/energy/scene"},
                   "maps": {"store": loader.DATA, "group": "/violations/maps"},
                   "events": {"collisions": []}, "causal_relations": []}
    if summary is not None:
        annotations["violation_summary"] = summary
    document = {
        "schema_version": 3,
        "metadata": {
            "sample_info": {"schema_version": 3, "dataset_version": "test-v3",
                            "sample_uid": uid, "pair_uid": pair_uid,
                            "valid_sample_uid": valid_uid, "label": label, "split": split,
                            "seed": seed, "render_seed": seed, "variant": variant,
                            "generation_config_id": "test:debug", "num_frames": frames,
                            "fps": 5.0, "duration_seconds": frames / 5.0,
                            "resolution": [side, side], "latent_frames": 2,
                            "latent_hw": 1, "framing_attempt": 0, "size_scale": 1.0,
                            "provenance": {"generator_commit": "test"}},
            "scene": {"info": scene_info,
                      "world": {"camera": {"motion": "static", "K": np.eye(3).tolist(),
                                             "positions": [[0, -5, 3]] * frames,
                                             "quaternions": [[1, 0, 0, 0]] * frames},
                                "environment": {"background": [0, 0, 0], "hdri": None,
                                                "lighting": [], "floor_datum": {"z": 0, "units": "m"}},
                                "physics": {"gravity": [0, 0, -9.81], "timestep_seconds": 1/100,
                                            "substeps_per_frame": 20,
                                            "units": {"length": "m", "time": "s", "mass": "kg", "energy": "J"}},
                                "objects_summary": {"n_objects": n_objects,
                                                    "n_subjects": n_objects - 1,
                                                    "n_support": 1,
                                                    "n_violators": int(label == "invalid"),
                                                    "n_affected": 0}}}},
        "observations": {"rgb": {"path": loader.RGB, "format": "mp4", "decode": "explicit"},
                         "dense": {"store": loader.DATA, "group": "/observations"}},
        "annotations": annotations,
        "storage": {"dense_store": loader.DATA, "format": "HDF5",
                    "compression": "gzip-4", "checksum": "fletcher32"},
    }
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
    sample_dir = write_sample(str(root), document, observations, instance, energy,
                              np.zeros((frames, side, side), np.float32),
                              violation_maps, per_object)
    if write_rgb:
        rgb = np.zeros((frames, side, side, 3), np.uint8)
        rgb[..., 1] = np.arange(frames, dtype=np.uint8)[:, None, None] * 20
        write_video(rgb, os.path.join(sample_dir, loader.RGB), fps=5)
    else:
        open(os.path.join(sample_dir, loader.RGB), "wb").close()
    return sample_dir


def make_pair(root, pair_uid="test-v3/L0/drop/0007_standard", **kwargs):
    valid = make_sample(root, pair_uid + "/valid", "valid", pair_uid=pair_uid,
                        valid_uid=pair_uid + "/valid", **kwargs)
    invalid = make_sample(root, pair_uid + "/invalid_solidity_strong", "invalid",
                          pair_uid=pair_uid, valid_uid=pair_uid + "/valid", **kwargs)
    return valid, invalid
