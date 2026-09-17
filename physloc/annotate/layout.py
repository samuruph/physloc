"""Names and discovery helpers for the single PhysLoc schema-v3 layout."""
from __future__ import annotations

import glob
import json
import os
from typing import Dict, List

from .. import loader

SCHEMA_VERSION = loader.SCHEMA_VERSION
SAMPLE_METADATA = loader.SAMPLE_METADATA
RGB = loader.RGB
DATA = loader.DATA
OVERLAY = "overlay.mp4"
CLOCKS = loader.CLOCKS
PASSES: Dict[str, str] = {
    "depth": "depth", "forward_flow": "forward_flow",
    "backward_flow": "backward_flow", "normal": "normal",
    "object_coordinates": "object_coordinates",
    "shadow_strength": "shadow_strength",
    "shadow_source_id": "shadow_source_id",
}


def identity(document) -> Dict[str, object]:
    return ((document or {}).get("metadata") or {}).get("sample_info") or {}


def read(sample_dir: str) -> Dict[str, object]:
    with open(os.path.join(sample_dir, SAMPLE_METADATA), encoding="utf-8") as handle:
        return json.load(handle)


def find(root: str) -> List[str]:
    return sorted(glob.glob(os.path.join(root, "samples", "**", SAMPLE_METADATA),
                            recursive=True))
