"""What the files in a clip directory are called, in one place.

Names follow Kubric's MOVi datasets (`refs/kubric/challenges/movi/README.md`)
wherever MOVi has a name for the thing -- `metadata.json`, `video`,
`segmentations`, `depth`, `forward_flow`, `backward_flow`, `normal`,
`object_coordinates` -- so a reader that knows MOVi finds the passes where it
expects them. The metadata inside keeps MOVi's top-level blocks (`metadata`,
`camera`, `instances`, `events`) and adds PhysLoc's own beside them; see
docs/schema.md.

Every consumer reads the names from here. They were spelled out in a dozen
files, and a rename that misses one is a reader that silently finds nothing.
"""
from __future__ import annotations

import glob
import json
import os
from typing import Dict, List

from .. import loader as _loader

#: 1 when the layout became MOVi's; 2 when the annotations shrank to the two
#: files below and everything derivable moved into `physloc/loader.py`. A reader
#: checks this before trusting any other field.
SCHEMA_VERSION = _loader.SCHEMA_VERSION

METADATA = _loader.METADATA
VIDEO = _loader.VIDEO
OVERLAY = "overlay.mp4"
SEGMENTATIONS = _loader.SEGMENTATIONS
#: Invalid clips only. `masks.npz`: the dense maps (`violation`, `causal`,
#: `causal_source`). `objects.npz`: the per-violator [K,T] table.
MASKS = _loader.MASKS
OBJECTS = _loader.OBJECTS
CLOCKS = _loader.CLOCKS
#: MOVi's per-instance, per-frame tensors -- positions, quaternions, velocities,
#: boxes, image positions, visibility -- as [k, s, ...] arrays. In an `.npz`
#: rather than in `metadata.json` because `pour` alone would be megabytes of
#: JSON per clip.
INSTANCES = "instances.npz"

#: Renderer pass -> the name it ships under, as both the file stem and the key.
PASSES: Dict[str, str] = {
    "depth": "depth",
    "forward_flow": "forward_flow",
    "backward_flow": "backward_flow",
    "normal": "normal",
    "object_coordinates": "object_coordinates",
}


def identity(meta) -> Dict[str, object]:
    """The `metadata` block: who a clip is, what it is of, and its geometry."""
    return (meta or {}).get("metadata") or {}


def read(cdir: str) -> Dict[str, object]:
    with open(os.path.join(cdir, METADATA)) as fh:
        return json.load(fh)


def find(root: str) -> List[str]:
    """Every clip's metadata file under a release root, sorted."""
    return sorted(glob.glob(os.path.join(root, "clips", "**", METADATA),
                            recursive=True))
