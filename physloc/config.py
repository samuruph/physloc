"""Run settings as a file rather than a wall of flags.

    python -m physloc.cli generate --config review
    python -m physloc.cli generate --config configs/review.yaml --seed 42

**Precedence is CLI > config > argparse default.** A flag you type always wins,
so a config is a starting point you can lean on, never something that quietly
overrides what you asked for. That ordering is the whole reason this module
exists rather than a shell alias: an alias cannot be partially overridden.

Host-side only. Nothing the container imports may depend on PyYAML -- the
container's Python is 3.9 with Kubric's own pinned set, and `taxonomy.py` and
`scenarios/` are imported on both sides of the seam. `cli.py` is not.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional, Set

#: Applied to every subcommand, before the per-command block.
DEFAULTS_KEY = "defaults"


class ConfigError(Exception):
    pass


def resolve_path(name: str) -> str:
    """Accept a path, or a bare name resolved against `configs/`."""
    if os.path.sep in name or name.endswith((".yaml", ".yml")):
        return name
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(repo, "configs", name + ".yaml")


def load(name: Optional[str], command: str, valid: Set[str]) -> Dict[str, Any]:
    """Flatten a config file into `{dest: value}` for one subcommand.

    A file may be flat, or split into a `defaults:` block plus one block per
    command; the command's own block wins over `defaults`. Keys are argparse
    destinations, so `--keep-going` is `keep_going`.

    Unknown keys **in a command's own block** are an error, not a warning. A
    silently ignored typo there is a run that does not do what the file says it
    does, and being able to trust the file at a glance is the entire point.

    Keys in `defaults` (and bare top-level keys) are filtered instead of
    rejected, because they are cross-command by construction: `seed` is
    meaningful to `generate` and meaningless to `taxonomy`, and a shared block
    that could not contain it would not be worth having.
    """
    if not name:
        name = os.environ.get("PHYSLOC_CONFIG") or ""
    if not name:
        return {}
    path = resolve_path(name)
    if not os.path.exists(path):
        raise ConfigError("no such config: %s" % path)
    try:
        import yaml
    except ImportError:                                    # pragma: no cover
        raise ConfigError(
            "reading %s needs PyYAML: conda install -n physloc pyyaml" % path)

    with open(path) as fh:
        doc = yaml.safe_load(fh) or {}
    if not isinstance(doc, dict):
        raise ConfigError("%s: top level must be a mapping" % path)

    blocks = {k for k in doc if isinstance(doc.get(k), dict)}

    # Cross-command: filtered to what this subcommand understands.
    shared: Dict[str, Any] = {}
    section = doc.get(DEFAULTS_KEY)
    if isinstance(section, dict):
        shared.update(section)
    for k, v in doc.items():            # flat files: bare top-level scalars
        if k not in blocks and k != DEFAULTS_KEY:
            shared.setdefault(k, v)
    out = {k: v for k, v in shared.items() if k in valid}

    # This command's own block: strict, and it wins.
    own = doc.get(command)
    if isinstance(own, dict):
        unknown = sorted(set(own) - valid)
        if unknown:
            raise ConfigError(
                "%s: unknown setting(s) under `%s:`: %s\n  valid here: %s"
                % (path, command, ", ".join(unknown), ", ".join(sorted(valid))))
        out.update(own)
    return out
