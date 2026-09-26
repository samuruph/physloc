"""One isolated host annotation request; JSON on stdin and stdout."""
from __future__ import annotations

import contextlib
import json
import os
import sys


def main():
    request = json.load(sys.stdin)
    from .. import params
    params.apply(request["params"])
    from .pipeline import annotate_work

    # Keep diagnostics separate from the machine-readable result.
    with contextlib.redirect_stdout(sys.stderr):
        release = os.path.basename(os.path.normpath(request["outroot"])) or "physloc_v0"
        results = annotate_work(request["workdir"], request["outroot"],
                                release=release, only=request.get("only"))
        if request.get("overlay", True):
            from ..viz.overlay import build
            for result in results:
                result["overlay"] = build(result["samples"]["invalid"])["path"]
    json.dump(results, sys.stdout)


if __name__ == "__main__":
    main()
