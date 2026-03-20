from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REBUILD_ROOT = _SCRIPT_DIR.parents[1]
_PROJECT_ROOT = _REBUILD_ROOT.parents[1]


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--refresh-legacy",
        action="store_true",
        help="Re-run legacy RQ2 scripts that write to the original results paths.",
    )
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    cmds = []
    if args.refresh_legacy:
        cmds.extend(
            [
                [sys.executable, str(_PROJECT_ROOT / "experiments" / "rq2" / "build_modified_graph_rq2_pruned_20260314.py")],
                [sys.executable, str(_PROJECT_ROOT / "experiments" / "rq2" / "evaluate_rq2_hybrid_penalized_20260314.py")],
            ]
        )
    cmds.append([sys.executable, str(_SCRIPT_DIR / "summarize_rq2_current_20260315.py")])
    for cmd in cmds:
        print("[RUN]", " ".join(cmd))
        subprocess.run(cmd, cwd=str(_PROJECT_ROOT), check=True)


if __name__ == "__main__":
    main()
