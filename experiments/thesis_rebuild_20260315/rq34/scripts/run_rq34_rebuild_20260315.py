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
    ap.add_argument("--cases-per-dataset", type=int, default=15)
    ap.add_argument("--noise-levels", type=str, default="0.0,0.2,0.4,0.6,0.8,1.0")
    ap.add_argument("--run-tag", type=str, default="sampled15x6_rebuild")
    ap.add_argument("--causal-graph-path", type=str, default="")
    ap.add_argument("--force-resample", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    run_cmd = [
        sys.executable,
        str(_SCRIPT_DIR / "run_rq34_resumable_20260315.py"),
        "--run-tag",
        args.run_tag,
        "--cases-per-dataset",
        str(args.cases_per_dataset),
        "--noise-levels",
        args.noise_levels,
    ]
    if args.causal_graph_path:
        run_cmd.extend(["--causal-graph-path", args.causal_graph_path])
    if args.force_resample:
        run_cmd.append("--force-resample")
    print("[RUN]", " ".join(run_cmd))
    subprocess.run(run_cmd, cwd=str(_PROJECT_ROOT), check=True)


if __name__ == "__main__":
    main()
