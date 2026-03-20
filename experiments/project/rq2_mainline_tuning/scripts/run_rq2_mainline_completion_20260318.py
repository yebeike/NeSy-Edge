from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    return ap.parse_args()


def _run_step(label: str, script_name: str, extra_args: list[str]) -> None:
    script_path = SCRIPT_DIR / script_name
    cmd = [sys.executable, str(script_path)] + extra_args
    print(label, flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    args = _parse_args()
    extra_args = ["--force"] if args.force else []
    _run_step("[1/4] Benchmark import + schema unification. Expected: <15s", "build_rq2_mainline_benchmark_20260318.py", extra_args)
    _run_step("[2/4] Hadoop feature-cap calibration. Expected: 1-15 min", "calibrate_rq2_mainline_hadoop_features_20260318.py", extra_args)
    _run_step("[3/4] Rebuild all 4 methods for all 3 datasets. Expected: 1-20 min", "build_rq2_mainline_graphs_20260318.py", extra_args)
    _run_step("[4/4] Evaluate all modes + final report synthesis. Expected: <1 min", "evaluate_rq2_mainline_20260318.py", extra_args)
    print("[Done] RQ2 mainline completion artifacts are ready.", flush=True)


if __name__ == "__main__":
    main()
