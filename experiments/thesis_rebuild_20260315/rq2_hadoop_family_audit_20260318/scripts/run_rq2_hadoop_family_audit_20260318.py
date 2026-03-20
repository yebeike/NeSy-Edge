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
    print(label)
    subprocess.run(cmd, check=True)


def main() -> None:
    args = _parse_args()
    extra_args = ["--force"] if args.force else []
    _run_step("[1/3] Hadoop benchmark rebuild. Expected: <20s", "build_rq2_hadoop_family_audit_benchmark_20260318.py", extra_args)
    _run_step("[2/3] Hadoop graph build. Expected: 10-60s", "build_rq2_hadoop_family_audit_graphs_20260318.py", extra_args)
    _run_step("[3/3] Hadoop evaluation. Expected: <20s", "evaluate_rq2_hadoop_family_audit_20260318.py", [])
    print("[Done] RQ2 Hadoop family audit artifacts are ready.")


if __name__ == "__main__":
    main()
