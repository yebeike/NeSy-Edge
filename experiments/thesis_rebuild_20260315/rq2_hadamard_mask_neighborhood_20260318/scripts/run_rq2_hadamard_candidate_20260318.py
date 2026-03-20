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
    cmd = [sys.executable, str(SCRIPT_DIR / script_name)] + extra_args
    print(label, flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    args = _parse_args()
    extra_args = ["--force"] if args.force else []
    _run_step(
        "[1/2] Build hadamard_mask_dynotears over frozen RQ2 inputs. Expected: 1-20 min",
        "build_rq2_hadamard_candidate_20260318.py",
        extra_args,
    )
    _run_step(
        "[2/2] Evaluate five-method paper-facing summary. Expected: <15s",
        "evaluate_rq2_hadamard_candidate_20260318.py",
        extra_args,
    )
    print("[Done] hadamard_mask_dynotears candidate artifacts are ready.", flush=True)


if __name__ == "__main__":
    main()
