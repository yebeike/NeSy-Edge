from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REBUILD_ROOT = SCRIPT_DIR.parent


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", type=Path, required=True)
    ap.add_argument("--progress", type=Path, required=True)
    ap.add_argument("--benchmark-id", type=str, required=True)
    ap.add_argument("--output-dir", type=Path, default=REBUILD_ROOT / "analysis")
    return ap.parse_args()


def run_cmd(cmd: list[str]) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    summary_path = (
        args.progress.parent / f"{args.progress.stem.replace('_progress', '')}_relaxed_summary.json"
    )
    new_spec_path = REBUILD_ROOT / "specs" / f"{args.benchmark_id}.json"
    package_output_dir = args.output_dir / args.benchmark_id

    run_cmd(
        [
            sys.executable,
            str(SCRIPT_DIR / "summarize_relaxed_full_probe_v1.py"),
            "--spec",
            str(args.spec),
            "--progress",
            str(args.progress),
            "--output",
            str(summary_path),
        ]
    )
    run_cmd(
        [
            sys.executable,
            str(SCRIPT_DIR / "build_relaxed_full_spec_v1.py"),
            "--benchmark-id",
            args.benchmark_id,
            "--progress",
            str(args.progress),
            "--drop-all-zero",
            "--drop-rag-agent-toxic",
            "--drop-oa-vanilla-toxic",
        ]
    )
    run_cmd(
        [
            sys.executable,
            str(SCRIPT_DIR / "build_proof_package_v1.py"),
            "--spec",
            str(new_spec_path),
            "--output-dir",
            str(package_output_dir),
        ]
    )
    print(summary_path)
    print(new_spec_path)
    print(package_output_dir)


if __name__ == "__main__":
    main()
