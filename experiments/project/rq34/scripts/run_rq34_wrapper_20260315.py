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
    ap.add_argument("--cases-per-dataset", type=int, default=10)
    ap.add_argument("--noise-levels", type=str, default="0.0,0.2,0.4,0.6,0.8,1.0")
    ap.add_argument("--run-tag", type=str, default="sampled10x6")
    ap.add_argument(
        "--skip-run",
        action="store_true",
        help="Only summarize an existing source file.",
    )
    ap.add_argument(
        "--refresh-legacy",
        action="store_true",
        help="Run the legacy stage4 script, which writes to the original results paths.",
    )
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    legacy_source = _PROJECT_ROOT / "results" / "rq123_e2e" / "stage4_noise_api_sampled_summary_20260314.json"
    rebuild_source = _REBUILD_ROOT / "rq34" / "results" / "stage4_noise_api_sampled_summary_20260314.json"
    source = legacy_source
    if not args.skip_run and args.refresh_legacy:
        cmd = [
            sys.executable,
            str(_PROJECT_ROOT / "experiments" / "rq123_e2e" / "stage4_noise_api_sampled_20260313.py"),
            "--cases-per-dataset",
            str(args.cases_per_dataset),
            "--noise-levels",
            args.noise_levels,
        ]
        print("[RUN]", " ".join(cmd))
        subprocess.run(cmd, cwd=str(_PROJECT_ROOT), check=True)
    elif not args.skip_run:
        cmd = [
            sys.executable,
            str(_SCRIPT_DIR / "run_rq34_rebuild_20260315.py"),
            "--cases-per-dataset",
            str(args.cases_per_dataset),
            "--noise-levels",
            args.noise_levels,
            "--run-tag",
            args.run_tag,
        ]
        print("[RUN]", " ".join(cmd))
        subprocess.run(cmd, cwd=str(_PROJECT_ROOT), check=True)
        source = rebuild_source
    subprocess.run(
        [
            sys.executable,
            str(_SCRIPT_DIR / "summarize_rq34_current_20260315.py"),
            "--source",
            str(source),
            "--run-tag",
            args.run_tag,
            "--cases-per-dataset",
            str(args.cases_per_dataset),
            "--noise-levels",
            args.noise_levels,
        ],
        cwd=str(_PROJECT_ROOT),
        check=True,
    )


if __name__ == "__main__":
    main()
