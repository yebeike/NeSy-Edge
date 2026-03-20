from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", type=Path, required=True)
    ap.add_argument("--spec", type=Path, required=True)
    ap.add_argument("--progress", type=Path, required=True)
    ap.add_argument("--next-benchmark-id", type=str, required=True)
    ap.add_argument("--poll-seconds", type=int, default=60)
    ap.add_argument("--log-path", type=Path, required=True)
    return ap.parse_args()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def log_line(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(text.rstrip() + "\n")


def run_and_log(cmd: list[str], log_path: Path) -> None:
    log_line(log_path, "$ " + " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.stdout:
        log_line(log_path, proc.stdout)
    if proc.stderr:
        log_line(log_path, proc.stderr)
    proc.check_returncode()


def main() -> None:
    args = parse_args()
    log_line(args.log_path, f"watcher_started state={args.state}")
    while True:
        if args.state.exists():
            try:
                state = load_json(args.state)
            except Exception as exc:
                log_line(args.log_path, f"state_read_error {exc}")
            else:
                completed = int(state.get("completed_steps", 0) or 0)
                total = int(state.get("total_steps", 0) or 0)
                log_line(args.log_path, f"poll completed={completed} total={total}")
                if total > 0 and completed >= total:
                    break
        time.sleep(max(5, int(args.poll_seconds)))

    run_and_log(
        [
            sys.executable,
            str(SCRIPT_DIR / "refresh_relaxed_full_after_probe_v1.py"),
            "--spec",
            str(args.spec),
            "--progress",
            str(args.progress),
            "--benchmark-id",
            args.next_benchmark_id,
        ],
        args.log_path,
    )
    run_and_log(
        [
            sys.executable,
            str(SCRIPT_DIR / "run_local_probe_v1.py"),
            "--benchmark-path",
            str(
                args.spec.parent.parent
                / "analysis"
                / args.next_benchmark_id
                / f"{args.next_benchmark_id}_package.json"
            ),
            "--output-dir",
            str(args.spec.parent.parent / "analysis" / f"local_probe_{args.next_benchmark_id}_highnoise_20260319"),
            "--run-tag",
            f"{args.next_benchmark_id}_highnoise_open_20260319",
            "--datasets",
            "HDFS,OpenStack,Hadoop",
            "--noise-levels",
            "1.0",
        ],
        args.log_path,
    )


if __name__ == "__main__":
    main()
