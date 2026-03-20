from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REBUILD_ROOT = SCRIPT_DIR.parent


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", type=Path, required=True)
    ap.add_argument("--full-spec", type=Path, required=True)
    ap.add_argument("--poll-seconds", type=int, default=60)
    ap.add_argument("--log-path", type=Path, required=True)
    return ap.parse_args()


def log(msg: str, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(msg.rstrip() + "\n")
    print(msg, flush=True)


def run_cmd(cmd: list[str], log_path: Path) -> None:
    log("$ " + " ".join(cmd), log_path)
    subprocess.run(cmd, check=True)


def wait_for_completion(state_path: Path, poll_seconds: int, log_path: Path) -> None:
    while True:
        if state_path.exists():
            data = json.loads(state_path.read_text(encoding="utf-8"))
            completed = int(data.get("completed_steps", 0) or 0)
            total = int(data.get("total_steps", 0) or 0)
            log(f"poll completed={completed} total={total}", log_path)
            if total > 0 and completed >= total:
                return
        time.sleep(max(5, poll_seconds))


def main() -> None:
    args = parse_args()
    full_6noise_dir = REBUILD_ROOT / "analysis" / "rq3_full_relaxed_144_v3_6noise_20260320"
    full_6noise_dir.mkdir(parents=True, exist_ok=True)
    full_6noise_pkg = full_6noise_dir / "rq3_full_relaxed_144_v3_20260320_package.json"

    core_spec = REBUILD_ROOT / "specs" / "rq3_core_subset_v3_20260320.json"
    core_dir = REBUILD_ROOT / "analysis" / "rq3_core_subset_v3_6noise_20260320"
    core_dir.mkdir(parents=True, exist_ok=True)
    core_pkg = core_dir / "rq3_core_subset_v3_20260320_package.json"

    log("waiting for high-noise api run to finish", args.log_path)
    wait_for_completion(args.state, args.poll_seconds, args.log_path)

    run_cmd(
        [
            sys.executable,
            str(SCRIPT_DIR / "build_proof_package_v1.py"),
            "--spec",
            str(args.full_spec),
            "--output-dir",
            str(full_6noise_dir),
            "--noise-levels",
            "0.0,0.2,0.4,0.6,0.8,1.0",
        ],
        args.log_path,
    )
    run_cmd(
        [
            sys.executable,
            str(SCRIPT_DIR / "run_api_probe_v1.py"),
            "--benchmark-path",
            str(full_6noise_pkg),
            "--output-dir",
            str(REBUILD_ROOT / "analysis" / "api_probe_full_relaxed_144_v3_6noise_20260320"),
            "--run-tag",
            "rq3_full_relaxed_144_v3_6noise_api_20260320",
            "--datasets",
            "HDFS,OpenStack,Hadoop",
            "--modes",
            "vanilla_open,rag_open,agent_open",
            "--noise-levels",
            "0.0,0.2,0.4,0.6,0.8,1.0",
            "--model",
            "deepseek-chat",
            "--temperature",
            "0.0",
            "--max-output-tokens",
            "220",
        ],
        args.log_path,
    )

    run_cmd(
        [
            sys.executable,
            str(SCRIPT_DIR / "build_core_subset_spec_v1.py"),
            "--spec",
            str(args.full_spec),
            "--benchmark-id",
            "rq3_core_subset_v3_20260320",
            "--output",
            str(core_spec),
        ],
        args.log_path,
    )
    run_cmd(
        [
            sys.executable,
            str(SCRIPT_DIR / "build_proof_package_v1.py"),
            "--spec",
            str(core_spec),
            "--output-dir",
            str(core_dir),
            "--noise-levels",
            "0.0,0.2,0.4,0.6,0.8,1.0",
        ],
        args.log_path,
    )
    run_cmd(
        [
            sys.executable,
            str(SCRIPT_DIR / "run_api_probe_v1.py"),
            "--benchmark-path",
            str(core_pkg),
            "--output-dir",
            str(REBUILD_ROOT / "analysis" / "api_probe_core_subset_v3_6noise_20260320"),
            "--run-tag",
            "rq3_core_subset_v3_6noise_api_20260320",
            "--datasets",
            "HDFS,OpenStack,Hadoop",
            "--modes",
            "vanilla_open,rag_open,agent_open",
            "--noise-levels",
            "0.0,0.2,0.4,0.6,0.8,1.0",
            "--model",
            "deepseek-chat",
            "--temperature",
            "0.0",
            "--max-output-tokens",
            "220",
        ],
        args.log_path,
    )

    log("api pipeline completed", args.log_path)


if __name__ == "__main__":
    main()
