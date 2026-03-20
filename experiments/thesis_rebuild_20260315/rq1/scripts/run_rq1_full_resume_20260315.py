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
    ap.add_argument("--run-tag", type=str, default="full_v3")
    ap.add_argument("--manifest-name", type=str, default="rq1_manifest_full_20260315.json")
    ap.add_argument("--hdfs-cases", type=int, default=300)
    ap.add_argument("--openstack-cases", type=int, default=200)
    ap.add_argument("--hadoop-cases", type=int, default=100)
    ap.add_argument("--hdfs-refs", type=int, default=24)
    ap.add_argument("--openstack-refs", type=int, default=43)
    ap.add_argument("--hadoop-refs", type=int, default=8)
    ap.add_argument("--noise-levels", type=str, default="0.0,0.2,0.4,0.6,0.8,1.0")
    ap.add_argument("--datasets", type=str, default="")
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    build_cmd = [
        sys.executable,
        str(_SCRIPT_DIR / "build_rq1_manifest_20260315.py"),
        "--manifest-name",
        args.manifest_name,
        "--hdfs-cases",
        str(args.hdfs_cases),
        "--openstack-cases",
        str(args.openstack_cases),
        "--hadoop-cases",
        str(args.hadoop_cases),
        "--hdfs-refs",
        str(args.hdfs_refs),
        "--openstack-refs",
        str(args.openstack_refs),
        "--hadoop-refs",
        str(args.hadoop_refs),
    ]
    resume_cmd = [
        sys.executable,
        str(_SCRIPT_DIR / "resume_rq1_run_20260315.py"),
        "--manifest-name",
        args.manifest_name,
        "--run-tag",
        args.run_tag,
        "--noise-levels",
        args.noise_levels,
    ]
    if args.datasets:
        resume_cmd.extend(["--datasets", args.datasets])
    report_cmd = [
        sys.executable,
        str(_SCRIPT_DIR / "summarize_rq1_run_20260315.py"),
        "--run-tag",
        args.run_tag,
    ]

    for cmd in [build_cmd, resume_cmd, report_cmd]:
        print("[RUN]", " ".join(cmd))
        subprocess.run(cmd, cwd=str(_PROJECT_ROOT), check=True)


if __name__ == "__main__":
    main()
