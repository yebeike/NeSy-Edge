from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


_SCRIPT_DIR = Path(__file__).resolve().parent


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-benchmark", action="store_true")
    ap.add_argument("--build-graphs", action="store_true")
    ap.add_argument("--evaluate", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--force-benchmark", action="store_true")
    ap.add_argument("--force-graphs", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    do_all = args.all or not (args.build_benchmark or args.build_graphs or args.evaluate)
    cmds = []
    if args.build_benchmark or do_all:
        cmd = [sys.executable, str(_SCRIPT_DIR / "build_rq2_fullcase_audit_benchmark_20260318.py")]
        if args.force_benchmark:
            cmd.append("--force")
        cmds.append(cmd)
    if args.build_graphs or do_all:
        cmd = [sys.executable, str(_SCRIPT_DIR / "build_rq2_fullcase_audit_graphs_20260318.py")]
        if args.force_graphs:
            cmd.append("--force")
        cmds.append(cmd)
    if args.evaluate or do_all:
        cmds.append([sys.executable, str(_SCRIPT_DIR / "evaluate_rq2_fullcase_audit_20260318.py")])

    for cmd in cmds:
        print("[RUN]", " ".join(cmd))
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
