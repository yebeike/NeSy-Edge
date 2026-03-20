from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


_SCRIPT_DIR = Path(__file__).resolve().parent


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-benchmark", action="store_true")
    ap.add_argument("--build-openstack", action="store_true")
    ap.add_argument("--build-graphs", action="store_true")
    ap.add_argument("--evaluate", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--force-benchmark", action="store_true")
    ap.add_argument("--force-openstack", action="store_true")
    ap.add_argument("--force-domains", default="")
    ap.add_argument("--force-graphs", default="")
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    do_all = args.all or not (args.build_benchmark or args.build_openstack or args.build_graphs or args.evaluate)
    cmds = []
    if args.build_benchmark or do_all:
        cmd = [sys.executable, str(_SCRIPT_DIR / "build_rq2_fullcase_benchmark_20260316.py")]
        if args.force_benchmark:
            cmd.append("--force")
        cmds.append(cmd)
    if args.build_openstack or do_all:
        cmd = [sys.executable, str(_SCRIPT_DIR / "build_openstack_semantic_timeseries_20260316.py")]
        if args.force_openstack:
            cmd.append("--force")
        cmds.append(cmd)
    if args.build_graphs or do_all:
        cmd = [sys.executable, str(_SCRIPT_DIR / "build_rq2_fullcase_graphs_20260316.py")]
        if args.force_domains:
            cmd.extend(["--force-domains", args.force_domains])
        if args.force_graphs:
            cmd.extend(["--force-graphs", args.force_graphs])
        cmds.append(cmd)
    if args.evaluate or do_all:
        cmds.append([sys.executable, str(_SCRIPT_DIR / "evaluate_rq2_fullcase_20260316.py")])

    for cmd in cmds:
        print("[RUN]", " ".join(cmd))
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
