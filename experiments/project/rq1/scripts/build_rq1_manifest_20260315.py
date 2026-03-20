from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REBUILD_ROOT = _SCRIPT_DIR.parents[1]
_PROJECT_ROOT = _REBUILD_ROOT.parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.thesis_rebuild_20260315.shared.case_builders.rq1_case_pool import build_rq1_manifest
from experiments.thesis_rebuild_20260315.shared.utils.io_utils import write_json
from experiments.thesis_rebuild_20260315.shared.utils.project_paths import MANIFEST_DIR, ensure_dirs


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest-name", type=str, default="rq1_manifest_pilot_20260315.json")
    ap.add_argument("--hdfs-cases", type=int, default=300)
    ap.add_argument("--openstack-cases", type=int, default=200)
    ap.add_argument("--hadoop-cases", type=int, default=100)
    ap.add_argument("--hdfs-refs", type=int, default=24)
    ap.add_argument("--openstack-refs", type=int, default=43)
    ap.add_argument("--hadoop-refs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=2026)
    return ap.parse_args()


def main() -> str:
    args = _parse_args()
    ensure_dirs()
    manifest = build_rq1_manifest(
        pilot_sizes={"HDFS": args.hdfs_cases, "OpenStack": args.openstack_cases, "Hadoop": args.hadoop_cases},
        ref_sizes={"HDFS": args.hdfs_refs, "OpenStack": args.openstack_refs, "Hadoop": args.hadoop_refs},
        seed=args.seed,
    )
    out_path = MANIFEST_DIR / args.manifest_name
    write_json(out_path, manifest)
    compact = {
        ds: {
            "pool_size": meta["pool_size"],
            "reference_count": meta["reference_count"],
            "eval_count": meta["eval_count"],
        }
        for ds, meta in manifest["datasets"].items()
    }
    print(json.dumps(compact, indent=2))
    print(f"[Saved] {out_path}")
    return str(out_path)


if __name__ == "__main__":
    main()
