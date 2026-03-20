from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.thesis_rebuild_20260315.rq1.components.edge_protocol_20260317 import (
    DEFAULT_PRESETS,
    build_edge_manifest,
    preset_manifest_name,
    save_manifest,
)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=sorted(DEFAULT_PRESETS), default="small")
    ap.add_argument("--suffix", type=str, default="v1")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--manifest-name", type=str, default="")
    ap.add_argument("--sampling-mode", choices=("natural", "unique_clean"), default="natural")
    return ap.parse_args()


def main() -> str:
    args = _parse_args()
    spec = DEFAULT_PRESETS[args.preset]
    manifest = build_edge_manifest(
        eval_sizes=spec["eval"],
        ref_sizes=spec["refs"],
        seed=args.seed,
        sampling_mode=args.sampling_mode,
    )
    name = args.manifest_name or preset_manifest_name(args.preset, args.suffix)
    out_path = save_manifest(manifest, name)
    compact = {
        ds: {
            "pool_size": meta["pool_size"],
            "sampling_mode": meta.get("sampling_mode", ""),
            "reference_count": meta["reference_count"],
            "eval_count": meta["eval_count"],
            "split_audit": meta.get("split_audit", {}),
        }
        for ds, meta in manifest["datasets"].items()
    }
    print(json.dumps({"preset": args.preset, "manifest": name, "datasets": compact}, indent=2))
    print(f"[Saved] {out_path}")
    return str(out_path)


if __name__ == "__main__":
    main()
