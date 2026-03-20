from __future__ import annotations

import argparse
import json
import time

from rq2_mainline_completion_common_20260318 import (
    build_original_graph,
    build_pc_graph,
    prepare_feature_space,
)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["Hadoop"])
    ap.add_argument("--method", required=True, choices=["original_dynotears", "pc_cpdag_hypothesis"])
    ap.add_argument("--cap", required=True, type=int)
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    df, tpl_map, profile = prepare_feature_space(args.dataset, max_cols=args.cap)
    started = time.time()
    if args.method == "original_dynotears":
        edges = build_original_graph(args.dataset, df, tpl_map)
    else:
        edges = build_pc_graph(args.dataset, df, tpl_map)
    payload = {
        "dataset": args.dataset,
        "method": args.method,
        "cap": args.cap,
        "selected_columns": int(profile["selected_columns"]),
        "edges": len(edges),
        "elapsed_sec": round(time.time() - started, 3),
    }
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
