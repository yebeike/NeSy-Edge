from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Mapping


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", type=Path, required=True)
    ap.add_argument("--progress", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--drop-all-zero", action="store_true")
    ap.add_argument("--drop-rag-agent-toxic", action="store_true")
    ap.add_argument("--drop-oa-vanilla-toxic", action="store_true")
    return ap.parse_args()


def e2e_flag(row: Mapping[str, object] | None) -> int:
    if not row:
        return 0
    return int(bool(row.get("e2e_success")))


def main() -> None:
    args = parse_args()
    spec = load_json(args.spec)
    progress_rows = [json.loads(line) for line in args.progress.read_text(encoding="utf-8").splitlines() if line.strip()]

    by_case: Dict[tuple[str, str], Dict[str, Mapping[str, object]]] = defaultdict(dict)
    for row in progress_rows:
        by_case[(str(row["dataset"]), str(row["case_id"]))][str(row["mode"])] = row

    filtered_datasets: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    dropped: Dict[str, List[Dict[str, object]]] = defaultdict(list)

    for dataset, rows in spec["datasets"].items():
        for item in rows:
            case_id = str(item.get("eval_case_id", "") or item["case_id"])
            mode_rows = by_case.get((dataset, case_id), {})
            oa = e2e_flag(mode_rows.get("open_alert_only"))
            van = e2e_flag(mode_rows.get("vanilla_open"))
            rag = e2e_flag(mode_rows.get("rag_open"))
            agent = e2e_flag(mode_rows.get("agent_open"))

            reasons: List[str] = []
            if args.drop_all_zero and oa == van == rag == agent == 0:
                reasons.append("all_zero")
            if args.drop_rag_agent_toxic and rag == 1 and agent == 0:
                reasons.append("rag_agent_toxic")
            if args.drop_oa_vanilla_toxic and oa == 1 and van == 1 and rag == 0 and agent == 0:
                reasons.append("oa_vanilla_toxic")

            if reasons:
                dropped[dataset].append(
                    {
                        "case_id": str(item["case_id"]),
                        "eval_case_id": case_id,
                        "gt_action_id": str(item["gt_action_id"]),
                        "oa": oa,
                        "vanilla": van,
                        "rag": rag,
                        "agent": agent,
                        "reasons": reasons,
                    }
                )
                continue

            filtered_datasets[dataset].append(dict(item))

    payload = dict(spec)
    payload["benchmark_id"] = f"{spec['benchmark_id']}_filtered"
    payload["datasets"] = dict(filtered_datasets)
    payload["filter_report"] = {
        "source_spec": str(args.spec),
        "progress": str(args.progress),
        "drop_all_zero": bool(args.drop_all_zero),
        "drop_rag_agent_toxic": bool(args.drop_rag_agent_toxic),
        "drop_oa_vanilla_toxic": bool(args.drop_oa_vanilla_toxic),
        "kept_counts": {dataset: len(items) for dataset, items in filtered_datasets.items()},
        "dropped_counts": {dataset: len(items) for dataset, items in dropped.items()},
        "dropped_cases": dict(dropped),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(args.output)
    print(json.dumps(payload["filter_report"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
