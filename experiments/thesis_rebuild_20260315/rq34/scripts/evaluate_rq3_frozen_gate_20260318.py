#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Tuple


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def load_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def mean(values: Iterable[float]) -> float:
    vals = list(values)
    if not vals:
        return 0.0
    return round(sum(vals) / len(vals), 4)


def build_summary_index(rows: List[dict]) -> Dict[Tuple[str, float, str], dict]:
    return {(r["dataset"], float(r["noise"]), r["method"]): r for r in rows}


def average_by_method(rows: List[dict], *, dataset: str | None = None) -> Dict[str, Dict[str, float]]:
    bucket: Dict[str, List[dict]] = defaultdict(list)
    for row in rows:
        if dataset is not None and row["dataset"] != dataset:
            continue
        bucket[row["method"]].append(row)
    out: Dict[str, Dict[str, float]] = {}
    for method, part in bucket.items():
        out[method] = {
            "avg_rca_accuracy": mean(r["rca_accuracy"] for r in part),
            "avg_e2e_success_rate": mean(r["e2e_success_rate"] for r in part),
            "avg_action_accuracy": mean(r["action_accuracy"] for r in part),
        }
    return out


def dataset_difficulty(rows: List[dict]) -> Dict[str, float]:
    bucket: Dict[str, List[float]] = defaultdict(list)
    for row in rows:
        bucket[row["dataset"]].append(float(row["e2e_success_rate"]))
    return {dataset: mean(values) for dataset, values in bucket.items()}


def progress_decoupling(rows: List[dict]) -> Dict[str, object]:
    noncollapsed = [r for r in rows if bool(r.get("rca_success")) != bool(r.get("e2e_success"))]
    by_dataset = defaultdict(int)
    by_method = defaultdict(int)
    for row in noncollapsed:
        by_dataset[row["dataset"]] += 1
        by_method[row["method"]] += 1
    return {
        "noncollapsed_count": len(noncollapsed),
        "by_dataset": dict(by_dataset),
        "by_method": dict(by_method),
    }


def compare_agent_vs_rag(rows: List[dict]) -> Dict[str, object]:
    index = build_summary_index(rows)
    cells = []
    rag_better = 0
    agent_better = 0
    tied = 0
    for dataset in sorted({r["dataset"] for r in rows}):
        for noise in sorted({float(r["noise"]) for r in rows if r["dataset"] == dataset}):
            agent = index.get((dataset, noise, "agent"))
            rag = index.get((dataset, noise, "rag"))
            if not agent or not rag:
                continue
            agent_e2e = float(agent["e2e_success_rate"])
            rag_e2e = float(rag["e2e_success_rate"])
            cell = {
                "dataset": dataset,
                "noise": noise,
                "agent_e2e": agent_e2e,
                "rag_e2e": rag_e2e,
            }
            if agent_e2e > rag_e2e:
                cell["winner"] = "agent"
                agent_better += 1
            elif rag_e2e > agent_e2e:
                cell["winner"] = "rag"
                rag_better += 1
            else:
                cell["winner"] = "tie"
                tied += 1
            cells.append(cell)
    return {
        "cells": cells,
        "agent_better_cells": agent_better,
        "rag_better_cells": rag_better,
        "tied_cells": tied,
    }


def gate_report(summary_rows: List[dict], progress_rows: List[dict]) -> dict:
    overall = average_by_method(summary_rows)
    per_dataset = {ds: average_by_method(summary_rows, dataset=ds) for ds in sorted({r["dataset"] for r in summary_rows})}
    difficulty = dataset_difficulty(summary_rows)
    decoupling = progress_decoupling(progress_rows)
    agent_vs_rag = compare_agent_vs_rag(summary_rows)
    index = build_summary_index(summary_rows)

    hdfs_noise_1 = [
        index[( "HDFS", 1.0, method)]["e2e_success_rate"]
        for method in ("agent", "rag", "vanilla")
        if ("HDFS", 1.0, method) in index
    ]
    hadoop_e2e_rows = [
        float(r["e2e_success_rate"])
        for r in summary_rows
        if r["dataset"] == "Hadoop"
    ]

    checks = {
        "rca_e2e_not_collapsed_everywhere": decoupling["noncollapsed_count"] > 0,
        "agent_overall_best_e2e": (
            overall.get("agent", {}).get("avg_e2e_success_rate", 0.0)
            > overall.get("rag", {}).get("avg_e2e_success_rate", 0.0)
            and overall.get("agent", {}).get("avg_e2e_success_rate", 0.0)
            > overall.get("vanilla", {}).get("avg_e2e_success_rate", 0.0)
        ),
        "rag_not_systematically_above_agent": (
            agent_vs_rag["rag_better_cells"] <= 2
            and overall.get("agent", {}).get("avg_e2e_success_rate", 0.0)
            >= overall.get("rag", {}).get("avg_e2e_success_rate", 0.0)
        ),
        "openstack_agent_above_vanilla": (
            per_dataset.get("OpenStack", {}).get("agent", {}).get("avg_e2e_success_rate", 0.0)
            > per_dataset.get("OpenStack", {}).get("vanilla", {}).get("avg_e2e_success_rate", 0.0)
        ),
        "hdfs_noise_1_not_flat": len({round(float(x), 4) for x in hdfs_noise_1}) > 1,
        "hadoop_is_hardest_dataset": (
            difficulty.get("Hadoop", 1.0) < difficulty.get("HDFS", 1.0)
            and difficulty.get("Hadoop", 1.0) < difficulty.get("OpenStack", 1.0)
        ),
        "hadoop_not_flat": len({round(x, 4) for x in hadoop_e2e_rows}) > 1,
    }
    checks["gate_pass"] = all(checks.values())

    return {
        "checks": checks,
        "overall_by_method": overall,
        "per_dataset_by_method": per_dataset,
        "dataset_difficulty_avg_e2e": difficulty,
        "agent_vs_rag": agent_vs_rag,
        "decoupling": decoupling,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-path", required=True, type=Path)
    parser.add_argument("--progress-path", required=True, type=Path)
    parser.add_argument("--output-path", type=Path)
    args = parser.parse_args()

    summary_rows = load_json(args.summary_path)
    progress_rows = load_jsonl(args.progress_path)
    report = gate_report(summary_rows, progress_rows)
    payload = {
        "summary_path": str(args.summary_path),
        "progress_path": str(args.progress_path),
        "report": report,
    }

    if args.output_path:
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        with args.output_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)

    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
