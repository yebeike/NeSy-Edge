from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Mapping


SCRIPT_DIR = Path(__file__).resolve().parent
REBUILD_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = REBUILD_ROOT.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.thesis_rebuild_20260315.shared.utils.io_utils import write_json
from experiments.thesis_rebuild_20260315.rq3_rebuild_v1.scripts import rebuild_utils_v1 as utils


DEFAULT_PROGRESS_PATH = (
    REBUILD_ROOT
    / "analysis"
    / "local_probe_v1_20260318"
    / "rq3_triad_proof_slice_local_v1_20260318_progress.jsonl"
)
DEFAULT_OUTPUT_PATH = (
    REBUILD_ROOT
    / "analysis"
    / "local_probe_v1_20260318"
    / "rq3_triad_proof_slice_local_v1_gate_20260318.json"
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--progress-path", type=Path, default=DEFAULT_PROGRESS_PATH)
    ap.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    return ap.parse_args()


def metric(rows: List[Mapping[str, object]], field: str) -> float:
    return round(sum(int(bool(row[field])) for row in rows) / max(1, len(rows)), 4)


def main() -> None:
    args = parse_args()
    rows = [json.loads(line) for line in args.progress_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    contract = utils.contract()["local_gate"]
    high_noise_rows = [row for row in rows if f"{float(row['noise']):.1f}" == "1.0"]
    heur_rows = [row for row in rows if str(row["mode"]) == "heuristic_alert"]
    vanilla_rows = [row for row in high_noise_rows if str(row["mode"]) == "vanilla_open"]
    rag_rows = [row for row in high_noise_rows if str(row["mode"]) == "rag_open"]
    agent_rows = [row for row in high_noise_rows if str(row["mode"]) == "agent_open"]

    by_dataset: Dict[str, Dict[str, float]] = {}
    for dataset in sorted({str(row["dataset"]) for row in high_noise_rows}):
        ds_vanilla = [row for row in vanilla_rows if str(row["dataset"]) == dataset]
        ds_rag = [row for row in rag_rows if str(row["dataset"]) == dataset]
        ds_agent = [row for row in agent_rows if str(row["dataset"]) == dataset]
        by_dataset[dataset] = {
            "vanilla_family_accuracy_noise_1": metric(ds_vanilla, "rca_success"),
            "rag_family_accuracy_noise_1": metric(ds_rag, "rca_success"),
            "agent_family_accuracy_noise_1": metric(ds_agent, "rca_success"),
            "agent_minus_rag_noise_1": round(metric(ds_agent, "rca_success") - metric(ds_rag, "rca_success"), 4),
            "rag_minus_vanilla_noise_1": round(metric(ds_rag, "rca_success") - metric(ds_vanilla, "rca_success"), 4),
            "agent_gap_rows": int(sum(int(bool(row["rca_success"]) and not bool(row["e2e_success"])) for row in ds_agent)),
        }

    checks = [
        {
            "name": "heuristic_alert_not_too_easy",
            "passed": metric(heur_rows, "rca_success") <= float(contract["heuristic_alert_max_family_accuracy"]),
            "actual": metric(heur_rows, "rca_success"),
            "threshold": float(contract["heuristic_alert_max_family_accuracy"]),
        },
        {
            "name": "vanilla_high_noise_not_too_easy",
            "passed": metric(vanilla_rows, "rca_success") <= float(contract["vanilla_open_max_family_accuracy_at_noise_1"]),
            "actual": metric(vanilla_rows, "rca_success"),
            "threshold": float(contract["vanilla_open_max_family_accuracy_at_noise_1"]),
        },
        {
            "name": "agent_beats_rag_at_high_noise",
            "passed": round(metric(agent_rows, "rca_success") - metric(rag_rows, "rca_success"), 4)
            >= float(contract["agent_minus_rag_min_family_gap_at_noise_1"]),
            "actual": round(metric(agent_rows, "rca_success") - metric(rag_rows, "rca_success"), 4),
            "threshold": float(contract["agent_minus_rag_min_family_gap_at_noise_1"]),
        },
        {
            "name": "rag_beats_vanilla_at_high_noise",
            "passed": round(metric(rag_rows, "rca_success") - metric(vanilla_rows, "rca_success"), 4)
            >= float(contract["rag_minus_vanilla_min_family_gap_at_noise_1"]),
            "actual": round(metric(rag_rows, "rca_success") - metric(vanilla_rows, "rca_success"), 4),
            "threshold": float(contract["rag_minus_vanilla_min_family_gap_at_noise_1"]),
        },
        {
            "name": "rca_e2e_decoupling_present",
            "passed": int(sum(int(bool(row["rca_success"]) and not bool(row["e2e_success"])) for row in rows))
            >= int(contract["min_rca_e2e_gap_rows"]),
            "actual": int(sum(int(bool(row["rca_success"]) and not bool(row["e2e_success"])) for row in rows)),
            "threshold": int(contract["min_rca_e2e_gap_rows"]),
        },
    ]

    flat_failures = []
    if bool(contract["require_all_methods_non_flat"]):
        for mode_name in ("vanilla_open", "rag_open", "agent_open"):
            low_rows = [row for row in rows if str(row["mode"]) == mode_name and f"{float(row['noise']):.1f}" == "0.0"]
            high_rows = [row for row in rows if str(row["mode"]) == mode_name and f"{float(row['noise']):.1f}" == "1.0"]
            low_acc = metric(low_rows, "rca_success")
            high_acc = metric(high_rows, "rca_success")
            if abs(low_acc - high_acc) < 1e-9:
                flat_failures.append({"mode": mode_name, "low_noise_family_accuracy": low_acc, "high_noise_family_accuracy": high_acc})

    output = {
        "progress_path": str(args.progress_path),
        "overall": {
            "heuristic_alert_family_accuracy": metric(heur_rows, "rca_success"),
            "vanilla_family_accuracy_noise_1": metric(vanilla_rows, "rca_success"),
            "rag_family_accuracy_noise_1": metric(rag_rows, "rca_success"),
            "agent_family_accuracy_noise_1": metric(agent_rows, "rca_success"),
            "agent_minus_rag_noise_1": round(metric(agent_rows, "rca_success") - metric(rag_rows, "rca_success"), 4),
            "rag_minus_vanilla_noise_1": round(metric(rag_rows, "rca_success") - metric(vanilla_rows, "rca_success"), 4),
            "rca_without_e2e_rows": int(sum(int(bool(row["rca_success"]) and not bool(row["e2e_success"])) for row in rows)),
        },
        "per_dataset_high_noise": by_dataset,
        "checks": checks,
        "flat_failures": flat_failures,
        "local_gate_passed": all(check["passed"] for check in checks) and not flat_failures,
    }
    write_json(args.output_path, output)
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
