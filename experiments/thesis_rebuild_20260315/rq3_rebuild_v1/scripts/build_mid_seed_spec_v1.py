from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
REBUILD_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = REBUILD_ROOT.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.thesis_rebuild_20260315.shared.utils.io_utils import write_json


DEFAULT_OUTPUT_DIR = REBUILD_ROOT / "specs"
TRIAD_SPEC_PATH = REBUILD_ROOT / "specs" / "rq3_triad_proof_slice_v3_20260318.json"
DATASET_ADMISSION_PATH = REBUILD_ROOT / "configs" / "dataset_admission_v1_20260318.json"

DEFAULT_RANKED_INPUTS = [
    REBUILD_ROOT
    / "analysis"
    / "local_probe_openstack_reanchor_bulk_v2_highnoise_20260319"
    / "rq3_openstack_reanchor_bulk_v2_highnoise_open_20260319_ranked_graphfix.json",
    REBUILD_ROOT
    / "analysis"
    / "local_probe_openstack_reanchor_bulk_v1_highnoise_20260319"
    / "rq3_openstack_reanchor_bulk_v1_highnoise_open_20260319_ranked_mid_relaxed.json",
    REBUILD_ROOT
    / "analysis"
    / "local_probe_openstack_keepdirect_bulk_v3_highnoise_20260319"
    / "rq3_openstack_keepdirect_bulk_v3_highnoise_open_20260319_ranked_mid_relaxed.json",
    REBUILD_ROOT
    / "analysis"
    / "local_probe_openstack_bulk_candidate_sweep_v2_highnoise_20260319"
    / "rq3_openstack_bulk_candidate_sweep_v2_highnoise_open_20260319_ranked_calibrated.json",
    REBUILD_ROOT
    / "analysis"
    / "local_probe_openstack_keepdirect_bulk_v3_highnoise_20260319"
    / "rq3_openstack_keepdirect_bulk_v3_highnoise_open_20260319_ranked_calibrated.json",
    REBUILD_ROOT
    / "analysis"
    / "local_probe_openstack_reanchor_bulk_v1_highnoise_20260319"
    / "rq3_openstack_reanchor_bulk_v1_highnoise_open_20260319_ranked_calibrated.json",
    REBUILD_ROOT
    / "analysis"
    / "local_probe_hdfs_reanchor_bulk_v1_highnoise_20260319"
    / "rq3_hdfs_reanchor_bulk_v1_highnoise_open_20260319_ranked_graphfix.json",
    REBUILD_ROOT
    / "analysis"
    / "local_probe_hdfs_keepdirect_pipeline_storage_v6_highnoise_20260319"
    / "rq3_hdfs_keepdirect_pipeline_storage_v6_highnoise_open_20260319_ranked_mid_relaxed.json",
    REBUILD_ROOT
    / "analysis"
    / "local_probe_hdfs_pipeline_candidate_sweep_v1_highnoise_20260319"
    / "rq3_hdfs_pipeline_candidate_sweep_v1_highnoise_open_20260319_ranked_mid_relaxed.json",
    REBUILD_ROOT
    / "analysis"
    / "local_probe_hdfs_transfer_only_sweep_v5_highnoise_20260319"
    / "rq3_hdfs_transfer_only_sweep_v5_highnoise_open_20260319_ranked_calibrated.json",
    REBUILD_ROOT
    / "analysis"
    / "local_probe_hdfs_pipeline_candidate_sweep_v1_highnoise_20260319"
    / "rq3_hdfs_pipeline_candidate_sweep_v1_highnoise_open_20260319_ranked_calibrated.json",
    REBUILD_ROOT
    / "analysis"
    / "local_probe_hdfs_bulk_candidate_sweep_v3_highnoise_20260319"
    / "rq3_hdfs_bulk_candidate_sweep_v3_highnoise_open_20260319_ranked_calibrated.json",
    REBUILD_ROOT
    / "analysis"
    / "local_probe_hdfs_keepdirect_pipeline_storage_v6_highnoise_20260319"
    / "rq3_hdfs_keepdirect_pipeline_storage_v6_highnoise_open_20260319_ranked_calibrated.json",
    REBUILD_ROOT
    / "analysis"
    / "local_probe_hadoop_control_reanchor_bulk_v1_highnoise_20260319"
    / "rq3_hadoop_control_reanchor_bulk_v1_highnoise_open_20260319_ranked_graphfix.json",
    REBUILD_ROOT
    / "analysis"
    / "local_probe_hadoop_keepdirect_bulk_v6_highnoise_20260319"
    / "rq3_hadoop_keepdirect_bulk_v6_highnoise_open_20260319_ranked_mid_relaxed.json",
    REBUILD_ROOT
    / "analysis"
    / "local_probe_hadoop_formal_candidate_sweep_v4_highnoise_20260319"
    / "rq3_hadoop_formal_candidate_sweep_v4_highnoise_open_20260319_ranked_mid_relaxed.json",
    REBUILD_ROOT
    / "analysis"
    / "local_probe_hadoop_formal_candidate_sweep_v4_highnoise_20260319"
    / "rq3_hadoop_formal_candidate_sweep_v4_highnoise_open_20260319_ranked_calibrated.json",
    REBUILD_ROOT
    / "analysis"
    / "local_probe_hadoop_bulk_candidate_sweep_v5_highnoise_20260319"
    / "rq3_hadoop_bulk_candidate_sweep_v5_highnoise_open_20260319_ranked_calibrated.json",
    REBUILD_ROOT
    / "analysis"
    / "local_probe_hadoop_keepdirect_bulk_v6_highnoise_20260319"
    / "rq3_hadoop_keepdirect_bulk_v6_highnoise_open_20260319_ranked_calibrated.json",
]

VERDICT_RANK = {
    "shortlist": 3,
    "provisional": 2,
    "reject": 1,
    "reject_direct": 0,
    "incomplete": -1,
}

MAX_PER_ACTION = 3

MANUAL_KEEP = {
    "OpenStack": {
        "openstack_48",
        "openstack_49",
        "openstack_51",
        "openstack_win_20",
    },
    "HDFS": {
        "hdfs_blk_blk_-3102267849859399193",
    },
    "Hadoop": {
        "hadoop_application_1445062781478_0015",
        "hadoop_application_1445087491445_0001",
        "hadoop_application_1445182159119_0001",
        "hadoop_application_1445182159119_0014",
    },
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--benchmark-id", type=str, default="rq3_mid_seed_v1_20260319")
    ap.add_argument("--per-dataset-cap", type=int, default=12)
    ap.add_argument("--probe-progress", type=Path, default=None)
    ap.add_argument("--drop-all-zero", action="store_true")
    ap.add_argument("--drop-rag-agent-toxic", action="store_true")
    return ap.parse_args()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def base_case_id(case_id: str) -> str:
    return re.sub(r"__[^_]+$", "", str(case_id))


def dedup_key(row: Mapping[str, object]) -> Tuple[str, str, str]:
    return (
        str(row["dataset"]),
        base_case_id(str(row["case_id"])),
        str(row["gt_action_id"]),
    )


def choose_better(existing: Mapping[str, object] | None, candidate: Mapping[str, object]) -> bool:
    if existing is None:
        return True
    current = (
        VERDICT_RANK[str(candidate["verdict"])],
        int(candidate.get("score", 0)),
        int(bool(candidate.get("agent_e2e", 0))),
        int(bool(candidate.get("rag_e2e", 0))),
        -int(bool(candidate.get("heuristic_alert_e2e", 0))),
    )
    prev = (
        VERDICT_RANK[str(existing["verdict"])],
        int(existing.get("score", 0)),
        int(bool(existing.get("agent_e2e", 0))),
        int(bool(existing.get("rag_e2e", 0))),
        -int(bool(existing.get("heuristic_alert_e2e", 0))),
    )
    return current > prev


def row_from_ranked(rank_row: Mapping[str, object], source_file: str) -> Dict[str, object]:
    modes = dict(rank_row.get("modes", {}) or {})
    eval_case_id = str(rank_row.get("eval_case_id", rank_row["case_id"]))
    pool_case_id = str(rank_row.get("pool_case_id", "") or base_case_id(eval_case_id))
    return {
        "dataset": str(rank_row["dataset"]),
        "case_id": pool_case_id,
        "eval_case_id": eval_case_id,
        "source": str(rank_row.get("source", "") or ""),
        "gt_family_id": str(rank_row.get("gt_family_id", "") or ""),
        "gt_action_id": str(rank_row["gt_action_id"]),
        "alert_match": "",
        "eligibility_note": str(rank_row.get("eligibility_note", "") or ""),
        "selection_score": float(rank_row.get("selection_score", 0.0) or 0.0),
        "difficulty_score": float(rank_row.get("difficulty_score", 0.0) or 0.0),
        "weak_mainline_alert": bool(rank_row.get("weak_mainline_alert", False)),
        "fixed_small_member": bool(rank_row.get("fixed_small_member", False)),
        "selected_alert_flags": dict(rank_row.get("selected_alert_flags", {}) or {}),
        "verdict": str(rank_row["verdict"]),
        "score": int(rank_row.get("score", 0)),
        "source_file": source_file,
        "heuristic_alert_e2e": int(bool((modes.get("heuristic_alert") or {}).get("e2e_success"))),
        "open_alert_only_e2e": int(bool((modes.get("open_alert_only") or {}).get("e2e_success"))),
        "vanilla_e2e": int(bool((modes.get("vanilla_open") or {}).get("e2e_success"))),
        "rag_e2e": int(bool((modes.get("rag_open") or {}).get("e2e_success"))),
        "agent_e2e": int(bool((modes.get("agent_open") or {}).get("e2e_success"))),
    }


def sort_rows(items: List[Dict[str, object]]) -> List[Dict[str, object]]:
    return sorted(
        items,
        key=lambda row: (
            -VERDICT_RANK[str(row["verdict"])],
            -int(row["score"]),
            -int(bool(row["agent_e2e"])),
            -int(bool(row["rag_e2e"])),
            int(bool(row["heuristic_alert_e2e"])),
            str(row["gt_action_id"]),
            str(row["case_id"]),
        ),
    )


def select_diverse_rows(items: List[Dict[str, object]], cap: int) -> List[Dict[str, object]]:
    chosen: List[Dict[str, object]] = []
    seen_base_ids: set[str] = set()
    action_counts: Dict[str, int] = defaultdict(int)
    grouped: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in sort_rows(items):
        grouped[str(row["gt_action_id"])].append(row)

    # First pass: take the best row from each action to guarantee minimum action coverage.
    for action_id in sorted(
        grouped,
        key=lambda action: (
            -VERDICT_RANK[str(grouped[action][0]["verdict"])],
            -int(grouped[action][0]["score"]),
            action,
        ),
    ):
        if len(chosen) >= cap:
            break
        for row in grouped[action_id]:
            base_id = base_case_id(str(row["case_id"]))
            if base_id in seen_base_ids:
                continue
            chosen.append(row)
            seen_base_ids.add(base_id)
            action_counts[action_id] += 1
            break

    # Second pass: fill remaining slots round-robin with a soft per-action cap.
    progress = True
    while len(chosen) < cap and progress:
        progress = False
        for action_id in sorted(
            grouped,
            key=lambda action: (
                action_counts[action],
                -VERDICT_RANK[str(grouped[action][0]["verdict"])],
                -int(grouped[action][0]["score"]),
                action,
            ),
        ):
            if len(chosen) >= cap:
                break
            if action_counts[action_id] >= MAX_PER_ACTION:
                continue
            for row in grouped[action_id]:
                base_id = base_case_id(str(row["case_id"]))
                if base_id in seen_base_ids:
                    continue
                chosen.append(row)
                seen_base_ids.add(base_id)
                action_counts[action_id] += 1
                progress = True
                break

    # Final pass: if the dataset still has capacity, fill with the strongest leftovers.
    if len(chosen) < cap:
        for row in sort_rows(items):
            if len(chosen) >= cap:
                break
            base_id = base_case_id(str(row["case_id"]))
            if base_id in seen_base_ids:
                continue
            chosen.append(row)
            seen_base_ids.add(base_id)
            action_counts[str(row["gt_action_id"])] += 1
    return chosen


def load_spec_metadata() -> Dict[Tuple[str, str, str, str], Dict[str, str]]:
    out: Dict[Tuple[str, str, str, str], Dict[str, str]] = {}
    specs_dir = REBUILD_ROOT / "specs"
    for path in sorted(specs_dir.glob("*.json")):
        try:
            obj = load_json(path)
        except Exception:
            continue
        datasets = obj.get("datasets")
        if not isinstance(datasets, dict):
            continue
        for dataset, items in datasets.items():
            if not isinstance(items, list):
                continue
            for item in items:
                case_id = str(item.get("case_id", "") or "")
                eval_case_id = str(item.get("eval_case_id", case_id) or case_id)
                source = str(item.get("source", "") or "")
                action_id = str(item.get("gt_action_id", "") or "")
                meta = {
                    "gt_family_id": str(item.get("gt_family_id", "") or ""),
                    "alert_match": str(item.get("alert_match", "") or ""),
                }
                for key_case in {case_id, eval_case_id}:
                    key = (str(dataset), source, key_case, action_id)
                    existing = out.get(key)
                    if existing is None:
                        out[key] = meta
                        continue
                    if (not existing.get("gt_family_id") and meta.get("gt_family_id")) or (
                        not existing.get("alert_match") and meta.get("alert_match")
                    ):
                        out[key] = meta
    return out


def load_conflicted_base_ids() -> Dict[str, set[str]]:
    conflicts: Dict[str, set[str]] = defaultdict(set)
    actions_by_base: Dict[Tuple[str, str], set[str]] = defaultdict(set)
    specs_dir = REBUILD_ROOT / "specs"
    for path in sorted(specs_dir.glob("*.json")):
        if "keepdirect" in path.name:
            continue
        try:
            obj = load_json(path)
        except Exception:
            continue
        datasets = obj.get("datasets")
        if not isinstance(datasets, dict):
            continue
        for dataset, items in datasets.items():
            if not isinstance(items, list):
                continue
            for item in items:
                case_id = str(item.get("case_id", "") or "")
                action_id = str(item.get("gt_action_id", "") or "")
                if not case_id or not action_id:
                    continue
                actions_by_base[(str(dataset), base_case_id(case_id))].add(action_id)
    for (dataset, base_id), actions in actions_by_base.items():
        if len(actions) > 1:
            conflicts[dataset].add(base_id)
    return conflicts


def load_manual_cases() -> Dict[Tuple[str, str], Dict[str, object]]:
    admission = load_json(DATASET_ADMISSION_PATH)
    allowed = {
        dataset: set(MANUAL_KEEP.get(dataset, set()))
        for dataset in ("OpenStack", "HDFS", "Hadoop")
    }
    validated = admission["datasets"]
    for dataset, payload in validated.items():
        for raw in payload.get("validated_cases", []):
            _, case_id = raw.split(":", 1)
            allowed.setdefault(dataset, set()).add(case_id)
    manual: Dict[Tuple[str, str], Dict[str, object]] = {}
    specs_dir = REBUILD_ROOT / "specs"
    for path in sorted(specs_dir.glob("*.json")):
        try:
            obj = load_json(path)
        except Exception:
            continue
        datasets = obj.get("datasets")
        if not isinstance(datasets, dict):
            continue
        for dataset, items in datasets.items():
            if not isinstance(items, list):
                continue
            for item in items:
                case_id = str(item.get("case_id", "") or "")
                if case_id not in allowed.get(dataset, set()):
                    continue
                key = (dataset, case_id)
                existing = manual.get(key)
                candidate = {
                    "case_id": case_id,
                    "eval_case_id": str(item.get("eval_case_id", "") or case_id),
                    "source": str(item.get("source", "") or ""),
                    "gt_family_id": str(item.get("gt_family_id", "") or ""),
                    "gt_action_id": str(item.get("gt_action_id", "") or ""),
                    "alert_match": str(item.get("alert_match", "") or ""),
                    "eligibility_note": "manual_keep: trusted rebuild proof/admission case",
                    "selection_score": 0.0,
                    "difficulty_score": 0.0,
                    "weak_mainline_alert": False,
                    "fixed_small_member": True,
                    "selected_alert_flags": {},
                }
                if existing is None:
                    manual[key] = candidate
                    continue
                replace = (
                    int(bool(candidate.get("gt_action_id"))),
                    int(bool(candidate.get("gt_family_id"))),
                    int(bool(candidate.get("alert_match"))),
                    int(bool(candidate.get("source"))),
                ) > (
                    int(bool(existing.get("gt_action_id"))),
                    int(bool(existing.get("gt_family_id"))),
                    int(bool(existing.get("alert_match"))),
                    int(bool(existing.get("source"))),
                )
                if replace:
                    manual[key] = candidate
    return manual


def load_probe_exclusions(
    progress_path: Path | None,
    *,
    drop_all_zero: bool,
    drop_rag_agent_toxic: bool,
) -> Dict[Tuple[str, str], List[str]]:
    if progress_path is None or not progress_path.exists():
        return {}
    rows = [
        json.loads(line)
        for line in progress_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_case: Dict[Tuple[str, str], Dict[str, Mapping[str, object]]] = defaultdict(dict)
    for row in rows:
        dataset = str(row.get("dataset", "") or "")
        case_id = str(row.get("case_id", "") or "")
        mode = str(row.get("mode", "") or "")
        if dataset and case_id and mode:
            by_case[(dataset, case_id)][mode] = row

    out: Dict[Tuple[str, str], List[str]] = {}
    for key, modes in by_case.items():
        oa = int(bool((modes.get("open_alert_only") or {}).get("e2e_success")))
        van = int(bool((modes.get("vanilla_open") or {}).get("e2e_success")))
        rag = int(bool((modes.get("rag_open") or {}).get("e2e_success")))
        agent = int(bool((modes.get("agent_open") or {}).get("e2e_success")))
        reasons: List[str] = []
        if drop_all_zero and oa == van == rag == agent == 0:
            reasons.append("all_zero")
        if drop_rag_agent_toxic and rag == 1 and agent == 0:
            reasons.append("rag_agent_toxic")
        if reasons:
            out[key] = reasons
    return out


def main() -> None:
    args = parse_args()
    manual_cases = load_manual_cases()
    spec_meta = load_spec_metadata()
    conflicted_base_ids = load_conflicted_base_ids()
    probe_exclusions = load_probe_exclusions(
        args.probe_progress,
        drop_all_zero=bool(args.drop_all_zero),
        drop_rag_agent_toxic=bool(args.drop_rag_agent_toxic),
    )
    aggregated: Dict[Tuple[str, str, str], Dict[str, object]] = {}

    for ranked_path in DEFAULT_RANKED_INPUTS:
        if not ranked_path.exists():
            continue
        obj = load_json(ranked_path)
        for rank_row in obj.get("ranked_cases", []):
            row = row_from_ranked(rank_row, ranked_path.name)
            meta = spec_meta.get((row["dataset"], row["source"], row["case_id"], row["gt_action_id"])) or spec_meta.get(
                (row["dataset"], row["source"], row["eval_case_id"], row["gt_action_id"])
            )
            if meta:
                if not row["gt_family_id"]:
                    row["gt_family_id"] = meta["gt_family_id"]
                if not row["alert_match"]:
                    row["alert_match"] = meta["alert_match"]
            if base_case_id(str(row["case_id"])) in conflicted_base_ids.get(str(row["dataset"]), set()):
                continue
            if (str(row["dataset"]), str(row["eval_case_id"])) in probe_exclusions:
                continue
            key = dedup_key(row)
            if choose_better(aggregated.get(key), row):
                aggregated[key] = row

    selected_by_dataset: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for (dataset, case_base, action_id), row in aggregated.items():
        if row["verdict"] not in {"shortlist", "provisional"}:
            continue
        selected_by_dataset[dataset].append(row)

    final_datasets: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    seen_dataset_base: Dict[str, set[str]] = defaultdict(set)
    for dataset, items in selected_by_dataset.items():
        for row in select_diverse_rows(items, int(args.per_dataset_cap)):
            base_id = base_case_id(str(row["case_id"]))
            eval_case_id = str(row.get("eval_case_id", "") or row["case_id"])
            final_datasets[dataset].append(
                {
                    "case_id": str(row["case_id"]),
                    "eval_case_id": eval_case_id,
                    "source": str(row["source"]),
                    "gt_family_id": str(row["gt_family_id"]),
                    "gt_action_id": str(row["gt_action_id"]),
                    "alert_match": str(row.get("alert_match", "") or ""),
                    "eligibility_note": (
                        f"{row['verdict']} score={row['score']} "
                        f"agent={row['agent_e2e']} rag={row['rag_e2e']} vanilla={row['vanilla_e2e']} "
                        f"heuristic={row['heuristic_alert_e2e']} source={row['source_file']} "
                        f"note={row['eligibility_note']}"
                    ),
                }
            )
            seen_dataset_base[dataset].add(base_id)

    for (dataset, case_id), item in manual_cases.items():
        if len(final_datasets[dataset]) >= int(args.per_dataset_cap):
            continue
        base_id = base_case_id(case_id)
        if base_id in conflicted_base_ids.get(dataset, set()):
            continue
        if base_id in seen_dataset_base[dataset]:
            continue
        if (dataset, str(item.get("eval_case_id", "") or case_id)) in probe_exclusions:
            continue
        final_datasets[dataset].append(item)
        seen_dataset_base[dataset].add(base_id)

    payload = {
        "benchmark_id": args.benchmark_id,
        "benchmark_kind": "mid_seed_candidate_pool",
        "purpose": "Mid-first candidate seed assembled from calibrated local probes plus trusted manual rebuild cases.",
        "formal_small_ready": False,
        "paid_api_allowed": False,
        "contract_path": str(REBUILD_ROOT / "configs" / "contract_v1_20260318.json"),
        "datasets": dict(final_datasets),
        "filter_report": {
            "probe_progress": str(args.probe_progress) if args.probe_progress else "",
            "drop_all_zero": bool(args.drop_all_zero),
            "drop_rag_agent_toxic": bool(args.drop_rag_agent_toxic),
            "excluded_cases": [
                {
                    "dataset": dataset,
                    "eval_case_id": case_id,
                    "reasons": reasons,
                }
                for (dataset, case_id), reasons in sorted(probe_exclusions.items())
            ],
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"{args.benchmark_id}.json"
    write_json(output_path, payload)
    print(output_path)
    print(json.dumps({dataset: len(items) for dataset, items in final_datasets.items()}, indent=2, ensure_ascii=False))
    if conflicted_base_ids:
        print(
            json.dumps(
                {
                    dataset: sorted(items)
                    for dataset, items in conflicted_base_ids.items()
                    if items
                },
                indent=2,
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
