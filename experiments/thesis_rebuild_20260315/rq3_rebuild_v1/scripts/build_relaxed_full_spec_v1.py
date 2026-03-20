from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
REBUILD_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = REBUILD_ROOT.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.thesis_rebuild_20260315.rq34.scripts.frozen_catalog_v2_20260318 import (
    ACTION_CATALOG,
    family_for_action,
)
from experiments.thesis_rebuild_20260315.shared.utils.io_utils import write_json


DEFAULT_OUTPUT_DIR = REBUILD_ROOT / "specs"
DEFAULT_CONTRACT_PATH = REBUILD_ROOT / "configs" / "contract_v1_20260318.json"
DEFAULT_HDFS_OPENSTACK_REPORT = (
    PROJECT_ROOT
    / "experiments"
    / "thesis_rebuild_20260315"
    / "rq34"
    / "analysis"
    / "rq3_small_v3_diagnostic_slice_20260318"
    / "rq3_small_v3_pool_anchor_opportunities_20260318.json"
)
DEFAULT_HADOOP_CONTROL_REPORT = (
    REBUILD_ROOT
    / "analysis"
    / "rq3_hadoop_control_anchor_opportunities_v1_20260319"
    / "rq3_hadoop_control_anchor_opportunities_v1_20260319.json"
)
DEFAULT_ANALYSIS_DIR = REBUILD_ROOT / "analysis"

TARGET_COUNTS = {"HDFS": 50, "OpenStack": 50, "Hadoop": 44}
MAX_PER_BASE_FULL = {"HDFS": 2, "OpenStack": 4, "Hadoop": 2}

QUALITY_RANK = {
    "core_hard": 0,
    "core_usable": 1,
    "filler_reanchor": 2,
    "excluded": 9,
}
SELECTION_BUCKET_RANK = {
    "shortlist": 0,
    "provisional": 1,
    "usable_easy": 2,
    "filler_reanchor": 3,
    "excluded": 9,
}
VERDICT_RANK = {
    "shortlist": 4,
    "provisional": 3,
    "reject": 2,
    "reject_direct": 1,
    "preview": 0,
    "manual_keep": 5,
    "incomplete": -1,
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark-id", type=str, default="rq3_full_relaxed_144_v1_20260319")
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    ap.add_argument("--contract-path", type=Path, default=DEFAULT_CONTRACT_PATH)
    ap.add_argument("--hdfs-openstack-report", type=Path, default=DEFAULT_HDFS_OPENSTACK_REPORT)
    ap.add_argument("--hadoop-control-report", type=Path, default=DEFAULT_HADOOP_CONTROL_REPORT)
    ap.add_argument("--progress", type=Path, default=None)
    ap.add_argument("--drop-all-zero", action="store_true")
    ap.add_argument("--drop-rag-agent-toxic", action="store_true")
    ap.add_argument("--drop-oa-vanilla-toxic", action="store_true")
    ap.add_argument("--hdfs-target", type=int, default=TARGET_COUNTS["HDFS"])
    ap.add_argument("--openstack-target", type=int, default=TARGET_COUNTS["OpenStack"])
    ap.add_argument("--hadoop-target", type=int, default=TARGET_COUNTS["Hadoop"])
    ap.add_argument("--max-openstack-per-base", type=int, default=MAX_PER_BASE_FULL["OpenStack"])
    ap.add_argument("--max-hdfs-per-base", type=int, default=MAX_PER_BASE_FULL["HDFS"])
    ap.add_argument("--max-hadoop-per-base", type=int, default=MAX_PER_BASE_FULL["Hadoop"])
    ap.add_argument("--preview-cap-openstack", type=int, default=4)
    ap.add_argument("--preview-cap-hdfs", type=int, default=4)
    ap.add_argument("--preview-cap-hadoop", type=int, default=6)
    return ap.parse_args()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def norm(text: str) -> str:
    return " ".join(str(text or "").strip().lower().split())


def base_case_id(case_id: str) -> str:
    return re.sub(r"__[^_]+$", "", str(case_id))


def short_action_tag(action_id: str) -> str:
    tail = str(action_id or "").split("_", 1)[-1]
    cleaned = re.sub(r"[^a-z0-9]+", "", tail.lower())
    return cleaned[:18] or "alt"


def mode_flag(modes: Mapping[str, Mapping[str, object]], mode: str, key: str) -> int:
    return int(bool((modes.get(mode) or {}).get(key)))


def toxicity_flags_from_modes(modes: Mapping[str, Mapping[str, object]]) -> List[str]:
    oa = mode_flag(modes, "open_alert_only", "e2e_success")
    van = mode_flag(modes, "vanilla_open", "e2e_success")
    rag = mode_flag(modes, "rag_open", "e2e_success")
    agent = mode_flag(modes, "agent_open", "e2e_success")
    flags: List[str] = []
    if oa == van == rag == agent == 0:
        flags.append("all_zero")
    if rag == 1 and agent == 0:
        flags.append("rag_agent_toxic")
    if oa == 1 and van == 1 and rag == 0 and agent == 0:
        flags.append("oa_vanilla_toxic")
    return flags


def exclusion_flags_from_progress(
    progress_path: Path | None,
    *,
    drop_all_zero: bool,
    drop_rag_agent_toxic: bool,
    drop_oa_vanilla_toxic: bool,
) -> Dict[Tuple[str, str], List[str]]:
    if progress_path is None or not progress_path.exists():
        return {}
    rows = [json.loads(line) for line in progress_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_case: Dict[Tuple[str, str], Dict[str, Mapping[str, object]]] = defaultdict(dict)
    for row in rows:
        by_case[(str(row["dataset"]), str(row["case_id"]))][str(row["mode"])] = row

    out: Dict[Tuple[str, str], List[str]] = {}
    for key, modes in by_case.items():
        oa = mode_flag(modes, "open_alert_only", "e2e_success")
        van = mode_flag(modes, "vanilla_open", "e2e_success")
        rag = mode_flag(modes, "rag_open", "e2e_success")
        agent = mode_flag(modes, "agent_open", "e2e_success")
        flags: List[str] = []
        if drop_all_zero and oa == van == rag == agent == 0:
            flags.append("all_zero")
        if drop_rag_agent_toxic and rag == 1 and agent == 0:
            flags.append("rag_agent_toxic")
        if drop_oa_vanilla_toxic and oa == 1 and van == 1 and rag == 0 and agent == 0:
            flags.append("oa_vanilla_toxic")
        if flags:
            out[key] = flags
    return out


def load_spec_metadata() -> Dict[Tuple[str, str, str, str], Dict[str, str]]:
    out: Dict[Tuple[str, str, str, str], Dict[str, str]] = {}
    for path in sorted((REBUILD_ROOT / "specs").glob("*.json")):
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
                    out[(str(dataset), source, key_case, action_id)] = meta
    return out


def directness_from_note(text: str) -> float:
    note = str(text or "")
    match = re.search(r"alt_directness=([0-9.]+)", note)
    if match:
        return float(match.group(1))
    return 999.0


def load_spec_fallback_candidates(
    *,
    progress_exclusions: Mapping[Tuple[str, str], Sequence[str]],
) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for path in sorted((REBUILD_ROOT / "specs").glob("*.json")):
        try:
            obj = load_json(path)
        except Exception:
            continue
        benchmark_kind = str(obj.get("benchmark_kind", "") or "")
        benchmark_id = str(obj.get("benchmark_id", "") or "")
        if benchmark_kind == "rq3_relaxed_full_local" or benchmark_id.startswith("rq3_full_relaxed_"):
            continue
        datasets = obj.get("datasets")
        if not isinstance(datasets, dict):
            continue
        for dataset, items in datasets.items():
            if not isinstance(items, list):
                continue
            for item in items:
                gt_action_id = str(item.get("gt_action_id", "") or "")
                raw_eval_case_id = str(item.get("eval_case_id", "") or item.get("case_id", ""))
                case_id = base_case_id(raw_eval_case_id)
                eval_case_id = raw_eval_case_id
                if raw_eval_case_id == case_id and gt_action_id:
                    eval_case_id = f"{case_id}__{short_action_tag(gt_action_id)}sf"
                gt_family_id = str(item.get("gt_family_id", "") or family_for_action(dataset, gt_action_id) or "")
                if not eval_case_id or not gt_action_id:
                    continue
                note = str(item.get("eligibility_note", "") or "")
                toxic_flags = list(progress_exclusions.get((str(dataset), eval_case_id), []))
                quality_tier = "excluded" if toxic_flags else (
                    "core_usable" if ("mid_seed" in benchmark_id or "triad_proof_slice" in benchmark_id) else "filler_reanchor"
                )
                selection_bucket = "excluded" if toxic_flags else (
                    "usable_easy" if quality_tier == "core_usable" else "filler_reanchor"
                )
                out.append(
                    {
                        "dataset": str(dataset),
                        "case_id": case_id,
                        "eval_case_id": eval_case_id,
                        "source": str(item.get("source", "") or ""),
                        "gt_family_id": gt_family_id,
                        "gt_action_id": gt_action_id,
                        "alert_match": str(item.get("alert_match", "") or ""),
                        "selection_score": float(item.get("selection_score", 0.0) or 0.0),
                        "difficulty_score": float(item.get("difficulty_score", 0.0) or 0.0),
                        "weak_mainline_alert": bool(item.get("weak_mainline_alert", False)),
                        "selected_alert_flags": dict(item.get("selected_alert_flags", {}) or {}),
                        "verdict": "manual_keep",
                        "score": 0,
                        "selection_bucket": selection_bucket,
                        "quality_tier": quality_tier,
                        "toxicity_flags": toxic_flags,
                        "base_incident_id": base_case_id(eval_case_id),
                        "reanchor_group_id": f"{dataset}:{str(item.get('source', '') or '')}:{case_id}:{gt_action_id}",
                        "is_duplicate_reanchor": False,
                        "candidate_origin": "spec_fallback",
                        "scored_by_probe": False,
                        "source_file": path.name,
                        "directness_score": directness_from_note(note),
                        "heuristic_alert_e2e": 0,
                        "open_alert_only_e2e": 0,
                        "vanilla_e2e": 0,
                        "rag_e2e": 0,
                        "agent_e2e": 0,
                        "inclusion_reason": f"fallback from spec {path.name}",
                        "eligibility_note": note,
                    }
                )
    return out


def choose_better(existing: Mapping[str, object] | None, candidate: Mapping[str, object]) -> bool:
    if existing is None:
        return True
    current = (
        QUALITY_RANK[str(candidate["quality_tier"])],
        SELECTION_BUCKET_RANK[str(candidate["selection_bucket"])],
        -VERDICT_RANK[str(candidate["verdict"])],
        -int(candidate.get("score", 0)),
        -int(bool(candidate.get("agent_e2e", 0))),
        -int(bool(candidate.get("rag_e2e", 0))),
        int(bool(candidate.get("open_alert_only_e2e", 0))),
        float(candidate.get("directness_score", 999.0)),
        -float(candidate.get("difficulty_score", 0.0)),
        -float(candidate.get("selection_score", 0.0)),
    )
    prev = (
        QUALITY_RANK[str(existing["quality_tier"])],
        SELECTION_BUCKET_RANK[str(existing["selection_bucket"])],
        -VERDICT_RANK[str(existing["verdict"])],
        -int(existing.get("score", 0)),
        -int(bool(existing.get("agent_e2e", 0))),
        -int(bool(existing.get("rag_e2e", 0))),
        int(bool(existing.get("open_alert_only_e2e", 0))),
        float(existing.get("directness_score", 999.0)),
        -float(existing.get("difficulty_score", 0.0)),
        -float(existing.get("selection_score", 0.0)),
    )
    return current < prev


def quality_tier_for_ranked(verdict: str, modes: Mapping[str, Mapping[str, object]], toxic_flags: Sequence[str]) -> str:
    if toxic_flags:
        return "excluded"
    oa = mode_flag(modes, "open_alert_only", "e2e_success")
    van = mode_flag(modes, "vanilla_open", "e2e_success")
    rag = mode_flag(modes, "rag_open", "e2e_success")
    agent = mode_flag(modes, "agent_open", "e2e_success")
    if verdict == "shortlist" and oa == 0 and not (van == rag == agent == 1):
        return "core_hard"
    if verdict in {"shortlist", "provisional"}:
        return "core_usable"
    if verdict == "reject_direct":
        return "filler_reanchor"
    if max(van, rag, agent) > 0:
        return "core_usable" if oa == 0 else "filler_reanchor"
    return "filler_reanchor"


def selection_bucket_for_ranked(verdict: str, modes: Mapping[str, Mapping[str, object]], toxic_flags: Sequence[str]) -> str:
    if toxic_flags:
        return "excluded"
    if verdict == "shortlist":
        return "shortlist"
    if verdict == "provisional":
        return "provisional"
    oa = mode_flag(modes, "open_alert_only", "e2e_success")
    van = mode_flag(modes, "vanilla_open", "e2e_success")
    rag = mode_flag(modes, "rag_open", "e2e_success")
    agent = mode_flag(modes, "agent_open", "e2e_success")
    if verdict == "reject_direct":
        return "filler_reanchor"
    if max(van, rag, agent) > 0 or (oa == 0 and any(mode_flag(modes, name, "rca_success") for name in ("vanilla_open", "rag_open", "agent_open"))):
        return "usable_easy"
    return "filler_reanchor"


def ranked_candidate_from_row(
    row: Mapping[str, object],
    *,
    source_file: str,
    spec_meta: Mapping[Tuple[str, str, str, str], Mapping[str, str]],
    progress_exclusions: Mapping[Tuple[str, str], Sequence[str]],
) -> Dict[str, object]:
    dataset = str(row["dataset"])
    eval_case_id = str(row.get("eval_case_id", "") or row["case_id"])
    pool_case_id = base_case_id(eval_case_id)
    source = str(row.get("source", "") or "")
    action_id = str(row["gt_action_id"])
    meta = spec_meta.get((dataset, source, pool_case_id, action_id)) or spec_meta.get((dataset, source, eval_case_id, action_id)) or {}
    gt_family_id = str(row.get("gt_family_id", "") or meta.get("gt_family_id") or family_for_action(dataset, action_id) or "")
    alert_match = str(meta.get("alert_match", "") or "")
    modes = dict(row.get("modes", {}) or {})
    toxic_flags = list(dict.fromkeys([*toxicity_flags_from_modes(modes), *progress_exclusions.get((dataset, eval_case_id), [])]))
    verdict = str(row.get("verdict", "reject"))
    selection_bucket = selection_bucket_for_ranked(verdict, modes, toxic_flags)
    quality_tier = quality_tier_for_ranked(verdict, modes, toxic_flags)
    oa = mode_flag(modes, "open_alert_only", "e2e_success")
    heuristic = mode_flag(modes, "heuristic_alert", "e2e_success")
    vanilla = mode_flag(modes, "vanilla_open", "e2e_success")
    rag = mode_flag(modes, "rag_open", "e2e_success")
    agent = mode_flag(modes, "agent_open", "e2e_success")
    note = str(row.get("eligibility_note", "") or "")
    directness_score = directness_from_note(note)
    if directness_score >= 999.0 and "reject_direct" == verdict:
        directness_score = 0.0
    return {
        "dataset": dataset,
        "case_id": pool_case_id,
        "eval_case_id": eval_case_id,
        "source": source,
        "gt_family_id": gt_family_id,
        "gt_action_id": action_id,
        "alert_match": alert_match,
        "selection_score": float(row.get("selection_score", 0.0) or 0.0),
        "difficulty_score": float(row.get("difficulty_score", 0.0) or 0.0),
        "weak_mainline_alert": bool(row.get("weak_mainline_alert", False)),
        "selected_alert_flags": dict(row.get("selected_alert_flags", {}) or {}),
        "verdict": verdict,
        "score": int(row.get("score", 0) or 0),
        "selection_bucket": selection_bucket,
        "quality_tier": quality_tier,
        "toxicity_flags": toxic_flags,
        "base_incident_id": base_case_id(eval_case_id),
        "reanchor_group_id": f"{dataset}:{source}:{pool_case_id}:{action_id}",
        "is_duplicate_reanchor": False,
        "candidate_origin": "ranked_probe",
        "scored_by_probe": True,
        "source_file": source_file,
        "directness_score": directness_score,
        "heuristic_alert_e2e": heuristic,
        "open_alert_only_e2e": oa,
        "vanilla_e2e": vanilla,
        "rag_e2e": rag,
        "agent_e2e": agent,
        "inclusion_reason": (
            f"ranked:{verdict} score={int(row.get('score', 0) or 0)} "
            f"oa={oa} vanilla={vanilla} rag={rag} agent={agent} source={source_file}"
        ),
        "eligibility_note": note,
    }


def preview_candidate(
    *,
    dataset: str,
    source: str,
    case_id: str,
    gt_action_id: str,
    preview: Mapping[str, object],
    preview_rank: int,
    selection_score: float,
    difficulty_score: float,
    selected_alert_flags: Mapping[str, object],
    current_directness: float,
    eligibility_note: str,
) -> Dict[str, object]:
    action_tag = short_action_tag(gt_action_id)
    eval_case_id = f"{case_id}__{action_tag}pv{preview_rank:02d}"
    directness_score = float(preview.get("directness_score", 999.0) or 999.0)
    return {
        "dataset": dataset,
        "case_id": case_id,
        "eval_case_id": eval_case_id,
        "source": source,
        "gt_family_id": str(family_for_action(dataset, gt_action_id) or ""),
        "gt_action_id": gt_action_id,
        "alert_match": str(preview.get("line", "") or ""),
        "selection_score": float(selection_score),
        "difficulty_score": float(difficulty_score),
        "weak_mainline_alert": directness_score <= 2.0,
        "selected_alert_flags": dict(selected_alert_flags or {}),
        "verdict": "preview",
        "score": 0,
        "selection_bucket": "filler_reanchor",
        "quality_tier": "filler_reanchor",
        "toxicity_flags": [],
        "base_incident_id": base_case_id(eval_case_id),
        "reanchor_group_id": f"{dataset}:{source}:{case_id}:{gt_action_id}",
        "is_duplicate_reanchor": False,
        "candidate_origin": "report_preview",
        "scored_by_probe": False,
        "source_file": "preview_report",
        "directness_score": directness_score,
        "heuristic_alert_e2e": 0,
        "open_alert_only_e2e": 0,
        "vanilla_e2e": 0,
        "rag_e2e": 0,
        "agent_e2e": 0,
        "inclusion_reason": (
            f"preview current_directness={current_directness:.1f} alt_directness={directness_score:.1f} "
            f"offset={int(preview.get('offset', 0) or 0)} rank={preview_rank}"
        ),
        "eligibility_note": eligibility_note,
    }


def iter_ranked_files(analysis_dir: Path) -> Iterable[Path]:
    for path in sorted(analysis_dir.rglob("*ranked*.json")):
        yield path


def load_ranked_candidates(
    *,
    analysis_dir: Path,
    spec_meta: Mapping[Tuple[str, str, str, str], Mapping[str, str]],
    progress_exclusions: Mapping[Tuple[str, str], Sequence[str]],
) -> Dict[Tuple[str, str, str], Dict[str, object]]:
    aggregated: Dict[Tuple[str, str, str], Dict[str, object]] = {}
    for ranked_path in iter_ranked_files(analysis_dir):
        try:
            obj = load_json(ranked_path)
        except Exception:
            continue
        ranked_rows = obj.get("ranked_cases")
        if not isinstance(ranked_rows, list):
            continue
        for row in ranked_rows:
            candidate = ranked_candidate_from_row(
                row,
                source_file=ranked_path.name,
                spec_meta=spec_meta,
                progress_exclusions=progress_exclusions,
            )
            key = (
                candidate["dataset"],
                str(candidate["eval_case_id"]),
                candidate["gt_action_id"],
            )
            if choose_better(aggregated.get(key), candidate):
                aggregated[key] = candidate
    return aggregated


def load_hdfs_openstack_preview_candidates(
    report_path: Path,
    *,
    preview_caps: Mapping[str, int],
) -> List[Dict[str, object]]:
    payload = load_json(report_path)
    out: List[Dict[str, object]] = []
    for dataset in ("HDFS", "OpenStack"):
        action_map = (((payload.get("datasets") or {}).get(dataset) or {}).get("actions") or {})
        for action_id, rows in action_map.items():
            for row in rows:
                source = str(row.get("pool_source", "") or "")
                case_id = str(row.get("case_id", "") or "")
                previews = list(row.get("candidate_preview") or [])
                if not source or not case_id or not previews:
                    continue
                for idx, preview in enumerate(previews[: int(preview_caps[dataset])], start=1):
                    out.append(
                        preview_candidate(
                            dataset=dataset,
                            source=source,
                            case_id=case_id,
                            gt_action_id=str(action_id),
                            preview=preview,
                            preview_rank=idx,
                            selection_score=float(row.get("selection_score", 0.0) or 0.0),
                            difficulty_score=float(row.get("difficulty_score", 0.0) or 0.0),
                            selected_alert_flags=dict(row.get("selected_alert_flags", {}) or {}),
                            current_directness=float(((row.get("current_metrics") or {}).get("directness_score", 0.0) or 0.0)),
                            eligibility_note="preview from rq3_small_v3_pool_anchor_opportunities",
                        )
                    )
    return out


def load_hadoop_control_preview_candidates(report_path: Path, *, preview_cap: int) -> List[Dict[str, object]]:
    payload = load_json(report_path)
    out: List[Dict[str, object]] = []
    for action_id, rows in (payload.get("actions") or {}).items():
        for row in rows:
            source = str(row.get("pool_source", "") or "benchmark_v2")
            case_id = str(row.get("case_id", "") or "")
            previews = list(row.get("candidate_preview") or [])
            if not case_id or not previews:
                continue
            for idx, preview in enumerate(previews[:preview_cap], start=1):
                out.append(
                    preview_candidate(
                        dataset="Hadoop",
                        source=source,
                        case_id=case_id,
                        gt_action_id=str(action_id),
                        preview=preview,
                        preview_rank=idx,
                        selection_score=0.0,
                        difficulty_score=0.0,
                        selected_alert_flags={},
                        current_directness=float(((row.get("current_metrics") or {}).get("directness_score", 0.0) or 0.0)),
                        eligibility_note="preview from rq3_hadoop_control_anchor_opportunities",
                    )
                )
    return out


def sort_candidates(candidates: Sequence[Mapping[str, object]]) -> List[Mapping[str, object]]:
    return sorted(
        candidates,
        key=lambda row: (
            QUALITY_RANK[str(row["quality_tier"])],
            SELECTION_BUCKET_RANK[str(row["selection_bucket"])],
            -VERDICT_RANK[str(row["verdict"])],
            int(bool(row.get("is_duplicate_reanchor", False))),
            int(bool(row.get("open_alert_only_e2e", 0))),
            -int(bool(row.get("agent_e2e", 0))),
            -int(bool(row.get("rag_e2e", 0))),
            -int(bool(row.get("vanilla_e2e", 0))),
            float(row.get("directness_score", 999.0)),
            -float(row.get("difficulty_score", 0.0)),
            -float(row.get("selection_score", 0.0)),
            str(row.get("eval_case_id", "")),
        ),
    )


def action_soft_cap(dataset: str, target: int) -> int:
    action_count = max(1, len(ACTION_CATALOG.get(dataset, {})))
    return int(math.ceil(target / action_count) + 2)


def add_selected(
    selected: List[Dict[str, object]],
    chosen_keys: set[Tuple[str, str, str]],
    base_counts: MutableMapping[str, int],
    action_counts: MutableMapping[str, int],
    candidate: Mapping[str, object],
) -> None:
    key = (str(candidate["dataset"]), str(candidate["eval_case_id"]), str(candidate["gt_action_id"]))
    if key in chosen_keys:
        return
    selected.append(dict(candidate))
    chosen_keys.add(key)
    base_counts[str(candidate["base_incident_id"])] += 1
    action_counts[str(candidate["gt_action_id"])] += 1


def select_dataset_candidates(
    dataset: str,
    candidates: Sequence[Mapping[str, object]],
    *,
    target: int,
    max_per_base: int,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    usable = [dict(row) for row in candidates if str(row["quality_tier"]) != "excluded"]
    chosen: List[Dict[str, object]] = []
    chosen_keys: set[Tuple[str, str, str]] = set()
    base_counts: Dict[str, int] = defaultdict(int)
    action_counts: Dict[str, int] = defaultdict(int)
    soft_cap = action_soft_cap(dataset, target)

    by_action: Dict[str, List[Mapping[str, object]]] = defaultdict(list)
    for row in sort_candidates(usable):
        by_action[str(row["gt_action_id"])].append(row)

    action_order = sorted(
        by_action,
        key=lambda action_id: (
            QUALITY_RANK[str(by_action[action_id][0]["quality_tier"])],
            SELECTION_BUCKET_RANK[str(by_action[action_id][0]["selection_bucket"])],
            -VERDICT_RANK[str(by_action[action_id][0]["verdict"])],
            float(by_action[action_id][0].get("directness_score", 999.0)),
            action_id,
        ),
    )

    for action_id in action_order:
        if len(chosen) >= target:
            break
        for row in by_action[action_id]:
            base_id = str(row["base_incident_id"])
            if base_counts[base_id] >= max_per_base:
                continue
            add_selected(chosen, chosen_keys, base_counts, action_counts, row)
            break

    for row in sort_candidates(usable):
        if len(chosen) >= target:
            break
        base_id = str(row["base_incident_id"])
        action_id = str(row["gt_action_id"])
        key = (str(row["dataset"]), str(row["eval_case_id"]), action_id)
        if key in chosen_keys:
            continue
        if base_counts[base_id] >= max_per_base:
            continue
        if action_counts[action_id] >= soft_cap and str(row["selection_bucket"]) == "filler_reanchor":
            continue
        add_selected(chosen, chosen_keys, base_counts, action_counts, row)

    if len(chosen) < target:
        for row in sort_candidates(usable):
            if len(chosen) >= target:
                break
            base_id = str(row["base_incident_id"])
            action_id = str(row["gt_action_id"])
            key = (str(row["dataset"]), str(row["eval_case_id"]), action_id)
            if key in chosen_keys:
                continue
            if base_counts[base_id] >= max_per_base:
                continue
            add_selected(chosen, chosen_keys, base_counts, action_counts, row)

    if len(chosen) < target:
        for row in sort_candidates(usable):
            if len(chosen) >= target:
                break
            action_id = str(row["gt_action_id"])
            key = (str(row["dataset"]), str(row["eval_case_id"]), action_id)
            if key in chosen_keys:
                continue
            overflow_row = dict(row)
            overflow_row["inclusion_reason"] = (
                f"{str(row.get('inclusion_reason', '') or '')}; overflow_fill_after_base_cap"
            ).strip("; ")
            add_selected(chosen, chosen_keys, base_counts, action_counts, overflow_row)

    ordered = sort_candidates(chosen)
    seen_base: set[str] = set()
    for row in ordered:
        duplicate = str(row["base_incident_id"]) in seen_base
        row["is_duplicate_reanchor"] = duplicate
        seen_base.add(str(row["base_incident_id"]))

    selected_keys = {(str(row["dataset"]), str(row["eval_case_id"]), str(row["gt_action_id"])) for row in ordered}
    reserve = [
        dict(row)
        for row in sort_candidates(usable)
        if (str(row["dataset"]), str(row["eval_case_id"]), str(row["gt_action_id"])) not in selected_keys
    ]
    return ordered[:target], reserve


def summarize_selected(spec: Mapping[str, object]) -> Dict[str, object]:
    dataset_case_counts = {dataset: len(items) for dataset, items in spec["datasets"].items()}
    quality_counts = {
        dataset: dict(Counter(str(item["quality_tier"]) for item in items))
        for dataset, items in spec["datasets"].items()
    }
    bucket_counts = {
        dataset: dict(Counter(str(item["selection_bucket"]) for item in items))
        for dataset, items in spec["datasets"].items()
    }
    duplicate_counts = {
        dataset: int(sum(int(bool(item.get("is_duplicate_reanchor", False))) for item in items))
        for dataset, items in spec["datasets"].items()
    }
    action_counts = {
        dataset: dict(Counter(str(item["gt_action_id"]) for item in items))
        for dataset, items in spec["datasets"].items()
    }
    core_subset_counts = {
        dataset: int(
            sum(
                int(
                    str(item["quality_tier"]) in {"core_hard", "core_usable"}
                    and not bool(item.get("is_duplicate_reanchor", False))
                )
                for item in items
            )
        )
        for dataset, items in spec["datasets"].items()
    }
    return {
        "benchmark_id": str(spec["benchmark_id"]),
        "dataset_case_counts": dataset_case_counts,
        "quality_tier_counts": quality_counts,
        "selection_bucket_counts": bucket_counts,
        "duplicate_reanchor_counts": duplicate_counts,
        "action_counts": action_counts,
        "core_subset_counts": core_subset_counts,
        "total_cases": int(sum(dataset_case_counts.values())),
    }


def main() -> None:
    args = parse_args()
    targets = {"HDFS": int(args.hdfs_target), "OpenStack": int(args.openstack_target), "Hadoop": int(args.hadoop_target)}
    max_per_base = {
        "HDFS": int(args.max_hdfs_per_base),
        "OpenStack": int(args.max_openstack_per_base),
        "Hadoop": int(args.max_hadoop_per_base),
    }
    spec_meta = load_spec_metadata()
    progress_exclusions = exclusion_flags_from_progress(
        args.progress,
        drop_all_zero=bool(args.drop_all_zero),
        drop_rag_agent_toxic=bool(args.drop_rag_agent_toxic),
        drop_oa_vanilla_toxic=bool(args.drop_oa_vanilla_toxic),
    )

    aggregated_ranked = load_ranked_candidates(
        analysis_dir=args.analysis_dir,
        spec_meta=spec_meta,
        progress_exclusions=progress_exclusions,
    )
    preview_candidates = (
        load_hdfs_openstack_preview_candidates(
            args.hdfs_openstack_report,
            preview_caps={"HDFS": int(args.preview_cap_hdfs), "OpenStack": int(args.preview_cap_openstack)},
        )
        + load_hadoop_control_preview_candidates(
            args.hadoop_control_report,
            preview_cap=int(args.preview_cap_hadoop),
        )
    )
    spec_fallback_candidates = load_spec_fallback_candidates(progress_exclusions=progress_exclusions)

    aggregated: Dict[Tuple[str, str, str], Dict[str, object]] = dict(aggregated_ranked)
    for candidate in [*preview_candidates, *spec_fallback_candidates]:
        key = (candidate["dataset"], str(candidate["eval_case_id"]), candidate["gt_action_id"])
        if choose_better(aggregated.get(key), candidate):
            aggregated[key] = candidate

    by_dataset: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for candidate in aggregated.values():
        by_dataset[str(candidate["dataset"])].append(candidate)

    final_datasets: Dict[str, List[Dict[str, object]]] = {}
    reserve_by_dataset: Dict[str, List[Dict[str, object]]] = {}
    for dataset in ("HDFS", "OpenStack", "Hadoop"):
        selected, reserve = select_dataset_candidates(
            dataset,
            by_dataset.get(dataset, []),
            target=targets[dataset],
            max_per_base=max_per_base[dataset],
        )
        final_datasets[dataset] = selected
        reserve_by_dataset[dataset] = reserve[: max(0, targets[dataset] // 2)]

    spec = {
        "benchmark_id": args.benchmark_id,
        "benchmark_kind": "rq3_relaxed_full_local",
        "purpose": (
            "Coverage-first relaxed full RQ3 benchmark for local qwen3.5 evaluation. "
            "Keeps method fairness fixed while mixing core and filler reanchors to reach 144 cases."
        ),
        "formal_small_ready": False,
        "paid_api_allowed": False,
        "contract_path": str(args.contract_path),
        "datasets": final_datasets,
        "selection_policy": {
            "targets": targets,
            "max_per_base": max_per_base,
            "drop_all_zero": bool(args.drop_all_zero),
            "drop_rag_agent_toxic": bool(args.drop_rag_agent_toxic),
            "drop_oa_vanilla_toxic": bool(args.drop_oa_vanilla_toxic),
            "preview_caps": {
                "HDFS": int(args.preview_cap_hdfs),
                "OpenStack": int(args.preview_cap_openstack),
                "Hadoop": int(args.preview_cap_hadoop),
            },
            "progress_path": str(args.progress) if args.progress else "",
        },
        "reserve_candidates": reserve_by_dataset,
    }

    summary = summarize_selected(spec)
    output_path = args.output_dir / f"{args.benchmark_id}.json"
    summary_path = args.output_dir / f"{args.benchmark_id}_summary.json"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_path, spec)
    write_json(summary_path, summary)
    print(output_path)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
