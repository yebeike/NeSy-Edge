from __future__ import annotations

import json
import os
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

_SCRIPT_DIR = Path(__file__).resolve().parent
_REDESIGN_ROOT = _SCRIPT_DIR.parent
_REBUILD_ROOT = _REDESIGN_ROOT.parents[0]
_PROJECT_ROOT = _REDESIGN_ROOT.parents[2]

_AUDIT_ROOT = _REBUILD_ROOT / "rq2_fullcase_audit_20260318"
_AUDIT_SCRIPTS = _AUDIT_ROOT / "scripts"
_AUDIT_RESULTS = _AUDIT_ROOT / "results"

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_PROJECT_ROOT / "experiments" / "rq123_e2e") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "experiments" / "rq123_e2e"))
if str(_AUDIT_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_AUDIT_SCRIPTS))

from build_e2e_scaled_benchmark import _extract_target_line_and_template  # type: ignore
from rq2_fullcase_audit_common_20260318 import (  # type: ignore
    OLD_MODIFIED_GRAPH_PATH,
    RAW_OPENSTACK_2,
    UNDIRECTED_RELATIONS,
    best_graph_template,
    dataset_to_domain,
    domain_template_pool,
    evaluate_graph_rows,
    family_of,
    fuzzy_match,
    iter_with_progress,
    load_json,
    manual_prior_pair_overlap,
    map_prior_templates_to_graph_candidates,
    summarize_rows,
    write_json,
)


RESULTS_DIR = _REDESIGN_ROOT / "results"
REPORTS_DIR = _REDESIGN_ROOT / "reports"

OPENSTACK_WINDOW_RADIUS = 18
EXTENDED_MIN_GAP = 12
CORE_MIN_GAP = 240
CORE_MAX_PER_PAIR_TOTAL = 36
CORE_MAX_PER_PAIR_PER_FILE = 12

OPENSTACK_CASE_SPECS: List[Dict[str, object]] = [
    {
        "name": "unknown_base",
        "keyword": "unknown base file:",
        "count": 10,
        "effect_template": "Unknown base file: <*>",
        "root_template": "Removable base files: <*>",
    },
    {
        "name": "base_too_young",
        "keyword": "base or swap file too young to remove:",
        "count": 15,
        "effect_template": "Base or swap file too young to remove: <*>",
        "root_template": "Removable base files: <*>",
    },
    {
        "name": "removable_base",
        "keyword": "removable base files:",
        "count": 10,
        "effect_template": "Removable base files: <*>",
        "root_template": "Active base files: <*>",
    },
    {
        "name": "pending_task",
        "keyword": "pending task",
        "count": 8,
        "effect_template": "nova-compute.log.<*>.<*>-<*>-<*>_<*>:<*>:<*>-<*>-<*>:<*>:<*>.<*> INFO nova.compute.manager [-] [instance: <*>] During sync_power_state the instance has a pending task <*> Skip.",
        "root_template": "While synchronizing instance power states, found <*> instances in the database and <*> instances on the hypervisor.",
    },
    {
        "name": "instance_sync_recreated",
        "keyword": "re-created its instancelist",
        "count": 5,
        "effect_template": "The instance sync for host 'cp-<*>.slowvm<*>.tcloud-pg<*>.utah.cloudlab.us' did not match. Re-created its InstanceList.",
        "root_template": "Successfully synced instances from host 'cp-<*>.slowvm<*>.tcloud-pg<*>.utah.cloudlab.us'.",
    },
    {
        "name": "token_validation",
        "keyword": "bad response code while validating token",
        "count": 2,
        "effect_template": "Bad response code while validating token: <*>",
        "root_template": "Identity response: <!DOCTYPE HTML PUBLIC \"-<*> HTML <*>.<*><*>\">",
    },
]

CANDIDATE_POOL_PATH = RESULTS_DIR / "rq2_openstack_redesign_candidate_pool_20260318.json"
EXTENDED_BENCH_PATH = RESULTS_DIR / "rq2_openstack_redesign_openstack_extended_benchmark_20260318.json"
CORE_BENCH_PATH = RESULTS_DIR / "rq2_openstack_redesign_openstack_core_benchmark_20260318.json"
HDFS_FROZEN_BENCH_PATH = RESULTS_DIR / "rq2_openstack_redesign_hdfs_frozen_benchmark_20260318.json"
COMBINED_EVAL_BENCH_PATH = RESULTS_DIR / "rq2_openstack_redesign_eval_benchmark_20260318.json"

GRAPH_FILES = {
    "modified": RESULTS_DIR / "gt_causal_knowledge_modified_frozen_redesign_20260318.json",
    "original_dynotears": RESULTS_DIR / "gt_causal_knowledge_original_dynotears_redesign_20260318.json",
    "pearson_hypothesis": RESULTS_DIR / "gt_causal_knowledge_pearson_hypothesis_redesign_20260318.json",
    "pc_cpdag_hypothesis": RESULTS_DIR / "gt_causal_knowledge_pc_cpdag_hypothesis_redesign_20260318.json",
}

SOURCE_GRAPH_FILES = {
    "modified": _AUDIT_RESULTS / "gt_causal_knowledge_modified_frozen_audit_20260318.json",
    "original_dynotears": _AUDIT_RESULTS / "gt_causal_knowledge_original_dynotears_audit_20260318.json",
    "pearson_hypothesis": _AUDIT_RESULTS / "gt_causal_knowledge_pearson_hypothesis_audit_20260318.json",
    "pc_cpdag_hypothesis": _AUDIT_RESULTS / "gt_causal_knowledge_pc_cpdag_hypothesis_audit_20260318.json",
}

AUDIT_BENCHMARK_EVAL_PATH = _AUDIT_RESULTS / "rq2_fullcase_audit_benchmark_evaluable_20260318.json"


def ensure_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _window_lines(lines: List[str], idx: int, radius: int = OPENSTACK_WINDOW_RADIUS) -> List[str]:
    lo = max(0, idx - radius)
    hi = min(len(lines), idx + radius + 1)
    return [line for line in lines[lo:hi] if line.strip()]


def _parse_templates_from_lines(lines: List[str]) -> List[str]:
    parsed: List[str] = []
    for line in lines:
        _, tpl = _extract_target_line_and_template([line], "OpenStack")
        tpl = str(tpl or "").strip()
        if tpl:
            parsed.append(tpl)
    return parsed


def _raw_openstack_files() -> List[Path]:
    return [RAW_OPENSTACK_2 / name for name in sorted(os.listdir(RAW_OPENSTACK_2)) if name.endswith(".log")]


def _same_family_candidates(effect_graph: str, mapped_prior_candidates: List[str]) -> List[str]:
    effect_family = family_of("OpenStack", effect_graph)
    same_family: List[str] = []
    seen = set()
    for candidate in mapped_prior_candidates:
        if family_of("OpenStack", candidate) != effect_family:
            continue
        if (
            candidate == effect_graph
            or fuzzy_match(candidate, effect_graph)
            or candidate in seen
        ):
            continue
        seen.add(candidate)
        same_family.append(candidate)
    return same_family


def build_openstack_hit_candidate(
    spec: Dict[str, object],
    path: Path,
    idx: int,
    lines: List[str],
    template_pool: List[str],
) -> Dict[str, object]:
    window = _window_lines(lines, idx, OPENSTACK_WINDOW_RADIUS)
    local_idx = max(0, min(len(window) - 1, idx - max(0, idx - OPENSTACK_WINDOW_RADIUS)))
    hit_line = window[local_idx]
    _, hit_template = _extract_target_line_and_template([hit_line], "OpenStack")
    effect_graph = best_graph_template("OpenStack", [str(hit_template or ""), hit_line], template_pool)
    prior_lines = window[:local_idx]
    prior_templates = _parse_templates_from_lines(prior_lines)
    mapped_prior_candidates = map_prior_templates_to_graph_candidates("OpenStack", prior_templates, template_pool)
    same_family_candidates = _same_family_candidates(effect_graph, mapped_prior_candidates)
    cross_family_candidates = []
    seen_cross = set()
    for candidate in mapped_prior_candidates:
        if candidate in same_family_candidates or candidate == effect_graph or candidate in seen_cross:
            continue
        seen_cross.add(candidate)
        cross_family_candidates.append(candidate)

    root_graph = ""
    status = "missing_same_family_root"
    if len(same_family_candidates) == 1:
        root_graph = same_family_candidates[0]
        if manual_prior_pair_overlap("OpenStack", root_graph, effect_graph):
            status = "manual_prior_overlap"
        else:
            status = "core_eligible"
    elif len(same_family_candidates) > 1:
        status = "ambiguous_same_family_root"

    return {
        "candidate_id": f"{spec['name']}::{path.name}:{idx}",
        "dataset": "OpenStack",
        "audit_case_spec": str(spec["name"]),
        "audit_keyword": str(spec["keyword"]),
        "spec_effect_template": str(spec["effect_template"]),
        "spec_root_template": str(spec["root_template"]),
        "source": f"openstack_raw_redesign_20260318::{path.name}",
        "audit_hit_file": path.name,
        "audit_hit_line_index": idx,
        "audit_hit_local_index": local_idx,
        "audit_hit_line": hit_line,
        "audit_hit_template": str(hit_template or ""),
        "raw_log_window": window,
        "raw_log": "\n".join(window),
        "audit_prior_lines": prior_lines,
        "audit_prior_templates": prior_templates,
        "audit_mapped_prior_graph_candidates": mapped_prior_candidates,
        "audit_same_family_prior_graph_candidates": same_family_candidates,
        "audit_cross_family_prior_graph_candidates": cross_family_candidates,
        "audit_cross_family_prior_count": len(cross_family_candidates),
        "audit_noisy_cross_family_window": bool(cross_family_candidates),
        "ground_truth_template_graph": effect_graph,
        "ground_truth_root_cause_template_graph": root_graph,
        "manual_prior_pair_overlap": status == "manual_prior_overlap",
        "audit_root_status": status,
        "audit_derivation_complete": bool(effect_graph and hit_line and hit_template),
    }


def candidate_priority(candidate: Dict[str, object]) -> Tuple[int, int, int]:
    status = str(candidate.get("audit_root_status", ""))
    priority = {
        "core_eligible": 4,
        "manual_prior_overlap": 3,
        "ambiguous_same_family_root": 2,
        "missing_same_family_root": 1,
    }.get(status, 0)
    cross_family = int(candidate.get("audit_cross_family_prior_count", 0) or 0)
    return (priority, -cross_family, -len(candidate.get("audit_prior_templates", [])))


def select_diverse_candidates(
    candidates: List[Dict[str, object]],
    count: int,
    min_gap: int,
) -> List[Dict[str, object]]:
    ordered = sorted(
        candidates,
        key=lambda row: (
            -candidate_priority(row)[0],
            -candidate_priority(row)[1],
            -candidate_priority(row)[2],
            str(row.get("audit_hit_file", "")),
            int(row.get("audit_hit_line_index", 0) or 0),
        ),
    )
    selected: List[Dict[str, object]] = []
    seen_by_file: Dict[str, List[int]] = defaultdict(list)
    for candidate in ordered:
        if len(selected) >= count:
            break
        file_name = str(candidate.get("audit_hit_file", ""))
        idx = int(candidate.get("audit_hit_line_index", 0) or 0)
        if all(abs(idx - prev) >= min_gap for prev in seen_by_file[file_name]):
            selected.append(candidate)
            seen_by_file[file_name].append(idx)
    if len(selected) >= count:
        return selected[:count]
    for candidate in ordered:
        if len(selected) >= count:
            break
        if candidate in selected:
            continue
        selected.append(candidate)
    return selected[:count]


def select_core_candidates(
    candidates: List[Dict[str, object]],
    min_gap: int = CORE_MIN_GAP,
) -> List[Dict[str, object]]:
    core_candidates = [row for row in candidates if str(row.get("audit_root_status", "")) == "core_eligible"]
    ordered = sorted(
        core_candidates,
        key=lambda row: (
            int(row.get("audit_cross_family_prior_count", 0) or 0),
            str(row.get("audit_hit_file", "")),
            int(row.get("audit_hit_line_index", 0) or 0),
        ),
    )
    selected: List[Dict[str, object]] = []
    seen_by_file: Dict[str, List[int]] = defaultdict(list)
    seen_per_pair: Counter = Counter()
    seen_per_pair_file: Dict[Tuple[str, str], Counter] = defaultdict(Counter)
    for candidate in ordered:
        file_name = str(candidate.get("audit_hit_file", ""))
        idx = int(candidate.get("audit_hit_line_index", 0) or 0)
        pair = (
            str(candidate.get("ground_truth_root_cause_template_graph", "") or ""),
            str(candidate.get("ground_truth_template_graph", "") or ""),
        )
        if seen_per_pair[pair] >= CORE_MAX_PER_PAIR_TOTAL:
            continue
        if seen_per_pair_file[pair][file_name] >= CORE_MAX_PER_PAIR_PER_FILE:
            continue
        if all(abs(idx - prev) >= min_gap for prev in seen_by_file[file_name]):
            selected.append(candidate)
            seen_by_file[file_name].append(idx)
            seen_per_pair[pair] += 1
            seen_per_pair_file[pair][file_name] += 1
    return selected


def copy_graph_files() -> Dict[str, Path]:
    ensure_dirs()
    for name, src in SOURCE_GRAPH_FILES.items():
        shutil.copy2(src, GRAPH_FILES[name])
    return GRAPH_FILES


def frozen_hdfs_rows() -> List[Dict[str, object]]:
    rows = load_json(AUDIT_BENCHMARK_EVAL_PATH)
    return [row for row in rows if str(row.get("dataset", "")) == "HDFS"]


def build_path_supplement_rows(
    graph_paths: Dict[str, Path],
    openstack_core_rows: List[Dict[str, object]],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for graph_name, path in graph_paths.items():
        kb = load_json(path)
        adjacency: Dict[str, set[str]] = defaultdict(set)
        for edge in kb:
            if str(edge.get("domain", "")).lower() != dataset_to_domain("OpenStack"):
                continue
            src = str(edge.get("source_template", "") or "")
            tgt = str(edge.get("target_template", "") or "")
            if not src or not tgt:
                continue
            adjacency[src].add(tgt)
            if str(edge.get("relation", "") or "") in UNDIRECTED_RELATIONS:
                adjacency[tgt].add(src)

        counter = Counter()
        for case in openstack_core_rows:
            root = str(case.get("ground_truth_root_cause_template_graph", "") or "")
            effect = str(case.get("ground_truth_template_graph", "") or "")
            depth = shortest_path_depth(adjacency, root, effect, max_hops=3)
            if depth == 1:
                counter["direct"] += 1
            elif depth == 2:
                counter["two_hop"] += 1
            elif depth == 3:
                counter["three_hop"] += 1
            else:
                counter["none"] += 1
        rows.append(
            {
                "graph": graph_name,
                "cases": len(openstack_core_rows),
                "direct": counter["direct"],
                "two_hop": counter["two_hop"],
                "three_hop": counter["three_hop"],
                "none": counter["none"],
            }
        )
    return rows


def shortest_path_depth(
    adjacency: Dict[str, set[str]],
    source: str,
    target: str,
    max_hops: int,
) -> int | None:
    if not source or not target:
        return None
    frontier = {source}
    seen = {source}
    for depth in range(1, max_hops + 1):
        nxt = set()
        for node in frontier:
            for neighbor in adjacency.get(node, set()):
                if neighbor == target:
                    return depth
                if neighbor not in seen:
                    seen.add(neighbor)
                    nxt.add(neighbor)
        if not nxt:
            break
        frontier = nxt
    return None
