from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_FINAL_ROOT = _SCRIPT_DIR.parent
_REBUILD_ROOT = _FINAL_ROOT.parents[0]
_PROJECT_ROOT = _FINAL_ROOT.parents[2]
_PENALIZED_ROOT = _REBUILD_ROOT / "rq2_mainline_penalized_20260318"

for path in [
    _PROJECT_ROOT,
    _PROJECT_ROOT / "experiments" / "rq123_e2e",
    _REBUILD_ROOT / "rq2_fullcase_audit_20260318" / "scripts",
    _REBUILD_ROOT / "rq2_hadoop_family_audit_20260318" / "scripts",
    _PENALIZED_ROOT / "scripts",
    _FINAL_ROOT / "scripts",
]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from rq2_fullcase_audit_common_20260318 import _safe_standardize, build_eval_cases as build_hdfs_eval_cases, calc_rank as calc_hdfs_rank  # type: ignore
from rq2_hadoop_family_audit_common_20260318 import calc_family_rank  # type: ignore
from rq2_mainline_completion_common_20260318 import (  # type: ignore
    DATASET_DOMAIN,
    FEATURE_SPACE_PATH as FROZEN_FEATURE_SPACE_PATH,
    GRAPH_PARAMS,
    OTHER_FAMILY,
    SOURCE_HDFS_BENCH_PATH,
    UNIFIED_BENCHMARK_MAINLINE_PATH,
    _clean_edges_for_domain,
    calc_edge_rank,
    calc_path2_rank,
    collect_mapped_symbolic_prior_edges,
    ensure_dirs as _ensure_dirs,
    exact_relaxed_match,
    family_of,
    graph_relation_stats,
    load_json,
    prepare_feature_space,
    write_json,
)

from vendor_dynotears_masked_causalnex_20260318 import dynotears_from_standardized_matrix_masked


RESULTS_DIR = _FINAL_ROOT / "results"
REPORTS_DIR = _FINAL_ROOT / "reports"
CACHE_DIR = RESULTS_DIR / "graph_cache_20260318"

HADAMARD_GRAPH_PATH = RESULTS_DIR / "gt_causal_knowledge_hadamard_mask_dynotears_20260318.json"
MASK_PROFILE_PATH = RESULTS_DIR / "rq2_hadamard_mask_profiles_20260318.json"
PAPER_SUMMARY_JSON = RESULTS_DIR / "rq2_hadamard_paper_metrics_summary_20260318.json"
PAPER_SUMMARY_MD = REPORTS_DIR / "rq2_hadamard_paper_metrics_summary_20260318.md"
ASSESSMENT_MD = REPORTS_DIR / "rq2_hadamard_candidate_assessment_20260318.md"

MASK_BACKGROUND = 2.0
MASK_SYMBOLIC = 0.10
MASK_CURATED = 0.02
MASK_REVERSE = 8.0
MASK_OTHER_SOURCE = 4.0
PILOT_MAX_ITER = 40
PILOT_SUPPORT_FLOOR = 0.05
PILOT_ACTIVE_PENALTY = 6.0
PILOT_OTHER_ACTIVE_PENALTY = 10.0
PRIOR_THRESHOLD_SCALE = 0.55
CURATED_THRESHOLD_SCALE = 0.25

FROZEN_COMPARE_GRAPH_FILES = {
    "modified": _PENALIZED_ROOT / "results" / "gt_causal_knowledge_modified_mainline_completion_20260318.json",
    "original_dynotears": _PENALIZED_ROOT / "results" / "gt_causal_knowledge_original_dynotears_mainline_completion_20260318.json",
    "pearson_hypothesis": _PENALIZED_ROOT / "results" / "gt_causal_knowledge_pearson_hypothesis_mainline_completion_20260318.json",
    "pc_cpdag_hypothesis": _PENALIZED_ROOT / "results" / "gt_causal_knowledge_pc_cpdag_hypothesis_mainline_completion_20260318.json",
    "hadamard_mask_dynotears": HADAMARD_GRAPH_PATH,
}


def ensure_dirs() -> None:
    _ensure_dirs()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def graph_cache_path(dataset: str) -> Path:
    return CACHE_DIR / f"{dataset.lower()}_hadamard_mask_dynotears_20260318.json"


def load_frozen_feature_columns() -> Dict[str, List[str]]:
    profiles = load_json(FROZEN_FEATURE_SPACE_PATH)
    return {str(row["dataset"]): list(row["column_ids"]) for row in profiles}


def prepare_frozen_feature_space(dataset: str) -> Tuple[object, Dict[str, str], Dict[str, object]]:
    frozen_cols = load_frozen_feature_columns()[dataset]
    return prepare_feature_space(dataset, explicit_columns=frozen_cols)


def _template_to_columns(cols: List[str], tpl_map: Dict[str, str]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = defaultdict(list)
    for col in cols:
        out[str(tpl_map.get(col, "") or "")].append(col)
    return out


def _collect_symbolic_edges_and_pairs(
    dataset: str,
    tpl_map: Dict[str, str],
) -> Tuple[List[Dict[str, object]], set[Tuple[str, str]], set[Tuple[str, str]]]:
    template_pool = list(dict.fromkeys(str(v or "") for v in tpl_map.values() if str(v or "").strip()))
    symbolic_edges = collect_mapped_symbolic_prior_edges(dataset, template_pool)
    prior_pairs = {
        (
            str(edge.get("source_template", "") or ""),
            str(edge.get("target_template", "") or ""),
        )
        for edge in symbolic_edges
    }
    curated_pairs = {
        (
            str(edge.get("source_template", "") or ""),
            str(edge.get("target_template", "") or ""),
        )
        for edge in symbolic_edges
        if bool(edge.get("curated_symbolic_prior"))
    }
    return symbolic_edges, prior_pairs, curated_pairs


def build_prior_masks(
    dataset: str,
    cols: List[str],
    tpl_map: Dict[str, str],
    symbolic_edges: List[Dict[str, object]],
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    d = len(cols)
    w_mask = np.full((d, d), MASK_BACKGROUND, dtype=float)
    a_mask = np.full((d, d), MASK_BACKGROUND, dtype=float)
    np.fill_diagonal(w_mask, 1.0)
    col_to_idx = {col: idx for idx, col in enumerate(cols)}

    tpl_to_cols = _template_to_columns(cols, tpl_map)

    for i, src_col in enumerate(cols):
        src_tpl = str(tpl_map.get(src_col, "") or "")
        if family_of(dataset, src_tpl) == OTHER_FAMILY[dataset]:
            w_mask[i, :] = np.maximum(w_mask[i, :], MASK_OTHER_SOURCE)
            a_mask[i, :] = np.maximum(a_mask[i, :], MASK_OTHER_SOURCE)
            w_mask[i, i] = 1.0

    forward_pairs = {
        (
            str(edge.get("source_template", "") or ""),
            str(edge.get("target_template", "") or ""),
        )
        for edge in symbolic_edges
    }
    applied = 0
    reverse_applied = 0
    curated_hits = 0

    for edge in symbolic_edges:
        src_tpl = str(edge.get("source_template", "") or "")
        tgt_tpl = str(edge.get("target_template", "") or "")
        if src_tpl not in tpl_to_cols or tgt_tpl not in tpl_to_cols:
            continue
        is_curated = bool(edge.get("curated_symbolic_prior"))
        soft_penalty = MASK_CURATED if is_curated else MASK_SYMBOLIC
        reverse_pair = (tgt_tpl, src_tpl)
        reverse_penalty = MASK_REVERSE if reverse_pair not in forward_pairs else MASK_BACKGROUND
        if is_curated:
            curated_hits += 1
        for src_col in tpl_to_cols[src_tpl]:
            i = col_to_idx[src_col]
            for tgt_col in tpl_to_cols[tgt_tpl]:
                j = col_to_idx[tgt_col]
                if i == j:
                    continue
                w_mask[i, j] = min(w_mask[i, j], soft_penalty)
                a_mask[i, j] = min(a_mask[i, j], soft_penalty)
                applied += 1
                if reverse_penalty > MASK_BACKGROUND:
                    w_mask[j, i] = max(w_mask[j, i], reverse_penalty)
                    a_mask[j, i] = max(a_mask[j, i], reverse_penalty)
                    reverse_applied += 1

    profile = {
        "dataset": dataset,
        "selected_columns": d,
        "symbolic_edges_mapped": len(symbolic_edges),
        "mask_entries_softened": int(applied),
        "mask_entries_reverse_penalized": int(reverse_applied),
        "curated_symbolic_edges": int(curated_hits),
        "w_mask_min": float(np.min(w_mask)),
        "w_mask_max": float(np.max(w_mask)),
        "a_mask_min": float(np.min(a_mask)),
        "a_mask_max": float(np.max(a_mask)),
    }
    return w_mask, a_mask, profile


def _build_stage2_masks(
    dataset: str,
    cols: List[str],
    tpl_map: Dict[str, str],
    base_w_mask: np.ndarray,
    base_a_mask: np.ndarray,
    W_pilot: np.ndarray,
    A_pilot: np.ndarray,
    prior_pairs: set[Tuple[str, str]],
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    w_mask = np.array(base_w_mask, copy=True)
    a_mask = np.array(base_a_mask, copy=True)
    d = len(cols)
    w_penalized = 0
    a_penalized = 0
    other_family = OTHER_FAMILY[dataset]

    for i in range(d):
        src_tpl = str(tpl_map.get(cols[i], "") or "")
        src_other = family_of(dataset, src_tpl) == other_family
        for j in range(d):
            if i == j:
                continue
            tgt_tpl = str(tpl_map.get(cols[j], "") or "")
            pair = (src_tpl, tgt_tpl)
            if pair in prior_pairs:
                continue
            if abs(float(W_pilot[i, j])) >= PILOT_SUPPORT_FLOOR:
                penalty = PILOT_OTHER_ACTIVE_PENALTY if src_other else PILOT_ACTIVE_PENALTY
                if penalty > w_mask[i, j]:
                    w_mask[i, j] = penalty
                    w_penalized += 1
            if abs(float(A_pilot[i, j])) >= PILOT_SUPPORT_FLOOR:
                penalty = PILOT_OTHER_ACTIVE_PENALTY if src_other else PILOT_ACTIVE_PENALTY
                if penalty > a_mask[i, j]:
                    a_mask[i, j] = penalty
                    a_penalized += 1
    np.fill_diagonal(w_mask, 1.0)
    return w_mask, a_mask, {
        "stage2_penalized_w_entries": int(w_penalized),
        "stage2_penalized_a_entries": int(a_penalized),
    }


def _edge_threshold(
    base_threshold: float,
    src_tpl: str,
    tgt_tpl: str,
    prior_pairs: set[Tuple[str, str]],
    curated_pairs: set[Tuple[str, str]],
) -> float:
    pair = (src_tpl, tgt_tpl)
    if pair in curated_pairs:
        return base_threshold * CURATED_THRESHOLD_SCALE
    if pair in prior_pairs:
        return base_threshold * PRIOR_THRESHOLD_SCALE
    return base_threshold


def build_hadamard_graph(dataset: str):
    df, tpl_map, feature_profile = prepare_frozen_feature_space(dataset)
    X, cols = _safe_standardize(df)
    if X.size == 0 or not cols:
        return [], {**feature_profile, "symbolic_edges_mapped": 0}
    symbolic_edges, prior_pairs, curated_pairs = _collect_symbolic_edges_and_pairs(dataset, tpl_map)
    w_mask, a_mask, mask_profile = build_prior_masks(dataset, cols, tpl_map, symbolic_edges)
    params = GRAPH_PARAMS[dataset]
    W_pilot, A_pilot = dynotears_from_standardized_matrix_masked(
        X,
        w_mask=w_mask,
        a_mask=a_mask,
        p=1,
        lambda_w=float(params["lambda_w"]),
        lambda_a=float(params["lambda_a"]),
        max_iter=PILOT_MAX_ITER,
        h_tol=1e-8,
        w_threshold=0.0,
    )
    stage2_w_mask, stage2_a_mask, stage2_profile = _build_stage2_masks(
        dataset,
        cols,
        tpl_map,
        w_mask,
        a_mask,
        W_pilot,
        A_pilot,
        prior_pairs,
    )
    W, A = dynotears_from_standardized_matrix_masked(
        X,
        w_mask=stage2_w_mask,
        a_mask=stage2_a_mask,
        p=1,
        lambda_w=float(params["lambda_w"]),
        lambda_a=float(params["lambda_a"]),
        max_iter=100,
        h_tol=1e-8,
        w_threshold=0.0,
    )
    edges: List[Dict[str, object]] = []
    threshold = float(params["threshold"])
    kept_prior_edges = 0
    kept_curated_edges = 0
    for mat, rel in ((A, "temporally_causes"), (W, "instantly_triggers")):
        mat = np.array(mat, copy=True)
        s_idx, t_idx = np.where(mat != 0)
        for s, t in zip(s_idx, t_idx):
            src_id = cols[s]
            tgt_id = cols[t]
            src_tpl = str(tpl_map.get(src_id, "Unknown") or "Unknown")
            tgt_tpl = str(tpl_map.get(tgt_id, "Unknown") or "Unknown")
            edge_threshold = _edge_threshold(threshold, src_tpl, tgt_tpl, prior_pairs, curated_pairs)
            if abs(float(mat[s, t])) < edge_threshold:
                continue
            pair = (src_tpl, tgt_tpl)
            if pair in prior_pairs:
                kept_prior_edges += 1
            if pair in curated_pairs:
                kept_curated_edges += 1
            edges.append(
                {
                    "domain": DATASET_DOMAIN[dataset],
                    "source_template": src_tpl,
                    "relation": rel,
                    "target_template": tgt_tpl,
                    "weight": float(round(float(mat[s, t]), 4)),
                }
            )
    cleaned = _clean_edges_for_domain(dataset, edges)
    return cleaned, {
        **feature_profile,
        **mask_profile,
        **stage2_profile,
        "pilot_support_floor": PILOT_SUPPORT_FLOOR,
        "pilot_max_iter": PILOT_MAX_ITER,
        "prior_threshold_scale": PRIOR_THRESHOLD_SCALE,
        "curated_threshold_scale": CURATED_THRESHOLD_SCALE,
        "kept_prior_edges": int(kept_prior_edges),
        "kept_curated_edges": int(kept_curated_edges),
        "edges": len(cleaned),
    }


def _penalized_case_rank(rank: int, sparsity: int) -> float:
    return float(rank if rank >= 0 else sparsity + 1)


def evaluate_paper_rows(graph_paths: Dict[str, Path]) -> List[Dict[str, object]]:
    kb_by_name = {name: load_json(path) for name, path in graph_paths.items()}
    hdfs_cases = [c for c in build_hdfs_eval_cases(SOURCE_HDFS_BENCH_PATH) if c["dataset"] == "HDFS"]
    mainline_rows = [row for row in load_json(UNIFIED_BENCHMARK_MAINLINE_PATH) if str(row.get("benchmark_tier", "") or "") == "mainline"]
    openstack_rows = [row for row in mainline_rows if str(row.get("dataset", "") or "") == "OpenStack"]
    hadoop_cases = [
        {
            "dataset": "Hadoop",
            "case_id": str(row["case_id"]),
            "gt_effect": str(row["effect_target_value"]),
            "gt_root_family": str(row["root_target_value"]),
        }
        for row in mainline_rows
        if str(row.get("dataset", "") or "") == "Hadoop"
    ]

    rows: List[Dict[str, object]] = []
    for method, kb in kb_by_name.items():
        hdfs_sparsity_sum = 0.0
        hdfs_rank_sum = 0.0
        for case in hdfs_cases:
            sparsity, rank = calc_hdfs_rank(kb, "HDFS", case["gt_root"], case["gt_effect"], match_mode="task_aligned")
            hdfs_sparsity_sum += float(sparsity)
            hdfs_rank_sum += _penalized_case_rank(rank, sparsity)
        rows.append(
            {
                "dataset": "HDFS",
                "method": method,
                "evaluator": "audit_task_aligned_penalized",
                "sparsity_mean": round(hdfs_sparsity_sum / len(hdfs_cases), 4),
                "avg_rank": round(hdfs_rank_sum / len(hdfs_cases), 4),
            }
        )

        os_edge_sparsity_sum = 0.0
        os_edge_rank_sum = 0.0
        for row in openstack_rows:
            sparsity, rank = calc_edge_rank(kb, row, "task_aligned_edge")
            os_edge_sparsity_sum += float(sparsity)
            os_edge_rank_sum += _penalized_case_rank(rank, sparsity)
        rows.append(
            {
                "dataset": "OpenStack",
                "method": method,
                "evaluator": "redesign_task_aligned_edge_penalized",
                "sparsity_mean": round(os_edge_sparsity_sum / len(openstack_rows), 4),
                "avg_rank": round(os_edge_rank_sum / len(openstack_rows), 4),
            }
        )

        os_path_sparsity_sum = 0.0
        os_path_rank_sum = 0.0
        for row in openstack_rows:
            sparsity, rank = calc_path2_rank(kb, row)
            os_path_sparsity_sum += float(sparsity)
            os_path_rank_sum += _penalized_case_rank(rank, sparsity)
        rows.append(
            {
                "dataset": "OpenStack",
                "method": method,
                "evaluator": "redesign_task_aligned_path2_penalized",
                "sparsity_mean": round(os_path_sparsity_sum / len(openstack_rows), 4),
                "avg_rank": round(os_path_rank_sum / len(openstack_rows), 4),
            }
        )

        hd_sparsity_sum = 0.0
        hd_rank_sum = 0.0
        for case in hadoop_cases:
            sparsity, rank = calc_family_rank(kb, case["gt_root_family"], case["gt_effect"], match_mode="task_aligned")
            hd_sparsity_sum += float(sparsity)
            hd_rank_sum += _penalized_case_rank(rank, sparsity)
        rows.append(
            {
                "dataset": "Hadoop",
                "method": method,
                "evaluator": "family_audit_core80_task_aligned_penalized",
                "sparsity_mean": round(hd_sparsity_sum / len(hadoop_cases), 4),
                "avg_rank": round(hd_rank_sum / len(hadoop_cases), 4),
            }
        )
    return rows


def summarize_rows(rows: List[Dict[str, object]]) -> str:
    lines = [
        "> `Avg_Rank` is the penalized mean over all cases; a miss is assigned `E + 1`, where `E` is the graph edge count for that dataset/method.",
        "",
        "| Dataset | Method | Evaluator | Sparsity_mean | Avg_Rank |",
        "|---|---|---|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['method']} | {row['evaluator']} | {row['sparsity_mean']} | {row['avg_rank']} |"
        )
    return "\n".join(lines) + "\n"


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
