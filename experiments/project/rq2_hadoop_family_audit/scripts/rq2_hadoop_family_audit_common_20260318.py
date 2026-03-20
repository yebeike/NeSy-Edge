from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

_SCRIPT_DIR = Path(__file__).resolve().parent
_AUDIT_ROOT = _SCRIPT_DIR.parent
_REBUILD_ROOT = _AUDIT_ROOT.parents[0]
_PROJECT_ROOT = _AUDIT_ROOT.parents[2]

_FULLCASE_SCRIPTS = _REBUILD_ROOT / "rq2_fullcase" / "scripts"
_FULLCASE_RESULTS = _REBUILD_ROOT / "rq2_fullcase" / "results"
_PHASE1_AUDIT_SCRIPTS = _REBUILD_ROOT / "rq2_fullcase_audit_20260318" / "scripts"

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_PROJECT_ROOT / "experiments" / "rq123_e2e") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "experiments" / "rq123_e2e"))
if str(_FULLCASE_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_FULLCASE_SCRIPTS))
if str(_PHASE1_AUDIT_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_PHASE1_AUDIT_SCRIPTS))

from build_e2e_scaled_benchmark import _extract_target_line_and_template  # type: ignore
from rq2_fullcase_common_20260316 import (  # type: ignore
    BENCH_V2_PATH,
    _trim_top_variance_columns,
    canonical_tokens,
    exact_relaxed_match,
    family_of,
    fuzzy_match,
    load_hadoop_timeseries,
)
from rq2_fullcase_audit_common_20260318 import (  # type: ignore
    UNDIRECTED_RELATIONS,
    build_original_dynotears_edges as _build_original_dynotears_edges,
    build_pc_cpdag_hypothesis_edges as _build_pc_cpdag_hypothesis_edges,
    build_pearson_hypothesis_edges as _build_pearson_hypothesis_edges,
    iter_with_progress,
    load_json,
    merge_edges_prefer_stronger,
    write_json,
)


RESULTS_DIR = _AUDIT_ROOT / "results"
REPORTS_DIR = _AUDIT_ROOT / "reports"

SOURCE_BENCH_PATH = BENCH_V2_PATH
OLD_MODIFIED_GRAPH_PATH = _FULLCASE_RESULTS / "gt_causal_knowledge_nesydy_fullcase_20260316.json"

HADOOP_BENCHMARK_FULL_PATH = RESULTS_DIR / "rq2_hadoop_family_audit_benchmark_full_20260318.json"
HADOOP_BENCHMARK_CORE_PATH = RESULTS_DIR / "rq2_hadoop_family_audit_benchmark_local_core_20260318.json"

GRAPH_FILES = {
    "modified": RESULTS_DIR / "gt_causal_knowledge_modified_hadoop_family_audit_20260318.json",
    "original_dynotears": RESULTS_DIR / "gt_causal_knowledge_original_dynotears_hadoop_family_audit_20260318.json",
    "pearson_hypothesis": RESULTS_DIR / "gt_causal_knowledge_pearson_hadoop_family_audit_20260318.json",
    "pc_cpdag_hypothesis": RESULTS_DIR / "gt_causal_knowledge_pc_hadoop_family_audit_20260318.json",
}

CORE_LOCAL_RADIUS = 40
DIAGNOSTIC_RADIUS = 80


def ensure_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def _best_graph_template_hadoop(
    query_texts: List[str],
    pool: List[str],
    desired_family: str = "",
) -> str:
    queries = [str(x or "").strip() for x in query_texts if str(x or "").strip()]
    if not queries:
        return ""
    query_token_sets = [set(canonical_tokens(q)) for q in queries]

    best_tpl = ""
    best_score = -1.0
    for tpl in pool:
        fam = family_of("Hadoop", tpl)
        score = 0.0
        for q, toks in zip(queries, query_token_sets):
            if exact_relaxed_match(tpl, q):
                score = max(score, 1000.0)
            if desired_family and fam == desired_family and fam != "HADOOP_UNKNOWN":
                score += 120.0
            if family_of("Hadoop", q) == fam and fam != "HADOOP_UNKNOWN":
                score += 80.0
            score += len(set(canonical_tokens(tpl)) & toks) * 4.0
            if q.lower() in tpl.lower() or tpl.lower() in q.lower():
                score += 20.0
        if score > best_score:
            best_score = score
            best_tpl = tpl
    return best_tpl


def _unique_root_graph_candidates(
    prior_lines: List[str],
    template_pool: List[str],
    desired_root_family: str,
    effect_graph: str,
) -> Tuple[List[str], List[str]]:
    root_graphs: List[str] = []
    root_lines: List[str] = []
    seen_graphs = set()
    for line in prior_lines:
        if family_of("Hadoop", line) != desired_root_family:
            continue
        _, tpl = _extract_target_line_and_template([line], "Hadoop")
        graph = _best_graph_template_hadoop(
            [str(tpl or ""), line],
            template_pool,
            desired_family=desired_root_family,
        )
        if not graph:
            continue
        if exact_relaxed_match(graph, effect_graph):
            continue
        if graph in seen_graphs:
            continue
        seen_graphs.add(graph)
        root_graphs.append(graph)
        root_lines.append(line)
    return root_graphs, root_lines


def _window_prior_lines(lines: List[str], target_idx: int, radius: int) -> List[str]:
    start = max(0, target_idx - radius)
    return [line for line in lines[start:target_idx] if line.strip()]


def build_hadoop_family_case(
    row: Dict[str, object],
    template_pool: List[str],
) -> Dict[str, object]:
    lines = [x for x in str(row.get("raw_log", "") or "").splitlines() if x.strip()]
    target_line, target_tpl = _extract_target_line_and_template(lines, "Hadoop")
    target_idx = len(lines) - 1
    if target_line:
        try:
            target_idx = lines.index(target_line)
        except ValueError:
            target_idx = len(lines) - 1

    effect_graph = _best_graph_template_hadoop(
        [
            str(row.get("ground_truth_template", "") or ""),
            str(target_tpl or ""),
            str(target_line or ""),
        ],
        template_pool,
    )
    root_label = str(row.get("ground_truth_root_cause_template", "") or "")
    root_family = family_of("Hadoop", root_label)

    prior40 = _window_prior_lines(lines, target_idx, CORE_LOCAL_RADIUS)
    prior80 = _window_prior_lines(lines, target_idx, DIAGNOSTIC_RADIUS)
    root_graphs40, root_lines40 = _unique_root_graph_candidates(prior40, template_pool, root_family, effect_graph)
    root_graphs80, root_lines80 = _unique_root_graph_candidates(prior80, template_pool, root_family, effect_graph)

    status40 = "missing"
    if len(root_graphs40) == 1:
        status40 = "unique_graph"
    elif len(root_graphs40) > 1:
        status40 = "ambiguous_graph"

    status80 = "missing"
    if len(root_graphs80) == 1:
        status80 = "unique_graph"
    elif len(root_graphs80) > 1:
        status80 = "ambiguous_graph"

    row2 = dict(row)
    row2["ground_truth_template_label"] = str(row.get("ground_truth_template", "") or "")
    row2["ground_truth_root_cause_label"] = root_label
    row2["ground_truth_template_graph"] = effect_graph
    row2["ground_truth_root_family"] = root_family
    row2["audit_target_line"] = str(target_line or "")
    row2["audit_target_template"] = str(target_tpl or "")
    row2["audit_target_line_index"] = target_idx
    row2["audit_local_root_graphs_radius40"] = root_graphs40
    row2["audit_local_root_lines_radius40"] = root_lines40
    row2["audit_local_root_status_radius40"] = status40
    row2["audit_local_root_graphs_radius80"] = root_graphs80
    row2["audit_local_root_lines_radius80"] = root_lines80
    row2["audit_local_root_status_radius80"] = status80
    row2["audit_local_core_eligible"] = status40 == "unique_graph"
    row2["audit_derivation_complete"] = bool(effect_graph and target_line and target_tpl and root_family != "HADOOP_UNKNOWN")
    return row2


def build_eval_cases(bench_path: Path) -> List[Dict[str, str]]:
    rows = load_json(bench_path)
    cases: List[Dict[str, str]] = []
    for row in rows:
        if str(row.get("dataset", "") or "") != "Hadoop":
            continue
        gt_effect = str(row.get("ground_truth_template_graph", "") or "").strip()
        gt_root_family = str(row.get("ground_truth_root_family", "") or "").strip()
        if not gt_effect or not gt_root_family:
            continue
        cases.append(
            {
                "dataset": "Hadoop",
                "case_id": str(row.get("case_id", "") or ""),
                "gt_effect": gt_effect,
                "gt_root_family": gt_root_family,
            }
        )
    return cases


def clean_hadoop_edges(edges: List[Dict[str, object]]) -> List[Dict[str, object]]:
    filtered = [
        edge
        for edge in edges
        if str(edge.get("domain", "")).lower() == "hadoop"
        and not exact_relaxed_match(
            str(edge.get("source_template", "") or ""),
            str(edge.get("target_template", "") or ""),
        )
    ]
    return merge_edges_prefer_stronger(filtered)


def build_original_dynotears_edges(
    df,
    tpl_map: Dict[str, str],
    lambda_w: float,
    lambda_a: float,
    threshold: float,
    p: int = 1,
    max_iter: int = 100,
) -> List[Dict[str, object]]:
    edges = _build_original_dynotears_edges(
        df,
        tpl_map,
        "hadoop",
        lambda_w=lambda_w,
        lambda_a=lambda_a,
        threshold=threshold,
        p=p,
        max_iter=max_iter,
    )
    return clean_hadoop_edges(edges)


def build_pearson_hypothesis_edges(
    df,
    tpl_map: Dict[str, str],
    threshold: float,
) -> List[Dict[str, object]]:
    edges = _build_pearson_hypothesis_edges(df, tpl_map, "hadoop", threshold=threshold)
    return clean_hadoop_edges(edges)


def build_pc_cpdag_hypothesis_edges(
    df,
    tpl_map: Dict[str, str],
    alpha: float,
    max_vars: int = 0,
) -> List[Dict[str, object]]:
    work_df = df
    if max_vars > 0 and getattr(df, "shape", (0, 0))[1] > max_vars:
        work_df = _trim_top_variance_columns(df, max_vars)
    edges = _build_pc_cpdag_hypothesis_edges(work_df, tpl_map, "hadoop", alpha=alpha)
    return clean_hadoop_edges(edges)


def _relation_penalty(edge: Dict[str, object]) -> int:
    rel = str(edge.get("relation", "") or "")
    if rel == "pearson_undirected":
        return 1
    if rel == "pc_undirected":
        return 2
    if rel in {"pc_partially_oriented", "pc_bidirected", "pc_ambiguous"}:
        return 1
    return 0


def _effect_match_kind(pred_effect: str, gt_effect: str, match_mode: str) -> str:
    if exact_relaxed_match(pred_effect, gt_effect):
        return "exact"
    if match_mode != "task_aligned":
        return "none"
    pred_family = family_of("Hadoop", pred_effect)
    gt_family = family_of("Hadoop", gt_effect)
    if pred_family == gt_family and pred_family != "HADOOP_UNKNOWN":
        return "family"
    if fuzzy_match(pred_effect, gt_effect):
        return "fuzzy"
    return "none"


def _candidate_buckets(
    edges_domain: List[Dict[str, object]],
    gt_effect: str,
    match_mode: str,
) -> Dict[str, List[Dict[str, object]]]:
    buckets: Dict[str, List[Dict[str, object]]] = {"exact": [], "family": [], "fuzzy": []}

    def _maybe_add(candidate_root: str, effect_side: str, edge: Dict[str, object]) -> None:
        kind = _effect_match_kind(effect_side, gt_effect, match_mode)
        if kind in buckets:
            buckets[kind].append({"candidate_root": candidate_root, "edge": edge})

    for edge in edges_domain:
        src = str(edge.get("source_template", "") or "")
        tgt = str(edge.get("target_template", "") or "")
        rel = str(edge.get("relation", "") or "")
        if rel in UNDIRECTED_RELATIONS:
            _maybe_add(tgt, src, edge)
            _maybe_add(src, tgt, edge)
        else:
            _maybe_add(src, tgt, edge)

    return buckets


def calc_family_rank(
    kb: List[Dict[str, object]],
    gt_root_family: str,
    gt_effect: str,
    match_mode: str,
) -> Tuple[int, int]:
    edges_domain = [e for e in kb if str(e.get("domain", "")).lower() == "hadoop"]
    sparsity = len(edges_domain)
    buckets = _candidate_buckets(edges_domain, gt_effect, match_mode)
    pool_order: List[Tuple[str, int]] = [("exact", 0)]
    if match_mode == "task_aligned":
        pool_order.extend([("family", 0), ("fuzzy", 2)])

    for kind, target_penalty in pool_order:
        candidates = buckets[kind]
        if not candidates:
            continue
        scored = sorted(
            ((abs(float(c["edge"].get("weight", 0.0) or 0.0)), c) for c in candidates),
            key=lambda x: x[0],
            reverse=True,
        )
        best_rank = -1
        for idx, (_, candidate) in enumerate(scored, start=1):
            cand_root = str(candidate["candidate_root"] or "")
            if exact_relaxed_match(cand_root, gt_effect):
                continue
            if family_of("Hadoop", cand_root) != gt_root_family or gt_root_family == "HADOOP_UNKNOWN":
                continue
            cand_rank = idx + _relation_penalty(candidate["edge"]) + target_penalty
            if best_rank < 0 or cand_rank < best_rank:
                best_rank = cand_rank
        if best_rank >= 0:
            return sparsity, best_rank
    return sparsity, -1


def evaluate_family_graph_rows(
    graph_paths: Dict[str, Path],
    bench_path: Path,
    match_mode: str,
) -> List[Dict[str, object]]:
    eval_cases = build_eval_cases(bench_path)
    kb_by_name = {name: load_json(path) for name, path in graph_paths.items()}
    rows: List[Dict[str, object]] = []
    jobs = [(name, kb) for name, kb in kb_by_name.items()]
    for name, kb in iter_with_progress(jobs, f"Evaluating {match_mode}"):
        sparsity_sum = 0.0
        rank_sum = 0.0
        rankable = 0
        for case in eval_cases:
            sparsity, rank = calc_family_rank(
                kb,
                gt_root_family=case["gt_root_family"],
                gt_effect=case["gt_effect"],
                match_mode=match_mode,
            )
            sparsity_sum += float(sparsity)
            if rank >= 0:
                rankable += 1
                rank_sum += float(rank)
        n = len(eval_cases) or 1
        rows.append(
            {
                "dataset": "Hadoop",
                "graph": name,
                "cases": len(eval_cases),
                "rankable": rankable,
                "sparsity_mean": round(sparsity_sum / n, 4),
                "avg_rank": None if rankable == 0 else round(rank_sum / rankable, 4),
            }
        )
    return rows


def summarize_rows(rows: List[Dict[str, object]]) -> str:
    lines = [
        "| Dataset | Graph | Cases | Rankable | Sparsity_mean | Avg_Rank |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        avg_rank = "nan" if row["avg_rank"] is None else f"{row['avg_rank']}"
        lines.append(
            f"| {row['dataset']} | {row['graph']} | {row['cases']} | {row['rankable']} | {row['sparsity_mean']} | {avg_rank} |"
        )
    return "\n".join(lines) + "\n"


def relation_stats(edges: List[Dict[str, object]]) -> Dict[str, object]:
    domain_edges = [e for e in edges if str(e.get("domain", "")).lower() == "hadoop"]
    self_edges = [
        e
        for e in domain_edges
        if exact_relaxed_match(
            str(e.get("source_template", "") or ""),
            str(e.get("target_template", "") or ""),
        )
    ]
    return {
        "edges": len(domain_edges),
        "relations": dict(Counter(str(e.get("relation", "")) for e in domain_edges)),
        "self_edges_detected": len(self_edges),
        "edges_after_clean": len(clean_hadoop_edges(edges)),
    }
