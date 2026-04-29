"""Public reasoning evaluation helpers with paper-aligned rank semantics."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

from experiments.project.reasoning.core.matching import (
    HDFS_EFFECT_PROXY_RULES,
    UNDIRECTED_RELATIONS,
    effect_match_kind,
    exact_relaxed_match,
    family_of,
)


_CANONICAL_METHOD_ORDER = ("nesy_edge_reasoning", "dynotears", "pc", "pearson")
_ROOT_OTHER = {"HDFS": "HDFS_OTHER", "OpenStack": "OS_OTHER", "Hadoop": "HADOOP_UNKNOWN"}


def canonical_evaluator_names(hadoop_prefix: str = "family_audit_full44") -> dict[str, dict[str, str]]:
    """Return canonical evaluator names used by the public reasoning package."""
    return {
        "dataset_aligned": {
            "HDFS": "audit_task_aligned",
            "OpenStack": "redesign_task_aligned_path2",
            "Hadoop": f"{hadoop_prefix}_task_aligned",
        },
        "penalized": {
            "HDFS": "audit_task_aligned_penalized",
            "OpenStack": "redesign_task_aligned_path2_penalized",
            "Hadoop": f"{hadoop_prefix}_task_aligned_penalized",
        },
    }


def _target_penalty(kind: str) -> int:
    return 2 if kind == "fuzzy" else 0


def _relation_penalty(edge: Mapping[str, Any]) -> int:
    relation = str(edge.get("relation", "") or "")
    if relation == "pearson_undirected":
        return 1
    if relation == "pc_undirected":
        return 2
    if relation in {"pc_partially_oriented", "pc_bidirected", "pc_ambiguous"}:
        return 1
    return 0


def _root_matches(row: Mapping[str, Any], candidate_root: str) -> bool:
    dataset = str(row["dataset"])
    root_type = str(row["root_target_type"])
    root_value = str(row["root_target_value"])
    if root_type == "family":
        return family_of(dataset, candidate_root) == root_value and root_value != _ROOT_OTHER[dataset]
    return exact_relaxed_match(candidate_root, root_value)


def _edge_candidates_for_case(
    kb: list[dict[str, Any]],
    row: Mapping[str, Any],
    mode: str,
) -> list[dict[str, Any]]:
    dataset = str(row["dataset"])
    domain = dataset.lower()
    gt_effect = str(row["effect_target_value"])
    candidates: list[dict[str, Any]] = []
    for edge in kb:
        if str(edge.get("domain", "")).lower() != domain:
            continue
        src = str(edge.get("source_template", "") or "")
        tgt = str(edge.get("target_template", "") or "")
        relation = str(edge.get("relation", "") or "")
        pairs = [(src, tgt)]
        if relation in UNDIRECTED_RELATIONS:
            pairs = [(src, tgt), (tgt, src)]
        for candidate_root, effect_side in pairs:
            kind = "exact" if mode == "exact_only_edge" and exact_relaxed_match(effect_side, gt_effect) else effect_match_kind(dataset, effect_side, gt_effect)
            if mode == "exact_only_edge" and kind != "exact":
                continue
            if kind == "none":
                continue
            candidates.append({"candidate_root": candidate_root, "kind": kind, "edge": edge})
    return candidates


def calc_hdfs_rank(kb: list[dict[str, Any]], row: Mapping[str, Any], *, match_mode: str = "task_aligned") -> tuple[int, int]:
    dataset = "HDFS"
    edges_domain = [edge for edge in kb if str(edge.get("domain", "")).lower() == "hdfs"]
    sparsity = len(edges_domain)
    gt_root = str(row["root_target_value"])
    gt_effect = str(row["effect_target_value"])

    def candidate_buckets() -> dict[str, list[dict[str, Any]]]:
        buckets = {"exact": [], "family": [], "fuzzy": []}

        def maybe_add(candidate_root: str, effect_side: str, edge: Mapping[str, Any]) -> None:
            if match_mode == "exact_only":
                if exact_relaxed_match(effect_side, gt_effect):
                    buckets["exact"].append({"candidate_root": candidate_root, "edge": edge})
                return
            kind = effect_match_kind(dataset, effect_side, gt_effect)
            if kind in buckets:
                buckets[kind].append({"candidate_root": candidate_root, "edge": edge})

        for edge in edges_domain:
            src = str(edge.get("source_template", "") or "")
            tgt = str(edge.get("target_template", "") or "")
            relation = str(edge.get("relation", "") or "")
            if relation in UNDIRECTED_RELATIONS:
                maybe_add(tgt, src, edge)
                maybe_add(src, tgt, edge)
            else:
                maybe_add(src, tgt, edge)
        return buckets

    def rank_from_candidates(candidates: list[dict[str, Any]], *, root_options: list[tuple[str, int]]) -> int:
        scored = sorted(
            ((abs(float(candidate["edge"].get("weight", 0.0) or 0.0)), candidate) for candidate in candidates),
            key=lambda item: item[0],
            reverse=True,
        )
        best_rank = -1
        for index, (_, candidate) in enumerate(scored, start=1):
            cand_root = str(candidate["candidate_root"] or "")
            edge = candidate["edge"]
            for root_option, root_penalty in root_options:
                if match_mode == "exact_only":
                    matched = exact_relaxed_match(cand_root, root_option)
                else:
                    matched = exact_relaxed_match(cand_root, root_option) or (
                        family_of(dataset, cand_root) == family_of(dataset, root_option) != _ROOT_OTHER[dataset]
                    )
                if not matched:
                    continue
                cand_rank = index + root_penalty + _relation_penalty(edge)
                if best_rank < 0 or cand_rank < best_rank:
                    best_rank = cand_rank
        return best_rank

    buckets = candidate_buckets()
    pool_order: list[tuple[str, int]] = [("exact", 0)]
    if match_mode == "task_aligned":
        pool_order.extend([("family", 0), ("fuzzy", 2)])
    for kind, target_penalty in pool_order:
        candidates = buckets[kind]
        if not candidates:
            continue
        direct_rank = rank_from_candidates(candidates, root_options=[(gt_root, 0)])
        if direct_rank >= 0:
            return sparsity, direct_rank + target_penalty

    if match_mode == "task_aligned":
        rule = HDFS_EFFECT_PROXY_RULES.get(gt_effect)
        if rule:
            for proxy_target, proxy_penalty in rule["target_proxies"]:
                proxy_candidates = []
                for edge in edges_domain:
                    src = str(edge.get("source_template", "") or "")
                    tgt = str(edge.get("target_template", "") or "")
                    relation = str(edge.get("relation", "") or "")
                    pairs = [(src, tgt), (tgt, src)] if relation in UNDIRECTED_RELATIONS else [(src, tgt)]
                    for candidate_root, effect_side in pairs:
                        if exact_relaxed_match(effect_side, proxy_target):
                            proxy_candidates.append({"candidate_root": candidate_root, "edge": edge})
                if not proxy_candidates:
                    continue
                proxy_rank = rank_from_candidates(proxy_candidates, root_options=list(rule["root_proxies"]))
                if proxy_rank >= 0:
                    return sparsity, proxy_rank + proxy_penalty
    return sparsity, -1


def calc_openstack_edge_rank(kb: list[dict[str, Any]], row: Mapping[str, Any], *, mode: str = "task_aligned_edge") -> tuple[int, int]:
    edges_domain = [edge for edge in kb if str(edge.get("domain", "")).lower() == "openstack"]
    sparsity = len(edges_domain)
    candidates = _edge_candidates_for_case(kb, row, mode)
    if not candidates:
        return sparsity, -1
    scored = sorted(
        ((abs(float(candidate["edge"].get("weight", 0.0) or 0.0)), candidate) for candidate in candidates),
        key=lambda item: item[0],
        reverse=True,
    )
    best_rank = -1
    for index, (_, candidate) in enumerate(scored, start=1):
        if not _root_matches(row, str(candidate["candidate_root"] or "")):
            continue
        cand_rank = index + _relation_penalty(candidate["edge"]) + _target_penalty(str(candidate["kind"]))
        if best_rank < 0 or cand_rank < best_rank:
            best_rank = cand_rank
    return sparsity, best_rank


def calc_openstack_path2_rank(kb: list[dict[str, Any]], row: Mapping[str, Any]) -> tuple[int, int]:
    sparsity, direct_rank = calc_openstack_edge_rank(kb, row, mode="task_aligned_edge")
    if direct_rank >= 0:
        return sparsity, direct_rank
    transitions: list[dict[str, Any]] = []
    for edge in kb:
        if str(edge.get("domain", "")).lower() != "openstack":
            continue
        src = str(edge.get("source_template", "") or "")
        tgt = str(edge.get("target_template", "") or "")
        relation = str(edge.get("relation", "") or "")
        transitions.append({"src": src, "tgt": tgt, "edge": edge})
        if relation in UNDIRECTED_RELATIONS:
            transitions.append({"src": tgt, "tgt": src, "edge": edge})

    by_src: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for transition in transitions:
        by_src[str(transition["src"])].append(transition)

    path_candidates: list[tuple[tuple[float, float], dict[str, Any]]] = []
    gt_effect = str(row["effect_target_value"])
    for transition1 in transitions:
        for transition2 in by_src.get(str(transition1["tgt"]), []):
            kind = effect_match_kind("OpenStack", str(transition2["tgt"]), gt_effect)
            if kind == "none":
                continue
            root_node = str(transition1["src"])
            if not _root_matches(row, root_node):
                continue
            score = (
                min(
                    abs(float(transition1["edge"].get("weight", 0.0) or 0.0)),
                    abs(float(transition2["edge"].get("weight", 0.0) or 0.0)),
                ),
                abs(float(transition1["edge"].get("weight", 0.0) or 0.0))
                + abs(float(transition2["edge"].get("weight", 0.0) or 0.0)),
            )
            path_candidates.append((score, {"kind": kind, "edge1": transition1["edge"], "edge2": transition2["edge"]}))
    if not path_candidates:
        return sparsity, -1
    path_candidates.sort(key=lambda item: item[0], reverse=True)
    best_rank = -1
    for index, (_, candidate) in enumerate(path_candidates, start=1):
        cand_rank = (
            index
            + _relation_penalty(candidate["edge1"])
            + _relation_penalty(candidate["edge2"])
            + _target_penalty(str(candidate["kind"]))
            + 1
        )
        if best_rank < 0 or cand_rank < best_rank:
            best_rank = cand_rank
    return sparsity, best_rank


def calc_hadoop_family_rank(kb: list[dict[str, Any]], row: Mapping[str, Any], *, match_mode: str = "task_aligned") -> tuple[int, int]:
    edges_domain = [edge for edge in kb if str(edge.get("domain", "")).lower() == "hadoop"]
    sparsity = len(edges_domain)
    gt_root_family = str(row["root_target_value"])
    gt_effect = str(row["effect_target_value"])
    buckets: dict[str, list[dict[str, Any]]] = {"exact": [], "family": [], "fuzzy": []}
    for edge in edges_domain:
        src = str(edge.get("source_template", "") or "")
        tgt = str(edge.get("target_template", "") or "")
        relation = str(edge.get("relation", "") or "")
        pairs = [(src, tgt)]
        if relation in UNDIRECTED_RELATIONS:
            pairs = [(src, tgt), (tgt, src)]
        for candidate_root, effect_side in pairs:
            if match_mode == "exact_only":
                kind = "exact" if exact_relaxed_match(effect_side, gt_effect) else "none"
            else:
                kind = effect_match_kind("Hadoop", effect_side, gt_effect)
            if kind == "none":
                continue
            buckets[kind].append({"candidate_root": candidate_root, "edge": edge})
    pool_order: list[tuple[str, int]] = [("exact", 0)]
    if match_mode == "task_aligned":
        pool_order.extend([("family", 0), ("fuzzy", 2)])
    for kind, target_penalty in pool_order:
        candidates = buckets[kind]
        if not candidates:
            continue
        scored = sorted(
            ((abs(float(candidate["edge"].get("weight", 0.0) or 0.0)), candidate) for candidate in candidates),
            key=lambda item: item[0],
            reverse=True,
        )
        best_rank = -1
        for index, (_, candidate) in enumerate(scored, start=1):
            cand_root = str(candidate["candidate_root"] or "")
            if exact_relaxed_match(cand_root, gt_effect):
                continue
            if family_of("Hadoop", cand_root) != gt_root_family or gt_root_family == "HADOOP_UNKNOWN":
                continue
            cand_rank = index + _relation_penalty(candidate["edge"]) + target_penalty
            if best_rank < 0 or cand_rank < best_rank:
                best_rank = cand_rank
        if best_rank >= 0:
            return sparsity, best_rank
    return sparsity, -1


def _evaluate_dataset(
    *,
    dataset: str,
    evaluator: str,
    cases: list[dict[str, Any]],
    kb: list[dict[str, Any]],
    rank_fn,
    penalized: bool,
    method: str,
) -> dict[str, Any]:
    rankable = 0
    sparsity_sum = 0.0
    rank_sum = 0.0
    penalized_sum = 0.0
    for row in cases:
        sparsity, rank = rank_fn(kb, row)
        sparsity_sum += float(sparsity)
        penalized_sum += float(rank if rank >= 0 else sparsity + 1)
        if rank >= 0:
            rankable += 1
            rank_sum += float(rank)
    n = len(cases) or 1
    payload: dict[str, Any] = {
        "dataset": dataset,
        "method": method,
        "evaluator": evaluator,
        "cases": len(cases),
        "sparsity_mean": round(sparsity_sum / n, 4),
    }
    if penalized:
        payload["avg_rank"] = round(penalized_sum / n, 4)
    else:
        payload["rankable"] = rankable
        payload["avg_rank"] = None if not rankable else round(rank_sum / rankable, 4)
    return payload


def evaluate_reasoning_graphs(
    *,
    case_rows: Iterable[Mapping[str, Any]],
    graph_rows_by_method: Mapping[str, Mapping[str, Iterable[Mapping[str, Any]]]],
    hadoop_prefix: str = "family_audit_full44",
) -> dict[str, list[dict[str, Any]]]:
    """Evaluate multiple graph methods over the provided reasoning benchmark rows."""
    rows = [dict(row) for row in case_rows]
    hdfs_rows = [row for row in rows if str(row.get("dataset", "")) == "HDFS"]
    openstack_rows = [row for row in rows if str(row.get("dataset", "")) == "OpenStack"]
    hadoop_rows = [row for row in rows if str(row.get("dataset", "")) == "Hadoop"]
    evaluator_names = canonical_evaluator_names(hadoop_prefix)

    dataset_aligned: list[dict[str, Any]] = []
    penalized_rows: list[dict[str, Any]] = []
    ordered_methods = [m for m in _CANONICAL_METHOD_ORDER if m in graph_rows_by_method]
    ordered_methods += sorted(set(graph_rows_by_method) - set(ordered_methods))

    for method in ordered_methods:
        dataset_graphs = {dataset: [dict(edge) for edge in edges] for dataset, edges in graph_rows_by_method[method].items()}
        dataset_aligned.append(
            _evaluate_dataset(
                dataset="HDFS",
                evaluator=evaluator_names["dataset_aligned"]["HDFS"],
                cases=hdfs_rows,
                kb=dataset_graphs.get("HDFS", []),
                rank_fn=calc_hdfs_rank,
                penalized=False,
                method=method,
            )
        )
        penalized_rows.append(
            _evaluate_dataset(
                dataset="HDFS",
                evaluator=evaluator_names["penalized"]["HDFS"],
                cases=hdfs_rows,
                kb=dataset_graphs.get("HDFS", []),
                rank_fn=calc_hdfs_rank,
                penalized=True,
                method=method,
            )
        )
        dataset_aligned.append(
            _evaluate_dataset(
                dataset="OpenStack",
                evaluator=evaluator_names["dataset_aligned"]["OpenStack"],
                cases=openstack_rows,
                kb=dataset_graphs.get("OpenStack", []),
                rank_fn=calc_openstack_path2_rank,
                penalized=False,
                method=method,
            )
        )
        penalized_rows.append(
            _evaluate_dataset(
                dataset="OpenStack",
                evaluator=evaluator_names["penalized"]["OpenStack"],
                cases=openstack_rows,
                kb=dataset_graphs.get("OpenStack", []),
                rank_fn=calc_openstack_path2_rank,
                penalized=True,
                method=method,
            )
        )
        dataset_aligned.append(
            _evaluate_dataset(
                dataset="Hadoop",
                evaluator=evaluator_names["dataset_aligned"]["Hadoop"],
                cases=hadoop_rows,
                kb=dataset_graphs.get("Hadoop", []),
                rank_fn=calc_hadoop_family_rank,
                penalized=False,
                method=method,
            )
        )
        penalized_rows.append(
            _evaluate_dataset(
                dataset="Hadoop",
                evaluator=evaluator_names["penalized"]["Hadoop"],
                cases=hadoop_rows,
                kb=dataset_graphs.get("Hadoop", []),
                rank_fn=calc_hadoop_family_rank,
                penalized=True,
                method=method,
            )
        )
    return {"dataset_aligned": dataset_aligned, "penalized": penalized_rows}
