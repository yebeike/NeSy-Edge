"""Project-owned symbolic-prior helpers for reasoning graph builders."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

from experiments.project.reasoning.core.matching import (
    OTHER_FAMILY,
    canonical_tokens,
    exact_relaxed_match,
    family_of,
    fuzzy_match,
    normalize_template,
)
from experiments.project.reasoning.paths import ProjectPaths, project_paths
from experiments.project.shared.jsonio import read_json


_DATASET_TO_DOMAIN = {"HDFS": "hdfs", "OpenStack": "openstack", "Hadoop": "hadoop"}
SYMBOLIC_PRIOR_WEIGHT_BOOST = 0.5
MERGED_WEIGHT_BOOST = 0.25
MAX_INCOMING_PER_TARGET = 2
PRUNE_GENERIC_SOURCE_BACKBONE = True


def _template_lookup(template_pool: Iterable[str]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for template in template_pool:
        key = normalize_template(template)
        if key and key not in lookup:
            lookup[key] = str(template)
    return lookup


def _resolve_exact_template(
    template_lookup: Mapping[str, str],
    raw_template: str,
) -> str | None:
    key = normalize_template(raw_template)
    if not key:
        return None
    return template_lookup.get(key)


def _score_template_candidate(
    dataset: str,
    raw_template: str,
    candidate_template: str,
    *,
    desired_family: str,
) -> float:
    raw_norm = normalize_template(raw_template)
    candidate_norm = normalize_template(candidate_template)
    if not raw_norm or not candidate_norm:
        return 0.0
    if raw_norm == candidate_norm:
        return 1000.0

    score = 0.0
    raw_family = desired_family or family_of(dataset, raw_template)
    candidate_family = family_of(dataset, candidate_template)
    if (
        raw_family
        and raw_family == candidate_family
        and raw_family != OTHER_FAMILY[dataset]
    ):
        score += 120.0
    elif raw_family != candidate_family:
        score -= 40.0

    raw_tokens = set(canonical_tokens(raw_template))
    candidate_tokens = set(canonical_tokens(candidate_template))
    overlap = len(raw_tokens & candidate_tokens)
    score += overlap * 8.0
    if fuzzy_match(raw_template, candidate_template):
        score += 20.0
    if raw_norm in candidate_norm or candidate_norm in raw_norm:
        score += 12.0
    return score


def _resolve_template(
    dataset: str,
    template_lookup: Mapping[str, str],
    raw_template: str,
    *,
    template_pool: Iterable[str],
    desired_family: str,
) -> str | None:
    exact = _resolve_exact_template(template_lookup, raw_template)
    if exact:
        return exact

    scored: list[tuple[float, str]] = []
    for candidate in template_pool:
        score = _score_template_candidate(
            dataset,
            raw_template,
            candidate,
            desired_family=desired_family,
        )
        if score > 0.0:
            scored.append((score, str(candidate)))
    if not scored:
        return None
    scored.sort(key=lambda item: item[0], reverse=True)
    best_score, best_candidate = scored[0]
    if best_score >= 1000.0:
        return best_candidate

    raw_family = desired_family or family_of(dataset, raw_template)
    candidate_family = family_of(dataset, best_candidate)
    if raw_family != candidate_family or raw_family == OTHER_FAMILY[dataset]:
        return None
    if best_score < 28.0:
        return None
    return best_candidate


def _copy_edge(row: Mapping[str, Any], *, domain: str) -> dict[str, object]:
    return {
        "domain": domain,
        "source_template": str(row.get("source_template", "") or ""),
        "relation": str(row.get("relation", "") or ""),
        "target_template": str(row.get("target_template", "") or ""),
        "weight": float(row.get("weight", 0.0) or 0.0),
        "provenance": str(row.get("provenance", "") or ""),
    }


def _merge_symbolic_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, object]]:
    merged: dict[tuple[str, str, str], dict[str, object]] = {}
    for row in rows:
        source = str(row.get("source_template", "") or "")
        relation = str(row.get("relation", "") or "")
        target = str(row.get("target_template", "") or "")
        key = (normalize_template(source), relation, normalize_template(target))
        if not key[0] or not key[2]:
            continue
        current = merged.get(key)
        if current is None or abs(float(row.get("weight", 0.0) or 0.0)) > abs(
            float(current.get("weight", 0.0) or 0.0)
        ):
            merged[key] = dict(row)
    return list(merged.values())


def _clean_edges_for_domain(
    dataset: str,
    rows: Iterable[Mapping[str, Any]],
) -> list[dict[str, object]]:
    domain = _DATASET_TO_DOMAIN[dataset]
    cleaned: list[dict[str, object]] = []
    for row in rows:
        if str(row.get("domain", "") or "").lower() != domain:
            continue
        source = str(row.get("source_template", "") or "")
        target = str(row.get("target_template", "") or "")
        if not source or not target:
            continue
        if exact_relaxed_match(source, target):
            continue
        cleaned.append(
            {
                "domain": domain,
                "source_template": source,
                "relation": str(row.get("relation", "") or ""),
                "target_template": target,
                "weight": float(row.get("weight", 0.0) or 0.0),
                "provenance": str(row.get("provenance", "") or ""),
            }
        )
    return cleaned


def load_reasoning_symbolic_sources(
    *,
    paths: ProjectPaths | None = None,
    symbolic_knowledge_asset: str = "symbolic_knowledge_v1.json",
    curated_priors_asset: str = "curated_symbolic_priors_v1.json",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    resolved_paths = paths or project_paths()
    symbolic_rows = list(
        read_json(resolved_paths.reasoning_knowledge_path(symbolic_knowledge_asset))
    )
    curated_rows = list(
        read_json(resolved_paths.reasoning_knowledge_path(curated_priors_asset))
    )
    return symbolic_rows, curated_rows


def collect_symbolic_prior_edges(
    dataset: str,
    template_pool: Iterable[str],
    *,
    symbolic_rows: Iterable[Mapping[str, Any]],
    curated_rows: Iterable[Mapping[str, Any]],
) -> list[dict[str, object]]:
    domain = _DATASET_TO_DOMAIN[dataset]
    materialized_pool = list(template_pool)
    lookup = _template_lookup(materialized_pool)
    priors: list[dict[str, object]] = []

    for row in list(symbolic_rows) + list(curated_rows):
        if str(row.get("domain", "") or "").lower() != domain:
            continue
        raw_source = str(row.get("source_template", "") or "")
        raw_target = str(row.get("target_template", "") or "")
        mapped_source = _resolve_template(
            dataset,
            lookup,
            raw_source,
            template_pool=materialized_pool,
            desired_family=family_of(dataset, raw_source),
        )
        mapped_target = _resolve_template(
            dataset,
            lookup,
            raw_target,
            template_pool=materialized_pool,
            desired_family=family_of(dataset, raw_target),
        )
        if not mapped_source or not mapped_target:
            continue
        if exact_relaxed_match(mapped_source, mapped_target):
            continue
        priors.append(
            {
                "domain": domain,
                "source_template": mapped_source,
                "relation": "symbolic_prior",
                "target_template": mapped_target,
                "weight": float(
                    round(max(0.55, abs(float(row.get("weight", 0.0) or 0.0))), 4)
                ),
                "provenance": "symbolic_prior",
            }
        )
    return _merge_symbolic_rows(priors)


def merge_reasoning_edges(
    dataset: str,
    *,
    original_rows: Iterable[Mapping[str, Any]],
    symbolic_rows: Iterable[Mapping[str, Any]],
) -> list[dict[str, object]]:
    domain = _DATASET_TO_DOMAIN[dataset]
    merged: dict[tuple[str, str], dict[str, object]] = {}

    def _upsert(row: Mapping[str, Any], provenance_label: str) -> None:
        source = str(row.get("source_template", "") or "")
        target = str(row.get("target_template", "") or "")
        key = (normalize_template(source), normalize_template(target))
        current = merged.get(key)
        if current is None:
            edge = _copy_edge(row, domain=domain)
            edge["provenance"] = provenance_label
            merged[key] = edge
            return
        sources = set(str(current.get("provenance", "") or "").split("+"))
        sources.discard("")
        sources.add(provenance_label)
        current["provenance"] = "merged" if len(sources) > 1 else provenance_label
        if abs(float(row.get("weight", 0.0) or 0.0)) > abs(float(current.get("weight", 0.0) or 0.0)):
            current["weight"] = float(row.get("weight", 0.0) or 0.0)
            current["relation"] = str(row.get("relation", "") or current.get("relation", ""))

    for row in original_rows:
        if str(row.get("domain", "") or "").lower() != domain:
            continue
        _upsert(row, "original_backbone")
    for row in symbolic_rows:
        if str(row.get("domain", "") or "").lower() != domain:
            continue
        _upsert(row, "symbolic_prior")

    rows = list(merged.values())
    if PRUNE_GENERIC_SOURCE_BACKBONE:
        rows = [
            row
            for row in rows
            if not (
                str(row.get("provenance", "") or "") == "original_backbone"
                and family_of(dataset, str(row.get("source_template", "") or ""))
                == OTHER_FAMILY[dataset]
            )
        ]
    rows = _clean_edges_for_domain(dataset, rows)
    for row in rows:
        provenance = str(row.get("provenance", "") or "")
        if provenance == "symbolic_prior":
            row["weight"] = float(round(abs(float(row.get("weight", 0.0) or 0.0)) + SYMBOLIC_PRIOR_WEIGHT_BOOST, 4))
        elif provenance == "merged":
            weight = float(row.get("weight", 0.0) or 0.0)
            sign = -1.0 if weight < 0 else 1.0
            row["weight"] = float(round(sign * (abs(weight) + MERGED_WEIGHT_BOOST), 4))

    if MAX_INCOMING_PER_TARGET > 0:
        buckets: dict[str, list[dict[str, object]]] = defaultdict(list)
        for row in rows:
            buckets[str(row.get("target_template", "") or "")].append(row)
        limited: list[dict[str, object]] = []
        for target_rows in buckets.values():
            target_rows = sorted(
                target_rows,
                key=lambda row: abs(float(row.get("weight", 0.0) or 0.0)),
                reverse=True,
            )
            limited.extend(target_rows[:MAX_INCOMING_PER_TARGET])
        rows = limited
    return rows
