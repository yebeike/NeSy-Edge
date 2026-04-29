"""Public in-memory graph builders for the reasoning layer."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from experiments.project.reasoning.core.matching import exact_relaxed_match, normalize_template
from experiments.project.reasoning.core.priors import (
    collect_symbolic_prior_edges,
    merge_reasoning_edges,
)


def _normalize_excluded_pairs(
    pairs: Iterable[tuple[str, str]] | None,
) -> set[tuple[str, str]]:
    normalized: set[tuple[str, str]] = set()
    for source, target in pairs or []:
        source_key = normalize_template(source)
        target_key = normalize_template(target)
        if source_key and target_key:
            normalized.add((source_key, target_key))
    return normalized


def _filter_edges(
    rows: Iterable[Mapping[str, Any]],
    *,
    excluded_pairs: set[tuple[str, str]],
) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    for row in rows:
        source = str(row.get("source_template", "") or "")
        target = str(row.get("target_template", "") or "")
        source_key = normalize_template(source)
        target_key = normalize_template(target)
        if not source_key or not target_key:
            continue
        if exact_relaxed_match(source, target):
            continue
        if (source_key, target_key) in excluded_pairs:
            continue
        filtered.append(dict(row))
    return filtered


def build_nesy_edge_reasoning_graph(
    *,
    dataset: str,
    backbone_rows: Iterable[Mapping[str, Any]],
    template_pool: Iterable[str],
    symbolic_rows: Iterable[Mapping[str, Any]],
    curated_rows: Iterable[Mapping[str, Any]],
    excluded_pairs: Iterable[tuple[str, str]] | None = None,
) -> list[dict[str, Any]]:
    """Build one NeSy-Edge candidate graph from a backbone and symbolic priors."""
    symbolic_edges = collect_symbolic_prior_edges(
        dataset,
        template_pool,
        symbolic_rows=symbolic_rows,
        curated_rows=curated_rows,
    )
    merged = merge_reasoning_edges(
        dataset,
        original_rows=backbone_rows,
        symbolic_rows=symbolic_edges,
    )
    return _filter_edges(
        merged,
        excluded_pairs=_normalize_excluded_pairs(excluded_pairs),
    )


def build_reasoning_candidate_graphs(
    *,
    template_pools_by_dataset: Mapping[str, Iterable[str]],
    backbones_by_dataset: Mapping[str, Iterable[Mapping[str, Any]]],
    symbolic_rows: Iterable[Mapping[str, Any]],
    curated_rows: Iterable[Mapping[str, Any]],
    excluded_pairs_by_dataset: Mapping[str, Iterable[tuple[str, str]]] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Build candidate graphs for each dataset in one pass."""
    outputs: dict[str, list[dict[str, Any]]] = {}
    for dataset, template_pool in template_pools_by_dataset.items():
        outputs[dataset] = build_nesy_edge_reasoning_graph(
            dataset=dataset,
            backbone_rows=backbones_by_dataset.get(dataset, []),
            template_pool=template_pool,
            symbolic_rows=symbolic_rows,
            curated_rows=curated_rows,
            excluded_pairs=(excluded_pairs_by_dataset or {}).get(dataset, []),
        )
    return outputs
