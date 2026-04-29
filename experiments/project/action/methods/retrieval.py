"""Retrieval helpers."""

from __future__ import annotations

from dataclasses import dataclass

from experiments.project.action.core.schema import ActionCase, ActionQuery
from experiments.project.action.core.text import cosine_similarity


@dataclass(frozen=True)
class RetrievedCase:
    case_id: str
    score: float
    case: ActionCase
    raw_score: float | None = None
    graph_score: float | None = None


def topk_raw(
    query: ActionQuery,
    support_cases: list[ActionCase],
    *,
    feature_name: str,
    k: int,
) -> list[RetrievedCase]:
    scored = []
    query_feature = query.raw_features[feature_name]
    for support_case in support_cases:
        score = cosine_similarity(query_feature, support_case.raw_features[feature_name])
        scored.append(
            RetrievedCase(
                case_id=support_case.case_id,
                score=score,
                case=support_case,
                raw_score=score,
                graph_score=None,
            )
        )
    scored.sort(key=lambda item: (item.score, item.case.case_id), reverse=True)
    return scored[:k]


def topk_combined(
    query: ActionQuery,
    support_cases: list[ActionCase],
    *,
    raw_feature_name: str,
    graph_feature_name: str,
    raw_weight: float,
    graph_weight: float,
    k: int,
) -> list[RetrievedCase]:
    scored = []
    query_raw = query.raw_features[raw_feature_name]
    query_graph = query.base_case.graph_features[graph_feature_name]
    for support_case in support_cases:
        raw_score = cosine_similarity(query_raw, support_case.raw_features[raw_feature_name])
        graph_score = cosine_similarity(query_graph, support_case.graph_features[graph_feature_name])
        total_score = raw_weight * raw_score + graph_weight * graph_score
        scored.append(
            RetrievedCase(
                case_id=support_case.case_id,
                score=total_score,
                case=support_case,
                raw_score=raw_score,
                graph_score=graph_score,
            )
        )
    scored.sort(key=lambda item: (item.score, item.case.case_id), reverse=True)
    return scored[:k]
