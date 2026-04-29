"""Prompt rendering and prediction parsing helpers."""

from __future__ import annotations

import json

from experiments.project.action.core.schema import ActionCase, ActionQuery, LabelEntry
from experiments.project.action.methods.retrieval import RetrievedCase


def render_labelbook(entries: list[LabelEntry], *, title: str) -> str:
    lines = [title]
    for entry in entries:
        lines.append(f"- {entry.label}: {entry.description}")
    return "\n".join(lines)


def render_retrieved_cases(items: list[RetrievedCase]) -> str:
    if not items:
        return "No retrieved cases."
    lines = []
    for index, item in enumerate(items, start=1):
        lines.append(
            f"[Retrieved #{index}] score={item.score:.4f} case_id={item.case.case_id} "
            f"root={item.case.root_label} action={item.case.action_label}"
        )
        lines.append(item.case.support_summary)
    return "\n".join(lines)


def render_graph_facts(case: ActionCase, *, fact_key: str) -> str:
    facts = case.graph_facts.get(fact_key, [])
    if not facts:
        return "No graph facts available."
    return "\n".join(f"- {fact}" for fact in facts)


def json_output_contract(dataset: str) -> str:
    return (
        "Return strict JSON only with keys "
        '{"root_label": "...", "action_label": "...", "reasoning": "...", "confidence": 0.0}. '
        f"The selected labels must belong to the {dataset} labelbook."
    )


def build_vanilla_messages(
    query: ActionQuery,
    *,
    root_entries: list[LabelEntry],
    action_entries: list[LabelEntry],
) -> list[dict[str, str]]:
    system = (
        "You are the Vanilla LLM baseline for incident diagnosis. "
        "Use only the current incident evidence. Do not invent retrieved cases or graph facts. "
        + json_output_contract(query.base_case.dataset)
    )
    user = "\n\n".join(
        [
            render_labelbook(root_entries, title="Allowed root labels"),
            render_labelbook(action_entries, title="Allowed action labels"),
            f"Current incident evidence\n{query.incident_text}",
        ]
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def build_rag_messages(
    query: ActionQuery,
    *,
    root_entries: list[LabelEntry],
    action_entries: list[LabelEntry],
    retrieved: list[RetrievedCase],
) -> list[dict[str, str]]:
    system = (
        "You are the RAG-only baseline for incident diagnosis. "
        "Use the current incident evidence and the retrieved troubleshooting cases only. "
        "Do not use graph-derived causal facts. "
        + json_output_contract(query.base_case.dataset)
    )
    user = "\n\n".join(
        [
            render_labelbook(root_entries, title="Allowed root labels"),
            render_labelbook(action_entries, title="Allowed action labels"),
            f"Current incident evidence\n{query.incident_text}",
            f"Retrieved troubleshooting cases\n{render_retrieved_cases(retrieved)}",
        ]
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def build_nesy_messages(
    query: ActionQuery,
    *,
    root_entries: list[LabelEntry],
    action_entries: list[LabelEntry],
    retrieved: list[RetrievedCase],
    graph_fact_key: str,
) -> list[dict[str, str]]:
    system = (
        "You are an incident diagnosis assistant. "
        "Use only the current incident evidence, retrieved troubleshooting cases, "
        "and graph-derived causal facts explicitly listed below. "
        "Do not claim causal links beyond those graph facts. "
        + json_output_contract(query.base_case.dataset)
    )
    user = "\n\n".join(
        [
            render_labelbook(root_entries, title="Allowed root labels"),
            render_labelbook(action_entries, title="Allowed action labels"),
            f"Current incident evidence\n{query.incident_text}",
            f"Graph-derived causal facts\n{render_graph_facts(query.base_case, fact_key=graph_fact_key)}",
            f"Retrieved troubleshooting cases\n{render_retrieved_cases(retrieved)}",
        ]
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def parse_prediction_payload(text: str) -> dict[str, str | float | None]:
    stripped = str(text or "").strip()
    if not stripped:
        return {"root_label": None, "action_label": None, "reasoning": None, "confidence": None}
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        try:
            payload = json.loads(stripped[start : end + 1])
            return {
                "root_label": payload.get("root_label"),
                "action_label": payload.get("action_label"),
                "reasoning": payload.get("reasoning"),
                "confidence": payload.get("confidence"),
            }
        except json.JSONDecodeError:
            pass
    return {"root_label": None, "action_label": None, "reasoning": stripped, "confidence": None}
