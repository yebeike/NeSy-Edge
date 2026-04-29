"""Shared schema types."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class LabelEntry:
    dataset: str
    label: str
    description: str
    kind: str


@dataclass
class ActionCase:
    dataset: str
    case_id: str
    benchmark_label: str
    root_label: str
    root_description: str
    action_label: str
    action_description: str
    title: str
    incident_text: str
    support_summary: str
    raw_features: dict[str, Counter[str]] = field(default_factory=dict)
    graph_features: dict[str, Counter[str]] = field(default_factory=dict)
    graph_facts: dict[str, list[str]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "case_id": self.case_id,
            "benchmark_label": self.benchmark_label,
            "root_label": self.root_label,
            "root_description": self.root_description,
            "action_label": self.action_label,
            "action_description": self.action_description,
            "title": self.title,
            "incident_text": self.incident_text,
            "support_summary": self.support_summary,
            "raw_features": {
                name: dict(counter)
                for name, counter in self.raw_features.items()
            },
            "graph_features": {
                name: dict(counter)
                for name, counter in self.graph_features.items()
            },
            "graph_facts": self.graph_facts,
            "metadata": self.metadata,
        }


@dataclass
class ActionQuery:
    base_case: ActionCase
    noise_level: float
    seed: int
    incident_text: str
    raw_features: dict[str, Counter[str]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_case_id": self.base_case.case_id,
            "dataset": self.base_case.dataset,
            "noise_level": self.noise_level,
            "seed": self.seed,
            "incident_text": self.incident_text,
            "raw_features": {
                name: dict(counter)
                for name, counter in self.raw_features.items()
            },
            "metadata": self.metadata,
        }


@dataclass
class BenchmarkBundle:
    cases: list[ActionCase]
    labelbook: dict[str, dict[str, list[LabelEntry]]]
    benchmark_source: str
    method_source: str

    def cases_by_dataset(self) -> dict[str, list[ActionCase]]:
        payload: dict[str, list[ActionCase]] = {}
        for case in self.cases:
            payload.setdefault(case.dataset, []).append(case)
        for dataset in payload:
            payload[dataset].sort(key=lambda item: item.case_id)
        return payload

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark_source": self.benchmark_source,
            "method_source": self.method_source,
            "case_count": len(self.cases),
            "cases": [case.to_dict() for case in self.cases],
            "labelbook": {
                dataset: {
                    kind: [
                        {
                            "dataset": entry.dataset,
                            "label": entry.label,
                            "description": entry.description,
                            "kind": entry.kind,
                        }
                        for entry in entries
                    ]
                    for kind, entries in payload.items()
                }
                for dataset, payload in self.labelbook.items()
            },
        }
