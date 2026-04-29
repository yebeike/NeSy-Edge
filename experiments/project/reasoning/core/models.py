"""Typed models for reasoning benchmark rows and graph payloads."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


JsonDict = dict[str, Any]


def _required(data: Mapping[str, Any], field_name: str) -> Any:
    if field_name not in data:
        raise KeyError(f"Missing required field: {field_name}")
    return data[field_name]


@dataclass(frozen=True)
class ReasoningBenchmarkVariant:
    variant_id: str
    manifest_id: str
    status: str
    dataset_counts: JsonDict
    total_cases: int
    root_semantics: str
    evaluator_semantics: str
    source_summary: str
    notes: str = ""

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ReasoningBenchmarkVariant":
        return cls(
            variant_id=str(_required(data, "variant_id")),
            manifest_id=str(_required(data, "manifest_id")),
            status=str(_required(data, "status")),
            dataset_counts=dict(_required(data, "dataset_counts")),
            total_cases=int(_required(data, "total_cases")),
            root_semantics=str(_required(data, "root_semantics")),
            evaluator_semantics=str(_required(data, "evaluator_semantics")),
            source_summary=str(_required(data, "source_summary")),
            notes=str(data.get("notes", "")),
        )

    def to_dict(self) -> JsonDict:
        return asdict(self)


@dataclass(frozen=True)
class ReasoningBenchmarkRow:
    dataset: str
    case_id: str
    effect_target_type: str
    effect_target_value: str
    root_target_type: str
    root_target_value: str
    benchmark_tier: str
    benchmark_source_workspace: str
    manual_prior_pair_overlap: bool
    effect_target_label: str
    root_target_label: str
    metadata: JsonDict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ReasoningBenchmarkRow":
        return cls(
            dataset=str(_required(data, "dataset")),
            case_id=str(_required(data, "case_id")),
            effect_target_type=str(_required(data, "effect_target_type")),
            effect_target_value=str(_required(data, "effect_target_value")),
            root_target_type=str(_required(data, "root_target_type")),
            root_target_value=str(_required(data, "root_target_value")),
            benchmark_tier=str(_required(data, "benchmark_tier")),
            benchmark_source_workspace=str(_required(data, "benchmark_source_workspace")),
            manual_prior_pair_overlap=bool(
                _required(data, "manual_prior_pair_overlap")
            ),
            effect_target_label=str(_required(data, "effect_target_label")),
            root_target_label=str(_required(data, "root_target_label")),
            metadata=dict(data.get("metadata", {})),
        )

    def to_dict(self) -> JsonDict:
        return asdict(self)


@dataclass(frozen=True)
class CausalGraphEdge:
    dataset: str
    method: str
    source_template: str
    relation: str
    target_template: str
    weight: float
    provenance: str = ""

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
        *,
        dataset: str,
        method: str,
    ) -> "CausalGraphEdge":
        return cls(
            dataset=dataset,
            method=method,
            source_template=str(_required(data, "source_template")),
            relation=str(_required(data, "relation")),
            target_template=str(_required(data, "target_template")),
            weight=float(data.get("weight", 0.0) or 0.0),
            provenance=str(data.get("provenance", "")),
        )

    def to_dict(self) -> JsonDict:
        return asdict(self)


@dataclass(frozen=True)
class CausalGraphArtifact:
    manifest_id: str
    bundle_id: str
    graph_id: str
    dataset: str
    method: str
    source_summary: str
    edge_count: int
    relation_counts: JsonDict
    provenance_counts: JsonDict
    edges: list[CausalGraphEdge] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CausalGraphArtifact":
        edges = [
            CausalGraphEdge.from_dict(
                item,
                dataset=str(_required(data, "dataset")),
                method=str(_required(data, "method")),
            )
            for item in data.get("edges", [])
        ]
        return cls(
            manifest_id=str(_required(data, "manifest_id")),
            bundle_id=str(_required(data, "bundle_id")),
            graph_id=str(_required(data, "graph_id")),
            dataset=str(_required(data, "dataset")),
            method=str(_required(data, "method")),
            source_summary=str(_required(data, "source_summary")),
            edge_count=int(_required(data, "edge_count")),
            relation_counts=dict(data.get("relation_counts", {})),
            provenance_counts=dict(data.get("provenance_counts", {})),
            edges=edges,
        )

    def to_dict(self) -> JsonDict:
        payload = asdict(self)
        payload["edges"] = [edge.to_dict() for edge in self.edges]
        return payload
