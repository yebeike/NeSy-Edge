"""Typed models for perception-layer artifacts and outputs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


JsonDict = dict[str, Any]


def _required(data: Mapping[str, Any], field_name: str) -> Any:
    if field_name not in data:
        raise KeyError(f"Missing required field: {field_name}")
    return data[field_name]


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    raise TypeError(f"Expected list, got {type(value)!r}")


@dataclass(frozen=True)
class EmbeddingBundleMetadata:
    """Metadata for one precomputed reference-embedding bundle."""

    schema_version: str
    bundle_id: str
    manifest_id: str
    dataset_id: str
    reference_count: int
    preprocessing_version: str
    embedding_model_id: str
    embedding_dimension: int
    created_at: str
    notes: str = ""

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EmbeddingBundleMetadata":
        return cls(
            schema_version=str(_required(data, "schema_version")),
            bundle_id=str(_required(data, "bundle_id")),
            manifest_id=str(_required(data, "manifest_id")),
            dataset_id=str(_required(data, "dataset_id")),
            reference_count=int(_required(data, "reference_count")),
            preprocessing_version=str(_required(data, "preprocessing_version")),
            embedding_model_id=str(_required(data, "embedding_model_id")),
            embedding_dimension=int(_required(data, "embedding_dimension")),
            created_at=str(_required(data, "created_at")),
            notes=str(data.get("notes", "")),
        )

    def to_dict(self) -> JsonDict:
        return asdict(self)


@dataclass(frozen=True)
class ReferenceRow:
    """Stable project-owned reference row used by perception retrieval."""

    reference_id: str
    case_id: str
    dataset_id: str
    raw_log: str
    clean_alert: str
    gt_template: str
    metadata: JsonDict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ReferenceRow":
        raw_window = data.get("raw_window", {})
        if not isinstance(raw_window, Mapping):
            raw_window = {}
        raw_lines = raw_window.get("lines", [])
        if not isinstance(raw_lines, list):
            raw_lines = []
        fallback_raw_log = ""
        if raw_lines:
            fallback_raw_log = str(raw_lines[-1])
        elif raw_window.get("source_path") is not None:
            fallback_raw_log = str(raw_window.get("source_path"))

        metadata = dict(data.get("metadata", {}))
        gt_template = data.get("gt_template")
        if gt_template is None:
            gt_template = metadata.get("gt_template", "")

        return cls(
            reference_id=str(_required(data, "reference_id")),
            case_id=str(data.get("case_id", data.get("reference_id", ""))),
            dataset_id=str(_required(data, "dataset_id")),
            raw_log=str(data.get("raw_log", fallback_raw_log)),
            clean_alert=str(data.get("clean_alert", data.get("raw_log", fallback_raw_log))),
            gt_template=str(gt_template),
            metadata=metadata,
        )

    def to_dict(self) -> JsonDict:
        return asdict(self)


@dataclass(frozen=True)
class RetrievalCandidate:
    """One perception-layer retrieval candidate."""

    reference_id: str
    case_id: str
    template: str
    score: float
    matched_text: str

    def to_dict(self) -> JsonDict:
        return asdict(self)


@dataclass(frozen=True)
class PerceptionRouteDiagnostics:
    """Route-level diagnostics for one parsed alert."""

    route: str
    query_text: str
    query_chars: int
    best_score: float | None
    candidate_count: int
    metadata: JsonDict = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        return asdict(self)


@dataclass(frozen=True)
class StructuredEvent:
    """One structured event output from the perception layer."""

    event_id: str
    template: str
    source_alert: str
    metadata: JsonDict = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        return asdict(self)


@dataclass(frozen=True)
class StructuredEventSet:
    """Canonical perception output container."""

    incident_id: str
    dataset_id: str
    method_id: str
    events: list[StructuredEvent]
    diagnostics: JsonDict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "StructuredEventSet":
        return cls(
            incident_id=str(_required(data, "incident_id")),
            dataset_id=str(_required(data, "dataset_id")),
            method_id=str(_required(data, "method_id")),
            events=[
                StructuredEvent(
                    event_id=str(_required(event, "event_id")),
                    template=str(_required(event, "template")),
                    source_alert=str(_required(event, "source_alert")),
                    metadata=dict(event.get("metadata", {})),
                )
                for event in _as_list(_required(data, "events"))
            ],
            diagnostics=dict(data.get("diagnostics", {})),
        )

    def to_dict(self) -> JsonDict:
        payload = asdict(self)
        payload["events"] = [event.to_dict() for event in self.events]
        return payload


@dataclass(frozen=True)
class PerceptionParseResult:
    """Canonical result container for one perception parse step."""

    template: str
    route: str
    diagnostics: PerceptionRouteDiagnostics

    def to_dict(self) -> JsonDict:
        return {
            "template": self.template,
            "route": self.route,
            "diagnostics": self.diagnostics.to_dict(),
        }
