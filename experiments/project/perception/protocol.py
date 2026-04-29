"""Project-owned perception full-protocol runner and artifact writer."""

from __future__ import annotations

import statistics
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from experiments.project.perception.core.evaluation import exact_match_hit
from experiments.project.perception.core.models import PerceptionParseResult
from experiments.project.perception.core.noise import NOISE_LEVELS, SemanticNoisePolicy
from experiments.project.perception.core.preprocessing import prepare_runtime_alert
from experiments.project.perception.manifest_io import iter_cases
from experiments.project.perception.paths import ProjectPaths, project_paths
from experiments.project.shared.jsonio import write_json, write_jsonl


class ParserProtocol(Protocol):
    """Minimal parser protocol required by the perception full-protocol runner."""

    def parse(self, query_text: str, dataset_id: str) -> PerceptionParseResult:
        """Parse one alert string into a predicted template."""

    def reset_cache(self) -> None:
        """Reset parser-local caches when the chosen cache scope requires it."""


ParserFactory = Callable[[], ParserProtocol]


@dataclass(frozen=True)
class PerceptionProtocolCase:
    """Stable project-owned protocol case derived from a manifest row."""

    incident_id: str
    case_id: str
    dataset_id: str
    clean_alert: str
    gt_template: str
    gt_source: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PerceptionMethodSpec:
    """Config for one method evaluated by the perception full protocol."""

    method_id: str
    parser_factory: ParserFactory
    cache_scope: str = "per_case"
    method_role: str | None = None


@dataclass(frozen=True)
class PerceptionProtocolRow:
    """One evaluated perception row emitted by the project-owned runner."""

    dataset_id: str
    incident_id: str
    case_id: str
    noise_level: float
    method_id: str
    method_role: str | None
    input_alert: str
    runtime_alert: str
    prediction: str
    gt_template: str
    pa_hit: float
    latency_ms: float
    route: str
    query_chars: int
    best_score: float | None
    candidate_count: int
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "incident_id": self.incident_id,
            "case_id": self.case_id,
            "noise_level": round(self.noise_level, 1),
            "method_id": self.method_id,
            "method_role": self.method_role,
            "input_alert": self.input_alert,
            "runtime_alert": self.runtime_alert,
            "prediction": self.prediction,
            "gt_template": self.gt_template,
            "pa_hit": self.pa_hit,
            "latency_ms": round(self.latency_ms, 3),
            "route": self.route,
            "query_chars": self.query_chars,
            "best_score": self.best_score,
            "candidate_count": self.candidate_count,
            "diagnostics": dict(self.diagnostics),
        }


def _case_input_alert(row: Mapping[str, Any]) -> str:
    selected_alert = str(row.get("selected_alert", "") or "").strip()
    if selected_alert:
        return selected_alert
    alert_event = row.get("alert_event", {})
    if isinstance(alert_event, Mapping):
        trigger_text = str(alert_event.get("trigger_text", "") or "").strip()
        if trigger_text:
            return trigger_text
    clean_alert = str(row.get("clean_alert", "") or "").strip()
    if clean_alert:
        return clean_alert
    return str(row.get("raw_alert", "") or "").strip()


def load_protocol_cases(
    manifest_id: str,
    *,
    paths: ProjectPaths | None = None,
) -> list[PerceptionProtocolCase]:
    """Load full-protocol perception cases from one project-owned manifest."""
    cases: list[PerceptionProtocolCase] = []
    for row in iter_cases(manifest_id, paths=paths):
        gt_template = str(
            row.get("gt_template")
            or (row.get("ground_truth", {}) or {}).get("template", "")
        )
        cases.append(
            PerceptionProtocolCase(
                incident_id=str(row.get("incident_id") or row.get("case_id") or ""),
                case_id=str(row.get("case_id") or row.get("incident_id") or ""),
                dataset_id=str(row["dataset_id"]),
                clean_alert=_case_input_alert(row),
                gt_template=gt_template,
                gt_source=str(row.get("gt_source", "")),
                metadata=dict(row.get("metadata", {})),
            )
        )
    return cases


def _select_protocol_cases(
    cases: list[PerceptionProtocolCase],
    *,
    dataset_ids: Sequence[str] | None = None,
    case_limit_per_dataset: int = 0,
    case_start: int = 0,
    case_stop: int = 0,
) -> list[PerceptionProtocolCase]:
    selected_datasets = {
        dataset_id.lower() for dataset_id in (dataset_ids or []) if dataset_id
    }
    filtered = [
        case
        for case in cases
        if not selected_datasets or case.dataset_id.lower() in selected_datasets
    ]
    if case_limit_per_dataset <= 0 and case_start <= 0 and case_stop <= 0:
        return filtered

    sliced: list[PerceptionProtocolCase] = []
    by_dataset: dict[str, list[PerceptionProtocolCase]] = defaultdict(list)
    for case in filtered:
        by_dataset[case.dataset_id].append(case)
    for dataset_id in sorted(by_dataset):
        dataset_cases = by_dataset[dataset_id]
        start = max(0, case_start)
        stop = case_stop or None
        window = dataset_cases[start:stop]
        if case_limit_per_dataset > 0:
            window = window[:case_limit_per_dataset]
        sliced.extend(window)
    return sliced


def _parser_cache_key(
    cache_scope: str,
    method_id: str,
    dataset_id: str,
    noise_level: float,
) -> tuple[str, ...]:
    if cache_scope == "global":
        return (method_id,)
    if cache_scope == "per_dataset":
        return (method_id, dataset_id)
    if cache_scope == "per_noise":
        return (method_id, dataset_id, f"{noise_level:.1f}")
    if cache_scope == "per_case":
        return (method_id, dataset_id, f"{noise_level:.1f}", "__case__")
    raise ValueError(f"Unsupported cache_scope: {cache_scope}")


def _get_parser(
    parser_cache: dict[tuple[str, ...], ParserProtocol],
    spec: PerceptionMethodSpec,
    dataset_id: str,
    noise_level: float,
) -> ParserProtocol:
    cache_key = _parser_cache_key(
        spec.cache_scope,
        spec.method_id,
        dataset_id,
        noise_level,
    )
    parser = parser_cache.get(cache_key)
    if parser is None:
        parser = spec.parser_factory()
        parser_cache[cache_key] = parser
    return parser


def summarize_protocol_rows(
    rows: Iterable[PerceptionProtocolRow],
    *,
    manifest_id: str,
    noise_levels: list[float],
    method_specs: list[PerceptionMethodSpec],
) -> dict[str, Any]:
    """Summarize full-protocol perception rows into paper-facing metrics."""
    grouped: dict[tuple[str, str, str], list[PerceptionProtocolRow]] = defaultdict(list)
    route_counts: dict[str, dict[str, dict[str, int]]] = defaultdict(dict)
    summary: dict[str, dict[str, dict[str, float | int]]] = defaultdict(dict)
    curves: dict[tuple[str, str], list[tuple[str, float]]] = defaultdict(list)
    role_map = {
        spec.method_role: spec.method_id
        for spec in method_specs
        if spec.method_role is not None
    }

    materialized_rows = list(rows)
    for row in materialized_rows:
        grouped[
            (
                row.dataset_id,
                f"{row.noise_level:.1f}",
                row.method_id,
            )
        ].append(row)

    for (dataset_id, noise_key, method_id), part in sorted(grouped.items()):
        latencies = [row.latency_ms for row in part]
        pa = sum(row.pa_hit for row in part) / len(part)
        route_counter = Counter(row.route for row in part)
        summary[dataset_id][f"{noise_key}:{method_id}"] = {
            "cases": len(part),
            "pa": round(pa, 4),
            "latency_ms": round(sum(latencies) / len(latencies), 3),
            "median_latency_ms": round(statistics.median(latencies), 3),
            "max_latency_ms": round(max(latencies), 3),
        }
        route_counts[dataset_id][f"{noise_key}:{method_id}"] = dict(route_counter)
        curves[(dataset_id, method_id)].append((noise_key, round(pa, 4)))

    acceptance_flags: list[str] = []
    highest_noise_key = f"{max(noise_levels):.1f}"
    for dataset_id, methods in summary.items():
        drain_method = role_map.get("drain")
        qwen_method = role_map.get("qwen")
        nesy_method = role_map.get("nesy")

        if drain_method and f"0.0:{drain_method}" in methods:
            if methods[f"0.0:{drain_method}"]["pa"] < 0.5:
                acceptance_flags.append(f"{dataset_id}: drain clean PA below 0.5")
        if qwen_method and f"0.0:{qwen_method}" in methods:
            if methods[f"0.0:{qwen_method}"]["pa"] < 0.5:
                acceptance_flags.append(f"{dataset_id}: qwen clean PA below 0.5")
        if nesy_method and drain_method and qwen_method:
            if (
                f"0.0:{nesy_method}" in methods
                and f"0.0:{drain_method}" in methods
                and f"0.0:{qwen_method}" in methods
            ):
                nesy_clean = float(methods[f"0.0:{nesy_method}"]["pa"])
                best_baseline = max(
                    float(methods[f"0.0:{drain_method}"]["pa"]),
                    float(methods[f"0.0:{qwen_method}"]["pa"]),
                )
                if nesy_clean + 0.02 < best_baseline:
                    acceptance_flags.append(
                        f"{dataset_id}: nesy clean PA not competitive"
                    )
            if (
                f"0.0:{drain_method}" in methods
                and f"{highest_noise_key}:{drain_method}" in methods
                and f"{highest_noise_key}:{nesy_method}" in methods
            ):
                drain_clean = float(methods[f"0.0:{drain_method}"]["pa"])
                drain_high = float(methods[f"{highest_noise_key}:{drain_method}"]["pa"])
                nesy_high = float(methods[f"{highest_noise_key}:{nesy_method}"]["pa"])
                if drain_clean - drain_high < 0.1:
                    acceptance_flags.append(
                        f"{dataset_id}: drain does not degrade enough under noise"
                    )
                if nesy_high <= drain_high:
                    acceptance_flags.append(
                        f"{dataset_id}: nesy not above drain at highest noise"
                    )
            if qwen_method:
                qwen_latencies = [
                    float(meta["latency_ms"])
                    for key, meta in methods.items()
                    if key.endswith(f":{qwen_method}")
                ]
                nesy_latencies = [
                    float(meta["latency_ms"])
                    for key, meta in methods.items()
                    if key.endswith(f":{nesy_method}")
                ]
                if qwen_latencies and nesy_latencies:
                    if statistics.mean(nesy_latencies) >= statistics.mean(qwen_latencies):
                        acceptance_flags.append(
                            f"{dataset_id}: nesy latency is not below qwen latency"
                        )

    for (dataset_id, method_id), seq in curves.items():
        seq.sort(key=lambda item: float(item[0]))
        pa_values = [pa for _, pa in seq]
        qwen_method = role_map.get("qwen")
        if qwen_method and method_id == qwen_method and len(pa_values) >= 3:
            if len(set(pa_values)) == 1:
                acceptance_flags.append(f"{dataset_id}: qwen PA exactly flat across noise levels")

    return {
        "manifest_id": manifest_id,
        "noise_levels": [round(level, 1) for level in noise_levels],
        "methods": [spec.method_id for spec in method_specs],
        "method_roles": {
            spec.method_id: spec.method_role
            for spec in method_specs
            if spec.method_role is not None
        },
        "query_count": len(materialized_rows),
        "route_counts": route_counts,
        "acceptance_flags": acceptance_flags,
        "summary": summary,
    }


def run_perception_protocol(
    manifest_id: str,
    *,
    method_specs: list[PerceptionMethodSpec],
    noise_levels: list[float] | None = None,
    noise_policy: SemanticNoisePolicy | None = None,
    dataset_ids: Sequence[str] | None = None,
    case_limit_per_dataset: int = 0,
    case_start: int = 0,
    case_stop: int = 0,
    paths: ProjectPaths | None = None,
) -> dict[str, Any]:
    """Run one project-owned perception full protocol over a manifest."""
    if not method_specs:
        raise ValueError("At least one method spec is required.")

    resolved_noise_levels = list(noise_levels or NOISE_LEVELS)
    policy = noise_policy or SemanticNoisePolicy()
    cases = _select_protocol_cases(
        load_protocol_cases(manifest_id, paths=paths),
        dataset_ids=dataset_ids,
        case_limit_per_dataset=case_limit_per_dataset,
        case_start=case_start,
        case_stop=case_stop,
    )
    parser_cache: dict[tuple[str, ...], ParserProtocol] = {}
    warmed_parsers: set[tuple[str, ...]] = set()
    rows: list[PerceptionProtocolRow] = []

    for dataset_id in sorted({case.dataset_id for case in cases}):
        dataset_cases = [case for case in cases if case.dataset_id == dataset_id]
        for noise_level in resolved_noise_levels:
            for case in dataset_cases:
                noisy_alert = policy.inject(case.clean_alert, dataset_id, noise_level)
                runtime_alert = prepare_runtime_alert(noisy_alert, dataset_id)
                for spec in method_specs:
                    parser = _get_parser(
                        parser_cache,
                        spec,
                        dataset_id,
                        noise_level,
                    )
                    warm_key = (
                        spec.method_id,
                        dataset_id,
                        f"{noise_level:.1f}",
                        spec.cache_scope,
                    )
                    if warm_key not in warmed_parsers and hasattr(parser, "warmup"):
                        parser.warmup(dataset_id)
                        warmed_parsers.add(warm_key)
                    if spec.cache_scope == "per_case" and hasattr(parser, "reset_cache"):
                        parser.reset_cache()
                    start = time.perf_counter()
                    result = parser.parse(runtime_alert, dataset_id)
                    latency_ms = (time.perf_counter() - start) * 1000.0
                    diagnostics = result.diagnostics
                    rows.append(
                        PerceptionProtocolRow(
                            dataset_id=dataset_id,
                            incident_id=case.incident_id,
                            case_id=case.case_id,
                            noise_level=noise_level,
                            method_id=spec.method_id,
                            method_role=spec.method_role,
                            input_alert=noisy_alert,
                            runtime_alert=runtime_alert,
                            prediction=result.template,
                            gt_template=case.gt_template,
                            pa_hit=exact_match_hit(result.template, case.gt_template),
                            latency_ms=latency_ms,
                            route=result.route,
                            query_chars=diagnostics.query_chars,
                            best_score=diagnostics.best_score,
                            candidate_count=diagnostics.candidate_count,
                            diagnostics=diagnostics.to_dict(),
                        )
                    )

    summary = summarize_protocol_rows(
        rows,
        manifest_id=manifest_id,
        noise_levels=resolved_noise_levels,
        method_specs=method_specs,
    )
    spec_payload = {
        "manifest_id": manifest_id,
        "noise_levels": [round(level, 1) for level in resolved_noise_levels],
        "methods": [
            {
                "method_id": spec.method_id,
                "cache_scope": spec.cache_scope,
                "method_role": spec.method_role,
            }
            for spec in method_specs
        ],
        "noise_policy_id": "semantic_noise_v1",
        "dataset_ids": [dataset_id.lower() for dataset_id in (dataset_ids or [])],
        "case_limit_per_dataset": case_limit_per_dataset,
        "case_start": case_start,
        "case_stop": case_stop,
    }
    return {
        "spec": spec_payload,
        "summary": summary,
        "rows": [row.to_dict() for row in rows],
    }


def write_protocol_artifact(
    manifest_id: str,
    run_id: str,
    payload: Mapping[str, Any],
    *,
    paths: ProjectPaths | None = None,
) -> dict[str, str]:
    """Write one perception full-protocol artifact bundle."""
    resolved_paths = paths or project_paths()
    run_paths = resolved_paths.perception_protocol_run(manifest_id, run_id)
    run_paths.run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_paths.spec_path, payload["spec"])
    write_json(run_paths.summary_path, payload["summary"])
    write_jsonl(run_paths.rows_path, payload["rows"])
    return {
        "spec_path": str(run_paths.spec_path),
        "summary_path": str(run_paths.summary_path),
        "rows_path": str(run_paths.rows_path),
    }
