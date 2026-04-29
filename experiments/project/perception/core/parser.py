"""Probe-ready `L1 -> L2 -> L3` controller for the perception layer."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from experiments.project.perception.core.models import (
    PerceptionParseResult,
    PerceptionRouteDiagnostics,
    RetrievalCandidate,
)
from experiments.project.perception.retrieval.retrieval import fingerprint_text


TemplateResolver = Callable[[str, str], str | None]
Retriever = Callable[[str, str, int], list[RetrievalCandidate]]
FallbackParser = Callable[[str, str, list[RetrievalCandidate]], tuple[str, dict[str, Any]]]


class PerceptionParser:
    """Minimal route controller for the rebuilt perception layer."""

    def __init__(
        self,
        *,
        retriever: Retriever,
        fallback_parser: FallbackParser,
        template_resolver: TemplateResolver | None = None,
        shortcut_thresholds: dict[str, float] | None = None,
        top_k: int = 5,
    ) -> None:
        self.retriever = retriever
        self.fallback_parser = fallback_parser
        self.template_resolver = template_resolver
        self.shortcut_thresholds = dict(shortcut_thresholds or {})
        self.top_k = top_k
        self._cache: dict[str, str] = {}

    def reset_cache(self) -> None:
        """Reset the per-run exact-match cache."""
        self._cache = {}

    def warmup(self, dataset_id: str) -> None:
        """Warm the fallback path when the attached fallback parser supports it."""
        if hasattr(self.retriever, "warmup"):
            self.retriever.warmup(dataset_id)
        if hasattr(self.fallback_parser, "warmup"):
            self.fallback_parser.warmup(dataset_id)

    def parse(self, query_text: str, dataset_id: str) -> PerceptionParseResult:
        """Parse one alert string through the `L1 -> L2 -> L3` route stack."""
        cache_key = f"{dataset_id}:{fingerprint_text(query_text)}"
        retrieval_method_id = getattr(self.retriever, "method_id", None)
        retrieval_metadata = (
            {"retrieval_method_id": str(retrieval_method_id)}
            if retrieval_method_id
            else {}
        )
        if cache_key in self._cache:
            return PerceptionParseResult(
                template=self._cache[cache_key],
                route="L1_cache",
                diagnostics=PerceptionRouteDiagnostics(
                    route="L1_cache",
                    query_text=query_text,
                    query_chars=len(query_text),
                    best_score=None,
                    candidate_count=0,
                ),
            )

        if self.template_resolver is not None:
            canonical = self.template_resolver(query_text, dataset_id)
            if canonical:
                self._cache[cache_key] = canonical
                return PerceptionParseResult(
                    template=canonical,
                    route="symbolic_canonical",
                    diagnostics=PerceptionRouteDiagnostics(
                        route="symbolic_canonical",
                        query_text=query_text,
                        query_chars=len(query_text),
                        best_score=None,
                        candidate_count=0,
                    ),
                )

        candidates = self.retriever(query_text, dataset_id, self.top_k)
        if candidates:
            best = candidates[0]
            threshold = self.shortcut_thresholds.get(dataset_id, 0.66)
            if best.score >= threshold:
                self._cache[cache_key] = best.template
                return PerceptionParseResult(
                    template=best.template,
                    route="symbolic_shortcut",
                    diagnostics=PerceptionRouteDiagnostics(
                        route="symbolic_shortcut",
                        query_text=query_text,
                        query_chars=len(query_text),
                        best_score=best.score,
                        candidate_count=len(candidates),
                        metadata=retrieval_metadata,
                    ),
                )

            predicted_template, metadata = self.fallback_parser(
                query_text,
                dataset_id,
                candidates,
            )
            self._cache[cache_key] = predicted_template
            return PerceptionParseResult(
                template=predicted_template,
                route="llm_fallback",
                diagnostics=PerceptionRouteDiagnostics(
                    route="llm_fallback",
                    query_text=query_text,
                    query_chars=len(query_text),
                    best_score=best.score,
                    candidate_count=len(candidates),
                    metadata={**retrieval_metadata, **metadata},
                ),
            )

        predicted_template, metadata = self.fallback_parser(query_text, dataset_id, [])
        self._cache[cache_key] = predicted_template
        return PerceptionParseResult(
            template=predicted_template,
            route="llm_cold_start",
            diagnostics=PerceptionRouteDiagnostics(
                route="llm_cold_start",
                query_text=query_text,
                query_chars=len(query_text),
                best_score=None,
                candidate_count=0,
                metadata={**retrieval_metadata, **metadata},
            ),
        )
