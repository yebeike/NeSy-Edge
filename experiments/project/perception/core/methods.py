"""Factory helpers for project-owned perception protocol method specs."""

from __future__ import annotations

from experiments.project.perception.llm.fallbacks import (
    FallbackParser,
    top_reference_template_fallback,
)
from experiments.project.perception.llm.baselines import (
    DirectLocalLlmParser,
    DrainReplayProtocolParser,
)
from experiments.project.perception.llm.local_llm import (
    ChatTemplateGenerator,
    LocalLlmFallbackParser,
    LocalLlmGenerationConfig,
    TransformersChatTemplateGenerator,
)
from experiments.project.perception.core.parser import PerceptionParser
from experiments.project.perception.paths import ProjectPaths
from experiments.project.perception.protocol import PerceptionMethodSpec
from experiments.project.perception.retrieval.runtime import (
    EmbeddingManifestRetriever,
    LexicalManifestRetriever,
    QueryEmbeddingBackend,
)


def _local_generator(
    *,
    generator: ChatTemplateGenerator | None = None,
    model_path: str | None = None,
) -> ChatTemplateGenerator:
    if generator is not None:
        return generator
    if model_path:
        return TransformersChatTemplateGenerator(
            LocalLlmGenerationConfig(model_path=model_path)
        )
    return TransformersChatTemplateGenerator(LocalLlmGenerationConfig())


def build_lexical_protocol_method(
    manifest_id: str,
    *,
    method_id: str = "lexical_l2_legacy_protocol",
    fallback_parser: FallbackParser = top_reference_template_fallback,
    shortcut_thresholds: dict[str, float] | None = None,
    top_k: int = 5,
    cache_scope: str = "per_case",
    method_role: str | None = None,
    paths: ProjectPaths | None = None,
) -> PerceptionMethodSpec:
    """Build one protocol method spec backed by the lexical retriever."""
    return PerceptionMethodSpec(
        method_id=method_id,
        cache_scope=cache_scope,
        method_role=method_role,
        parser_factory=lambda: PerceptionParser(
            retriever=LexicalManifestRetriever(manifest_id, paths=paths),
            fallback_parser=fallback_parser,
            shortcut_thresholds=shortcut_thresholds,
            top_k=top_k,
        ),
    )


def build_drain_replay_protocol_method(
    manifest_id: str,
    *,
    method_id: str = "baseline_drain",
    cache_scope: str = "per_case",
    method_role: str = "drain",
    paths: ProjectPaths | None = None,
) -> PerceptionMethodSpec:
    """Build one protocol method spec backed by the official-style Drain replay baseline."""
    return PerceptionMethodSpec(
        method_id=method_id,
        cache_scope=cache_scope,
        method_role=method_role,
        parser_factory=lambda: DrainReplayProtocolParser(manifest_id, paths=paths),
    )


def build_direct_local_llm_protocol_method(
    *,
    generator: ChatTemplateGenerator | None = None,
    model_path: str | None = None,
    method_id: str = "baseline_qwen_direct",
    cache_scope: str = "per_case",
    method_role: str = "qwen",
) -> PerceptionMethodSpec:
    """Build one protocol method spec backed by a direct local-Qwen parser."""
    local_generator = _local_generator(generator=generator, model_path=model_path)
    return PerceptionMethodSpec(
        method_id=method_id,
        cache_scope=cache_scope,
        method_role=method_role,
        parser_factory=lambda: DirectLocalLlmParser(generator=local_generator),
    )


def build_nesy_lexical_local_llm_protocol_method(
    manifest_id: str,
    *,
    generator: ChatTemplateGenerator | None = None,
    model_path: str | None = None,
    method_id: str = "nesy_edge_lexical_local_qwen",
    cache_scope: str = "per_case",
    method_role: str = "nesy",
    shortcut_thresholds: dict[str, float] | None = None,
    top_k: int = 5,
    paths: ProjectPaths | None = None,
) -> PerceptionMethodSpec:
    """Build one NeSy-Edge protocol method with lexical `L2` and local `L3`."""
    local_generator = _local_generator(generator=generator, model_path=model_path)
    return PerceptionMethodSpec(
        method_id=method_id,
        cache_scope=cache_scope,
        method_role=method_role,
        parser_factory=lambda: PerceptionParser(
            retriever=LexicalManifestRetriever(manifest_id, paths=paths),
            fallback_parser=LocalLlmFallbackParser(local_generator),
            shortcut_thresholds=shortcut_thresholds,
            top_k=top_k,
        ),
    )


def build_embedding_protocol_method(
    manifest_id: str,
    bundle_id: str,
    *,
    query_backend: QueryEmbeddingBackend,
    retriever_method_id: str,
    method_id: str | None = None,
    fallback_parser: FallbackParser = top_reference_template_fallback,
    shortcut_thresholds: dict[str, float] | None = None,
    top_k: int = 5,
    cache_scope: str = "per_case",
    method_role: str | None = None,
    paths: ProjectPaths | None = None,
) -> PerceptionMethodSpec:
    """Build one protocol method spec backed by an embedding retriever."""
    protocol_method_id = method_id or f"{retriever_method_id}_protocol"
    return PerceptionMethodSpec(
        method_id=protocol_method_id,
        cache_scope=cache_scope,
        method_role=method_role,
        parser_factory=lambda: PerceptionParser(
            retriever=EmbeddingManifestRetriever(
                manifest_id,
                bundle_id,
                query_backend=query_backend,
                method_id=retriever_method_id,
                paths=paths,
            ),
            fallback_parser=fallback_parser,
            shortcut_thresholds=shortcut_thresholds,
            top_k=top_k,
        ),
    )


def build_nesy_embedding_local_llm_protocol_method(
    manifest_id: str,
    bundle_id: str,
    *,
    query_backend: QueryEmbeddingBackend,
    retriever_method_id: str,
    generator: ChatTemplateGenerator | None = None,
    model_path: str | None = None,
    method_id: str | None = None,
    cache_scope: str = "per_case",
    method_role: str = "nesy",
    shortcut_thresholds: dict[str, float] | None = None,
    top_k: int = 5,
    paths: ProjectPaths | None = None,
) -> PerceptionMethodSpec:
    """Build one NeSy-Edge protocol method with embedding `L2` and local `L3`."""
    protocol_method_id = method_id or f"{retriever_method_id}_local_qwen_protocol"
    local_generator = _local_generator(generator=generator, model_path=model_path)
    return PerceptionMethodSpec(
        method_id=protocol_method_id,
        cache_scope=cache_scope,
        method_role=method_role,
        parser_factory=lambda: PerceptionParser(
            retriever=EmbeddingManifestRetriever(
                manifest_id,
                bundle_id,
                query_backend=query_backend,
                method_id=retriever_method_id,
                paths=paths,
            ),
            fallback_parser=LocalLlmFallbackParser(local_generator),
            shortcut_thresholds=shortcut_thresholds,
            top_k=top_k,
        ),
    )
