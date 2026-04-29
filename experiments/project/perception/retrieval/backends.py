"""Optional local embedding backends for the perception layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


MINILM_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
QWEN3_EMBEDDING_MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
QWEN3_PERCEPTION_QUERY_INSTRUCTION = (
    "Given an alert log line, retrieve semantically similar reference alert "
    "logs from the same system family."
)


def build_instructed_query(task_instruction: str, query_text: str) -> str:
    """Format one query-side instruction string for instruction-aware models."""
    return f"Instruct: {task_instruction}\nQuery: {query_text}"


@dataclass(frozen=True)
class EmbeddingBackendConfig:
    """Configuration for one local embedding backend."""

    model_id: str
    normalize_embeddings: bool = True
    query_instruction: str | None = None
    model_kwargs: dict[str, Any] = field(default_factory=dict)
    tokenizer_kwargs: dict[str, Any] = field(default_factory=dict)


class SentenceTransformerEmbeddingBackend:
    """Lazy-loading sentence-transformers backend for local embedding probes."""

    def __init__(self, config: EmbeddingBackendConfig) -> None:
        self.config = config
        self._model: Any | None = None

    def _load_model(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                self.config.model_id,
                model_kwargs=self.config.model_kwargs or None,
                tokenizer_kwargs=self.config.tokenizer_kwargs or None,
            )
        return self._model

    def warmup(self) -> None:
        """Load weights before timed protocol rows begin."""
        self._load_model()

    def prepare_queries(self, queries: list[str]) -> list[str]:
        """Apply the optional instruction policy to query-side texts."""
        if not self.config.query_instruction:
            return list(queries)
        return [
            build_instructed_query(self.config.query_instruction, query_text)
            for query_text in queries
        ]

    def prepare_references(self, references: list[str]) -> list[str]:
        """Prepare reference-side texts without query instructions."""
        return list(references)

    def embed_queries(self, queries: list[str]) -> list[list[float]]:
        """Encode query texts with the configured local backend."""
        model = self._load_model()
        prepared = self.prepare_queries(queries)
        vectors = model.encode(
            prepared,
            normalize_embeddings=self.config.normalize_embeddings,
        )
        return vectors.tolist()

    def embed_references(self, references: list[str]) -> list[list[float]]:
        """Encode reference texts with the configured local backend."""
        model = self._load_model()
        prepared = self.prepare_references(references)
        vectors = model.encode(
            prepared,
            normalize_embeddings=self.config.normalize_embeddings,
        )
        return vectors.tolist()


def config_for_backend_preset(preset_id: str) -> EmbeddingBackendConfig:
    """Return one project-owned embedding backend preset."""
    normalized = preset_id.strip().lower()
    if normalized == "minilm-symmetric":
        return EmbeddingBackendConfig(model_id=MINILM_MODEL_ID)
    if normalized in {"qwen-symmetric", "qwen3-embedding-0.6b"}:
        return EmbeddingBackendConfig(
            model_id=QWEN3_EMBEDDING_MODEL_ID,
            tokenizer_kwargs={"padding_side": "left"},
        )
    if normalized in {"qwen-instruct", "qwen3-embedding-0.6b-instruct"}:
        return EmbeddingBackendConfig(
            model_id=QWEN3_EMBEDDING_MODEL_ID,
            query_instruction=QWEN3_PERCEPTION_QUERY_INSTRUCTION,
            tokenizer_kwargs={"padding_side": "left"},
        )
    raise KeyError(f"Unknown embedding backend preset: {preset_id}")
