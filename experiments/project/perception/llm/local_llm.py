"""Project-owned local `L3` fallback contract for the perception layer."""

from __future__ import annotations

import re
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from experiments.project.perception.core.models import RetrievalCandidate
from experiments.project.perception.core.preprocessing import bounded_embedding_text


DEFAULT_L3_MODEL_PATH = "models/qwen3-0.6b"
DEFAULT_L3_TOP_K = {"hdfs": 2, "openstack": 2, "hadoop": 3}
DEFAULT_L3_MAX_NEW_TOKENS = {"hdfs": 16, "openstack": 24, "hadoop": 28}


@dataclass(frozen=True)
class LocalLlmGenerationConfig:
    """Configuration for one project-owned local perception fallback backend."""

    model_path: str = DEFAULT_L3_MODEL_PATH
    max_new_tokens: dict[str, int] = field(
        default_factory=lambda: dict(DEFAULT_L3_MAX_NEW_TOKENS)
    )
    do_sample: bool = False
    top_k_refs: dict[str, int] = field(default_factory=lambda: dict(DEFAULT_L3_TOP_K))


class ChatTemplateGenerator(Protocol):
    """Minimal generator protocol required by the local fallback adapter."""

    def generate(
        self,
        messages: list[dict[str, str]],
        dataset_id: str,
    ) -> tuple[str, float]:
        """Generate one template string and return generation latency."""


def _system_prompt() -> str:
    return (
        "You are a log parser.\n"
        "Output exactly one event template line.\n"
        "Keep all constant words, punctuation, and field labels from the input.\n"
        "Do not invent prefixes or drop constant prefixes when they are present.\n"
        "Preserve HTTP method/path scaffolds and status/len/time fields when they are present.\n"
        "Preserve decimal placeholders as <*>.<*> when decimals are present.\n"
        "Replace dynamic values with <*>.\n"
        "Never explain and never add commentary."
    )


def _first_line(text: str) -> str:
    stripped = (text or "").strip()
    if not stripped:
        return ""
    return stripped.splitlines()[0].strip()


def normalize_generated_template(text: str) -> str:
    """Normalize one raw local LLM response into a single template line."""
    value = _first_line(text)
    value = re.sub(
        r"^(Template:|Output:|Result:|Event Template Line:|Event:)\s*",
        "",
        value,
        flags=re.IGNORECASE,
    )
    value = re.sub(r"^\[\s*event_template\s*:\s*", "[", value, flags=re.IGNORECASE)
    value = re.sub(r"<[^>]+>", "<*>", value)
    value = re.sub(r"\b\d{1,3}(?:\.\d{1,3}){3}(?::\d+)?\b", "<*>", value)
    value = re.sub(
        r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
        "<*>",
        value,
        flags=re.IGNORECASE,
    )
    value = re.sub(r"\b(?:[a-z0-9_-]+\.){2,}[a-z]{2,}\b", "<*>", value, flags=re.IGNORECASE)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def budget_prompt_text(text: str, dataset_id: str) -> str:
    """Prepare bounded local-fallback prompt text."""
    value = bounded_embedding_text(text, dataset_id)
    if dataset_id.lower() == "openstack":
        value = re.sub(
            r"\b(?:[a-z0-9_-]+\.){2,}[a-z]{2,}\b",
            "<*>",
            value,
            flags=re.IGNORECASE,
        )
    return value


def build_direct_messages(query_text: str, dataset_id: str) -> list[dict[str, str]]:
    """Build a no-reference local fallback prompt."""
    budgeted_query = budget_prompt_text(query_text, dataset_id)
    return [
        {"role": "system", "content": _system_prompt()},
        {
            "role": "user",
            "content": f"Input Log: {budgeted_query}\nTemplate:",
        },
    ]


def build_reference_messages(
    query_text: str,
    dataset_id: str,
    candidates: Sequence[RetrievalCandidate],
    *,
    top_k: int,
) -> list[dict[str, str]]:
    """Build a reference-guided local fallback prompt."""
    budgeted_query = budget_prompt_text(query_text, dataset_id)
    ref_block = "\n".join(
        (
            f"Reference Log: {budget_prompt_text(candidate.matched_text, dataset_id)}\n"
            f"Reference Template: {candidate.template}"
        )
        for candidate in list(candidates)[:top_k]
    )
    return [
        {"role": "system", "content": _system_prompt()},
        {
            "role": "user",
            "content": f"{ref_block}\nInput Log: {budgeted_query}\nTemplate:",
        },
    ]


class TransformersChatTemplateGenerator:
    """Lazy-loading local transformers backend for perception fallback parsing."""

    def __init__(self, config: LocalLlmGenerationConfig) -> None:
        self.config = config
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._device: str | None = None

    def _load(self) -> tuple[Any, Any, str]:
        if self._tokenizer is not None and self._model is not None and self._device:
            return self._tokenizer, self._model, self._device

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_path = Path(self.config.model_path)
        if not model_path.is_absolute():
            repo_root = Path(__file__).resolve().parents[4]
            model_path = repo_root / model_path

        device = "mps" if torch.backends.mps.is_available() else "cpu"
        if torch.cuda.is_available():
            device = "cuda"

        tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            torch_dtype="auto",
            device_map=device,
            trust_remote_code=True,
        )
        model.eval()

        self._tokenizer = tokenizer
        self._model = model
        self._device = device
        return tokenizer, model, device

    def generate(
        self,
        messages: list[dict[str, str]],
        dataset_id: str,
    ) -> tuple[str, float]:
        import torch

        tokenizer, model, device = self._load()
        try:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except Exception:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        inputs = tokenizer([text], return_tensors="pt").to(device)
        start = time.perf_counter()
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=self.config.max_new_tokens.get(dataset_id, 24),
                do_sample=self.config.do_sample,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        latency_ms = (time.perf_counter() - start) * 1000.0
        prompt_len = int(inputs.input_ids.shape[1])
        response = tokenizer.decode(
            generated_ids[0][prompt_len:],
            skip_special_tokens=True,
        )
        return normalize_generated_template(response), latency_ms

    def warmup(self, _dataset_id: str) -> None:
        """Load tokenizer and weights before timed protocol rows begin."""
        self._load()


class LocalLlmFallbackParser:
    """Reference-aware local fallback parser for the project-owned perception path."""

    def __init__(
        self,
        generator: ChatTemplateGenerator,
        *,
        top_k_refs: dict[str, int] | None = None,
    ) -> None:
        self.generator = generator
        self.top_k_refs = dict(top_k_refs or DEFAULT_L3_TOP_K)

    def warmup(self, dataset_id: str) -> None:
        """Warm the attached generator when it exposes a warmup hook."""
        if hasattr(self.generator, "warmup"):
            self.generator.warmup(dataset_id)

    def __call__(
        self,
        query_text: str,
        dataset_id: str,
        candidates: list[RetrievalCandidate],
    ) -> tuple[str, dict[str, Any]]:
        dataset_key = dataset_id.lower()
        if candidates:
            top_k = self.top_k_refs.get(dataset_key, 2)
            messages = build_reference_messages(
                query_text,
                dataset_key,
                candidates,
                top_k=top_k,
            )
            template, latency_ms = self.generator.generate(messages, dataset_key)
            return template, {
                "fallback_mode": "local_llm_with_references",
                "llm_latency_ms": round(latency_ms, 3),
                "ref_count": min(len(candidates), top_k),
            }

        messages = build_direct_messages(query_text, dataset_key)
        template, latency_ms = self.generator.generate(messages, dataset_key)
        return template, {
            "fallback_mode": "local_llm_direct",
            "llm_latency_ms": round(latency_ms, 3),
            "ref_count": 0,
        }
