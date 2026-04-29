"""Public perception-layer package for NeSy-Edge."""

from experiments.project.perception.core.methods import (
    build_direct_local_llm_protocol_method,
    build_drain_replay_protocol_method,
    build_embedding_protocol_method,
    build_nesy_embedding_local_llm_protocol_method,
)
from experiments.project.perception.core.parser import PerceptionParser
from experiments.project.perception.protocol import (
    PerceptionMethodSpec,
    load_protocol_cases,
    run_perception_protocol,
    summarize_protocol_rows,
)

__all__ = [
    "PerceptionMethodSpec",
    "PerceptionParser",
    "build_direct_local_llm_protocol_method",
    "build_drain_replay_protocol_method",
    "build_embedding_protocol_method",
    "build_nesy_embedding_local_llm_protocol_method",
    "load_protocol_cases",
    "run_perception_protocol",
    "summarize_protocol_rows",
]
