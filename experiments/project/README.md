# NeSy-Edge Public Code Package

Public package for the NeSy-Edge.

Install dependencies with:

```bash
pip install -r experiments/project/requirements.txt
```

## Layout

- `perception/`: log parsing and perception evaluation code
- `reasoning/`: in-memory causal-graph construction and evaluation code
- `action/`: action schema, noise, prompting, and method code
- `shared/`: shared utilities

## Perception

`perception/` is organized as:

- `core/`: preprocessing, noise, evaluation, parser routing, method factories
- `llm/`: local fallback parser and baseline parsers
- `retrieval/`: lexical/embedding retrieval, bundle builders, runtime helpers
- `protocol.py`: end-to-end perception protocol runner


```text
experiments/project/data/manifests/<manifest_id>/
experiments/project/artifacts/manifests/<manifest_id>/
```

Minimal usage:

```python
from experiments.project.perception import (
    build_direct_local_llm_protocol_method,
    build_drain_replay_protocol_method,
    build_nesy_embedding_local_llm_protocol_method,
    run_perception_protocol,
)
from experiments.project.perception.retrieval.backends import (
    SentenceTransformerEmbeddingBackend,
    config_for_backend_preset,
)
from experiments.project.perception.retrieval.bundles import build_embedding_bundle

manifest_id = "your_manifest_id"
bundle_id = "embedding_l2_qwen_your_manifest_id"

config = config_for_backend_preset("qwen-symmetric")
backend = SentenceTransformerEmbeddingBackend(config)

build_embedding_bundle(
    manifest_id,
    bundle_id,
    backend=backend,
    embedding_model_id=config.model_id,
)

methods = [
    build_drain_replay_protocol_method(manifest_id),
    build_direct_local_llm_protocol_method(),
    build_nesy_embedding_local_llm_protocol_method(
        manifest_id,
        bundle_id,
        query_backend=backend,
        retriever_method_id="embedding_l2_qwen",
    ),
]

payload = run_perception_protocol(manifest_id, method_specs=methods)
print(payload["summary"])
```

## Reasoning

`reasoning/` contains:

- `core/`: graph building, matching, symbolic priors, evaluation
- `paths.py`: optional local symbolic-knowledge paths
- `protocol.py`: public core entry points

Minimal usage:

```python
from experiments.project.reasoning import (
    build_reasoning_candidate_graphs,
    load_reasoning_symbolic_sources,
    evaluate_reasoning_graphs,
)
```

## Action

`action/` contains:

- `core/`: schema, noise, text features, evaluation helpers
- `methods/`: `vanilla`, `rag`, and `nesy` action methods
- `query.py`: query construction helpers from user-owned cases

Minimal usage:

```python
from experiments.project.action import (
    BenchmarkBundle,
    NeSyFormalRunner,
    build_query,
    support_cases,
)
```
