# RQ3: Causal-Augmented Diagnostic Agent

Lightweight ReAct-style agent (no LangChain) that uses RQ1 log parser, RQ2 causal graph, and ChromaDB for root cause diagnosis.

## Prerequisites

- **Data**: `data/processed/causal_knowledge.json` (from RQ2 exporter), `data/processed/rq3_test_set.json`
- **API**: `.env`: `gemini_api_key` / `gemini_api_key_2` (Gemini), or `deepseek_api_key` (DeepSeek). Default batch uses DeepSeek to save cost.
- **Env**: Use project venv (needs `openai`, `transformers`, and RQ1/RQ2 deps). RQ1 KB and models for `log_parser` (NuSyEdgeNode) required for full pipeline.

## Usage (from project root)

```bash
export PYTHONPATH=.

# --- Without API (baseline RCA) ---
python experiments/rq3/run_batch_offline.py --limit 50
python experiments/rq3/evaluate.py --predictions results/rq3/predictions_offline.json

# --- With API: full 36 causal_edge cases (recommended for RQ3 scale) ---
python experiments/rq3/run_batch.py --backend deepseek --limit 36 --max-tokens 1024 --delay 1
python experiments/rq3/evaluate.py

# --- With API: small test (e.g. 10 cases) ---
python experiments/rq3/run_batch.py --backend deepseek --limit 10 --max-tokens 1024 --delay 1
python experiments/rq3/evaluate.py
# Gemini (if quota): --backend gemini

# --- Baselines (comparison experiment) ---
python experiments/rq3/run_batch_vanilla.py --limit 36 --delay 1   # Baseline 1: no tools
python experiments/rq3/run_batch_rag.py --limit 36 --delay 1      # Baseline 2: RAG only (no causal)
python experiments/rq3/compare_rq3.py   # Evaluates ours + vanilla + rag, writes results/rq3/comparison.json

# --- Full 66 cases (scale + diversity: HDFS + OpenStack) ---
python experiments/rq3/run_batch.py --backend deepseek --limit 0 --max-tokens 1024 --delay 1
python experiments/rq3/run_batch_vanilla.py --limit 0 --delay 1
python experiments/rq3/run_batch_rag.py --limit 0 --delay 1
python experiments/rq3/compare_rq3.py

# --- Token (precise 66-case counts for comparison.json) ---
# Run Vanilla + RAG with token recording, then compare. Ours already has token in predictions.json.
python experiments/rq3/run_token_baselines.py --limit 0 --delay 1

# --- Data Efficiency (25% / 50% / 75% / 100% train -> causal graphs -> same 66 test -> curve) ---
# Export 4 causal graphs only (no API):
python experiments/rq3/run_data_efficiency.py --export-only
# Full: export + run RQ3 with each graph + evaluate + write results/rq3/data_efficiency_curve.json
python experiments/rq3/run_data_efficiency.py --limit 0 --delay 1
# Run RQ3 with a specific causal graph (e.g. for ablation):
python experiments/rq3/run_batch.py --backend deepseek --limit 0 --causal data/processed/causal_knowledge_25.json --output results/rq3/predictions_eff_25.json
```

## Files

- **run_batch_vanilla.py**: Baseline 1 (Vanilla LLM): raw log → LLM only. Output: `predictions_vanilla.json`.
- **run_batch_rag.py**: Baseline 2 (Standard RAG): raw log → ChromaDB retrieval → LLM. Output: `predictions_rag.json`.
- **compare_rq3.py**: Run evaluate on ours / vanilla / rag and write `comparison.json`.
- **run_token_baselines.py**: One-shot Vanilla + RAG (66 cases with token) then compare; fills `comparison.json` token/cost.
- **run_data_efficiency.py**: Export causal graphs at 25/50/75/100% train data; run RQ3 with each; write `data_efficiency_curve.json`.
- **tools.py**: `log_parser`, `causal_navigator`, `knowledge_retriever`; tool defs for API
- **agent.py**: Gemini REST API + tool-calling loop; `run_agent_with_instruction()`
- **run_batch.py**: Load test set, run agent per case, save predictions
- **evaluate.py**: RCA accuracy (cases with known GT root cause), hallucination rate

## Notes

- **Token/cost (DeepSeek)**: Use `deepseek-chat` (not reasoner), `--max-tokens 1024`. Input 2 CNY/M, output 3 CNY/M; cache hits 0.2 CNY/M. To save tokens: (1) short system prompt is used for DeepSeek; (2) user prompt is compact; (3) max_rounds capped at 6 for DeepSeek. ~36 cases typically stay within a few CNY.
- **429**: Batch retries once after 25s. Use `--delay` between cases. For Gemini both keys are tried.
- **No openai package**: Agent falls back to urllib for DeepSeek when `openai` is not installed (e.g. in CI); full run still requires venv with `openai` and `transformers` for `log_parser`.
- **Offline baseline**: `run_batch_offline.py` runs parser + causal_navigator only (no LLM). On the full set this gives RCA 100% on the 36 causal_edge cases.
- RCA is computed only where `ground_truth_root_cause_template` != `"unknown"`.
- Evaluator uses `extracted_root_cause` from causal_navigator (or heuristics from `model_answer`).
- Eval summary includes `by_dataset` (HDFS / OpenStack) for RCA breakdown.
- **Hallucination (causal-edge)**: A case is counted as hallucination if the model asserts a (cause → effect) link that does not exist in `causal_knowledge.json`. Effect = this case’s observed log template; cause = the stated root cause. Ours (causal_navigator) only outputs graph-backed causes → low hallucination; RAG/Vanilla often assert links not in the graph → higher hallucination.
- **Token usage**: Batch scripts record `token_usage` (prompt_tokens, completion_tokens) per case; `compare_rq3.py` aggregates and adds `token_usage` and `estimated_cost_cny` (DeepSeek 2/3 CNY/M) to `comparison.json`.
