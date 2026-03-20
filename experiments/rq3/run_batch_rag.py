# Baseline 2: Standard RAG — retrieve by raw log (no causal graph), then LLM. Same test set, same eval format.
# Usage (from project root): PYTHONPATH=. python experiments/rq3/run_batch_rag.py [--limit N]

import os
import sys
import json
import argparse
import time

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

DATA_PROCESSED = os.path.join(_PROJECT_ROOT, "data", "processed")
RESULTS_RQ3 = os.path.join(_PROJECT_ROOT, "results", "rq3")

RAG_PROMPT = """You are an ops diagnostic expert. Use ONLY the reference logs below to infer root cause. Do not invent causes not supported by the references.

Reference logs (from knowledge base):
{refs}

---
Current log to diagnose:
{raw_log}
Dataset: {dataset}

Give: 1) Root cause (brief, consistent with references). 2) One recommended action."""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="Max cases (0 = all)")
    ap.add_argument("--test-set", default="", help="Path to test set JSON")
    ap.add_argument("--delay", type=float, default=1.0, help="Seconds between API calls")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--top-k", type=int, default=3, help="Retrieval top_k")
    ap.add_argument("--output", default="", help="Output JSON path (default: results/rq3/predictions_rag.json)")
    args = ap.parse_args()

    test_set_path = args.test_set or os.path.join(DATA_PROCESSED, "rq3_test_set.json")
    if not os.path.exists(test_set_path):
        print(f"Test set not found: {test_set_path}")
        sys.exit(1)

    with open(test_set_path, "r", encoding="utf-8") as f:
        cases = json.load(f)
    if args.limit > 0:
        cases = cases[: args.limit]
    print(f"Standard RAG baseline: {len(cases)} cases, top_k={args.top_k}")

    from experiments.rq3.agent import run_chat_only_deepseek, _get_deepseek_api_key
    from experiments.rq3.tools import knowledge_retriever

    api_key = _get_deepseek_api_key()
    if not api_key:
        print("No DeepSeek API key. Set deepseek_api_key in .env")
        sys.exit(1)
    model = "deepseek-chat"

    os.makedirs(RESULTS_RQ3, exist_ok=True)
    predictions = []
    for i, c in enumerate(cases):
        case_id = c.get("case_id", f"case_{i}")
        raw_log = c.get("raw_log", "")
        dataset = c.get("dataset", "HDFS")
        query = raw_log[:500]
        try:
            refs = knowledge_retriever(query, dataset, top_k=args.top_k)
        except Exception as e:
            refs = f"(retrieve_error: {e})"
        user_msg = RAG_PROMPT.format(refs=refs, raw_log=raw_log, dataset=dataset)
        try:
            answer, usage = run_chat_only_deepseek(user_msg, api_key=api_key, model=model, max_tokens=args.max_tokens)
        except RuntimeError as e:
            if "429" in str(e):
                time.sleep(25)
                answer, usage = run_chat_only_deepseek(user_msg, api_key=api_key, model=model, max_tokens=args.max_tokens)
            else:
                raise e
        usage = usage or {}
        pred = {
            "case_id": case_id,
            "dataset": dataset,
            "ground_truth_template": c.get("ground_truth_template"),
            "ground_truth_root_cause_template": c.get("ground_truth_root_cause_template"),
            "source": c.get("source"),
            "model_answer": answer,
            "tool_calls": [{"name": "knowledge_retriever", "result": refs, "args": {"query": query[:80], "dataset": dataset}}],
            "extracted_root_cause": _extract_root_cause_from_text(answer),
            "token_usage": usage,
        }
        predictions.append(pred)
        print(f"  [{i+1}/{len(cases)}] {case_id} -> len={len(answer)}")
        if i < len(cases) - 1 and args.delay > 0:
            time.sleep(args.delay)

    out_path = args.output or os.path.join(RESULTS_RQ3, "predictions_rag.json")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)
    total_in = sum(p.get("token_usage", {}).get("prompt_tokens", 0) for p in predictions)
    total_out = sum(p.get("token_usage", {}).get("completion_tokens", 0) for p in predictions)
    print(f"Saved to {out_path}")
    if total_in or total_out:
        print(f"Token usage: input={total_in}, output={total_out}, total={total_in + total_out}")


def _extract_root_cause_from_text(text: str) -> str:
    if not text:
        return ""
    for line in text.split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "root cause" in line.lower() or "cause:" in line.lower():
            return line[:400]
    return text[:400]


if __name__ == "__main__":
    main()
