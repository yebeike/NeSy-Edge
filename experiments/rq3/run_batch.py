# Run RQ3 agent on test set and save predictions to results/rq3/
# Usage (from project root): PYTHONPATH=. python experiments/rq3/run_batch.py [--limit N] [--test-set path]

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="Max cases to run (0 = all)")
    ap.add_argument("--test-set", default="", help="Path to test set JSON (default: data/processed/rq3_test_set.json)")
    ap.add_argument("--model", default="gemini-2.0-flash", help="Model name (gemini-2.0-flash or deepseek-chat)")
    ap.add_argument("--delay", type=float, default=2.0, help="Seconds between API calls (avoid 429)")
    ap.add_argument("--backend", default="deepseek", choices=("gemini", "deepseek"), help="API backend (default: deepseek to save cost)")
    ap.add_argument("--max-tokens", type=int, default=1024, help="Max output tokens for DeepSeek (default 1024 to save cost)")
    ap.add_argument("--output", default="", help="Output JSON path (default: results/rq3/predictions.json)")
    ap.add_argument("--causal", default="", help="Path to causal_knowledge JSON (default: data/processed/causal_knowledge.json). For Data Efficiency use e.g. causal_knowledge_25.json")
    args = ap.parse_args()

    if args.causal and args.causal.strip():
        os.environ["CAUSAL_KNOWLEDGE_PATH"] = os.path.abspath(args.causal.strip())
        print(f"Using causal graph: {os.environ['CAUSAL_KNOWLEDGE_PATH']}")

    test_set_path = args.test_set or os.path.join(DATA_PROCESSED, "rq3_test_set.json")
    if not os.path.exists(test_set_path):
        print(f"Test set not found: {test_set_path}")
        sys.exit(1)

    with open(test_set_path, "r", encoding="utf-8") as f:
        cases = json.load(f)
    if args.limit > 0:
        cases = cases[: args.limit]
    print(f"Running agent on {len(cases)} cases (test set: {test_set_path})")

    from experiments.rq3.agent import run_agent_with_instruction, _get_gemini_api_keys, _get_deepseek_api_key
    if args.backend == "deepseek":
        api_key = _get_deepseek_api_key()
        if not api_key:
            print("No DeepSeek API key. Set deepseek_api_key in .env")
            sys.exit(1)
        model = "deepseek-chat" if "gemini" in (args.model or "").lower() else (args.model or "deepseek-chat")
    else:
        keys = _get_gemini_api_keys()
        api_key = keys[0] if keys else None
        if not api_key:
            print("No Gemini API key. Set GEMINI_API_KEY or add to .env")
            sys.exit(1)
        model = args.model or "gemini-2.0-flash"
    print(f"Backend: {args.backend}, model: {model}, max_tokens: {args.max_tokens}")

    os.makedirs(RESULTS_RQ3, exist_ok=True)
    predictions = []
    for i, c in enumerate(cases):
        case_id = c.get("case_id", f"case_{i}")
        raw_log = c.get("raw_log", "")
        dataset = c.get("dataset", "HDFS")
        # Short prompt to save input tokens (DeepSeek 2 CNY/M input)
        user_msg = f"Log diagnosis: root cause + one action.\n\nLog: {raw_log}\nDataset: {dataset}"
        try:
            answer, tool_log, usage = run_agent_with_instruction(
                user_msg,
                api_key=api_key,
                model=model,
                backend=args.backend,
                max_tokens=args.max_tokens,
            )
        except RuntimeError as e429:
            if "429" in str(e429):
                time.sleep(25)
                try:
                    answer, tool_log, usage = run_agent_with_instruction(user_msg, api_key=api_key, model=model, backend=args.backend, max_tokens=args.max_tokens)
                except Exception as e2:
                    raise e2
            else:
                raise e429
        usage = usage or {}
        try:
            pred = {
                "case_id": case_id,
                "dataset": dataset,
                "ground_truth_template": c.get("ground_truth_template"),
                "ground_truth_root_cause_template": c.get("ground_truth_root_cause_template"),
                "source": c.get("source"),
                "model_answer": answer,
                "tool_calls": [{"name": t["name"], "result": t["result"], "args": t.get("args", {})} for t in tool_log],
                "extracted_root_cause": _extract_root_cause_from_answer(answer, tool_log),
                "token_usage": usage,
            }
            predictions.append(pred)
            print(f"  [{i+1}/{len(cases)}] {case_id} -> answer len={len(answer)}")
        except Exception as e:
            predictions.append({
                "case_id": case_id,
                "dataset": dataset,
                "ground_truth_template": c.get("ground_truth_template"),
                "ground_truth_root_cause_template": c.get("ground_truth_root_cause_template"),
                "source": c.get("source"),
                "model_answer": f"(error: {e})",
                "tool_calls": [],
                "extracted_root_cause": "",
                "token_usage": {},
            })
            print(f"  [{i+1}/{len(cases)}] {case_id} ERROR: {e}")
        if i < len(cases) - 1 and args.delay > 0:
            time.sleep(args.delay)
    out_path = args.output or os.path.join(RESULTS_RQ3, "predictions.json")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)
    total_in = sum(p.get("token_usage", {}).get("prompt_tokens", 0) for p in predictions)
    total_out = sum(p.get("token_usage", {}).get("completion_tokens", 0) for p in predictions)
    print(f"Saved {len(predictions)} predictions to {out_path}")
    if total_in or total_out:
        print(f"Token usage: input={total_in}, output={total_out}, total={total_in + total_out}")


def _extract_root_cause_from_answer(answer: str, tool_log: list) -> str:
    """Heuristic: if causal_navigator was called, last result may be root cause list; else search answer text."""
    for t in reversed(tool_log):
        if t["name"] == "causal_navigator" and t.get("result"):
            try:
                data = json.loads(t["result"])
                if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
                    return data[0].get("source_template", "")
                if isinstance(data, dict) and "error" not in data:
                    return str(data.get("source_template", data))[:200]
            except json.JSONDecodeError:
                pass
    # Fallback: look for "root cause" or "source_template" in answer
    lower = answer.lower()
    for line in answer.split("\n"):
        if "root cause" in line.lower() or "source" in line.lower():
            return line.strip()[:300]
    return ""


if __name__ == "__main__":
    main()
