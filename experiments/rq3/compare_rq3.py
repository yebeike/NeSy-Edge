# Run evaluate on Ours, Vanilla, RAG and write comparison JSON.
# Usage (from project root): PYTHONPATH=. python experiments/rq3/compare_rq3.py

import os
import sys
import json
import subprocess

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
RESULTS_RQ3 = os.path.join(_PROJECT_ROOT, "results", "rq3")

CONFIGS = [
    ("ours", "predictions.json", "eval_ours.json"),
    ("vanilla", "predictions_vanilla.json", "eval_vanilla.json"),
    ("rag", "predictions_rag.json", "eval_rag.json"),
]


def main():
    os.makedirs(RESULTS_RQ3, exist_ok=True)
    comparison = {}
    for name, pred_file, out_file in CONFIGS:
        pred_path = os.path.join(RESULTS_RQ3, pred_file)
        out_path = os.path.join(RESULTS_RQ3, out_file)
        if not os.path.exists(pred_path):
            print(f"Skip {name}: {pred_path} not found")
            comparison[name] = {"error": f"missing {pred_file}"}
            continue
        env = os.environ.copy()
        env["PYTHONPATH"] = _PROJECT_ROOT
        r = subprocess.run(
            [sys.executable, "experiments/rq3/evaluate.py", "--predictions", pred_path, "--output", out_path],
            cwd=_PROJECT_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if r.returncode != 0:
            comparison[name] = {"error": r.stderr or r.stdout or "eval failed"}
            print(f"Eval {name} failed: {r.stderr[:200] if r.stderr else r.stdout[:200]}")
            continue
        with open(out_path, "r", encoding="utf-8") as f:
            comparison[name] = json.load(f)
        with open(pred_path, "r", encoding="utf-8") as f:
            preds = json.load(f)
        pin = sum(p.get("token_usage", {}).get("prompt_tokens", 0) for p in preds)
        pout = sum(p.get("token_usage", {}).get("completion_tokens", 0) for p in preds)
        comparison[name]["token_usage"] = {"prompt_tokens": pin, "completion_tokens": pout, "total_tokens": pin + pout}
        cost_cny = (pin / 1e6 * 2) + (pout / 1e6 * 3) if (pin or pout) else 0
        comparison[name]["estimated_cost_cny"] = round(cost_cny, 4)
        print(f"{name}: RCA={comparison[name].get('rca_accuracy_percent')}% ({comparison[name].get('rca_correct')}/{comparison[name].get('rca_total')}), hallu={comparison[name].get('hallucination_rate_percent')}%, tokens={pin + pout} (in={pin}, out={pout})")

    out_path = os.path.join(RESULTS_RQ3, "comparison.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
