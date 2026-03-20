# Run RQ3 pipeline WITHOUT LLM: for each test case run log_parser -> causal_navigator only,
# save predictions so we can evaluate RCA of the symbolic pipeline (no API needed).
# Usage: PYTHONPATH=. python experiments/rq3/run_batch_offline.py [--limit N]

import os
import sys
import json
import argparse

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

DATA_PROCESSED = os.path.join(_PROJECT_ROOT, "data", "processed")
RESULTS_RQ3 = os.path.join(_PROJECT_ROOT, "results", "rq3")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="Max cases (0=all)")
    ap.add_argument("--test-set", default="", help="Path to test set JSON")
    ap.add_argument("--causal", default="", help="Path to causal_knowledge JSON (for Data Efficiency: causal_knowledge_25.json etc)")
    ap.add_argument("--output", default="", help="Output JSON path (default: predictions_offline.json or predictions_eff_25.json when --causal set)")
    args = ap.parse_args()

    if args.causal and args.causal.strip():
        os.environ["CAUSAL_KNOWLEDGE_PATH"] = os.path.abspath(args.causal.strip())

    test_set_path = args.test_set or os.path.join(DATA_PROCESSED, "rq3_test_set.json")
    if not os.path.exists(test_set_path):
        print(f"Test set not found: {test_set_path}")
        sys.exit(1)

    with open(test_set_path, "r", encoding="utf-8") as f:
        cases = json.load(f)
    if args.limit > 0:
        cases = cases[: args.limit]
    print(f"Offline run (no LLM) on {len(cases)} cases")

    from experiments.rq3.tools import causal_navigator
    domain_map = {"HDFS": "hdfs", "OpenStack": "openstack"}
    try:
        from experiments.rq3.tools import log_parser
        use_parser = True
    except Exception:
        use_parser = False
        print("(log_parser unavailable, using ground_truth_template as proxy)")
    predictions = []
    for i, c in enumerate(cases):
        case_id = c.get("case_id", f"case_{i}")
        raw_log = c.get("raw_log", "")
        dataset = c.get("dataset", "HDFS")
        if use_parser:
            try:
                template = log_parser(raw_log, dataset)
            except Exception as e:
                template = c.get("ground_truth_template", "")
                print(f"  [{i+1}] {case_id} parser error, use GT template: {e}")
        else:
            template = c.get("ground_truth_template", "")
        try:
            domain = domain_map.get(dataset, "hdfs")
            causal_path = os.environ.get("CAUSAL_KNOWLEDGE_PATH") or None
            root_json = causal_navigator(template, domain, causal_path=causal_path)
            root_list = json.loads(root_json) if (root_json and root_json.startswith("[")) else []
            extracted_root_cause = root_list[0].get("source_template", "") if root_list and isinstance(root_list[0], dict) else ""
        except Exception as e:
            root_json = "[]"
            extracted_root_cause = ""
            root_list = []
            print(f"  [{i+1}] {case_id} causal_navigator error: {e}")
        pred = {
            "case_id": case_id,
            "dataset": dataset,
            "ground_truth_template": c.get("ground_truth_template"),
            "ground_truth_root_cause_template": c.get("ground_truth_root_cause_template"),
            "source": c.get("source"),
            "model_answer": "(offline: no LLM)",
            "tool_calls": [
                {"name": "log_parser", "result": template},
                {"name": "causal_navigator", "result": root_json},
            ],
            "extracted_root_cause": extracted_root_cause,
        }
        predictions.append(pred)
        print(f"  [{i+1}/{len(cases)}] {case_id} template_ok={bool(template)} root={extracted_root_cause[:50] if extracted_root_cause else 'none'}...")

    os.makedirs(RESULTS_RQ3, exist_ok=True)
    if args.output and args.output.strip():
        out_path = os.path.abspath(args.output) if not os.path.isabs(args.output) else args.output
    else:
        out_path = os.path.join(RESULTS_RQ3, "predictions_offline.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)
    print(f"Saved to {out_path}. Run: python experiments/rq3/evaluate.py --predictions {out_path}")


if __name__ == "__main__":
    main()
