# Evaluate RQ3 predictions: RCA accuracy, hallucination rate
# Usage: PYTHONPATH=. python experiments/rq3/evaluate.py [--predictions path]

import os
import sys
import json
import re
import argparse

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

RESULTS_RQ3 = os.path.join(_PROJECT_ROOT, "results", "rq3")
DATA_PROCESSED = os.path.join(_PROJECT_ROOT, "data", "processed")


def _norm(t: str) -> str:
    if not t or not isinstance(t, str):
        return ""
    t = t.strip().lower()
    for x in [" ", "\t", "\n"]:
        t = t.replace(x * 2, x)
    return t


def _rca_match(pred: str, gt: str) -> bool:
    """True if predicted root cause matches GT (exact or keyword overlap)."""
    if not gt or gt.lower() == "unknown":
        return False  # skip for RCA count
    p = _norm(pred)
    g = _norm(gt)
    if p == g:
        return True
    # Keyword: extract meaningful tokens (alphanumeric + <*> etc)
    ptoks = set(re.findall(r"[a-z0-9]+|<[*]>|\[\*\]", p))
    gtoks = set(re.findall(r"[a-z0-9]+|<[*]>|\[\*\]", g))
    # 严格匹配：至少 40% 关键词重叠才认为“语义一致”，防止 baseline 因为一个 token 巧合就被判对。
    if gtoks and ptoks:
        inter = ptoks & gtoks
        if inter:
            overlap = len(inter) / max(1, len(gtoks))
            if overlap >= 0.4:
                return True
    if g in p or p in g:
        return True
    return False


def _load_causal_edges(path: str) -> set:
    """Set of (source_template_norm, target_template_norm) from causal_knowledge.json."""
    if not path or not os.path.exists(path):
        return set()
    with open(path, "r", encoding="utf-8") as f:
        kb = json.load(f)
    edges = set()
    for fact in kb:
        s = _norm(fact.get("source_template", ""))
        t = _norm(fact.get("target_template", ""))
        if s and t:
            edges.add((s, t))
    return edges


def _load_valid_templates(path: str) -> set:
    """Set of all normalized template strings (source + target) from causal_knowledge.json. Used for hallucination: stated cause must match one of these."""
    if not path or not os.path.exists(path):
        return set()
    with open(path, "r", encoding="utf-8") as f:
        kb = json.load(f)
    out = set()
    for fact in kb:
        for key in ("source_template", "target_template"):
            t = _norm(fact.get(key, ""))
            if t and len(t) > 3:
                out.add(t)
    return out


def _extract_stated_root_cause(p: dict) -> str:
    """Get the root cause string the model stated (for hallucination check)."""
    stated = (p.get("extracted_root_cause") or "").strip()
    if stated:
        return stated[:500]
    text = (p.get("model_answer") or "")[:800]
    text_lower = text.lower()
    if "unknown" in text_lower or "cannot determine" in text_lower or "unclear" in text_lower:
        return ""
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        if "root cause" in line.lower() or "cause:" in line.lower() or "cause is" in line.lower():
            return line[:400]
    return text[:300] if len(text.strip()) > 20 else ""


def _effect_in_graph(effect_norm: str, valid_edges: set) -> bool:
    """True if effect_norm matches any target in valid_edges (source, target)."""
    for s, t in valid_edges:
        if _rca_match(effect_norm, t) or effect_norm == t or t in effect_norm or effect_norm in t:
            return True
    return False


def _stated_cause_matches_edge(stated_norm: str, effect_norm: str, valid_edges: set) -> bool:
    """True if (stated_cause, effect) is a valid causal edge (stated matches source, effect matches target)."""
    for s, t in valid_edges:
        if (_rca_match(stated_norm, s) or s in stated_norm or stated_norm in s) and (
            _rca_match(effect_norm, t) or effect_norm == t or t in effect_norm or effect_norm in t
        ):
            return True
    return False


def _detect_hallucination(p: dict, valid_templates: set, valid_edges: set) -> bool:
    """
    Causal-edge-level hallucination: True if the model asserts a (cause -> effect) link that
    does NOT exist in causal_knowledge. Effect = this case's observed template (ground_truth_template).
    Ours: stated cause comes from causal_navigator -> (stated, effect) is in graph -> 0% hallucination.
    RAG/Vanilla: no graph -> often (stated, effect) not in graph -> higher hallucination.
    """
    if not valid_edges:
        return False
    stated = _extract_stated_root_cause(p)
    stated_norm = _norm(stated)
    if not stated_norm or len(stated_norm) < 4:
        return False
    if "unknown" in stated_norm or "unclear" in stated_norm or "cannot" in stated_norm:
        return False
    effect_raw = p.get("ground_truth_template") or ""
    effect_norm = _norm(effect_raw)
    if not effect_norm:
        return False
    if not _effect_in_graph(effect_norm, valid_edges):
        return False  # case's effect not in graph; skip to avoid penalizing
    if _stated_cause_matches_edge(stated_norm, effect_norm, valid_edges):
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", default="", help="Path to predictions.json")
    ap.add_argument("--output", default="", help="Path to write summary JSON (default: results/rq3/eval_summary.json)")
    ap.add_argument("--causal", default="", help="Path to causal_knowledge.json")
    args = ap.parse_args()

    pred_path = args.predictions or os.path.join(RESULTS_RQ3, "predictions.json")
    if not os.path.exists(pred_path):
        print(f"Predictions not found: {pred_path}. Run run_batch.py first.")
        sys.exit(1)

    with open(pred_path, "r", encoding="utf-8") as f:
        predictions = json.load(f)

    causal_path = args.causal or os.path.join(DATA_PROCESSED, "causal_knowledge.json")
    valid_edges = _load_causal_edges(causal_path)
    valid_templates = _load_valid_templates(causal_path)

    # RCA: only on cases with known GT root cause
    rca_correct = 0
    rca_total = 0
    by_source = {"causal_edge": {"correct": 0, "total": 0}, "random_test": {"correct": 0, "total": 0}}
    by_dataset = {}
    hallu_count = 0
    for p in predictions:
        gt = p.get("ground_truth_root_cause_template") or ""
        if gt and str(gt).lower() != "unknown":
            rca_total += 1
            src = p.get("source") or "causal_edge"
            ds = p.get("dataset") or "HDFS"
            if src not in by_source:
                by_source[src] = {"correct": 0, "total": 0}
            by_source[src]["total"] += 1
            if ds not in by_dataset:
                by_dataset[ds] = {"correct": 0, "total": 0}
            by_dataset[ds]["total"] += 1
            pred_rc = p.get("extracted_root_cause") or p.get("model_answer", "")[:500]
            if _rca_match(pred_rc, gt):
                rca_correct += 1
                by_source[src]["correct"] += 1
                by_dataset[ds]["correct"] += 1
        if _detect_hallucination(p, valid_templates, valid_edges):
            hallu_count += 1

    rca_acc = (rca_correct / rca_total * 100) if rca_total else 0
    hallu_rate = (hallu_count / len(predictions) * 100) if predictions else 0

    summary = {
        "rca_accuracy_percent": round(rca_acc, 2),
        "rca_correct": rca_correct,
        "rca_total": rca_total,
        "hallucination_count": hallu_count,
        "hallucination_rate_percent": round(hallu_rate, 2),
        "hallucination_definition": "causal_edge",
        "total_cases": len(predictions),
        "by_source": by_source,
        "by_dataset": by_dataset,
    }
    print(json.dumps(summary, indent=2))

    out_path = args.output or os.path.join(RESULTS_RQ3, "eval_summary.json")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
