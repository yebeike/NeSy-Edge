# Data Efficiency: 25% / 50% / 75% / 100% training data -> causal graphs -> same 66-case test -> curve.
# Step 1: Export 4 causal graphs (no API). Step 2: Run RQ3 agent with each graph (API). Step 3: Evaluate each; write curve JSON.
# Usage (from project root):
#   PYTHONPATH=. python experiments/rq3/run_data_efficiency.py --export-only   # only generate causal_knowledge_25/50/75/100.json
#   PYTHONPATH=. python experiments/rq3/run_data_efficiency.py                  # export + run + evaluate + curve
#   PYTHONPATH=. python experiments/rq3/run_data_efficiency.py --run-only       # assume graphs exist; run RQ3 + eval + curve

import os
import sys
import json
import subprocess

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
DATA_PROCESSED = os.path.join(_PROJECT_ROOT, "data", "processed")
RESULTS_RQ3 = os.path.join(_PROJECT_ROOT, "results", "rq3")

FRACTIONS = [0.25, 0.5, 0.75, 1.0]


def _python_exe():
    """Prefer project venv so log_parser (edge_node -> transformers) is available."""
    for rel in ("venv/bin/python", ".venv/bin/python"):
        exe = os.path.join(_PROJECT_ROOT, rel)
        if os.path.isfile(exe):
            return exe
    return sys.executable


def _run(cmd, cwd=None, env=None):
    env = env or os.environ.copy()
    env.setdefault("PYTHONPATH", _PROJECT_ROOT)
    # Use venv python for run_batch so log_parser (edge_node -> transformers) works
    if cmd and "run_batch.py" in str(cmd):
        cmd = [_python_exe()] + list(cmd)[1:]
    r = subprocess.run(cmd, cwd=cwd or _PROJECT_ROOT, env=env)
    if r.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}")
    return r


def export_graphs():
    """Write causal_knowledge_25.json, _50, _75, _100 in data/processed (run from project root)."""
    for frac in FRACTIONS:
        out = os.path.join(DATA_PROCESSED, f"causal_knowledge_{int(frac*100)}.json")
        print(f"[Export] train_fraction={frac} -> {out}")
        _run([
            sys.executable, "experiments/run_rq2_unified_knowledge_exporter.py",
            "--train-fraction", str(frac), "--output", out
        ])


def run_rq3_for_fraction(frac: float, limit: int = 0, delay: float = 1.0):
    """Run RQ3 batch with causal_knowledge_{frac*100}.json; write predictions_eff_{frac*100}.json."""
    causal = os.path.join(DATA_PROCESSED, f"causal_knowledge_{int(frac*100)}.json")
    if not os.path.isfile(causal):
        raise FileNotFoundError(f"Run --export-only first: {causal}")
    out_pred = os.path.join(RESULTS_RQ3, f"predictions_eff_{int(frac*100)}.json")
    cmd = [
        sys.executable, "experiments/rq3/run_batch.py",
        "--backend", "deepseek", "--limit", str(limit), "--delay", str(delay),
        "--causal", causal, "--output", out_pred
    ]
    print(f"[RQ3] fraction={frac} -> {out_pred}")
    _run(cmd)


def evaluate_fraction(frac: float):
    """Evaluate predictions_eff_{frac*100}.json with same causal graph; return (rca_percent, hallu_percent)."""
    causal = os.path.join(DATA_PROCESSED, f"causal_knowledge_{int(frac*100)}.json")
    pred = os.path.join(RESULTS_RQ3, f"predictions_eff_{int(frac*100)}.json")
    out_eval = os.path.join(RESULTS_RQ3, f"eval_eff_{int(frac*100)}.json")
    if not os.path.isfile(pred):
        return None, None
    _run([
        sys.executable, "experiments/rq3/evaluate.py",
        "--predictions", pred, "--causal", causal, "--output", out_eval
    ])
    with open(out_eval, "r", encoding="utf-8") as f:
        s = json.load(f)
    return s.get("rca_accuracy_percent"), s.get("hallucination_rate_percent")


def build_curve():
    """Collect eval results into data_efficiency_curve.json."""
    curve = []
    for frac in FRACTIONS:
        eval_path = os.path.join(RESULTS_RQ3, f"eval_eff_{int(frac*100)}.json")
        if not os.path.isfile(eval_path):
            curve.append({"train_fraction": frac, "rca_percent": None, "hallucination_percent": None})
            continue
        with open(eval_path, "r", encoding="utf-8") as f:
            s = json.load(f)
        curve.append({
            "train_fraction": frac,
            "rca_percent": s.get("rca_accuracy_percent"),
            "hallucination_percent": s.get("hallucination_rate_percent"),
            "rca_correct": s.get("rca_correct"),
            "rca_total": s.get("rca_total"),
        })
    out = os.path.join(RESULTS_RQ3, "data_efficiency_curve.json")
    os.makedirs(RESULTS_RQ3, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(curve, f, indent=2)
    print(f"Curve saved to {out}")
    return curve


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--export-only", action="store_true", help="Only export 4 causal graphs; do not run RQ3")
    ap.add_argument("--run-only", action="store_true", help="Skip export; run RQ3 + evaluate + curve (graphs must exist)")
    ap.add_argument("--build-curve-only", action="store_true", help="Only build data_efficiency_curve.json from existing eval_eff_*.json")
    ap.add_argument("--limit", type=int, default=0, help="Max test cases (0 = all 66)")
    ap.add_argument("--delay", type=float, default=1.0, help="Delay between API calls")
    args = ap.parse_args()

    if args.build_curve_only:
        build_curve()
        return
    if not args.run_only:
        export_graphs()
    if args.export_only:
        print("Export done. Run without --export-only to run RQ3 and build curve.")
        return

    for frac in FRACTIONS:
        run_rq3_for_fraction(frac, limit=args.limit, delay=args.delay)
    for frac in FRACTIONS:
        evaluate_fraction(frac)
    build_curve()


if __name__ == "__main__":
    main()
