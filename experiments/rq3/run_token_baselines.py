# One-shot: run Vanilla and RAG on full 66 cases (with token recording), then compare.
# Use this to fill comparison.json with precise token_usage and estimated_cost_cny.
# Usage (from project root): PYTHONPATH=. python experiments/rq3/run_token_baselines.py [--delay 1] [--limit 0]
# --limit 0 = all 66; use --limit 5 to test quickly.

import os
import sys
import subprocess

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="Max cases (0 = all 66)")
    ap.add_argument("--delay", type=float, default=1.0, help="Seconds between API calls")
    ap.add_argument("--skip-vanilla", action="store_true", help="Skip Vanilla run")
    ap.add_argument("--skip-rag", action="store_true", help="Skip RAG run")
    args = ap.parse_args()

    env = os.environ.copy()
    env["PYTHONPATH"] = _PROJECT_ROOT
    cwd = _PROJECT_ROOT

    if not args.skip_vanilla:
        print("[1/3] Running Vanilla LLM baseline (66 cases, token recorded)...")
        r = subprocess.run(
            [sys.executable, "experiments/rq3/run_batch_vanilla.py", "--limit", str(args.limit), "--delay", str(args.delay)],
            cwd=cwd, env=env
        )
        if r.returncode != 0:
            print("Vanilla run failed.")
            sys.exit(r.returncode)

    if not args.skip_rag:
        print("[2/3] Running Standard RAG baseline (66 cases, token recorded)...")
        r = subprocess.run(
            [sys.executable, "experiments/rq3/run_batch_rag.py", "--limit", str(args.limit), "--delay", str(args.delay)],
            cwd=cwd, env=env
        )
        if r.returncode != 0:
            print("RAG run failed.")
            sys.exit(r.returncode)

    print("[3/3] Running compare_rq3.py -> comparison.json")
    r = subprocess.run(
        [sys.executable, "experiments/rq3/compare_rq3.py"],
        cwd=cwd, env=env
    )
    if r.returncode != 0:
        print("Compare failed.")
        sys.exit(r.returncode)
    print("Done. results/rq3/comparison.json now has token_usage and estimated_cost_cny for all three methods.")


if __name__ == "__main__":
    main()
