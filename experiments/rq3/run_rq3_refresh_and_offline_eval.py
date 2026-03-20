"""
Task 3 runner (non-destructive):
1) Re-export latest causal_knowledge.json from current RQ2 pipeline (overwrite data/processed/causal_knowledge.json)
2) Run RQ3 offline batch on 36 cases (causal_edge subset is typically the first 36)
3) Run RQ3 evaluate.py and print the summary JSON

This script only orchestrates existing scripts; it does not modify any existing code.
"""

import os
import sys
import subprocess


def _run(cmd, cwd):
    print("\n" + "-" * 100)
    print("RUN:", " ".join(cmd))
    print("-" * 100)
    p = subprocess.run(cmd, cwd=cwd, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed (exit={p.returncode}): {' '.join(cmd)}")


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = dict(os.environ)
    env["PYTHONPATH"] = project_root

    # 1) Export latest causal knowledge (overwrite default path)
    _run(
        [sys.executable, "experiments/run_rq2_unified_knowledge_exporter.py"],
        cwd=project_root,
    )

    # 2) Run RQ3 offline on 36 cases
    _run(
        [sys.executable, "experiments/rq3/run_batch_offline.py", "--limit", "36"],
        cwd=project_root,
    )

    # 3) Evaluate
    _run(
        [sys.executable, "experiments/rq3/evaluate.py", "--predictions", "results/rq3/predictions_offline.json"],
        cwd=project_root,
    )


if __name__ == "__main__":
    main()

