import os
import sys


"""
Temporary 1-run wrapper for RQ1 large-scale robustness.

Constraints (per user request):
- Do NOT modify existing scripts under src/ or experiments/.
- Reuse the existing large-scale pipeline logic but force:
  - RUN_SEEDS = [2026]  (single Monte Carlo run)
  - Keep BASE_POOL_SIZE=20_000, TRAIN_SIZE=10_000, TEST_POOL_SIZE=10_000.
  - From TEST_POOL (10_000), sample 2_000 test logs per run.
  - Vanilla LLM only evaluated on 200 logs per run.

Implementation:
- Import experiments.rq1_robustness.run_rq1_large_scale as a module.
- Override its RUN_SEEDS at runtime to [2026] without touching the original file.
- Optionally, point EXPORT_DIR to a smoke-test-specific folder to avoid clutter.
"""


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # Import the existing large-scale RQ1 runner as a module
    from experiments.rq1_robustness import run_rq1_large_scale as rq1_ls

    # Force single-run configuration (seed=2026) without editing the original file
    rq1_ls.RUN_SEEDS = [2026]

    # Optionally redirect outputs to a dedicated smoke-test folder so that
    # we can easily distinguish them from the canonical 5-run results.
    try:
        project_root_in_module = rq1_ls._PROJECT_ROOT  # type: ignore[attr-defined]
    except Exception:
        project_root_in_module = project_root
    smoke_dir = os.path.join(project_root_in_module, "results", "rq1_smoke_1run")
    rq1_ls.EXPORT_DIR = smoke_dir  # type: ignore[attr-defined]

    os.makedirs(smoke_dir, exist_ok=True)

    print("=" * 120)
    print("RQ1 LARGE-SCALE SMOKE TEST (1-run)")
    print("Config:")
    print("  - RUN_SEEDS = [2026]")
    print("  - BASE_POOL_SIZE = 20,000 logs per dataset (from data/raw)")
    print("  - TRAIN_SIZE = 10,000 (first half -> RAG KB / cache warm-up)")
    print("  - TEST_POOL_SIZE = 10,000 (second half)")
    print("  - TEST_SAMPLE_PER_RUN = 2,000 (per run sample from TEST_POOL)")
    print("  - VANILLA_SAMPLE_PER_RUN = 200 (subset of the 2,000)")
    print("=" * 120)

    rq1_ls.run_rq1_large_scale()


if __name__ == "__main__":
    main()

