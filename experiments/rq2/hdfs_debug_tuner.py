"""
RQ2 HDFS – Debug & Hyperparameter Tuner
--------------------------------------
Wrapper around `run_rq2_debug_comprehensive_check.py`.

Used to:
  - Check data health
  - Validate DYNOTEARS kernel on toy data
  - Run grid search for (lambda_w, lambda_a, threshold)
"""

import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

import run_rq2_debug_comprehensive_check as _dbg  # type: ignore


def main():
    _dbg.check_data_health()
    print("\n")
    if _dbg.check_algorithm_kernel():
        print("\n")
        import pandas as pd
        from _dbg import DATA_PATH  # type: ignore[attr-defined]

        df = pd.read_csv(DATA_PATH, index_col=0)
        _dbg.run_grid_search(df)


if __name__ == "__main__":
    main()

