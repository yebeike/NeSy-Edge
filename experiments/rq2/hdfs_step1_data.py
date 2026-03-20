"""
RQ2 HDFS – Step 1: Oracle Parsing & Time-Series Construction
------------------------------------------------------------
Thin wrapper around the original script `run_rq2_step1_process_data.py`.

Keeps all original logic and paths intact, but provides a clean entrypoint
under `experiments/rq2/` for RQ2 experiments.
"""

import os
import sys

# Make the parent `experiments/` directory importable
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

import run_rq2_step1_process_data as _step1  # type: ignore


def main():
    _step1.main()


if __name__ == "__main__":
    main()

