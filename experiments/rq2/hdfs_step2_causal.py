"""
RQ2 HDFS – Step 2: Causal Analysis (Oracle Time-Series)
-------------------------------------------------------
Thin wrapper around `run_rq2_step2_causal_analysis.py`.

This script is the main entrypoint for:
  - Single-edge fault localization (lagged & intra)
  - Confounder resistance case study
"""

import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

import run_rq2_step2_causal_analysis as _step2  # type: ignore


def main():
    _step2.main()


if __name__ == "__main__":
    main()

