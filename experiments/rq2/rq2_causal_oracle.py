"""
RQ2 – Oracle Causal Validation Pipeline
---------------------------------------
Wrapper around `run_rq2_causal_validation.py`.

End-to-end "oracle" experiment:
  - Flash matching using ground-truth templates
  - Time-series aggregation
  - Causal injection (lagged + confounder)
  - Baseline vs DYNOTEARS comparison
"""

import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

import run_rq2_causal_validation as _oracle  # type: ignore


def main():
    _oracle.main()


if __name__ == "__main__":
    main()

