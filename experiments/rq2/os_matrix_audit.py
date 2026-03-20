"""
RQ2 OpenStack – Scientific Matrix Audit
---------------------------------------
Wrapper around `run_rq2_os_matrix_final_audit.py`.

Checks:
  - Stationarity (ADF)
  - Condition number
  - Information entropy
"""

import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

import run_rq2_os_matrix_final_audit as _audit  # type: ignore


def main():
    from run_rq2_os_matrix_final_audit import run_scientific_audit  # type: ignore

    AUDIT_FILE = "data/processed/openstack_refined_ts.csv"
    run_scientific_audit(AUDIT_FILE)


if __name__ == "__main__":
    main()

