"""
RQ2 OpenStack – Preprocess & Refine Metrics Matrix
--------------------------------------------------
Wrapper around `run_rq2_os_preprocessor.py`.

Generates:
  - `openstack_refined_ts.csv`
  - `openstack_id_map.json`
"""

import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

import run_rq2_os_preprocessor as _osp  # type: ignore


def main():
    _osp.main()


if __name__ == "__main__":
    main()

