"""
RQ2 OpenStack – Cascade Path Recovery
-------------------------------------
Wrapper around `run_rq2_os_path_reconstruction.py`.

Main metric: Full Path Recovery Rate (FPRR) and hop-wise ranks.
"""

import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

import run_rq2_os_path_reconstruction as _cascade  # type: ignore


def main():
    _cascade.main()


if __name__ == "__main__":
    main()

