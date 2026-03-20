"""
RQ2 HDFS – Failure Analysis Scanner
-----------------------------------
Wrapper around `debug_rq2_failure_analysis.py`.

This script is non-essential for main thesis results, but useful for
explaining and visualising different failure modes of DYNOTEARS.
"""

import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

import debug_rq2_failure_analysis as _fail  # type: ignore


def main():
    _fail.main()


if __name__ == "__main__":
    main()

