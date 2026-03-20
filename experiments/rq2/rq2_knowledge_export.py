"""
RQ2 – Causal Knowledge Export & Refinement
------------------------------------------
Wrapper around:
  - `run_rq2_unified_knowledge_exporter.py`
  - `run_rq2_verify_and_repair_knowledge.py`

This script:
  1) Exports causal facts from HDFS & OpenStack time-series.
  2) Runs the semantic repair pass to fill Unknown templates.
"""

import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

import run_rq2_unified_knowledge_exporter as _exporter  # type: ignore
import run_rq2_verify_and_repair_knowledge as _refiner  # type: ignore


def main():
    # Step 1: export causal knowledge
    _exporter.main()
    # Step 2: refine / repair semantic gaps (if KB exists)
    if os.path.exists("data/processed/causal_knowledge.json"):
        ref = _refiner.CausalKnowledgeRefiner("data/processed/causal_knowledge.json")
        ref.validate_and_fix()


if __name__ == "__main__":
    main()

