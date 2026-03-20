from __future__ import annotations

import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.thesis_rebuild_20260315.rq1.scripts import run_rq1_edge_chunked_resumable_20260317 as base_chunked


base_chunked.RUNNER = Path(__file__).resolve().parent / "run_rq1_edge_resumable_official_drain_20260318.py"


if __name__ == "__main__":
    base_chunked.main()
