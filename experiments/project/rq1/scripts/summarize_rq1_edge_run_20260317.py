from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.thesis_rebuild_20260315.rq1.components.edge_protocol_20260317 import write_markdown_report


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-tag", type=str, required=True)
    return ap.parse_args()


def main() -> str:
    args = _parse_args()
    root = Path(__file__).resolve().parent.parent
    summary_path = root / "results" / f"rq1_{args.run_tag}_summary_20260317.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    report_path = write_markdown_report(
        summary_path=summary_path,
        payload=payload,
        report_name=f"rq1_{args.run_tag}_report_20260317.md",
    )
    print(f"[Saved] {report_path}")
    return str(report_path)


if __name__ == "__main__":
    main()
