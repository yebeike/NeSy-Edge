from __future__ import annotations

import argparse
import json
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-tag", type=str, required=True)
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    root = Path(__file__).resolve().parent.parent
    summary_path = root / "results" / f"rq1_{args.run_tag}_summary_20260315.json"
    report_path = root.parent / "reports" / f"rq1_{args.run_tag}_report_20260315.md"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))

    lines = [f"# RQ1 Report: {args.run_tag}", ""]
    lines.append(f"Source summary: `{summary_path}`")
    lines.append("")
    lines.append("## Clean sanity")
    for dataset, methods in payload.get("clean_sanity", {}).items():
        lines.append(f"- {dataset}: {methods}")
    lines.append("")
    lines.append("## Degenerate flags")
    flags = payload.get("degenerate_flags", [])
    if not flags:
        lines.append("- None")
    else:
        for flag in flags:
            lines.append(f"- {flag}")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[Saved] {report_path}")


if __name__ == "__main__":
    main()
