from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent.parent / "results"
    summary_path = root / "rq1_pilot_summary_20260315.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary not found: {summary_path}")
    payload = json.loads(summary_path.read_text(encoding="utf-8"))

    print("\n## Clean sanity")
    for dataset, methods in payload.get("clean_sanity", {}).items():
        print(dataset, methods)

    print("\n## Degenerate baseline flags")
    flags = payload.get("degenerate_flags", [])
    if not flags:
        print("None")
    else:
        for flag in flags:
            print(flag)


if __name__ == "__main__":
    main()
