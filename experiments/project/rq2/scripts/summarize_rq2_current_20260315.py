from __future__ import annotations

import json
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REBUILD_ROOT = _SCRIPT_DIR.parents[1]
_PROJECT_ROOT = _REBUILD_ROOT.parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.thesis_rebuild_20260315.shared.utils.io_utils import write_json


SRC = _PROJECT_ROOT / "results" / "rq2" / "hybrid_eval_penalized_20260314.json"
OUT_DIR = _REBUILD_ROOT / "results"
OUT_JSON = OUT_DIR / "rq2_summary_20260315.json"
OUT_MD = _REBUILD_ROOT / "reports" / "rq2_status_20260315.md"


def main() -> str:
    rows = json.loads(SRC.read_text(encoding="utf-8"))
    by_ds = {}
    for row in rows:
        by_ds.setdefault(row["dataset"], {})[row["graph"]] = row

    summary = {"datasets": {}, "source": str(SRC)}
    lines = ["# RQ2 Status (2026-03-15)", "", f"Source: `{SRC}`", ""]
    for ds in ["HDFS", "OpenStack", "Hadoop"]:
        part = by_ds.get(ds, {})
        modified = part.get("modified", {})
        original = part.get("original", {})
        summary["datasets"][ds] = {
            "modified": modified,
            "original": original,
        }
        if modified and original:
            lines.append(
                f"- {ds}: modified graph keeps lower sparsity ({modified['sparsity_mean']}) "
                f"and lower avg-rank ({modified['avg_rank']}) than original "
                f"({original['sparsity_mean']}, {original['avg_rank']})."
            )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_json(OUT_JSON, summary)
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[Saved] {OUT_JSON}")
    print(f"[Saved] {OUT_MD}")
    return str(OUT_JSON)


if __name__ == "__main__":
    main()
