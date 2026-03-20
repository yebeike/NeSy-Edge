from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REBUILD_ROOT = _SCRIPT_DIR.parents[1]
_PROJECT_ROOT = _REBUILD_ROOT.parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.thesis_rebuild_20260315.shared.utils.io_utils import write_json


OUT_DIR = _REBUILD_ROOT / "results"
RQ34_RESULTS_DIR = _REBUILD_ROOT / "rq34" / "results"


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=str, required=True)
    ap.add_argument("--run-tag", type=str, required=True)
    ap.add_argument("--cases-per-dataset", type=int, default=10)
    ap.add_argument("--noise-levels", type=str, default="0.0,0.2,0.4,0.6,0.8,1.0")
    return ap.parse_args()


def main() -> str:
    args = _parse_args()
    src = Path(args.source).resolve()
    rows = json.loads(src.read_text(encoding="utf-8"))
    summary = {
        "source": str(src),
        "run_tag": args.run_tag,
        "cases_per_dataset": args.cases_per_dataset,
        "noise_levels": [float(x.strip()) for x in args.noise_levels.split(",") if x.strip()],
        "datasets": {},
    }
    lines = [
        f"# RQ34 Action-Aware Status ({args.run_tag})",
        "",
        f"Source: `{src}`",
        f"- Cases per dataset: `{args.cases_per_dataset}`",
        f"- Noise levels: `{args.noise_levels}`",
        "",
    ]

    datasets = sorted({str(r["dataset"]) for r in rows})
    for ds in datasets:
        ds_rows = [r for r in rows if r["dataset"] == ds]
        if not ds_rows:
            continue
        summary["datasets"][ds] = {}
        lines.append(f"## {ds}")
        for method in ["agent", "rag", "vanilla"]:
            part = [r for r in ds_rows if r["method"] == method]
            if not part:
                continue
            avg_rca = round(sum(float(r["rca_accuracy"]) for r in part) / len(part), 4)
            avg_action = round(sum(float(r["action_accuracy"]) for r in part) / len(part), 4)
            avg_text = round(sum(float(r["action_text_success_rate"]) for r in part) / len(part), 4)
            avg_e2e = round(sum(float(r["e2e_success_rate"]) for r in part) / len(part), 4)
            summary["datasets"][ds][method] = {
                "avg_rca_accuracy": avg_rca,
                "avg_action_accuracy": avg_action,
                "avg_action_text_success_rate": avg_text,
                "avg_e2e_success_rate": avg_e2e,
                "points": len(part),
            }
            lines.append(
                f"- {method}: RCA={avg_rca:.4f}, action-id={avg_action:.4f}, "
                f"action-text={avg_text:.4f}, E2E={avg_e2e:.4f}."
            )
        lines.append("")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RQ34_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_json = OUT_DIR / "rq34_actionaware_summary_20260316.json"
    out_tagged_json = RQ34_RESULTS_DIR / f"rq34_{args.run_tag}_summary_20260316.json"
    out_md = _REBUILD_ROOT / "reports" / "rq34_actionaware_status_20260316.md"
    out_tagged_md = _REBUILD_ROOT / "reports" / f"rq34_{args.run_tag}_status_20260316.md"
    write_json(out_json, summary)
    write_json(out_tagged_json, summary)
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    out_tagged_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[Saved] {out_json}")
    print(f"[Saved] {out_tagged_json}")
    print(f"[Saved] {out_md}")
    print(f"[Saved] {out_tagged_md}")
    return str(out_json)


if __name__ == "__main__":
    main()
