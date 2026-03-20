from __future__ import annotations

import argparse
import hashlib
import os
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from rq2_fullcase_common_20260316 import REPORTS_DIR, RESULTS_DIR, ensure_dirs, write_json

import sys

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.perception.drain_parser import DrainParser
from src.system.edge_node import NuSyEdgeNode


RAW_OPENSTACK_2 = _PROJECT_ROOT / "data" / "raw" / "OpenStack_2"
DEFAULT_TS_PATH = RESULTS_DIR / "openstack_semantic_timeseries_20260316.csv"
DEFAULT_ID_MAP_PATH = RESULTS_DIR / "openstack_semantic_id_map_20260316.json"
DEFAULT_REPORT_PATH = REPORTS_DIR / "openstack_semantic_timeseries_20260316.md"

LEVEL_RE = re.compile(r" (INFO|WARNING|ERROR|DEBUG) ")
TS_RE = re.compile(r"^\S+\s+(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)")


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", default="1min", help="Pandas resample window, default 1min.")
    ap.add_argument("--out-ts", default=str(DEFAULT_TS_PATH))
    ap.add_argument("--out-id-map", default=str(DEFAULT_ID_MAP_PATH))
    ap.add_argument("--report", default=str(DEFAULT_REPORT_PATH))
    ap.add_argument("--force", action="store_true")
    return ap.parse_args()


def _template_id(template: str) -> str:
    return hashlib.md5(template.encode("utf-8")).hexdigest()[:12]


def _iter_openstack_events(parser: DrainParser) -> Tuple[List[Dict[str, object]], Dict[str, str]]:
    events: List[Dict[str, object]] = []
    id_map: Dict[str, str] = {}

    for f_name in sorted(os.listdir(RAW_OPENSTACK_2)):
        if not f_name.endswith(".log"):
            continue
        path = RAW_OPENSTACK_2 / f_name
        with open(path, "r", encoding="latin-1", errors="ignore") as f:
            for raw_line in f:
                raw_line = raw_line.rstrip("\n")
                if not raw_line.strip():
                    continue
                ts_match = TS_RE.search(raw_line)
                lvl_match = LEVEL_RE.search(raw_line)
                if not ts_match or not lvl_match:
                    continue
                clean = NuSyEdgeNode.preprocess_header(raw_line, "OpenStack") or raw_line
                try:
                    template = parser.parse(clean)
                except Exception:
                    template = clean
                template = " ".join(str(template or clean).split())
                if not template:
                    continue
                event_id = _template_id(template)
                id_map[event_id] = template
                events.append({"Timestamp": ts_match.group(1), "EventID": event_id})
    return events, id_map


def main() -> None:
    args = _parse_args()
    ensure_dirs()
    out_ts = Path(args.out_ts)
    out_id_map = Path(args.out_id_map)
    report_path = Path(args.report)

    if out_ts.exists() and out_id_map.exists() and report_path.exists() and not args.force:
        print(f"[*] Reusing OpenStack semantic timeseries: {out_ts}")
        print(f"[*] Reusing OpenStack semantic id map: {out_id_map}")
        print(f"[*] Reusing OpenStack semantic report: {report_path}")
        return

    parser = DrainParser()
    events, id_map = _iter_openstack_events(parser)
    if not events:
        raise RuntimeError("No OpenStack events were parsed from OpenStack_2 logs.")

    df = pd.DataFrame(events)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df = df.dropna(subset=["Timestamp"])
    df = df.sort_values("Timestamp")
    matrix = df.groupby("EventID").resample(args.window, on="Timestamp").size().unstack(level=0).fillna(0)
    matrix = matrix.sort_index(axis=1)
    matrix.index = matrix.index.astype(str)

    matrix.to_csv(out_ts)
    write_json(out_id_map, {k: id_map[k] for k in matrix.columns if k in id_map})

    top_templates = Counter(df["EventID"]).most_common(12)
    lines = [
        "# OpenStack Semantic Timeseries (2026-03-16)",
        "",
        f"- Raw logs: `{RAW_OPENSTACK_2}`",
        f"- Parsed events: `{len(df)}`",
        f"- Unique semantic templates: `{len(id_map)}`",
        f"- Matrix shape: `{matrix.shape[0]} x {matrix.shape[1]}`",
        f"- Window: `{args.window}`",
        "",
        "## Top templates",
        "",
        "| EventID | Count | Template |",
        "|---|---:|---|",
    ]
    for event_id, count in top_templates:
        lines.append(f"| {event_id} | {count} | {id_map.get(event_id, '')} |")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[Saved] {out_ts}")
    print(f"[Saved] {out_id_map}")
    print(f"[Saved] {report_path}")
    print(f"[Info] matrix shape={matrix.shape}, unique_templates={len(id_map)}")


if __name__ == "__main__":
    main()
