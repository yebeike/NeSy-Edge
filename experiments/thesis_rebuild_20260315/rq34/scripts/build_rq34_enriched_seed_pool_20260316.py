from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Mapping, Sequence


_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parents[3]
_DATA_RAW = _PROJECT_ROOT / "data" / "raw"
_DATA_PROCESSED = _PROJECT_ROOT / "data" / "processed"
_RESULTS_DIR = _PROJECT_ROOT / "experiments" / "thesis_rebuild_20260315" / "rq34" / "results"

RQ3_TEST_SET_PATH = _DATA_PROCESSED / "rq3_test_set.json"
HDFS_STRUCTURED = _DATA_RAW / "HDFS" / "HDFS_2k.log_structured.csv"
OPENSTACK_STRUCTURED = _DATA_RAW / "OpenStack" / "OpenStack_2k.log_structured.csv"
OUTPUT_PATH = _RESULTS_DIR / "rq34_enriched_seed_pool_20260316.json"

HDFS_RADIUS = 6
OPENSTACK_RADIUS = 12


def _norm(text: str) -> str:
    return " ".join(str(text or "").strip().lower().split())


def _hdfs_line(row: Mapping[str, str]) -> str:
    return f"{row['Date']} {row['Time']} {row['Pid']} {row['Level']} {row['Component']}: {row['Content']}"


def _openstack_line(row: Mapping[str, str]) -> str:
    return (
        f"{row['Logrecord']} {row['Date']} {row['Time']} {row['Pid']} {row['Level']} "
        f"{row['Component']} [{row['ADDR']}] {row['Content']}"
    )


def _load_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _match_index(seed_raw: str, reconstructed: Sequence[str]) -> int:
    key = _norm(seed_raw)
    for idx, line in enumerate(reconstructed):
        if _norm(line) == key:
            return idx
    raise KeyError(f"seed line not found: {seed_raw[:120]}")


def _extract_hdfs_window(rows: Sequence[Mapping[str, str]], idx: int) -> str:
    anchor_row = dict(rows[idx])
    anchor = _hdfs_line(anchor_row)
    block_match = re.search(r"(blk_[A-Za-z0-9\\-]+)", anchor)
    block_id = block_match.group(1) if block_match else ""
    component_family = str(anchor_row.get("Component", "")).split("$", 1)[0]
    lo = max(0, idx - 30)
    hi = min(len(rows), idx + 31)

    chosen: List[str] = []
    for i in range(lo, hi):
        row = dict(rows[i])
        line = _hdfs_line(row)
        same_block = bool(block_id and block_id in line)
        same_family = str(row.get("Component", "")).split("$", 1)[0] == component_family
        if same_block or same_family:
            chosen.append(line)

    if len(chosen) < 3:
        lo2 = max(0, idx - HDFS_RADIUS)
        hi2 = min(len(rows), idx + HDFS_RADIUS + 1)
        chosen = [_hdfs_line(dict(rows[i])) for i in range(lo2, hi2)]
    return "\n".join(chosen)


def _addr_key(addr: str) -> str:
    parts = str(addr or "").split()
    return parts[0] if parts else ""


def _extract_openstack_window(rows: Sequence[Mapping[str, str]], idx: int) -> str:
    anchor = dict(rows[idx])
    anchor_logrecord = anchor.get("Logrecord", "")
    anchor_addr = _addr_key(anchor.get("ADDR", ""))
    anchor_line = _openstack_line(anchor)
    req_match = re.search(r"(req-[A-Za-z0-9\\-]+)", anchor_line)
    req_id = req_match.group(1) if req_match else ""
    lo = max(0, idx - OPENSTACK_RADIUS)
    hi = min(len(rows), idx + OPENSTACK_RADIUS + 1)

    chosen: List[str] = []
    for i in range(lo, hi):
        row = dict(rows[i])
        same_file = row.get("Logrecord", "") == anchor_logrecord
        same_addr = anchor_addr and _addr_key(row.get("ADDR", "")) == anchor_addr
        line = _openstack_line(row)
        same_req = bool(req_id and req_id in line)
        if same_req or same_addr or same_file or i == idx:
            chosen.append(line)

    if len(chosen) < 5:
        chosen = [_openstack_line(dict(rows[i])) for i in range(lo, hi)]
    return "\n".join(chosen)


def build_enriched_seed_pool() -> List[Dict[str, object]]:
    seeds = json.loads(RQ3_TEST_SET_PATH.read_text(encoding="utf-8"))
    hdfs_rows = _load_rows(HDFS_STRUCTURED)
    os_rows = _load_rows(OPENSTACK_STRUCTURED)
    hdfs_reconstructed = [_hdfs_line(r) for r in hdfs_rows]
    os_reconstructed = [_openstack_line(r) for r in os_rows]

    enriched: List[Dict[str, object]] = []
    for case in seeds:
        dataset = str(case.get("dataset", ""))
        if dataset not in {"HDFS", "OpenStack"}:
            continue
        seed_raw = str(case.get("raw_log", "") or "")
        if not seed_raw:
            continue
        if dataset == "HDFS":
            idx = _match_index(seed_raw, hdfs_reconstructed)
            raw_window = _extract_hdfs_window(hdfs_rows, idx)
        else:
            idx = _match_index(seed_raw, os_reconstructed)
            raw_window = _extract_openstack_window(os_rows, idx)

        item = dict(case)
        item["raw_log_seed"] = seed_raw
        item["raw_log"] = raw_window
        item["source"] = "rq3_test_set_enriched"
        item["enriched_anchor_index"] = idx
        item["window_line_count"] = len([x for x in raw_window.splitlines() if x.strip()])
        enriched.append(item)
    return enriched


def write_enriched_seed_pool(output_path: Path = OUTPUT_PATH) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pool = build_enriched_seed_pool()
    output_path.write_text(json.dumps(pool, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


if __name__ == "__main__":
    out = write_enriched_seed_pool()
    print(out)
