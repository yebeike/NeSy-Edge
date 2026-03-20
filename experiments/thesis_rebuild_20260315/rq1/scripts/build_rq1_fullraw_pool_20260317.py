from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd
import random

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parents[3]
_REBUILD_ROOT = _SCRIPT_DIR.parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.thesis_rebuild_20260315.rq1.components.edge_protocol_20260317 import (
    _canonicalize_for_retrieval,
    prepare_runtime_alert,
)
from experiments.thesis_rebuild_20260315.shared.case_builders.rq1_case_pool import RQ1Case
from experiments.thesis_rebuild_20260315.shared.evaluators.normalized_pa import normalize_template
from experiments.thesis_rebuild_20260315.shared.utils.project_paths import RAW_DATA_DIR, REPORT_DIR
from src.system.edge_node import NuSyEdgeNode


ARTIFACT_DIR = _REBUILD_ROOT / "rq1" / "artifacts"
DEFAULT_POOL_PATH = ARTIFACT_DIR / "rq1_fullraw_pool_20260317.json"
DEFAULT_STATE_PATH = ARTIFACT_DIR / "rq1_fullraw_pool_state_20260317.json"
DEFAULT_REPORT_PATH = REPORT_DIR / "rq1_fullraw_pool_report_20260317.md"


@dataclass
class CatalogEntry:
    event_id: str
    template: str
    normalized_template: str
    regex: str


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="HDFS,OpenStack,Hadoop")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--chunk-lines", type=int, default=200000)
    ap.add_argument("--hdfs-cap-per-template", type=int, default=2000)
    ap.add_argument("--openstack-cap-per-template", type=int, default=1200)
    ap.add_argument("--hadoop-cap-per-template", type=int, default=800)
    ap.add_argument("--max-lines-per-file", type=int, default=0)
    ap.add_argument("--pool-path", default=str(DEFAULT_POOL_PATH))
    ap.add_argument("--state-path", default=str(DEFAULT_STATE_PATH))
    ap.add_argument("--report-path", default=str(DEFAULT_REPORT_PATH))
    return ap.parse_args()


def _selected_datasets(raw: str) -> List[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def _dataset_files(dataset: str) -> List[Path]:
    if dataset == "HDFS":
        return [RAW_DATA_DIR / "HDFS_v1" / "HDFS.log"]
    if dataset == "OpenStack":
        base = RAW_DATA_DIR / "OpenStack_2"
        return [base / "openstack_abnormal.log", base / "openstack_normal1.log", base / "openstack_normal2.log"]
    if dataset == "Hadoop":
        return sorted((RAW_DATA_DIR / "Hadoop").rglob("*.log"))
    raise ValueError(f"Unsupported dataset: {dataset}")


def _catalog_path(dataset: str) -> Path:
    if dataset == "HDFS":
        return RAW_DATA_DIR / "HDFS" / "HDFS_2k.log_templates.csv"
    if dataset == "OpenStack":
        return RAW_DATA_DIR / "OpenStack" / "OpenStack_2k.log_templates.csv"
    if dataset == "Hadoop":
        return _PROJECT_ROOT / "loghub" / "Hadoop" / "Hadoop_2k.log_structured.csv"
    raise ValueError(f"Unsupported dataset: {dataset}")


def _template_to_regex(template: str) -> str:
    value = " ".join(str(template or "").strip().split())
    escaped = re.escape(value)
    escaped = escaped.replace(re.escape("<*>"), r".+?")
    escaped = escaped.replace(r"\ ", r"\s+")
    return rf"^{escaped}$"


def _load_catalog(dataset: str) -> List[CatalogEntry]:
    df = pd.read_csv(_catalog_path(dataset))
    rows: List[CatalogEntry] = []
    seen_templates: set[str] = set()
    for _, row in df.iterrows():
        template = str(row.get("EventTemplate", "") or "").strip()
        if not template:
            continue
        if dataset == "Hadoop" and template in seen_templates:
            continue
        seen_templates.add(template)
        rows.append(
            CatalogEntry(
                event_id=str(row.get("EventId", "") or ""),
                template=template,
                normalized_template=normalize_template(template),
                regex=_template_to_regex(template),
            )
        )
    return rows


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _load_or_init_state(path: Path, datasets: List[str], seed: int) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    state = {
        "protocol": "rq1_fullraw_pool_20260317",
        "seed": seed,
        "datasets": {},
    }
    for dataset in datasets:
        state["datasets"][dataset] = {
            "files": [str(p) for p in _dataset_files(dataset)],
            "file_index": 0,
            "line_no": 0,
            "completed": False,
            "total_seen": 0,
            "matched": 0,
            "ambiguous": 0,
            "unmatched": 0,
            "per_template_seen": {},
            "reservoirs": {},
        }
    return state


def _save_json(path: Path, payload: object) -> None:
    _ensure_parent(path)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _clean_line(raw_line: str, dataset: str) -> str:
    raw = raw_line.rstrip("\n")
    clean = NuSyEdgeNode.preprocess_header(raw, dataset) or raw
    return prepare_runtime_alert(clean, dataset)


def _match_catalog(clean_line: str, dataset: str, catalog: List[CatalogEntry]) -> CatalogEntry | None | str:
    if not clean_line:
        return None
    canonical = _canonicalize_for_retrieval(clean_line, dataset)
    regex_hits = [entry for entry in catalog if re.fullmatch(entry.regex, clean_line)]
    if len(regex_hits) == 1:
        return regex_hits[0]
    if len(regex_hits) > 1:
        return "ambiguous"

    norm = normalize_template(canonical)
    norm_hits = [entry for entry in catalog if entry.normalized_template == norm]
    if len(norm_hits) == 1:
        return norm_hits[0]
    if len(norm_hits) > 1:
        return "ambiguous"

    canonical_hits = [entry for entry in catalog if re.fullmatch(entry.regex, canonical)]
    if len(canonical_hits) == 1:
        return canonical_hits[0]
    if len(canonical_hits) > 1:
        return "ambiguous"
    return None


def _case_dict(dataset: str, source_file: Path, line_no: int, raw_line: str, clean_line: str, entry: CatalogEntry) -> dict:
    case = RQ1Case(
        case_id=f"{dataset.lower()}_fullraw_{source_file.stem}_{line_no}",
        dataset=dataset,
        raw_alert=raw_line.rstrip("\n"),
        clean_alert=clean_line,
        gt_template=entry.template,
        gt_source="fullraw_2k_catalog_match",
        meta={"source_file": source_file.name, "line_no": line_no, "event_id": entry.event_id},
    )
    return asdict(case)


def _reservoir_add(bucket: List[dict], seen_count: int, case: dict, cap: int, rng: random.Random) -> None:
    if len(bucket) < cap:
        bucket.append(case)
        return
    pick = rng.randrange(seen_count)
    if pick < cap:
        bucket[pick] = case


def _process_dataset_chunk(
    dataset: str,
    ds_state: dict,
    catalog: List[CatalogEntry],
    rng: random.Random,
    cap_per_template: int,
    chunk_lines: int,
    max_lines_per_file: int,
) -> int:
    processed = 0
    files = [Path(p) for p in ds_state["files"]]
    while processed < chunk_lines and not ds_state["completed"]:
        if ds_state["file_index"] >= len(files):
            ds_state["completed"] = True
            break
        path = files[ds_state["file_index"]]
        with path.open("r", encoding="latin-1", errors="ignore") as f:
            for line_no, raw_line in enumerate(f, start=1):
                if line_no <= ds_state["line_no"]:
                    continue
                if max_lines_per_file > 0 and line_no > max_lines_per_file:
                    break
                clean_line = _clean_line(raw_line, dataset)
                match = _match_catalog(clean_line, dataset, catalog)
                ds_state["line_no"] = line_no
                ds_state["total_seen"] += 1
                processed += 1
                if match == "ambiguous":
                    ds_state["ambiguous"] += 1
                elif match is None:
                    ds_state["unmatched"] += 1
                else:
                    entry = match
                    ds_state["matched"] += 1
                    per_tpl = ds_state["per_template_seen"]
                    per_tpl[entry.template] = int(per_tpl.get(entry.template, 0)) + 1
                    reservoirs = ds_state["reservoirs"]
                    bucket = reservoirs.setdefault(entry.template, [])
                    _reservoir_add(
                        bucket=bucket,
                        seen_count=per_tpl[entry.template],
                        case=_case_dict(dataset, path, line_no, raw_line, clean_line, entry),
                        cap=cap_per_template,
                        rng=rng,
                    )
                if processed >= chunk_lines:
                    break
            else:
                ds_state["file_index"] += 1
                ds_state["line_no"] = 0
                continue
            if max_lines_per_file > 0 and ds_state["line_no"] >= max_lines_per_file:
                ds_state["file_index"] += 1
                ds_state["line_no"] = 0
        if ds_state["file_index"] >= len(files):
            ds_state["completed"] = True
    return processed


def _build_pool_payload(state: dict) -> dict:
    payload = {
        "protocol": state["protocol"],
        "seed": state["seed"],
        "datasets": {},
    }
    for dataset, ds_state in state["datasets"].items():
        cases = [case for bucket in ds_state["reservoirs"].values() for case in bucket]
        payload["datasets"][dataset] = {
            "completed": ds_state["completed"],
            "files": ds_state["files"],
            "total_seen": ds_state["total_seen"],
            "matched": ds_state["matched"],
            "ambiguous": ds_state["ambiguous"],
            "unmatched": ds_state["unmatched"],
            "coverage": round(ds_state["matched"] / max(ds_state["total_seen"], 1), 6),
            "template_count": len(ds_state["per_template_seen"]),
            "per_template_seen": ds_state["per_template_seen"],
            "cases": cases,
        }
    return payload


def _write_report(path: Path, payload: dict, state_path: Path, pool_path: Path) -> None:
    lines = [
        "# RQ1 Full-Raw Pool Build",
        "",
        f"- Pool JSON: `{pool_path}`",
        f"- State JSON: `{state_path}`",
        "",
    ]
    for dataset, meta in payload["datasets"].items():
        lines.extend(
            [
                f"## {dataset}",
                "",
                f"- completed: `{meta['completed']}`",
                f"- total_seen: `{meta['total_seen']}`",
                f"- matched: `{meta['matched']}`",
                f"- ambiguous: `{meta['ambiguous']}`",
                f"- unmatched: `{meta['unmatched']}`",
                f"- coverage: `{meta['coverage']}`",
                f"- matched_templates: `{meta['template_count']}`",
                f"- pooled_cases: `{len(meta['cases'])}`",
                "",
                "| Template | Seen |",
                "|---|---:|",
            ]
        )
        top = sorted(meta["per_template_seen"].items(), key=lambda item: item[1], reverse=True)[:20]
        for template, count in top:
            lines.append(f"| {template} | {count} |")
        lines.append("")
    _ensure_parent(path)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = _parse_args()
    datasets = _selected_datasets(args.datasets)
    if not datasets:
        raise SystemExit("No datasets selected.")

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    state_path = Path(args.state_path)
    pool_path = Path(args.pool_path)
    report_path = Path(args.report_path)

    state = _load_or_init_state(state_path, datasets, args.seed)
    catalogs = {dataset: _load_catalog(dataset) for dataset in datasets}
    caps = {
        "HDFS": args.hdfs_cap_per_template,
        "OpenStack": args.openstack_cap_per_template,
        "Hadoop": args.hadoop_cap_per_template,
    }
    rng = random.Random(args.seed)

    total_processed = 0
    for dataset in datasets:
        ds_state = state["datasets"][dataset]
        if ds_state["completed"]:
            continue
        processed = _process_dataset_chunk(
            dataset=dataset,
            ds_state=ds_state,
            catalog=catalogs[dataset],
            rng=rng,
            cap_per_template=caps[dataset],
            chunk_lines=max(1, args.chunk_lines - total_processed),
            max_lines_per_file=args.max_lines_per_file,
        )
        total_processed += processed
        if total_processed >= args.chunk_lines:
            break

    _save_json(state_path, state)
    payload = _build_pool_payload(state)
    _save_json(pool_path, payload)
    _write_report(report_path, payload, state_path=state_path, pool_path=pool_path)

    compact = {
        dataset: {
            "completed": meta["completed"],
            "total_seen": meta["total_seen"],
            "matched": meta["matched"],
            "coverage": meta["coverage"],
            "pooled_cases": len(meta["cases"]),
            "template_count": meta["template_count"],
        }
        for dataset, meta in payload["datasets"].items()
    }
    print(json.dumps({"processed_this_run": total_processed, "datasets": compact}, indent=2))
    print(f"[Saved] {state_path}")
    print(f"[Saved] {pool_path}")
    print(f"[Saved] {report_path}")


if __name__ == "__main__":
    main()
