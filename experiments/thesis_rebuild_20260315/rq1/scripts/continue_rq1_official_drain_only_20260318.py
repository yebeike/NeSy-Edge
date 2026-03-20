from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[4]
REBUILD_ROOT = PROJECT_ROOT / "experiments" / "thesis_rebuild_20260315"
RQ1_RESULTS_DIR = REBUILD_ROOT / "rq1" / "results"
REPORT_DIR = REBUILD_ROOT / "reports"
RUNNER = REBUILD_ROOT / "rq1" / "scripts" / "run_rq1_edge_chunked_resumable_official_drain_only_20260318.py"
MERGER = REBUILD_ROOT / "rq1" / "scripts" / "merge_rq1_official_drain_composite_20260318.py"
AUDITOR = REBUILD_ROOT / "rq1" / "scripts" / "audit_rq1_artifacts_20260317.py"
STATUS_MD = REPORT_DIR / "rq1_official_drain_only_pipeline_20260318.md"


STAGES = [
    {
        "name": "small",
        "manifest": "rq1_manifest_edge_small_fullraw_strict_v3_20260317.json",
        "canonical_rows": RQ1_RESULTS_DIR / "rq1_edge_small_fullraw_strict_v3_direct_rows_20260317.csv",
        "drain_tag": "edge_small_fullraw_strict_v6_official_drain_only",
        "composite_tag": "edge_small_fullraw_strict_v6_official_drain_only_composite",
        "cases_per_chunk_spec": "",
    },
    {
        "name": "mid",
        "manifest": "rq1_manifest_edge_mid_fullraw_strict_v3_20260317.json",
        "canonical_rows": RQ1_RESULTS_DIR / "rq1_edge_mid_fullraw_strict_v3_chunked_direct_rows_20260317.csv",
        "drain_tag": "edge_mid_fullraw_strict_v6_official_drain_only",
        "composite_tag": "edge_mid_fullraw_strict_v6_official_drain_only_composite",
        "cases_per_chunk_spec": "",
    },
    {
        "name": "full",
        "manifest": "rq1_manifest_edge_full_fullraw_strict_v3_20260317.json",
        "canonical_rows": RQ1_RESULTS_DIR / "rq1_edge_full_fullraw_strict_v4_sharded_direct_rows_20260317.csv",
        "drain_tag": "edge_full_fullraw_strict_v6_official_drain_only",
        "composite_tag": "edge_full_fullraw_strict_v6_official_drain_only_composite",
        "cases_per_chunk_spec": "HDFS:500,OpenStack:400,Hadoop:350",
    },
]


def _timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _log(message: str) -> None:
    line = f"[{_timestamp()}] {message}"
    print(line, flush=True)
    STATUS_MD.parent.mkdir(parents=True, exist_ok=True)
    with STATUS_MD.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _run(cmd: list[str]) -> None:
    _log("RUN " + " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))


def _run_stage(stage: dict) -> int:
    drain_rows = RQ1_RESULTS_DIR / f"rq1_{stage['drain_tag']}_rows_20260317.csv"
    drain_summary = RQ1_RESULTS_DIR / f"rq1_{stage['drain_tag']}_summary_20260317.json"
    composite_rows = RQ1_RESULTS_DIR / f"rq1_{stage['composite_tag']}_rows_20260317.csv"
    composite_summary = RQ1_RESULTS_DIR / f"rq1_{stage['composite_tag']}_summary_20260317.json"
    audit_json = RQ1_RESULTS_DIR / f"rq1_{stage['composite_tag']}_audit_20260317.json"
    audit_md = REPORT_DIR / f"rq1_{stage['composite_tag']}_audit_20260318.md"

    runner_cmd = [
        sys.executable,
        str(RUNNER),
        "--run-tag",
        stage["drain_tag"],
        "--manifest-name",
        stage["manifest"],
        "--cpu-threads",
        "2",
        "--memory-target-mb",
        "2304",
    ]
    if stage["cases_per_chunk_spec"]:
        runner_cmd.extend(["--cases-per-chunk-spec", stage["cases_per_chunk_spec"]])
    _run(runner_cmd)

    _run(
        [
            sys.executable,
            str(MERGER),
            "--run-tag",
            stage["composite_tag"],
            "--manifest-name",
            stage["manifest"],
            "--canonical-rows",
            str(stage["canonical_rows"]),
            "--drain-rows",
            str(drain_rows),
            "--drain-summary",
            str(drain_summary),
        ]
    )
    _run(
        [
            sys.executable,
            str(AUDITOR),
            "--manifest-name",
            stage["manifest"],
            "--rows-path",
            str(composite_rows),
            "--json-out",
            str(audit_json),
            "--md-out",
            str(audit_md),
        ]
    )

    summary = _load_json(composite_summary)
    audit = _load_json(audit_json)
    if summary.get("acceptance_flags"):
        _log(f"{stage['name']} acceptance failed: {summary['acceptance_flags']}")
        return 1
    if audit.get("flags"):
        _log(f"{stage['name']} audit failed: {audit['flags']}")
        return 1
    _log(f"{stage['name']} gate passed")
    return 0


def main() -> int:
    STATUS_MD.write_text("# RQ1 Official Drain-Only Pipeline\n\n", encoding="utf-8")
    for stage in STAGES:
        rc = _run_stage(stage)
        if rc != 0:
            return rc
    _log("full gate passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
