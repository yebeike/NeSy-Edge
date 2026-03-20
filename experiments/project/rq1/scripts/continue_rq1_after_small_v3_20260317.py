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
RUNNER = REBUILD_ROOT / "rq1" / "scripts" / "run_rq1_edge_chunked_resumable_20260317.py"
AUDITOR = REBUILD_ROOT / "rq1" / "scripts" / "audit_rq1_artifacts_20260317.py"

SMALL_TAG = "edge_small_fullraw_strict_v3_direct"
SMALL_MANIFEST = "rq1_manifest_edge_small_fullraw_strict_v3_20260317.json"
SMALL_SUMMARY = RQ1_RESULTS_DIR / "rq1_edge_small_fullraw_strict_v3_direct_summary_20260317.json"
SMALL_ROWS = RQ1_RESULTS_DIR / "rq1_edge_small_fullraw_strict_v3_direct_rows_20260317.csv"
SMALL_AUDIT_JSON = RQ1_RESULTS_DIR / "rq1_edge_small_fullraw_strict_v3_direct_audit_20260317.json"
SMALL_AUDIT_MD = REPORT_DIR / "rq1_edge_small_fullraw_strict_v3_direct_audit_20260317.md"

MID_TAG = "edge_mid_fullraw_strict_v3_chunked_direct"
MID_MANIFEST = "rq1_manifest_edge_mid_fullraw_strict_v3_20260317.json"
MID_SUMMARY = RQ1_RESULTS_DIR / "rq1_edge_mid_fullraw_strict_v3_chunked_direct_summary_20260317.json"
MID_ROWS = RQ1_RESULTS_DIR / "rq1_edge_mid_fullraw_strict_v3_chunked_direct_rows_20260317.csv"
MID_AUDIT_JSON = RQ1_RESULTS_DIR / "rq1_edge_mid_fullraw_strict_v3_chunked_direct_audit_20260317.json"
MID_AUDIT_MD = REPORT_DIR / "rq1_edge_mid_fullraw_strict_v3_chunked_direct_audit_20260317.md"

FULL_TAG = "edge_full_fullraw_strict_v3_chunked_direct"
FULL_MANIFEST = "rq1_manifest_edge_full_fullraw_strict_v3_20260317.json"
FULL_SUMMARY = RQ1_RESULTS_DIR / "rq1_edge_full_fullraw_strict_v3_chunked_direct_summary_20260317.json"
FULL_ROWS = RQ1_RESULTS_DIR / "rq1_edge_full_fullraw_strict_v3_chunked_direct_rows_20260317.csv"
FULL_AUDIT_JSON = RQ1_RESULTS_DIR / "rq1_edge_full_fullraw_strict_v3_chunked_direct_audit_20260317.json"
FULL_AUDIT_MD = REPORT_DIR / "rq1_edge_full_fullraw_strict_v3_chunked_direct_audit_20260317.md"

STATUS_MD = REPORT_DIR / "rq1_small_to_full_continuation_v3_20260317.md"


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


def _wait_for_summary(path: Path, label: str, poll_s: int = 20) -> None:
    while not path.exists():
        _log(f"waiting for {label} summary: {path.name}")
        time.sleep(poll_s)
    _log(f"{label} summary detected: {path.name}")


def _audit(manifest_name: str, rows: Path, audit_json: Path, audit_md: Path) -> dict:
    _run(
        [
            sys.executable,
            str(AUDITOR),
            "--manifest-name",
            manifest_name,
            "--rows-path",
            str(rows),
            "--json-out",
            str(audit_json),
            "--md-out",
            str(audit_md),
        ]
    )
    return _load_json(audit_json)


def _launch_chunked(run_tag: str, manifest_name: str) -> None:
    summary_path = RQ1_RESULTS_DIR / f"rq1_{run_tag}_summary_20260317.json"
    if summary_path.exists():
        _log(f"summary already exists for {run_tag}; skip launch")
        return
    _run(
        [
            sys.executable,
            str(RUNNER),
            "--run-tag",
            run_tag,
            "--manifest-name",
            manifest_name,
            "--cpu-threads",
            "2",
            "--memory-target-mb",
            "2304",
            "--qwen-mode",
            "direct",
        ]
    )


def main() -> int:
    STATUS_MD.write_text("# RQ1 Small-to-Full Continuation v3\n\n", encoding="utf-8")

    _wait_for_summary(SMALL_SUMMARY, "small")
    small_summary = _load_json(SMALL_SUMMARY)
    small_audit = _audit(SMALL_MANIFEST, SMALL_ROWS, SMALL_AUDIT_JSON, SMALL_AUDIT_MD)
    if small_summary.get("acceptance_flags"):
        _log(f"small acceptance failed: {small_summary['acceptance_flags']}")
        return 1
    if small_audit.get("flags"):
        _log(f"small audit failed: {small_audit['flags']}")
        return 1
    _log("small gate passed; launching mid")

    _launch_chunked(MID_TAG, MID_MANIFEST)
    mid_summary = _load_json(MID_SUMMARY)
    mid_audit = _audit(MID_MANIFEST, MID_ROWS, MID_AUDIT_JSON, MID_AUDIT_MD)
    if mid_summary.get("acceptance_flags"):
        _log(f"mid acceptance failed: {mid_summary['acceptance_flags']}")
        return 1
    if mid_audit.get("flags"):
        _log(f"mid audit failed: {mid_audit['flags']}")
        return 1
    _log("mid gate passed; launching full")

    _launch_chunked(FULL_TAG, FULL_MANIFEST)
    full_summary = _load_json(FULL_SUMMARY)
    full_audit = _audit(FULL_MANIFEST, FULL_ROWS, FULL_AUDIT_JSON, FULL_AUDIT_MD)
    if full_summary.get("acceptance_flags"):
        _log(f"full acceptance failed: {full_summary['acceptance_flags']}")
        return 1
    if full_audit.get("flags"):
        _log(f"full audit failed: {full_audit['flags']}")
        return 1
    _log("full gate passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
