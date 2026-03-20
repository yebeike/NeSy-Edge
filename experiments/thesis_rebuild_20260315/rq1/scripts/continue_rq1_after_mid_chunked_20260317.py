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

MID_TAG = "edge_mid_fullraw_strict_v4_chunked_direct"
MID_MANIFEST = "rq1_manifest_edge_mid_fullraw_strict_v2_20260317.json"
MID_SUMMARY = RQ1_RESULTS_DIR / "rq1_edge_mid_fullraw_strict_v4_chunked_direct_summary_20260317.json"
MID_ROWS = RQ1_RESULTS_DIR / "rq1_edge_mid_fullraw_strict_v4_chunked_direct_rows_20260317.csv"
MID_AUDIT_JSON = RQ1_RESULTS_DIR / "rq1_edge_mid_fullraw_strict_v4_chunked_direct_audit_20260317.json"
MID_AUDIT_MD = REPORT_DIR / "rq1_edge_mid_fullraw_strict_v4_chunked_direct_audit_20260317.md"

FULL_TAG = "edge_full_fullraw_strict_v4_chunked_direct"
FULL_MANIFEST = "rq1_manifest_edge_full_fullraw_strict_v2_20260317.json"
FULL_SUMMARY = RQ1_RESULTS_DIR / "rq1_edge_full_fullraw_strict_v4_chunked_direct_summary_20260317.json"
FULL_ROWS = RQ1_RESULTS_DIR / "rq1_edge_full_fullraw_strict_v4_chunked_direct_rows_20260317.csv"
FULL_AUDIT_JSON = RQ1_RESULTS_DIR / "rq1_edge_full_fullraw_strict_v4_chunked_direct_audit_20260317.json"
FULL_AUDIT_MD = REPORT_DIR / "rq1_edge_full_fullraw_strict_v4_chunked_direct_audit_20260317.md"

STATUS_MD = REPORT_DIR / "rq1_mid_to_full_continuation_v4_chunked_20260317.md"


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


def _launch_mid() -> None:
    if MID_SUMMARY.exists():
        _log("mid summary already exists; skip mid launch")
        return
    _run(
        [
            sys.executable,
            str(RUNNER),
            "--run-tag",
            MID_TAG,
            "--manifest-name",
            MID_MANIFEST,
            "--cpu-threads",
            "2",
            "--memory-target-mb",
            "2304",
            "--qwen-mode",
            "direct",
        ]
    )


def _audit_mid() -> tuple[dict, dict]:
    _run(
        [
            sys.executable,
            str(AUDITOR),
            "--manifest-name",
            MID_MANIFEST,
            "--rows-path",
            str(MID_ROWS),
            "--json-out",
            str(MID_AUDIT_JSON),
            "--md-out",
            str(MID_AUDIT_MD),
        ]
    )
    return _load_json(MID_SUMMARY), _load_json(MID_AUDIT_JSON)


def _launch_full() -> None:
    if FULL_SUMMARY.exists():
        _log("full summary already exists; skip full launch")
        return
    _run(
        [
            sys.executable,
            str(RUNNER),
            "--run-tag",
            FULL_TAG,
            "--manifest-name",
            FULL_MANIFEST,
            "--cpu-threads",
            "2",
            "--memory-target-mb",
            "2304",
            "--qwen-mode",
            "direct",
        ]
    )


def _audit_full() -> tuple[dict, dict]:
    _run(
        [
            sys.executable,
            str(AUDITOR),
            "--manifest-name",
            FULL_MANIFEST,
            "--rows-path",
            str(FULL_ROWS),
            "--json-out",
            str(FULL_AUDIT_JSON),
            "--md-out",
            str(FULL_AUDIT_MD),
        ]
    )
    return _load_json(FULL_SUMMARY), _load_json(FULL_AUDIT_JSON)


def main() -> int:
    STATUS_MD.write_text("# RQ1 Mid-to-Full Continuation v4 Chunked\n\n", encoding="utf-8")
    _launch_mid()
    mid_summary, mid_audit = _audit_mid()
    if mid_summary.get("acceptance_flags"):
        _log(f"mid acceptance failed: {mid_summary['acceptance_flags']}")
        return 1
    if mid_audit.get("flags"):
        _log(f"mid audit failed: {mid_audit['flags']}")
        return 1
    _log("mid gate passed; launching full")
    _launch_full()
    full_summary, full_audit = _audit_full()
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
