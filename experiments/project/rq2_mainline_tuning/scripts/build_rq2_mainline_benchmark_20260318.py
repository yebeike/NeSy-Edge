from __future__ import annotations

import argparse
from collections import Counter

from rq2_mainline_completion_common_20260318 import (
    REPORTS_DIR,
    UNIFIED_BENCHMARK_ALL_PATH,
    UNIFIED_BENCHMARK_APPENDIX_PATH,
    UNIFIED_BENCHMARK_MAINLINE_PATH,
    build_unified_benchmark_rows,
    ensure_dirs,
    write_json,
)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    ensure_dirs()
    report_path = REPORTS_DIR / "rq2_mainline_benchmark_20260318.md"
    if (
        UNIFIED_BENCHMARK_ALL_PATH.exists()
        and UNIFIED_BENCHMARK_MAINLINE_PATH.exists()
        and UNIFIED_BENCHMARK_APPENDIX_PATH.exists()
        and report_path.exists()
        and not args.force
    ):
        print(f"[*] Reusing unified benchmark artifacts under {UNIFIED_BENCHMARK_ALL_PATH.parent}")
        print(f"[*] Reusing benchmark report: {report_path}")
        return

    print("[1/2] Importing frozen benchmark rows from audit/redesign workspaces. Expected: <10s")
    all_rows, mainline_rows, appendix_rows = build_unified_benchmark_rows()

    print("[2/2] Writing unified benchmark artifacts and report. Expected: <5s")
    write_json(UNIFIED_BENCHMARK_ALL_PATH, all_rows)
    write_json(UNIFIED_BENCHMARK_MAINLINE_PATH, mainline_rows)
    write_json(UNIFIED_BENCHMARK_APPENDIX_PATH, appendix_rows)

    mainline_counts = Counter(str(row["dataset"]) for row in mainline_rows)
    appendix_counts = Counter(str(row["dataset"]) for row in appendix_rows)
    root_types = {str(row["dataset"]): str(row["root_target_type"]) for row in mainline_rows}

    lines = [
        "# RQ2 Mainline Unified Benchmark (2026-03-18)",
        "",
        f"- Workspace: `{UNIFIED_BENCHMARK_ALL_PATH.parent.parent}`",
        "- HDFS rows are copied unchanged from the audit-fork evaluable benchmark.",
        "- OpenStack rows are copied unchanged from the redesigned 97-case core benchmark.",
        "- Hadoop mainline rows are a quality-first core: family-audit cases with complete derivation and a unique local root graph within radius 80.",
        "- Hadoop full 44-case family benchmark and the stricter local40 subset are kept as appendix-only references.",
        "- HDFS/OpenStack selection is frozen; Hadoop tiering is recomputed only to separate thesis-eligible core from weakly supervised appendix rows.",
        "",
        "## Outputs",
        "",
        f"- Unified all rows: `{UNIFIED_BENCHMARK_ALL_PATH}`",
        f"- Unified mainline rows: `{UNIFIED_BENCHMARK_MAINLINE_PATH}`",
        f"- Unified appendix rows: `{UNIFIED_BENCHMARK_APPENDIX_PATH}`",
        "",
        "## Counts",
        "",
        f"- Mainline total rows: `{len(mainline_rows)}`",
        f"- Appendix total rows: `{len(appendix_rows)}`",
    ]
    for dataset in ["HDFS", "OpenStack", "Hadoop"]:
        lines.append(f"- Mainline `{dataset}` rows: `{mainline_counts[dataset]}`")
    for dataset, count in appendix_counts.items():
        lines.append(f"- Appendix `{dataset}` rows: `{count}`")

    lines.extend(
        [
            "",
            "## Root Target Types",
            "",
        ]
    )
    for dataset in ["HDFS", "OpenStack", "Hadoop"]:
        lines.append(f"- `{dataset}` root_target_type: `{root_types.get(dataset, 'unknown')}`")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[Saved] {UNIFIED_BENCHMARK_ALL_PATH}")
    print(f"[Saved] {UNIFIED_BENCHMARK_MAINLINE_PATH}")
    print(f"[Saved] {UNIFIED_BENCHMARK_APPENDIX_PATH}")
    print(f"[Saved] {report_path}")


if __name__ == "__main__":
    main()
