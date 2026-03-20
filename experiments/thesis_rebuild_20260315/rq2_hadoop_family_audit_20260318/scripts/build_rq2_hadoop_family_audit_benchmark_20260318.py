from __future__ import annotations

import argparse
from collections import Counter

from rq2_hadoop_family_audit_common_20260318 import (
    HADOOP_BENCHMARK_CORE_PATH,
    HADOOP_BENCHMARK_FULL_PATH,
    REPORTS_DIR,
    SOURCE_BENCH_PATH,
    build_hadoop_family_case,
    ensure_dirs,
    load_hadoop_timeseries,
    load_json,
    write_json,
)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    ensure_dirs()
    report_path = REPORTS_DIR / "rq2_hadoop_family_audit_benchmark_20260318.md"
    if (
        HADOOP_BENCHMARK_FULL_PATH.exists()
        and HADOOP_BENCHMARK_CORE_PATH.exists()
        and report_path.exists()
        and not args.force
    ):
        print(f"[*] Reusing Hadoop family benchmark artifacts under {HADOOP_BENCHMARK_FULL_PATH.parent}")
        print(f"[*] Reusing benchmark report: {report_path}")
        return

    print("[1/3] Loading source Hadoop benchmark and template pool. Expected: <5s")
    rows = [
        row
        for row in load_json(SOURCE_BENCH_PATH)
        if str(row.get("dataset", "") or "") == "Hadoop"
    ]
    _, tpl_map = load_hadoop_timeseries()
    template_pool = sorted(set(str(x or "") for x in tpl_map.values() if str(x or "").strip()))

    print("[2/3] Rebuilding Hadoop benchmark with family-level root supervision. Expected: <10s")
    full_rows = [build_hadoop_family_case(row, template_pool) for row in rows]
    core_rows = [
        row
        for row in full_rows
        if bool(row.get("audit_derivation_complete")) and bool(row.get("audit_local_core_eligible"))
    ]

    print("[3/3] Writing benchmark artifacts and report. Expected: <5s")
    write_json(HADOOP_BENCHMARK_FULL_PATH, full_rows)
    write_json(HADOOP_BENCHMARK_CORE_PATH, core_rows)

    root_family_counts = Counter(str(row.get("ground_truth_root_family", "")) for row in full_rows)
    effect_counts = Counter(str(row.get("ground_truth_template_graph", "")) for row in full_rows)
    status40 = Counter(str(row.get("audit_local_root_status_radius40", "")) for row in full_rows)
    status80 = Counter(str(row.get("audit_local_root_status_radius80", "")) for row in full_rows)
    pair_counts = Counter(
        (
            str(row.get("ground_truth_root_family", "")),
            str(row.get("ground_truth_template_graph", "")),
        )
        for row in full_rows
    )
    derivation_incomplete = sum(not bool(row.get("audit_derivation_complete")) for row in full_rows)
    core_missing_root = sum(not bool(row.get("audit_local_root_graphs_radius40")) for row in core_rows)
    core_ambiguous = sum(
        str(row.get("audit_local_root_status_radius40", "")) == "ambiguous_graph"
        for row in core_rows
    )

    lines = [
        "# RQ2 Hadoop Family Audit Benchmark (2026-03-18)",
        "",
        f"- Workspace: `{HADOOP_BENCHMARK_FULL_PATH.parent.parent}`",
        "- Hadoop is rebuilt as an exact-effect plus root-family task.",
        "- The old representative-root-template forcing is not used.",
        "- The full benchmark keeps all Hadoop rows whose effect anchoring and root-family derivation are available.",
        "- The local-core subset is only a sanity slice where the prior window within 40 lines yields exactly one non-effect root-graph candidate.",
        "",
        "## Outputs",
        "",
        f"- Full benchmark: `{HADOOP_BENCHMARK_FULL_PATH}`",
        f"- Local-core benchmark: `{HADOOP_BENCHMARK_CORE_PATH}`",
        "",
        "## Final Sizes",
        "",
        f"- Full Hadoop rows: `{len(full_rows)}`",
        f"- Local-core rows: `{len(core_rows)}`",
        f"- Derivation incomplete rows in full set: `{derivation_incomplete}`",
        f"- Missing local root rows in local-core set: `{core_missing_root}`",
        f"- Ambiguous local root rows in local-core set: `{core_ambiguous}`",
        "",
        "## Root Family Counts",
        "",
    ]
    for family, count in sorted(root_family_counts.items()):
        lines.append(f"- `{family}`: `{count}`")

    lines.extend(
        [
            "",
            "## Local Root Status",
            "",
            "### Radius 40",
            "",
        ]
    )
    for status, count in sorted(status40.items()):
        lines.append(f"- `{status}`: `{count}`")

    lines.extend(
        [
            "",
            "### Radius 80",
            "",
        ]
    )
    for status, count in sorted(status80.items()):
        lines.append(f"- `{status}`: `{count}`")

    lines.extend(
        [
            "",
            "## Effect Template Counts",
            "",
        ]
    )
    for effect, count in effect_counts.most_common():
        lines.append(f"- `{effect}`: `{count}`")

    lines.extend(
        [
            "",
            "## Root Family x Effect Template",
            "",
        ]
    )
    for (family, effect), count in pair_counts.most_common():
        lines.append(f"- `{family}` -> `{effect}`: `{count}`")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[Saved] {HADOOP_BENCHMARK_FULL_PATH}")
    print(f"[Saved] {HADOOP_BENCHMARK_CORE_PATH}")
    print(f"[Saved] {report_path}")


if __name__ == "__main__":
    main()
