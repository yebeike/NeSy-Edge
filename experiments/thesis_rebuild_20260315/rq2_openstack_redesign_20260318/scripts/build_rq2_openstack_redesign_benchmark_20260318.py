from __future__ import annotations

import argparse
import os
from collections import Counter
from pathlib import Path
from typing import Dict, List

from rq2_openstack_redesign_common_20260318 import (
    CANDIDATE_POOL_PATH,
    COMBINED_EVAL_BENCH_PATH,
    CORE_BENCH_PATH,
    CORE_MAX_PER_PAIR_PER_FILE,
    CORE_MAX_PER_PAIR_TOTAL,
    CORE_MIN_GAP,
    EXTENDED_BENCH_PATH,
    EXTENDED_MIN_GAP,
    HDFS_FROZEN_BENCH_PATH,
    OPENSTACK_CASE_SPECS,
    RAW_OPENSTACK_2,
    REPORTS_DIR,
    build_openstack_hit_candidate,
    domain_template_pool,
    ensure_dirs,
    frozen_hdfs_rows,
    iter_with_progress,
    select_core_candidates,
    select_diverse_candidates,
    write_json,
)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    return ap.parse_args()


def _raw_files() -> List[Path]:
    return [RAW_OPENSTACK_2 / name for name in sorted(os.listdir(RAW_OPENSTACK_2)) if name.endswith(".log")]


def main() -> None:
    args = _parse_args()
    ensure_dirs()
    report_path = REPORTS_DIR / "rq2_openstack_redesign_benchmark_20260318.md"
    if (
        CANDIDATE_POOL_PATH.exists()
        and EXTENDED_BENCH_PATH.exists()
        and CORE_BENCH_PATH.exists()
        and HDFS_FROZEN_BENCH_PATH.exists()
        and COMBINED_EVAL_BENCH_PATH.exists()
        and report_path.exists()
        and not args.force
    ):
        print(f"[*] Reusing benchmark artifacts under {CANDIDATE_POOL_PATH.parent}")
        print(f"[*] Reusing benchmark report: {report_path}")
        return

    print("[1/4] Loading template pool and raw OpenStack logs. Expected: <10s")
    openstack_pool = domain_template_pool()["OpenStack"]
    files = _raw_files()

    print("[2/4] Enumerating and annotating raw trigger hits. Expected: 30-90s")
    candidate_pool: List[Dict[str, object]] = []
    per_spec_stats: Dict[str, Counter] = {}
    per_spec_candidates: Dict[str, List[Dict[str, object]]] = {}
    for spec in OPENSTACK_CASE_SPECS:
        spec_name = str(spec["name"])
        keyword = str(spec["keyword"]).lower()
        spec_rows: List[Dict[str, object]] = []
        stats = Counter()
        for path in files:
            lines = path.read_text(encoding="latin-1", errors="ignore").splitlines()
            indexed_hits = [(idx, lines) for idx, line in enumerate(lines) if keyword in line.lower()]
            for idx, current_lines in iter_with_progress(indexed_hits, f"{spec_name} hits from {path.name}"):
                row = build_openstack_hit_candidate(spec, path, idx, current_lines, openstack_pool)
                spec_rows.append(row)
                stats["total_hits"] += 1
                stats[str(row.get("audit_root_status", ""))] += 1
        candidate_pool.extend(spec_rows)
        per_spec_stats[spec_name] = stats
        per_spec_candidates[spec_name] = spec_rows

    print("[3/4] Selecting extended and core tracks. Expected: <10s")
    extended_rows: List[Dict[str, object]] = []
    core_rows: List[Dict[str, object]] = []
    extended_case_idx = 0
    core_case_idx = 0
    extended_counts = Counter()
    core_counts = Counter()
    for spec in OPENSTACK_CASE_SPECS:
        spec_name = str(spec["name"])
        requested = int(spec["count"])
        selected_extended = select_diverse_candidates(
            per_spec_candidates[spec_name],
            count=requested,
            min_gap=EXTENDED_MIN_GAP,
        )
        selected_core = select_core_candidates(
            per_spec_candidates[spec_name],
            min_gap=CORE_MIN_GAP,
        )
        for row in selected_extended:
            row2 = dict(row)
            row2["track"] = "extended"
            row2["case_id"] = f"openstack_redesign_extended_{extended_case_idx:03d}"
            row2["audit_selection_reason"] = "quality_ranked_extended_sample"
            extended_rows.append(row2)
            extended_case_idx += 1
            extended_counts[spec_name] += 1
        for row in selected_core:
            row2 = dict(row)
            row2["track"] = "core"
            row2["case_id"] = f"openstack_redesign_core_{core_case_idx:03d}"
            row2["audit_selection_reason"] = "core_eligible_quality_first"
            core_rows.append(row2)
            core_case_idx += 1
            core_counts[spec_name] += 1

    hdfs_rows = frozen_hdfs_rows()
    combined_eval_rows = hdfs_rows + core_rows

    print("[4/4] Writing benchmark artifacts and report. Expected: <5s")
    write_json(CANDIDATE_POOL_PATH, candidate_pool)
    write_json(EXTENDED_BENCH_PATH, extended_rows)
    write_json(CORE_BENCH_PATH, core_rows)
    write_json(HDFS_FROZEN_BENCH_PATH, hdfs_rows)
    write_json(COMBINED_EVAL_BENCH_PATH, combined_eval_rows)

    extended_status = Counter(str(row.get("audit_root_status", "")) for row in extended_rows)
    core_pairs = Counter(
        (
            str(row.get("ground_truth_root_cause_template_graph", "")),
            str(row.get("ground_truth_template_graph", "")),
        )
        for row in core_rows
    )
    core_manual_overlap = sum(bool(row.get("manual_prior_pair_overlap")) for row in core_rows)
    core_missing_root = sum(not bool(row.get("ground_truth_root_cause_template_graph")) for row in core_rows)
    core_derivation_incomplete = sum(not bool(row.get("audit_derivation_complete")) for row in core_rows)

    report_lines = [
        "# RQ2 OpenStack Redesign Benchmark (2026-03-18)",
        "",
        f"- Workspace: `{CANDIDATE_POOL_PATH.parent.parent}`",
        "- HDFS is copied unchanged from the audit fork and treated as frozen.",
        "- OpenStack effects are anchored to the sampled hit line itself.",
        "- OpenStack core keeps only rows with exactly one non-effect same-family prior graph candidate, zero manual-prior overlap, and no semantic-near-duplicate root/effect mapping.",
        f"- OpenStack core applies pair/file redundancy caps: at least `{CORE_MIN_GAP}` line gap within a file, at most `{CORE_MAX_PER_PAIR_PER_FILE}` rows per pair per file, and at most `{CORE_MAX_PER_PAIR_TOTAL}` rows per exact graph pair.",
        "- OpenStack extended keeps a balanced quality-ranked sample for sensitivity analysis, with full annotations preserved.",
        "",
        "## Outputs",
        "",
        f"- Candidate pool: `{CANDIDATE_POOL_PATH}`",
        f"- OpenStack extended benchmark: `{EXTENDED_BENCH_PATH}`",
        f"- OpenStack core benchmark: `{CORE_BENCH_PATH}`",
        f"- HDFS frozen benchmark: `{HDFS_FROZEN_BENCH_PATH}`",
        f"- Combined eval benchmark (HDFS frozen + OpenStack core): `{COMBINED_EVAL_BENCH_PATH}`",
        "",
        "## Selection Counts",
        "",
        "| Spec | Requested Extended | Total Hits | Core-Eligible Hits | Manual Overlap Hits | Ambiguous Hits | Missing Hits | Selected Extended | Selected Core |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for spec in OPENSTACK_CASE_SPECS:
        spec_name = str(spec["name"])
        stats = per_spec_stats[spec_name]
        report_lines.append(
            "| {spec} | {requested} | {total_hits} | {core_hits} | {overlap_hits} | {ambiguous_hits} | {missing_hits} | {sel_ext} | {sel_core} |".format(
                spec=spec_name,
                requested=int(spec["count"]),
                total_hits=stats["total_hits"],
                core_hits=stats["core_eligible"],
                overlap_hits=stats["manual_prior_overlap"],
                ambiguous_hits=stats["ambiguous_same_family_root"],
                missing_hits=stats["missing_same_family_root"],
                sel_ext=extended_counts[spec_name],
                sel_core=core_counts[spec_name],
            )
        )

    report_lines.extend(
        [
            "",
            "## Final Benchmark Sizes",
            "",
            f"- HDFS frozen evaluable rows: `{len(hdfs_rows)}`",
            f"- OpenStack extended rows: `{len(extended_rows)}`",
            f"- OpenStack core rows: `{len(core_rows)}`",
            f"- Combined eval rows: `{len(combined_eval_rows)}`",
            "",
            "## Extended Track Root Status",
            "",
        ]
    )
    for status, count in sorted(extended_status.items()):
        report_lines.append(f"- `{status}`: `{count}`")

    report_lines.extend(
        [
            "",
            "## OpenStack Core Integrity",
            "",
            f"- manual_prior_pair_overlap rows: `{core_manual_overlap}`",
            f"- missing root rows: `{core_missing_root}`",
            f"- derivation incomplete rows: `{core_derivation_incomplete}`",
            "",
            "## OpenStack Core Graph Pairs",
            "",
        ]
    )
    for (root_graph, effect_graph), count in core_pairs.most_common(20):
        report_lines.append(f"- `{root_graph}` -> `{effect_graph}`: `{count}`")

    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"[Saved] {CANDIDATE_POOL_PATH}")
    print(f"[Saved] {EXTENDED_BENCH_PATH}")
    print(f"[Saved] {CORE_BENCH_PATH}")
    print(f"[Saved] {HDFS_FROZEN_BENCH_PATH}")
    print(f"[Saved] {COMBINED_EVAL_BENCH_PATH}")
    print(f"[Saved] {report_path}")


if __name__ == "__main__":
    main()
