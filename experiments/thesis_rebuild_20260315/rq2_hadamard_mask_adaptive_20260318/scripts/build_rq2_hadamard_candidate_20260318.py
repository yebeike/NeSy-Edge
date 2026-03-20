from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_FINAL_ROOT = _SCRIPT_DIR.parent
_REBUILD_ROOT = _FINAL_ROOT.parents[0]
_PROJECT_ROOT = _FINAL_ROOT.parents[2]

for path in [
    _PROJECT_ROOT,
    _PROJECT_ROOT / "experiments" / "rq123_e2e",
    _REBUILD_ROOT / "rq2_mainline_penalized_20260318" / "scripts",
    _FINAL_ROOT / "scripts",
]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from rq2_hadamard_candidate_common_20260318 import (  # type: ignore
    FROZEN_COMPARE_GRAPH_FILES,
    HADAMARD_GRAPH_PATH,
    MASK_BACKGROUND,
    MASK_CURATED,
    MASK_OTHER_SOURCE,
    MASK_PROFILE_PATH,
    MASK_REVERSE,
    MASK_SYMBOLIC,
    CURATED_THRESHOLD_SCALE,
    PILOT_ACTIVE_PENALTY,
    PILOT_MAX_ITER,
    PILOT_OTHER_ACTIVE_PENALTY,
    PILOT_SUPPORT_FLOOR,
    PRIOR_THRESHOLD_SCALE,
    REPORTS_DIR,
    build_hadamard_graph,
    ensure_dirs,
    graph_cache_path,
    graph_relation_stats,
    load_json,
    write_json,
    write_text,
)
from rq2_mainline_completion_common_20260318 import Heartbeat  # type: ignore


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    return ap.parse_args()


def _load_or_build(dataset: str, force: bool):
    cache_path = graph_cache_path(dataset)
    if cache_path.exists() and not force:
        print(f"[*] Reusing {dataset}/hadamard_mask_dynotears: {cache_path}", flush=True)
        rows = load_json(cache_path)
        return rows, None
    with Heartbeat(f"{dataset}/hadamard_mask_dynotears", interval_sec=30, remaining="masked DYNOTEARS fit"):
        rows, profile = build_hadamard_graph(dataset)
    write_json(cache_path, rows)
    print(f"[Saved] {cache_path}", flush=True)
    return rows, profile


def main() -> None:
    args = _parse_args()
    ensure_dirs()
    report_path = REPORTS_DIR / "rq2_hadamard_graph_build_20260318.md"
    if HADAMARD_GRAPH_PATH.exists() and MASK_PROFILE_PATH.exists() and report_path.exists() and not args.force:
        print(f"[*] Reusing hadamard candidate artifacts under {HADAMARD_GRAPH_PATH.parent}", flush=True)
        print(f"[*] Reusing graph build report: {report_path}", flush=True)
        return

    print("[1/2] Building hadamard_mask_dynotears for HDFS/OpenStack/Hadoop. Expected: 1-20 min", flush=True)
    datasets = ["HDFS", "OpenStack", "Hadoop"]
    combined = []
    graph_stats_rows = []
    mask_profiles = []
    reused_profiles = {}
    if MASK_PROFILE_PATH.exists() and not args.force:
        for row in load_json(MASK_PROFILE_PATH):
            reused_profiles[str(row["dataset"])] = row

    for dataset in datasets:
        rows, profile = _load_or_build(dataset, args.force)
        combined.extend(rows)
        graph_stats_rows.append(graph_relation_stats(rows, dataset))
        mask_profiles.append(profile or reused_profiles.get(dataset) or {"dataset": dataset, "reused_without_profile": True})
        print(f"[dataset] {dataset}: edges={len(rows)}", flush=True)

    print("[2/2] Writing combined candidate graph and reports. Expected: <5s", flush=True)
    write_json(HADAMARD_GRAPH_PATH, combined)
    write_json(MASK_PROFILE_PATH, mask_profiles)
    print(f"[Saved] {HADAMARD_GRAPH_PATH}", flush=True)
    print(f"[Saved] {MASK_PROFILE_PATH}", flush=True)

    lines = [
        "# RQ2 Hadamard-Mask Candidate Graph Build (2026-03-18)",
        "",
        f"- Workspace: `{_FINAL_ROOT}`",
        "- This workspace adds only one new candidate method: `hadamard_mask_dynotears`.",
        "- All four comparison methods remain frozen from `rq2_mainline_penalized_20260318`.",
        "- Feature spaces, graph hyperparameters, benchmark rows, and paper metrics are reused unchanged from the frozen penalized mainline.",
        "- The new candidate keeps the Hadamard prior-mask DYNOTEARS objective and adds an adaptive second pass that penalizes pilot-active non-prior edges.",
        "- No benchmark-derived edge or case-level supervision is injected into the mask construction.",
        f"- Adaptive mask profile: background={MASK_BACKGROUND}, symbolic={MASK_SYMBOLIC}, curated={MASK_CURATED}, reverse={MASK_REVERSE}, other_source={MASK_OTHER_SOURCE}.",
        f"- Adaptive stage-2 profile: pilot_max_iter={PILOT_MAX_ITER}, pilot_support_floor={PILOT_SUPPORT_FLOOR}, pilot_active_penalty={PILOT_ACTIVE_PENALTY}, pilot_other_active_penalty={PILOT_OTHER_ACTIVE_PENALTY}.",
        f"- Prior-aware export thresholds: symbolic={PRIOR_THRESHOLD_SCALE}x base threshold, curated={CURATED_THRESHOLD_SCALE}x base threshold.",
        "",
        "## Frozen Comparison Artifacts",
        "",
    ]
    for name, path in FROZEN_COMPARE_GRAPH_FILES.items():
        label = "candidate" if name == "hadamard_mask_dynotears" else "frozen"
        lines.append(f"- `{name}` ({label}): `{path}`")
    lines.extend(
        [
            "",
            "## Mask Profiles",
            "",
            "| Dataset | Selected_Columns | Symbolic_Edges_Mapped | Curated_Symbolic_Edges | Mask_Entries_Softened | Reverse_Penalized | Stage2_W | Stage2_A | Kept_Prior | Kept_Curated | W_Mask_Min | W_Mask_Max |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in mask_profiles:
        lines.append(
            f"| {row['dataset']} | {row.get('selected_columns', 'NA')} | {row.get('symbolic_edges_mapped', 'NA')} | {row.get('curated_symbolic_edges', 'NA')} | {row.get('mask_entries_softened', 'NA')} | {row.get('mask_entries_reverse_penalized', 'NA')} | {row.get('stage2_penalized_w_entries', 'NA')} | {row.get('stage2_penalized_a_entries', 'NA')} | {row.get('kept_prior_edges', 'NA')} | {row.get('kept_curated_edges', 'NA')} | {row.get('w_mask_min', 'NA')} | {row.get('w_mask_max', 'NA')} |"
        )
    lines.extend(
        [
            "",
            "## Graph Stats",
            "",
            "| Dataset | Edges | Relations |",
            "|---|---:|---|",
        ]
    )
    for row in graph_stats_rows:
        lines.append(f"| {row['dataset']} | {row['edges']} | `{row['relations']}` |")
    write_text(report_path, "\n".join(lines) + "\n")
    print(f"[Saved] {report_path}", flush=True)


if __name__ == "__main__":
    main()
