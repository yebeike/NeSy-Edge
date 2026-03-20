from __future__ import annotations

import argparse
from pathlib import Path

from rq2_mainline_completion_common_20260318 import (
    CACHE_DIR,
    CURATED_PRIOR_PATH,
    FEATURE_SPACE_PATH,
    GRAPH_FILES,
    HADOOP_CALIBRATION_PATH,
    MODIFIED_PROVENANCE_PATH,
    REPORTS_DIR,
    Heartbeat,
    build_original_graph,
    build_pc_graph,
    build_pearson_graph,
    collect_mapped_symbolic_prior_edges,
    ensure_dirs,
    graph_cache_path,
    graph_relation_stats,
    load_json,
    merge_modified_edges,
    prepare_feature_space,
    refresh_curated_prior_file,
    write_json,
)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    return ap.parse_args()


def _load_or_build(cache_path: Path, label: str, force: bool, builder):
    if cache_path.exists() and not force:
        print(f"[*] Reusing {label}: {cache_path}")
        return load_json(cache_path)
    with Heartbeat(label, interval_sec=30):
        rows = builder()
    write_json(cache_path, rows)
    print(f"[Saved] {cache_path}")
    return rows


def main() -> None:
    args = _parse_args()
    ensure_dirs()
    report_path = REPORTS_DIR / "rq2_mainline_graph_build_20260318.md"
    if (
        FEATURE_SPACE_PATH.exists()
        and MODIFIED_PROVENANCE_PATH.exists()
        and all(path.exists() for path in GRAPH_FILES.values())
        and report_path.exists()
        and not args.force
    ):
        print(f"[*] Reusing mainline graph artifacts under {GRAPH_FILES['modified'].parent}")
        print(f"[*] Reusing graph build report: {report_path}")
        return

    if not HADOOP_CALIBRATION_PATH.exists():
        raise FileNotFoundError(f"Missing Hadoop calibration artifact: {HADOOP_CALIBRATION_PATH}")

    refresh_curated_prior_file()
    print(f"[*] Refreshed curated symbolic priors: {CURATED_PRIOR_PATH}")

    print("[1/3] Loading shared feature spaces. Expected: <10s")
    hadoop_calibration = load_json(HADOOP_CALIBRATION_PATH)
    datasets = ["HDFS", "OpenStack", "Hadoop"]
    feature_profiles = []
    feature_spaces = {}
    for dataset in datasets:
        if dataset == "Hadoop":
            df, tpl_map, profile = prepare_feature_space(
                dataset,
                explicit_columns=list(hadoop_calibration["column_ids"]),
            )
        else:
            df, tpl_map, profile = prepare_feature_space(dataset)
        feature_profiles.append(profile)
        feature_spaces[dataset] = (df, tpl_map)
    write_json(FEATURE_SPACE_PATH, feature_profiles)
    print(f"[Saved] {FEATURE_SPACE_PATH}")

    print("[2/3] Building 4 methods for all 3 datasets. Expected: 1-20 min")
    combined = {name: [] for name in GRAPH_FILES}
    provenance_rows = []
    graph_stats_rows = []

    for dataset in datasets:
        df, tpl_map = feature_spaces[dataset]
        template_pool = list(dict.fromkeys(str(v or "") for v in tpl_map.values() if str(v or "").strip()))
        print(f"[dataset] {dataset}: selected_columns={df.shape[1]}")

        original_rows = _load_or_build(
            graph_cache_path(dataset, "original_dynotears"),
            f"{dataset}/original_dynotears",
            args.force,
            lambda dataset=dataset, df=df, tpl_map=tpl_map: build_original_graph(dataset, df, tpl_map),
        )
        pearson_rows = _load_or_build(
            graph_cache_path(dataset, "pearson_hypothesis"),
            f"{dataset}/pearson_hypothesis",
            args.force,
            lambda dataset=dataset, df=df, tpl_map=tpl_map: build_pearson_graph(dataset, df, tpl_map),
        )
        pc_rows = _load_or_build(
            graph_cache_path(dataset, "pc_cpdag_hypothesis"),
            f"{dataset}/pc_cpdag_hypothesis",
            args.force,
            lambda dataset=dataset, df=df, tpl_map=tpl_map: build_pc_graph(dataset, df, tpl_map),
        )

        symbolic_rows = collect_mapped_symbolic_prior_edges(dataset, template_pool)
        modified_rows, dataset_provenance = merge_modified_edges(dataset, original_rows, symbolic_rows)
        write_json(graph_cache_path(dataset, "modified"), modified_rows)
        print(f"[Saved] {graph_cache_path(dataset, 'modified')}")

        combined["original_dynotears"].extend(original_rows)
        combined["pearson_hypothesis"].extend(pearson_rows)
        combined["pc_cpdag_hypothesis"].extend(pc_rows)
        combined["modified"].extend(modified_rows)
        provenance_rows.extend(dataset_provenance)

        for method, rows in [
            ("original_dynotears", original_rows),
            ("pearson_hypothesis", pearson_rows),
            ("pc_cpdag_hypothesis", pc_rows),
            ("modified", modified_rows),
        ]:
            graph_stats_rows.append({"method": method, **graph_relation_stats(rows, dataset)})

    print("[3/3] Writing combined graph artifacts and report. Expected: <5s")
    for name, path in GRAPH_FILES.items():
        write_json(path, combined[name])
        print(f"[Saved] {path}")
    write_json(MODIFIED_PROVENANCE_PATH, provenance_rows)
    print(f"[Saved] {MODIFIED_PROVENANCE_PATH}")

    zero_benchmark_derived = all(not bool(row.get("benchmark_derived")) for row in provenance_rows)
    lines = [
        "# RQ2 Mainline Graph Build (2026-03-18)",
        "",
        f"- Workspace: `{GRAPH_FILES['modified'].parent.parent}`",
        "- All four methods are rebuilt in this workspace.",
        "- modified is rebuilt from the new original_dynotears backbone plus symbolic KB edges mapped into the current dataset template pool.",
        "- modified also loads a small curated supplemental prior set stored separately from the benchmark and applied uniformly by domain.",
        "- The tuned curated prior set adds only raw-timeseries-supported HDFS/Hadoop direct priors; no case-derived benchmark hints are used.",
        "- modified prunes unsupported original-backbone edges whose source stays in the dataset OTHER/UNKNOWN family; merged and curated prior edges remain eligible.",
        "- modified applies a provenance-aware confidence boost to prior-supported edges (`symbolic_prior +0.5`, `merged +0.25`) without adding benchmark-derived signals.",
        "- No benchmark-derived edges are injected into modified.",
        f"- Provenance guarantee holds: `{zero_benchmark_derived}`",
        "",
        "## Outputs",
        "",
    ]
    for name, path in GRAPH_FILES.items():
        lines.append(f"- `{name}`: `{path}`")
    lines.append(f"- `modified_provenance`: `{MODIFIED_PROVENANCE_PATH}`")
    lines.append(f"- `curated_symbolic_priors`: `{CURATED_PRIOR_PATH}`")
    lines.append(f"- `feature_spaces`: `{FEATURE_SPACE_PATH}`")
    lines.extend(
        [
            "",
            "## Feature Profiles",
            "",
            "| Dataset | Original_Columns | Constant_Removed | Duplicate_Observed | Selected_Columns | Cap |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for profile in feature_profiles:
        lines.append(
            f"| {profile['dataset']} | {profile['original_columns']} | {profile['constant_removed']} | {profile.get('duplicate_observed', profile['duplicate_removed'])} | {profile['selected_columns']} | {profile.get('cap', 'full')} |"
        )
    lines.extend(
        [
            "",
            "## Graph Stats",
            "",
            "| Dataset | Method | Edges | Relations |",
            "|---|---|---:|---|",
        ]
    )
    for row in graph_stats_rows:
        lines.append(f"| {row['dataset']} | {row['method']} | {row['edges']} | `{row['relations']}` |")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[Saved] {report_path}")


if __name__ == "__main__":
    main()
