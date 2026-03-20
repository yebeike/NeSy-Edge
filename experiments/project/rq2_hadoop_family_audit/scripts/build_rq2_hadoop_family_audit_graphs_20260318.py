from __future__ import annotations

import argparse

from rq2_hadoop_family_audit_common_20260318 import (
    GRAPH_FILES,
    OLD_MODIFIED_GRAPH_PATH,
    REPORTS_DIR,
    build_original_dynotears_edges,
    build_pc_cpdag_hypothesis_edges,
    build_pearson_hypothesis_edges,
    clean_hadoop_edges,
    ensure_dirs,
    load_hadoop_timeseries,
    load_json,
    relation_stats,
    write_json,
)


ORIGINAL_PARAMS = {
    "lambda_w": 0.025,
    "lambda_a": 0.05,
    "threshold": 0.30,
    "p": 1,
    "max_iter": 100,
}
PEARSON_THRESHOLD = 0.93
PC_ALPHA = 0.05
PC_MAX_VARS = 64


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    ensure_dirs()
    report_path = REPORTS_DIR / "rq2_hadoop_family_audit_graph_build_20260318.md"
    if all(path.exists() for path in GRAPH_FILES.values()) and report_path.exists() and not args.force:
        print(f"[*] Reusing Hadoop graph artifacts under {GRAPH_FILES['modified'].parent}")
        print(f"[*] Reusing graph build report: {report_path}")
        return

    print("[1/4] Loading Hadoop timeseries and old modified graph. Expected: <5s")
    df, tpl_map = load_hadoop_timeseries()
    old_modified = load_json(OLD_MODIFIED_GRAPH_PATH)

    print("[2/4] Preparing frozen modified Hadoop slice. Expected: <5s")
    modified_edges = clean_hadoop_edges(old_modified)

    print("[3/4] Rebuilding clean baselines. Expected: 10-60s")
    original_edges = build_original_dynotears_edges(df, tpl_map, **ORIGINAL_PARAMS)
    pearson_edges = build_pearson_hypothesis_edges(df, tpl_map, threshold=PEARSON_THRESHOLD)
    pc_edges = build_pc_cpdag_hypothesis_edges(df, tpl_map, alpha=PC_ALPHA, max_vars=PC_MAX_VARS)

    print("[4/4] Writing graph artifacts and report. Expected: <5s")
    graph_payloads = {
        "modified": modified_edges,
        "original_dynotears": original_edges,
        "pearson_hypothesis": pearson_edges,
        "pc_cpdag_hypothesis": pc_edges,
    }
    for name, path in GRAPH_FILES.items():
        write_json(path, graph_payloads[name])

    lines = [
        "# RQ2 Hadoop Family Audit Graph Build (2026-03-18)",
        "",
        f"- Workspace: `{GRAPH_FILES['modified'].parent.parent}`",
        "- modified is copied from the old fullcase artifact and cleaned only by domain slicing, self-loop removal, and undirected-safe deduplication.",
        "- original_dynotears is rebuilt with the vendored official-style DYNOTEARS implementation from the audit fork.",
        "- pearson_hypothesis is an undirected correlation hypothesis graph.",
        "- pc_cpdag_hypothesis is a CPDAG-faithful hypothesis graph.",
        "",
        "## Parameters",
        "",
        f"- original_dynotears `(lambda_w, lambda_a, threshold, p, max_iter)`: `{ORIGINAL_PARAMS['lambda_w']}, {ORIGINAL_PARAMS['lambda_a']}, {ORIGINAL_PARAMS['threshold']}, {ORIGINAL_PARAMS['p']}, {ORIGINAL_PARAMS['max_iter']}`",
        f"- pearson_hypothesis threshold: `{PEARSON_THRESHOLD}`",
        f"- pc_cpdag_hypothesis `(alpha, max_vars)`: `{PC_ALPHA}, {PC_MAX_VARS}`",
        "",
        "## Outputs",
        "",
    ]
    for name, path in GRAPH_FILES.items():
        lines.append(f"- `{name}`: `{path}`")

    lines.extend(
        [
            "",
            "## Graph Stats",
            "",
            "| Graph | Edges | Edges_After_Clean | Self_Edges_Detected | Relation_Breakdown |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for name in ["modified", "original_dynotears", "pearson_hypothesis", "pc_cpdag_hypothesis"]:
        stats = relation_stats(graph_payloads[name])
        lines.append(
            f"| {name} | {stats['edges']} | {stats['edges_after_clean']} | {stats['self_edges_detected']} | `{stats['relations']}` |"
        )

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for name, path in GRAPH_FILES.items():
        print(f"[Saved] {name}: {path}")
    print(f"[Saved] {report_path}")


if __name__ == "__main__":
    main()
