from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

from rq2_fullcase_audit_common_20260318 import (
    GRAPH_FILES,
    OLD_MODIFIED_GRAPH_PATH,
    OLD_OPENSTACK_ID_MAP_PATH,
    OLD_OPENSTACK_TS_PATH,
    REPORTS_DIR,
    build_original_dynotears_edges,
    build_pc_cpdag_hypothesis_edges,
    build_pearson_hypothesis_edges,
    ensure_dirs,
    load_hdfs_timeseries,
    load_json,
    load_openstack_semantic_timeseries,
    relation_stats,
    write_json,
)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", default="hdfs,openstack")
    ap.add_argument("--orig-hdfs-lambda-w", type=float, default=0.025)
    ap.add_argument("--orig-hdfs-lambda-a", type=float, default=0.05)
    ap.add_argument("--orig-hdfs-threshold", type=float, default=0.30)
    ap.add_argument("--orig-openstack-lambda-w", type=float, default=0.030)
    ap.add_argument("--orig-openstack-lambda-a", type=float, default=0.06)
    ap.add_argument("--orig-openstack-threshold", type=float, default=0.40)
    ap.add_argument("--orig-p", type=int, default=1)
    ap.add_argument("--orig-max-iter", type=int, default=100)
    ap.add_argument("--pearson-thresh-hdfs", type=float, default=0.30)
    ap.add_argument("--pearson-thresh-openstack", type=float, default=0.91)
    ap.add_argument("--pc-alpha-hdfs", type=float, default=0.05)
    ap.add_argument("--pc-alpha-openstack", type=float, default=0.30)
    ap.add_argument("--force", action="store_true")
    return ap.parse_args()


def _load_domain_data(args: argparse.Namespace, domains: List[str]) -> Dict[str, Tuple[object, object]]:
    data: Dict[str, Tuple[object, object]] = {}
    for dom in domains:
        if dom == "hdfs":
            data[dom] = load_hdfs_timeseries()
        elif dom == "openstack":
            data[dom] = load_openstack_semantic_timeseries(Path(OLD_OPENSTACK_TS_PATH), Path(OLD_OPENSTACK_ID_MAP_PATH))
        else:
            raise ValueError(f"Unsupported domain: {dom}")
    return data


def main() -> None:
    args = _parse_args()
    ensure_dirs()
    report_path = REPORTS_DIR / "rq2_fullcase_audit_graph_build_20260318.md"
    domains = [x.strip().lower() for x in args.domains.split(",") if x.strip()]
    print("[1/4] Loading domain time series and ID maps. Expected: <10s")
    data = _load_domain_data(args, domains)

    if all(path.exists() for path in GRAPH_FILES.values()) and report_path.exists() and not args.force:
        print("[*] Reusing all audit graph artifacts.")
        print(f"[*] Reusing graph build report: {report_path}")
        return

    print("[2/4] Copying frozen modified graph slice. Expected: <5s")
    modified_source = load_json(OLD_MODIFIED_GRAPH_PATH)
    modified_edges = [
        edge
        for edge in modified_source
        if str(edge.get("domain", "")).lower() in set(domains)
    ]

    original_edges: List[Dict[str, object]] = []
    pearson_edges: List[Dict[str, object]] = []
    pc_edges: List[Dict[str, object]] = []

    total_domains = len(domains) or 1
    for dom in domains:
        position = domains.index(dom) + 1
        df, id_map = data[dom]
        if dom == "hdfs":
            print(f"[3/4][{position}/{total_domains}] Building HDFS graphs. Expected: 15-45s")
            print("  - original_dynotears")
            original_edges.extend(
                build_original_dynotears_edges(
                    df,
                    id_map,
                    dom,
                    lambda_w=args.orig_hdfs_lambda_w,
                    lambda_a=args.orig_hdfs_lambda_a,
                    threshold=args.orig_hdfs_threshold,
                    p=args.orig_p,
                    max_iter=args.orig_max_iter,
                )
            )
            print("  - pearson_hypothesis")
            pearson_edges.extend(
                build_pearson_hypothesis_edges(
                    df,
                    id_map,
                    dom,
                    threshold=args.pearson_thresh_hdfs,
                )
            )
            print("  - pc_cpdag_hypothesis")
            pc_edges.extend(
                build_pc_cpdag_hypothesis_edges(
                    df,
                    id_map,
                    dom,
                    alpha=args.pc_alpha_hdfs,
                )
            )
        elif dom == "openstack":
            print(f"[3/4][{position}/{total_domains}] Building OpenStack graphs. Expected: 15-60s")
            print("  - original_dynotears")
            original_edges.extend(
                build_original_dynotears_edges(
                    df,
                    id_map,
                    dom,
                    lambda_w=args.orig_openstack_lambda_w,
                    lambda_a=args.orig_openstack_lambda_a,
                    threshold=args.orig_openstack_threshold,
                    p=args.orig_p,
                    max_iter=args.orig_max_iter,
                )
            )
            print("  - pearson_hypothesis")
            pearson_edges.extend(
                build_pearson_hypothesis_edges(
                    df,
                    id_map,
                    dom,
                    threshold=args.pearson_thresh_openstack,
                )
            )
            print("  - pc_cpdag_hypothesis")
            pc_edges.extend(
                build_pc_cpdag_hypothesis_edges(
                    df,
                    id_map,
                    dom,
                    alpha=args.pc_alpha_openstack,
                )
            )

    print("[4/4] Writing graph artifacts and report. Expected: <5s")
    write_json(GRAPH_FILES["modified"], modified_edges)
    write_json(GRAPH_FILES["original_dynotears"], original_edges)
    write_json(GRAPH_FILES["pearson_hypothesis"], pearson_edges)
    write_json(GRAPH_FILES["pc_cpdag_hypothesis"], pc_edges)

    report_lines = [
        "# RQ2 Fullcase Audit Graph Build (2026-03-18)",
        "",
        f"- Domains: `{domains}`",
        f"- Modified source graph copied from: `{OLD_MODIFIED_GRAPH_PATH}`",
        "- Modified graph was not rebuilt in this phase; it was copied forward unchanged for the selected domains.",
        f"- Vendored DYNOTEARS lag order `p`: `{args.orig_p}`",
        f"- Vendored DYNOTEARS max_iter: `{args.orig_max_iter}`",
        f"- Original HDFS `(lambda_w, lambda_a, threshold)`: `{args.orig_hdfs_lambda_w}, {args.orig_hdfs_lambda_a}, {args.orig_hdfs_threshold}`",
        f"- Original OpenStack `(lambda_w, lambda_a, threshold)`: `{args.orig_openstack_lambda_w}, {args.orig_openstack_lambda_a}, {args.orig_openstack_threshold}`",
        f"- Pearson thresholds `(HDFS/OpenStack)`: `{args.pearson_thresh_hdfs}/{args.pearson_thresh_openstack}`",
        f"- PC alphas `(HDFS/OpenStack)`: `{args.pc_alpha_hdfs}/{args.pc_alpha_openstack}`",
        "",
    ]
    for name, edges in [
        ("modified", modified_edges),
        ("original_dynotears", original_edges),
        ("pearson_hypothesis", pearson_edges),
        ("pc_cpdag_hypothesis", pc_edges),
    ]:
        report_lines.append(f"## {name}")
        report_lines.append("")
        stats = relation_stats(edges)
        for dom in ("hdfs", "openstack"):
            report_lines.append(f"- {dom}: `{stats[dom]['edges']}` edges | relations={stats[dom]['relations']}")
        report_lines.append("")
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    for name, path in GRAPH_FILES.items():
        print(f"[Saved] {name}: {path}")
    print(f"[Saved] {report_path}")


if __name__ == "__main__":
    main()
