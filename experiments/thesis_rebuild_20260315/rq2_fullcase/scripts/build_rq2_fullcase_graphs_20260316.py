from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

from rq2_fullcase_common_20260316 import (
    RESULTS_DIR,
    REPORTS_DIR,
    build_modified_refined_edges,
    build_original_dynotears_edges,
    build_pc_edges,
    build_pearson_edges,
    collect_symbolic_prior_edges,
    ensure_dirs,
    load_hadoop_timeseries,
    load_hdfs_timeseries,
    load_openstack_semantic_timeseries,
    merge_edges_prefer_stronger,
    relation_stats,
    write_json,
)


GRAPH_FILES = {
    "modified": RESULTS_DIR / "gt_causal_knowledge_nesydy_fullcase_20260316.json",
    "original": RESULTS_DIR / "gt_causal_knowledge_dynotears_fullcase_20260316.json",
    "pearson": RESULTS_DIR / "gt_causal_knowledge_pearson_fullcase_20260316.json",
    "pc": RESULTS_DIR / "gt_causal_knowledge_pc_fullcase_20260316.json",
}
CACHE_DIR = RESULTS_DIR / "domain_cache_20260316"


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", default="hdfs,openstack,hadoop")
    ap.add_argument("--pc-alpha-hdfs", type=float, default=0.05)
    ap.add_argument("--pc-alpha-openstack", type=float, default=0.30)
    ap.add_argument("--pc-alpha-hadoop", type=float, default=0.05)
    ap.add_argument("--orig-hdfs-lambda-w", type=float, default=0.025)
    ap.add_argument("--orig-hdfs-lambda-a", type=float, default=0.05)
    ap.add_argument("--orig-hdfs-threshold", type=float, default=0.30)
    ap.add_argument("--orig-openstack-lambda-w", type=float, default=0.030)
    ap.add_argument("--orig-openstack-lambda-a", type=float, default=0.06)
    ap.add_argument("--orig-openstack-threshold", type=float, default=0.40)
    ap.add_argument("--orig-hadoop-lambda-w", type=float, default=0.025)
    ap.add_argument("--orig-hadoop-lambda-a", type=float, default=0.05)
    ap.add_argument("--orig-hadoop-threshold", type=float, default=0.30)
    ap.add_argument("--pc-max-vars-hadoop", type=int, default=96)
    ap.add_argument("--pc-max-vars-openstack", type=int, default=0)
    ap.add_argument("--pearson-thresh-hdfs", type=float, default=0.30)
    ap.add_argument("--pearson-thresh-openstack", type=float, default=0.91)
    ap.add_argument("--pearson-thresh-hadoop", type=float, default=0.93)
    ap.add_argument("--pearson-max-cols-hadoop", type=int, default=0)
    ap.add_argument("--pearson-max-incoming-hdfs", type=int, default=20)
    ap.add_argument("--pearson-max-incoming-openstack", type=int, default=3)
    ap.add_argument("--pearson-max-incoming-hadoop", type=int, default=8)
    ap.add_argument("--openstack-ts", default=str(RESULTS_DIR / "openstack_semantic_timeseries_20260316.csv"))
    ap.add_argument("--openstack-id-map", default=str(RESULTS_DIR / "openstack_semantic_id_map_20260316.json"))
    ap.add_argument("--force-domains", default="")
    ap.add_argument("--force-graphs", default="")
    return ap.parse_args()


def _load_domain_data(args: argparse.Namespace, domains: List[str]) -> Dict[str, Tuple[object, object]]:
    data: Dict[str, Tuple[object, object]] = {}
    for dom in domains:
        if dom == "hdfs":
            data[dom] = load_hdfs_timeseries()
        elif dom == "openstack":
            data[dom] = load_openstack_semantic_timeseries(Path(args.openstack_ts), Path(args.openstack_id_map))
        elif dom == "hadoop":
            data[dom] = load_hadoop_timeseries()
        else:
            raise ValueError(f"Unsupported domain: {dom}")
    return data


def _cache_path(graph_name: str, domain: str) -> Path:
    return CACHE_DIR / f"{domain}_{graph_name}_edges_20260316.json"


def _load_cached_domain_edges(graph_name: str, domain: str) -> List[Dict[str, object]] | None:
    path = _cache_path(graph_name, domain)
    if not path.exists():
        return None
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def _save_cached_domain_edges(graph_name: str, domain: str, edges: List[Dict[str, object]]) -> None:
    write_json(_cache_path(graph_name, domain), edges)


def main() -> None:
    args = _parse_args()
    ensure_dirs()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    domains = [x.strip().lower() for x in args.domains.split(",") if x.strip()]
    force_domains = {x.strip().lower() for x in args.force_domains.split(",") if x.strip()}
    force_graphs = {x.strip().lower() for x in args.force_graphs.split(",") if x.strip()}
    data = _load_domain_data(args, domains)

    modified_edges: List[Dict[str, object]] = []
    original_edges: List[Dict[str, object]] = []
    pearson_edges: List[Dict[str, object]] = []
    pc_edges: List[Dict[str, object]] = []

    for dom in domains:
        cached = {}
        for graph_name in GRAPH_FILES:
            cached[graph_name] = _load_cached_domain_edges(graph_name, dom)
        needs_rebuild = {
            graph_name: (
                cached.get(graph_name) is None
                or (dom in force_domains and (not force_graphs or graph_name in force_graphs))
            )
            for graph_name in GRAPH_FILES
        }
        if not any(needs_rebuild.values()):
            print(f"[*] Reusing cached graphs for {dom}")
            modified_edges.extend(cached["modified"] or [])
            original_edges.extend(cached["original"] or [])
            pearson_edges.extend(cached["pearson"] or [])
            pc_edges.extend(cached["pc"] or [])
        else:
            df, id_map = data[dom]
            if needs_rebuild["original"]:
                print(f"[*] Building original DYNOTEARS for {dom}")
                dom_original = build_original_dynotears_edges(
                    df,
                    id_map,
                    dom,
                    hdfs_lambda_w=args.orig_hdfs_lambda_w,
                    hdfs_lambda_a=args.orig_hdfs_lambda_a,
                    hdfs_threshold=args.orig_hdfs_threshold,
                    openstack_lambda_w=args.orig_openstack_lambda_w,
                    openstack_lambda_a=args.orig_openstack_lambda_a,
                    openstack_threshold=args.orig_openstack_threshold,
                    hadoop_lambda_w=args.orig_hadoop_lambda_w,
                    hadoop_lambda_a=args.orig_hadoop_lambda_a,
                    hadoop_threshold=args.orig_hadoop_threshold,
                )
                _save_cached_domain_edges("original", dom, dom_original)
            else:
                print(f"[*] Reusing cached original DYNOTEARS for {dom}")
                dom_original = cached["original"] or []
            original_edges.extend(dom_original)

            if needs_rebuild["pearson"]:
                print(f"[*] Building Pearson graph for {dom}")
                pearson_thresh = args.pearson_thresh_hdfs
                pearson_max_cols = 0
                pearson_max_incoming = args.pearson_max_incoming_hdfs
                if dom == "openstack":
                    pearson_thresh = args.pearson_thresh_openstack
                    pearson_max_incoming = args.pearson_max_incoming_openstack
                elif dom == "hadoop":
                    pearson_thresh = args.pearson_thresh_hadoop
                    pearson_max_cols = args.pearson_max_cols_hadoop
                    pearson_max_incoming = args.pearson_max_incoming_hadoop
                drop_duplicate_cols = (dom != "hadoop")
                merge_duplicates = (dom != "hadoop")
                if dom == "hdfs":
                    # Keep the raw transfer-chain channels separate for HDFS so
                    # the Pearson baseline does not collapse into an unrealistically
                    # tiny template graph.
                    drop_duplicate_cols = False
                    merge_duplicates = False
                pearson_df = df
                if pearson_max_cols > 0 and dom == "hadoop":
                    from rq2_fullcase_common_20260316 import _trim_top_variance_columns  # type: ignore

                    pearson_df = _trim_top_variance_columns(df, pearson_max_cols)
                dom_pearson = build_pearson_edges(
                    pearson_df,
                    id_map,
                    dom,
                    thresh=pearson_thresh,
                    max_incoming_per_target=pearson_max_incoming,
                    drop_duplicate_cols=drop_duplicate_cols,
                    require_recognized_source=(dom == "hadoop"),
                    merge_duplicates=merge_duplicates,
                )
                _save_cached_domain_edges("pearson", dom, dom_pearson)
            else:
                print(f"[*] Reusing cached Pearson graph for {dom}")
                dom_pearson = cached["pearson"] or []
            pearson_edges.extend(dom_pearson)

            # Save early artifacts before the more fragile PC stage.
            write_json(GRAPH_FILES["modified"], merge_edges_prefer_stronger(modified_edges + collect_symbolic_prior_edges(domains)))
            write_json(GRAPH_FILES["original"], original_edges)
            write_json(GRAPH_FILES["pearson"], pearson_edges)

            if needs_rebuild["pc"]:
                pc_max_vars = 0
                pc_alpha = args.pc_alpha_hdfs
                if dom == "hadoop":
                    pc_max_vars = args.pc_max_vars_hadoop
                    pc_alpha = args.pc_alpha_hadoop
                elif dom == "openstack":
                    pc_max_vars = args.pc_max_vars_openstack
                    pc_alpha = args.pc_alpha_openstack
                print(f"[*] Building PC graph for {dom} (alpha={pc_alpha}, max_vars={pc_max_vars})")
                dom_pc = build_pc_edges(df, id_map, dom, alpha=pc_alpha, max_vars=pc_max_vars)
                _save_cached_domain_edges("pc", dom, dom_pc)
            else:
                print(f"[*] Reusing cached PC graph for {dom}")
                dom_pc = cached["pc"] or []
            pc_edges.extend(dom_pc)

            if needs_rebuild["modified"]:
                print(f"[*] Building refined modified graph for {dom}")
                dom_modified = build_modified_refined_edges(dom_original, collect_symbolic_prior_edges([dom]), dom)
                _save_cached_domain_edges("modified", dom, dom_modified)
            else:
                print(f"[*] Reusing cached refined modified graph for {dom}")
                dom_modified = cached["modified"] or []
            modified_edges.extend(dom_modified)

    symbolic_priors = collect_symbolic_prior_edges(domains)
    modified_edges = merge_edges_prefer_stronger(modified_edges)

    write_json(GRAPH_FILES["modified"], modified_edges)
    write_json(GRAPH_FILES["original"], original_edges)
    write_json(GRAPH_FILES["pearson"], pearson_edges)
    write_json(GRAPH_FILES["pc"], pc_edges)

    report_lines = [
        "# RQ2 Fullcase Graph Build (2026-03-16)",
        "",
        f"- Domains: `{domains}`",
        f"- OpenStack semantic TS: `{args.openstack_ts}`",
        f"- Original DYNOTEARS params HDFS `(lambda_w, lambda_a, thr)`: `{args.orig_hdfs_lambda_w}, {args.orig_hdfs_lambda_a}, {args.orig_hdfs_threshold}`",
        f"- Original DYNOTEARS params OpenStack `(lambda_w, lambda_a, thr)`: `{args.orig_openstack_lambda_w}, {args.orig_openstack_lambda_a}, {args.orig_openstack_threshold}`",
        f"- Original DYNOTEARS params Hadoop `(lambda_w, lambda_a, thr)`: `{args.orig_hadoop_lambda_w}, {args.orig_hadoop_lambda_a}, {args.orig_hadoop_threshold}`",
        f"- PC alphas (HDFS/OpenStack/Hadoop): `{args.pc_alpha_hdfs}/{args.pc_alpha_openstack}/{args.pc_alpha_hadoop}`",
        f"- PC max vars (OpenStack/Hadoop): `{args.pc_max_vars_openstack}/{args.pc_max_vars_hadoop}`",
        f"- Pearson thresholds (HDFS/OpenStack/Hadoop): `{args.pearson_thresh_hdfs}/{args.pearson_thresh_openstack}/{args.pearson_thresh_hadoop}`",
        f"- Pearson max cols (Hadoop): `{args.pearson_max_cols_hadoop}`",
        f"- Pearson max incoming per target (HDFS/OpenStack/Hadoop): `{args.pearson_max_incoming_hdfs}/{args.pearson_max_incoming_openstack}/{args.pearson_max_incoming_hadoop}`",
        f"- Force domains: `{sorted(force_domains)}`",
        f"- Force graphs: `{sorted(force_graphs)}`",
        "",
    ]
    for name, edges in [
        ("modified", modified_edges),
        ("original", original_edges),
        ("pearson", pearson_edges),
        ("pc", pc_edges),
    ]:
        report_lines.append(f"## {name}")
        report_lines.append("")
        stats = relation_stats(edges)
        for dom in ("hdfs", "openstack", "hadoop"):
            report_lines.append(f"- {dom}: `{stats[dom]['edges']}` edges | relations={stats[dom]['relations']}")
        report_lines.append("")

    report_path = REPORTS_DIR / "rq2_fullcase_graph_build_20260316.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    for name, path in GRAPH_FILES.items():
        print(f"[Saved] {name}: {path}")
    print(f"[Saved] {report_path}")


if __name__ == "__main__":
    main()
