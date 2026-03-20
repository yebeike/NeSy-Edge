from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HUGGINGFACE_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

_SCRIPT_DIR = Path(__file__).resolve().parent
_REBUILD_ROOT = _SCRIPT_DIR.parents[1]
_PROJECT_ROOT = _REBUILD_ROOT.parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.rq123_e2e.noise_hadoop_aggressive_20260311 import get_hadoop_aggressive_injector
from experiments.rq123_e2e.noise_injector_rq1_20260311 import get_rq1_injector
from experiments.thesis_rebuild_20260315.shared.components.drain_baseline import DrainBaseline
from experiments.thesis_rebuild_20260315.shared.evaluators.normalized_pa import exact_match_hit
from experiments.thesis_rebuild_20260315.shared.utils.io_utils import write_csv, write_json
from experiments.thesis_rebuild_20260315.shared.utils.project_paths import MANIFEST_DIR, RQ1_RESULTS_DIR, ensure_dirs
from src.system.edge_node import NuSyEdgeNode
from src.system.knowledge_base import KnowledgeBase
from src.utils.llm_client import LLMClient
from src.utils.noise_injector import NoiseInjector


NOISE_LEVELS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
SHORTCUT_THRESHOLDS = {"HDFS": 0.62, "OpenStack": 0.66, "Hadoop": 0.68}


def _fingerprint(text: str) -> str:
    return "".join("N" if ch.isdigit() else ch for ch in text)


def _inject_noise(dataset: str, clean_alert: str, noise_level: float, generic_injector, hdfs_injector, hadoop_injector) -> str:
    if dataset == "HDFS":
        hdfs_injector.injection_rate = noise_level
        return hdfs_injector.inject(clean_alert, dataset_type="HDFS")
    if dataset == "OpenStack":
        generic_injector.injection_rate = noise_level
        return generic_injector.inject(clean_alert, dataset_type="OpenStack")
    hadoop_injector.injection_rate = noise_level
    return hadoop_injector.inject(clean_alert, dataset_type="HDFS")


def _score_ref(query: str, ref_log: str) -> float:
    query_fp = _fingerprint(query)
    ref_fp = _fingerprint(ref_log)
    seq = SequenceMatcher(None, query_fp, ref_fp).ratio()
    q_tokens = set(query.lower().split())
    r_tokens = set(ref_log.lower().split())
    overlap = len(q_tokens & r_tokens) / max(1, len(q_tokens | r_tokens))
    return 0.7 * seq + 0.3 * overlap


def _choose_top_refs(query: str, refs: List[tuple[str, str]], top_k: int = 3) -> List[tuple[str, str]]:
    if not refs:
        return []
    scored = [(_score_ref(query, ref_log), ref_log, ref_template) for ref_log, ref_template in refs]
    scored.sort(key=lambda item: (item[0], item[2], item[1]), reverse=True)
    return [(ref_log, ref_template) for _, ref_log, ref_template in scored[:top_k]]


def _score_kb_hit(edge_node: NuSyEdgeNode, query: str, ref_log: str) -> float:
    query_fp = edge_node._compute_fingerprint(query)
    ref_fp = edge_node._compute_fingerprint(ref_log)
    seq = edge_node._calculate_similarity(query_fp, ref_fp)
    q_tokens = set(query.lower().split())
    r_tokens = set(ref_log.lower().split())
    overlap = len(q_tokens & r_tokens) / max(1, len(q_tokens | r_tokens))
    return 0.7 * seq + 0.3 * overlap


def _parse_with_nusy_rebuild(edge_node: NuSyEdgeNode, raw_log: str, dataset_type: str) -> tuple[str, float, bool, str]:
    start_t = time.time()
    content = NuSyEdgeNode.preprocess_header(raw_log, dataset_type) or raw_log
    fingerprint = edge_node._compute_fingerprint(content)
    if fingerprint in edge_node.cache:
        latency = (time.time() - start_t) * 1000
        return edge_node.cache[fingerprint], latency, True, "L1_cache"

    template = ""
    route = "llm"
    if edge_node.kb:
        rag_results = edge_node.kb.search(content, dataset_type, top_k=5)
        if rag_results:
            scored = sorted(
                rag_results,
                key=lambda item: _score_kb_hit(edge_node, content, item["raw_log"]),
                reverse=True,
            )
            best_match = scored[0]
            best_score = _score_kb_hit(edge_node, content, best_match["raw_log"])
            if best_score >= SHORTCUT_THRESHOLDS.get(dataset_type, 0.66):
                template = best_match["template"]
                route = "symbolic_shortcut"
            else:
                refs = [(item["raw_log"], item["template"]) for item in scored[:3]]
                if len(refs) == 1:
                    template = edge_node.llm.parse_with_rag(content, refs[0][0], refs[0][1])
                    route = "llm_single_rag"
                else:
                    template = edge_node.llm.parse_with_multi_rag(content, refs)
                    route = "llm_multi_rag"
        else:
            template = edge_node.llm.parse_with_rag(content, "No reference", "No reference")
            route = "llm_cold_start"
    else:
        template = edge_node.llm.parse_with_rag(content, "No reference", "No reference")
        route = "llm_no_kb"

    edge_node.cache[fingerprint] = template
    latency = (time.time() - start_t) * 1000
    return template, latency, False, route


def _load_manifest(name: str) -> dict:
    path = MANIFEST_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}. Run build_rq1_manifest_20260315.py first.")
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-cases-per-dataset", type=int, default=0)
    parser.add_argument("--datasets", type=str, default="")
    parser.add_argument("--noise-levels", type=str, default="")
    parser.add_argument("--run-tag", type=str, default="pilot")
    parser.add_argument("--manifest-name", type=str, default="rq1_manifest_pilot_20260315.json")
    return parser.parse_args()


def main() -> str:
    args = _parse_args()
    ensure_dirs()
    manifest = _load_manifest(args.manifest_name)
    selected = {x.strip() for x in args.datasets.split(",") if x.strip()}
    noise_levels = NOISE_LEVELS
    if args.noise_levels.strip():
        noise_levels = [float(x.strip()) for x in args.noise_levels.split(",") if x.strip()]

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    edge_node = NuSyEdgeNode()
    qwen = LLMClient()
    generic_injector = NoiseInjector(seed=2026)
    hdfs_injector = get_rq1_injector(seed=2026)
    hadoop_injector = get_hadoop_aggressive_injector(seed=2026)

    rows: List[dict] = []
    out_csv = RQ1_RESULTS_DIR / f"rq1_{args.run_tag}_rows_20260315.csv"
    fieldnames = [
        "dataset",
        "case_id",
        "noise",
        "method",
        "gt_template",
        "prediction",
        "pa_hit",
        "latency_ms",
        "gt_source",
        "route",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for dataset, meta in manifest["datasets"].items():
            if selected and dataset not in selected:
                continue
            ref_cases = meta["reference_cases"]
            eval_cases = meta["eval_cases"]
            if args.max_cases_per_dataset > 0:
                eval_cases = eval_cases[: args.max_cases_per_dataset]
            drain = DrainBaseline(reference_logs=[c["clean_alert"] for c in ref_cases])
            qwen_refs = [(c["clean_alert"], c["gt_template"]) for c in ref_cases]
            edge_node.cache = {}
            edge_node.kb = KnowledgeBase(
                collection_name=f"thesis_rebuild_rq1_{args.run_tag}_{dataset.lower()}",
                persist_path=f"experiments/thesis_rebuild_20260315/rq1/results/chroma_{args.run_tag}_{dataset.lower()}",
            )
            if qwen_refs:
                edge_node.kb.add_knowledge(
                    raw_logs=[ref[0] for ref in qwen_refs],
                    templates=[ref[1] for ref in qwen_refs],
                    dataset_type=dataset,
                )
            print(f"[RQ1] dataset={dataset} refs={len(ref_cases)} eval={len(eval_cases)} noise_levels={noise_levels}")

            for case_idx, case in enumerate(eval_cases, start=1):
                print(f"[RQ1] case {case_idx}/{len(eval_cases)} {case['case_id']}")
                for noise in noise_levels:
                    noisy_alert = _inject_noise(
                        dataset=dataset,
                        clean_alert=case["clean_alert"],
                        noise_level=noise,
                        generic_injector=generic_injector,
                        hdfs_injector=hdfs_injector,
                        hadoop_injector=hadoop_injector,
                    )
                    clean_for_parse = NuSyEdgeNode.preprocess_header(noisy_alert, dataset) or noisy_alert

                    t0 = time.perf_counter()
                    pred_drain = drain.parse(clean_for_parse)
                    lat_drain = (time.perf_counter() - t0) * 1000.0

                    t0 = time.perf_counter()
                    pred_nusy, _, _, nusy_route = _parse_with_nusy_rebuild(edge_node, noisy_alert, dataset)
                    lat_nusy = (time.perf_counter() - t0) * 1000.0

                    t0 = time.perf_counter()
                    top_refs = _choose_top_refs(clean_for_parse, qwen_refs, top_k=3)
                    if len(top_refs) == 1:
                        pred_qwen = qwen.parse_with_rag(clean_for_parse, top_refs[0][0], top_refs[0][1])
                    elif len(top_refs) > 1:
                        pred_qwen = qwen.parse_with_multi_rag(clean_for_parse, top_refs)
                    else:
                        pred_qwen = qwen.parse_with_multi_rag(clean_for_parse, qwen_refs)
                    lat_qwen = (time.perf_counter() - t0) * 1000.0

                    gt = case["gt_template"]
                    batch_rows = [
                        {
                            "dataset": dataset,
                            "case_id": case["case_id"],
                            "noise": noise,
                            "method": "Drain",
                            "gt_template": gt,
                            "prediction": pred_drain,
                            "pa_hit": exact_match_hit(pred_drain, gt),
                            "latency_ms": round(lat_drain, 3),
                            "gt_source": case["gt_source"],
                            "route": "drain_replayed_refs",
                        },
                        {
                            "dataset": dataset,
                            "case_id": case["case_id"],
                            "noise": noise,
                            "method": "NuSy",
                            "gt_template": gt,
                            "prediction": pred_nusy,
                            "pa_hit": exact_match_hit(pred_nusy, gt),
                            "latency_ms": round(lat_nusy, 3),
                            "gt_source": case["gt_source"],
                            "route": nusy_route,
                        },
                        {
                            "dataset": dataset,
                            "case_id": case["case_id"],
                            "noise": noise,
                            "method": "Qwen",
                            "gt_template": gt,
                            "prediction": pred_qwen,
                            "pa_hit": exact_match_hit(pred_qwen, gt),
                            "latency_ms": round(lat_qwen, 3),
                            "gt_source": case["gt_source"],
                            "route": "qwen_refs",
                        },
                    ]
                    rows.extend(batch_rows)
                    for row in batch_rows:
                        writer.writerow(row)
                    f.flush()
                    print(
                        f"[RQ1] dataset={dataset} case={case['case_id']} noise={noise:.1f} "
                        f"PA(D/N/Q)={batch_rows[0]['pa_hit']}/{batch_rows[1]['pa_hit']}/{batch_rows[2]['pa_hit']}"
                    )

    summary: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(dict)
    clean_sanity: Dict[str, Dict[str, float]] = defaultdict(dict)

    grouped: Dict[tuple, List[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["noise"], row["method"])].append(row)

    for (dataset, noise, method), part in sorted(grouped.items()):
        pa = sum(r["pa_hit"] for r in part) / len(part)
        lat = sum(r["latency_ms"] for r in part) / len(part)
        summary.setdefault(dataset, {})[f"{noise:.1f}:{method}"] = {
            "cases": len(part),
            "pa": round(pa, 4),
            "latency_ms": round(lat, 3),
        }
        if noise == 0.0:
            clean_sanity.setdefault(dataset, {})[method] = round(pa, 4)

    degenerate_flags = []
    for dataset, methods in clean_sanity.items():
        for method, pa in methods.items():
            if method != "NuSy" and pa <= 0.05:
                degenerate_flags.append({"dataset": dataset, "method": method, "clean_pa": pa})

    out_json = RQ1_RESULTS_DIR / f"rq1_{args.run_tag}_summary_20260315.json"
    write_json(
        out_json,
        {
            "manifest": args.manifest_name,
            "run_tag": args.run_tag,
            "noise_levels": noise_levels,
            "clean_sanity": clean_sanity,
            "degenerate_flags": degenerate_flags,
            "summary": summary,
        },
    )

    print(json.dumps({"clean_sanity": clean_sanity, "degenerate_flags": degenerate_flags}, indent=2))
    print(f"[Saved] {out_csv}")
    print(f"[Saved] {out_json}")
    return str(out_json)


if __name__ == "__main__":
    main()
