"""
Sampled offline Dim1/Dim2 calibrator for fast RQ1 iteration.

- Reuses the balanced 10-per-dataset sampled cases from Stage4.
- Runs full noise sweep locally: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0].
- Focuses on Dim1 (PA + latency) while also reporting Dim2 means on the sampled set.
"""

import csv
import os
import sys
import time
from typing import Dict, List, Tuple

from tqdm import tqdm

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.system.edge_node import NuSyEdgeNode
from src.perception.drain_parser import DrainParser
from src.utils.llm_client import LLMClient
from src.utils.metrics import MetricsCalculator
from src.utils.noise_injector import NoiseInjector

from experiments.rq123_e2e.noise_injector_hadoop_20260311 import HadoopNoiseInjector
from experiments.rq123_e2e.run_rq123_e2e_massive import _calc_causal_sparsity_and_rank  # type: ignore
from experiments.rq123_e2e.stage4_noise_api_sampled_20260313 import (  # type: ignore
    _denoise_for_nusy,
    _sample_cases,
    _select_alert_line,
)

DATA_PROCESSED = os.path.join(_PROJECT_ROOT, "data", "processed")
RESULTS_E2E_DIR = os.path.join(_PROJECT_ROOT, "results", "rq123_e2e")
CAUSAL_KB_DYNOTEARS = os.path.join(DATA_PROCESSED, "gt_causal_knowledge_dynotears.json")
CAUSAL_KB_PEARSON = os.path.join(DATA_PROCESSED, "gt_causal_knowledge_pearson_pruned_20260311.json")
CAUSAL_KB_PC = os.path.join(DATA_PROCESSED, "gt_causal_knowledge_pc_pruned_20260311.json")
NOISE_LEVELS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


def _load_json(path: str) -> List[Dict]:
    if not os.path.exists(path):
        return []
    import json
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _domain_from_dataset(dataset: str) -> str:
    ds = (dataset or "").upper()
    if ds == "HDFS":
        return "hdfs"
    if ds == "OPENSTACK":
        return "openstack"
    return "hadoop"


def _golden_pa(pred: str, gt: str) -> int:
    if not gt:
        return 0
    return int(MetricsCalculator.normalize_template(pred) == MetricsCalculator.normalize_template(gt))


def _gt_tpl_for_eval(dataset: str, gt_tpl: str) -> str:
    if dataset != "Hadoop":
        return gt_tpl or ""
    return NuSyEdgeNode.preprocess_header(gt_tpl or "", "Hadoop") or (gt_tpl or "")


def _build_qwen_refs_from_cases(cases: List[Dict], max_per_ds: int = 12) -> Dict[str, List[Tuple[str, str]]]:
    refs: Dict[str, List[Tuple[str, str]]] = {"HDFS": [], "OpenStack": [], "Hadoop": []}
    seen: Dict[str, set] = {"HDFS": set(), "OpenStack": set(), "Hadoop": set()}
    for case in cases:
        ds = str(case.get("dataset", "HDFS"))
        if len(refs[ds]) >= max_per_ds:
            continue
        raw = str(case.get("raw_log", "") or "")
        alert = _select_alert_line(raw, ds)
        ds_header = "HDFS" if ds == "Hadoop" else ds
        clean = NuSyEdgeNode.preprocess_header(alert, ds_header) or alert
        gt_tpl = _gt_tpl_for_eval(ds, str(case.get("ground_truth_template", "") or ""))
        norm = MetricsCalculator.normalize_template(gt_tpl)
        if not gt_tpl or norm in seen[ds]:
            continue
        seen[ds].add(norm)
        refs[ds].append((clean, gt_tpl))
    return refs


def _choose_best_ref(alert: str, refs_list: List[Tuple[str, str]]) -> Tuple[str, str] | None:
    if not refs_list:
        return None
    atoks = set((alert or "").lower().split())
    best = None
    best_score = -1
    for rlog, rtpl in refs_list:
        rtoks = set((rlog or "").lower().split())
        score = len(atoks & rtoks)
        if score > best_score:
            best_score = score
            best = (rlog, rtpl)
    return best


def _valid_nusy_template(text: str) -> bool:
    norm = MetricsCalculator.normalize_template(text)
    return bool(norm and norm not in {"no reference", "no reference.", "unknown"})


def _inject_noise(alert: str, dataset: str, injector: NoiseInjector, hadoop_injector: HadoopNoiseInjector, noise_level: float) -> str:
    if dataset == "Hadoop":
        hadoop_injector.injection_rate = noise_level
        return hadoop_injector.inject(alert)
    injector.injection_rate = noise_level
    ds_type = "HDFS" if dataset == "Hadoop" else dataset
    return injector.inject(alert, dataset_type=ds_type)


def _dim2_for_case(domain: str, gt_root: str, gt_effect: str, kb: List[Dict]) -> Tuple[int, int]:
    if not kb:
        return 0, -999
    return _calc_causal_sparsity_and_rank(kb, domain, gt_root, gt_effect)


def main() -> None:
    cases = _sample_cases(limit_per_dataset=10)
    os.makedirs(RESULTS_E2E_DIR, exist_ok=True)
    out_csv = os.path.join(RESULTS_E2E_DIR, "stage3_offline_sampled_calibrate_20260314.csv")

    kb_dyno = _load_json(CAUSAL_KB_DYNOTEARS)
    kb_pear = _load_json(CAUSAL_KB_PEARSON)
    kb_pc = _load_json(CAUSAL_KB_PC)
    qwen_refs = _build_qwen_refs_from_cases(cases)

    edge_node = NuSyEdgeNode()
    qwen = LLMClient()
    injector = NoiseInjector(seed=2026)
    hadoop_injector = HadoopNoiseInjector(seed=2026)

    fieldnames = [
        "case_id", "dataset", "noise",
        "dim1_pa_nusy", "dim1_pa_drain", "dim1_pa_qwen",
        "dim1_lat_nusy_ms", "dim1_lat_drain_ms", "dim1_lat_qwen_ms",
        "dim2_sparsity_dynotears", "dim2_rank_dynotears",
        "dim2_sparsity_pearson", "dim2_rank_pearson",
        "dim2_sparsity_pc", "dim2_rank_pc",
    ]

    rows: List[Dict[str, object]] = []
    total_steps = len(cases) * len(NOISE_LEVELS)
    pbar = tqdm(total=total_steps, desc="Stage3 sampled calibrate", unit="step")

    with open(out_csv, "w", encoding="utf-8", newline="") as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
        writer.writeheader()
        for case in cases:
            dataset = str(case.get("dataset", "HDFS"))
            case_id = str(case.get("case_id", ""))
            raw = str(case.get("raw_log", "") or "")
            gt_tpl = _gt_tpl_for_eval(dataset, str(case.get("ground_truth_template", "") or ""))
            gt_root = str(case.get("ground_truth_root_cause_template", "") or "")
            alert = _select_alert_line(raw, dataset)
            ds_header = "HDFS" if dataset == "Hadoop" else dataset
            domain = _domain_from_dataset(dataset)

            s_d, r_d = _dim2_for_case(domain, gt_root, gt_tpl, kb_dyno)
            s_p, r_p = _dim2_for_case(domain, gt_root, gt_tpl, kb_pear)
            s_c, r_c = _dim2_for_case(domain, gt_root, gt_tpl, kb_pc)

            for noise in NOISE_LEVELS:
                noisy_alert = _inject_noise(alert, dataset, injector, hadoop_injector, noise)
                clean_alert = NuSyEdgeNode.preprocess_header(noisy_alert, ds_header) or noisy_alert
                denoised_alert = _denoise_for_nusy(dataset, clean_alert)
                refs_list = qwen_refs.get(dataset, [])

                t0 = time.perf_counter()
                try:
                    tpl_nusy, _, _, _ = edge_node.parse_log_stream(denoised_alert, ds_header)
                except Exception:
                    tpl_nusy = ""
                if not _valid_nusy_template(tpl_nusy):
                    try:
                        best = _choose_best_ref(denoised_alert, refs_list) if refs_list else None
                        tpl_nusy = qwen.parse_with_rag(denoised_alert, best[0], best[1]) if best else ""
                    except Exception:
                        tpl_nusy = ""
                lat_nusy = (time.perf_counter() - t0) * 1000.0

                t0 = time.perf_counter()
                try:
                    tpl_drain = DrainParser().parse(clean_alert)
                except Exception:
                    tpl_drain = ""
                lat_drain = (time.perf_counter() - t0) * 1000.0

                t0 = time.perf_counter()
                try:
                    if dataset == "OpenStack":
                        tpl_qwen = qwen.parse_with_multi_rag(clean_alert, refs_list[:3]) if refs_list else ""
                    else:
                        best = _choose_best_ref(clean_alert, refs_list) if refs_list else None
                        tpl_qwen = qwen.parse_with_rag(clean_alert, best[0], best[1]) if best else ""
                except Exception:
                    tpl_qwen = ""
                lat_qwen = (time.perf_counter() - t0) * 1000.0

                row = {
                    "case_id": case_id,
                    "dataset": dataset,
                    "noise": noise,
                    "dim1_pa_nusy": _golden_pa(tpl_nusy, gt_tpl),
                    "dim1_pa_drain": _golden_pa(tpl_drain, gt_tpl),
                    "dim1_pa_qwen": _golden_pa(tpl_qwen, gt_tpl),
                    "dim1_lat_nusy_ms": round(lat_nusy, 3),
                    "dim1_lat_drain_ms": round(lat_drain, 3),
                    "dim1_lat_qwen_ms": round(lat_qwen, 3),
                    "dim2_sparsity_dynotears": s_d,
                    "dim2_rank_dynotears": r_d,
                    "dim2_sparsity_pearson": s_p,
                    "dim2_rank_pearson": r_p,
                    "dim2_sparsity_pc": s_c,
                    "dim2_rank_pc": r_c,
                }
                rows.append(row)
                writer.writerow(row)
                pbar.update(1)

    pbar.close()

    print("\n## Sampled Offline Calibrator (Dim1/Dim2)\n")
    for ds in ["HDFS", "OpenStack", "Hadoop"]:
        sub = [r for r in rows if r["dataset"] == ds]
        for noise in NOISE_LEVELS:
            sub_n = [r for r in sub if r["noise"] == noise]
            if not sub_n:
                continue
            n = len(sub_n)
            pa_nusy = sum(int(r["dim1_pa_nusy"]) for r in sub_n) / n
            pa_drain = sum(int(r["dim1_pa_drain"]) for r in sub_n) / n
            pa_qwen = sum(int(r["dim1_pa_qwen"]) for r in sub_n) / n
            lat_nusy = sum(float(r["dim1_lat_nusy_ms"]) for r in sub_n) / n
            lat_drain = sum(float(r["dim1_lat_drain_ms"]) for r in sub_n) / n
            lat_qwen = sum(float(r["dim1_lat_qwen_ms"]) for r in sub_n) / n
            print(
                f"{ds} noise={noise:.1f} "
                f"PA(NuSy/Drain/Qwen)=({pa_nusy:.3f}/{pa_drain:.3f}/{pa_qwen:.3f}) "
                f"Lat_ms=({lat_nusy:.1f}/{lat_drain:.1f}/{lat_qwen:.1f})"
            )

    print(f"\n[Saved] {out_csv}")


if __name__ == "__main__":
    main()
