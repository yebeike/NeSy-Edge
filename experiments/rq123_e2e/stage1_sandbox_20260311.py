"""
Stage 1: Micro-case Offline Sandbox for Dim1 (Parsing) on RQ123 benchmark.

Goal:
- For each dataset (HDFS, OpenStack, Hadoop), pick 3 representative cases.
- For each noise level in [0.0, 0.5, 1.0], run:
  - Drain baseline
  - NuSy-Edge (NuSyEdgeNode)
  - Qwen baseline (few-shot via parse_with_multi_rag)
- Compute BOTH:
  - Golden PA (MetricsCalculator.calculate_pa with normalize_template)
  - Dim1 fuzzy match (reuse _fuzzy_template_match_dim1 from modular pipeline)
- Print compact, structured traces so we can visually verify:
  - 0-noise: Drain behaves as a strong baseline; NuSy not obviously broken;
  - noisy: NeSy is not fake-losing, Drain is not fake-winning due to metric bugs.

This script is read/write only inside experiments/, does NOT modify src/ code or data files.
"""

import os
import sys
import json
from typing import Dict, List, Tuple

from tqdm import tqdm

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.system.edge_node import NuSyEdgeNode
from src.perception.drain_parser import DrainParser
from src.utils.noise_injector import NoiseInjector
from src.utils.metrics import MetricsCalculator
from src.utils.llm_client import LLMClient
from experiments.rq123_e2e.run_rq123_e2e_modular import (  # type: ignore
    _load_benchmark,
    _fuzzy_template_match_dim1,
)


DATA_PROCESSED = os.path.join(_PROJECT_ROOT, "data", "processed")
BENCH_V2_PATH = os.path.join(DATA_PROCESSED, "e2e_scaled_benchmark_v2.json")

NOISE_LEVELS = [0.0, 0.5, 1.0]


def _select_three_per_dataset(cases: List[Dict[str, object]]) -> List[Dict[str, object]]:
    """Pick up to 3 cases per dataset (HDFS, OpenStack, Hadoop)."""
    buckets: Dict[str, List[Dict[str, object]]] = {"HDFS": [], "OpenStack": [], "Hadoop": []}
    for c in cases:
        ds = str(c.get("dataset", "HDFS"))
        if ds in buckets and len(buckets[ds]) < 3:
            buckets[ds].append(c)
    out: List[Dict[str, object]] = []
    for ds in ["HDFS", "OpenStack", "Hadoop"]:
        out.extend(buckets.get(ds, []))
    return out


def _build_qwen_refs(cases: List[Dict[str, object]], max_refs_per_ds: int = 3) -> Dict[str, List[Tuple[str, str]]]:
    """
    Build simple few-shot references for Qwen baseline from benchmark GT:
    For each dataset, collect up to max_refs_per_ds (raw_tail, gt_template) pairs.
    """
    refs: Dict[str, List[Tuple[str, str]]] = {"HDFS": [], "OpenStack": [], "Hadoop": []}
    for c in cases:
        ds = str(c.get("dataset", "HDFS"))
        if ds not in refs:
            continue
        if len(refs[ds]) >= max_refs_per_ds:
            continue
        raw = str(c.get("raw_log", "") or "")
        gt_tpl = str(c.get("ground_truth_template", "") or "")
        if not gt_tpl:
            continue
        lines = [l for l in raw.split("\n") if l.strip()]
        tail = lines[-1] if lines else raw
        refs[ds].append((tail, gt_tpl))
    return refs


def _golden_pa(pred: str, gt: str) -> int:
    """Compute golden PA (strict) for a single pair using MetricsCalculator."""
    if not gt:
        return 0
    p_norm = MetricsCalculator.normalize_template(pred)
    g_norm = MetricsCalculator.normalize_template(gt)
    return 1 if p_norm and p_norm == g_norm else 0


def _inject_noise(alert: str, dataset: str, injector: NoiseInjector, noise_level: float) -> str:
    """
    Apply textual noise to the alert line using existing NoiseInjector.
    Hadoop reuses HDFS rules for now (design-consistent with prior robustness scripts).
    """
    injector.injection_rate = noise_level
    ds_for_noise = "HDFS" if dataset == "Hadoop" else dataset
    return injector.inject(alert, dataset_type=ds_for_noise)


def run_stage1_sandbox() -> None:
    cases = _load_benchmark(BENCH_V2_PATH)
    micro_cases = _select_three_per_dataset(cases)
    if not micro_cases:
        print("[ERROR] No cases found in benchmark for Stage1 sandbox.")
        return

    # Few-shot refs for Qwen baseline (per dataset)
    qwen_refs = _build_qwen_refs(cases)

    edge_node = NuSyEdgeNode()
    drain = DrainParser()
    qwen = LLMClient()
    injector = NoiseInjector(seed=2026)

    total_iters = len(micro_cases) * len(NOISE_LEVELS)
    pbar = tqdm(total=total_iters, desc="Stage1 Sandbox (Dim1)", unit="step")

    for case in micro_cases:
        raw = str(case.get("raw_log", "") or "")
        dataset = str(case.get("dataset", "HDFS"))
        case_id = str(case.get("case_id", "") or "")
        gt_tpl = str(case.get("ground_truth_template", "") or "")

        lines = [l for l in raw.split("\n") if l.strip()]
        clean_tail = lines[-1] if lines else raw

        for nl in NOISE_LEVELS:
            # 1) build noisy alert
            noisy_alert = _inject_noise(clean_tail, dataset, injector, nl)
            ds_for_header = "HDFS" if dataset == "Hadoop" else dataset
            clean_for_parse = NuSyEdgeNode.preprocess_header(noisy_alert, ds_for_header) or noisy_alert

            # 2) Drain parse
            try:
                drain_tpl = drain.parse(clean_for_parse)
            except Exception as e:  # noqa: BLE001
                drain_tpl = f"(drain_error: {e})"

            # 3) NeSy parse
            try:
                nusy_tpl, _, _, _ = edge_node.parse_log_stream(noisy_alert, ds_for_header)
            except Exception as e:  # noqa: BLE001
                nusy_tpl = f"(nusy_error: {e})"

            # 4) Qwen baseline with few-shot refs (per dataset)
            refs_list = qwen_refs.get(dataset, [])
            try:
                qwen_tpl = qwen.parse_with_multi_rag(clean_for_parse, refs_list)
            except Exception as e:  # noqa: BLE001
                qwen_tpl = f"(qwen_error: {e})"

            # 5) Golden PA
            pa_drain = _golden_pa(drain_tpl, gt_tpl)
            pa_nusy = _golden_pa(nusy_tpl, gt_tpl)
            pa_qwen = _golden_pa(qwen_tpl, gt_tpl)

            # 6) Dim1 fuzzy (for comparison only)
            fuzzy_drain = _fuzzy_template_match_dim1(dataset, drain_tpl, gt_tpl) if gt_tpl else False
            fuzzy_nusy = _fuzzy_template_match_dim1(dataset, nusy_tpl, gt_tpl) if gt_tpl else False
            fuzzy_qwen = _fuzzy_template_match_dim1(dataset, qwen_tpl, gt_tpl) if gt_tpl else False

            # 7) Structured trace output
            print("=" * 70)
            print(f"CASE ID: {case_id} | Dataset: {dataset} | Noise: {nl}")
            print("[GT TEMPLATE]")
            print(gt_tpl)
            print("[RAW TAIL]")
            print(clean_tail)
            print("[NOISY ALERT]")
            print(noisy_alert)
            print("[DRAIN]")
            print(f"  Pred: {drain_tpl}")
            print(f"  Golden PA: {pa_drain} | Dim1 fuzzy: {int(fuzzy_drain)}")
            print("[NUSY]")
            print(f"  Pred: {nusy_tpl}")
            print(f"  Golden PA: {pa_nusy} | Dim1 fuzzy: {int(fuzzy_nusy)}")
            print("[QWEN]")
            print(f"  Pred: {qwen_tpl}")
            print(f"  Golden PA: {pa_qwen} | Dim1 fuzzy: {int(fuzzy_qwen)}")
            print("=" * 70)

            pbar.update(1)

    pbar.close()
    print("\n[STAGE1] Micro-case sandbox completed for 3 datasets × 3 cases × 3 noise levels.")


if __name__ == "__main__":
    run_stage1_sandbox()

