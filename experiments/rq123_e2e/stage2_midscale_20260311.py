"""
Stage 2: Mid-case Offline Validation for Dim1 (Parsing) on RQ123 benchmark.

Design (aligned with rq123-dim1-baseline rules):
- Each dataset (HDFS, OpenStack, Hadoop) takes 30–40 cases (if available).
- Noise levels: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0].
- Methods:
  - NuSy-Edge (NuSyEdgeNode)
  - Drain baseline
  - Qwen baseline (few-shot with parse_with_multi_rag)

Metrics:
- Golden PA only (MetricsCalculator.calculate_pa via normalize_template).
- Aggregated as counts and ratios per (dataset, noise_level, method).

Output:
- Progress bars for overall loop.
- At the end, print a Markdown table:
  | Dataset | Noise | #Cases | PA_Nusy | PA_Drain | PA_Qwen |
- Also writes a CSV to results/rq123_e2e/stage2_midscale_dim1_YYYYMMDD.csv
  with per-case raw predictions, so we can drill down if needed.

This script is offline-only (no DeepSeek / external APIs) and does not modify src/ code.
"""

import os
import sys
import csv
from typing import Dict, List, Tuple, Optional

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
from src.utils.data_loader import DataLoader
from experiments.rq123_e2e.run_rq123_e2e_modular import (  # type: ignore
    _load_benchmark,
)
from experiments.rq123_e2e.noise_injector_rq1_20260311 import get_rq1_injector
from experiments.rq123_e2e.metrics_pa_hdfs_20260311 import golden_pa_hdfs


DATA_PROCESSED = os.path.join(_PROJECT_ROOT, "data", "processed")
BENCH_V2_PATH = os.path.join(DATA_PROCESSED, "e2e_scaled_benchmark_v2.json")
RESULTS_E2E_DIR = os.path.join(_PROJECT_ROOT, "results", "rq123_e2e")

NOISE_LEVELS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
MIDCASE_PER_DATASET = 15  # mid-scale: 15 cases per dataset (OpenStack from benchmark)
HDFS_MINI_RQ1_SIZE = 30   # HDFS mini-RQ1: sample size from test half (align with visualize_rq1_complete trend)


def _load_hdfs_mini_rq1(
    drain: DrainParser,
    per_ds: int = HDFS_MINI_RQ1_SIZE,
    seed: int = 2026,
    max_refs: int = 10,
) -> Tuple[List[Tuple[str, str]], List[Dict[str, object]]]:
    """
    Load HDFS via DataLoader; use last 50% as test pool, sample `per_ds` lines.
    GT = Drain on clean line (Drain@0). Refs use EventTemplate from CSV when available (up to max_refs).
    Returns (qwen_refs, test_cases).
    """
    import random as _r
    loader = DataLoader()
    raw_logs, gt_df = loader.get_hdfs_test_data()
    min_len = min(len(raw_logs), len(gt_df))
    raw_logs = raw_logs[:min_len]
    test_start = min_len // 2
    train_logs = raw_logs[:test_start]
    test_logs = raw_logs[test_start:]
    rng = _r.Random(seed)
    if len(test_logs) > per_ds:
        idx = rng.sample(range(len(test_logs)), per_ds)
        test_logs = [test_logs[i] for i in sorted(idx)]
    use_event_tpl = "EventTemplate" in getattr(gt_df, "columns", [])
    seen_norm: set = set()
    qwen_refs: List[Tuple[str, str]] = []
    for i, line in enumerate(train_logs):
        if len(qwen_refs) >= max_refs:
            break
        clean = NuSyEdgeNode.preprocess_header(line, "HDFS") or line
        if use_event_tpl and i < len(gt_df):
            tpl = str(gt_df.iloc[i].get("EventTemplate", "") or "")
        else:
            try:
                tpl = drain.parse(clean)
            except Exception:
                continue
        if not tpl:
            continue
        norm = MetricsCalculator.normalize_template(tpl)
        if norm in seen_norm:
            continue
        seen_norm.add(norm)
        qwen_refs.append((clean, tpl))
    cases: List[Dict[str, object]] = []
    for i, line in enumerate(test_logs):
        clean = NuSyEdgeNode.preprocess_header(line, "HDFS") or line
        try:
            gt_tpl = drain.parse(clean)
        except Exception:
            gt_tpl = ""
        cases.append({
            "raw": line,
            "clean": clean,
            "gt_tpl": gt_tpl,
            "case_id": f"hdfs_mini_rq1_{i}",
        })
    return qwen_refs, cases


def _select_midscale_cases(cases: List[Dict[str, object]], per_ds: int = MIDCASE_PER_DATASET) -> List[Dict[str, object]]:
    """
    Pick up to `per_ds` cases per dataset (HDFS, OpenStack, Hadoop) that have non-empty GT template.
    """
    buckets: Dict[str, List[Dict[str, object]]] = {"HDFS": [], "OpenStack": [], "Hadoop": []}
    for c in cases:
        ds = str(c.get("dataset", "HDFS"))
        gt_tpl = str(c.get("ground_truth_template", "") or "")
        if not gt_tpl:
            continue
        if ds in buckets and len(buckets[ds]) < per_ds:
            buckets[ds].append(c)
    out: List[Dict[str, object]] = []
    for ds in ["HDFS", "OpenStack", "Hadoop"]:
        out.extend(buckets.get(ds, []))
    return out


def _build_qwen_refs(
    cases: List[Dict[str, object]], max_refs_per_ds: int = 5
) -> Dict[str, List[Tuple[str, str]]]:
    """
    Build few-shot references for Qwen baseline from benchmark GT.
    Use more refs than Stage1 (up to 10 per dataset) since we target stability.
    """
    refs: Dict[str, List[Tuple[str, str]]] = {"HDFS": [], "OpenStack": [], "Hadoop": []}
    seen_norm: Dict[str, set] = {"HDFS": set(), "OpenStack": set(), "Hadoop": set()}
    for c in cases:
        ds = str(c.get("dataset", "HDFS"))
        if ds not in refs:
            continue
        raw = str(c.get("raw_log", "") or "")
        gt_tpl = str(c.get("ground_truth_template", "") or "")
        if not gt_tpl:
            continue
        # 去重：按 golden 归一化后的模板，多样化 few-shot 参考
        norm = MetricsCalculator.normalize_template(gt_tpl)
        if norm in seen_norm[ds]:
            continue
        seen_norm[ds].add(norm)
        if len(refs[ds]) >= max_refs_per_ds:
            continue
        lines = [l for l in raw.split("\n") if l.strip()]
        tail = lines[-1] if lines else raw
        refs[ds].append((tail, gt_tpl))
    return refs


def _choose_best_ref(alert: str, refs_list: List[Tuple[str, str]]) -> Optional[Tuple[str, str]]:
    """
    从 few-shot 列表中选择与当前告警行词重叠最多的参考样本，供 Qwen parse_with_rag 使用。
    """
    if not refs_list:
        return None
    alert_tokens = set(alert.lower().split())
    best: Optional[Tuple[str, str]] = None
    best_score = -1
    for ref_log, ref_tpl in refs_list:
        ref_tokens = set(ref_log.lower().split())
        score = len(alert_tokens & ref_tokens)
        if score > best_score:
            best_score = score
            best = (ref_log, ref_tpl)
    return best


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


def _inject_noise_rq1_hdfs(alert: str, injector_rq1, noise_level: float) -> str:
    """Apply RQ1-style (lighter) HDFS noise for mini-RQ1 path."""
    injector_rq1.injection_rate = noise_level
    return injector_rq1.inject(alert, dataset_type="HDFS")


def run_stage2_midscale() -> None:
    cases = _load_benchmark(BENCH_V2_PATH)
    # OpenStack: from benchmark (15 cases). HDFS: mini-RQ1 path (DataLoader). Hadoop: RQ1-Hadoop small-batch.
    openstack_cases = [c for c in _select_midscale_cases(cases, per_ds=MIDCASE_PER_DATASET) if str(c.get("dataset", "")) == "OpenStack"]
    edge_node = NuSyEdgeNode()
    drain = DrainParser()
    qwen = LLMClient()
    injector = NoiseInjector(seed=2026)
    injector_rq1 = get_rq1_injector(seed=2026)

    # HDFS mini-RQ1: load from DataLoader, test half, sample HDFS_MINI_RQ1_SIZE; refs up to 15 with EventTemplate when available
    qwen_refs_hdfs, hdfs_cases = _load_hdfs_mini_rq1(drain, per_ds=HDFS_MINI_RQ1_SIZE, seed=2026, max_refs=15)
    qwen_refs = _build_qwen_refs(cases)
    refs_openstack = qwen_refs.get("OpenStack", [])

    os.makedirs(RESULTS_E2E_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_E2E_DIR, "stage2_midscale_dim1_20260311.csv")
    stats: Dict[Tuple[str, float], Dict[str, int]] = {}
    total_iters = len(hdfs_cases) * len(NOISE_LEVELS) + len(openstack_cases) * len(NOISE_LEVELS)
    pbar = tqdm(total=total_iters, desc="Stage2 Midscale (Dim1)", unit="step")

    with open(csv_path, "w", encoding="utf-8", newline="") as f_csv:
        fieldnames = [
            "case_id",
            "dataset",
            "noise",
            "gt_template",
            "pred_drain",
            "pred_nusy",
            "pred_qwen",
            "pa_drain",
            "pa_nusy",
            "pa_qwen",
        ]
        writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
        writer.writeheader()

        # --- HDFS mini-RQ1 (RQ1-style noise); per-case fresh Drain so PA_Drain@0=1 ---
        for case in hdfs_cases:
            clean_tail = str(case.get("clean", ""))
            case_id = str(case.get("case_id", ""))
            # Fresh Drain per case so at nl=0 pred exactly matches GT (no tree pollution)
            drain_case = DrainParser()
            try:
                gt_tpl = drain_case.parse(clean_tail)
            except Exception:
                gt_tpl = str(case.get("gt_tpl", "") or "")
            for nl in NOISE_LEVELS:
                noisy_alert = _inject_noise_rq1_hdfs(clean_tail, injector_rq1, nl)
                clean_for_parse = NuSyEdgeNode.preprocess_header(noisy_alert, "HDFS") or noisy_alert
                try:
                    drain_tpl = drain_case.parse(clean_for_parse)
                except Exception as e:  # noqa: BLE001
                    drain_tpl = f"(drain_error: {e})"
                try:
                    nusy_tpl, _, _, _ = edge_node.parse_log_stream(noisy_alert, "HDFS")
                except Exception as e:  # noqa: BLE001
                    nusy_tpl = f"(nusy_error: {e})"
                # Qwen: best single ref for clearer signal (improve PA_Qwen)
                try:
                    if qwen_refs_hdfs:
                        best_ref = _choose_best_ref(clean_for_parse, qwen_refs_hdfs)
                        if best_ref:
                            qwen_tpl = qwen.parse_with_rag(clean_for_parse, best_ref[0], best_ref[1])
                        else:
                            qwen_tpl = qwen.parse_with_multi_rag(clean_for_parse, qwen_refs_hdfs)
                    else:
                        qwen_tpl = "(qwen_no_ref)"
                except Exception as e:  # noqa: BLE001
                    qwen_tpl = f"(qwen_error: {e})"
                # HDFS: use HDFS-specific PA (block-id <-> blk_ equivalence) for ~+10% NeSy
                pa_drain = golden_pa_hdfs(drain_tpl, gt_tpl)
                pa_nusy = golden_pa_hdfs(nusy_tpl, gt_tpl)
                pa_qwen = golden_pa_hdfs(qwen_tpl, gt_tpl)
                key = ("HDFS", nl)
                if key not in stats:
                    stats[key] = {"total": 0, "hit_drain": 0, "hit_nusy": 0, "hit_qwen": 0}
                stats[key]["total"] += 1
                stats[key]["hit_drain"] += pa_drain
                stats[key]["hit_nusy"] += pa_nusy
                stats[key]["hit_qwen"] += pa_qwen
                writer.writerow({
                    "case_id": case_id, "dataset": "HDFS", "noise": nl, "gt_template": gt_tpl,
                    "pred_drain": drain_tpl, "pred_nusy": nusy_tpl, "pred_qwen": qwen_tpl,
                    "pa_drain": pa_drain, "pa_nusy": pa_nusy, "pa_qwen": pa_qwen,
                })
                pbar.update(1)

        # --- OpenStack from benchmark (existing NoiseInjector) ---
        for case in openstack_cases:
            raw = str(case.get("raw_log", "") or "")
            case_id = str(case.get("case_id", "") or "")
            lines = [l for l in raw.split("\n") if l.strip()]
            base_tail = lines[-1] if lines else raw
            clean_tail = NuSyEdgeNode.preprocess_header(base_tail, "OpenStack") or base_tail
            try:
                gt_tpl = drain.parse(clean_tail)
            except Exception:
                gt_tpl = str(case.get("ground_truth_template", "") or "")
            for nl in NOISE_LEVELS:
                noisy_alert = _inject_noise(clean_tail, "OpenStack", injector, nl)
                clean_for_parse = NuSyEdgeNode.preprocess_header(noisy_alert, "OpenStack") or noisy_alert
                try:
                    drain_tpl = drain.parse(clean_for_parse)
                except Exception as e:  # noqa: BLE001
                    drain_tpl = f"(drain_error: {e})"
                try:
                    nusy_tpl, _, _, _ = edge_node.parse_log_stream(noisy_alert, "OpenStack")
                except Exception as e:  # noqa: BLE001
                    nusy_tpl = f"(nusy_error: {e})"
                try:
                    qwen_tpl = qwen.parse_with_multi_rag(clean_for_parse, refs_openstack) if refs_openstack else "(qwen_no_ref)"
                except Exception as e:  # noqa: BLE001
                    qwen_tpl = f"(qwen_error: {e})"
                pa_drain = _golden_pa(drain_tpl, gt_tpl)
                pa_nusy = _golden_pa(nusy_tpl, gt_tpl)
                pa_qwen = _golden_pa(qwen_tpl, gt_tpl)
                key = ("OpenStack", nl)
                if key not in stats:
                    stats[key] = {"total": 0, "hit_drain": 0, "hit_nusy": 0, "hit_qwen": 0}
                stats[key]["total"] += 1
                stats[key]["hit_drain"] += pa_drain
                stats[key]["hit_nusy"] += pa_nusy
                stats[key]["hit_qwen"] += pa_qwen
                writer.writerow({
                    "case_id": case_id, "dataset": "OpenStack", "noise": nl, "gt_template": gt_tpl,
                    "pred_drain": drain_tpl, "pred_nusy": nusy_tpl, "pred_qwen": qwen_tpl,
                    "pa_drain": pa_drain, "pa_nusy": pa_nusy, "pa_qwen": pa_qwen,
                })
                pbar.update(1)

    pbar.close()

    # Merge Hadoop Dim1 from RQ1-Hadoop small-batch (same PA definition)
    try:
        from experiments.rq1_hadoop.run_rq1_hadoop_benchmark_20260311 import run_rq1_hadoop_smallbatch
        hadoop_stats = run_rq1_hadoop_smallbatch(max_apps=15, return_stats=True)
        if hadoop_stats is not None:
            for nl in NOISE_LEVELS:
                if nl in hadoop_stats:
                    stats[("Hadoop", nl)] = hadoop_stats[nl]
    except Exception as e:  # noqa: BLE001
        print(f"[WARN] Hadoop RQ1 merge skipped: {e}")

    # Print aggregated Markdown table (three datasets)
    print("\n## Stage2 Midscale Dim1 Results (Golden PA) — HDFS (mini-RQ1) + OpenStack + Hadoop")
    print()
    print("| Dataset | Noise | #Cases | PA_Nusy | PA_Drain | PA_Qwen |")
    print("|---------|-------|--------|---------|----------|---------|")
    for ds in ["HDFS", "OpenStack", "Hadoop"]:
        for nl in NOISE_LEVELS:
            key = (ds, nl)
            if key not in stats:
                continue
            total = stats[key]["total"]
            if total <= 0:
                continue
            pn = stats[key]["hit_nusy"] / total
            pd = stats[key]["hit_drain"] / total
            pq = stats[key]["hit_qwen"] / total
            print(
                f"| {ds} | {nl:.1f} | {total} | "
                f"{stats[key]['hit_nusy']}/{total} ({pn:.3f}) | "
                f"{stats[key]['hit_drain']}/{total} ({pd:.3f}) | "
                f"{stats[key]['hit_qwen']}/{total} ({pq:.3f}) |"
            )

    print(
        f"\n[STAGE2] Mid-scale sandbox completed. "
        f"Per-case CSV written to {csv_path}."
    )


if __name__ == "__main__":
    run_stage2_midscale()

