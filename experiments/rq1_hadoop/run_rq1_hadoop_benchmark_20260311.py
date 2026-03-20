"""
RQ1-Hadoop: small-batch parsing benchmark for NeSy-Edge vs Drain vs Qwen.

Design:
- Use abnormal applications (non-normal labels) from data/raw/Hadoop/abnormal_label.txt
  as fault cases.
- For each application_id, treat all its log lines as a window and use the last
  non-empty line as the "alert" log for Dim1 parsing.
- Define GT template as Drain's template on the clean (noise=0.0) alert line.
  This aligns GT style with Drain outputs and matches the golden PA definition.
- Methods:
  - NeSy-Edge (NuSyEdgeNode.parse_log_stream)
  - Drain (baseline)
  - Qwen baseline (few-shot via parse_with_multi_rag using a small reference set)
- Metrics:
  - Golden PA only (MetricsCalculator.calculate_pa, implemented per-pair).
  - Report counts/ratios per method, plus a Markdown summary table.

This script is intended for Stage1/Stage2 scale (e.g., 3–20 applications).
"""

import os
import sys
import csv
from typing import Dict, List, Tuple

from tqdm import tqdm

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from experiments.rq123_e2e import hadoop_loader  # type: ignore
from src.system.edge_node import NuSyEdgeNode
from src.perception.drain_parser import DrainParser
from src.utils.llm_client import LLMClient
from src.utils.metrics import MetricsCalculator
from src.utils.noise_injector import NoiseInjector
from src.system.knowledge_base import KnowledgeBase

try:
    from experiments.rq123_e2e.noise_hadoop_aggressive_20260311 import get_hadoop_aggressive_injector
except ImportError:
    get_hadoop_aggressive_injector = None


def _golden_pa(pred: str, gt: str) -> int:
    if not gt:
        return 0
    p_norm = MetricsCalculator.normalize_template(pred)
    g_norm = MetricsCalculator.normalize_template(gt)
    return 1 if p_norm and p_norm == g_norm else 0


def _select_abnormal_cases(
    hadoop_root: str,
    abnormal_label_path: str,
    max_apps: int = 20,
) -> List[Dict[str, object]]:
    rows = hadoop_loader.parse_abnormal_label_file(abnormal_label_path)
    abns: List[Dict[str, object]] = []
    for r in rows:
        lbl = r.get("label", "").lower()
        if lbl == "normal":
            continue
        abns.append(r)
    abns = abns[:max_apps]

    cases: List[Dict[str, object]] = []
    for r in abns:
        app_id = r["application_id"]
        try:
            lines = hadoop_loader.iter_hadoop_application_logs(hadoop_root, app_id)
        except FileNotFoundError:
            continue
        if not lines:
            continue
        non_empty = [l for l in lines if l.strip()]
        alert = non_empty[-1] if non_empty else ""
        if not alert:
            continue
        cases.append(
            {
                "case_id": app_id,
                "application_id": app_id,
                "label": r.get("label", ""),
                "reason": r.get("reason", ""),
                "alert": alert,
            }
        )
    return cases


def _build_qwen_refs_from_cases(
    cases: List[Dict[str, object]],
    drain: DrainParser,
    max_refs: int = 5,
) -> List[Tuple[str, str]]:
    """
    Build few-shot references for Qwen: (raw_alert, drain_template) for up to max_refs cases.
    """
    refs: List[Tuple[str, str]] = []
    for c in cases:
        if len(refs) >= max_refs:
            break
        alert = str(c.get("alert", "") or "")
        clean = NuSyEdgeNode.preprocess_header(alert, "Hadoop") or alert
        try:
            tpl = drain.parse(clean)
        except Exception:
            continue
        if not tpl:
            continue
        refs.append((clean, tpl))
    return refs


def run_rq1_hadoop_smallbatch(
    max_apps: int = 20,
    noise_levels: List[float] = None,
    return_stats: bool = False,
):
    """
    Run RQ1-Hadoop small-batch. If return_stats=True, return the stats dict and do not print table
    (for merging into Stage2 three-dataset Dim1 table).
    """
    if noise_levels is None:
        noise_levels = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    data_raw = os.path.join(_PROJECT_ROOT, "data", "raw")
    hadoop_root = os.path.join(data_raw, "Hadoop")
    abnormal_label_path = os.path.join(hadoop_root, "abnormal_label.txt")
    if not os.path.exists(hadoop_root) or not os.path.exists(abnormal_label_path):
        raise FileNotFoundError(f"Hadoop raw logs or abnormal_label.txt not found under {hadoop_root}")

    cases = _select_abnormal_cases(hadoop_root, abnormal_label_path, max_apps=max_apps)
    if not cases:
        print("[WARN] No abnormal Hadoop applications selected for RQ1 benchmark.")
        return None if return_stats else None

    print(f"[INFO] RQ1-Hadoop small-batch benchmark on {len(cases)} abnormal applications.")

    edge_node = NuSyEdgeNode()
    drain = DrainParser()
    qwen = LLMClient()
    if return_stats and get_hadoop_aggressive_injector is not None:
        injector = get_hadoop_aggressive_injector(seed=2026)
    else:
        injector = NoiseInjector(seed=2026)
    kb = KnowledgeBase()

    # Precompute GT templates (Drain on clean, noise=0) for evaluation only
    for c in cases:
        alert = str(c["alert"])
        clean = NuSyEdgeNode.preprocess_header(alert, "Hadoop") or alert
        try:
            gt_tpl = drain.parse(clean)
        except Exception:
            gt_tpl = ""
        c["gt_tpl"] = gt_tpl

    # Build few-shot refs for Qwen
    qwen_refs = _build_qwen_refs_from_cases(cases, drain, max_refs=5)

    # Aggregation
    stats: Dict[float, Dict[str, int]] = {}
    total_iters = len(cases) * len(noise_levels)
    pbar = tqdm(total=total_iters, desc="RQ1-Hadoop Dim1", unit="step", disable=return_stats)

    # Per-case CSV for diagnostics
    results_dir = os.path.join(_PROJECT_ROOT, "results", "rq1_hadoop")
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, "rq1_hadoop_smallbatch_dim1_20260311.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f_csv:
        fieldnames = [
            "case_id",
            "label",
            "noise",
            "gt_template",
            "pred_nusy",
            "pred_drain",
            "pred_qwen",
            "pa_nusy",
            "pa_drain",
            "pa_qwen",
        ]
        writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
        writer.writeheader()

        for c in cases:
            alert = str(c["alert"])
            gt_tpl = str(c.get("gt_tpl", "") or "")
            case_id = str(c.get("case_id", "") or "")
            label = str(c.get("label", "") or "")

            for nl in noise_levels:
                injector.injection_rate = nl
                noisy = injector.inject(alert, dataset_type="HDFS")  # noise still reuses HDFS rules
                clean_for_parse = NuSyEdgeNode.preprocess_header(noisy, "Hadoop") or noisy

                # Drain
                try:
                    drain_tpl = drain.parse(clean_for_parse)
                except Exception as e:  # noqa: BLE001
                    drain_tpl = f"(drain_error: {e})"

                # NeSy: use the full NuSy-Edge pipeline (Hadoop-aware KB + shortcut + LLM),
                # consistent with HDFS/OpenStack RQ1 design.
                try:
                    nusy_tpl, _, _, _ = edge_node.parse_log_stream(noisy, "Hadoop")
                except Exception as e:  # noqa: BLE001
                    nusy_tpl = f"(nusy_error: {e})"

                # Qwen
                try:
                    qwen_tpl = qwen.parse_with_multi_rag(clean_for_parse, qwen_refs)
                except Exception as e:  # noqa: BLE001
                    qwen_tpl = f"(qwen_error: {e})"

                pa_drain = _golden_pa(drain_tpl, gt_tpl)
                pa_nusy = _golden_pa(nusy_tpl, gt_tpl)
                pa_qwen = _golden_pa(qwen_tpl, gt_tpl)

                if nl not in stats:
                    stats[nl] = {"total": 0, "hit_drain": 0, "hit_nusy": 0, "hit_qwen": 0}
                stats[nl]["total"] += 1
                stats[nl]["hit_drain"] += pa_drain
                stats[nl]["hit_nusy"] += pa_nusy
                stats[nl]["hit_qwen"] += pa_qwen

                writer.writerow(
                    {
                        "case_id": case_id,
                        "label": label,
                        "noise": nl,
                        "gt_template": gt_tpl,
                        "pred_nusy": nusy_tpl,
                        "pred_drain": drain_tpl,
                        "pred_qwen": qwen_tpl,
                        "pa_nusy": pa_nusy,
                        "pa_drain": pa_drain,
                        "pa_qwen": pa_qwen,
                    }
                )

                pbar.update(1)

    pbar.close()

    if return_stats:
        return stats

    # Print Markdown summary
    print("\n## RQ1-Hadoop Small-batch Dim1 (Golden PA)")
    print()
    print("| Noise | #Cases | PA_Nusy | PA_Drain | PA_Qwen |")
    print("|-------|--------|---------|----------|---------|")
    for nl in sorted(stats.keys()):
        total = stats[nl]["total"]
        if total <= 0:
            continue
        pn = stats[nl]["hit_nusy"] / total
        pd = stats[nl]["hit_drain"] / total
        pq = stats[nl]["hit_qwen"] / total
        print(
            f"| {nl:.1f} | {total} | "
            f"{stats[nl]['hit_nusy']}/{total} ({pn:.3f}) | "
            f"{stats[nl]['hit_drain']}/{total} ({pd:.3f}) | "
            f"{stats[nl]['hit_qwen']}/{total} ({pq:.3f}) |"
        )

    print("\n[DONE] RQ1-Hadoop small-batch benchmark finished.")


if __name__ == "__main__":
    run_rq1_hadoop_smallbatch()

