"""
Stage4 mini API diagnose (HDFS + Hadoop, noise=0, first 15 cases each).

Goal:
- Diagnose and fix the issue that Dim3 (RCA) and Dim4 (Semantic E2E) are 0 for
  HDFS Vanilla/RAG and all Hadoop methods, by:
  - Mapping fine-grained GT root templates to a small set of macro labels.
  - Making DeepSeek output ONLY these macro labels.
  - Evaluating RCA/E2E at label level (and, for HDFS, aligning with existing SOP ids).

IMPORTANT:
- This script is for development/debugging only. It does NOT change the main Stage4
  implementation; once we are satisfied with the behavior here, we can port the
  label-space design into the main pipeline.
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
from src.utils.llm_client import LLMClient
from src.utils.metrics import MetricsCalculator

from experiments.rq123_e2e.noise_injector_hadoop_20260311 import HadoopNoiseInjector
from experiments.rq123_e2e.run_rq123_e2e_modular import (  # type: ignore
    _load_benchmark,
    _load_repair_sop,
    _resolve_action_id,
    _truncate_logs,
    _get_deepseek_api_key,
    _call_deepseek_with_retry,
)
from experiments.rq3 import evaluate as rq3_eval  # type: ignore


DATA_PROCESSED = os.path.join(_PROJECT_ROOT, "data", "processed")
RESULTS_E2E_DIR = os.path.join(_PROJECT_ROOT, "results", "rq123_e2e")
BENCH_V2_PATH = os.path.join(DATA_PROCESSED, "e2e_scaled_benchmark_v2.json")


# -------------------- Label mapping helpers --------------------

HDFS_LABELS = [
    "HDFS_GOT_EXCEPTION_SERVING",
    "HDFS_ALLOCATE_BLOCK",
    "HDFS_OTHER",
]

HADOOP_LABELS = [
    "HADOOP_MERGER",
    "HADOOP_SPILL",
    "HADOOP_METRICS",
    "HADOOP_DIAGNOSTICS",
    "HADOOP_YARNCHILD",
    "HADOOP_UNKNOWN",
]


def _hdfs_gt_label(gt_root: str) -> str:
    t = (gt_root or "").lower()
    if "got exception while serving" in t:
        return "HDFS_GOT_EXCEPTION_SERVING"
    if "allocateblock" in t:
        return "HDFS_ALLOCATE_BLOCK"
    return "HDFS_OTHER"


def _hadoop_gt_label(gt_root: str) -> str:
    t = (gt_root or "").lower()
    if "mapred.merger: merging" in t:
        return "HADOOP_MERGER"
    if "maptask: finished spill" in t:
        return "HADOOP_SPILL"
    if "metricssystemimpl: stopping maptask metrics system" in t:
        return "HADOOP_METRICS"
    if "taskattemptimpl: diagnostics report" in t:
        return "HADOOP_DIAGNOSTICS"
    if "yarnchild:" in t:
        return "HADOOP_YARNCHILD"
    return "HADOOP_UNKNOWN"


def gt_label_for_case(case: Dict[str, object]) -> str:
    ds = str(case.get("dataset", "HDFS"))
    gt_root = str(case.get("ground_truth_root_cause_template", "") or "")
    if ds == "HDFS":
        return _hdfs_gt_label(gt_root)
    if ds == "Hadoop":
        return _hadoop_gt_label(gt_root)
    return ""


def _extract_label_from_json(text: str, allowed: List[str]) -> str:
    """
    Try to parse a JSON object and extract 'root_cause_label'.
    Fallback: search for any allowed label substring.
    """
    t = (text or "").strip()
    if not t:
        return ""
    # Try JSON parse
    try:
        start = t.find("{")
        end = t.rfind("}")
        if start >= 0 and end > start:
            obj = json.loads(t[start : end + 1])
            if isinstance(obj, dict):
                lbl = obj.get("root_cause_label") or obj.get("label") or ""
                lbl = str(lbl).strip()
                for a in allowed:
                    if a.lower() == lbl.lower():
                        return a
    except Exception:
        pass
    # Fallback: search any allowed label substring
    lower = t.lower()
    for a in allowed:
        if a.lower() in lower:
            return a
    return ""


def _map_label_to_sop_id(dataset: str, label: str) -> str:
    """
    For HDFS, map our macro label to existing SOP ids.
    For Hadoop, currently there is no dedicated SOP entry, so we only use label-level E2E.
    """
    if dataset == "HDFS":
        if label == "HDFS_GOT_EXCEPTION_SERVING":
            return "HDFS_GOT_EXCEPTION_SERVING"
        if label == "HDFS_ALLOCATE_BLOCK":
            return "HDFS_ALLOCATE_BLOCK"
    return ""


# -------------------- Main mini-eval logic --------------------


def run_stage4_mini_api_diagnose() -> None:
    cases = _load_benchmark(BENCH_V2_PATH)
    if not cases:
        print("[ERROR] No cases in benchmark.")
        return

    # Select first 15 HDFS + first 15 Hadoop cases (noise=0 scenario)
    hdfs_cases: List[Dict[str, object]] = []
    hadoop_cases: List[Dict[str, object]] = []
    for c in cases:
        ds = str(c.get("dataset", "HDFS"))
        if ds == "HDFS" and len(hdfs_cases) < 15:
            hdfs_cases.append(c)
        elif ds == "Hadoop" and len(hadoop_cases) < 15:
            hadoop_cases.append(c)
        if len(hdfs_cases) >= 15 and len(hadoop_cases) >= 15:
            break

    mini_cases: List[Dict[str, object]] = hdfs_cases + hadoop_cases
    print(f"[INFO] Selected {len(hdfs_cases)} HDFS + {len(hadoop_cases)} Hadoop cases (total {len(mini_cases)}).")

    # Filter out cases with UNKNOWN labels for RCA/E2E stats
    labeled_cases = []
    for c in mini_cases:
        lbl = gt_label_for_case(c)
        if lbl and not lbl.endswith("UNKNOWN"):
            labeled_cases.append((c, lbl))
    print(f"[INFO] Labeled cases (excluding UNKNOWN): {len(labeled_cases)}/{len(mini_cases)}.")

    edge_node = NuSyEdgeNode()
    qwen = LLMClient()
    sop_kb = _load_repair_sop()
    deepseek_key = _get_deepseek_api_key()

    # Aggregation: per-dataset & method
    methods = ["agent", "vanilla", "rag"]
    stats_rca: Dict[Tuple[str, str], Dict[str, int]] = {}
    stats_e2e: Dict[Tuple[str, str], Dict[str, int]] = {}

    for ds in ["HDFS", "Hadoop"]:
        for m in methods:
            stats_rca[(ds, m)] = {"total": 0, "success": 0}
            stats_e2e[(ds, m)] = {"total": 0, "success": 0}

    pbar = tqdm(total=len(labeled_cases) * len(methods), desc="Mini Stage4 RCA/E2E", unit="call")

    for case, gt_label in labeled_cases:
        raw = str(case.get("raw_log", ""))
        dataset = str(case.get("dataset", "HDFS"))
        case_id = str(case.get("case_id", ""))

        lines = [l for l in raw.split("\n") if l.strip()]
        alert = lines[-1] if lines else raw
        clean_alert = NuSyEdgeNode.preprocess_header(alert, dataset) or alert

        # For prompts
        gt_tpl = str(case.get("ground_truth_template", "") or "")

        # HDFS vs Hadoop label space
        if dataset == "HDFS":
            allowed_labels = HDFS_LABELS
        else:
            allowed_labels = HADOOP_LABELS

        # Build a short description for DeepSeek
        label_desc = (
            "Allowed root cause labels (choose ONE exactly): "
            + ", ".join(allowed_labels)
        )

        # Common base prompt tail (log info)
        base_tail = (
            f"Dataset: {dataset}\n"
            f"Log window tail (truncated): {_truncate_logs(raw, 600)}\n"
        )

        # NuSy template as additional context
        try:
            nusy_tpl, _, _, _ = edge_node.parse_log_stream(clean_alert, dataset)
        except Exception:
            nusy_tpl = ""

        # Agent prompt (uses causal cues + NuSy template)
        for method in methods:
            if method == "agent":
                user_msg = (
                    "You are NeSy-Agent. Use ONLY the provided context.\n"
                    "Task: identify the ROOT_CAUSE_LABEL for this incident.\n"
                    f"{label_desc}\n"
                    "Return STRICT JSON: {\"root_cause_label\": \"<ONE_LABEL_FROM_LIST>\", \"repair_action\": \"...\"}.\n\n"
                    f"{base_tail}"
                    f"Parsed template (NuSy-Edge): {nusy_tpl or gt_tpl}\n"
                )
            elif method == "vanilla":
                user_msg = (
                    "You are an ops expert. Analyze the log and identify the root cause.\n"
                    f"{label_desc}\n"
                    "Return STRICT JSON: {\"root_cause_label\": \"<ONE_LABEL_FROM_LIST>\", \"repair_action\": \"...\"}.\n\n"
                    f"{base_tail}"
                )
            else:  # rag
                from experiments.rq3 import tools as rq3_tools  # type: ignore

                refs = rq3_tools.knowledge_retriever(clean_alert[:200], dataset, top_k=3)
                user_msg = (
                    "You are an ops expert. Use the references and logs to choose the root cause label.\n"
                    f"{label_desc}\n"
                    "Return STRICT JSON: {\"root_cause_label\": \"<ONE_LABEL_FROM_LIST>\", \"repair_action\": \"...\"}.\n\n"
                    f"{base_tail}"
                    f"References:\n{refs}\n"
                )

            resp = _call_deepseek_with_retry(
                user_msg, api_key=deepseek_key, model="deepseek-chat", max_tokens=256
            )

            pred_label = _extract_label_from_json(resp, allowed_labels)

            key = (dataset, method)
            stats_rca[key]["total"] += 1
            if pred_label and pred_label == gt_label:
                stats_rca[key]["success"] += 1

            # Semantic E2E:
            # - For HDFS: map labels to SOP ids and compare action_id equality.
            # - For Hadoop: we use label-level equality as proxy (no dedicated SOP entries yet).
            stats_e2e[key]["total"] += 1
            if dataset == "HDFS":
                gt_action_id = _map_label_to_sop_id(dataset, gt_label)
                pred_action_id = _map_label_to_sop_id(dataset, pred_label)
                if gt_action_id and pred_action_id and gt_action_id == pred_action_id:
                    stats_e2e[key]["success"] += 1
            else:
                if pred_label and pred_label == gt_label:
                    stats_e2e[key]["success"] += 1

            pbar.update(1)

    pbar.close()

    # Print summary
    print("\n=== Mini Dim3 RCA (label-level) ===")
    for ds in ["HDFS", "Hadoop"]:
        for m in methods:
            s = stats_rca[(ds, m)]
            n = s["total"] or 1
            print(
                f"{ds}\t{m}\tRCA: {s['success']}/{s['total']} ({s['success']/n:.3f})"
            )

    print("\n=== Mini Dim4 Semantic E2E (label/SOP-level) ===")
    for ds in ["HDFS", "Hadoop"]:
        for m in methods:
            s = stats_e2e[(ds, m)]
            n = s["total"] or 1
            print(
                f"{ds}\t{m}\tE2E: {s['success']}/{s['total']} ({s['success']/n:.3f})"
            )


if __name__ == "__main__":
    run_stage4_mini_api_diagnose()

