"""
Small Hadoop-only DeepSeek debug runner for Dim3/Dim4.

It prints:
- case id
- GT label
- prompt style / method
- raw model response
- extracted label
"""

import json
import os
import sys
from typing import Dict, List

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.system.edge_node import NuSyEdgeNode
from src.utils.noise_injector import NoiseInjector
from experiments.rq123_e2e.noise_injector_hadoop_20260311 import HadoopNoiseInjector
from experiments.rq123_e2e.stage4_noise_api_sampled_20260313 import (  # type: ignore
    BENCH_V2_PATH,
    CAUSAL_KB_DYNOTEARS,
    _denoise_for_nusy,
    _extract_label_from_json,
    _select_alert_line,
    _get_deepseek_api_key,
    _inject_noise,
    _truncate_and_inject_noise,
    _simple_allowed_labels,
    allowed_labels_for_dataset,
    describe_allowed_labels,
    gt_label_for_case,
)
from experiments.rq123_e2e.run_rq123_e2e_modular import _call_deepseek_with_retry  # type: ignore
from experiments.rq3 import tools as rq3_tools  # type: ignore


def _load_cases() -> List[Dict[str, object]]:
    with open(BENCH_V2_PATH, "r", encoding="utf-8") as f:
        all_cases = json.load(f)
    hadoop = [c for c in all_cases if str(c.get("dataset", "")) == "Hadoop"]
    chosen: List[Dict[str, object]] = []
    seen = set()
    for c in hadoop:
        label = gt_label_for_case(c)
        if label not in seen and label != "HADOOP_UNKNOWN":
            seen.add(label)
            chosen.append(c)
    return chosen


def main() -> None:
    cases = _load_cases()
    edge_node = NuSyEdgeNode()
    injector = NoiseInjector(seed=2026)
    injector_hadoop = HadoopNoiseInjector(seed=2026)
    api_key = _get_deepseek_api_key()

    for case in cases:
        dataset = "Hadoop"
        raw = str(case.get("raw_log", "") or "")
        gt_label = gt_label_for_case(case)
        alert = _select_alert_line(raw, dataset)
        noisy_alert = _inject_noise(alert, dataset, injector, injector_hadoop, 0.0)
        clean_for_parse = NuSyEdgeNode.preprocess_header(noisy_alert, dataset) or noisy_alert
        try:
            tpl_nusy, _, _, _ = edge_node.parse_log_stream(_denoise_for_nusy(dataset, clean_for_parse), dataset)
        except Exception:
            tpl_nusy = ""
        noised_context = _truncate_and_inject_noise(raw, dataset, injector, injector_hadoop, 0.0, max_chars=600)
        allowed = allowed_labels_for_dataset(dataset)

        prompts = {
            "agent": (
                "You are NeSy-Agent. Use ONLY the provided context.\n"
                "Task: identify the ROOT_CAUSE_LABEL for this incident.\n"
                f"{describe_allowed_labels(dataset, allowed)}\n"
                "Return STRICT JSON: {\"root_cause_label\": \"<ONE_LABEL_FROM_LIST>\", \"repair_action\": \"...\"}.\n\n"
                f"Dataset: {dataset}\n"
                f"Log window tail (truncated, noised):\n{noised_context}\n"
                f"Observed template (NuSy): {tpl_nusy or clean_for_parse}\n"
                f"Causal candidates (JSON): {rq3_tools.causal_navigator(tpl_nusy or clean_for_parse, 'hadoop', causal_path=CAUSAL_KB_DYNOTEARS)}\n"
            ),
            "vanilla": (
                "You are an ops expert. Analyze the log and identify the root cause label.\n"
                f"{_simple_allowed_labels(allowed)}\n"
                "Return STRICT JSON: {\"root_cause_label\": \"<ONE_LABEL_FROM_LIST>\", \"repair_action\": \"...\"}.\n\n"
                f"Dataset: {dataset}\n"
                f"Log window tail (truncated, noised):\n{noised_context}\n"
            ),
            "rag": (
                "You are an ops expert. Use the references and logs to choose the root cause label.\n"
                f"{_simple_allowed_labels(allowed)}\n"
                "Return STRICT JSON: {\"root_cause_label\": \"<ONE_LABEL_FROM_LIST>\", \"repair_action\": \"...\"}.\n\n"
                f"Dataset: {dataset}\n"
                f"Log window tail (truncated, noised):\n{noised_context}\n"
                f"References:\n{rq3_tools.knowledge_retriever(clean_for_parse[:200], dataset, top_k=3)}\n"
            ),
        }

        print(f"\n=== CASE {case.get('case_id')} | GT={gt_label} ===")
        for method, prompt in prompts.items():
            resp = _call_deepseek_with_retry(prompt, api_key=api_key, model="deepseek-chat", max_tokens=256)
            pred = _extract_label_from_json(resp, allowed)
            print(f"\n[{method}]")
            print(resp)
            print(f"--> extracted={pred or '(empty)'}")


if __name__ == "__main__":
    main()
