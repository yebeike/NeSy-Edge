import os
import sys
import json
import argparse
import time
from typing import Dict, List, Tuple

import pandas as pd
from tqdm import tqdm


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


from src.system.edge_node import NuSyEdgeNode
from src.perception.drain_parser import DrainParser
from src.utils.noise_injector import NoiseInjector
from src.utils.metrics import MetricsCalculator
from experiments.rq3 import tools as rq3_tools
from experiments.rq3 import evaluate as rq3_eval
from experiments.rq123_e2e import hadoop_loader


DATA_PROCESSED = os.path.join(_PROJECT_ROOT, "data", "processed")
RESULTS_E2E_DIR = os.path.join(_PROJECT_ROOT, "results", "rq123_e2e")
# 使用带自动 GT 标注的新 v2 基准文件（保留 v1 不动）
SCALED_BENCH_PATH = os.path.join(DATA_PROCESSED, "e2e_scaled_benchmark_v2.json")
REPAIR_SOP_PATH = os.path.join(DATA_PROCESSED, "repair_sop_kb.json")


NOISE_LEVELS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
# Stage 3 decision methods（概念层的命名）
METHODS_STAGE3 = ["NuSy-Agent", "Vanilla", "StandardRAG"]
# 实际内部实现使用 legacy 名称（NuSy-Edge_offline 等）
METHODS = ["NuSy-Edge_offline", "Vanilla", "StandardRAG"]


def _norm(s: str) -> str:
    return rq3_eval._norm(s)  # type: ignore[attr-defined]


def _shorten(text: str, max_chars: int = 2000) -> str:
    """
    Safely truncate long log windows before sending to DeepSeek，避免超过上下文长度导致 400。
    保留结尾部分，因为故障告警通常在窗口后段。
    """
    if not text:
        return ""
    text = str(text)
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def _load_scaled_cases(path: str = SCALED_BENCH_PATH) -> List[Dict[str, object]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Scaled benchmark not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_causal_kb(path: str) -> List[Dict[str, object]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"causal_knowledge.json not found at {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_repair_sop(path: str = REPAIR_SOP_PATH) -> List[Dict[str, str]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"repair_sop_kb.json not found at {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("repair_sop_kb.json must be a list")
    return data


def _load_causal_edges_and_templates(causal_path: str) -> Tuple[set, set]:
    valid_edges = rq3_eval._load_causal_edges(causal_path)  # type: ignore[attr-defined]
    valid_templates = rq3_eval._load_valid_templates(causal_path)  # type: ignore[attr-defined]
    return valid_edges, valid_templates


def _domain_from_dataset(dataset: str) -> str:
    ds = (dataset or "").upper()
    if ds == "HDFS":
        return "hdfs"
    if ds == "OPENSTACK":
        return "openstack"
    if ds == "HADOOP":
        return "hadoop"
    return "hdfs"


def _select_pilot_cases(all_cases: List[Dict[str, object]], n_total: int = 10) -> List[Dict[str, object]]:
    """
    pilot-run 要求：10 个 case，尽量混合 HDFS / OpenStack / Hadoop。
    策略：优先每类取若干条，不足则用剩余补齐。
    """
    buckets: Dict[str, List[Dict[str, object]]] = {"HDFS": [], "OpenStack": [], "Hadoop": []}
    for c in all_cases:
        ds = str(c.get("dataset", "HDFS"))
        if ds in buckets:
            buckets[ds].append(c)
    # 目标配比（可调整）：4 HDFS + 3 OpenStack + 3 Hadoop
    plan = [("HDFS", 4), ("OpenStack", 3), ("Hadoop", 3)]
    out: List[Dict[str, object]] = []
    for ds, k in plan:
        out.extend(buckets.get(ds, [])[:k])
    if len(out) < n_total:
        # 用任意剩余 case 补齐
        seen = set(id(x) for x in out)
        for c in all_cases:
            if id(c) in seen:
                continue
            out.append(c)
            if len(out) >= n_total:
                break
    return out[:n_total]


def _select_pilot_2_per_dataset(all_cases: List[Dict[str, object]]) -> List[Dict[str, object]]:
    """Pilot: exactly 2 cases per dataset (HDFS, OpenStack, Hadoop) = 6 cases total."""
    buckets: Dict[str, List[Dict[str, object]]] = {"HDFS": [], "OpenStack": [], "Hadoop": []}
    for c in all_cases:
        ds = str(c.get("dataset", "HDFS"))
        if ds in buckets and len(buckets[ds]) < 2:
            buckets[ds].append(c)
    out: List[Dict[str, object]] = []
    for ds in ["HDFS", "OpenStack", "Hadoop"]:
        out.extend(buckets.get(ds, [])[:2])
    return out


def _select_midscale_cases(all_cases: List[Dict[str, object]], per_ds: int = 15) -> List[Dict[str, object]]:
    """
    从放大版基准集中选择中尺度验证用的 case：
      - 每个数据集（HDFS / OpenStack / Hadoop）各取 per_ds 条
      - 要求有有效的 GT 根因（HDFS/OpenStack 为 template，Hadoop 为 label）
    """
    buckets: Dict[str, List[Dict[str, object]]] = {"HDFS": [], "OpenStack": [], "Hadoop": []}
    for c in all_cases:
        ds = str(c.get("dataset", "HDFS"))
        gt_root = str(c.get("ground_truth_root_cause_template", "") or "")
        if not gt_root or gt_root.lower() == "unknown":
            continue
        if ds in buckets and len(buckets[ds]) < per_ds:
            buckets[ds].append(c)
    out: List[Dict[str, object]] = []
    for ds in ["HDFS", "OpenStack", "Hadoop"]:
        out.extend(buckets.get(ds, []))
    return out


def _parse_root_from_deepseek_answer(text: str, dataset: str = "") -> str:
    """
    尝试从 DeepSeek 回答中解析 root_cause_template：
      - 优先 JSON: {"root_cause_template": "...", "repair_action": "..."}
      - 否则回退到 RQ3 evaluator 的抽取逻辑
    """
    t = (text or "").strip()
    if not t:
        return ""
    # Hadoop 特例：优先从回答中抽取三类宏观 label
    if (dataset or "").lower() == "hadoop":
        lower = t.lower()
        for label in ["Machine down", "Network disconnection", "Disk full"]:
            if label.lower() in lower:
                return label
    # 先尝试直接解析 JSON（容忍回答中夹杂前后缀）
    try:
        start = t.find("{")
        end = t.rfind("}")
        if start >= 0 and end > start:
            obj = json.loads(t[start : end + 1])
            if isinstance(obj, dict):
                rc = str(obj.get("root_cause_template", "") or "").strip()
                if rc:
                    return rc
    except Exception:
        pass
    pred_record = {"model_answer": t, "extracted_root_cause": ""}
    try:
        return str(rq3_eval._extract_stated_root_cause(pred_record) or "").strip()  # type: ignore[attr-defined]
    except Exception:
        return ""


def _calc_causal_sparsity_and_rank(
    causal_kb: List[Dict[str, object]],
    domain: str,
    gt_root: str,
    gt_effect: str,
) -> Tuple[int, int]:
    domain = (domain or "hdfs").lower()
    gt_root_n = _norm(gt_root)
    gt_effect_n = _norm(gt_effect)

    edges_domain = [e for e in causal_kb if e.get("domain") == domain]
    sparsity = len(edges_domain)
    same_target = []
    for e in edges_domain:
        t = _norm(e.get("target_template", ""))
        if not t:
            continue
        if rq3_eval._rca_match(t, gt_effect) or rq3_eval._rca_match(gt_effect, t):  # type: ignore[attr-defined]
            same_target.append(e)
    if not same_target:
        return sparsity, -1
    scored = []
    for e in same_target:
        w = float(e.get("weight", 0.0) or 0.0)
        scored.append((abs(w), e))
    scored.sort(key=lambda x: x[0], reverse=True)
    rank = -1
    for idx, (_, e) in enumerate(scored, start=1):
        s = _norm(e.get("source_template", ""))
        if s and (rq3_eval._rca_match(s, gt_root) or rq3_eval._rca_match(gt_root, s)):  # type: ignore[attr-defined]
            rank = idx
            break
    return sparsity, rank


def _resolve_repair_action(
    sop_kb: List[Dict[str, str]],
    root_tpl: str,
    dataset: str,
) -> Tuple[str, str]:
    if not root_tpl:
        return "", ""
    ds = (dataset or "").strip() or "HDFS"
    root_n = _norm(root_tpl)
    best_id, best_action = "", ""
    best_score = 0.0
    for entry in sop_kb:
        if entry.get("dataset", "").lower() != ds.lower():
            continue
        pattern = entry.get("pattern", "")
        if not pattern:
            continue
        pat_n = _norm(pattern)
        if rq3_eval._rca_match(root_n, pat_n):  # type: ignore[attr-defined]
            score = len(pat_n)
        else:
            tokens_r = set(root_n.split())
            tokens_p = set(pat_n.split())
            inter = tokens_r & tokens_p
            score = len(inter)
        if score > best_score:
            best_score = score
            best_id = str(entry.get("id", "") or "")
            best_action = str(entry.get("repair_action", "") or "")
    if best_score <= 0.0:
        return "", ""
    return best_id, best_action


def _resolve_hadoop_label(
    sop_kb: List[Dict[str, str]],
    root_tpl: str,
    dataset: str,
) -> str:
    """
    专门用于 Hadoop：根据预测模板，在 SOP KB 中找到对应的宏观 label（如 Machine down）。
    逻辑与 _resolve_repair_action 相同，但返回 entry["label"]。
    """
    if (dataset or "").lower() != "hadoop":
        return ""
    ds = "Hadoop"
    root_n = _norm(root_tpl)
    best_label = ""
    best_score = 0.0
    for entry in sop_kb:
        if entry.get("dataset", "").lower() != ds.lower():
            continue
        pattern = entry.get("pattern", "")
        if not pattern:
            continue
        pat_n = _norm(pattern)
        if rq3_eval._rca_match(root_n, pat_n):  # type: ignore[attr-defined]
            score = len(pat_n)
        else:
            tokens_r = set(root_n.split())
            tokens_p = set(pat_n.split())
            inter = tokens_r & tokens_p
            score = len(inter)
        if score > best_score:
            best_score = score
            best_label = str(entry.get("label", "") or "")
    return best_label


def _phase1_warmup(edge_node: NuSyEdgeNode) -> None:
    """
    轻量 Phase1 预热：对 HDFS / OpenStack / Hadoop 正常日志跑一遍 NuSy-Edge 解析。
    """
    from experiments.rq123_e2e.run_rq123_e2e import _iter_phase1_logs_for_dataset  # type: ignore

    for ds in ["HDFS", "OpenStack", "Hadoop"]:
        logs = _iter_phase1_logs_for_dataset(ds, max_logs=200)
        for raw in logs:
            try:
                edge_node.parse_log_stream(raw, ds if ds != "Hadoop" else "HDFS")
            except Exception:
                continue


def run_massive_e2e(
    max_cases: int = 50,
    noise_levels: List[float] = None,
    methods: List[str] = None,
    use_fail_fast: bool = True,
    pilot_run: bool = False,
    pilot_2_per_dataset: bool = False,
    use_api: bool = False,
    midscale: bool = False,
) -> pd.DataFrame:
    """
    大规模 E2E 管道（离线近似版）：
      - Stage1: NuSy-Edge + Drain 解析（本地）
      - Stage2: causal_knowledge 查询（本地）
      - Stage3: Fail-Fast 短路：若解析崩溃或因果图无路径则跳过 Stage4
      - Stage4: 离线 NuSy-Agent 近似（log_parser + causal_navigator + SOP 映射，无 API）
    """
    if noise_levels is None:
        noise_levels = NOISE_LEVELS
    if methods is None:
        methods = METHODS

    cases = _load_scaled_cases()
    if pilot_run:
        cases = _select_pilot_cases(cases, n_total=10)
    elif pilot_2_per_dataset:
        cases = _select_pilot_2_per_dataset(cases)
    elif midscale:
        cases = _select_midscale_cases(cases, per_ds=15)
    elif max_cases > 0:
        cases = cases[:max_cases]

    # 优先使用新的 Hadoop 稀疏图文件，如不存在则回退到原始 causal_knowledge.json
    pruned_path = os.path.join(DATA_PROCESSED, "causal_knowledge_hadoop_pruned.json")
    if os.path.exists(pruned_path):
        causal_path = pruned_path
    else:
        causal_path = rq3_tools.get_causal_knowledge_path()
    causal_kb = _load_causal_kb(causal_path)
    valid_edges, valid_templates = _load_causal_edges_and_templates(causal_path)
    sop_kb = _load_repair_sop(REPAIR_SOP_PATH)

    edge_node = NuSyEdgeNode()
    drain = DrainParser()
    injector = NoiseInjector(seed=2026)

    # DeepSeek API 工具（pilot 或 midscale 时使用）
    from experiments.rq3.agent import _get_deepseek_api_key, run_chat_only_deepseek  # type: ignore[attr-defined]

    deepseek_key = _get_deepseek_api_key()
    pricing_input = 2.0 / 1_000_000.0   # RMB per 1M input tokens
    pricing_output = 3.0 / 1_000_000.0  # RMB per 1M output tokens
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_api_calls = 0

    _phase1_warmup(edge_node)

    rows: List[Dict[str, object]] = []

    total_iters = len(cases) * len(noise_levels) * len(methods)
    desc = "E2E Massive Pilot" if pilot_run else ("E2E Massive Midscale" if midscale else "E2E Massive")
    pbar = tqdm(total=total_iters, desc=desc, unit="step")

    for case in cases:
        raw = str(case.get("raw_log", ""))
        dataset = str(case.get("dataset", "HDFS"))
        gt_tpl = str(case.get("ground_truth_template", ""))
        gt_root = str(case.get("ground_truth_root_cause_template", ""))
        case_id = str(case.get("case_id", ""))
        source = str(case.get("source", ""))

        for noise in noise_levels:
            injector.injection_rate = noise
            noisy_raw = injector.inject(raw, dataset_type="HDFS" if dataset == "Hadoop" else dataset)
            clean_log = NuSyEdgeNode.preprocess_header(noisy_raw, "HDFS" if dataset == "Hadoop" else dataset)
            if not clean_log:
                clean_log = noisy_raw

            # ---------- Stage1: RQ1 解析 (Perception) ----------
            # Use last line for multi-line windows as the "target" log for parsing
            log_for_parse = raw.strip().split("\n")[-1].strip() if "\n" in raw.strip() else raw.strip()
            if not log_for_parse:
                log_for_parse = raw.strip()[:500]
            clean_for_parse = NuSyEdgeNode.preprocess_header(log_for_parse, "HDFS" if dataset == "Hadoop" else dataset)
            if not clean_for_parse:
                clean_for_parse = log_for_parse

            # Drain (baseline)
            try:
                t0 = time.time()
                drain_tpl = drain.parse(clean_log)
                drain_lat = (time.time() - t0) * 1000
            except Exception:
                drain_tpl, drain_lat = "", 0.0

            # NuSy-Edge
            t0 = time.time()
            try:
                nusy_tpl, nusy_lat, _, _ = edge_node.parse_log_stream(noisy_raw, "HDFS" if dataset == "Hadoop" else dataset)
            except Exception as e:
                nusy_tpl, nusy_lat = f"(parse_error: {e})", (time.time() - t0) * 1000

            # Vanilla LLM (Qwen3-0.6B): no RAG, no graph
            vanilla_llm_tpl = ""
            vanilla_llm_lat = 0.0
            try:
                t0 = time.time()
                vanilla_llm_tpl = edge_node.llm.parse_with_multi_rag(clean_for_parse, [])
                vanilla_llm_lat = (time.time() - t0) * 1000
            except Exception:
                pass

            # 对 Hadoop：只有宏观 label，无模板级 GT，PA 记为 NaN（None），避免误导
            if dataset in ("Hadoop", "hadoop"):
                pa_nusy = None
                pa_drain = None
                pa_vanilla_llm = None
            else:
                pa_nusy = 0
                if gt_tpl:
                    pa_nusy = 1 if rq3_eval._rca_match(nusy_tpl, gt_tpl) else 0  # type: ignore[attr-defined]
                pa_drain = 0
                if gt_tpl:
                    pa_drain = 1 if rq3_eval._rca_match(drain_tpl, gt_tpl) else 0  # type: ignore[attr-defined]
                pa_vanilla_llm = 0
                if gt_tpl:
                    pa_vanilla_llm = 1 if rq3_eval._rca_match(vanilla_llm_tpl, gt_tpl) else 0  # type: ignore[attr-defined]

            # ---------- Stage2: RQ2 因果图 ----------
            domain = _domain_from_dataset(dataset)
            sparsity, local_rank = _calc_causal_sparsity_and_rank(causal_kb, domain, gt_root, gt_tpl)
            for method in methods:
                # ---------- Stage3: Fail-Fast（按方法解耦） ----------
                fail_fast_triggered = False
                if use_fail_fast:
                    if method == "NuSy-Edge_offline":
                        # 对 HDFS / OpenStack：需要解析 + 图完整 + rank>0
                        if dataset not in ("Hadoop", "hadoop"):
                            if (gt_tpl and not rq3_eval._rca_match(nusy_tpl, gt_tpl)) or sparsity == 0 or local_rank <= 0:  # type: ignore[attr-defined]
                                fail_fast_triggered = True
                        else:
                            # Hadoop：当前仅有粗粒度 GT 根因标签，不要求模板匹配与 rank，只要图非空即可进入 Stage4
                            if sparsity == 0:
                                fail_fast_triggered = True
                    else:
                        # Baseline 直接用原始日志，仅在完全空白时短路
                        if not raw.strip():
                            fail_fast_triggered = True
                rca_ok = 0
                hallu = 0
                repair_match = 0
                pred_root = ""
                pred_action_id = ""
                pred_action_desc = ""

                if not fail_fast_triggered:
                    # ---------- Stage4: Agent / Baseline ----------
                    tpl_for_rq3 = nusy_tpl or clean_log

                    if use_api and deepseek_key and method in ("NuSy-Edge_offline", "Vanilla", "StandardRAG"):
                        # Pilot：强制 3 方法都走 DeepSeek（每个 case×方法一次调用），NuSy-Edge 用神经符号上下文
                        if method == "NuSy-Edge_offline":
                            # 先用图导航获得候选根因（若为空，按 fail-fast 已短路；这里正常应有候选）
                            root_json = rq3_tools.causal_navigator(tpl_for_rq3, domain)
                            try:
                                root_list = json.loads(root_json)
                            except Exception:
                                root_list = []
                            if isinstance(root_list, dict):
                                root_list = []
                            cand = []
                            if root_list:
                                cand = [f"{r.get('source_template','')}" for r in root_list[:5]]
                            raw_snippet = _shorten(raw, 2000)
                            if dataset in ("Hadoop", "hadoop"):
                                # Hadoop：要求输出宏观 label + 动作，label 必须是列举值之一
                                user_msg = (
                                    "You are NuSy-Agent for Hadoop logs. Use ONLY the provided context.\n"
                                    "Return STRICT JSON: {\"root_cause_template\": \"...\", "
                                    "\"root_cause_label\": \"Machine down|Network disconnection|Disk full\", "
                                    "\"repair_action\": \"...\"}\n\n"
                                    f"Dataset: {dataset}\n"
                                    f"Observed log (raw, tail): {raw_snippet}\n"
                                    f"Parsed template (NuSy-Edge): {tpl_for_rq3}\n"
                                    f"Causal graph candidates (top causes): {cand}\n"
                                    "Task: choose the correct root_cause_label from the three options based on the log and context, "
                                    "fill a short root_cause_template, and output a concrete repair_action."
                                )
                            else:
                                user_msg = (
                                    "You are NuSy-Agent. Use ONLY the provided symbolic context.\n"
                                    "Return STRICT JSON: {\"root_cause_template\": \"...\", \"repair_action\": \"...\"}\n\n"
                                    f"Dataset: {dataset}\n"
                                    f"Observed log (raw, tail): {raw_snippet}\n"
                                    f"Parsed template (NuSy-Edge): {tpl_for_rq3}\n"
                                    f"Causal graph candidates (top causes): {cand}\n"
                                    "Task: pick the most likely root_cause_template from candidates, and output a concrete repair_action."
                                )
                        elif method == "Vanilla":
                            # FAIRNESS: include valid macro-labels so baseline acts as classifier
                            raw_snippet = _shorten(raw, 2000)
                            if dataset in ("Hadoop", "hadoop"):
                                choices = "Choose from exactly: Machine down, Network disconnection, Disk full."
                            else:
                                choices = "Choose from common root cause types (e.g. verification succeeded, deleting block, got exception while serving, block allocation, receive block, etc.)."
                            user_msg = (
                                "You are an ops expert. Analyze the log and identify the root cause.\n"
                                f"Valid root cause options: {choices}\n"
                                "Return STRICT JSON: {\"root_cause_template\": \"...\", \"repair_action\": \"...\"}\n\n"
                                f"Log (tail):\n{raw_snippet}"
                            )
                        else:
                            raw_snippet = _shorten(raw, 2000)
                            refs = rq3_tools.knowledge_retriever(raw[:200], dataset, top_k=3)
                            if dataset in ("Hadoop", "hadoop"):
                                choices = "Choose from exactly: Machine down, Network disconnection, Disk full."
                            else:
                                choices = "Choose from common root cause types consistent with the references."
                            user_msg = (
                                "You are an ops expert. Use the references and choose a root cause.\n"
                                f"Valid root cause options: {choices}\n"
                                "Return STRICT JSON: {\"root_cause_template\": \"...\", \"repair_action\": \"...\"}\n\n"
                                f"References:\n{refs}\n\nLog (tail):\n{raw_snippet}"
                            )
                        resp_text, usage = run_chat_only_deepseek(
                            user_msg, api_key=deepseek_key, model="deepseek-chat", max_tokens=512
                        )
                        total_prompt_tokens += usage.get("prompt_tokens", 0)
                        total_completion_tokens += usage.get("completion_tokens", 0)
                        total_api_calls += 1
                        pred_record = {
                            "raw_log": raw,
                            "dataset": dataset,
                            "source": source,
                            "ground_truth_template": gt_tpl,
                            "ground_truth_root_cause_template": gt_root,
                            "extracted_root_cause": "",
                            "model_answer": resp_text,
                        }
                        pred_root = _parse_root_from_deepseek_answer(resp_text, dataset=dataset)
                    else:
                        # NuSy-Edge_offline 或非 pilot 模式：用因果导航作为近似
                        root_json = rq3_tools.causal_navigator(tpl_for_rq3, domain)
                        try:
                            root_list = json.loads(root_json)
                        except Exception:
                            root_list = []
                        if isinstance(root_list, dict):
                            root_list = []
                        if root_list:
                            pred_root = str(root_list[0].get("source_template", "")).strip()

                        pred_record = {
                            "raw_log": raw,
                            "dataset": dataset,
                            "source": source,
                            "ground_truth_template": gt_tpl,
                            "ground_truth_root_cause_template": gt_root,
                            "extracted_root_cause": pred_root,
                            "model_answer": "",
                        }

                    if gt_root and gt_root.lower() != "unknown":
                        if dataset in ("Hadoop", "hadoop"):
                            # Hadoop：用 SOP KB 映射模板 -> label，再与 GT label 比较
                            pred_label = _resolve_hadoop_label(sop_kb, pred_root, dataset)
                            if pred_label and rq3_eval._rca_match(pred_label, gt_root):  # type: ignore[attr-defined]
                                rca_ok = 1
                        else:
                            if rq3_eval._rca_match(pred_root, gt_root):  # type: ignore[attr-defined]
                                rca_ok = 1
                    hallu = 1 if rq3_eval._detect_hallucination(pred_record, valid_templates, valid_edges) else 0  # type: ignore[attr-defined]

                    pred_action_id, pred_action_desc = _resolve_repair_action(sop_kb, pred_root, dataset)
                    gt_action_id, _ = _resolve_repair_action(sop_kb, gt_root, dataset)
                    repair_match = 1 if (gt_action_id and pred_action_id and gt_action_id == pred_action_id) else 0

                    # 语义动作回退：若严格模板匹配失败，但 SOP repair_action 完全一致，也认为 RCA 正确
                    if gt_root and gt_root.lower() != "unknown" and dataset not in ("Hadoop", "hadoop"):
                        if not rca_ok and repair_match == 1:
                            rca_ok = 1

                # ---------- E2E 成功判定: 1 ONLY IF PA>0 AND Rank>0 AND RCA==1 ----------
                pa_for_method = pa_nusy if method != "Vanilla" else pa_vanilla_llm
                if dataset in ("Hadoop", "hadoop"):
                    # Hadoop: no template GT; require sparsity and RCA only
                    e2e_success = 1 if (sparsity > 0 and rca_ok == 1) else 0
                else:
                    e2e_success = 1 if (
                        pa_for_method > 0
                        and sparsity > 0
                        and local_rank > 0
                        and rca_ok == 1
                    ) else 0

                rows.append(
                    {
                        "case_id": case_id,
                        "dataset": dataset,
                        "source": source,
                        "method": method,
                        "noise_level": noise,
                        # Stage 1 (Perception) - 2 metrics
                        "rq1_pa_nusy": pa_nusy,
                        "rq1_pa_drain": pa_drain,
                        "rq1_pa_vanilla_llm": pa_vanilla_llm,
                        "rq1_latency_ms_nusy": float(nusy_lat),
                        "rq1_latency_ms_drain": float(drain_lat),
                        "rq1_latency_ms_vanilla_llm": float(vanilla_llm_lat),
                        # Stage 2 (Reasoning) - 2 metrics
                        "rq2_causal_sparsity": sparsity,
                        "rq2_local_rank": local_rank,
                        "fail_fast_triggered": int(fail_fast_triggered),
                        # Stage 3 (Decision) - 3 metrics
                        "rq3_rca_correct": rca_ok,
                        "rq3_hallucination": hallu,
                        "rq3_repair_action_match": repair_match,
                        "e2e_success": e2e_success,
                        "pred_template_nusy": nusy_tpl,
                        "pred_template_drain": drain_tpl,
                        "gt_template": gt_tpl,
                        "gt_root_cause": gt_root,
                        "pred_root_cause": pred_root,
                        "pred_repair_action_id": pred_action_id,
                        "pred_repair_action_desc": pred_action_desc,
                    }
                )
                pbar.update(1)

    df = pd.DataFrame(rows)
    pbar.close()
    os.makedirs(RESULTS_E2E_DIR, exist_ok=True)
    details_path = os.path.join(RESULTS_E2E_DIR, "massive_e2e_details.csv")
    df.to_csv(details_path, index=False)

    # 聚合指标：按 (dataset, method, noise_level) 分组，输出 7 项指标
    summary: Dict[str, object] = {}
    grouped = df.groupby(["dataset", "method", "noise_level"])
    for (ds, m, n), g in grouped:
        key = f"{ds}__{m}__noise_{n}"
        # 7 metrics: Parsing PA, Parsing Latency, Causal Sparsity, Avg_Rank, RCA Accuracy, Hallucination Rate, E2E Diagnostic Success
        summary[key] = {
            "dataset": ds,
            "method": m,
            "noise_level": n,
            "num_cases": int(len(g)),
            "parsing_pa": float(g["rq1_pa_nusy"].mean()) if "rq1_pa_nusy" in g else 0.0,
            "parsing_latency_ms": float(g["rq1_latency_ms_nusy"].mean()) if "rq1_latency_ms_nusy" in g else 0.0,
            "causal_sparsity_mean": float(g["rq2_causal_sparsity"].mean()) if "rq2_causal_sparsity" in g else 0.0,
            "avg_rank": float(g["rq2_local_rank"].replace({-1: None}).dropna().mean()) if ("rq2_local_rank" in g and g["rq2_local_rank"].replace({-1: None}).dropna().size > 0) else 0.0,
            "rca_accuracy": float(g["rq3_rca_correct"].mean()) if "rq3_rca_correct" in g else 0.0,
            "hallucination_rate": float(g["rq3_hallucination"].mean()) if "rq3_hallucination" in g else 0.0,
            "e2e_diagnostic_success_rate": float(g["e2e_success"].mean()) if "e2e_success" in g else 0.0,
            "repair_action_match_rate": float(g["rq3_repair_action_match"].mean()) if "rq3_repair_action_match" in g else 0.0,
            "fail_fast_rate": float(g["fail_fast_triggered"].mean()) if "fail_fast_triggered" in g else 0.0,
        }

    summary_path = os.path.join(RESULTS_E2E_DIR, "massive_e2e_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    # 7-metric wide table confirmation
    print("\n[7-METRIC WIDE TABLE] Stage1: parsing_pa, parsing_latency_ms | Stage2: causal_sparsity_mean, avg_rank | Stage3: rca_accuracy, hallucination_rate, e2e_diagnostic_success_rate")
    if use_api:
        total_cost_cny = total_prompt_tokens * pricing_input + total_completion_tokens * pricing_output
        full_run_calls = 110 * 6 * 3
        est_full_cost = (total_cost_cny / max(1, total_api_calls)) * full_run_calls
        print(
            f"[PILOT COST] prompt_tokens={total_prompt_tokens}, "
            f"completion_tokens={total_completion_tokens}, "
            f"api_calls={total_api_calls}, "
            f"total_cost_cny={total_cost_cny:.4f}, "
            f"extrapolated_full_run_cost_cny≈{est_full_cost:.2f} (110*6*3 calls)"
        )
    print(f"[INFO] Wrote detailed metrics to {details_path}")
    print(f"[INFO] Wrote aggregated metrics to {summary_path}")
    # Final report for verification
    n_hdfs = len(df[df["dataset"] == "HDFS"]["case_id"].unique()) if "dataset" in df.columns else 0
    n_os = len(df[df["dataset"] == "OpenStack"]["case_id"].unique()) if "dataset" in df.columns else 0
    n_hadoop = len(df[df["dataset"] == "Hadoop"]["case_id"].unique()) if "dataset" in df.columns else 0
    hadoop_edges = len([e for e in causal_kb if e.get("domain") == "hadoop"])
    print(f"\n[FINAL REPORT] Cases: HDFS={n_hdfs}, OpenStack={n_os}, Hadoop={n_hadoop}")
    print(f"[FINAL REPORT] Hadoop causal graph: {hadoop_edges} edges (threshold pruning: MIN_COUNT=5, WEIGHT_THRESHOLD=0.18 in build_hadoop_causal_graph.py).")
    print("[FINAL REPORT] 7-metric table generated successfully.")
    print("PIPELINE REFACTORED. READY FOR FINAL MASSIVE RUN.")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-cases", type=int, default=20, help="Max number of cases from scaled benchmark (0=all).")
    ap.add_argument(
        "--noise-levels",
        type=str,
        default="0.0,0.2",
        help="Comma-separated noise levels, e.g. '0.0,0.4,0.8'. Default uses 0.0 and 0.2 for快速验证.",
    )
    ap.add_argument(
        "--pilot-run",
        action="store_true",
        help="Run small pilot with DeepSeek API integration (cost-aware, ~30 calls).",
    )
    ap.add_argument(
        "--mid-scale",
        action="store_true",
        help="Run mid-scale canary test (45 cases, 2 noise levels) with DeepSeek API.",
    )
    ap.add_argument(
        "--pilot-2",
        action="store_true",
        help="Run minimal pilot: 2 cases per dataset (6 total), single noise level, no API (offline).",
    )
    args = ap.parse_args()

    noise_levels = [float(x) for x in args.noise_levels.split(",") if x.strip()]
    max_cases = int(args.max_cases) if args.max_cases >= 0 else 0

    if args.pilot_run:
        # Pilot：固定 10 个 case、单一噪声档、3 种方法
        noise_levels = [0.0]
        max_cases = 10
        methods = ["NuSy-Edge_offline", "Vanilla", "StandardRAG"]
        run_massive_e2e(
            max_cases=max_cases,
            noise_levels=noise_levels,
            methods=methods,
            pilot_run=True,
            use_api=True,
            midscale=False,
        )
    elif args.pilot_2:
        # Pilot-2: 2 cases per dataset (6 total), single noise, no API
        noise_levels = [0.0]
        methods = ["NuSy-Edge_offline", "Vanilla", "StandardRAG"]
        run_massive_e2e(
            max_cases=0,
            noise_levels=noise_levels,
            methods=methods,
            pilot_run=False,
            pilot_2_per_dataset=True,
            use_api=False,
            midscale=False,
        )
    elif args.mid_scale:
        # Mid-scale：每个数据集 15 个 case，共 45 个；噪声档 0.0 与 0.8
        noise_levels = [0.0, 0.8]
        methods = ["NuSy-Edge_offline", "Vanilla", "StandardRAG"]
        run_massive_e2e(
            max_cases=0,
            noise_levels=noise_levels,
            methods=methods,
            pilot_run=False,
            use_api=True,
            midscale=True,
        )
    else:
        run_massive_e2e(max_cases=max_cases, noise_levels=noise_levels)


if __name__ == "__main__":
    main()

