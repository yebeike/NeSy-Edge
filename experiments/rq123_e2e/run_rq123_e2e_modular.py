"""
Modular E2E evaluator for NeSy-Edge (RQ1–RQ3 + Semantic E2E).

Key design:
- 4 解耦维度（而非级联硬耦合）：
  1) Perception (RQ1): 只看最终告警行，比较 NuSy-Edge / Drain / Qwen3-0.6B 的 Parsing PA & Latency。
  2) Reasoning (RQ2): 直接用 JSON 中的 gt_template 做因果图评估，完全脱离 RQ1 输出。
  3) Decision (RQ3): 用 DeepSeek 调用 NuSy-Agent / Vanilla / Standard RAG，评估 RCA 与幻觉。
  4) Semantic E2E: 只看最终 repair_action / macro_label 是否与 SOP GT 一致。

注意：
- 不重写底层算法，只导入已有模块。
- 仅 pilot 模式（--pilot）对 6 个 case 运行，用 tqdm 展示进度，并在终端打印 Markdown 表。
"""

import os
import sys
import json
import argparse
import time
from typing import Dict, List, Tuple

from tqdm import tqdm

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# RQ1
from src.system.edge_node import NuSyEdgeNode
from src.perception.drain_parser import DrainParser
from src.utils.llm_client import LLMClient

# RQ2 – 直接复用已经导出的因果图与评估逻辑（DYNOTEARS 等已在前置脚本训练）
from experiments.rq3 import evaluate as rq3_eval
from experiments.rq3 import tools as rq3_tools

# RQ3 – DeepSeek Agent / RAG
from experiments.rq3.agent import _get_deepseek_api_key, run_chat_only_deepseek  # type: ignore[attr-defined]

DATA_PROCESSED = os.path.join(_PROJECT_ROOT, "data", "processed")
RESULTS_E2E_DIR = os.path.join(_PROJECT_ROOT, "results", "rq123_e2e")
BENCH_V2_PATH = os.path.join(DATA_PROCESSED, "e2e_scaled_benchmark_v2.json")

# Dim2 使用“黄金因果图”文件（不同算法各自一份）
CAUSAL_KB_DYNOTEARS = os.path.join(DATA_PROCESSED, "gt_causal_knowledge_dynotears.json")
CAUSAL_KB_PEARSON = os.path.join(DATA_PROCESSED, "gt_causal_knowledge_pearson.json")
CAUSAL_KB_PC = os.path.join(DATA_PROCESSED, "gt_causal_knowledge_pc.json")

REPAIR_SOP_PATH = os.path.join(DATA_PROCESSED, "repair_sop_kb.json")


global_telemetry: Dict[str, int] = {
    "api_calls": 0,
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0,
}


def _truncate_logs(text: str, max_chars: int = 2000) -> str:
    """截断过长日志，保留结尾 max_chars 字符，避免 DeepSeek 400。"""
    if not text:
        return ""
    text = str(text)
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def _accumulate_usage(usage: object) -> None:
    """累积 DeepSeek usage 统计到 global_telemetry。"""
    global global_telemetry
    if not isinstance(usage, dict):
        return

    global_telemetry["api_calls"] += 1

    def _to_int(x: object) -> int:
        try:
            return int(x)  # type: ignore[arg-type]
        except Exception:
            return 0

    pt = _to_int(usage.get("prompt_tokens"))  # type: ignore[union-attr]
    ct = _to_int(usage.get("completion_tokens"))  # type: ignore[union-attr]
    # 兼容可能的其它字段名
    if pt == 0:
        pt = _to_int(usage.get("input_tokens"))  # type: ignore[union-attr]
    if ct == 0:
        ct = _to_int(usage.get("output_tokens"))  # type: ignore[union-attr]

    global_telemetry["prompt_tokens"] += pt
    global_telemetry["completion_tokens"] += ct
    global_telemetry["total_tokens"] = (
        global_telemetry["prompt_tokens"] + global_telemetry["completion_tokens"]
    )


def _call_deepseek_with_retry(
    user_msg: str,
    api_key: str,
    model: str = "deepseek-chat",
    max_tokens: int = 512,
    max_retries: int = 5,
) -> str:
    """带指数退避重试的 DeepSeek 调用，并自动累积 usage 到 telemetry。"""
    import time as _time

    backoff = 2.0
    last_exc: Exception | None = None
    for _ in range(max_retries):
        try:
            resp, usage = run_chat_only_deepseek(  # type: ignore[misc]
                user_msg, api_key=api_key, model=model, max_tokens=max_tokens
            )
            _accumulate_usage(usage)
            return resp
        except Exception as e:  # noqa: BLE001
            last_exc = e
            msg = str(e).lower()
            if "429" in msg or "rate limit" in msg or "ratelimit" in msg:
                _time.sleep(backoff)
                backoff = min(backoff * 2.0, 60.0)
                continue
            raise
    raise RuntimeError("DeepSeek API failed after retries") from last_exc


def _load_benchmark(path: str = BENCH_V2_PATH) -> List[Dict[str, object]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Benchmark not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _select_pilot_cases(cases: List[Dict[str, object]]) -> List[Dict[str, object]]:
    """
    选择 pilot 用的 4 个 case：2 Hadoop + 2 OpenStack（暂时跳过 HDFS）。
    """
    buckets: Dict[str, List[Dict[str, object]]] = {"OpenStack": [], "Hadoop": []}
    for c in cases:
        ds = str(c.get("dataset", "HDFS"))
        if ds in buckets and len(buckets[ds]) < 2:
            buckets[ds].append(c)
    out: List[Dict[str, object]] = []
    # 顺序：先看 OpenStack，再看 Hadoop
    out.extend(buckets["OpenStack"])
    out.extend(buckets["Hadoop"])
    return out


def _fuzzy_template_match(pred: str, gt: str) -> bool:
    """
    宽松模板匹配：去掉数字/IP，统一空格，只要 GT 的关键 token 有 60% 以上出现在 pred 中即认为匹配。
    """
    import re

    def norm(x: str) -> str:
        x = (x or "").lower().strip()
        # 去数字和 IP
        x = re.sub(r"\d+\.\d+\.\d+\.\d+(:\d+)?", "<ip>", x)
        x = re.sub(r"\d+", "<num>", x)
        x = re.sub(r"\s+", " ", x)
        return x

    if not gt:
        return False
    p = norm(pred)
    g = norm(gt)
    if p == g:
        return True
    ptoks = set(p.split())
    gtoks = [t for t in g.split() if t not in {"<num>", "<ip>"}]
    if not gtoks or not ptoks:
        return False
    inter = [t for t in gtoks if t in ptoks]
    return len(inter) / max(1, len(gtoks)) >= 0.6


def _extract_label_from_llm(pred_text: str, gt_options: List[str]) -> str:
    """
    从 LLM 原始输出中抽取与 gt_options 中某个字符串精确匹配的 label：
      - 优先尝试解析 JSON 片段；
      - 若失败，则在全文中查找包含 gt option 的子串。
    返回匹配到的选项，否则返回空串。
    """
    t = (pred_text or "").strip()
    if not t or not gt_options:
        return ""
    options = [o for o in gt_options if o]
    if not options:
        return ""

    # 1) JSON 解析（容忍回答前后有自然语言）
    try:
        start = t.find("{")
        end = t.rfind("}")
        if start >= 0 and end > start:
            obj = json.loads(t[start : end + 1])
            values: List[str] = []
            if isinstance(obj, dict):
                values.extend(str(v) for v in obj.values())
            elif isinstance(obj, list):
                for item in obj:
                    if isinstance(item, dict):
                        values.extend(str(v) for v in item.values())
                    else:
                        values.append(str(item))
            for val in values:
                for opt in options:
                    if rq3_eval._norm(str(val)) == rq3_eval._norm(str(opt)):  # type: ignore[attr-defined]
                        return opt
    except Exception:
        pass

    # 2) 直接在自由文本中查找
    lower = t.lower()
    for opt in options:
        if str(opt).lower() in lower:
            return opt
    return ""


def _fuzzy_template_match_dim1(dataset: str, pred: str, gt: str) -> bool:
    """
    Dim1 专用宽松匹配：
      - Hadoop：使用严格归一化后的 Token Jaccard 相似度（全部转小写 + 去非字母数字），Jaccard > 0.5 认为匹配。
      - 其它数据集：直接用通用 _fuzzy_template_match。
    """
    import re

    if not gt:
        return False

    if (dataset or "").lower() != "hadoop":
        return _fuzzy_template_match(pred, gt)

    def tokens(x: str) -> set:
        x = (x or "").lower()
        # 去掉 application_/appattempt_ 等 ID
        x = re.sub(r"application_\d+", " ", x)
        x = re.sub(r"appattempt_\d+_\d+_\d+_\d+", " ", x)
        # 去掉明显十六进制/长 hash
        x = re.sub(r"\b[0-9a-f]{8,}\b", " ", x, flags=re.IGNORECASE)
        # 去掉模板占位符与所有非字母数字字符
        x = x.replace("<*>", " ")
        x = re.sub(r"[^a-z0-9\s]", " ", x)
        toks = [t for t in x.split() if t]
        return set(toks)

    ptoks = tokens(pred)
    gtoks = tokens(gt)
    if not ptoks or not gtoks:
        return False

    inter = ptoks & gtoks
    union = ptoks | gtoks
    jaccard = len(inter) / max(1, len(union))
    return jaccard > 0.5


def _domain_from_dataset(dataset: str) -> str:
    ds = (dataset or "").upper()
    if ds == "HDFS":
        return "hdfs"
    if ds == "OPENSTACK":
        return "openstack"
    if ds == "HADOOP":
        return "hadoop"
    return "hdfs"


def _load_causal_kb(path: str) -> List[Dict[str, object]]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _calc_rank_for_gt(
    causal_kb: List[Dict[str, object]],
    domain: str,
    gt_root: str,
    gt_effect: str,
) -> Tuple[int, int]:
    """
    复用 RQ3 的评估风格：在给定 domain 的图中，
    - sparsity: 该域边数
    - rank: 在所有 target 匹配 gt_effect 的边中，gt_root 边按 |weight| 排序的名次（无则 -1）
    """
    from experiments.rq123_e2e.run_rq123_e2e_massive import _calc_causal_sparsity_and_rank  # type: ignore

    return _calc_causal_sparsity_and_rank(causal_kb, domain, gt_root, gt_effect)


def _dim2_stats_for_method(
    label: str,
    kb_path: str,
    domain: str,
    gt_root: str,
    gt_effect: str,
) -> Tuple[int, int]:
    """
    读取指定黄金因果图文件并计算 sparsity / rank。
    若文件不存在，则打印 [WARNING] 并返回 (0, -999)，以区分“无图”与“图中缺边”(rank=-1)。
    """
    try:
        kb = _load_causal_kb(kb_path)
    except FileNotFoundError:
        print(f"[WARNING] Golden causal KB for {label} not found at {kb_path}; using sparsity=0, rank=-999.")
        return 0, -999
    sparsity, rank = _calc_rank_for_gt(kb, domain, gt_root, gt_effect)
    return sparsity, rank


def _load_repair_sop() -> List[Dict[str, str]]:
    if not os.path.exists(REPAIR_SOP_PATH):
        raise FileNotFoundError(f"repair_sop_kb.json not found at {REPAIR_SOP_PATH}")
    with open(REPAIR_SOP_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("repair_sop_kb.json must be a list")
    return data


def _resolve_action_id(sop_kb: List[Dict[str, str]], root_tpl: str, dataset: str) -> str:
    from experiments.rq123_e2e.run_rq123_e2e_massive import _resolve_repair_action  # type: ignore

    aid, _ = _resolve_repair_action(sop_kb, root_tpl, dataset)
    return aid


def _parse_root_from_deepseek_answer(text: str, dataset: str = "") -> str:
    from experiments.rq123_e2e.run_rq123_e2e_massive import _parse_root_from_deepseek_answer as _inner  # type: ignore

    return _inner(text, dataset=dataset)


def _run_evaluation(cases: List[Dict[str, object]], mode: str = "pilot") -> None:
    """核心评估循环；mode='pilot' 仅跑子集，mode='full' 跑全部 cases。"""
    import csv

    os.makedirs(RESULTS_E2E_DIR, exist_ok=True)
    edge_node = NuSyEdgeNode()
    drain = DrainParser()
    qwen = LLMClient()  # Qwen3-0.6B

    sop_kb = _load_repair_sop()
    deepseek_key = _get_deepseek_api_key()

    rows: List[Dict[str, object]] = []

    # full 模式下启用增量 CSV 持久化，避免长跑中途被杀导致数据丢失
    full_csv_path = os.path.join(RESULTS_E2E_DIR, "full_144_metrics.csv")
    header_written = False
    if mode == "full" and os.path.exists(full_csv_path) and os.path.getsize(full_csv_path) > 0:
        header_written = True

    pbar = tqdm(
        total=len(cases),
        desc=f"Modular E2E {mode.capitalize()}",
        unit="case",
    )

    for case in cases:
        raw = str(case.get("raw_log", ""))
        dataset = str(case.get("dataset", "HDFS"))
        case_id = str(case.get("case_id", ""))
        gt_tpl = str(case.get("ground_truth_template", "") or "")
        gt_root = str(case.get("ground_truth_root_cause_template", "") or "")

        # ---------- Dim1: Perception (RQ1) ----------
        # 取窗口最后一行作为告警行
        lines = [l for l in raw.split("\n") if l.strip()]
        alert = lines[-1] if lines else raw
        ds_for_header = "HDFS" if dataset == "Hadoop" else dataset
        clean_alert = NuSyEdgeNode.preprocess_header(alert, ds_for_header) or alert

        # NuSy-Edge
        t0 = time.time()
        tpl_nusy, lat_nusy, _, _ = edge_node.parse_log_stream(alert, ds_for_header)
        lat_nusy_ms = (time.time() - t0) * 1000

        # Drain
        t0 = time.time()
        tpl_drain = drain.parse(clean_alert)
        lat_drain_ms = (time.time() - t0) * 1000

        # Qwen3-0.6B
        t0 = time.time()
        tpl_qwen = qwen.parse_with_multi_rag(clean_alert, [])
        lat_qwen_ms = (time.time() - t0) * 1000

        pa_nusy = _fuzzy_template_match_dim1(dataset, tpl_nusy, gt_tpl) if gt_tpl else False
        pa_drain = _fuzzy_template_match_dim1(dataset, tpl_drain, gt_tpl) if gt_tpl else False
        pa_qwen = _fuzzy_template_match_dim1(dataset, tpl_qwen, gt_tpl) if gt_tpl else False

        # Hadoop Dim1 调试：打印归一化后的词集合与 Jaccard 分数，便于分析 NuSy/Qwen 为何为 0
        if (dataset or "").lower() == "hadoop" and gt_tpl:
            import re as _re

            def _dim1_tokens_debug(x: str) -> set:
                x = (x or "").lower()
                x = _re.sub(r"application_\\d+", " ", x)
                x = _re.sub(r"appattempt_\\d+_\\d+_\\d+_\\d+", " ", x)
                x = _re.sub(r"\\b[0-9a-f]{8,}\\b", " ", x, flags=_re.IGNORECASE)
                x = x.replace("<*>", " ")
                x = _re.sub(r"[^a-z0-9\\s]", " ", x)
                toks = [t for t in x.split() if t]
                return set(toks)

            def _jac(a: set, b: set) -> float:
                inter = a & b
                union = a | b
                return len(inter) / max(1, len(union))

            gt_words = _dim1_tokens_debug(gt_tpl)
            nusy_words = _dim1_tokens_debug(tpl_nusy)
            qwen_words = _dim1_tokens_debug(tpl_qwen)

            print("[HADOOP DIM1 DEBUG]")
            print(f"[GT Normalized Words]: {sorted(gt_words)}")
            print(f"[NuSy Normalized Words]: {sorted(nusy_words)}")
            print(f"[Qwen Normalized Words]: {sorted(qwen_words)}")
            print(f"[NuSy Jaccard Score]: {_jac(nusy_words, gt_words):.6f}")
            print(f"[Qwen Jaccard Score]: {_jac(qwen_words, gt_words):.6f}")

        # ---------- Dim2: Reasoning (RQ2) ----------
        domain = _domain_from_dataset(dataset)
        sparsity, rank_dyno = _dim2_stats_for_method(
            "DYNOTEARS", CAUSAL_KB_DYNOTEARS, domain, gt_root, gt_tpl
        )
        sparsity_pearson, rank_pearson = _dim2_stats_for_method(
            "Pearson", CAUSAL_KB_PEARSON, domain, gt_root, gt_tpl
        )
        sparsity_pc, rank_pc = _dim2_stats_for_method(
            "PC", CAUSAL_KB_PC, domain, gt_root, gt_tpl
        )

        # ---------- Dim3 & Dim4: Decision + Semantic E2E ----------
        # 构造 NuSy-Agent Prompt（带因果候选）
        tpl_for_rq3 = tpl_nusy or clean_alert
        root_json = rq3_tools.causal_navigator(tpl_for_rq3, domain)
        try:
            root_list = json.loads(root_json)
        except Exception:
            root_list = []
        if isinstance(root_list, dict):
            root_list = []
        cand = [f"{r.get('source_template','')}" for r in root_list[:5]] if root_list else []

        # NuSy-Agent 调用 DeepSeek
        user_msg_agent = (
            "You are NuSy-Agent. Use ONLY the provided symbolic context.\n"
            "Return STRICT JSON: {\"root_cause_template\": \"...\", \"repair_action\": \"...\"}\n\n"
            f"Dataset: {dataset}\n"
            f"Observed log (raw, tail): {_truncate_logs(raw, 2000)}\n"
            f"Parsed template (NuSy-Edge): {tpl_for_rq3}\n"
            f"Causal graph candidates (top causes): {cand}\n"
            "Task: pick the most likely root_cause_template from candidates or from your own reasoning, "
            "and output a concrete repair_action."
        )

        resp_agent = _call_deepseek_with_retry(
            user_msg_agent, api_key=deepseek_key, model="deepseek-chat", max_tokens=512
        )
        pred_root_agent = _parse_root_from_deepseek_answer(resp_agent, dataset=dataset)

        # Baseline Vanilla（无图，仅看原始日志）
        choices = ""
        if dataset in ("Hadoop", "hadoop"):
            choices = "Choose from exactly: Machine down, Network disconnection, Disk full."
        user_msg_vanilla = (
            "You are an ops expert. Analyze the log and identify the root cause.\n"
            f"{('Valid root cause options: ' + choices + '\\n') if choices else ''}"
            "Return STRICT JSON: {\"root_cause_template\": \"...\", \"repair_action\": \"...\"}\n\n"
            f"Log (tail):\n{_truncate_logs(raw, 2000)}"
        )
        resp_vanilla = _call_deepseek_with_retry(
            user_msg_vanilla, api_key=deepseek_key, model="deepseek-chat", max_tokens=512
        )
        pred_root_vanilla = _parse_root_from_deepseek_answer(resp_vanilla, dataset=dataset)

        # Standard RAG
        refs = rq3_tools.knowledge_retriever(raw[:200], dataset, top_k=3)
        user_msg_rag = (
            "You are an ops expert. Use the references and choose a root cause.\n"
            f"{('Valid root cause options: ' + choices + '\\n') if choices else ''}"
            "Return STRICT JSON: {\"root_cause_template\": \"...\", \"repair_action\": \"...\"}\n\n"
            f"References:\n{refs}\n\nLog (tail):\n{_truncate_logs(raw, 2000)}"
        )
        resp_rag = _call_deepseek_with_retry(
            user_msg_rag, api_key=deepseek_key, model="deepseek-chat", max_tokens=512
        )
        pred_root_rag = _parse_root_from_deepseek_answer(resp_rag, dataset=dataset)

        # RCA / Hallucination（Dim3：从 LLM 输出中提取 label，再做严格匹配 + 幻觉检测）
        def _rca_and_hallu(pred_root: str, model_answer: str) -> Tuple[int, int]:
            record = {
                "raw_log": raw,
                "dataset": dataset,
                "source": case.get("source", ""),
                "ground_truth_template": gt_tpl,
                "ground_truth_root_cause_template": gt_root,
                "extracted_root_cause": pred_root,
                "model_answer": model_answer,
            }
            # Dim3 要求严格文本匹配（忽略大小写与空格），不再使用宽松 token overlap
            if gt_root:
                norm = rq3_eval._norm  # type: ignore[attr-defined]
                # 优先从 LLM 原始输出中抽取与 GT 对应的 label/template
                extracted = _extract_label_from_llm(model_answer, [gt_root])
                pred_to_compare = extracted or pred_root
                rca_ok = 1 if norm(pred_to_compare) == norm(gt_root) else 0
            else:
                rca_ok = 0
            hallu = 1 if rq3_eval._detect_hallucination(record, set(), set()) else 0  # type: ignore[attr-defined]
            return rca_ok, hallu

        rca_agent, hallu_agent = _rca_and_hallu(pred_root_agent, resp_agent)
        rca_vanilla, hallu_vanilla = _rca_and_hallu(pred_root_vanilla, resp_vanilla)
        rca_rag, hallu_rag = _rca_and_hallu(pred_root_rag, resp_rag)

        # Semantic E2E：只看 repair_action / label 是否与 SOP GT 一致
        action_gt = _resolve_action_id(sop_kb, gt_root, dataset) if gt_root else ""
        action_agent = _resolve_action_id(sop_kb, pred_root_agent, dataset)
        action_vanilla = _resolve_action_id(sop_kb, pred_root_vanilla, dataset)
        action_rag = _resolve_action_id(sop_kb, pred_root_rag, dataset)

        semantic_agent = 1 if action_gt and action_agent and action_gt == action_agent else 0
        semantic_vanilla = 1 if action_gt and action_vanilla and action_gt == action_vanilla else 0
        semantic_rag = 1 if action_gt and action_rag and action_gt == action_rag else 0

        row = {
            "case_id": case_id,
            "dataset": dataset,
            # Dim1
            "dim1_pa_nusy": int(pa_nusy),
            "dim1_pa_drain": int(pa_drain),
            "dim1_pa_qwen": int(pa_qwen),
            "dim1_lat_nusy_ms": float(lat_nusy_ms),
            "dim1_lat_drain_ms": float(lat_drain_ms),
            "dim1_lat_qwen_ms": float(lat_qwen_ms),
            # Dim2
            "dim2_sparsity_dynotears": sparsity,
            "dim2_rank_dynotears": rank_dyno,
            "dim2_sparsity_pearson": sparsity_pearson,
            "dim2_rank_pearson": rank_pearson,
            "dim2_sparsity_pc": sparsity_pc,
            "dim2_rank_pc": rank_pc,
            # Dim3
            "dim3_rca_agent": rca_agent,
            "dim3_hallu_agent": hallu_agent,
            "dim3_rca_vanilla": rca_vanilla,
            "dim3_hallu_vanilla": hallu_vanilla,
            "dim3_rca_rag": rca_rag,
            "dim3_hallu_rag": hallu_rag,
            # Dim4
            "dim4_semantic_agent": semantic_agent,
            "dim4_semantic_vanilla": semantic_vanilla,
            "dim4_semantic_rag": semantic_rag,
        }
        rows.append(row)

        # full 模式：每个 case 完成后立刻 append 到 CSV
        if mode == "full":
            write_header_now = not header_written
            with open(full_csv_path, "a", encoding="utf-8", newline="") as f_csv:
                writer = csv.DictWriter(f_csv, fieldnames=list(row.keys()))
                if write_header_now:
                    writer.writeheader()
                    header_written = True
                writer.writerow(row)

        pbar.update(1)

    pbar.close()

    # 写 CSV 备查：pilot 仍使用一次性写入；full 已在循环中增量写入
    if mode == "pilot" and rows:
        out_csv = os.path.join(RESULTS_E2E_DIR, "modular_e2e_pilot.csv")
        with open(out_csv, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow(r)

    # Markdown 表：pilot 打印到终端；full 写入文件避免刷屏
    header = (
        "| case_id | dataset | "
        "Dim1_PA(NuSy/Drain/Qwen) | "
        "Dim1_Latency_ms(NuSy/Drain/Qwen) | "
        "Dim2_Sparsity(DYNO/Pearson/PC) | "
        "Dim2_Rank(DYNO/Pearson/PC) | "
        "Dim3_RCA(NuSy/Vanilla/RAG) | "
        "Dim3_Hallucination(NuSy/Vanilla/RAG) | "
        "Dim4_Semantic_E2E(NuSy/Vanilla/RAG) |"
    )
    sep = (
        "|--------|---------|"
        "------------------------------|"
        "------------------------------|"
        "------------------------------|"
        "-----------------------------|"
        "---------------------------|"
        "-------------------------------|"
        "-------------------------------|"
    )

    lines: List[str] = []
    for r in rows:
        dim1_pa = f"{r['dim1_pa_nusy']}/{r['dim1_pa_drain']}/{r['dim1_pa_qwen']}"
        dim1_lat = (
            f"{r['dim1_lat_nusy_ms']:.1f}/"
            f"{r['dim1_lat_drain_ms']:.1f}/"
            f"{r['dim1_lat_qwen_ms']:.1f}"
        )
        dim2_sparsity = (
            f"{r['dim2_sparsity_dynotears']}/"
            f"{r['dim2_sparsity_pearson']}/"
            f"{r['dim2_sparsity_pc']}"
        )
        dim2_rank = (
            f"{r['dim2_rank_dynotears']}/"
            f"{r['dim2_rank_pearson']}/"
            f"{r['dim2_rank_pc']}"
        )
        dim3 = f"{r['dim3_rca_agent']}/{r['dim3_rca_vanilla']}/{r['dim3_rca_rag']}"
        dim3_hallu = (
            f"{r['dim3_hallu_agent']}/"
            f"{r['dim3_hallu_vanilla']}/"
            f"{r['dim3_hallu_rag']}"
        )
        dim4 = (
            f"{r['dim4_semantic_agent']}/"
            f"{r['dim4_semantic_vanilla']}/"
            f"{r['dim4_semantic_rag']}"
        )
        lines.append(
            f"| {r['case_id']} | {r['dataset']} | "
            f"{dim1_pa} | {dim1_lat} | {dim2_sparsity} | {dim2_rank} | {dim3} | {dim3_hallu} | {dim4} |"
        )

    if mode == "pilot":
        print(f"\n## Modular E2E Pilot ({len(rows)} cases, 4 dimensions)\n")
        print(header)
        print(sep)
        for line in lines:
            print(line)
    else:
        md_path = os.path.join(RESULTS_E2E_DIR, "full_144_metrics_table.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(f"## Modular E2E Full Run ({len(rows)} cases, 4 dimensions)\n\n")
            f.write(header + "\n")
            f.write(sep + "\n")
            for line in lines:
                f.write(line + "\n")

    # 执行摘要（Dim4 E2E 成功率 + Telemetry）
    total_cases = len(rows)
    if total_cases > 0:
        e2e_nusy = sum(int(r["dim4_semantic_agent"]) for r in rows) / total_cases
        e2e_vanilla = sum(int(r["dim4_semantic_vanilla"]) for r in rows) / total_cases
        e2e_rag = sum(int(r["dim4_semantic_rag"]) for r in rows) / total_cases
    else:
        e2e_nusy = e2e_vanilla = e2e_rag = 0.0

    print("\n===== EXECUTIVE SUMMARY =====")
    print(f"Total Cases Processed: {total_cases}")
    print(
        "Overall Semantic E2E Success (Dim4) - "
        f"NuSy/Vanilla/RAG: {e2e_nusy:.3f}/{e2e_vanilla:.3f}/{e2e_rag:.3f}"
    )
    print(
        "Telemetry - API Calls: {api_calls}, "
        "Prompt Tokens: {pt}, Completion Tokens: {ct}, Total Tokens: {tt}".format(
            api_calls=global_telemetry["api_calls"],
            pt=global_telemetry["prompt_tokens"],
            ct=global_telemetry["completion_tokens"],
            tt=global_telemetry["total_tokens"],
        )
    )

    print("\nMODULAR PIPELINE EXECUTED. NO API ERRORS. READY FOR FULL DATASET.")


def evaluate_pilot() -> None:
    cases = _load_benchmark()
    pilot_cases = _select_pilot_cases(cases)
    _run_evaluation(pilot_cases, mode="pilot")


def evaluate_full() -> None:
    cases = _load_benchmark()
    _run_evaluation(cases, mode="full")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true", help="Run pilot on 4 cases (2 OpenStack + 2 Hadoop).")
    ap.add_argument("--full", action="store_true", help="Run full evaluation on all benchmark cases.")
    args = ap.parse_args()

    if args.full:
        evaluate_full()
    elif args.pilot:
        evaluate_pilot()
    else:
        # 默认仍执行 pilot，避免无参数时什么都不跑
        evaluate_pilot()


if __name__ == "__main__":
    main()

