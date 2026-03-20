import os
import sys
import time
import json
from typing import Any, Dict, List, Tuple

from tqdm import tqdm
import torch  # 本地 Qwen 使用

"""
Offline robustness evaluation for NeSy-Edge on RQ1/RQ2 (Dim1 & Dim2)
with log-level noise injection.

约束：
- 仅使用本地组件：NuSyEdgeNode / DrainParser / LLMClient(Qwen3-0.6B) / NoiseInjector / Golden Graph JSON。
- 不进行任何 DeepSeek / OpenAI 或其它远程 LLM API 调用。
- 只读现有数据与图，不改写基准文件。

评估维度：
- Dim1: Parsing PA & Latency (NuSy / Drain / Qwen)
- Dim2: Causal Graph sparsity & Rank (DYNOTEARS / Pearson / PC)

噪声注入：
- 复用 src.utils.noise_injector.NoiseInjector（先前 RQ1 实验的噪声逻辑）。
- 噪声水平：noise_levels = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
"""


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


from src.system.edge_node import NuSyEdgeNode  # type: ignore
from src.perception.drain_parser import DrainParser  # type: ignore
from src.utils.llm_client import LLMClient  # type: ignore
from src.utils.noise_injector import NoiseInjector  # type: ignore

from experiments.rq123_e2e.run_rq123_e2e_modular import (  # type: ignore
    _fuzzy_template_match_dim1,
    _domain_from_dataset,
)
from experiments.rq123_e2e.run_rq123_e2e_massive import (  # type: ignore
    _calc_causal_sparsity_and_rank,
)


DATA_PROCESSED = os.path.join(_PROJECT_ROOT, "data", "processed")
BENCH_V2_PATH = os.path.join(DATA_PROCESSED, "e2e_scaled_benchmark_v2.json")

CAUSAL_KB_DYNOTEARS = os.path.join(DATA_PROCESSED, "gt_causal_knowledge_dynotears.json")
CAUSAL_KB_PEARSON = os.path.join(DATA_PROCESSED, "gt_causal_knowledge_pearson.json")
CAUSAL_KB_PC = os.path.join(DATA_PROCESSED, "gt_causal_knowledge_pc.json")

# Micro-sandbox 噪声水平（节省时间）
# NOISE_LEVELS = [0.0, 0.5, 1.0]
NOISE_LEVELS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


def _load_json(path: str) -> Any:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _pick_tail_alert_line(raw_log: str) -> str:
    """近似 Dim1 行为：取窗口最后一条非空日志行。"""
    lines = [l for l in str(raw_log).split("\n") if l.strip()]
    return lines[-1] if lines else str(raw_log)


def _build_qwen_refs(
    cases: List[Dict[str, Any]],
    max_refs_per_dataset: int = 10,
) -> Dict[str, List[Tuple[str, str]]]:
    """
    为 Qwen 构造多参考 few-shot 列表：
      - 对每个数据集，取最多 max_refs_per_dataset 条 (alert_tail, gt_template)。
    """
    refs: Dict[str, List[Tuple[str, str]]] = {"HDFS": [], "OpenStack": [], "Hadoop": []}
    for c in cases:
        ds = str(c.get("dataset", "")).strip()
        if ds not in refs:
            continue
        if len(refs[ds]) >= max_refs_per_dataset:
            continue
        raw_log = str(c.get("raw_log", "") or "")
        gt_tpl = str(c.get("ground_truth_template", "") or "")
        if not raw_log or not gt_tpl:
            continue
        alert_line = _pick_tail_alert_line(raw_log)
        refs[ds].append((alert_line, gt_tpl))
    return refs


def _qwen_parse_with_refs(
    qwen: LLMClient,
    target_log: str,
    refs_list: List[Tuple[str, str]],
) -> str:
    """
    自定义的 Qwen 调用：
      - 等价于 parse_with_multi_rag，但增加 repetition_penalty=1.15。
      - refs_list: [(ref_log, ref_template), ...]
    """
    system_content = (
        "You are a Log Parser.\n"
        "Task: Transform the [Input Log] into an Event Template, following the style of the [References].\n"
        "Rules:\n"
        "1. Replace dynamic variables (Numbers, IPs, BlockIDs, UUIDs) with <*>.\n"
        "2. Keep static keywords like the References.\n"
        "3. Output ONLY the template string."
    )
    ref_block = "\n".join(
        f"Log: {r[0]}\nTemplate: {r[1]}" for r in refs_list
    )
    user_content = f"--- References ---\n{ref_block}\n\n--- Input Log ---\n{target_log}"
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]
    try:
        text = qwen.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except Exception:
        text = qwen.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    inputs = qwen.tokenizer([text], return_tensors="pt").to(qwen.device)  # type: ignore[attr-defined]
    with torch.no_grad():  # type: ignore[name-defined]
        generated_ids = qwen.model.generate(  # type: ignore[attr-defined]
            inputs.input_ids,
            max_new_tokens=64,
            do_sample=False,
            pad_token_id=qwen.tokenizer.eos_token_id,  # type: ignore[attr-defined]
            eos_token_id=qwen.tokenizer.eos_token_id,  # type: ignore[attr-defined]
            repetition_penalty=1.15,
        )
    gen_tokens = generated_ids[0][len(inputs.input_ids[0]) :]
    response = qwen.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()  # type: ignore[attr-defined]
    # 复用客户端内部清洗逻辑
    response = qwen._symbolic_cleanup(response)  # type: ignore[attr-defined]
    return response.strip()


def _compute_rank(
    causal_kb: List[Dict[str, Any]],
    domain: str,
    gt_root: str,
    effect_tpl: str,
) -> int:
    """
    使用与 modular pipeline 一致的 helper 计算 Dim2 rank。
    - effect_tpl: 这里使用「预测模板」而不是 benchmark 中的 GT 模板。
    - 返回值:
        >0: 在所有 target==effect_tpl 的边中的名次（1 表示 Rank1）
        -1: 图中没有该边 / or 无匹配
        -999: 图文件缺失（此处不期望出现）
    """
    if not effect_tpl:
        return -1
    _, rank = _calc_causal_sparsity_and_rank(causal_kb, domain, gt_root, effect_tpl)
    try:
        return int(rank)
    except Exception:
        return -1


def _compute_sparsity_per_domain(
    edges: List[Dict[str, Any]],
) -> Dict[str, int]:
    """
    统计每个 domain 的总边数，用于 Dim2 稀疏度指标。
    """
    out: Dict[str, int] = {}
    for e in edges:
        dom = str(e.get("domain", "")).strip().lower()
        out[dom] = out.get(dom, 0) + 1
    return out


def main() -> None:
    # 1) 加载基础数据与 Golden Graphs
    cases: List[Dict[str, Any]] = _load_json(BENCH_V2_PATH)
    dyn_edges: List[Dict[str, Any]] = _load_json(CAUSAL_KB_DYNOTEARS)
    pear_edges: List[Dict[str, Any]] = _load_json(CAUSAL_KB_PEARSON)
    pc_edges: List[Dict[str, Any]] = _load_json(CAUSAL_KB_PC)

    dyn_sparsity = _compute_sparsity_per_domain(dyn_edges)
    pear_sparsity = _compute_sparsity_per_domain(pear_edges)
    pc_sparsity = _compute_sparsity_per_domain(pc_edges)

    datasets = ["HDFS", "OpenStack", "Hadoop"]

    # 2) 初始化本地组件
    nusy = NuSyEdgeNode()
    drain = DrainParser()
    qwen = LLMClient()

    # 3) 构造 Qwen few-shot 参考
    qwen_refs = _build_qwen_refs(cases, max_refs_per_dataset=10)

    # 4) 统计结构：按 (dataset, noise_level) 聚合
    dim_stats: Dict[Tuple[str, float], Dict[str, Any]] = {}
    for ds in datasets:
        for nl in NOISE_LEVELS:
            key = (ds, nl)
            dim_stats[key] = {
                "count": 0,
                # Dim1 PA & Lat
                "nusy_hit": 0,
                "drain_hit": 0,
                "qwen_hit": 0,
                "nusy_lat_sum": 0.0,
                "drain_lat_sum": 0.0,
                "qwen_lat_sum": 0.0,
                # Dim2 Rank
                "dyn_rank1": 0,
                "pear_rank1": 0,
                "pc_rank1": 0,
                "dyn_rank_sum": 0.0,
                "pear_rank_sum": 0.0,
                "pc_rank_sum": 0.0,
                "dyn_rank_cnt": 0,
                "pear_rank_cnt": 0,
                "pc_rank_cnt": 0,
            }

    # 5) 噪声水平 -> 数据集 -> case 主循环
    for nl in NOISE_LEVELS:
        injector = NoiseInjector(injection_rate=nl, seed=2026)

        for ds in datasets:
            # 为该数据集筛选 case
            ds_cases = [c for c in cases if str(c.get("dataset", "")).strip() == ds]
            if not ds_cases:
                continue

            # 针对当前 (dataset, noise_level) 的进度条
            pbar = tqdm(
                ds_cases,
                desc=f"[Robustness] ds={ds} noise={nl}",
                unit="case",
            )

            # Qwen few-shot 参考
            refs_list = qwen_refs.get(ds, [])

            for c in pbar:
                key = (ds, nl)
                stat = dim_stats[key]
                stat["count"] += 1

                raw_log = str(c.get("raw_log", "") or "")
                gt_tpl = str(c.get("ground_truth_template", "") or "")
                gt_root = str(c.get("ground_truth_root_cause_template", "") or "")

                # 5.1 构造告警行并注入噪声
                alert_line = _pick_tail_alert_line(raw_log)

                # NoiseInjector 仅区分 OpenStack/HDFS；Hadoop 映射至 HDFS 通道
                inj_dataset = "HDFS" if ds == "Hadoop" else ds
                noisy_alert = injector.inject(alert_line, dataset_type=inj_dataset)

                # 5.2 Dim1: Parsing（NuSy / Drain / Qwen）
                # NuSy-Edge
                t0 = time.time()
                try:
                    nusy_pred, nusy_lat, _, _ = nusy.parse_log_stream(noisy_alert, ds)
                except Exception:
                    nusy_pred = ""
                    nusy_lat = (time.time() - t0) * 1000.0
                stat["nusy_lat_sum"] += float(nusy_lat)
                if _fuzzy_template_match_dim1(ds, nusy_pred, gt_tpl):
                    stat["nusy_hit"] += 1

                # Drain
                t0 = time.time()
                try:
                    drain_pred = drain.parse(noisy_alert)
                except Exception:
                    drain_pred = ""
                drain_lat = (time.time() - t0) * 1000.0
                stat["drain_lat_sum"] += float(drain_lat)
                if _fuzzy_template_match_dim1(ds, drain_pred, gt_tpl):
                    stat["drain_hit"] += 1

                # Qwen（少样本 + repetition_penalty=1.15）
                t0 = time.time()
                try:
                    if refs_list:
                        qwen_pred = _qwen_parse_with_refs(qwen, noisy_alert, refs_list)
                    else:
                        qwen_pred = qwen.parse_with_multi_rag(noisy_alert, [])
                except Exception:
                    qwen_pred = ""
                qwen_lat = (time.time() - t0) * 1000.0
                stat["qwen_lat_sum"] += float(qwen_lat)
                if _fuzzy_template_match_dim1(ds, qwen_pred, gt_tpl):
                    stat["qwen_hit"] += 1

                # 5.3 Dim2: Graph Sparsity & Rank (Pred tpl vs gt_root)
                domain = _domain_from_dataset(ds)

                # 使用 NuSy 预测模板作为 "effect"（更贴近 NeSy-Edge 的实际路径）
                eff_tpl = nusy_pred or drain_pred

                # DYNOTEARS
                r_dyn = _compute_rank(dyn_edges, domain, gt_root, eff_tpl)
                if r_dyn > 0:
                    stat["dyn_rank_sum"] += float(r_dyn)
                    stat["dyn_rank_cnt"] += 1
                    if r_dyn == 1:
                        stat["dyn_rank1"] += 1

                # Pearson
                r_pear = _compute_rank(pear_edges, domain, gt_root, eff_tpl)
                if r_pear > 0:
                    stat["pear_rank_sum"] += float(r_pear)
                    stat["pear_rank_cnt"] += 1
                    if r_pear == 1:
                        stat["pear_rank1"] += 1

                # PC
                r_pc = _compute_rank(pc_edges, domain, gt_root, eff_tpl)
                if r_pc > 0:
                    stat["pc_rank_sum"] += float(r_pc)
                    stat["pc_rank_cnt"] += 1
                    if r_pc == 1:
                        stat["pc_rank1"] += 1

    # 6) 打印 Markdown 汇总表
    print("## Offline Robustness Evaluation (Dim1 & Dim2, Local Only)")
    print()
    print(
        "| Dataset | Noise Level | "
        "PA (NuSy/Drain/Qwen) | "
        "Latency ms (NuSy/Drain/Qwen) | "
        "Sparsity (DYNO/Pear/PC) | "
        "Avg_Rank & Rank1 (DYNO/Pear/PC) |"
    )
    print(
        "|---------|-------------|"
        "------------------------|"
        "-------------------------------|"
        "----------------------------|"
        "---------------------------------|"
    )

    for ds in datasets:
        dom = _domain_from_dataset(ds)
        s_dyn = dyn_sparsity.get(dom, 0)
        s_pear = pear_sparsity.get(dom, 0)
        s_pc = pc_sparsity.get(dom, 0)

        for nl in NOISE_LEVELS:
            key = (ds, nl)
            stat = dim_stats[key]
            cnt = max(1, stat["count"])

            # Dim1 PA
            pa_nusy = stat["nusy_hit"] / cnt
            pa_drain = stat["drain_hit"] / cnt
            pa_qwen = stat["qwen_hit"] / cnt

            # Dim1 Lat
            lat_nusy = stat["nusy_lat_sum"] / cnt
            lat_drain = stat["drain_lat_sum"] / cnt
            lat_qwen = stat["qwen_lat_sum"] / cnt

            # Dim2 Avg Rank（排除 rank <= 0）
            def _avg(sum_v: float, n: int) -> float:
                return sum_v / n if n > 0 else 0.0

            avg_dyn = _avg(stat["dyn_rank_sum"], stat["dyn_rank_cnt"])
            avg_pear = _avg(stat["pear_rank_sum"], stat["pear_rank_cnt"])
            avg_pc = _avg(stat["pc_rank_sum"], stat["pc_rank_cnt"])

            row_pa = f"{pa_nusy:.3f}/{pa_drain:.3f}/{pa_qwen:.3f}"
            row_lat = f"{lat_nusy:.1f}/{lat_drain:.1f}/{lat_qwen:.1f}"
            row_sparsity = f"{s_dyn}/{s_pear}/{s_pc}"
            row_rank = (
                f"{avg_dyn:.2f} & {stat['dyn_rank1']} | "
                f"{avg_pear:.2f} & {stat['pear_rank1']} | "
                f"{avg_pc:.2f} & {stat['pc_rank1']}"
            )

            print(
                f"| {ds:<7} | {nl:11.1f} | "
                f"{row_pa:22} | "
                f"{row_lat:29} | "
                f"{row_sparsity:26} | "
                f"{row_rank} |"
            )


if __name__ == "__main__":
    main()

