import os
import sys
import json
import argparse
import time
from typing import Dict, List, Tuple

import pandas as pd


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


from src.system.edge_node import NuSyEdgeNode
from experiments.rq3 import tools as rq3_tools
from experiments.rq3 import evaluate as rq3_eval
from experiments.rq123_e2e import hadoop_loader


DATA_PROCESSED = os.path.join(_PROJECT_ROOT, "data", "processed")
RESULTS_E2E_DIR = os.path.join(_PROJECT_ROOT, "results", "rq123_e2e")
REPAIR_SOP_PATH = os.path.join(DATA_PROCESSED, "repair_sop_kb.json")


def _norm(s: str) -> str:
    return rq3_eval._norm(s)  # type: ignore[attr-defined]


def _load_rq3_cases() -> List[Dict[str, object]]:
    path = os.path.join(DATA_PROCESSED, "rq3_test_set.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"rq3_test_set.json not found at {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_causal_edges_and_templates(
    causal_path: str,
) -> Tuple[set, set]:
    valid_edges = rq3_eval._load_causal_edges(causal_path)  # type: ignore[attr-defined]
    valid_templates = rq3_eval._load_valid_templates(causal_path)  # type: ignore[attr-defined]
    return valid_edges, valid_templates


def _load_repair_sop(path: str = REPAIR_SOP_PATH) -> List[Dict[str, str]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"repair_sop_kb.json not found at {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("repair_sop_kb.json must be a list of entries")
    return data


def _domain_from_dataset(dataset: str) -> str:
    return "hdfs" if dataset.upper() == "HDFS" else "openstack"


def _resolve_repair_action(
    sop_kb: List[Dict[str, str]],
    root_tpl: str,
    dataset: str,
) -> Tuple[str, str]:
    """
    根据根因模板与数据集，从 repair_sop_kb.json 中解析出标准修复动作。
    匹配策略：dataset 一致且 pattern 与 root_tpl 在归一化后有明显 token 重叠。
    返回 (action_id, action_text)，若未命中则返回 ("", "")。
    """
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
        # 直接重用 RQ3 的匹配逻辑，保证与 RCA 一致
        if rq3_eval._rca_match(root_n, pat_n):  # type: ignore[attr-defined]
            score = len(pat_n)
        else:
            # 简单 token 重叠度
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


def _calc_causal_sparsity_and_rank(
    causal_kb: List[Dict[str, object]],
    domain: str,
    gt_root: str,
    gt_effect: str,
) -> Tuple[int, int]:
    """
    Reuse causal_knowledge.json as a static graph and define:
      - sparsity: number of edges in this domain
      - rank: for this case, rank of the true (root, effect) edge among all edges
              that share the same target (sorted by |weight| desc). If edge is
              missing, rank = -1.
    """
    domain = (domain or "hdfs").lower()
    gt_root_n = _norm(gt_root)
    gt_effect_n = _norm(gt_effect)

    edges_domain = [e for e in causal_kb if e.get("domain") == domain]
    sparsity = len(edges_domain)

    # Filter edges with same target (effect)
    same_target = []
    for e in edges_domain:
        t = _norm(e.get("target_template", ""))
        if not t:
            continue
        if rq3_eval._rca_match(t, gt_effect) or rq3_eval._rca_match(gt_effect, t):  # type: ignore[attr-defined]
            same_target.append(e)

    if not same_target:
        return sparsity, -1

    # Rank edges by absolute weight
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


def _iter_phase1_logs_for_dataset(dataset: str, max_logs: int = 200) -> List[str]:
    """
    Phase1 预热：从原始日志中抽取少量正常日志，用于 RQ1 cache/KB 预热。
    为了不依赖 RQ1 脚本内部函数，这里实现一个轻量版扫描：
      - HDFS: data/raw/HDFS_v1 下的 .log 文件
      - OpenStack: data/raw/OpenStack_2 下的 .log 文件
      - Hadoop: 通过 hadoop_loader 的 normal application（若存在 abnormal_label）
    若路径不存在，则返回空列表，保证 E2E 脚本可在缺失原始日志的环境下运行。
    """
    logs: List[str] = []
    root_raw = os.path.join(_PROJECT_ROOT, "data", "raw")
    ds = dataset.upper()

    if ds == "HDFS":
        base = os.path.join(root_raw, "HDFS_v1")
        if os.path.isdir(base):
            for name in sorted(os.listdir(base)):
                if not name.endswith(".log"):
                    continue
                path = os.path.join(base, name)
                if not os.path.isfile(path):
                    continue
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        if len(logs) >= max_logs:
                            return logs
                        line = line.rstrip("\n")
                        if line.strip():
                            logs.append(line)
    elif ds == "OPENSTACK":
        base = os.path.join(root_raw, "OpenStack_2")
        if os.path.isdir(base):
            for name in sorted(os.listdir(base)):
                if not name.endswith(".log"):
                    continue
                path = os.path.join(base, name)
                if not os.path.isfile(path):
                    continue
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        if len(logs) >= max_logs:
                            return logs
                        line = line.rstrip("\n")
                        if line.strip():
                            logs.append(line)
    elif ds == "HADOOP":
        # 利用 abnormal_label.txt 中的 normal application 作为 Phase1 正常运行
        abnormal_path = os.path.join(root_raw, "Hadoop", "abnormal_label.txt")
        h_root = os.path.join(root_raw, "Hadoop")
        if os.path.exists(abnormal_path) and os.path.isdir(h_root):
            try:
                rows = hadoop_loader.parse_abnormal_label_file(abnormal_path)
                for r in rows:
                    if len(logs) >= max_logs:
                        break
                    if r.get("label", "").lower() != "normal":
                        continue
                    app_logs = hadoop_loader.iter_hadoop_application_logs(h_root, r["application_id"])
                    for line in app_logs:
                        if len(logs) >= max_logs:
                            break
                        if line.strip():
                            logs.append(line)
            except Exception:
                # 若 Hadoop 数据缺失或结构不匹配，静默跳过
                return logs
    return logs


def _phase1_warmup_rq1(edge_node: NuSyEdgeNode) -> None:
    """
    Phase1：针对 HDFS / OpenStack / Hadoop 做轻量预热，以模拟真实部署中的长期运行缓存状态。
    仅调用 parse_log_stream，不写任何结果文件。
    """
    for ds in ["HDFS", "OpenStack", "Hadoop"]:
        logs = _iter_phase1_logs_for_dataset(ds, max_logs=200)
        if not logs:
            continue
        for raw in logs:
            try:
                edge_node.parse_log_stream(raw, ds if ds != "Hadoop" else "HDFS")
            except Exception:
                # 预热阶段允许个别解析失败，不影响主流程
                continue


def run_e2e(limit: int = 10, dataset_filter: str = "", causal_path: str = "") -> pd.DataFrame:
    """
    E2E pipeline over rq3_test_set:
      - RQ1: NuSy-Edge real-time parsing (template + latency)
      - RQ2: causal_knowledge.json as static graph (sparsity + local rank)
      - RQ3 (offline): log_parser + causal_navigator to pick root cause,
        reuse RQ3 evaluation logic for RCA & hallucination.
    """
    cases = _load_rq3_cases()
    if dataset_filter:
        cases = [c for c in cases if str(c.get("dataset", "")).upper() == dataset_filter.upper()]
    if limit > 0:
        cases = cases[:limit]
    if not cases:
        raise RuntimeError("No cases to run in E2E pipeline.")

    causal_path = causal_path or rq3_tools.get_causal_knowledge_path()
    if not os.path.exists(causal_path):
        raise FileNotFoundError(f"causal_knowledge.json not found at {causal_path}")
    with open(causal_path, "r", encoding="utf-8") as f:
        causal_kb = json.load(f)
    valid_edges, valid_templates = _load_causal_edges_and_templates(causal_path)

    sop_kb = _load_repair_sop(REPAIR_SOP_PATH)

    edge_node = NuSyEdgeNode()
    # Phase1：先做一次轻量预热，模拟长期运行中的缓存与 KB 状态
    _phase1_warmup_rq1(edge_node)

    rows: List[Dict[str, object]] = []
    for idx, c in enumerate(cases):
        raw = str(c.get("raw_log", ""))
        dataset = str(c.get("dataset", "HDFS"))
        gt_tpl = str(c.get("ground_truth_template", ""))
        gt_root = str(c.get("ground_truth_root_cause_template", ""))
        source = str(c.get("source", ""))
        case_id = str(c.get("case_id", f"case_{idx}"))

        # ---------- RQ1: parsing ----------
        t_start = time.time()
        try:
            pred_tpl, lat_ms, _, _ = edge_node.parse_log_stream(raw, dataset)
        except Exception as e:
            pred_tpl, lat_ms = f"(parse_error: {e})", (time.time() - t_start) * 1000
        pa_flag = 1 if rq3_eval._rca_match(pred_tpl, gt_tpl) else 0  # type: ignore[attr-defined]

        # ---------- RQ2: graph sparsity + local rank ----------
        domain = _domain_from_dataset(dataset)
        sparsity, local_rank = _calc_causal_sparsity_and_rank(causal_kb, domain, gt_root, gt_tpl)

        # ---------- RQ3 (offline): causal_navigator root cause ----------
        tpl_for_rq3 = pred_tpl or raw
        root_json = rq3_tools.causal_navigator(tpl_for_rq3, domain)
        try:
            root_list = json.loads(root_json)
        except Exception:
            root_list = []
        if isinstance(root_list, dict):
            # error case
            root_list = []
        extracted_root = ""
        if root_list:
            extracted_root = str(root_list[0].get("source_template", "")).strip()

        # Reuse RQ3 evaluation helpers for RCA & hallucination
        pred_record = {
            "raw_log": raw,
            "dataset": dataset,
            "source": source,
            "ground_truth_template": gt_tpl,
            "ground_truth_root_cause_template": gt_root,
            "extracted_root_cause": extracted_root,
            "model_answer": "",  # offline, no free-form text
        }
        rca_ok = 0
        if gt_root and gt_root.lower() != "unknown":
            if rq3_eval._rca_match(extracted_root, gt_root):  # type: ignore[attr-defined]
                rca_ok = 1
        hallu = 1 if rq3_eval._detect_hallucination(pred_record, valid_templates, valid_edges) else 0  # type: ignore[attr-defined]
        # ---------- Self-Healing：根据根因模板解析修复动作 ----------
        pred_action_id, pred_action = _resolve_repair_action(sop_kb, extracted_root, dataset)
        gt_action_id, _ = _resolve_repair_action(sop_kb, gt_root, dataset)
        repair_match = 1 if (gt_action_id and pred_action_id and gt_action_id == pred_action_id) else 0

        # ---------- E2E 成功判定（包含修复动作） ----------
        e2e_success = 1 if (
            pa_flag == 1
            and sparsity > 0
            and local_rank > 0
            and rca_ok == 1
            and repair_match == 1
        ) else 0

        rows.append(
            {
                "case_id": case_id,
                "dataset": dataset,
                "source": source,
                "rq1_parsing_pa": pa_flag,
                "rq1_latency_ms": float(lat_ms),
                "rq2_causal_sparsity": sparsity,
                "rq2_local_rank": local_rank,
                "rq3_rca_correct": rca_ok,
                "rq3_hallucination": hallu,
                "rq3_repair_action_id_pred": pred_action_id,
                "rq3_repair_action_id_gt": gt_action_id,
                "rq3_repair_action_desc_pred": pred_action,
                "rq3_repair_action_match": repair_match,
                "e2e_success": e2e_success,
                "pred_template": pred_tpl,
                "gt_template": gt_tpl,
                "gt_root_cause": gt_root,
                "extracted_root_cause": extracted_root,
            }
        )

    df = pd.DataFrame(rows)
    os.makedirs(RESULTS_E2E_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_E2E_DIR, "e2e_summary.csv")
    df.to_csv(out_path, index=False)

    # Also print a compact aggregated view
    agg = {
        "num_cases": len(df),
        "parsing_pa": float(df["rq1_parsing_pa"].mean()) if "rq1_parsing_pa" in df else 0.0,
        "parsing_latency_ms": float(df["rq1_latency_ms"].mean()) if "rq1_latency_ms" in df else 0.0,
        "causal_sparsity_mean": float(df["rq2_causal_sparsity"].mean()) if "rq2_causal_sparsity" in df else 0.0,
        "avg_rank": float(df["rq2_local_rank"].replace({-1: None}).dropna().mean()) if "rq2_local_rank" in df else 0.0,
        "rca_accuracy": float(df["rq3_rca_correct"].mean()) if "rq3_rca_correct" in df else 0.0,
        "hallucination_rate": float(df["rq3_hallucination"].mean()) if "rq3_hallucination" in df else 0.0,
        "repair_action_match_rate": float(df["rq3_repair_action_match"].mean())
        if "rq3_repair_action_match" in df
        else 0.0,
        "e2e_success_rate": float(df["e2e_success"].mean()) if "e2e_success" in df else 0.0,
    }
    # 保留原版本文件名的同时，新增 v2 版本以体现 Self-Healing 语义
    agg_path_v1 = os.path.join(RESULTS_E2E_DIR, "e2e_summary_agg.json")
    agg_path_v2 = os.path.join(RESULTS_E2E_DIR, "e2e_summary_agg_v2.json")
    with open(agg_path_v1, "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2)
    with open(agg_path_v2, "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2)

    print(json.dumps(agg, indent=2))
    print(f"[INFO] Wrote per-case E2E metrics to {out_path}")
    print(f"[INFO] Wrote aggregated E2E metrics to {agg_path_v1}")
    print(f"[INFO] Wrote aggregated E2E metrics (v2, with repair actions) to {agg_path_v2}")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=8, help="Number of rq3 cases to run (0 = all).")
    ap.add_argument("--dataset", type=str, default="", help="Optional dataset filter: HDFS or OpenStack.")
    ap.add_argument("--causal-path", type=str, default="", help="Override path to causal_knowledge.json.")
    args = ap.parse_args()

    run_e2e(limit=args.limit, dataset_filter=args.dataset, causal_path=args.causal_path)


if __name__ == "__main__":
    main()

