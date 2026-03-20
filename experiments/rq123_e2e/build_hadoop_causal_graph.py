import os
import sys
import json
from collections import Counter, defaultdict
from typing import Dict, List, Tuple


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


from src.perception.drain_parser import DrainParser
from src.system.edge_node import NuSyEdgeNode
from experiments.rq123_e2e import hadoop_loader


RAW_ROOT = os.path.join(_PROJECT_ROOT, "data", "raw")
DATA_PROCESSED = os.path.join(_PROJECT_ROOT, "data", "processed")
# 原始 RQ2 因果图（只读）
CAUSAL_KB_BASE = os.path.join(DATA_PROCESSED, "causal_knowledge.json")
# 本脚本导出的 Hadoop 稀疏图会写入新的文件，避免覆盖旧数据
CAUSAL_KB_HADOOP = os.path.join(DATA_PROCESSED, "causal_knowledge_hadoop_pruned.json")


def _build_hadoop_template_sequences(max_apps: int = 200) -> List[List[str]]:
    """
    为每个 Hadoop application 构造模板序列：
      - 使用 NuSy-Edge 的 header 预处理去除噪声前缀
      - 使用 DrainParser 解析为模板字符串
      - 只保留长度 >= 2 的序列
    """
    hadoop_root = os.path.join(RAW_ROOT, "Hadoop")
    label_path = os.path.join(hadoop_root, "abnormal_label.txt")
    if not (os.path.isdir(hadoop_root) and os.path.exists(label_path)):
        print("[WARN] Hadoop raw dir or abnormal_label.txt not found; skip Hadoop causal graph export.")
        return []

    rows = hadoop_loader.parse_abnormal_label_file(label_path)
    edge_node = NuSyEdgeNode()
    drain = DrainParser()

    seqs: List[List[str]] = []
    for row in rows:
        if len(seqs) >= max_apps:
            break
        app_id = row["application_id"]
        try:
            lines = hadoop_loader.iter_hadoop_application_logs(hadoop_root, app_id)
        except FileNotFoundError:
            continue
        if not lines:
            continue
        tpl_seq: List[str] = []
        for raw in lines:
            clean = NuSyEdgeNode.preprocess_header(raw, "HDFS")
            if not clean:
                clean = raw
            try:
                tpl = drain.parse(clean)
            except Exception:
                continue
            if tpl:
                tpl_seq.append(tpl)
        if len(tpl_seq) >= 2:
            seqs.append(tpl_seq)
    return seqs


def _build_cooccurrence_edges(seqs: List[List[str]]) -> List[Dict[str, object]]:
    """
    使用简单的顺序共现近似 DYNOTEARS 的时间依赖：
      - 对每条序列中相邻模板对 (Ti -> Tj) 计数
      - 将出现次数视为权重，并阈值过滤掉极少出现的边

    输出与现有 causal_knowledge.json 一致的结构：
      {
        "domain": "hadoop",
        "source_template": "<Ti>",
        "relation": "instantly_triggers",
        "target_template": "<Tj>",
        "weight": <float>
      }
    """
    pair_counter: Counter = Counter()
    # 为了让图更加“连通但仍然稀疏”，不仅统计相邻 (i, i+1)，也统计邻近的 (i, i+2), (i, i+3) 等局部顺序对。
    MAX_LAG = 3
    for seq in seqs:
        L = len(seq)
        for i in range(L - 1):
            s = seq[i]
            if not s:
                continue
            for j in range(i + 1, min(L, i + 1 + MAX_LAG)):
                t = seq[j]
                if not t or s == t:
                    continue
                pair_counter[(s, t)] += 1

    if not pair_counter:
        return []

    max_cnt = max(pair_counter.values())
    # Threshold pruning + sparsity filter: only strong conditional dependencies.
    # 通过较低的 MIN_COUNT 与中等 w_threshold，让 Hadoop 图落在 50~150 条边的健康稀疏范围。
    MIN_COUNT = 3
    WEIGHT_THRESHOLD = 0.08
    edges: List[Dict[str, object]] = []
    for (s, t), c in pair_counter.items():
        if c < MIN_COUNT:
            continue
        w = float(c) / float(max_cnt)
        if w <= WEIGHT_THRESHOLD:
            continue
        edges.append(
            {
                "domain": "hadoop",
                "source_template": s,
                "relation": "instantly_triggers",
                "target_template": t,
                "weight": round(w, 4),
            }
        )
    return edges


def append_hadoop_causal_edges():
    """
    基于 Hadoop 模板序列建立一个轻量因果图，并与原始 RQ2 因果图合并：
      - 读取 data/processed/causal_knowledge.json 作为基础（只读）
      - 去除其中旧的 domain='hadoop' 边
      - 追加新的 hadoop 边
      - 将结果写入 data/processed/causal_knowledge_hadoop_pruned.json
    """
    seqs = _build_hadoop_template_sequences()
    if not seqs:
        print("[WARN] No Hadoop template sequences built; skip updating causal_knowledge.json.")
        return

    new_edges = _build_cooccurrence_edges(seqs)
    if not new_edges:
        print("[WARN] No Hadoop causal edges mined; skip updating causal_knowledge.json.")
        return

    # 优先从已经存在的 pruned 文件读取；否则从原始 causal_knowledge.json 读取
    if os.path.exists(CAUSAL_KB_HADOOP):
        with open(CAUSAL_KB_HADOOP, "r", encoding="utf-8") as f:
            kb = json.load(f)
        if not isinstance(kb, list):
            kb = []
    elif os.path.exists(CAUSAL_KB_BASE):
        with open(CAUSAL_KB_BASE, "r", encoding="utf-8") as f:
            kb = json.load(f)
        if not isinstance(kb, list):
            kb = []
    else:
        kb = []

    # 去除旧的 hadoop domain 边
    kb = [e for e in kb if e.get("domain") != "hadoop"]
    kb.extend(new_edges)

    os.makedirs(os.path.dirname(CAUSAL_KB_HADOOP) or ".", exist_ok=True)
    with open(CAUSAL_KB_HADOOP, "w", encoding="utf-8") as f:
        json.dump(kb, f, indent=4, ensure_ascii=False)

    print(f"[INFO] Appended {len(new_edges)} Hadoop causal edges to {CAUSAL_KB_HADOOP}")


def main():
    append_hadoop_causal_edges()


if __name__ == "__main__":
    main()

