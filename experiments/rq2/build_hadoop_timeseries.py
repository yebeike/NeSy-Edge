import os
import sys
import json
from datetime import datetime
from typing import Dict, List

import pandas as pd
from tqdm import tqdm

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.perception.drain_parser import DrainParser  # type: ignore


RAW_HADOOP_ROOT = os.path.join(_PROJECT_ROOT, "data", "raw", "Hadoop")
DATA_PROCESSED = os.path.join(_PROJECT_ROOT, "data", "processed")
OUT_MATRIX = os.path.join(DATA_PROCESSED, "hadoop_timeseries.csv")
OUT_ID_MAP = os.path.join(DATA_PROCESSED, "hadoop_id_map.json")

# 时间窗口粒度：与 HDFS/OpenStack 保持一致，这里用 1 分钟
WINDOW = "1min"


def _iter_hadoop_log_lines(root: str):
    """
    遍历 data/raw/Hadoop 下所有 application_*/container 等日志文件，逐行返回原始日志。
    """
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            # 保守一些，只看 .log 和 .out 文件
            if not (name.endswith(".log") or name.endswith(".out")):
                continue
            fpath = os.path.join(dirpath, name)
            try:
                with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        yield line.rstrip("\n")
            except (IOError, OSError):
                continue


def build_hadoop_timeseries():
    if not os.path.isdir(RAW_HADOOP_ROOT):
        raise FileNotFoundError(f"Hadoop raw root not found: {RAW_HADOOP_ROOT}")

    parser = DrainParser()
    events: List[Dict[str, object]] = []
    eid_map: Dict[str, str] = {}  # Template -> EventID
    next_eid_idx = 1

    # 预估总行数很难，这里让 tqdm 动态更新
    print(f"[*] Scanning Hadoop logs under {RAW_HADOOP_ROOT} ...")
    for line in tqdm(list(_iter_hadoop_log_lines(RAW_HADOOP_ROOT)), desc="Parsing Hadoop", unit="line"):
        if not line.strip():
            continue
        # 1) 解析时间戳：形如 "2015-10-17 21:46:38,510 ..."
        ts = None
        try:
            # 取前 19 个字符（到秒），忽略毫秒
            prefix = line[:19]
            ts = datetime.strptime(prefix, "%Y-%m-%d %H:%M:%S")
        except Exception:
            # 尝试在整行中搜索
            import re

            m = re.search(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", line)
            if m:
                try:
                    ts = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
                except Exception:
                    ts = None
        if ts is None:
            continue

        # 2) 模板解析：直接把整行交给 DrainParser 做抽象
        try:
            tpl = parser.parse(line)
        except Exception:
            tpl = ""
        if not tpl:
            continue

        # 3) EventID 分配：Template -> E1, E2, ...
        if tpl not in eid_map:
            eid_map[tpl] = f"E{next_eid_idx}"
            next_eid_idx += 1
        eid = eid_map[tpl]

        events.append({"Timestamp": ts, "EventID": eid})

    if not events:
        print("[WARN] No Hadoop events parsed; timeseries will be empty.")
        return

    df = pd.DataFrame(events)
    df.set_index("Timestamp", inplace=True)

    print("[*] Aggregating Hadoop events into timeseries matrix ...")
    # 行：时间窗口；列：EventID；值：次数
    matrix = df.groupby("EventID").resample(WINDOW).size().unstack(level=0).fillna(0)
    matrix.sort_index(inplace=True)

    os.makedirs(DATA_PROCESSED, exist_ok=True)
    matrix.to_csv(OUT_MATRIX)

    # 生成 EventID -> Template 的映射
    id_to_tpl = {eid: tpl for tpl, eid in eid_map.items()}
    with open(OUT_ID_MAP, "w", encoding="utf-8") as f:
        json.dump(id_to_tpl, f, indent=2, ensure_ascii=False)

    print(f"[INFO] Hadoop timeseries matrix saved to: {OUT_MATRIX}")
    print(f"[INFO] Hadoop ID map saved to: {OUT_ID_MAP}")
    print(f"[INFO] Matrix shape: {matrix.shape}, num_event_ids: {len(id_to_tpl)}")


def main():
    build_hadoop_timeseries()


if __name__ == "__main__":
    main()

