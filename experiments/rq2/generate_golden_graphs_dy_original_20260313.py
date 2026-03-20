"""
Generate 'original' DYNOTEARS golden graphs (2026-03-13, with tqdm).

Differences from modified generator:
- Uses StandardScaler directly on full timeseries (no zero-variance filtering / custom tweaks).
- Uses per-domain hyperparameters closer to the earlier unified exporter:
  - HDFS:      lambda_w=0.025, lambda_a=0.05, thr=0.30
  - OpenStack: lambda_w=0.030, lambda_a=0.06, thr=0.40
  - Hadoop:    lambda_w=0.025, lambda_a=0.05, thr=0.30
- Adds explicit tqdm progress for each domain.

Output:
- data/processed/gt_causal_knowledge_dynotears_original_20260313.json
"""

import os
import sys
import json
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from tqdm import tqdm

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

_EXPERIMENTS_DIR = os.path.join(_PROJECT_ROOT, "experiments")
if _EXPERIMENTS_DIR not in sys.path:
    sys.path.insert(0, _EXPERIMENTS_DIR)

from run_rq2_unified_knowledge_exporter import dynotears_engine  # type: ignore

warnings.filterwarnings("ignore")

DATA_PROCESSED = os.path.join(_PROJECT_ROOT, "data", "processed")
RAW_HDFS_PRE = os.path.join(_PROJECT_ROOT, "data", "raw", "HDFS_v1", "preprocessed")
RAW_HDFS_ALT = os.path.join(_PROJECT_ROOT, "data", "raw", "HDFS")

OUT_DYNO_ORIG = os.path.join(DATA_PROCESSED, "gt_causal_knowledge_dynotears_original_20260313.json")


def _load_hdfs_timeseries() -> Tuple[pd.DataFrame, Dict[str, str]]:
    ts_path = os.path.join(DATA_PROCESSED, "hdfs_timeseries.csv")
    if not os.path.exists(ts_path):
        raise FileNotFoundError(ts_path)
    df = pd.read_csv(ts_path, index_col=0)

    hdfs_tpl_csv = os.path.join(RAW_HDFS_PRE, "HDFS.log_templates.csv")
    if not os.path.exists(hdfs_tpl_csv):
        alt = os.path.join(RAW_HDFS_ALT, "HDFS_templates.csv")
        if os.path.exists(alt):
            hdfs_tpl_csv = alt
        else:
            raise FileNotFoundError(
                "Neither HDFS_v1/preprocessed/HDFS.log_templates.csv nor data/raw/HDFS/HDFS_templates.csv found"
            )
    tpl_df = pd.read_csv(hdfs_tpl_csv)
    id_col = "EventId" if "EventId" in tpl_df.columns else ("EventID" if "EventID" in tpl_df.columns else None)
    if id_col is None or "EventTemplate" not in tpl_df.columns:
        raise ValueError(f"Unexpected HDFS template CSV format: {hdfs_tpl_csv}")
    id_map = dict(zip(tpl_df[id_col], tpl_df["EventTemplate"]))
    return df, id_map


def _load_openstack_timeseries() -> Tuple[pd.DataFrame, Dict[str, str]]:
    ts_path = os.path.join(DATA_PROCESSED, "openstack_refined_ts.csv")
    if not os.path.exists(ts_path):
        raise FileNotFoundError(ts_path)
    df = pd.read_csv(ts_path, index_col=0)
    id_map_path = os.path.join(DATA_PROCESSED, "openstack_id_map.json")
    if not os.path.exists(id_map_path):
        raise FileNotFoundError(id_map_path)
    with open(id_map_path, "r", encoding="utf-8") as f:
        id_map = json.load(f)
    return df, id_map


def _load_hadoop_timeseries() -> Tuple[pd.DataFrame, Dict[str, str]]:
    ts_path = os.path.join(DATA_PROCESSED, "hadoop_timeseries.csv")
    if not os.path.exists(ts_path):
        raise FileNotFoundError(ts_path)
    df = pd.read_csv(ts_path, index_col=0)
    id_map_path = os.path.join(DATA_PROCESSED, "hadoop_id_map.json")
    if not os.path.exists(id_map_path):
        raise FileNotFoundError(id_map_path)
    with open(id_map_path, "r", encoding="utf-8") as f:
        id_map = json.load(f)
    return df, id_map


def _dynotears_edges_for_original(
    df: pd.DataFrame,
    tpl_map: Dict[str, str],
    domain: str,
) -> List[Dict[str, object]]:
    if df.empty:
        return []
    X = df.values
    if X.size == 0:
        return []
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    if domain == "hdfs":
        lambda_w, lambda_a, thr = 0.025, 0.05, 0.30
    elif domain == "openstack":
        lambda_w, lambda_a, thr = 0.030, 0.06, 0.40
    else:  # hadoop
        lambda_w, lambda_a, thr = 0.025, 0.05, 0.30

    W, A = dynotears_engine(X, lambda_w, lambda_a)
    edges: List[Dict[str, object]] = []
    for mat, rel in ((A, "temporally_causes"), (W, "instantly_triggers")):
        mat = np.array(mat, copy=True)
        mat[np.abs(mat) < thr] = 0.0
        s_idx, t_idx = np.where(mat != 0)
        for s, t in zip(s_idx, t_idx):
            src_id = df.columns[s]
            tgt_id = df.columns[t]
            src_tpl = tpl_map.get(src_id, "Unknown")
            tgt_tpl = tpl_map.get(tgt_id, "Unknown")
            edges.append(
                {
                    "domain": domain,
                    "source_template": src_tpl,
                    "relation": rel,
                    "target_template": tgt_tpl,
                    "weight": float(round(float(mat[s, t]), 4)),
                }
            )
    return edges


HADOOP_TAIL_ROWS = 1000  # cap Hadoop timeseries for fast build (~5–10 min vs 39 min)


def main() -> None:
    print("[*] Loading timeseries for original DYNOTEARS graphs (20260313)...")
    hdfs_df, hdfs_map = _load_hdfs_timeseries()
    os_df, os_map = _load_openstack_timeseries()
    hadoop_df, hadoop_map = _load_hadoop_timeseries()

    # Truncate Hadoop to avoid long run (e.g. 45+ min)
    if len(hadoop_df) > HADOOP_TAIL_ROWS:
        hadoop_df = hadoop_df.tail(HADOOP_TAIL_ROWS)
        print(f"[*] Hadoop timeseries truncated to last {HADOOP_TAIL_ROWS} rows.")

    dyno_edges: List[Dict[str, object]] = []

    domains = [("hdfs", hdfs_df, hdfs_map), ("openstack", os_df, os_map), ("hadoop", hadoop_df, hadoop_map)]

    for dom, df, m in tqdm(domains, desc="Original DYNOTEARS by domain", unit="domain"):
        print(f"[*] Building original DYNOTEARS edges for {dom} ...")
        dyno_edges.extend(_dynotears_edges_for_original(df, m, dom))

    os.makedirs(DATA_PROCESSED, exist_ok=True)
    with open(OUT_DYNO_ORIG, "w", encoding="utf-8") as f:
        json.dump(dyno_edges, f, indent=2, ensure_ascii=False)

    from collections import Counter

    cnt = Counter(e.get("domain", "") for e in dyno_edges)
    print(f"[SUCCESS] Original DYNOTEARS edges written to {OUT_DYNO_ORIG}")
    for dom, n in cnt.items():
        print(f"  - {dom}: {n} edges")


if __name__ == "__main__":
    main()

