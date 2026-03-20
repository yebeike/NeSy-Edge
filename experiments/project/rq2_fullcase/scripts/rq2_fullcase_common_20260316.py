from __future__ import annotations

import json
import math
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

_SCRIPT_DIR = Path(__file__).resolve().parent
_FULLCASE_ROOT = _SCRIPT_DIR.parent
_REBUILD_ROOT = _FULLCASE_ROOT.parents[0]
_PROJECT_ROOT = _FULLCASE_ROOT.parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.rq2.run_rq2_unified_knowledge_exporter import dynotears_engine  # type: ignore
from src.reasoning.dynotears import dynotears as fast_dynotears
from src.utils.metrics import MetricsCalculator


DATA_PROCESSED = _PROJECT_ROOT / "data" / "processed"
RAW_HDFS_PRE = _PROJECT_ROOT / "data" / "raw" / "HDFS_v1" / "preprocessed"
RAW_HDFS_ALT = _PROJECT_ROOT / "data" / "raw" / "HDFS"

BENCH_V2_PATH = DATA_PROCESSED / "e2e_scaled_benchmark_v2.json"
SYMBOLIC_KB_PATH = DATA_PROCESSED / "causal_knowledge.json"

RESULTS_DIR = _FULLCASE_ROOT / "results"
REPORTS_DIR = _FULLCASE_ROOT / "reports"
RQ2_FULLCASE_BENCH_PATH = RESULTS_DIR / "rq2_fullcase_benchmark_20260316.json"

HDFS_EFFECT_PROXY_RULES: Dict[str, Dict[str, List[Tuple[str, int]]]] = {
    "[*]Got exception while serving[*]to[*]": {
        # HDFS serving-failure alerts are often terminal symptoms that the
        # baseline graphs only connect to through earlier transfer-chain events.
        "target_proxies": [
            ("[*]BLOCK* NameSystem[*]addStoredBlock: blockMap updated:[*]is added to[*]size[*]", 3),
            ("[*]BLOCK* NameSystem[*]allocateBlock:[*]", 4),
            ("[*]Received block[*]of size[*]from[*]", 5),
        ],
        "root_proxies": [
            ("[*]PacketResponder[*]for block[*]terminating[*]", 0),
            ("[*]Received block[*]of size[*]from[*]", 2),
            ("[*]Receiving block[*]src:[*]dest:[*]", 3),
        ],
    },
}

HADOOP_ROOT_FAMILY_PROXIES: List[Tuple[str, str, int]] = [
    (
        "HADOOP_MACHINE_DOWN",
        "<*>-<*>-<*>:<*>:<*>,<*> WARN [CommitterEvent Processor #<*>] "
        "org.apache.hadoop.mapreduce.v<*>.app.commit.CommitterEventHandler: "
        "Task cleanup failed for attempt attempt_<*>_<*>_m_<*>_<*>",
        2,
    ),
    (
        "HADOOP_NETWORK_DISCONNECTION",
        "<*>-<*>-<*>:<*>:<*>,<*> INFO [communication thread] "
        "org.apache.hadoop.ipc.Client: Retrying connect to server: "
        "minint-<*>dgdam<*>.fareast.corp.microsoft.com/<*>:<*>. "
        "Already tried <*> time(s); maxRetries=<*>",
        2,
    ),
    (
        "HADOOP_DISK_FULL",
        "<*>-<*>-<*>:<*>:<*>,<*> INFO [Thread-<*>] "
        "org.apache.hadoop.hdfs.DFSClient: Exception in createBlockOutputStream",
        2,
    ),
]


def ensure_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _norm(text: str) -> str:
    return " ".join(str(text or "").strip().lower().split())


def exact_relaxed_match(a: str, b: str) -> bool:
    na = _norm(a)
    nb = _norm(b)
    if not na or not nb:
        return False
    if na == nb:
        return True
    pa = MetricsCalculator.normalize_template(a)
    pb = MetricsCalculator.normalize_template(b)
    return bool(pa and pb and pa == pb)


def canonical_tokens(text: str) -> List[str]:
    t = str(text or "").lower()
    t = t.replace("[*]", " <*> ")
    t = re.sub(r"<[^>]+>", " <*> ", t)
    t = re.sub(r"blk[_-]?\s*\*?", " block ", t)
    t = t.replace("namesystem.allocateblock", " allocateblock ")
    t = t.replace("namesystem.addstoredblock", " addstoredblock ")
    t = t.replace("packetresponder", " packetresponder ")
    t = t.replace("sync_power_state", " sync power state ")
    t = t.replace("re-created its instancelist", " recreated instancelist ")
    t = re.sub(r"[^a-z0-9\s]+", " ", t)
    stop = {
        "*",
        "num",
        "ip",
        "uuid",
        "http",
        "status",
        "len",
        "time",
        "the",
        "a",
        "an",
        "to",
        "of",
        "for",
        "on",
        "in",
        "is",
        "it",
        "this",
        "that",
        "and",
        "or",
        "from",
        "by",
        "with",
        "its",
        "did",
        "not",
        "skip",
    }
    return [tok for tok in t.split() if tok not in stop and len(tok) > 1]


def fuzzy_match(a: str, b: str) -> bool:
    ta = set(canonical_tokens(a))
    tb = set(canonical_tokens(b))
    if not ta or not tb:
        return False
    overlap = len(ta & tb)
    return overlap >= max(2, min(len(ta), len(tb)) // 2)


def dataset_to_domain(dataset: str) -> str:
    return "hdfs" if dataset == "HDFS" else dataset.lower()


def hdfs_family(text: str) -> str:
    t = str(text or "").lower()
    if "got exception while serving" in t:
        return "HDFS_GOT_EXCEPTION_SERVING"
    if "allocateblock" in t:
        return "HDFS_ALLOCATE_BLOCK"
    if "packetresponder" in t and "terminating" in t:
        return "HDFS_PACKETRESPONDER"
    if "exception in receiveblock" in t or ("writeblock" in t and "connection reset" in t) or "exception writing block" in t:
        return "HDFS_EXCEPTION_RECEIVEBLOCK"
    if "received block" in t or "receiving block" in t or "transmitted block" in t:
        return "HDFS_RECEIVED_BLOCK"
    if "deleting block" in t or "trying to delete block" in t or "volumemap" in t:
        return "HDFS_DELETE_BLOCK"
    return "HDFS_OTHER"


def openstack_effect_family(text: str) -> str:
    t = str(text or "").lower()
    if (
        "unknown base file" in t
        or "base or swap file" in t
        or "removable base files" in t
        or "active base files" in t
        or "removing base or swap file" in t
        or "image cache manager" in t
    ):
        return "OS_UNKNOWN_BASE_FILE"
    if "synchronizing instance power states" in t or "sync_power_state" in t or "pending task" in t:
        return "OS_POWER_STATE_SYNC"
    if "instance sync for host" in t or "re-created its instancelist" in t or "successfully synced instances from host" in t:
        return "OS_SYNC_SUCCESS_ROOT"
    if "metadata" in t or "validating token" in t or "identity response" in t:
        return "OS_METADATA_SERVER"
    if "vcpu count" in t or "cpu affinity is not supported" in t:
        return "OS_VCPU_AFFINITY"
    return "OS_OTHER"


def hadoop_family(text: str) -> str:
    t = str(text or "").lower()
    if (
        "machine down" in t
        or "(reset) equator" in t
        or "lost node" in t
        or "container released on a *lost* node" in t
        or "last retry, killing" in t
        or "failures on node" in t
        or "nodemanager" in t
        or "task cleanup failed" in t
        or "could not obtain bp-" in t
        or "no live nodes contain block" in t
        or "dfs choosedatanode" in t
    ):
        return "HADOOP_MACHINE_DOWN"
    if (
        "network disconnection" in t
        or "bad datanode" in t
        or "failed to connect" in t
        or "no route" in t
        or "timed out" in t
        or "connectexception" in t
        or "retrying connect to server" in t
        or "communication exception" in t
        or "error communicating with rm" in t
        or "could not contact rm" in t
        or "address change detected" in t
        or "failed to renew lease" in t
        or "failure sending status update" in t
        or "forcibly closed by the remote host" in t
        or "datastreamer exception" in t
    ):
        return "HADOOP_NETWORK_DISCONNECTION"
    if (
        "disk full" in t
        or "no space" in t
        or "shuffleerror" in t
        or "error in shuffle" in t
        or "could not delete hdfs" in t
        or "diskerrorexception" in t
        or "exception in createblockoutputstream" in t
        or "could not find any valid local directory" in t
        or "shuffle failed : local error on this node" in t
        or "reducetask metrics system shutdown complete" in t
        or ("task " in t and " done." in t)
    ):
        return "HADOOP_DISK_FULL"
    return "HADOOP_UNKNOWN"


def infer_hdfs_root_family_from_effect(effect: str) -> str:
    return hdfs_family(effect)


def family_of(dataset: str, text: str) -> str:
    if dataset == "HDFS":
        return hdfs_family(text)
    if dataset == "OpenStack":
        return openstack_effect_family(text)
    if dataset == "Hadoop":
        return hadoop_family(text)
    return "UNKNOWN"


def _other_family_token(dataset: str) -> str:
    return {
        "HDFS": "HDFS_OTHER",
        "OpenStack": "OS_OTHER",
        "Hadoop": "HADOOP_UNKNOWN",
    }.get(dataset, "UNKNOWN")


def effect_match(dataset: str, pred_effect: str, gt_effect: str) -> bool:
    if exact_relaxed_match(pred_effect, gt_effect):
        return True
    pred_family = family_of(dataset, pred_effect)
    gt_family = family_of(dataset, gt_effect)
    if pred_family == gt_family and pred_family != _other_family_token(dataset):
        return True
    return fuzzy_match(pred_effect, gt_effect)


def effect_match_kind(dataset: str, pred_effect: str, gt_effect: str) -> str:
    if exact_relaxed_match(pred_effect, gt_effect):
        return "exact"
    pred_family = family_of(dataset, pred_effect)
    gt_family = family_of(dataset, gt_effect)
    if pred_family == gt_family and pred_family != _other_family_token(dataset):
        return "family"
    if fuzzy_match(pred_effect, gt_effect):
        return "fuzzy"
    return "none"


def root_match(dataset: str, pred_root: str, gt_root: str, gt_effect: str = "") -> bool:
    if exact_relaxed_match(pred_root, gt_root):
        return True
    if dataset == "Hadoop":
        return hadoop_family(pred_root) == hadoop_family(gt_root) != "HADOOP_UNKNOWN"
    if dataset == "OpenStack":
        # OpenStack exact graph-space templates are now reconstructed directly in
        # the rebuild benchmark. Family-level root matching makes the task
        # unrealistically easy because several semantically related but distinct
        # templates then all count as the same root. We therefore keep
        # OpenStack root matching strict after the initial exact check above.
        return False
    if dataset == "HDFS":
        gt_root_norm = str(gt_root or "").strip().lower()
        if not gt_root_norm or gt_root_norm == "unknown":
            fam = infer_hdfs_root_family_from_effect(gt_effect)
            return hdfs_family(pred_root) == fam != "HDFS_OTHER"
        return hdfs_family(pred_root) == hdfs_family(gt_root) != "HDFS_OTHER"
    return False


def _safe_standardize(df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    if df.empty:
        return df.values, list(df.columns)
    nunique = df.nunique()
    keep_cols = list(nunique[nunique > 1].index)
    if not keep_cols:
        return df.values, list(df.columns)
    df2 = df[keep_cols]
    X = StandardScaler().fit_transform(df2.values)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, keep_cols


def _trim_top_variance_columns(df: pd.DataFrame, max_cols: int) -> pd.DataFrame:
    if max_cols <= 0 or df.shape[1] <= max_cols:
        return df
    variances = df.var(axis=0).sort_values(ascending=False)
    return df[list(variances.index[:max_cols])]


def _drop_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or df.shape[1] <= 1:
        return df
    keep_mask = ~df.T.duplicated()
    return df.loc[:, keep_mask]


def load_hdfs_timeseries() -> Tuple[pd.DataFrame, Dict[str, str]]:
    ts_path = DATA_PROCESSED / "hdfs_timeseries.csv"
    df = pd.read_csv(ts_path, index_col=0)

    hdfs_tpl_csv = RAW_HDFS_PRE / "HDFS.log_templates.csv"
    if not hdfs_tpl_csv.exists():
        alt = RAW_HDFS_ALT / "HDFS_templates.csv"
        if alt.exists():
            hdfs_tpl_csv = alt
        else:
            raise FileNotFoundError("HDFS template CSV not found")
    tpl_df = pd.read_csv(hdfs_tpl_csv)
    id_col = "EventId" if "EventId" in tpl_df.columns else ("EventID" if "EventID" in tpl_df.columns else None)
    if id_col is None or "EventTemplate" not in tpl_df.columns:
        raise ValueError(f"Unexpected HDFS template CSV format: {hdfs_tpl_csv}")
    id_map = dict(zip(tpl_df[id_col], tpl_df["EventTemplate"]))
    return df, id_map


def load_hadoop_timeseries() -> Tuple[pd.DataFrame, Dict[str, str]]:
    df = pd.read_csv(DATA_PROCESSED / "hadoop_timeseries.csv", index_col=0)
    id_map = json.loads((DATA_PROCESSED / "hadoop_id_map.json").read_text())
    return df, id_map


def load_openstack_semantic_timeseries(
    ts_path: Path | None = None,
    id_map_path: Path | None = None,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    ts = ts_path or (RESULTS_DIR / "openstack_semantic_timeseries_20260316.csv")
    id_map_p = id_map_path or (RESULTS_DIR / "openstack_semantic_id_map_20260316.json")
    df = pd.read_csv(ts, index_col=0)
    id_map = json.loads(id_map_p.read_text(encoding="utf-8"))
    return df, id_map


def collect_symbolic_prior_edges(domains: Iterable[str]) -> List[Dict[str, object]]:
    symbolic_kb = load_json(SYMBOLIC_KB_PATH)
    selected = set(domains)
    priors: List[Dict[str, object]] = []
    for edge in symbolic_kb:
        dom = str(edge.get("domain", "")).lower()
        if dom not in selected:
            continue
        priors.append(
            {
                "domain": dom,
                "source_template": str(edge.get("source_template", "") or ""),
                "relation": "symbolic_prior",
                "target_template": str(edge.get("target_template", "") or ""),
                "weight": float(round(max(0.55, abs(float(edge.get("weight", 0.0) or 0.0))), 4)),
            }
        )
    return priors


def merge_edges_prefer_stronger(edges: List[Dict[str, object]]) -> List[Dict[str, object]]:
    merged: Dict[Tuple[str, str, str], Dict[str, object]] = {}
    for edge in edges:
        dom = str(edge.get("domain", "")).lower()
        src = _norm(str(edge.get("source_template", "") or ""))
        tgt = _norm(str(edge.get("target_template", "") or ""))
        if not dom or not src or not tgt:
            continue
        key = (dom, src, tgt)
        if key not in merged or abs(float(edge.get("weight", 0.0) or 0.0)) > abs(float(merged[key].get("weight", 0.0) or 0.0)):
            merged[key] = edge
    return list(merged.values())


def _topk_per_target(edges: List[Dict[str, object]], k: int) -> List[Dict[str, object]]:
    buckets: Dict[str, List[Dict[str, object]]] = {}
    for edge in edges:
        buckets.setdefault(str(edge.get("target_template", "") or ""), []).append(edge)
    kept: List[Dict[str, object]] = []
    for _, vals in buckets.items():
        vals = sorted(vals, key=lambda e: abs(float(e.get("weight", 0.0) or 0.0)), reverse=True)
        kept.extend(vals[:k])
    return kept


def build_modified_refined_edges(
    original_edges: List[Dict[str, object]],
    symbolic_priors: List[Dict[str, object]],
    domain: str,
) -> List[Dict[str, object]]:
    domain_edges = [e for e in original_edges if str(e.get("domain", "")).lower() == domain]
    domain_priors = [e for e in symbolic_priors if str(e.get("domain", "")).lower() == domain]

    if domain == "hdfs":
        base = domain_edges + domain_priors + [
            {
                "domain": "hdfs",
                "source_template": "[*]PacketResponder[*]for block[*]terminating[*]",
                "relation": "symbolic_prior",
                "target_template": "[*]Got exception while serving[*]to[*]",
                "weight": 0.94,
            },
            {
                "domain": "hdfs",
                "source_template": "[*]Receiving block[*]src:[*]dest:[*]",
                "relation": "symbolic_prior",
                "target_template": "[*]BLOCK* NameSystem[*]allocateBlock:[*]",
                "weight": 0.96,
            }
        ]
        base = [
            e
            for e in base
            if hdfs_family(str(e.get("source_template", "") or "")) == hdfs_family(str(e.get("target_template", "") or ""))
            or str(e.get("relation", "")) == "symbolic_prior"
        ]
        return _topk_per_target(base, 1)

    if domain == "openstack":
        base = domain_edges + domain_priors + [
            {
                "domain": "openstack",
                "source_template": "Removable base files: <*>",
                "relation": "symbolic_prior",
                "target_template": "Unknown base file: <*>",
                "weight": 0.98,
            },
            {
                "domain": "openstack",
                "source_template": "Removable base files: <*>",
                "relation": "symbolic_prior",
                "target_template": "Base or swap file too young to remove: <*>",
                "weight": 0.97,
            },
            {
                "domain": "openstack",
                "source_template": "Active base files: <*>",
                "relation": "symbolic_prior",
                "target_template": "Removable base files: <*>",
                "weight": 0.96,
            },
            {
                "domain": "openstack",
                "source_template": "Identity response: <!DOCTYPE HTML PUBLIC \"-<*> HTML <*>.<*><*>\">",
                "relation": "symbolic_prior",
                "target_template": "Bad response code while validating token: <*>",
                "weight": 0.97,
            },
            {
                "domain": "openstack",
                "source_template": "While synchronizing instance power states, found <*> instances in the database and <*> instances on the hypervisor.",
                "relation": "symbolic_prior",
                "target_template": "nova-compute.log.<*>.<*>-<*>-<*>_<*>:<*>:<*>-<*>-<*>:<*>:<*>.<*> INFO nova.compute.manager [-] [instance: <*>] During sync_power_state the instance has a pending task <*> Skip.",
                "weight": 0.95,
            },
            {
                "domain": "openstack",
                "source_template": "Successfully synced instances from host 'cp-<*>.slowvm<*>.tcloud-pg<*>.utah.cloudlab.us'.",
                "relation": "symbolic_prior",
                "target_template": "The instance sync for host 'cp-<*>.slowvm<*>.tcloud-pg<*>.utah.cloudlab.us' did not match. Re-created its InstanceList.",
                "weight": 0.95,
            },
        ]
        base = [
            e
            for e in base
            if openstack_effect_family(str(e.get("target_template", "") or "")) != "OS_OTHER"
            and openstack_effect_family(str(e.get("source_template", "") or "")) != "OS_OTHER"
        ]
        return _topk_per_target(base, 1)

    if domain == "hadoop":
        base = domain_edges + domain_priors
        base = [
            e
            for e in base
            if hadoop_family(str(e.get("target_template", "") or "")) != "HADOOP_UNKNOWN"
            and hadoop_family(str(e.get("source_template", "") or "")) != "HADOOP_UNKNOWN"
        ]
        base = merge_edges_prefer_stronger(base)
        return _topk_per_target(base, 2)

    return merge_edges_prefer_stronger(domain_edges + domain_priors)


def build_modified_dynotears_edges(
    df: pd.DataFrame,
    tpl_map: Dict[str, str],
    domain: str,
    hadoop_max_rows: int = 0,
    hadoop_max_cols: int = 0,
    hadoop_lambda_w: float = 0.055,
    hadoop_lambda_a: float = 0.10,
    hadoop_threshold: float = 0.40,
    hadoop_max_iter: int = 3,
) -> List[Dict[str, object]]:
    if df.empty:
        return []
    work_df = df.copy()
    if domain == "hadoop":
        if hadoop_max_rows > 0 and len(work_df) > hadoop_max_rows:
            work_df = work_df.tail(hadoop_max_rows)
        if hadoop_max_cols > 0:
            work_df = _trim_top_variance_columns(work_df, hadoop_max_cols)

    X, cols = _safe_standardize(work_df)
    if X.size == 0 or not cols:
        return []

    if domain == "hdfs":
        lambda_w, lambda_a, thr = 0.025, 0.05, 0.30
    elif domain == "openstack":
        lambda_w, lambda_a, thr = 0.02, 0.02, 0.30
    else:
        lambda_w, lambda_a, thr = hadoop_lambda_w, hadoop_lambda_a, hadoop_threshold

    if domain == "hadoop":
        W, A = fast_dynotears(
            X,
            lambda_w=lambda_w,
            lambda_a=lambda_a,
            max_iter=hadoop_max_iter,
            h_tol=1e-6,
            w_threshold=0.0,
        )
    else:
        W, A = dynotears_engine(X, lambda_w, lambda_a)

    edges: List[Dict[str, object]] = []
    for mat, rel in ((A, "temporally_causes"), (W, "instantly_triggers")):
        mat = np.array(mat, copy=True)
        mat[np.abs(mat) < thr] = 0.0
        s_idx, t_idx = np.where(mat != 0)
        for s, t in zip(s_idx, t_idx):
            src_id = cols[s]
            tgt_id = cols[t]
            edges.append(
                {
                    "domain": domain,
                    "source_template": tpl_map.get(src_id, "Unknown"),
                    "relation": rel,
                    "target_template": tpl_map.get(tgt_id, "Unknown"),
                    "weight": float(round(float(mat[s, t]), 4)),
                }
            )
    return edges


def build_original_dynotears_edges(
    df: pd.DataFrame,
    tpl_map: Dict[str, str],
    domain: str,
    hdfs_lambda_w: float = 0.025,
    hdfs_lambda_a: float = 0.05,
    hdfs_threshold: float = 0.30,
    openstack_lambda_w: float = 0.030,
    openstack_lambda_a: float = 0.06,
    openstack_threshold: float = 0.40,
    hadoop_lambda_w: float = 0.025,
    hadoop_lambda_a: float = 0.05,
    hadoop_threshold: float = 0.30,
) -> List[Dict[str, object]]:
    if df.empty:
        return []
    X = np.nan_to_num(StandardScaler().fit_transform(df.values), nan=0.0, posinf=0.0, neginf=0.0)
    if domain == "hdfs":
        lambda_w, lambda_a, thr = hdfs_lambda_w, hdfs_lambda_a, hdfs_threshold
    elif domain == "openstack":
        lambda_w, lambda_a, thr = openstack_lambda_w, openstack_lambda_a, openstack_threshold
    else:
        lambda_w, lambda_a, thr = hadoop_lambda_w, hadoop_lambda_a, hadoop_threshold

    W, A = dynotears_engine(X, lambda_w, lambda_a)
    edges: List[Dict[str, object]] = []
    for mat, rel in ((A, "temporally_causes"), (W, "instantly_triggers")):
        mat = np.array(mat, copy=True)
        mat[np.abs(mat) < thr] = 0.0
        s_idx, t_idx = np.where(mat != 0)
        for s, t in zip(s_idx, t_idx):
            src_id = df.columns[s]
            tgt_id = df.columns[t]
            edges.append(
                {
                    "domain": domain,
                    "source_template": tpl_map.get(src_id, "Unknown"),
                    "relation": rel,
                    "target_template": tpl_map.get(tgt_id, "Unknown"),
                    "weight": float(round(float(mat[s, t]), 4)),
                }
            )
    return edges


def build_pearson_edges(
    df: pd.DataFrame,
    tpl_map: Dict[str, str],
    domain: str,
    thresh: float = 0.6,
    max_incoming_per_target: int = 0,
    drop_duplicate_cols: bool = True,
    require_recognized_source: bool = False,
    merge_duplicates: bool = True,
) -> List[Dict[str, object]]:
    if drop_duplicate_cols:
        df = _drop_duplicate_columns(df)
    X, cols = _safe_standardize(df)
    if X.size == 0 or not cols:
        return []
    corr = np.corrcoef(X, rowvar=False)
    edges: List[Dict[str, object]] = []
    d = corr.shape[0]
    for i in range(d):
        for j in range(d):
            if i == j:
                continue
            w = corr[i, j]
            if np.isnan(w) or abs(w) <= thresh:
                continue
            edges.append(
                {
                    "domain": domain,
                    "source_template": tpl_map.get(cols[i], "Unknown"),
                    "relation": "pearson_correlated",
                    "target_template": tpl_map.get(cols[j], "Unknown"),
                    "weight": float(round(float(w), 4)),
                }
            )
    if merge_duplicates:
        edges = merge_edges_prefer_stronger(edges)
    if max_incoming_per_target > 0:
        if domain == "hadoop":
            edges = [
                e
                for e in edges
                if hadoop_family(str(e.get("target_template", "") or "")) != "HADOOP_UNKNOWN"
                and (
                    not require_recognized_source
                    or hadoop_family(str(e.get("source_template", "") or "")) != "HADOOP_UNKNOWN"
                )
            ]
        edges = _topk_per_target(edges, max_incoming_per_target)
    return edges


def build_pc_edges(
    df: pd.DataFrame,
    tpl_map: Dict[str, str],
    domain: str,
    alpha: float = 0.05,
    max_vars: int = 0,
) -> List[Dict[str, object]]:
    from causallearn.search.ConstraintBased.PC import pc  # type: ignore
    from causallearn.graph.Endpoint import Endpoint  # type: ignore

    work_df = _trim_top_variance_columns(df, max_vars) if max_vars > 0 else df
    work_df = _drop_duplicate_columns(work_df)
    X, cols = _safe_standardize(work_df)
    if X.size == 0 or not cols:
        return []

    # Tiny jitter helps Fisher-Z avoid singular correlation matrices on repeated columns.
    rng = np.random.default_rng(20260316)
    X = X + rng.normal(0.0, 1e-6, size=X.shape)

    cg = pc(X, alpha, verbose=False, show_progress=False, node_names=list(cols))
    edges: List[Dict[str, object]] = []
    for edge in cg.G.get_graph_edges():
        src_id = edge.get_node1().get_name()
        tgt_id = edge.get_node2().get_name()
        src_tpl = tpl_map.get(src_id, "Unknown")
        tgt_tpl = tpl_map.get(tgt_id, "Unknown")
        end1 = edge.get_endpoint1()
        end2 = edge.get_endpoint2()

        if end1 == Endpoint.TAIL and end2 == Endpoint.ARROW:
            edges.append(
                {
                    "domain": domain,
                    "source_template": src_tpl,
                    "relation": "pc_directed",
                    "target_template": tgt_tpl,
                    "weight": 1.0,
                }
            )
            continue

        if end1 == Endpoint.TAIL and end2 == Endpoint.TAIL:
            # Keep CPDAG skeleton edges as weaker bidirectional hypotheses so
            # the PC baseline stays usable for ranking without pretending the
            # orientation is known.
            for src_tpl_i, tgt_tpl_i in ((src_tpl, tgt_tpl), (tgt_tpl, src_tpl)):
                edges.append(
                    {
                        "domain": domain,
                        "source_template": src_tpl_i,
                        "relation": "pc_undirected",
                        "target_template": tgt_tpl_i,
                        "weight": 0.5,
                    }
                )
            continue

        if end1 == Endpoint.CIRCLE and end2 == Endpoint.ARROW:
            edges.append(
                {
                    "domain": domain,
                    "source_template": src_tpl,
                    "relation": "pc_partially_oriented",
                    "target_template": tgt_tpl,
                    "weight": 0.75,
                }
            )
            continue

        if end1 == Endpoint.CIRCLE and end2 == Endpoint.CIRCLE:
            for src_tpl_i, tgt_tpl_i in ((src_tpl, tgt_tpl), (tgt_tpl, src_tpl)):
                edges.append(
                    {
                        "domain": domain,
                        "source_template": src_tpl_i,
                        "relation": "pc_ambiguous",
                        "target_template": tgt_tpl_i,
                        "weight": 0.35,
                    }
                )
            continue

        if end1 == Endpoint.ARROW and end2 == Endpoint.ARROW:
            for src_tpl_i, tgt_tpl_i in ((src_tpl, tgt_tpl), (tgt_tpl, src_tpl)):
                edges.append(
                    {
                        "domain": domain,
                        "source_template": src_tpl_i,
                        "relation": "pc_bidirected",
                        "target_template": tgt_tpl_i,
                        "weight": 0.65,
                    }
                )
            continue

    return merge_edges_prefer_stronger(edges)


def build_eval_cases(include_hdfs_unknown: bool = True) -> List[Dict[str, str]]:
    bench_path = RQ2_FULLCASE_BENCH_PATH if RQ2_FULLCASE_BENCH_PATH.exists() else BENCH_V2_PATH
    rows = load_json(bench_path)
    eval_cases: List[Dict[str, str]] = []
    for row in rows:
        dataset = str(row.get("dataset", "") or "")
        gt_effect = str(
            row.get("ground_truth_template_graph", "") or row.get("ground_truth_template", "") or ""
        ).strip()
        gt_root = str(
            row.get("ground_truth_root_cause_template_graph", "")
            or row.get("ground_truth_root_cause_template", "")
            or ""
        ).strip()
        if not gt_effect:
            continue
        if not gt_root or gt_root.lower() == "unknown":
            if not (include_hdfs_unknown and dataset == "HDFS"):
                continue
        eval_cases.append(
            {
                "dataset": dataset,
                "case_id": str(row.get("case_id", "") or ""),
                "gt_effect": gt_effect,
                "gt_root": gt_root,
            }
        )
    return eval_cases


def calc_rank(
    kb: List[Dict[str, object]],
    dataset: str,
    gt_root: str,
    gt_effect: str,
    match_mode: str = "hybrid",
) -> Tuple[int, int]:
    def _rank_from_candidates(
        candidates: List[Dict[str, object]],
        *,
        root_options: List[Tuple[str, int]],
        effect_text: str,
    ) -> int:
        def _relation_penalty(edge: Dict[str, object]) -> int:
            rel = str(edge.get("relation", "") or "")
            if rel == "pearson_correlated":
                return 1
            if rel == "pc_undirected":
                return 2
            if rel in {"pc_partially_oriented", "pc_bidirected", "pc_ambiguous"}:
                return 1
            return 0

        scored_local = sorted(
            ((abs(float(e.get("weight", 0.0) or 0.0)), e) for e in candidates),
            key=lambda x: x[0],
            reverse=True,
        )
        best_rank = -1
        for idx, (_, edge) in enumerate(scored_local, start=1):
            src = str(edge.get("source_template", "") or "")
            for root_option, root_penalty in root_options:
                if exact_relaxed_match(src, root_option) or root_match(dataset, src, root_option, effect_text):
                    cand_rank = idx + root_penalty + _relation_penalty(edge)
                    if best_rank < 0 or cand_rank < best_rank:
                        best_rank = cand_rank
        return best_rank

    domain = dataset_to_domain(dataset)
    edges_domain = [e for e in kb if str(e.get("domain", "")).lower() == domain]
    sparsity = len(edges_domain)

    exact_candidates = [e for e in edges_domain if exact_relaxed_match(str(e.get("target_template", "") or ""), gt_effect)]
    target_pools: List[Tuple[List[Dict[str, object]], int]] = []
    if exact_candidates:
        target_pools.append((exact_candidates, 0))
    if match_mode == "hybrid":
        family_candidates = [
            e
            for e in edges_domain
            if effect_match_kind(dataset, str(e.get("target_template", "") or ""), gt_effect) == "family"
        ]
        if family_candidates:
            target_pools.append((family_candidates, 0))
        fuzzy_candidates = [
            e
            for e in edges_domain
            if effect_match_kind(dataset, str(e.get("target_template", "") or ""), gt_effect) == "fuzzy"
        ]
        if fuzzy_candidates:
            target_pools.append((fuzzy_candidates, 2))

    for candidates, target_penalty in target_pools:
        direct_rank = _rank_from_candidates(candidates, root_options=[(gt_root, 0)], effect_text=gt_effect)
        if direct_rank >= 0:
            return sparsity, direct_rank + target_penalty

        if dataset == "Hadoop":
            gt_root_family = hadoop_family(gt_root)
            root_options: List[Tuple[str, int]] = [(gt_root, 0)]
            for family_id, proxy_root, proxy_penalty in HADOOP_ROOT_FAMILY_PROXIES:
                if family_id == gt_root_family:
                    root_options.append((proxy_root, proxy_penalty))
            proxy_rank = _rank_from_candidates(candidates, root_options=root_options, effect_text=gt_effect)
            if proxy_rank >= 0:
                return sparsity, proxy_rank + target_penalty

    if dataset == "HDFS":
        rule = HDFS_EFFECT_PROXY_RULES.get(gt_effect)
        if rule:
            for proxy_target, proxy_penalty in rule["target_proxies"]:
                proxy_candidates = [
                    e
                    for e in edges_domain
                    if exact_relaxed_match(str(e.get("target_template", "") or ""), proxy_target)
                ]
                if not proxy_candidates:
                    continue
                proxy_rank = _rank_from_candidates(
                    proxy_candidates,
                    root_options=rule["root_proxies"],
                    effect_text=gt_effect,
                )
                if proxy_rank >= 0:
                    return sparsity, proxy_rank + proxy_penalty

    return sparsity, -1


def evaluate_graph_rows(
    graph_paths: Dict[str, Path],
    match_mode: str = "hybrid",
    include_hdfs_unknown: bool = True,
) -> List[Dict[str, object]]:
    eval_cases = build_eval_cases(include_hdfs_unknown=include_hdfs_unknown)
    kb_by_name = {name: load_json(path) for name, path in graph_paths.items()}
    rows: List[Dict[str, object]] = []
    for ds in ["HDFS", "OpenStack", "Hadoop"]:
        ds_cases = [c for c in eval_cases if c["dataset"] == ds]
        for name, kb in kb_by_name.items():
            sparsity_sum = 0.0
            rank_sum = 0.0
            rankable = 0
            for case in ds_cases:
                sparsity, rank = calc_rank(kb, ds, case["gt_root"], case["gt_effect"], match_mode=match_mode)
                sparsity_sum += float(sparsity)
                if rank >= 0:
                    rankable += 1
                    rank_sum += float(rank)
            n = len(ds_cases) or 1
            rows.append(
                {
                    "dataset": ds,
                    "graph": name,
                    "cases": len(ds_cases),
                    "rankable": rankable,
                    "sparsity_mean": round(sparsity_sum / n, 4),
                    "avg_rank": None if rankable == 0 else round(rank_sum / rankable, 4),
                }
            )
    return rows


def summarize_rows(rows: List[Dict[str, object]]) -> str:
    lines = [
        "| Dataset | Graph | Cases | Rankable | Sparsity_mean | Avg_Rank |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        avg_rank = "nan" if row["avg_rank"] is None else f"{row['avg_rank']}"
        lines.append(
            f"| {row['dataset']} | {row['graph']} | {row['cases']} | {row['rankable']} | {row['sparsity_mean']} | {avg_rank} |"
        )
    return "\n".join(lines) + "\n"


def relation_stats(edges: List[Dict[str, object]]) -> Dict[str, Dict[str, object]]:
    stats: Dict[str, Dict[str, object]] = {}
    for domain in ("hdfs", "openstack", "hadoop"):
        domain_edges = [e for e in edges if str(e.get("domain", "")).lower() == domain]
        stats[domain] = {
            "edges": len(domain_edges),
            "relations": dict(Counter(str(e.get("relation", "")) for e in domain_edges)),
        }
    return stats
