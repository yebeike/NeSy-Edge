from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from vendor_dynotears_causalnex_20260318 import dynotears_from_standardized_matrix

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

_SCRIPT_DIR = Path(__file__).resolve().parent
_AUDIT_ROOT = _SCRIPT_DIR.parent
_REBUILD_ROOT = _AUDIT_ROOT.parents[0]
_PROJECT_ROOT = _AUDIT_ROOT.parents[2]

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils.metrics import MetricsCalculator

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - progress is optional
    tqdm = None


DATA_PROCESSED = _PROJECT_ROOT / "data" / "processed"
RAW_HDFS_PRE = _PROJECT_ROOT / "data" / "raw" / "HDFS_v1" / "preprocessed"
RAW_HDFS_ALT = _PROJECT_ROOT / "data" / "raw" / "HDFS"
RAW_OPENSTACK_2 = _PROJECT_ROOT / "data" / "raw" / "OpenStack_2"

OLD_FULLCASE_ROOT = _REBUILD_ROOT / "rq2_fullcase"
OLD_RESULTS_DIR = OLD_FULLCASE_ROOT / "results"

RESULTS_DIR = _AUDIT_ROOT / "results"
REPORTS_DIR = _AUDIT_ROOT / "reports"

SOURCE_BENCH_PATH = DATA_PROCESSED / "e2e_scaled_benchmark_v2.json"
OLD_MODIFIED_GRAPH_PATH = OLD_RESULTS_DIR / "gt_causal_knowledge_nesydy_fullcase_20260316.json"
OLD_OPENSTACK_TS_PATH = OLD_RESULTS_DIR / "openstack_semantic_timeseries_20260316.csv"
OLD_OPENSTACK_ID_MAP_PATH = OLD_RESULTS_DIR / "openstack_semantic_id_map_20260316.json"

AUDIT_BENCHMARK_FULL_PATH = RESULTS_DIR / "rq2_fullcase_audit_benchmark_full_20260318.json"
AUDIT_BENCHMARK_EVAL_PATH = RESULTS_DIR / "rq2_fullcase_audit_benchmark_evaluable_20260318.json"

GRAPH_FILES = {
    "modified": RESULTS_DIR / "gt_causal_knowledge_modified_frozen_audit_20260318.json",
    "original_dynotears": RESULTS_DIR / "gt_causal_knowledge_original_dynotears_audit_20260318.json",
    "pearson_hypothesis": RESULTS_DIR / "gt_causal_knowledge_pearson_hypothesis_audit_20260318.json",
    "pc_cpdag_hypothesis": RESULTS_DIR / "gt_causal_knowledge_pc_cpdag_hypothesis_audit_20260318.json",
}

HDFS_EFFECT_PROXY_RULES: Dict[str, Dict[str, List[Tuple[str, int]]]] = {
    "[*]Got exception while serving[*]to[*]": {
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

_MANUAL_PRIOR_PAIRS = {
    ("hdfs", "[*]PacketResponder[*]for block[*]terminating[*]", "[*]Got exception while serving[*]to[*]"),
    ("hdfs", "[*]Receiving block[*]src:[*]dest:[*]", "[*]BLOCK* NameSystem[*]allocateBlock:[*]"),
    ("openstack", "Removable base files: <*>", "Unknown base file: <*>"),
    ("openstack", "Removable base files: <*>", "Base or swap file too young to remove: <*>"),
    ("openstack", "Active base files: <*>", "Removable base files: <*>"),
    ("openstack", "Identity response: <!DOCTYPE HTML PUBLIC \"-<*> HTML <*>.<*><*>\">", "Bad response code while validating token: <*>"),
    (
        "openstack",
        "While synchronizing instance power states, found <*> instances in the database and <*> instances on the hypervisor.",
        "nova-compute.log.<*>.<*>-<*>-<*>_<*>:<*>:<*>-<*>-<*>:<*>:<*>.<*> INFO nova.compute.manager [-] [instance: <*>] During sync_power_state the instance has a pending task <*> Skip.",
    ),
    (
        "openstack",
        "Successfully synced instances from host 'cp-<*>.slowvm<*>.tcloud-pg<*>.utah.cloudlab.us'.",
        "The instance sync for host 'cp-<*>.slowvm<*>.tcloud-pg<*>.utah.cloudlab.us' did not match. Re-created its InstanceList.",
    ),
}

UNDIRECTED_RELATIONS = {"pearson_undirected", "pc_undirected", "pc_ambiguous", "pc_bidirected"}
DIRECTED_RELATIONS = {
    "symbolic_prior",
    "temporally_causes",
    "instantly_triggers",
    "pc_directed",
    "pc_partially_oriented",
}


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


def family_of(dataset: str, text: str) -> str:
    if dataset == "HDFS":
        return hdfs_family(text)
    if dataset == "OpenStack":
        return openstack_effect_family(text)
    return "UNKNOWN"


def _other_family_token(dataset: str) -> str:
    return {
        "HDFS": "HDFS_OTHER",
        "OpenStack": "OS_OTHER",
    }.get(dataset, "UNKNOWN")


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
    if dataset == "OpenStack":
        return False
    if dataset == "HDFS":
        gt_root_norm = str(gt_root or "").strip().lower()
        if not gt_root_norm or gt_root_norm == "unknown":
            fam = hdfs_family(gt_effect)
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


def load_openstack_semantic_timeseries(
    ts_path: Path | None = None,
    id_map_path: Path | None = None,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    ts = ts_path or OLD_OPENSTACK_TS_PATH
    id_map_p = id_map_path or OLD_OPENSTACK_ID_MAP_PATH
    df = pd.read_csv(ts, index_col=0)
    id_map = json.loads(id_map_p.read_text(encoding="utf-8"))
    return df, id_map


def domain_template_pool() -> Dict[str, List[str]]:
    _, hdfs_map = load_hdfs_timeseries()
    _, openstack_map = load_openstack_semantic_timeseries()
    return {
        "HDFS": list(hdfs_map.values()),
        "OpenStack": list(openstack_map.values()),
    }


def best_graph_template(
    dataset: str,
    query_texts: List[str],
    pool: List[str],
    desired_family: str = "",
) -> str:
    queries = [str(x or "").strip() for x in query_texts if str(x or "").strip()]
    if not queries:
        return ""
    other_family = {"HDFS": "HDFS_OTHER", "OpenStack": "OS_OTHER"}[dataset]
    query_token_sets = [set(canonical_tokens(q)) for q in queries]

    best_tpl = ""
    best_score = -1.0
    for tpl in pool:
        fam = family_of(dataset, tpl)
        score = 0.0
        for q, toks in zip(queries, query_token_sets):
            if exact_relaxed_match(tpl, q):
                score = max(score, 1000.0)
            if desired_family and fam == desired_family and fam != other_family:
                score += 120.0
            qfam = family_of(dataset, q)
            if qfam == fam and fam != other_family:
                score += 80.0
            overlap = len(set(canonical_tokens(tpl)) & toks)
            score += overlap * 4.0
            if q.lower() in tpl.lower() or tpl.lower() in q.lower():
                score += 20.0
        if score > best_score:
            best_score = score
            best_tpl = tpl
    return best_tpl


def iter_with_progress(items: List[object], desc: str) -> Iterable[object]:
    if tqdm is None:
        return items
    return tqdm(items, desc=desc, unit="item")


def last_prior_by_family(dataset: str, prior_lines: List[str], desired_family: str) -> str:
    for line in reversed(prior_lines):
        if family_of(dataset, line) == desired_family:
            return line
    return ""


def manual_prior_pair_overlap(dataset: str, root_graph: str, effect_graph: str) -> bool:
    key = (dataset_to_domain(dataset), _norm(root_graph), _norm(effect_graph))
    normalized_pairs = {(dom, _norm(root), _norm(effect)) for dom, root, effect in _MANUAL_PRIOR_PAIRS}
    return key in normalized_pairs


def map_prior_templates_to_graph_candidates(
    dataset: str,
    prior_template_texts: List[str],
    pool: List[str],
) -> List[str]:
    mapped_candidates: List[str] = []
    for tpl in prior_template_texts:
        mapped = best_graph_template(dataset, [tpl], pool, desired_family=family_of(dataset, tpl))
        if mapped:
            mapped_candidates.append(mapped)
    return mapped_candidates


def pick_root_graph_from_prior_templates(
    dataset: str,
    prior_template_texts: List[str],
    pool: List[str],
    graph_effect: str,
    root_label_hint: str = "",
) -> str:
    other_family = _other_family_token(dataset)
    normalized_effect = _norm(graph_effect)
    label_hint = str(root_label_hint or "").strip()
    valid_label_hint = bool(label_hint) and label_hint.lower() != "unknown" and not exact_relaxed_match(label_hint, graph_effect)

    mapped_candidates = map_prior_templates_to_graph_candidates(dataset, prior_template_texts, pool)

    if dataset == "OpenStack":
        effect_family = family_of(dataset, graph_effect)
        family_candidates = [
            candidate
            for candidate in mapped_candidates
            if family_of(dataset, candidate) == effect_family and family_of(dataset, candidate) != other_family
        ]
        if family_candidates:
            last_effect_idx = -1
            for idx, candidate in enumerate(family_candidates):
                if exact_relaxed_match(candidate, graph_effect):
                    last_effect_idx = idx
            if last_effect_idx >= 0:
                for candidate in family_candidates[last_effect_idx + 1 :]:
                    if _norm(candidate) != normalized_effect:
                        return candidate
            for candidate in reversed(family_candidates):
                if _norm(candidate) != normalized_effect:
                    return candidate

    for candidate in reversed(mapped_candidates):
        if _norm(candidate) == normalized_effect:
            continue
        if family_of(dataset, candidate) == other_family:
            continue
        return candidate

    if valid_label_hint:
        hinted = best_graph_template(dataset, [label_hint], pool, desired_family=family_of(dataset, label_hint))
        if hinted and _norm(hinted) != normalized_effect:
            return hinted

    for candidate in reversed(mapped_candidates):
        if _norm(candidate) != normalized_effect:
            return candidate

    return ""


def merge_edges_prefer_stronger(edges: List[Dict[str, object]]) -> List[Dict[str, object]]:
    merged: Dict[Tuple[str, str, str, str], Dict[str, object]] = {}
    for edge in edges:
        dom = str(edge.get("domain", "")).lower()
        rel = str(edge.get("relation", ""))
        src = str(edge.get("source_template", "") or "")
        tgt = str(edge.get("target_template", "") or "")
        if rel in UNDIRECTED_RELATIONS:
            a, b = sorted([src, tgt])
            key = (dom, rel, a, b)
            edge = dict(edge)
            edge["source_template"] = a
            edge["target_template"] = b
        else:
            key = (dom, rel, src, tgt)
        if key not in merged or abs(float(edge.get("weight", 0.0) or 0.0)) > abs(float(merged[key].get("weight", 0.0) or 0.0)):
            merged[key] = edge
    return list(merged.values())


def build_original_dynotears_edges(
    df: pd.DataFrame,
    tpl_map: Dict[str, str],
    domain: str,
    lambda_w: float,
    lambda_a: float,
    threshold: float,
    p: int = 1,
    max_iter: int = 100,
    h_tol: float = 1e-8,
) -> List[Dict[str, object]]:
    if df.empty:
        return []
    X, cols = _safe_standardize(df)
    if X.size == 0 or not cols:
        return []
    W, A = dynotears_from_standardized_matrix(
        X,
        p=p,
        lambda_w=lambda_w,
        lambda_a=lambda_a,
        max_iter=max_iter,
        h_tol=h_tol,
        w_threshold=0.0,
    )
    edges: List[Dict[str, object]] = []
    for mat, rel in ((A, "temporally_causes"), (W, "instantly_triggers")):
        mat = np.array(mat, copy=True)
        mat[np.abs(mat) < threshold] = 0.0
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
    return merge_edges_prefer_stronger(edges)


def build_pearson_hypothesis_edges(
    df: pd.DataFrame,
    tpl_map: Dict[str, str],
    domain: str,
    threshold: float,
) -> List[Dict[str, object]]:
    X, cols = _safe_standardize(df)
    if X.size == 0 or not cols:
        return []
    corr = np.corrcoef(X, rowvar=False)
    edges: List[Dict[str, object]] = []
    d = corr.shape[0]
    for i in range(d):
        for j in range(i + 1, d):
            w = corr[i, j]
            if np.isnan(w) or abs(w) <= threshold:
                continue
            edges.append(
                {
                    "domain": domain,
                    "source_template": tpl_map.get(cols[i], "Unknown"),
                    "relation": "pearson_undirected",
                    "target_template": tpl_map.get(cols[j], "Unknown"),
                    "weight": float(round(float(w), 4)),
                }
            )
    return merge_edges_prefer_stronger(edges)


def build_pc_cpdag_hypothesis_edges(
    df: pd.DataFrame,
    tpl_map: Dict[str, str],
    domain: str,
    alpha: float,
) -> List[Dict[str, object]]:
    from causallearn.graph.Endpoint import Endpoint  # type: ignore
    from causallearn.search.ConstraintBased.PC import pc  # type: ignore

    work_df = _drop_duplicate_columns(df)
    X, cols = _safe_standardize(work_df)
    if X.size == 0 or not cols:
        return []

    rng = np.random.default_rng(20260318)
    X = X + rng.normal(0.0, 1e-6, size=X.shape)

    cg = pc(X, alpha=alpha, verbose=False, show_progress=False, node_names=list(cols))
    edges: List[Dict[str, object]] = []
    for edge in cg.G.get_graph_edges():
        src_id = edge.get_node1().get_name()
        tgt_id = edge.get_node2().get_name()
        src_tpl = tpl_map.get(src_id, "Unknown")
        tgt_tpl = tpl_map.get(tgt_id, "Unknown")
        end1 = edge.get_endpoint1()
        end2 = edge.get_endpoint2()

        if end1 == Endpoint.TAIL and end2 == Endpoint.ARROW:
            rel = "pc_directed"
            weight = 1.0
        elif end1 == Endpoint.TAIL and end2 == Endpoint.TAIL:
            rel = "pc_undirected"
            weight = 0.5
        elif end1 == Endpoint.CIRCLE and end2 == Endpoint.ARROW:
            rel = "pc_partially_oriented"
            weight = 0.75
        elif end1 == Endpoint.CIRCLE and end2 == Endpoint.CIRCLE:
            rel = "pc_ambiguous"
            weight = 0.35
        elif end1 == Endpoint.ARROW and end2 == Endpoint.ARROW:
            rel = "pc_bidirected"
            weight = 0.65
        else:
            continue

        edges.append(
            {
                "domain": domain,
                "source_template": src_tpl,
                "relation": rel,
                "target_template": tgt_tpl,
                "weight": weight,
            }
        )
    return merge_edges_prefer_stronger(edges)


def build_eval_cases(bench_path: Path) -> List[Dict[str, str]]:
    rows = load_json(bench_path)
    eval_cases: List[Dict[str, str]] = []
    for row in rows:
        dataset = str(row.get("dataset", "") or "")
        if dataset not in {"HDFS", "OpenStack"}:
            continue
        gt_effect = str(row.get("ground_truth_template_graph", "") or "").strip()
        gt_root = str(row.get("ground_truth_root_cause_template_graph", "") or "").strip()
        if not gt_effect or not gt_root:
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


def _relation_penalty(edge: Dict[str, object]) -> int:
    rel = str(edge.get("relation", "") or "")
    if rel == "pearson_undirected":
        return 1
    if rel == "pc_undirected":
        return 2
    if rel in {"pc_partially_oriented", "pc_bidirected", "pc_ambiguous"}:
        return 1
    return 0


def _candidate_buckets(
    edges_domain: List[Dict[str, object]],
    dataset: str,
    gt_effect: str,
    match_mode: str,
) -> Dict[str, List[Dict[str, object]]]:
    buckets: Dict[str, List[Dict[str, object]]] = {"exact": [], "family": [], "fuzzy": []}

    def _maybe_add(candidate_root: str, effect_side: str, edge: Dict[str, object]) -> None:
        if match_mode == "exact_only":
            if exact_relaxed_match(effect_side, gt_effect):
                buckets["exact"].append({"candidate_root": candidate_root, "edge": edge})
            return
        kind = effect_match_kind(dataset, effect_side, gt_effect)
        if kind in buckets:
            buckets[kind].append({"candidate_root": candidate_root, "edge": edge})

    for edge in edges_domain:
        src = str(edge.get("source_template", "") or "")
        tgt = str(edge.get("target_template", "") or "")
        rel = str(edge.get("relation", "") or "")
        if rel in UNDIRECTED_RELATIONS:
            _maybe_add(tgt, src, edge)
            _maybe_add(src, tgt, edge)
        else:
            _maybe_add(src, tgt, edge)

    return buckets


def _rank_from_candidates(
    candidates: List[Dict[str, object]],
    dataset: str,
    gt_effect: str,
    root_options: List[Tuple[str, int]],
    match_mode: str,
) -> int:
    scored = sorted(
        ((abs(float(c["edge"].get("weight", 0.0) or 0.0)), c) for c in candidates),
        key=lambda x: x[0],
        reverse=True,
    )
    best_rank = -1
    for idx, (_, candidate) in enumerate(scored, start=1):
        cand_root = str(candidate["candidate_root"] or "")
        edge = candidate["edge"]
        for root_option, root_penalty in root_options:
            if match_mode == "exact_only":
                matched = exact_relaxed_match(cand_root, root_option)
            else:
                matched = exact_relaxed_match(cand_root, root_option) or root_match(dataset, cand_root, root_option, gt_effect)
            if not matched:
                continue
            cand_rank = idx + root_penalty + _relation_penalty(edge)
            if best_rank < 0 or cand_rank < best_rank:
                best_rank = cand_rank
    return best_rank


def calc_rank(
    kb: List[Dict[str, object]],
    dataset: str,
    gt_root: str,
    gt_effect: str,
    match_mode: str = "task_aligned",
) -> Tuple[int, int]:
    domain = dataset_to_domain(dataset)
    edges_domain = [e for e in kb if str(e.get("domain", "")).lower() == domain]
    sparsity = len(edges_domain)
    buckets = _candidate_buckets(edges_domain, dataset, gt_effect, match_mode)

    pool_order: List[Tuple[str, int]] = [("exact", 0)]
    if match_mode == "task_aligned":
        pool_order.extend([("family", 0), ("fuzzy", 2)])

    for kind, target_penalty in pool_order:
        candidates = buckets[kind]
        if not candidates:
            continue
        direct_rank = _rank_from_candidates(
            candidates,
            dataset=dataset,
            gt_effect=gt_effect,
            root_options=[(gt_root, 0)],
            match_mode=match_mode,
        )
        if direct_rank >= 0:
            return sparsity, direct_rank + target_penalty

    if match_mode == "task_aligned" and dataset == "HDFS":
        rule = HDFS_EFFECT_PROXY_RULES.get(gt_effect)
        if rule:
            for proxy_target, proxy_penalty in rule["target_proxies"]:
                proxy_candidates = _candidate_buckets(edges_domain, dataset, proxy_target, "exact_only")["exact"]
                if not proxy_candidates:
                    continue
                proxy_rank = _rank_from_candidates(
                    proxy_candidates,
                    dataset=dataset,
                    gt_effect=gt_effect,
                    root_options=rule["root_proxies"],
                    match_mode="task_aligned",
                )
                if proxy_rank >= 0:
                    return sparsity, proxy_rank + proxy_penalty

    return sparsity, -1


def evaluate_graph_rows(
    graph_paths: Dict[str, Path],
    bench_path: Path,
    match_mode: str = "task_aligned",
) -> List[Dict[str, object]]:
    eval_cases = build_eval_cases(bench_path)
    kb_by_name = {name: load_json(path) for name, path in graph_paths.items()}
    rows: List[Dict[str, object]] = []
    jobs = [
        (ds, [c for c in eval_cases if c["dataset"] == ds], name, kb)
        for ds in ["HDFS", "OpenStack"]
        for name, kb in kb_by_name.items()
    ]
    for ds, ds_cases, name, kb in iter_with_progress(jobs, f"Evaluating {match_mode}"):
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
    for domain in ("hdfs", "openstack"):
        domain_edges = [e for e in edges if str(e.get("domain", "")).lower() == domain]
        stats[domain] = {
            "edges": len(domain_edges),
            "relations": dict(Counter(str(e.get("relation", "")) for e in domain_edges)),
        }
    return stats
