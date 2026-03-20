from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

_SCRIPT_DIR = Path(__file__).resolve().parent
_FINAL_ROOT = _SCRIPT_DIR.parent
_REBUILD_ROOT = _FINAL_ROOT.parents[0]
_PROJECT_ROOT = _FINAL_ROOT.parents[2]

_FULLCASE_SCRIPTS = _REBUILD_ROOT / "rq2_fullcase" / "scripts"
_AUDIT_SCRIPTS = _REBUILD_ROOT / "rq2_fullcase_audit_20260318" / "scripts"
_OPENSTACK_REDESIGN_SCRIPTS = _REBUILD_ROOT / "rq2_openstack_redesign_20260318" / "scripts"
_HADOOP_AUDIT_SCRIPTS = _REBUILD_ROOT / "rq2_hadoop_family_audit_20260318" / "scripts"

for path in [
    _PROJECT_ROOT,
    _PROJECT_ROOT / "experiments" / "rq123_e2e",
    _FULLCASE_SCRIPTS,
    _AUDIT_SCRIPTS,
    _OPENSTACK_REDESIGN_SCRIPTS,
    _HADOOP_AUDIT_SCRIPTS,
]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from rq2_fullcase_common_20260316 import (  # type: ignore
    SYMBOLIC_KB_PATH,
    _trim_top_variance_columns,
    canonical_tokens,
    exact_relaxed_match,
    family_of,
    fuzzy_match,
    load_hadoop_timeseries,
    load_hdfs_timeseries,
    load_openstack_semantic_timeseries,
)
from rq2_fullcase_audit_common_20260318 import (  # type: ignore
    OLD_OPENSTACK_ID_MAP_PATH,
    OLD_OPENSTACK_TS_PATH,
    UNDIRECTED_RELATIONS,
    build_original_dynotears_edges as _build_original_dynotears_edges,
    build_pc_cpdag_hypothesis_edges as _build_pc_cpdag_hypothesis_edges,
    build_pearson_hypothesis_edges as _build_pearson_hypothesis_edges,
    merge_edges_prefer_stronger,
)


RESULTS_DIR = _FINAL_ROOT / "results"
REPORTS_DIR = _FINAL_ROOT / "reports"
CACHE_DIR = RESULTS_DIR / "graph_cache_20260318"

SOURCE_HDFS_BENCH_PATH = (
    _REBUILD_ROOT
    / "rq2_fullcase_audit_20260318"
    / "results"
    / "rq2_fullcase_audit_benchmark_evaluable_20260318.json"
)
SOURCE_OPENSTACK_BENCH_PATH = (
    _REBUILD_ROOT
    / "rq2_openstack_redesign_20260318"
    / "results"
    / "rq2_openstack_redesign_openstack_core_benchmark_20260318.json"
)
SOURCE_HADOOP_FULL_BENCH_PATH = (
    _REBUILD_ROOT
    / "rq2_hadoop_family_audit_20260318"
    / "results"
    / "rq2_hadoop_family_audit_benchmark_full_20260318.json"
)
SOURCE_HADOOP_LOCAL_CORE_PATH = (
    _REBUILD_ROOT
    / "rq2_hadoop_family_audit_20260318"
    / "results"
    / "rq2_hadoop_family_audit_benchmark_local_core_20260318.json"
)

UNIFIED_BENCHMARK_ALL_PATH = RESULTS_DIR / "rq2_mainline_benchmark_all_20260318.json"
UNIFIED_BENCHMARK_MAINLINE_PATH = RESULTS_DIR / "rq2_mainline_benchmark_mainline_20260318.json"
UNIFIED_BENCHMARK_APPENDIX_PATH = RESULTS_DIR / "rq2_mainline_benchmark_appendix_20260318.json"

HADOOP_CALIBRATION_PATH = RESULTS_DIR / "rq2_mainline_hadoop_feature_calibration_20260318.json"
FEATURE_SPACE_PATH = RESULTS_DIR / "rq2_mainline_feature_spaces_20260318.json"
MODIFIED_PROVENANCE_PATH = RESULTS_DIR / "rq2_mainline_modified_provenance_20260318.json"
PATH_DIAGNOSTIC_PATH = RESULTS_DIR / "rq2_mainline_path_diagnostic_20260318.json"
CURATED_PRIOR_PATH = RESULTS_DIR / "rq2_mainline_curated_symbolic_priors_20260318.json"

GRAPH_FILES = {
    "modified": RESULTS_DIR / "gt_causal_knowledge_modified_mainline_completion_20260318.json",
    "original_dynotears": RESULTS_DIR / "gt_causal_knowledge_original_dynotears_mainline_completion_20260318.json",
    "pearson_hypothesis": RESULTS_DIR / "gt_causal_knowledge_pearson_hypothesis_mainline_completion_20260318.json",
    "pc_cpdag_hypothesis": RESULTS_DIR / "gt_causal_knowledge_pc_cpdag_hypothesis_mainline_completion_20260318.json",
}

DATASET_DOMAIN = {"HDFS": "hdfs", "OpenStack": "openstack", "Hadoop": "hadoop"}
OTHER_FAMILY = {"HDFS": "HDFS_OTHER", "OpenStack": "OS_OTHER", "Hadoop": "HADOOP_UNKNOWN"}
GRAPH_PARAMS = {
    "HDFS": {
        "lambda_w": 0.025,
        "lambda_a": 0.05,
        "threshold": 0.30,
        "pearson_threshold": 0.30,
        "pc_alpha": 0.05,
    },
    "OpenStack": {
        "lambda_w": 0.030,
        "lambda_a": 0.06,
        "threshold": 0.40,
        "pearson_threshold": 0.91,
        "pc_alpha": 0.30,
    },
    "Hadoop": {
        "lambda_w": 0.025,
        "lambda_a": 0.05,
        "threshold": 0.30,
        "pearson_threshold": 0.93,
        "pc_alpha": 0.05,
    },
}
HADOOP_CAP_CANDIDATES = [64, 48, 32]
HADOOP_ORIGINAL_TIMEOUT_SEC = 15 * 60
HADOOP_PC_TIMEOUT_SEC = 10 * 60
SYMBOLIC_PRIOR_WEIGHT_BOOST = 0.5
MERGED_WEIGHT_BOOST = 0.25
MODIFIED_MAX_INCOMING_PER_TARGET = 2
PRUNE_GENERIC_SOURCE_BACKBONE_FOR_MODIFIED = True
PRUNE_TRANSITIVE_SHORTCUTS_FOR_MODIFIED = True
TRANSITIVE_SHORTCUT_MIN_GAIN = 0.0

DEFAULT_CURATED_PRIORS = [
    {
        "domain": "openstack",
        "source_template": "Unknown base file: <*>",
        "target_template": "Removable base files: <*>",
        "weight": 0.98,
        "evidence": "raw_log_temporal_dominance",
    },
    {
        "domain": "openstack",
        "source_template": "Removable base files: <*>",
        "target_template": "Base or swap file too young to remove: <*>",
        "weight": 0.97,
        "evidence": "raw_log_temporal_dominance",
    },
    {
        "domain": "openstack",
        "source_template": "Active base files: <*>",
        "target_template": "Unknown base file: <*>",
        "weight": 0.96,
        "evidence": "raw_log_temporal_dominance",
    },
    {
        "domain": "openstack",
        "source_template": 'Identity response: <!DOCTYPE HTML PUBLIC "-<*> HTML <*>.<*><*>">',
        "target_template": "Bad response code while validating token: <*>",
        "weight": 0.97,
        "evidence": "domain_semantic_prior",
    },
    {
        "domain": "openstack",
        "source_template": "While synchronizing instance power states, found <*> instances in the database and <*> instances on the hypervisor.",
        "target_template": "nova-compute.log.<*>.<*>-<*>-<*>_<*>:<*>:<*>-<*>-<*>:<*>:<*>.<*> INFO nova.compute.manager [-] [instance: <*>] During sync_power_state the instance has a pending task <*> Skip.",
        "weight": 0.95,
        "evidence": "raw_log_temporal_dominance",
    },
    {
        "domain": "openstack",
        "source_template": "Successfully synced instances from host 'cp-<*>.slowvm<*>.tcloud-pg<*>.utah.cloudlab.us'.",
        "target_template": "The instance sync for host 'cp-<*>.slowvm<*>.tcloud-pg<*>.utah.cloudlab.us' did not match. Re-created its InstanceList.",
        "weight": 0.95,
        "evidence": "raw_log_temporal_dominance",
    },
    {
        "domain": "hdfs",
        "source_template": "[*]PacketResponder[*]for block[*]terminating[*]",
        "target_template": "[*]Got exception while serving[*]to[*]",
        "weight": 0.94,
        "evidence": "domain_semantic_prior",
    },
    {
        "domain": "hdfs",
        "source_template": "[*]Received block[*]src:[*]dest:[*]of size[*]",
        "target_template": "[*]Got exception while serving[*]to[*]",
        "weight": 0.97,
        "evidence": "raw_log_temporal_dominance",
    },
    {
        "domain": "hdfs",
        "source_template": "[*]Receiving block[*]src:[*]dest:[*]",
        "target_template": "[*]BLOCK* NameSystem[*]allocateBlock:[*]",
        "weight": 0.96,
        "evidence": "domain_semantic_prior",
    },
    {
        "domain": "hadoop",
        "source_template": "<*>-<*>-<*>:<*>:<*>,<*> INFO [main] org.apache.hadoop.mapred.MapTask: (RESET) equator <*> kv <*>(<*>) kvi <*>(<*>)",
        "target_template": "<*>-<*>-<*>:<*>:<*>,<*> WARN [CommitterEvent Processor #<*>] org.apache.hadoop.mapreduce.lib.output.FileOutputCommitter: Could not delete hdfs:<*>:<*><*>",
        "weight": 0.94,
        "evidence": "raw_log_temporal_dominance",
    },
    {
        "domain": "hadoop",
        "source_template": "<*>-<*>-<*>:<*>:<*>,<*> INFO [main] org.apache.hadoop.mapred.MapTask: (RESET) equator <*> kv <*>(<*>) kvi <*>(<*>)",
        "target_template": "<*>-<*>-<*>:<*>:<*>,<*> WARN <*> org.apache.hadoop.ipc.Client: Address change detected. Old: msra-sa-<*>/<*>:<*> New: msra-sa-<*>:<*>",
        "weight": 0.92,
        "evidence": "raw_log_temporal_dominance",
    },
    {
        "domain": "hadoop",
        "source_template": "<*>-<*>-<*>:<*>:<*>,<*> WARN [LeaseRenewer:msrabi@msra-sa-<*>:<*>] org.apache.hadoop.hdfs.LeaseRenewer: Failed to renew lease for [DFSClient_NONMAPREDUCE_<*>_<*>] for <*> seconds. Will retry shortly ...",
        "target_template": "<*>-<*>-<*>:<*>:<*>,<*> WARN [CommitterEvent Processor #<*>] org.apache.hadoop.mapreduce.lib.output.FileOutputCommitter: Could not delete hdfs:<*>:<*><*>",
        "weight": 0.91,
        "evidence": "raw_log_temporal_dominance",
    },
]


def ensure_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def refresh_curated_prior_file() -> None:
    write_json(CURATED_PRIOR_PATH, DEFAULT_CURATED_PRIORS)


def _norm(text: str) -> str:
    return " ".join(str(text or "").strip().lower().split())


class Heartbeat:
    def __init__(self, label: str, interval_sec: int = 30, remaining: str = "") -> None:
        self.label = label
        self.interval_sec = interval_sec
        self.remaining = remaining
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._start = 0.0

    def __enter__(self) -> "Heartbeat":
        self._start = time.time()

        def _run() -> None:
            while not self._stop.wait(self.interval_sec):
                elapsed = int(time.time() - self._start)
                msg = f"[heartbeat] {self.label} running for {elapsed}s"
                if self.remaining:
                    msg += f" | remaining substeps: {self.remaining}"
                print(msg, flush=True)

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)


def _effect_target_type(dataset: str) -> str:
    return "template"


def _root_target_type(dataset: str) -> str:
    return "family" if dataset == "Hadoop" else "template"


def _unify_hdfs_or_openstack_row(
    row: Dict[str, object],
    dataset: str,
    benchmark_source_workspace: str,
) -> Dict[str, object]:
    return {
        "dataset": dataset,
        "case_id": str(row.get("case_id", "") or ""),
        "effect_target_type": "template",
        "effect_target_value": str(row.get("ground_truth_template_graph", "") or ""),
        "root_target_type": "template",
        "root_target_value": str(row.get("ground_truth_root_cause_template_graph", "") or ""),
        "benchmark_tier": "mainline",
        "benchmark_source_workspace": benchmark_source_workspace,
        "manual_prior_pair_overlap": bool(row.get("manual_prior_pair_overlap")),
        "effect_target_label": str(row.get("ground_truth_template_label", row.get("ground_truth_template", "")) or ""),
        "root_target_label": str(row.get("ground_truth_root_cause_label", row.get("ground_truth_root_cause_template", "")) or ""),
    }


def _unify_hadoop_row(
    row: Dict[str, object],
    benchmark_tier: str,
    benchmark_source_workspace: str,
    case_id_prefix: str = "",
) -> Dict[str, object]:
    raw_case_id = str(row.get("case_id", "") or "")
    case_id = raw_case_id if not case_id_prefix else f"{case_id_prefix}::{raw_case_id}"
    return {
        "dataset": "Hadoop",
        "case_id": case_id,
        "effect_target_type": "template",
        "effect_target_value": str(row.get("ground_truth_template_graph", "") or ""),
        "root_target_type": "family",
        "root_target_value": str(row.get("ground_truth_root_family", "") or ""),
        "benchmark_tier": benchmark_tier,
        "benchmark_source_workspace": benchmark_source_workspace,
        "manual_prior_pair_overlap": bool(row.get("manual_prior_pair_overlap", False)),
        "effect_target_label": str(row.get("ground_truth_template_label", row.get("ground_truth_template", "")) or ""),
        "root_target_label": str(row.get("ground_truth_root_cause_label", "") or ""),
    }


def build_unified_benchmark_rows() -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]]]:
    hdfs_openstack_rows = load_json(SOURCE_HDFS_BENCH_PATH)
    openstack_core_rows = load_json(SOURCE_OPENSTACK_BENCH_PATH)
    hadoop_full_rows = load_json(SOURCE_HADOOP_FULL_BENCH_PATH)

    mainline_rows: List[Dict[str, object]] = []
    appendix_rows: List[Dict[str, object]] = []

    for row in hdfs_openstack_rows:
        if str(row.get("dataset", "") or "") == "HDFS":
            mainline_rows.append(_unify_hdfs_or_openstack_row(row, "HDFS", "rq2_fullcase_audit_20260318"))
    for row in openstack_core_rows:
        mainline_rows.append(_unify_hdfs_or_openstack_row(row, "OpenStack", "rq2_openstack_redesign_20260318"))
    for row in hadoop_full_rows:
        if bool(row.get("audit_derivation_complete")) and str(row.get("audit_local_root_status_radius80", "") or "") == "unique_graph":
            mainline_rows.append(_unify_hadoop_row(row, "mainline", "rq2_hadoop_family_audit_20260318::core80"))
        appendix_rows.append(
            _unify_hadoop_row(
                row,
                "appendix",
                "rq2_hadoop_family_audit_20260318::full44",
                case_id_prefix="hadoop_full_appendix",
            )
        )
    for row in load_json(SOURCE_HADOOP_LOCAL_CORE_PATH):
        appendix_rows.append(
            _unify_hadoop_row(
                row,
                "appendix",
                "rq2_hadoop_family_audit_20260318::local40",
                case_id_prefix="hadoop_local_core",
            )
        )
    all_rows = mainline_rows + appendix_rows
    return all_rows, mainline_rows, appendix_rows


def _drop_constant_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    nunique = df.nunique()
    keep_cols = list(nunique[nunique > 1].index)
    if not keep_cols:
        return df.iloc[:, :0]
    return df[keep_cols]


def _scored_graph_templates(
    dataset: str,
    query_texts: List[str],
    pool: List[str],
    desired_family: str = "",
) -> List[Tuple[float, str]]:
    queries = [str(x or "").strip() for x in query_texts if str(x or "").strip()]
    if not queries:
        return []
    other_family = OTHER_FAMILY[dataset]
    query_token_sets = [set(canonical_tokens(q)) for q in queries]
    scored: List[Tuple[float, str]] = []
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
            score += len(set(canonical_tokens(tpl)) & toks) * 4.0
            if q.lower() in tpl.lower() or tpl.lower() in q.lower():
                score += 20.0
        scored.append((score, tpl))
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored


def candidate_graph_templates_any(
    dataset: str,
    query_texts: List[str],
    pool: List[str],
    desired_family: str = "",
    max_candidates: int = 3,
    min_relative_score: float = 0.85,
    min_abs_score: float = 150.0,
) -> List[str]:
    scored = _scored_graph_templates(dataset, query_texts, pool, desired_family=desired_family)
    if not scored:
        return []
    best_score = float(scored[0][0])
    if best_score <= 0.0:
        return []

    keep: List[str] = []
    seen = set()
    exact_band = best_score >= 1000.0
    cutoff = max(min_abs_score, best_score * min_relative_score)
    for score, tpl in scored:
        if len(keep) >= max_candidates:
            break
        if exact_band:
            if score < 1000.0:
                continue
        elif score < cutoff:
            continue
        norm = _norm(tpl)
        if norm in seen:
            continue
        seen.add(norm)
        keep.append(tpl)
    if keep:
        return keep
    return [str(scored[0][1])]


def best_graph_template_any(
    dataset: str,
    query_texts: List[str],
    pool: List[str],
    desired_family: str = "",
) -> str:
    candidates = candidate_graph_templates_any(dataset, query_texts, pool, desired_family=desired_family, max_candidates=1)
    return candidates[0] if candidates else ""


def _dataset_template_pool(dataset: str, tpl_map: Dict[str, str]) -> List[str]:
    pool = [str(v or "") for v in tpl_map.values() if str(v or "").strip()]
    return list(dict.fromkeys(pool))


def _required_effect_templates(dataset: str) -> List[str]:
    if not UNIFIED_BENCHMARK_MAINLINE_PATH.exists():
        return []
    rows = load_json(UNIFIED_BENCHMARK_MAINLINE_PATH)
    templates = [
        str(row.get("effect_target_value", "") or "")
        for row in rows
        if str(row.get("dataset", "") or "") == dataset
        and str(row.get("benchmark_tier", "") or "") == "mainline"
    ]
    seen = set()
    unique = []
    for tpl in templates:
        norm = _norm(tpl)
        if not tpl or norm in seen:
            continue
        seen.add(norm)
        unique.append(tpl)
    return unique


def _load_dataset_timeseries(dataset: str) -> Tuple[pd.DataFrame, Dict[str, str]]:
    if dataset == "HDFS":
        return load_hdfs_timeseries()
    if dataset == "OpenStack":
        return load_openstack_semantic_timeseries(OLD_OPENSTACK_TS_PATH, OLD_OPENSTACK_ID_MAP_PATH)
    if dataset == "Hadoop":
        return load_hadoop_timeseries()
    raise ValueError(dataset)


def prepare_feature_space(
    dataset: str,
    max_cols: int = 0,
    explicit_columns: List[str] | None = None,
) -> Tuple[pd.DataFrame, Dict[str, str], Dict[str, object]]:
    df, tpl_map = _load_dataset_timeseries(dataset)
    original_cols = list(df.columns)
    work_df = _drop_constant_columns(df)
    constant_removed = len(original_cols) - work_df.shape[1]
    duplicate_observed = int(work_df.T.duplicated().sum()) if not work_df.empty else 0
    required_templates = _required_effect_templates(dataset) if dataset == "Hadoop" and max_cols > 0 and explicit_columns is None else []

    if explicit_columns is not None:
        missing = [col for col in explicit_columns if col not in work_df.columns]
        if missing:
            raise ValueError(f"Missing explicit columns for {dataset}: {missing[:5]}")
        work_df = work_df[explicit_columns]
    elif max_cols > 0 and work_df.shape[1] > max_cols:
        mandatory_cols: List[str] = []
        for target_tpl in required_templates:
            matches = [col for col in work_df.columns if exact_relaxed_match(str(tpl_map.get(col, "") or ""), target_tpl)]
            if not matches:
                continue
            if len(matches) == 1:
                chosen = matches[0]
            else:
                variances = work_df[matches].var(axis=0).sort_values(ascending=False)
                chosen = str(variances.index[0])
            if chosen not in mandatory_cols:
                mandatory_cols.append(chosen)
        if len(mandatory_cols) > max_cols:
            raise ValueError(f"Required templates for {dataset} exceed cap {max_cols}: {len(mandatory_cols)}")
        remaining_budget = max_cols - len(mandatory_cols)
        remaining_df = work_df.drop(columns=mandatory_cols, errors="ignore")
        if remaining_budget > 0 and not remaining_df.empty:
            remaining_df = _trim_top_variance_columns(remaining_df, remaining_budget)
            work_df = pd.concat([work_df[mandatory_cols], remaining_df], axis=1)
        else:
            work_df = work_df[mandatory_cols]

    selected_cols = list(work_df.columns)
    selected_tpl_map = {col: tpl_map[col] for col in selected_cols}
    profile = {
        "dataset": dataset,
        "original_columns": len(original_cols),
        "constant_removed": constant_removed,
        "duplicate_removed": 0,
        "duplicate_observed": duplicate_observed,
        "selected_columns": len(selected_cols),
        "column_ids": selected_cols,
        "required_effect_templates": required_templates,
    }
    if max_cols > 0:
        profile["cap"] = max_cols
    return work_df, selected_tpl_map, profile


def _measure_cache_path(dataset: str, method: str, cap: int) -> Path:
    return CACHE_DIR / f"measure_{dataset.lower()}_{method}_{cap}_v2_20260318.json"


def run_hadoop_measurement_subprocess(method: str, cap: int, timeout_sec: int) -> Dict[str, object]:
    cache_path = _measure_cache_path("Hadoop", method, cap)
    if cache_path.exists():
        return load_json(cache_path)

    helper = _SCRIPT_DIR / "measure_rq2_mainline_candidate_20260318.py"
    cmd = [sys.executable, str(helper), "--dataset", "Hadoop", "--method", method, "--cap", str(cap)]
    started = time.time()
    timed_out = False
    result_data: Dict[str, object]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=True,
        )
        stdout = proc.stdout.strip().splitlines()
        payload = json.loads(stdout[-1]) if stdout else {}
        result_data = {
            "dataset": "Hadoop",
            "method": method,
            "cap": cap,
            "timed_out": False,
            "elapsed_sec": round(time.time() - started, 3),
            "edges": int(payload.get("edges", 0)),
            "selected_columns": int(payload.get("selected_columns", 0)),
        }
    except subprocess.TimeoutExpired:
        timed_out = True
        result_data = {
            "dataset": "Hadoop",
            "method": method,
            "cap": cap,
            "timed_out": True,
            "elapsed_sec": round(time.time() - started, 3),
            "edges": None,
            "selected_columns": None,
        }
    write_json(cache_path, result_data)
    if timed_out:
        return result_data
    return result_data


def graph_cache_path(dataset: str, method: str) -> Path:
    return CACHE_DIR / f"{dataset.lower()}_{method}_20260318.json"


def _clean_edges_for_domain(dataset: str, edges: List[Dict[str, object]]) -> List[Dict[str, object]]:
    domain = DATASET_DOMAIN[dataset]
    filtered = []
    for edge in edges:
        if str(edge.get("domain", "")).lower() != domain:
            continue
        src = str(edge.get("source_template", "") or "")
        tgt = str(edge.get("target_template", "") or "")
        if exact_relaxed_match(src, tgt):
            continue
        filtered.append(edge)
    return merge_edges_prefer_stronger(filtered)


def build_original_graph(dataset: str, df: pd.DataFrame, tpl_map: Dict[str, str]) -> List[Dict[str, object]]:
    params = GRAPH_PARAMS[dataset]
    edges = _build_original_dynotears_edges(
        df,
        tpl_map,
        DATASET_DOMAIN[dataset],
        lambda_w=float(params["lambda_w"]),
        lambda_a=float(params["lambda_a"]),
        threshold=float(params["threshold"]),
        p=1,
        max_iter=100,
    )
    return _clean_edges_for_domain(dataset, edges)


def build_pearson_graph(dataset: str, df: pd.DataFrame, tpl_map: Dict[str, str]) -> List[Dict[str, object]]:
    params = GRAPH_PARAMS[dataset]
    edges = _build_pearson_hypothesis_edges(
        df,
        tpl_map,
        DATASET_DOMAIN[dataset],
        threshold=float(params["pearson_threshold"]),
    )
    return _clean_edges_for_domain(dataset, edges)


def build_pc_graph(dataset: str, df: pd.DataFrame, tpl_map: Dict[str, str]) -> List[Dict[str, object]]:
    from causallearn.graph.Endpoint import Endpoint  # type: ignore
    from causallearn.search.ConstraintBased.PC import pc  # type: ignore

    params = GRAPH_PARAMS[dataset]
    if df.empty:
        return []
    X = StandardScaler().fit_transform(df.values)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    rng = np.random.default_rng(20260318)
    X = X + rng.normal(0.0, 1e-6, size=X.shape)
    cols = list(df.columns)

    cg = pc(X, alpha=float(params["pc_alpha"]), verbose=False, show_progress=False, node_names=cols)
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
                "domain": DATASET_DOMAIN[dataset],
                "source_template": src_tpl,
                "relation": rel,
                "target_template": tgt_tpl,
                "weight": weight,
            }
        )
    return _clean_edges_for_domain(dataset, edges)


def collect_mapped_symbolic_prior_edges(
    dataset: str,
    template_pool: List[str],
) -> List[Dict[str, object]]:
    symbolic_rows = load_json(SYMBOLIC_KB_PATH)
    domain = DATASET_DOMAIN[dataset]
    priors: List[Dict[str, object]] = []
    for edge in symbolic_rows:
        if str(edge.get("domain", "")).lower() != domain:
            continue
        raw_src = str(edge.get("source_template", "") or "")
        raw_tgt = str(edge.get("target_template", "") or "")
        if not raw_src or not raw_tgt:
            continue
        mapped_src = best_graph_template_any(
            dataset,
            [raw_src],
            template_pool,
            desired_family=family_of(dataset, raw_src),
        )
        mapped_tgt = best_graph_template_any(
            dataset,
            [raw_tgt],
            template_pool,
            desired_family=family_of(dataset, raw_tgt),
        )
        if not mapped_src or not mapped_tgt:
            continue
        if exact_relaxed_match(mapped_src, mapped_tgt):
            continue
        priors.append(
            {
                "domain": domain,
                "source_template": mapped_src,
                "relation": "symbolic_prior",
                "target_template": mapped_tgt,
                "weight": float(round(max(0.55, abs(float(edge.get("weight", 0.0) or 0.0))), 4)),
                "raw_symbolic_source": raw_src,
                "raw_symbolic_target": raw_tgt,
            }
        )
    if CURATED_PRIOR_PATH.exists():
        curated_rows = load_json(CURATED_PRIOR_PATH)
        template_set = {_norm(tpl) for tpl in template_pool}
        for edge in curated_rows:
            if str(edge.get("domain", "")).lower() != domain:
                continue
            src = str(edge.get("source_template", "") or "")
            tgt = str(edge.get("target_template", "") or "")
            if _norm(src) not in template_set or _norm(tgt) not in template_set:
                continue
            if exact_relaxed_match(src, tgt):
                continue
            priors.append(
                {
                    "domain": domain,
                    "source_template": src,
                    "relation": "symbolic_prior",
                    "target_template": tgt,
                    "weight": float(round(max(0.55, abs(float(edge.get("weight", 0.0) or 0.0))), 4)),
                    "raw_symbolic_source": str(edge.get("source_template", "") or ""),
                    "raw_symbolic_target": str(edge.get("target_template", "") or ""),
                    "curated_symbolic_prior": True,
                }
            )
    return merge_edges_prefer_stronger(priors)


def merge_modified_edges(
    dataset: str,
    original_edges: List[Dict[str, object]],
    symbolic_edges: List[Dict[str, object]],
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    merged: Dict[Tuple[str, str, str], Dict[str, object]] = {}
    provenance_rows: List[Dict[str, object]] = []
    domain = DATASET_DOMAIN[dataset]

    def _upsert(edge: Dict[str, object], provenance_label: str) -> None:
        key = (
            domain,
            str(edge.get("source_template", "") or ""),
            str(edge.get("target_template", "") or ""),
        )
        current = merged.get(key)
        if current is None:
            edge2 = dict(edge)
            edge2["provenance"] = provenance_label
            merged[key] = edge2
            return
        current_sources = set(str(current.get("provenance", "") or "").split("+"))
        current_sources.discard("")
        current_sources.add(provenance_label)
        current["provenance"] = "merged" if len(current_sources) > 1 else provenance_label
        if abs(float(edge.get("weight", 0.0) or 0.0)) > abs(float(current.get("weight", 0.0) or 0.0)):
            current["weight"] = float(edge.get("weight", 0.0) or 0.0)
            current["relation"] = str(edge.get("relation", "") or current.get("relation", ""))

    for edge in original_edges:
        _upsert(edge, "original_backbone")
    for edge in symbolic_edges:
        _upsert(edge, "symbolic_prior")

    rows = list(merged.values())
    if PRUNE_GENERIC_SOURCE_BACKBONE_FOR_MODIFIED:
        other_family = OTHER_FAMILY[dataset]
        rows = [
            row
            for row in rows
            if not (
                str(row.get("provenance", "") or "") == "original_backbone"
                and family_of(dataset, str(row.get("source_template", "") or "")) == other_family
            )
        ]
    rows = _clean_edges_for_domain(dataset, rows)
    for row in rows:
        provenance = str(row.get("provenance", "") or "")
        boost = 0.0
        if provenance == "symbolic_prior":
            boost = SYMBOLIC_PRIOR_WEIGHT_BOOST
        elif provenance == "merged":
            boost = MERGED_WEIGHT_BOOST
        if boost > 0.0:
            weight = float(row.get("weight", 0.0) or 0.0)
            sign = -1.0 if weight < 0 else 1.0
            row["weight"] = float(round(sign * (abs(weight) + boost), 4))
    if MODIFIED_MAX_INCOMING_PER_TARGET > 0:
        buckets: Dict[str, List[Dict[str, object]]] = defaultdict(list)
        for row in rows:
            buckets[str(row.get("target_template", "") or "")].append(row)
        pruned_rows: List[Dict[str, object]] = []
        for target_rows in buckets.values():
            target_rows = sorted(
                target_rows,
                key=lambda row: abs(float(row.get("weight", 0.0) or 0.0)),
                reverse=True,
            )
            pruned_rows.extend(target_rows[:MODIFIED_MAX_INCOMING_PER_TARGET])
        rows = pruned_rows
    rows = _prune_transitive_shortcuts_for_modified(rows)
    for row in rows:
        provenance_rows.append(
            {
                "dataset": dataset,
                "domain": domain,
                "source_template": str(row.get("source_template", "") or ""),
                "target_template": str(row.get("target_template", "") or ""),
                "relation": str(row.get("relation", "") or ""),
                "weight": float(row.get("weight", 0.0) or 0.0),
                "provenance": str(row.get("provenance", "") or ""),
                "benchmark_derived": False,
            }
        )
    return rows, provenance_rows


def _prune_transitive_shortcuts_for_modified(
    rows: List[Dict[str, object]],
) -> List[Dict[str, object]]:
    if not PRUNE_TRANSITIVE_SHORTCUTS_FOR_MODIFIED:
        return rows

    by_source: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_source[str(row.get("source_template", "") or "")].append(row)

    removable: set[Tuple[str, str]] = set()
    for row in rows:
        if str(row.get("provenance", "") or "") != "original_backbone":
            continue
        src = str(row.get("source_template", "") or "")
        tgt = str(row.get("target_template", "") or "")
        direct_weight = abs(float(row.get("weight", 0.0) or 0.0))
        best_two_hop = -1.0
        for edge1 in by_source.get(src, []):
            mid = str(edge1.get("target_template", "") or "")
            if mid in (src, tgt):
                continue
            if str(edge1.get("provenance", "") or "") == "original_backbone":
                continue
            for edge2 in by_source.get(mid, []):
                if str(edge2.get("target_template", "") or "") != tgt:
                    continue
                if str(edge2.get("provenance", "") or "") == "original_backbone":
                    continue
                two_hop_support = min(
                    abs(float(edge1.get("weight", 0.0) or 0.0)),
                    abs(float(edge2.get("weight", 0.0) or 0.0)),
                )
                if two_hop_support > best_two_hop:
                    best_two_hop = two_hop_support
        if best_two_hop >= direct_weight + TRANSITIVE_SHORTCUT_MIN_GAIN:
            removable.add((src, tgt))

    if not removable:
        return rows
    return [
        row
        for row in rows
        if (str(row.get("source_template", "") or ""), str(row.get("target_template", "") or "")) not in removable
    ]


def graph_relation_stats(edges: List[Dict[str, object]], dataset: str) -> Dict[str, object]:
    domain = DATASET_DOMAIN[dataset]
    rows = [e for e in edges if str(e.get("domain", "")).lower() == domain]
    return {
        "dataset": dataset,
        "edges": len(rows),
        "relations": dict(Counter(str(e.get("relation", "") or "") for e in rows)),
    }


def _effect_match_kind(dataset: str, pred_effect: str, gt_effect: str, mode: str) -> str:
    if exact_relaxed_match(pred_effect, gt_effect):
        return "exact"
    if mode == "exact_only_edge":
        return "none"
    pred_family = family_of(dataset, pred_effect)
    gt_family = family_of(dataset, gt_effect)
    if pred_family == gt_family and pred_family != OTHER_FAMILY[dataset]:
        return "family"
    if fuzzy_match(pred_effect, gt_effect):
        return "fuzzy"
    return "none"


def _target_penalty(kind: str) -> int:
    if kind == "fuzzy":
        return 2
    return 0


def _relation_penalty(edge: Dict[str, object]) -> int:
    rel = str(edge.get("relation", "") or "")
    if rel == "pearson_undirected":
        return 1
    if rel == "pc_undirected":
        return 2
    if rel in {"pc_partially_oriented", "pc_bidirected", "pc_ambiguous"}:
        return 1
    return 0


def _root_matches(row: Dict[str, object], candidate_root: str) -> bool:
    dataset = str(row["dataset"])
    root_type = str(row["root_target_type"])
    root_value = str(row["root_target_value"])
    if root_type == "family":
        return family_of(dataset, candidate_root) == root_value and root_value != OTHER_FAMILY[dataset]
    return exact_relaxed_match(candidate_root, root_value)


def _edge_candidates_for_case(
    kb: List[Dict[str, object]],
    row: Dict[str, object],
    mode: str,
) -> List[Dict[str, object]]:
    dataset = str(row["dataset"])
    domain = DATASET_DOMAIN[dataset]
    gt_effect = str(row["effect_target_value"])
    candidates: List[Dict[str, object]] = []
    for edge in kb:
        if str(edge.get("domain", "")).lower() != domain:
            continue
        src = str(edge.get("source_template", "") or "")
        tgt = str(edge.get("target_template", "") or "")
        rel = str(edge.get("relation", "") or "")
        pairs = [(src, tgt)]
        if rel in UNDIRECTED_RELATIONS:
            pairs = [(src, tgt), (tgt, src)]
        for candidate_root, effect_side in pairs:
            kind = _effect_match_kind(dataset, effect_side, gt_effect, mode)
            if kind == "none":
                continue
            candidates.append(
                {
                    "candidate_root": candidate_root,
                    "effect_side": effect_side,
                    "kind": kind,
                    "edge": edge,
                }
            )
    return candidates


def calc_edge_rank(kb: List[Dict[str, object]], row: Dict[str, object], mode: str) -> Tuple[int, int]:
    dataset = str(row["dataset"])
    domain = DATASET_DOMAIN[dataset]
    edges_domain = [e for e in kb if str(e.get("domain", "")).lower() == domain]
    sparsity = len(edges_domain)
    candidates = _edge_candidates_for_case(kb, row, mode)
    if not candidates:
        return sparsity, -1
    scored = sorted(
        ((abs(float(c["edge"].get("weight", 0.0) or 0.0)), c) for c in candidates),
        key=lambda x: x[0],
        reverse=True,
    )
    best_rank = -1
    for idx, (_, candidate) in enumerate(scored, start=1):
        if not _root_matches(row, str(candidate["candidate_root"] or "")):
            continue
        cand_rank = idx + _relation_penalty(candidate["edge"]) + _target_penalty(str(candidate["kind"]))
        if best_rank < 0 or cand_rank < best_rank:
            best_rank = cand_rank
    return sparsity, best_rank


def _build_transitions(kb: List[Dict[str, object]], dataset: str) -> List[Dict[str, object]]:
    domain = DATASET_DOMAIN[dataset]
    transitions: List[Dict[str, object]] = []
    for edge in kb:
        if str(edge.get("domain", "")).lower() != domain:
            continue
        src = str(edge.get("source_template", "") or "")
        tgt = str(edge.get("target_template", "") or "")
        rel = str(edge.get("relation", "") or "")
        transitions.append({"src": src, "tgt": tgt, "edge": edge})
        if rel in UNDIRECTED_RELATIONS:
            transitions.append({"src": tgt, "tgt": src, "edge": edge})
    return transitions


def calc_path2_rank(kb: List[Dict[str, object]], row: Dict[str, object]) -> Tuple[int, int]:
    sparsity, direct_rank = calc_edge_rank(kb, row, "task_aligned_edge")
    if direct_rank >= 0:
        return sparsity, direct_rank

    dataset = str(row["dataset"])
    gt_effect = str(row["effect_target_value"])
    transitions = _build_transitions(kb, dataset)
    by_src: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for trans in transitions:
        by_src[str(trans["src"])].append(trans)

    path_candidates: List[Tuple[Tuple[float, float], Dict[str, object]]] = []
    for t1 in transitions:
        for t2 in by_src.get(str(t1["tgt"]), []):
            kind = _effect_match_kind(dataset, str(t2["tgt"]), gt_effect, "task_aligned_edge")
            if kind == "none":
                continue
            root_node = str(t1["src"])
            if not _root_matches(row, root_node):
                continue
            score = (
                min(abs(float(t1["edge"].get("weight", 0.0) or 0.0)), abs(float(t2["edge"].get("weight", 0.0) or 0.0))),
                abs(float(t1["edge"].get("weight", 0.0) or 0.0)) + abs(float(t2["edge"].get("weight", 0.0) or 0.0)),
            )
            path_candidates.append(
                (
                    score,
                    {
                        "kind": kind,
                        "edge1": t1["edge"],
                        "edge2": t2["edge"],
                    },
                )
            )
    if not path_candidates:
        return sparsity, -1
    path_candidates.sort(key=lambda item: item[0], reverse=True)
    best_rank = -1
    for idx, (_, candidate) in enumerate(path_candidates, start=1):
        cand_rank = (
            idx
            + _relation_penalty(candidate["edge1"])
            + _relation_penalty(candidate["edge2"])
            + _target_penalty(str(candidate["kind"]))
            + 1
        )
        if best_rank < 0 or cand_rank < best_rank:
            best_rank = cand_rank
    return sparsity, best_rank


def _matching_root_nodes(row: Dict[str, object], nodes: Iterable[str]) -> List[str]:
    return [node for node in nodes if _root_matches(row, node)]


def shortest_matching_path(kb: List[Dict[str, object]], row: Dict[str, object], max_hops: int = 3) -> int:
    dataset = str(row["dataset"])
    gt_effect = str(row["effect_target_value"])
    transitions = _build_transitions(kb, dataset)
    nodes = set()
    by_src: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for trans in transitions:
        nodes.add(str(trans["src"]))
        nodes.add(str(trans["tgt"]))
        by_src[str(trans["src"])].append(trans)
    roots = _matching_root_nodes(row, nodes)
    if not roots:
        return -1
    effects = {
        node
        for node in nodes
        if _effect_match_kind(dataset, node, gt_effect, "task_aligned_edge") != "none"
    }
    if not effects:
        return -1
    for root in roots:
        frontier = [(root, 0)]
        seen = {root}
        while frontier:
            node, depth = frontier.pop(0)
            if depth >= max_hops:
                continue
            for trans in by_src.get(node, []):
                nxt = str(trans["tgt"])
                if nxt in effects:
                    return depth + 1
                if nxt in seen:
                    continue
                seen.add(nxt)
                frontier.append((nxt, depth + 1))
    return -1


def evaluate_rows(
    graph_paths: Dict[str, Path],
    benchmark_rows: List[Dict[str, object]],
    mode: str,
) -> List[Dict[str, object]]:
    kb_by_name = {name: load_json(path) for name, path in graph_paths.items()}
    rows: List[Dict[str, object]] = []
    for dataset in ["HDFS", "OpenStack", "Hadoop"]:
        dataset_rows = [row for row in benchmark_rows if str(row["dataset"]) == dataset and str(row["benchmark_tier"]) == "mainline"]
        for name, kb in kb_by_name.items():
            sparsity_sum = 0.0
            rankable = 0
            rank_sum = 0.0
            for row in dataset_rows:
                if mode == "task_aligned_path2":
                    sparsity, rank = calc_path2_rank(kb, row)
                else:
                    sparsity, rank = calc_edge_rank(kb, row, mode)
                sparsity_sum += float(sparsity)
                if rank >= 0:
                    rankable += 1
                    rank_sum += float(rank)
            n = len(dataset_rows) or 1
            rows.append(
                {
                    "dataset": dataset,
                    "method": name,
                    "mode": mode,
                    "cases": len(dataset_rows),
                    "rankable": rankable,
                    "sparsity_mean": round(sparsity_sum / n, 4),
                    "avg_rank": None if rankable == 0 else round(rank_sum / rankable, 4),
                }
            )
    return rows


def summarize_rows(rows: List[Dict[str, object]]) -> str:
    lines = [
        "| Dataset | Method | Mode | Cases | Rankable | Sparsity_mean | Avg_Rank |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        avg = "nan" if row["avg_rank"] is None else str(row["avg_rank"])
        lines.append(
            f"| {row['dataset']} | {row['method']} | {row['mode']} | {row['cases']} | {row['rankable']} | {row['sparsity_mean']} | {avg} |"
        )
    return "\n".join(lines) + "\n"


def build_path_diagnostic_rows(
    graph_paths: Dict[str, Path],
    benchmark_rows: List[Dict[str, object]],
) -> List[Dict[str, object]]:
    kb_by_name = {name: load_json(path) for name, path in graph_paths.items()}
    rows: List[Dict[str, object]] = []
    for dataset in ["HDFS", "OpenStack", "Hadoop"]:
        dataset_rows = [row for row in benchmark_rows if str(row["dataset"]) == dataset and str(row["benchmark_tier"]) == "mainline"]
        for name, kb in kb_by_name.items():
            counts = Counter()
            for row in dataset_rows:
                depth = shortest_matching_path(kb, row, max_hops=3)
                if depth == 1:
                    counts["direct"] += 1
                elif depth == 2:
                    counts["two_hop"] += 1
                elif depth == 3:
                    counts["three_hop"] += 1
                else:
                    counts["none"] += 1
            rows.append(
                {
                    "dataset": dataset,
                    "method": name,
                    "cases": len(dataset_rows),
                    "direct": counts["direct"],
                    "two_hop": counts["two_hop"],
                    "three_hop": counts["three_hop"],
                    "none": counts["none"],
                }
            )
    return rows


def path_diagnostic_markdown(rows: List[Dict[str, object]]) -> str:
    lines = [
        "| Dataset | Method | Cases | Direct | Two_Hop | Three_Hop | None |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['method']} | {row['cases']} | {row['direct']} | {row['two_hop']} | {row['three_hop']} | {row['none']} |"
        )
    return "\n".join(lines) + "\n"
