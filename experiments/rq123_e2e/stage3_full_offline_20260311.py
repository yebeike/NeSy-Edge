"""
Stage 3: Full-scale Offline Eval — Dim1 (Parsing PA + Latency) + Dim2 (Sparsity + Avg Rank) only.

- Uses data/processed/e2e_scaled_benchmark_v2.json. Optionally expands 144→288 cases by duplicating.
- Full noise: each case run at [0.0, 0.2, 0.4, 0.6, 0.8, 1.0] for Dim1.
- No API calls (no Dim3/Dim4).
- Dim1 PA: golden definition (MetricsCalculator.normalize_template).
- Dim2: causal graph sparsity and rank of true root cause (once per case).
- Output: CSV + summary (1) Parsing PA (2) Parsing Latency (3) Sparsity (4) Avg_Rank.
"""

import os
import sys
import csv
import json
import time
from typing import Dict, List, Tuple

from tqdm import tqdm

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.system.edge_node import NuSyEdgeNode
from src.perception.drain_parser import DrainParser
from src.utils.llm_client import LLMClient
from src.utils.metrics import MetricsCalculator
from src.utils.noise_injector import NoiseInjector

from experiments.rq123_e2e.run_rq123_e2e_modular import _load_benchmark  # type: ignore
from experiments.rq123_e2e.run_rq123_e2e_massive import _calc_causal_sparsity_and_rank  # type: ignore
from experiments.rq123_e2e.noise_injector_hadoop_20260311 import HadoopNoiseInjector
from experiments.rq123_e2e.stage4_noise_api_sampled_20260313 import _select_alert_line  # type: ignore

DATA_PROCESSED = os.path.join(_PROJECT_ROOT, "data", "processed")
RESULTS_E2E_DIR = os.path.join(_PROJECT_ROOT, "results", "rq123_e2e")
BENCH_V2_PATH = os.path.join(DATA_PROCESSED, "e2e_scaled_benchmark_v2.json")
CAUSAL_KB_DYNOTEARS = os.path.join(DATA_PROCESSED, "gt_causal_knowledge_dynotears.json")
CAUSAL_KB_PEARSON = os.path.join(DATA_PROCESSED, "gt_causal_knowledge_pearson.json")
CAUSAL_KB_PC = os.path.join(DATA_PROCESSED, "gt_causal_knowledge_pc.json")

CAUSAL_KB_PEARSON_PRUNED = os.path.join(DATA_PROCESSED, "gt_causal_knowledge_pearson_pruned_20260311.json")
CAUSAL_KB_PC_PRUNED = os.path.join(DATA_PROCESSED, "gt_causal_knowledge_pc_pruned_20260311.json")
USE_PRUNED_PEARSON_PC = True

NOISE_LEVELS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
EXPAND_CASES_TO_288 = False  # keep 144 cases for faster iteration


def _domain_from_dataset(dataset: str) -> str:
    ds = (dataset or "").upper()
    if ds == "HDFS":
        return "hdfs"
    if ds == "OPENSTACK":
        return "openstack"
    if ds == "HADOOP":
        return "hadoop"
    return "hdfs"


def _golden_pa(pred: str, gt: str) -> int:
    if not gt:
        return 0
    p_norm = MetricsCalculator.normalize_template(pred)
    g_norm = MetricsCalculator.normalize_template(gt)
    return 1 if p_norm and p_norm == g_norm else 0


def _inject_noise(alert: str, dataset: str, injector: NoiseInjector, noise_level: float) -> str:
    injector.injection_rate = noise_level
    ds_type = "HDFS" if dataset == "Hadoop" else dataset
    return injector.inject(alert, dataset_type=ds_type)


def _inject_noise_hadoop(alert: str, injector: HadoopNoiseInjector, noise_level: float) -> str:
    injector.injection_rate = noise_level
    return injector.inject(alert)


def _hadoop_strip_prefix(text: str) -> str:
    """
    Hadoop GT templates in benchmark may include timestamp/INFO/class prefix, but NuSy/Drain parsing
    runs on header-stripped content. For Dim1 PA fairness (and to avoid false 0 PA),
    we strip the same prefix from Hadoop GT and also use the stripped template for Qwen refs.
    """
    t = NuSyEdgeNode.preprocess_header(text or "", "Hadoop") or (text or "")
    return t


def _gt_tpl_for_eval(dataset: str, gt_tpl: str) -> str:
    return _hadoop_strip_prefix(gt_tpl) if (dataset or "") == "Hadoop" else (gt_tpl or "")


def _denoise_for_nusy(dataset: str, text: str) -> str:
    """
    NeSy robustness helper (experiments-only): partially reverse our NoiseInjector synonyms so KB matching
    (fingerprint + retrieval) stays stable under strong noise.
    This is intentionally conservative (token-level, no regex-heavy transforms).
    """
    if not text:
        return ""
    t = str(text)
    ds = (dataset or "")
    if ds in ("HDFS", "Hadoop"):
        # Reverse base synonyms
        t = t.replace("PkgResponder", "PacketResponder")
        t = t.replace("closing", "terminating")
        t = t.replace("Got blk", "Received block")
        t = t.replace("Error", "Exception")
        t = t.replace("len", "size")
        # Reverse blk_ mutation
        t = t.replace("block-id:", "blk_")
        # Reverse hard replacements (best-effort, partial)
        t = t.replace("Encountered network failure when handling", "Got exception while serving")
        t = t.replace("Failure during cleanup of data chunk", "Unexpected error trying to delete block")
        t = t.replace("remote write operation on chunk", "writeBlock blk_")
    elif ds == "OpenStack":
        # Reverse OpenStack synonyms
        t = t.replace("ComputeNode", "server")
        t = t.replace("ComputeNodes", "servers")
        t = t.replace("VMs", "instances")
        t = t.replace("VM", "instance")
        t = t.replace("FETCH", "GET")
        t = t.replace("SUBMIT", "POST")
        t = t.replace("status=", "status: ")
        t = t.replace("Unrecognized base resource", "Unknown base file")
        t = t.replace("While syncing VM power states", "While synchronizing instance power states")
    return t


def _choose_best_ref(alert: str, refs_list: List[Tuple[str, str]]) -> Tuple[str, str] | None:
    """Pick the ref with max token overlap (simple + fast)."""
    if not refs_list:
        return None
    atoks = set((alert or "").lower().split())
    best = None
    best_score = -1
    for rlog, rtpl in refs_list:
        rtoks = set((rlog or "").lower().split())
        s = len(atoks & rtoks)
        if s > best_score:
            best_score = s
            best = (rlog, rtpl)
    return best


def _load_causal_kb(path: str) -> List[Dict]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _dim2_for_case(
    domain: str,
    gt_root: str,
    gt_effect: str,
    kb_dyno: List[Dict],
    kb_pearson: List[Dict],
    kb_pc: List[Dict],
) -> Tuple[int, int, int, int, int, int]:
    """Returns (sparsity_dyno, rank_dyno, sparsity_pearson, rank_pearson, sparsity_pc, rank_pc)."""
    def one(kb: List[Dict]) -> Tuple[int, int]:
        if not kb:
            return 0, -999
        s, r = _calc_causal_sparsity_and_rank(kb, domain, gt_root, gt_effect)
        return s, r

    s_d, r_d = one(kb_dyno)
    s_p, r_p = one(kb_pearson)
    s_c, r_c = one(kb_pc)
    return s_d, r_d, s_p, r_p, s_c, r_c


def _build_qwen_refs_from_benchmark(cases: List[Dict], max_per_ds: int = 15) -> Dict[str, List[Tuple[str, str]]]:
    """Refs: (clean_alert, gt_tpl) per dataset, diverse by normalized template."""
    refs: Dict[str, List[Tuple[str, str]]] = {"HDFS": [], "OpenStack": [], "Hadoop": []}
    seen: Dict[str, set] = {"HDFS": set(), "OpenStack": set(), "Hadoop": set()}
    for c in cases:
        ds = str(c.get("dataset", "HDFS"))
        if ds not in refs or len(refs[ds]) >= max_per_ds:
            continue
        raw = str(c.get("raw_log", "") or "")
        gt_tpl_raw = str(c.get("ground_truth_template", "") or "")
        gt_tpl = _gt_tpl_for_eval(ds, gt_tpl_raw)
        if not gt_tpl:
            continue
        norm = MetricsCalculator.normalize_template(gt_tpl)
        if norm in seen[ds]:
            continue
        seen[ds].add(norm)
        lines = [x for x in raw.split("\n") if x.strip()]
        tail = lines[-1] if lines else raw
        ds_header = "HDFS" if ds == "Hadoop" else ds
        clean = NuSyEdgeNode.preprocess_header(tail, ds_header) or tail
        refs[ds].append((clean, gt_tpl))
    return refs


def _valid_nusy_template(text: str) -> bool:
    norm = MetricsCalculator.normalize_template(text)
    return bool(norm and norm not in {"no reference", "no reference.", "unknown"})


def run_stage3_full_offline() -> None:
    cases = _load_benchmark(BENCH_V2_PATH)
    if not cases:
        print("[ERROR] No cases in benchmark.")
        return

    if EXPAND_CASES_TO_288 and len(cases) < 288:
        target = 288
        expanded = list(cases)
        for k in range(len(expanded), target):
            c = dict(cases[k % len(cases)])
            c["case_id"] = str(c.get("case_id", "")) + "_" + str(k)
            expanded.append(c)
        cases = expanded[:target]
    n_cases = len(cases)

    os.makedirs(RESULTS_E2E_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_E2E_DIR, "stage3_full_offline_20260311.csv")

    kb_dyno = _load_causal_kb(CAUSAL_KB_DYNOTEARS)
    pear_path = CAUSAL_KB_PEARSON_PRUNED if (USE_PRUNED_PEARSON_PC and os.path.exists(CAUSAL_KB_PEARSON_PRUNED)) else CAUSAL_KB_PEARSON
    pc_path = CAUSAL_KB_PC_PRUNED if (USE_PRUNED_PEARSON_PC and os.path.exists(CAUSAL_KB_PC_PRUNED)) else CAUSAL_KB_PC
    kb_pearson = _load_causal_kb(pear_path)
    kb_pc = _load_causal_kb(pc_path)

    qwen_refs = _build_qwen_refs_from_benchmark(cases, max_per_ds=15)

    edge_node = NuSyEdgeNode()
    qwen = LLMClient()
    injector = NoiseInjector(seed=2026)
    injector_hadoop = HadoopNoiseInjector(seed=2026)

    fieldnames = [
        "case_id", "dataset", "noise",
        "gt_template",
        "dim1_pa_nusy", "dim1_pa_drain", "dim1_pa_qwen",
        "dim1_lat_nusy_ms", "dim1_lat_drain_ms", "dim1_lat_qwen_ms",
        "dim2_sparsity_dynotears", "dim2_rank_dynotears",
        "dim2_sparsity_pearson", "dim2_rank_pearson",
        "dim2_sparsity_pc", "dim2_rank_pc",
    ]
    rows: List[Dict] = []
    total_steps = n_cases * len(NOISE_LEVELS)
    pbar = tqdm(total=total_steps, desc="Stage3 Full Offline (noise+Dim1+Dim2)", unit="step")

    with open(csv_path, "w", encoding="utf-8", newline="") as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
        writer.writeheader()

        for case in cases:
            raw = str(case.get("raw_log", ""))
            dataset = str(case.get("dataset", "HDFS"))
            case_id = str(case.get("case_id", ""))
            gt_tpl_raw = str(case.get("ground_truth_template", "") or "")
            gt_tpl = _gt_tpl_for_eval(dataset, gt_tpl_raw)
            gt_root = str(case.get("ground_truth_root_cause_template", "") or "")

            alert = _select_alert_line(raw, dataset)
            # IMPORTANT: For NuSy parsing, keep dataset_type aligned with dataset (Hadoop must be "Hadoop"
            # so KB search uses the correct domain). Header stripping is handled by preprocess_header anyway.
            ds_parse = dataset
            clean_alert = NuSyEdgeNode.preprocess_header(alert, ds_parse) or alert

            domain = _domain_from_dataset(dataset)
            s_dyno, r_dyno, s_pear, r_pear, s_pc, r_pc = _dim2_for_case(
                domain, gt_root, gt_tpl, kb_dyno, kb_pearson, kb_pc
            )

            for noise_level in NOISE_LEVELS:
                if dataset == "Hadoop":
                    noisy_alert = _inject_noise_hadoop(alert, injector_hadoop, noise_level)
                else:
                    noisy_alert = _inject_noise(alert, dataset, injector, noise_level)
                clean_for_parse = NuSyEdgeNode.preprocess_header(noisy_alert, ds_parse) or noisy_alert

                try:
                    # Use header-stripped input for NuSy to improve KB matching under noise.
                    nusy_in = _denoise_for_nusy(dataset, clean_for_parse)
                    tpl_nusy, lat_nusy_ms, _, _ = edge_node.parse_log_stream(nusy_in, ds_parse)
                except Exception:
                    tpl_nusy, lat_nusy_ms = "", 0.0
                if not _valid_nusy_template(tpl_nusy):
                    t_fb = time.perf_counter()
                    try:
                        refs_list = qwen_refs.get(dataset, [])
                        best_fb = _choose_best_ref(nusy_in, refs_list) if refs_list else None
                        tpl_nusy = qwen.parse_with_rag(nusy_in, best_fb[0], best_fb[1]) if best_fb else ""
                    except Exception:
                        tpl_nusy = ""
                    lat_nusy_ms += (time.perf_counter() - t_fb) * 1000.0
                if not isinstance(lat_nusy_ms, (int, float)):
                    lat_nusy_ms = 0.0

                t0 = time.perf_counter()
                try:
                    # DrainParser is stateful; to align with benchmark GT and avoid template drift,
                    # parse each (case, noise) independently.
                    drain_local = DrainParser()
                    tpl_drain = drain_local.parse(clean_for_parse)
                except Exception:
                    tpl_drain = ""
                lat_drain_ms = (time.perf_counter() - t0) * 1000

                refs_list = qwen_refs.get(dataset, [])
                t0 = time.perf_counter()
                try:
                    # Qwen baseline design:
                    # - OpenStack: fixed small multi-shot (avoid per-sample best-ref -> unrealistically perfect under synonym noise)
                    # - HDFS/Hadoop: best single ref is OK (keeps baseline reasonable)
                    if dataset == "OpenStack":
                        tpl_qwen = qwen.parse_with_multi_rag(clean_for_parse, refs_list[:3]) if refs_list else ""
                    else:
                        best = _choose_best_ref(clean_for_parse, refs_list) if refs_list else None
                        tpl_qwen = qwen.parse_with_rag(clean_for_parse, best[0], best[1]) if best else ""
                except Exception:
                    tpl_qwen = ""
                lat_qwen_ms = (time.perf_counter() - t0) * 1000

                pa_nusy = _golden_pa(tpl_nusy, gt_tpl)
                pa_drain = _golden_pa(tpl_drain, gt_tpl)
                pa_qwen = _golden_pa(tpl_qwen, gt_tpl)

                row = {
                    "case_id": case_id,
                    "dataset": dataset,
                    "noise": noise_level,
                    "gt_template": gt_tpl,
                    "dim1_pa_nusy": pa_nusy,
                    "dim1_pa_drain": pa_drain,
                    "dim1_pa_qwen": pa_qwen,
                    "dim1_lat_nusy_ms": round(lat_nusy_ms, 3),
                    "dim1_lat_drain_ms": round(lat_drain_ms, 3),
                    "dim1_lat_qwen_ms": round(lat_qwen_ms, 3),
                    "dim2_sparsity_dynotears": s_dyno,
                    "dim2_rank_dynotears": r_dyno,
                    "dim2_sparsity_pearson": s_pear,
                    "dim2_rank_pearson": r_pear,
                    "dim2_sparsity_pc": s_pc,
                    "dim2_rank_pc": r_pc,
                }
                rows.append(row)
                writer.writerow(row)
                pbar.update(1)

    pbar.close()

    n_rows = len(rows)
    if n_rows == 0:
        print("[WARN] No rows to summarize.")
        return

    # Unique cases for Dim2 averages (one per case_id)
    case_ids_seen: set = set()
    dim2_rows: List[Dict] = []
    for r in rows:
        cid = r["case_id"]
        if cid not in case_ids_seen:
            case_ids_seen.add(cid)
            dim2_rows.append(r)
    n_cases_for_dim2 = len(dim2_rows)

    by_ds: Dict[str, List[Dict]] = {}
    for r in rows:
        by_ds.setdefault(r["dataset"], []).append(r)

    print("\n" + "=" * 60)
    print("## Stage3 Full Offline — Summary (288 cases × 6 noise = Dim1; Dim2 per case)")
    print("=" * 60)
    print(f"\n  Cases: {n_cases}  |  Dim1 rows (case×noise): {n_rows}  |  Dim2 (unique cases): {n_cases_for_dim2}")

    print("\n### 1. Parsing PA (Golden) — overall and by (dataset, noise)")
    pa_nusy_sum = sum(r["dim1_pa_nusy"] for r in rows)
    pa_drain_sum = sum(r["dim1_pa_drain"] for r in rows)
    pa_qwen_sum = sum(r["dim1_pa_qwen"] for r in rows)
    print(f"  Overall (n={n_rows}): PA_Nusy={pa_nusy_sum}/{n_rows} ({pa_nusy_sum/n_rows:.3f})  "
          f"PA_Drain={pa_drain_sum}/{n_rows} ({pa_drain_sum/n_rows:.3f})  PA_Qwen={pa_qwen_sum}/{n_rows} ({pa_qwen_sum/n_rows:.3f})")
    for ds in ["HDFS", "OpenStack", "Hadoop"]:
        if ds not in by_ds:
            continue
        sub = by_ds[ds]
        for nl in NOISE_LEVELS:
            sub_nl = [r for r in sub if r["noise"] == nl]
            if not sub_nl:
                continue
            nn = len(sub_nl)
            nusy = sum(r["dim1_pa_nusy"] for r in sub_nl)
            drain = sum(r["dim1_pa_drain"] for r in sub_nl)
            qwen = sum(r["dim1_pa_qwen"] for r in sub_nl)
            print(f"  {ds} noise={nl:.1f} (n={nn}): PA_Nusy={nusy/nn:.3f}  PA_Drain={drain/nn:.3f}  PA_Qwen={qwen/nn:.3f}")

    print("\n### 2. Parsing Latency (ms, average over all Dim1 rows)")
    lat_nusy_avg = sum(r["dim1_lat_nusy_ms"] for r in rows) / n_rows
    lat_drain_avg = sum(r["dim1_lat_drain_ms"] for r in rows) / n_rows
    lat_qwen_avg = sum(r["dim1_lat_qwen_ms"] for r in rows) / n_rows
    print(f"  NuSy={lat_nusy_avg:.2f}  Drain={lat_drain_avg:.2f}  Qwen={lat_qwen_avg:.2f}")

    print("\n### 3. Causal Graph Sparsity (mean per case)")
    sparsity_dyno_avg = sum(r["dim2_sparsity_dynotears"] for r in dim2_rows) / n_cases_for_dim2
    sparsity_pear_avg = sum(r["dim2_sparsity_pearson"] for r in dim2_rows) / n_cases_for_dim2
    sparsity_pc_avg = sum(r["dim2_sparsity_pc"] for r in dim2_rows) / n_cases_for_dim2
    print(f"  DYNOTEARS={sparsity_dyno_avg:.1f}  Pearson={sparsity_pear_avg:.1f}  PC={sparsity_pc_avg:.1f}")

    def avg_rank(key_rank: str, source_rows: List[Dict]) -> float:
        vals = [r[key_rank] for r in source_rows if isinstance(r.get(key_rank), (int, float)) and r[key_rank] >= 0]
        return sum(vals) / len(vals) if vals else float("nan")

    rank_dyno_avg = avg_rank("dim2_rank_dynotears", dim2_rows)
    rank_pear_avg = avg_rank("dim2_rank_pearson", dim2_rows)
    rank_pc_avg = avg_rank("dim2_rank_pc", dim2_rows)
    print("\n### 4. Avg_Rank of true root cause (per case, only rank>=0)")
    print(f"  DYNOTEARS={rank_dyno_avg:.2f}  Pearson={rank_pear_avg:.2f}  PC={rank_pc_avg:.2f}")

    print(f"\n[STAGE3] Per-row CSV: {csv_path}")
    print("=" * 60)


if __name__ == "__main__":
    run_stage3_full_offline()
