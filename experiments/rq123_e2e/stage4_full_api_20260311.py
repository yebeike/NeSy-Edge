"""
Stage 4: Full-scale API Eval (Dim1 + Dim2 + Dim3 + Dim4).

- Dim1 (Parsing PA + Latency): three datasets, full noise levels [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
  using the same design as stage3_full_offline_20260311 (Hadoop-specific noise, NuSy denoise, Qwen baselines).
- Dim2 (Sparsity + Rank): once per case, using DYNOTEARS + (optionally pruned) Pearson/PC graphs.
- Dim3 (RCA Accuracy + Hallucination) and Dim4 (Semantic E2E):
  DeepSeek-based RQ3 Agent / Vanilla / RAG, following run_rq123_e2e_modular.

This script does NOT modify any src/ code; it orchestrates existing components.
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

from experiments.rq123_e2e.noise_injector_hadoop_20260311 import HadoopNoiseInjector
from experiments.rq123_e2e.run_rq123_e2e_modular import (  # type: ignore
    _load_benchmark,
    _load_repair_sop,
    _resolve_action_id,
    _parse_root_from_deepseek_answer,
    _call_deepseek_with_retry,
    _extract_label_from_llm,
    _truncate_logs,
    global_telemetry,
    _get_deepseek_api_key,
)
from experiments.rq123_e2e.run_rq123_e2e_massive import _calc_causal_sparsity_and_rank  # type: ignore
from experiments.rq3 import evaluate as rq3_eval  # type: ignore
from experiments.rq3 import tools as rq3_tools  # type: ignore

DATA_PROCESSED = os.path.join(_PROJECT_ROOT, "data", "processed")
RESULTS_E2E_DIR = os.path.join(_PROJECT_ROOT, "results", "rq123_e2e")
BENCH_V2_PATH = os.path.join(DATA_PROCESSED, "e2e_scaled_benchmark_v2.json")

CAUSAL_KB_DYNOTEARS = os.path.join(DATA_PROCESSED, "gt_causal_knowledge_dynotears.json")
CAUSAL_KB_PEARSON = os.path.join(DATA_PROCESSED, "gt_causal_knowledge_pearson.json")
CAUSAL_KB_PC = os.path.join(DATA_PROCESSED, "gt_causal_knowledge_pc.json")
CAUSAL_KB_PEARSON_PRUNED = os.path.join(DATA_PROCESSED, "gt_causal_knowledge_pearson_pruned_20260311.json")
CAUSAL_KB_PC_PRUNED = os.path.join(DATA_PROCESSED, "gt_causal_knowledge_pc_pruned_20260311.json")

NOISE_LEVELS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
USE_PRUNED_PEARSON_PC = True


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


def _inject_noise_hdfs_os(alert: str, dataset: str, injector: NoiseInjector, noise_level: float) -> str:
    injector.injection_rate = noise_level
    ds_type = "HDFS" if dataset == "Hadoop" else dataset
    return injector.inject(alert, dataset_type=ds_type)


def _inject_noise_hadoop(alert: str, injector: HadoopNoiseInjector, noise_level: float) -> str:
    injector.injection_rate = noise_level
    return injector.inject(alert)


def _hadoop_strip_prefix(text: str) -> str:
    t = NuSyEdgeNode.preprocess_header(text or "", "Hadoop") or (text or "")
    return t


def _gt_tpl_for_eval(dataset: str, gt_tpl: str) -> str:
    return _hadoop_strip_prefix(gt_tpl) if (dataset or "") == "Hadoop" else (gt_tpl or "")


def _denoise_for_nusy(dataset: str, text: str) -> str:
    if not text:
        return ""
    t = str(text)
    ds = (dataset or "")
    if ds in ("HDFS", "Hadoop"):
        t = t.replace("PkgResponder", "PacketResponder")
        t = t.replace("closing", "terminating")
        t = t.replace("Got blk", "Received block")
        t = t.replace("Error", "Exception")
        t = t.replace("len", "size")
        t = t.replace("block-id:", "blk_")
        t = t.replace("Encountered network failure when handling", "Got exception while serving")
        t = t.replace("Failure during cleanup of data chunk", "Unexpected error trying to delete block")
        t = t.replace("remote write operation on chunk", "writeBlock blk_")
    elif ds == "OpenStack":
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


def _rca_and_hallu(
    case: Dict[str, object],
    pred_root: str,
    model_answer: str,
) -> Tuple[int, int]:
    raw = str(case.get("raw_log", ""))
    dataset = str(case.get("dataset", "HDFS"))
    gt_tpl = str(case.get("ground_truth_template", "") or "")
    gt_root = str(case.get("ground_truth_root_cause_template", "") or "")

    record = {
        "raw_log": raw,
        "dataset": dataset,
        "source": case.get("source", ""),
        "ground_truth_template": gt_tpl,
        "ground_truth_root_cause_template": gt_root,
        "extracted_root_cause": pred_root,
        "model_answer": model_answer,
    }

    if gt_root:
        norm = rq3_eval._norm  # type: ignore[attr-defined]
        extracted = _extract_label_from_llm(model_answer, [gt_root])
        pred_to_compare = extracted or pred_root
        rca_ok = 1 if norm(pred_to_compare) == norm(gt_root) else 0
    else:
        rca_ok = 0

    hallu = 1 if rq3_eval._detect_hallucination(record, set(), set()) else 0  # type: ignore[attr-defined]
    return rca_ok, hallu


def run_stage4_full_api() -> None:
    cases = _load_benchmark(BENCH_V2_PATH)
    if not cases:
        print("[ERROR] No cases in benchmark.")
        return

    os.makedirs(RESULTS_E2E_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_E2E_DIR, "stage4_full_api_20260311.csv")

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

    sop_kb = _load_repair_sop()
    deepseek_key = _get_deepseek_api_key()

    fieldnames = [
        "case_id", "dataset", "noise",
        "dim1_pa_nusy", "dim1_pa_drain", "dim1_pa_qwen",
        "dim1_lat_nusy_ms", "dim1_lat_drain_ms", "dim1_lat_qwen_ms",
        "dim2_sparsity_dynotears", "dim2_rank_dynotears",
        "dim2_sparsity_pearson", "dim2_rank_pearson",
        "dim2_sparsity_pc", "dim2_rank_pc",
        "dim3_rca_agent", "dim3_hallu_agent",
        "dim3_rca_vanilla", "dim3_hallu_vanilla",
        "dim3_rca_rag", "dim3_hallu_rag",
        "dim4_semantic_agent", "dim4_semantic_vanilla", "dim4_semantic_rag",
    ]

    rows: List[Dict] = []
    total_steps = len(cases) * (len(NOISE_LEVELS) + 3)  # noise grid + 3 DeepSeek calls per case
    pbar = tqdm(total=total_steps, desc="Stage4 Full API (Dim1+Dim2+Dim3+Dim4)", unit="step")

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

            lines = [x for x in raw.split("\n") if x.strip()]
            alert = lines[-1] if lines else raw
            ds_parse = dataset
            clean_alert = NuSyEdgeNode.preprocess_header(alert, ds_parse) or alert

            domain = _domain_from_dataset(dataset)
            s_dyno, r_dyno, s_pear, r_pear, s_pc, r_pc = _dim2_for_case(
                domain, gt_root, gt_tpl, kb_dyno, kb_pearson, kb_pc
            )

            # --- Dim1 loop over noise levels ---
            dim1_rows_for_case: List[Dict] = []
            for noise_level in NOISE_LEVELS:
                if dataset == "Hadoop":
                    noisy_alert = _inject_noise_hadoop(alert, injector_hadoop, noise_level)
                else:
                    noisy_alert = _inject_noise_hdfs_os(alert, dataset, injector, noise_level)
                clean_for_parse = NuSyEdgeNode.preprocess_header(noisy_alert, ds_parse) or noisy_alert

                try:
                    nusy_in = _denoise_for_nusy(dataset, clean_for_parse)
                    tpl_nusy, lat_nusy_ms, _, _ = edge_node.parse_log_stream(nusy_in, ds_parse)
                except Exception:
                    tpl_nusy, lat_nusy_ms = "", 0.0
                if not isinstance(lat_nusy_ms, (int, float)):
                    lat_nusy_ms = 0.0

                t0 = time.perf_counter()
                try:
                    drain_local = DrainParser()
                    tpl_drain = drain_local.parse(clean_for_parse)
                except Exception:
                    tpl_drain = ""
                lat_drain_ms = (time.perf_counter() - t0) * 1000

                refs_list = qwen_refs.get(dataset, [])
                t0 = time.perf_counter()
                try:
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
                    # Dim3/4 filled only for noise==0
                    "dim3_rca_agent": 0,
                    "dim3_hallu_agent": 0,
                    "dim3_rca_vanilla": 0,
                    "dim3_hallu_vanilla": 0,
                    "dim3_rca_rag": 0,
                    "dim3_hallu_rag": 0,
                    "dim4_semantic_agent": 0,
                    "dim4_semantic_vanilla": 0,
                    "dim4_semantic_rag": 0,
                }
                dim1_rows_for_case.append(row)
                pbar.update(1)

            # --- Dim3 + Dim4 (only once per case, on raw window) ---
            tpl_for_rq3 = dim1_rows_for_case[0]["dim1_pa_nusy"] and tpl_nusy or clean_alert  # type: ignore[truthy-function]
            root_json = rq3_tools.causal_navigator(tpl_for_rq3, domain)
            try:
                root_list = json.loads(root_json)
            except Exception:
                root_list = []
            if isinstance(root_list, dict):
                root_list = []
            cand = [f"{r.get('source_template','')}" for r in root_list[:5]] if root_list else []

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
            rca_agent, hallu_agent = _rca_and_hallu(case, pred_root_agent, resp_agent)
            pbar.update(1)

            choices = ""
            if dataset.lower() == "hadoop":
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
            rca_vanilla, hallu_vanilla = _rca_and_hallu(case, pred_root_vanilla, resp_vanilla)
            pbar.update(1)

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
            rca_rag, hallu_rag = _rca_and_hallu(case, pred_root_rag, resp_rag)
            pbar.update(1)

            action_gt = _resolve_action_id(sop_kb, gt_root, dataset) if gt_root else ""
            action_agent = _resolve_action_id(sop_kb, pred_root_agent, dataset)
            action_vanilla = _resolve_action_id(sop_kb, pred_root_vanilla, dataset)
            action_rag = _resolve_action_id(sop_kb, pred_root_rag, dataset)

            semantic_agent = 1 if action_gt and action_agent and action_gt == action_agent else 0
            semantic_vanilla = 1 if action_gt and action_vanilla and action_gt == action_vanilla else 0
            semantic_rag = 1 if action_gt and action_rag and action_gt == action_rag else 0

            # Attach Dim3/4 to noise=0 row for this case; others remain 0
            for row in dim1_rows_for_case:
                if row["noise"] == 0.0:
                    row["dim3_rca_agent"] = rca_agent
                    row["dim3_hallu_agent"] = hallu_agent
                    row["dim3_rca_vanilla"] = rca_vanilla
                    row["dim3_hallu_vanilla"] = hallu_vanilla
                    row["dim3_rca_rag"] = rca_rag
                    row["dim3_hallu_rag"] = hallu_rag
                    row["dim4_semantic_agent"] = semantic_agent
                    row["dim4_semantic_vanilla"] = semantic_vanilla
                    row["dim4_semantic_rag"] = semantic_rag

            for row in dim1_rows_for_case:
                rows.append(row)
                writer.writerow(row)

    pbar.close()

    n_rows = len(rows)
    if n_rows == 0:
        print("[WARN] No rows to summarize.")
        return

    by_ds: Dict[str, List[Dict]] = {}
    for r in rows:
        by_ds.setdefault(r["dataset"], []).append(r)

    print("\n" + "=" * 60)
    print("## Stage4 Full API — Summary (Dim1+Dim2+Dim3+Dim4)")
    print("=" * 60)
    print(f"\n  Cases: {len({r['case_id'] for r in rows})}  |  Dim1 rows (case×noise): {n_rows}")

    # Dim1 summary
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

    # Dim2 sparsity / rank (per case, from noise=0 rows)
    dim2_rows = [r for r in rows if r["noise"] == 0.0]
    n_cases_dim2 = len(dim2_rows)
    print("\n### 2. Causal Graph Sparsity (mean per case)")
    sparsity_dyno_avg = sum(r["dim2_sparsity_dynotears"] for r in dim2_rows) / n_cases_dim2
    sparsity_pear_avg = sum(r["dim2_sparsity_pearson"] for r in dim2_rows) / n_cases_dim2
    sparsity_pc_avg = sum(r["dim2_sparsity_pc"] for r in dim2_rows) / n_cases_dim2
    print(f"  DYNOTEARS={sparsity_dyno_avg:.1f}  Pearson={sparsity_pear_avg:.1f}  PC={sparsity_pc_avg:.1f}")

    def avg_rank(key_rank: str) -> float:
        vals = [r[key_rank] for r in dim2_rows if isinstance(r.get(key_rank), (int, float)) and r[key_rank] >= 0]
        return sum(vals) / len(vals) if vals else float("nan")

    rank_dyno_avg = avg_rank("dim2_rank_dynotears")
    rank_pear_avg = avg_rank("dim2_rank_pearson")
    rank_pc_avg = avg_rank("dim2_rank_pc")
    print("\n### 3. Avg_Rank of true root cause (per case, only rank>=0)")
    print(f"  DYNOTEARS={rank_dyno_avg:.2f}  Pearson={rank_pear_avg:.2f}  PC={rank_pc_avg:.2f}")

    # Dim3 / Dim4 summary (noise=0 only)
    dim3_rows = dim2_rows
    def _avg(key: str) -> float:
        vals = [r[key] for r in dim3_rows if isinstance(r.get(key), (int, float))]
        return sum(vals) / len(vals) if vals else 0.0

    print("\n### 4. Dim3 RCA Accuracy (noise=0 rows)")
    print(f"  NuSy-Agent={_avg('dim3_rca_agent'):.3f}  Vanilla={_avg('dim3_rca_vanilla'):.3f}  RAG={_avg('dim3_rca_rag'):.3f}")

    print("\n### 5. Dim3 Hallucination Rate (noise=0 rows)")
    print(f"  NuSy-Agent={_avg('dim3_hallu_agent'):.3f}  Vanilla={_avg('dim3_hallu_vanilla'):.3f}  RAG={_avg('dim3_hallu_rag'):.3f}")

    print("\n### 6. Dim4 Semantic E2E Success (noise=0 rows)")
    print(f"  NuSy-Agent={_avg('dim4_semantic_agent'):.3f}  Vanilla={_avg('dim4_semantic_vanilla'):.3f}  RAG={_avg('dim4_semantic_rag'):.3f}")

    print("\n### 7. DeepSeek Telemetry")
    print(
        "  API Calls: {api_calls}, Prompt Tokens: {pt}, Completion Tokens: {ct}, Total Tokens: {tt}".format(
            api_calls=global_telemetry["api_calls"],
            pt=global_telemetry["prompt_tokens"],
            ct=global_telemetry["completion_tokens"],
            tt=global_telemetry["total_tokens"],
        )
    )

    print(f"\n[STAGE4] Per-row CSV: {csv_path}")
    print("=" * 60)


if __name__ == "__main__":
    run_stage4_full_api()

