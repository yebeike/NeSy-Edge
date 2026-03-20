from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Tuple

from tqdm import tqdm

_SCRIPT_DIR = Path(__file__).resolve().parent
_REBUILD_ROOT = _SCRIPT_DIR.parents[1]
_PROJECT_ROOT = _REBUILD_ROOT.parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.thesis_rebuild_20260315.shared.utils.io_utils import append_jsonl_row, write_json


LEGACY_STAGE4 = _PROJECT_ROOT / "experiments" / "rq123_e2e" / "stage4_noise_api_sampled_20260313.py"
REBUILD_RESULTS_DIR = _REBUILD_ROOT / "rq34" / "results"
RQ2_FULLCASE_MODIFIED_GRAPH = (
    _REBUILD_ROOT / "rq2_fullcase" / "results" / "gt_causal_knowledge_nesydy_fullcase_20260316.json"
)
METHODS = ["agent", "vanilla", "rag"]
DATASETS = ["HDFS", "OpenStack", "Hadoop"]


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases-per-dataset", type=int, default=15)
    ap.add_argument("--noise-levels", type=str, default="0.0,0.2,0.4,0.6,0.8,1.0")
    ap.add_argument("--run-tag", type=str, default="sampled15x6_rebuild")
    ap.add_argument(
        "--causal-graph-path",
        type=str,
        default=str(RQ2_FULLCASE_MODIFIED_GRAPH),
        help="Causal graph JSON used by the agent path.",
    )
    ap.add_argument(
        "--force-resample",
        action="store_true",
        help="Ignore any existing manifest and rebuild the sampled case list.",
    )
    return ap.parse_args()


def _load_legacy_module():
    spec = importlib.util.spec_from_file_location("stage4_rebuild_legacy", LEGACY_STAGE4)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load legacy stage4 script from {LEGACY_STAGE4}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _artifact_paths(run_tag: str) -> Dict[str, Path]:
    return {
        "manifest": REBUILD_RESULTS_DIR / f"rq34_{run_tag}_manifest_20260315.json",
        "progress": REBUILD_RESULTS_DIR / f"rq34_{run_tag}_progress_20260315.jsonl",
        "state": REBUILD_RESULTS_DIR / f"rq34_{run_tag}_state_20260315.json",
        "summary_rows": REBUILD_RESULTS_DIR / f"rq34_{run_tag}_summary_rows_20260315.json",
        "legacy_summary_alias": REBUILD_RESULTS_DIR / "stage4_noise_api_sampled_summary_20260314.json",
    }


def _noise_key(noise: float) -> str:
    return f"{float(noise):.1f}"


def _step_key(dataset: str, case_id: str, noise: float, method: str) -> str:
    return f"{dataset}|{case_id}|{_noise_key(noise)}|{method}"


def _build_stats(noise_levels: Iterable[float]) -> Dict[Tuple[str, float, str], Dict[str, int]]:
    stats: Dict[Tuple[str, float, str], Dict[str, int]] = {}
    for ds in DATASETS:
        for nl in noise_levels:
            for method in METHODS:
                stats[(ds, nl, method)] = {
                    "rca_total": 0,
                    "rca_success": 0,
                    "e2e_total": 0,
                    "e2e_success": 0,
                }
    return stats


def _summary_rows_from_stats(
    stats: Dict[Tuple[str, float, str], Dict[str, int]],
    noise_levels: Iterable[float],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for ds in DATASETS:
        for nl in noise_levels:
            for method in METHODS:
                s = stats[(ds, nl, method)]
                n_rca = s["rca_total"] or 1
                n_e2e = s["e2e_total"] or 1
                rows.append(
                    {
                        "dataset": ds,
                        "noise": float(nl),
                        "method": method,
                        "rca_success": s["rca_success"],
                        "rca_total": s["rca_total"],
                        "rca_accuracy": round(s["rca_success"] / n_rca, 4),
                        "e2e_success": s["e2e_success"],
                        "e2e_total": s["e2e_total"],
                        "e2e_success_rate": round(s["e2e_success"] / n_e2e, 4),
                    }
                )
    return rows


def _write_checkpoint(
    paths: Mapping[str, Path],
    stats: Dict[Tuple[str, float, str], Dict[str, int]],
    noise_levels: Iterable[float],
    *,
    run_tag: str,
    cases_per_dataset: int,
    total_steps: int,
    completed_steps: int,
    actual_api_calls: int,
    agent_local_shortcuts: int,
) -> None:
    summary_rows = _summary_rows_from_stats(stats, noise_levels)
    write_json(paths["summary_rows"], summary_rows)
    write_json(paths["legacy_summary_alias"], summary_rows)
    state = {
        "run_tag": run_tag,
        "cases_per_dataset": cases_per_dataset,
        "noise_levels": [float(x) for x in noise_levels],
        "total_steps": total_steps,
        "completed_steps": completed_steps,
        "actual_api_calls": actual_api_calls,
        "agent_local_shortcuts": agent_local_shortcuts,
        "progress_path": str(paths["progress"]),
        "summary_rows_path": str(paths["summary_rows"]),
    }
    write_json(paths["state"], state)


def _label_distribution(labeled_cases: List[Dict[str, object]], dataset: str) -> Dict[str, int]:
    dist: Dict[str, int] = {}
    for item in labeled_cases:
        case = item["case"]
        if str(case.get("dataset", "")) != dataset:
            continue
        label = str(item["gt_label"])
        dist[label] = dist.get(label, 0) + 1
    return dist


def _load_or_create_manifest(
    legacy,
    *,
    run_tag: str,
    cases_per_dataset: int,
    noise_levels: List[float],
    causal_graph_path: str,
    manifest_path: Path,
    force_resample: bool,
) -> Dict[str, object]:
    if manifest_path.exists() and not force_resample:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("cases_per_dataset") != cases_per_dataset
            or [float(x) for x in manifest.get("noise_levels", [])] != [float(x) for x in noise_levels]
        ):
            raise RuntimeError(
                f"Existing manifest {manifest_path} does not match current args. "
                "Use --force-resample or a new --run-tag."
            )
        return manifest

    sampled_cases = legacy._sample_cases(limit_per_dataset=cases_per_dataset)
    labeled_cases: List[Dict[str, object]] = []
    for case in sampled_cases:
        gt_label = legacy.gt_label_for_case(case)
        if not gt_label:
            continue
        labeled_cases.append({"case": case, "gt_label": gt_label})

    manifest = {
        "run_tag": run_tag,
        "cases_per_dataset": cases_per_dataset,
        "noise_levels": [float(x) for x in noise_levels],
        "methods": list(METHODS),
        "causal_graph_path": causal_graph_path,
        "labeled_cases": labeled_cases,
        "label_distribution": {ds: _label_distribution(labeled_cases, ds) for ds in DATASETS},
    }
    write_json(manifest_path, manifest)
    return manifest


def _load_progress(progress_path: Path, noise_levels: List[float]) -> Tuple[set[str], Dict[Tuple[str, float, str], Dict[str, int]], int, int]:
    completed: set[str] = set()
    stats = _build_stats(noise_levels)
    actual_api_calls = 0
    agent_local_shortcuts = 0

    if not progress_path.exists():
        return completed, stats, actual_api_calls, agent_local_shortcuts

    with progress_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            key = _step_key(
                str(row["dataset"]),
                str(row["case_id"]),
                float(row["noise"]),
                str(row["method"]),
            )
            if key in completed:
                continue
            completed.add(key)
            stat = stats[(str(row["dataset"]), float(row["noise"]), str(row["method"]))]
            stat["rca_total"] += 1
            stat["rca_success"] += int(bool(row.get("rca_success", False)))
            stat["e2e_total"] += 1
            stat["e2e_success"] += int(bool(row.get("e2e_success", False)))
            actual_api_calls += int(bool(row.get("api_call", False)))
            agent_local_shortcuts += int(bool(row.get("agent_local_shortcut", False)))
    return completed, stats, actual_api_calls, agent_local_shortcuts


def _call_method(
    legacy,
    *,
    method: str,
    dataset: str,
    noise: float,
    allowed: List[str],
    label_desc: str,
    simple_label_desc: str,
    noised_context: str,
    clean_for_parse: str,
    tpl_agent: str,
    cand_json: str,
    symbolic_label: str,
    deepseek_key: str,
) -> Tuple[str, bool, bool]:
    pred_label = ""
    api_call = False
    agent_local_shortcut = False

    if method == "agent":
        if symbolic_label:
            return symbolic_label, False, True
        base_tail = (
            f"Dataset: {dataset}\n"
            f"Noise level: {noise}\n"
            f"Log window tail (truncated, noised):\n{noised_context}\n"
            f"Observed template (NuSy): {tpl_agent or clean_for_parse}\n"
        )
        user_msg = (
            "You are NeSy-Agent. Use ONLY the provided context.\n"
            "Task: identify the ROOT_CAUSE_LABEL for this incident.\n"
            f"{label_desc}\n"
            "Return STRICT JSON: {\"root_cause_label\": \"<ONE_LABEL_FROM_LIST>\", \"repair_action\": \"...\"}.\n\n"
            f"{base_tail}"
            f"Causal candidates (JSON): {cand_json}\n"
        )
    else:
        base_tail = (
            f"Dataset: {dataset}\n"
            f"Noise level: {noise}\n"
            f"Log window tail (truncated, noised):\n{noised_context}\n"
            "Observed template: (infer from log above)\n"
        )
        if method == "vanilla":
            user_msg = (
                "You are an ops expert. Analyze the log and identify the root cause label.\n"
                f"{simple_label_desc}\n"
                "Return STRICT JSON: {\"root_cause_label\": \"<ONE_LABEL_FROM_LIST>\", \"repair_action\": \"...\"}.\n\n"
                f"{base_tail}"
            )
        else:
            refs = legacy.rq3_tools.knowledge_retriever(clean_for_parse[:200], dataset, top_k=3)
            user_msg = (
                "You are an ops expert. Use the references and logs to choose the root cause label.\n"
                f"{simple_label_desc}\n"
                "Return STRICT JSON: {\"root_cause_label\": \"<ONE_LABEL_FROM_LIST>\", \"repair_action\": \"...\"}.\n\n"
                f"{base_tail}"
                f"References:\n{refs}\n"
            )

    resp = legacy._call_deepseek_with_retry(
        user_msg, api_key=deepseek_key, model="deepseek-chat", max_tokens=256
    )
    api_call = True
    pred_label = legacy._extract_label_from_json(resp, allowed)
    return pred_label, api_call, agent_local_shortcut


def _run_once(args: argparse.Namespace) -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    REBUILD_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    legacy = _load_legacy_module()
    noise_levels = [float(x.strip()) for x in args.noise_levels.split(",") if x.strip()]
    paths = _artifact_paths(args.run_tag)
    manifest = _load_or_create_manifest(
        legacy,
        run_tag=args.run_tag,
        cases_per_dataset=args.cases_per_dataset,
        noise_levels=noise_levels,
        causal_graph_path=args.causal_graph_path,
        manifest_path=paths["manifest"],
        force_resample=args.force_resample,
    )

    labeled_cases = list(manifest["labeled_cases"])
    print(f"[INFO] Sampled {len(labeled_cases)} cases ({args.cases_per_dataset} per dataset).")
    for ds in DATASETS:
        print(f"[INFO] {ds} sampled label distribution: {manifest['label_distribution'].get(ds, {})}")
    print(f"[INFO] Labeled cases (all labels including OTHER): {len(labeled_cases)}/{len(labeled_cases)}.")
    print(f"[INFO] Agent causal graph: {args.causal_graph_path}")

    completed, stats, actual_api_calls, agent_local_shortcuts = _load_progress(paths["progress"], noise_levels)
    total_steps = len(labeled_cases) * len(noise_levels) * len(METHODS)
    print(f"[INFO] Resume state: {len(completed)}/{total_steps} completed steps.")

    edge_node = legacy.NuSyEdgeNode()
    legacy.LLMClient()  # keep parity with original script init path
    injector = legacy.NoiseInjector(seed=2026)
    injector_hadoop = legacy.HadoopNoiseInjector(seed=2026)
    deepseek_key = legacy._get_deepseek_api_key()

    pbar = tqdm(
        total=total_steps,
        desc="Stage4 noise sampled (Dim1+Dim3+Dim4)",
        unit="step",
        initial=len(completed),
    )

    for item in labeled_cases:
        case = dict(item["case"])
        gt_label = str(item["gt_label"])
        raw = str(case.get("raw_log", "") or "")
        dataset = str(case.get("dataset", "HDFS"))
        case_id = str(case.get("case_id", ""))

        alert = legacy._select_alert_line(raw, dataset)
        ds_parse = dataset
        for noise in noise_levels:
            noisy_alert = legacy._inject_noise(alert, dataset, injector, injector_hadoop, noise)
            clean_for_parse = legacy.NuSyEdgeNode.preprocess_header(noisy_alert, ds_parse) or noisy_alert
            denoised_for_agent = legacy._denoise_for_nusy(dataset, clean_for_parse)
            try:
                tpl_nusy, _, _, _ = edge_node.parse_log_stream(denoised_for_agent, ds_parse)
            except Exception:
                tpl_nusy = ""
            try:
                tpl_drain = legacy._DRAIN.parse(denoised_for_agent)
            except Exception:
                tpl_drain = ""
            tpl_agent = tpl_nusy if legacy._valid_template(tpl_nusy) else tpl_drain
            noised_context = legacy._truncate_and_inject_noise(
                raw, dataset, injector, injector_hadoop, noise, max_chars=600
            )
            allowed = legacy.allowed_labels_for_dataset(dataset)
            label_desc = legacy.describe_allowed_labels(dataset, allowed)
            simple_label_desc = legacy._simple_allowed_labels(allowed)
            domain = "hdfs" if dataset == "HDFS" else ("openstack" if dataset == "OpenStack" else "hadoop")
            causal_path = args.causal_graph_path
            cand_json = legacy.rq3_tools.causal_navigator(
                tpl_agent or denoised_for_agent, domain, causal_path=causal_path
            )
            symbolic_label, _ = legacy._agent_symbolic_vote(
                dataset,
                noised_context,
                clean_for_parse,
                denoised_for_agent,
                tpl_agent,
                cand_json,
            )

            for method in METHODS:
                step = _step_key(dataset, case_id, noise, method)
                if step in completed:
                    continue

                pred_label, api_call, agent_shortcut = _call_method(
                    legacy,
                    method=method,
                    dataset=dataset,
                    noise=noise,
                    allowed=allowed,
                    label_desc=label_desc,
                    simple_label_desc=simple_label_desc,
                    noised_context=noised_context,
                    clean_for_parse=clean_for_parse,
                    tpl_agent=tpl_agent,
                    cand_json=cand_json,
                    symbolic_label=symbolic_label,
                    deepseek_key=deepseek_key,
                )

                rca_success = bool(pred_label and pred_label == gt_label)
                if dataset in ("HDFS", "OpenStack"):
                    gt_action_id = legacy._map_label_to_sop_id(dataset, gt_label)
                    pred_action_id = legacy._map_label_to_sop_id(dataset, pred_label)
                    e2e_success = bool(
                        (gt_action_id and pred_action_id and gt_action_id == pred_action_id)
                        or (pred_label and pred_label == gt_label)
                    )
                else:
                    e2e_success = bool(pred_label and pred_label == gt_label)

                row = {
                    "dataset": dataset,
                    "case_id": case_id,
                    "noise": float(noise),
                    "method": method,
                    "gt_label": gt_label,
                    "pred_label": pred_label,
                    "rca_success": rca_success,
                    "e2e_success": e2e_success,
                    "api_call": api_call,
                    "agent_local_shortcut": agent_shortcut,
                }
                append_jsonl_row(paths["progress"], row)

                completed.add(step)
                stat = stats[(dataset, noise, method)]
                stat["rca_total"] += 1
                stat["e2e_total"] += 1
                stat["rca_success"] += int(rca_success)
                stat["e2e_success"] += int(e2e_success)
                actual_api_calls += int(api_call)
                agent_local_shortcuts += int(agent_shortcut)
                pbar.update(1)

                _write_checkpoint(
                    paths,
                    stats,
                    noise_levels,
                    run_tag=args.run_tag,
                    cases_per_dataset=args.cases_per_dataset,
                    total_steps=total_steps,
                    completed_steps=len(completed),
                    actual_api_calls=actual_api_calls,
                    agent_local_shortcuts=agent_local_shortcuts,
                )

    pbar.close()
    print(f"[INFO] Actual DeepSeek calls: {actual_api_calls} / logical steps {total_steps}.")
    print(f"[INFO] Agent local symbolic shortcuts: {agent_local_shortcuts}.")

    summarize_cmd = [
        sys.executable,
        str(_SCRIPT_DIR / "summarize_rq34_current_20260315.py"),
        "--source",
        str(paths["summary_rows"]),
        "--run-tag",
        args.run_tag,
        "--cases-per-dataset",
        str(args.cases_per_dataset),
        "--noise-levels",
        args.noise_levels,
    ]
    print("[RUN]", " ".join(summarize_cmd))
    subprocess.run(summarize_cmd, cwd=str(_PROJECT_ROOT), check=True)


def main() -> None:
    args = _parse_args()
    _run_once(args)


if __name__ == "__main__":
    main()
