from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Mapping

from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
REBUILD_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = REBUILD_ROOT.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.thesis_rebuild_20260315.rq34.scripts.frozen_catalog_v2_20260318 import (
    family_for_action,
    infer_action_id_from_text,
    infer_family_from_text,
)
from experiments.thesis_rebuild_20260315.shared.utils.io_utils import append_jsonl_row, write_json
from experiments.thesis_rebuild_20260315.rq3_rebuild_v1.scripts import rebuild_utils_v1 as utils


DEFAULT_CONFIG_PATH = REBUILD_ROOT / "configs" / "local_probe_v1_20260318.json"
DEFAULT_BENCHMARK_PATH = (
    REBUILD_ROOT
    / "analysis"
    / "rq3_triad_proof_slice_v1_20260318"
    / "rq3_triad_proof_slice_v1_20260318_package.json"
)
DEFAULT_OUTPUT_DIR = REBUILD_ROOT / "analysis" / "local_probe_v1_20260318"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    ap.add_argument("--benchmark-path", type=Path, default=DEFAULT_BENCHMARK_PATH)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--run-tag", type=str, default="rq3_triad_proof_slice_local_v1_20260318")
    ap.add_argument("--datasets", type=str, default="HDFS,OpenStack,Hadoop")
    ap.add_argument("--modes", type=str, default="")
    ap.add_argument("--noise-levels", type=str, default="")
    ap.add_argument("--max-llm-calls", type=int, default=0)
    return ap.parse_args()


def artifact_paths(output_dir: Path, run_tag: str) -> Dict[str, Path]:
    return {
        "progress": output_dir / f"{run_tag}_progress.jsonl",
        "state": output_dir / f"{run_tag}_state.json",
        "summary": output_dir / f"{run_tag}_summary.json",
        "manifest": output_dir / f"{run_tag}_manifest.json",
    }


def step_key(dataset: str, case_id: str, noise_key: str, mode_name: str) -> str:
    return f"{dataset}|{case_id}|{noise_key}|{mode_name}"


def call_ollama(*, url: str, model: str, prompt: str, temperature: float, num_predict: int) -> Dict[str, object]:
    payload = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "think": False,
            "keep_alive": "30m",
            "options": {
                "temperature": temperature,
                "num_predict": num_predict,
            },
        }
    ).encode("utf-8")
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        raw_obj = json.loads(resp.read().decode("utf-8"))
    response_text = str(raw_obj.get("response", "") or "").strip()
    parsed = {}
    if response_text:
        try:
            parsed = json.loads(response_text)
        except json.JSONDecodeError:
            start = response_text.find("{")
            end = response_text.rfind("}")
            if start >= 0 and end > start:
                parsed = json.loads(response_text[start : end + 1])
    return {
        "raw_response": response_text,
        "parsed": parsed,
        "meta": {
            "total_duration": raw_obj.get("total_duration"),
            "eval_count": raw_obj.get("eval_count"),
            "prompt_eval_count": raw_obj.get("prompt_eval_count"),
        },
    }


def salvage_json_fields(response_text: str) -> Dict[str, object]:
    def _extract(key: str) -> str:
        pattern = rf'"{key}"\s*:\s*"((?:\\.|[^"\\])*)"'
        match = re.search(pattern, response_text, flags=re.DOTALL)
        if not match:
            return ""
        try:
            return json.loads(f'"{match.group(1)}"')
        except json.JSONDecodeError:
            return match.group(1)

    diagnosis = _extract("diagnosis")
    repair_action = _extract("repair_action")
    evidence_phrases = []
    list_match = re.search(r'"evidence_phrases"\s*:\s*\[(.*?)\]', response_text, flags=re.DOTALL)
    if list_match:
        for quoted in re.findall(r'"((?:\\.|[^"\\])*)"', list_match.group(1), flags=re.DOTALL):
            try:
                evidence_phrases.append(json.loads(f'"{quoted}"'))
            except json.JSONDecodeError:
                evidence_phrases.append(quoted)
    if not diagnosis and not repair_action and not evidence_phrases:
        return {}
    return {
        "diagnosis": diagnosis,
        "repair_action": repair_action,
        "evidence_phrases": evidence_phrases,
    }


def build_prompt(mode_name: str, dataset: str, view: Mapping[str, object]) -> str:
    selected_alert = str(view.get("selected_alert", "") or "")
    context_text = str(view.get("context_text", "") or "")
    base = [
        "You are diagnosing a noisy production incident from logs.",
        "Use only the evidence shown below.",
        "Do not invent benchmark labels, hidden graph state, or symptoms that are not present.",
        "Return only one raw JSON object with keys `diagnosis`, `repair_action`, and `evidence_phrases`.",
        "`diagnosis` should be a concise plain-language root-cause statement.",
        "`repair_action` should be a concise plain-language remediation.",
        "`evidence_phrases` should be a short list of exact phrases copied from the evidence.",
        "",
        f"Dataset: {dataset}",
    ]
    if mode_name == "open_alert_only":
        base.append(f"Selected alert:\n{selected_alert}")
        return "\n".join(base)

    base.extend(
        [
            f"Selected alert:\n{selected_alert}",
            "",
            f"Incident context:\n{context_text}",
        ]
    )
    if mode_name in {"rag_open", "agent_open"}:
        refs = view.get("rag_references", [])
        if refs:
            base.append("")
            base.append("Historical incident snippets:")
            for idx, ref in enumerate(refs, start=1):
                base.append(f"[{idx}] {str(ref.get('text', '')).strip()}")
    if mode_name == "agent_open":
        graph_evidence = view.get("graph_evidence", {}) or {}
        summary_lines = list(graph_evidence.get("summary_lines", []))
        if summary_lines:
            base.append("")
            base.append("Causal graph cues:")
            base.extend(f"- {line}" for line in summary_lines)
            base.append(
                "Use these only as structural constraints. If a graph cue conflicts with the visible logs or the historical snippets, ignore it and follow the evidence."
            )
            base.append("Use graph cues mainly to disambiguate between otherwise plausible root causes or repairs.")
    return "\n".join(base)


def extract_open_prediction(dataset: str, parsed: Mapping[str, object], raw_response: str) -> Dict[str, str]:
    if not parsed:
        parsed = salvage_json_fields(raw_response)
    diagnosis = str(parsed.get("diagnosis", "") or "")
    repair_action = str(parsed.get("repair_action", "") or "")
    pred = utils.infer_from_open_text(dataset, diagnosis, repair_action, raw_response)
    return {
        "diagnosis": diagnosis,
        "repair_action": repair_action or diagnosis or raw_response,
        "pred_family_id": pred["pred_family_id"],
        "pred_action_id": pred["pred_action_id"],
    }


def load_progress(progress_path: Path) -> set[str]:
    completed: set[str] = set()
    if not progress_path.exists():
        return completed
    for line in progress_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        completed.add(step_key(str(row["dataset"]), str(row["case_id"]), f"{float(row['noise']):.1f}", str(row["mode"])))
    return completed


def summarize_rows(rows: List[Mapping[str, object]]) -> Dict[str, object]:
    def _metric(part: List[Mapping[str, object]]) -> Dict[str, float]:
        total = len(part)
        return {
            "rows": total,
            "family_accuracy": round(sum(int(bool(row["rca_success"])) for row in part) / max(1, total), 4),
            "action_accuracy": round(sum(int(bool(row["action_success"])) for row in part) / max(1, total), 4),
            "action_text_success_rate": round(sum(int(bool(row["action_text_success"])) for row in part) / max(1, total), 4),
            "e2e_accuracy": round(sum(int(bool(row["e2e_success"])) for row in part) / max(1, total), 4),
            "rca_e2e_gap_rows": int(sum(int(bool(row["rca_success"]) and not bool(row["e2e_success"])) for row in part)),
        }

    by_mode: Dict[str, Dict[str, float]] = {}
    by_mode_dataset_noise: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = defaultdict(lambda: defaultdict(dict))
    for mode in sorted({str(row["mode"]) for row in rows}):
        part = [row for row in rows if str(row["mode"]) == mode]
        by_mode[mode] = _metric(part)
        for dataset in sorted({str(row["dataset"]) for row in part}):
            ds_part = [row for row in part if str(row["dataset"]) == dataset]
            for noise_key in sorted({f"{float(row['noise']):.1f}" for row in ds_part}):
                noise_part = [row for row in ds_part if f"{float(row['noise']):.1f}" == noise_key]
                by_mode_dataset_noise[mode][dataset][noise_key] = _metric(noise_part)
    return {
        "by_mode": by_mode,
        "by_mode_dataset_noise": by_mode_dataset_noise,
    }


def format_duration(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes > 0:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def main() -> None:
    args = parse_args()
    config = utils.load_json(args.config)
    package = utils.load_json(args.benchmark_path)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = artifact_paths(output_dir, args.run_tag)

    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    selected_modes = {x.strip() for x in args.modes.split(",") if x.strip()}
    modes = [mode for mode in config["modes"] if not selected_modes or str(mode["name"]) in selected_modes]
    noise_keys = (
        [f"{float(x.strip()):.1f}" for x in args.noise_levels.split(",") if x.strip()]
        if args.noise_levels.strip()
        else [f"{float(x):.1f}" for x in package["noise_levels"]]
    )
    cases = [case for case in package["cases"] if str(case["dataset"]) in datasets]
    completed = load_progress(paths["progress"])
    llm_calls = 0
    if paths["state"].exists():
        try:
            llm_calls = int(json.loads(paths["state"].read_text(encoding="utf-8")).get("llm_calls", 0))
        except Exception:
            llm_calls = 0
    total_steps = len(cases) * len(noise_keys) * len(modes)
    write_json(
        paths["manifest"],
        {
            "run_tag": args.run_tag,
            "benchmark_path": str(args.benchmark_path),
            "config_path": str(args.config),
            "datasets": datasets,
            "noise_keys": noise_keys,
            "mode_names": [str(mode["name"]) for mode in modes],
            "case_ids": [f"{case['dataset']}:{case['case_id']}" for case in cases],
        },
    )

    run_started_at = time.time()
    pbar = tqdm(total=total_steps, desc="RQ3 rebuild local probe", unit="step", initial=len(completed))
    budget_exhausted = False
    for case in cases:
        dataset = str(case["dataset"])
        case_id = str(case["case_id"])
        gt_family = str(case["gt_family_id"])
        gt_action = str(case["gt_action_id"])
        for noise_key in noise_keys:
            view = dict(case["noise_views"][noise_key])
            for mode in modes:
                mode_name = str(mode["name"])
                step = step_key(dataset, case_id, noise_key, mode_name)
                if step in completed:
                    continue
                row: Dict[str, object] = {
                    "dataset": dataset,
                    "case_id": case_id,
                    "noise": float(noise_key),
                    "mode": mode_name,
                    "gt_family_id": gt_family,
                    "gt_action_id": gt_action,
                    "quality_tier": str(case.get("quality_tier", "") or ""),
                    "selection_bucket": str(case.get("selection_bucket", "") or ""),
                    "toxicity_flags": list(case.get("toxicity_flags", []) or []),
                    "base_incident_id": str(case.get("base_incident_id", "") or case.get("pool_case_id", "") or case_id),
                    "reanchor_group_id": str(case.get("reanchor_group_id", "") or ""),
                    "is_duplicate_reanchor": bool(case.get("is_duplicate_reanchor", False)),
                    "candidate_origin": str(case.get("candidate_origin", "") or ""),
                    "scored_by_probe": bool(case.get("scored_by_probe", False)),
                    "inclusion_reason": str(case.get("inclusion_reason", "") or ""),
                }
                if str(mode["kind"]) == "heuristic":
                    evidence = (
                        str(view.get("selected_alert", "") or "")
                        if str(mode["evidence"]) == "alert_only"
                        else "\n".join(
                            part
                            for part in (
                                str(view.get("selected_alert", "") or ""),
                                str(view.get("context_text", "") or ""),
                            )
                            if part
                        )
                    )
                    pred_action = infer_action_id_from_text(dataset, evidence)
                    pred_family = family_for_action(dataset, pred_action)
                    if not pred_family:
                        pred_family = infer_family_from_text(dataset, evidence)
                    repair_action = ""
                    row["evidence_preview"] = evidence[:900]
                    row["pred_family_id"] = pred_family
                    row["pred_action_id"] = pred_action
                    row["repair_action"] = repair_action
                else:
                    if args.max_llm_calls > 0 and llm_calls >= args.max_llm_calls:
                        budget_exhausted = True
                        break
                    prompt = build_prompt(mode_name, dataset, view)
                    result = call_ollama(
                        url=str(config["ollama_url"]),
                        model=str(config["model"]),
                        prompt=prompt,
                        temperature=float(config.get("temperature", 0)),
                        num_predict=int(config.get("num_predict", 220)),
                    )
                    llm_calls += 1
                    prediction = extract_open_prediction(dataset, dict(result["parsed"]), str(result["raw_response"]))
                    row["prompt_preview"] = prompt[:2800]
                    row["response_preview"] = str(result["raw_response"])[:1800]
                    row["ollama_meta"] = result["meta"]
                    row["diagnosis"] = prediction["diagnosis"]
                    row["repair_action"] = prediction["repair_action"]
                    row["pred_family_id"] = prediction["pred_family_id"]
                    row["pred_action_id"] = prediction["pred_action_id"]

                row["selected_alert"] = str(view.get("selected_alert", ""))[:1200]
                row["context_text"] = str(view.get("context_text", ""))[:1600]
                row["graph_summary_lines"] = list((view.get("graph_evidence", {}) or {}).get("summary_lines", []))
                row["reference_ids"] = [ref.get("reference_id") for ref in view.get("rag_references", [])]
                row["rca_success"] = bool(row["pred_family_id"] and row["pred_family_id"] == gt_family)
                row["action_success"] = bool(row["pred_action_id"] and row["pred_action_id"] == gt_action)
                row["action_text_success"] = bool(utils.action_text_ok(dataset, gt_action, str(row.get("repair_action", "") or "")))
                row["e2e_success"] = bool(row["rca_success"] and (row["action_success"] or row["action_text_success"]))
                append_jsonl_row(paths["progress"], row)
                completed.add(step)
                pbar.update(1)
                elapsed_seconds = time.time() - run_started_at
                completed_steps = len(completed)
                remaining_steps = max(0, total_steps - completed_steps)
                steps_per_second = completed_steps / elapsed_seconds if elapsed_seconds > 0 else 0.0
                eta_seconds = remaining_steps / steps_per_second if steps_per_second > 0 else 0.0
                pbar.set_postfix_str(
                    f"done={completed_steps}/{total_steps} llm={llm_calls} eta={format_duration(eta_seconds)}"
                )
                write_json(
                    paths["state"],
                    {
                        "run_tag": args.run_tag,
                        "completed_steps": completed_steps,
                        "total_steps": total_steps,
                        "llm_calls": llm_calls,
                        "elapsed_seconds": round(elapsed_seconds, 2),
                        "estimated_remaining_seconds": round(eta_seconds, 2),
                        "steps_per_second": round(steps_per_second, 4),
                        "progress_path": str(paths["progress"]),
                        "summary_path": str(paths["summary"]),
                    },
                )
            if budget_exhausted:
                break
        if budget_exhausted:
            break
    pbar.close()

    rows = []
    if paths["progress"].exists():
        rows = [json.loads(line) for line in paths["progress"].read_text(encoding="utf-8").splitlines() if line.strip()]
    summary = summarize_rows(rows)
    summary.update(
        {
            "run_tag": args.run_tag,
            "model": config["model"],
            "rows": len(rows),
            "llm_calls": llm_calls,
            "budget_exhausted": budget_exhausted,
            "progress_path": str(paths["progress"]),
        }
    )
    write_json(paths["summary"], summary)
    write_json(
        paths["state"],
        {
            "run_tag": args.run_tag,
            "completed_steps": len(completed),
            "total_steps": total_steps,
            "llm_calls": llm_calls,
            "elapsed_seconds": round(time.time() - run_started_at, 2),
            "estimated_remaining_seconds": 0.0,
            "steps_per_second": round((len(completed) / max(1e-9, time.time() - run_started_at)), 4),
            "progress_path": str(paths["progress"]),
            "summary_path": str(paths["summary"]),
        },
    )


if __name__ == "__main__":
    main()
