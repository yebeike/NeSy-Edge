from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
REBUILD_ROOT = SCRIPT_DIR.parents[1]
PROJECT_ROOT = REBUILD_ROOT.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.thesis_rebuild_20260315.rq34.scripts import run_rq34_frozen_small_v2_resumable_20260318 as runner_v2
from experiments.thesis_rebuild_20260315.rq34.scripts.frozen_catalog_v2_20260318 import (
    ACTION_CATALOG,
    RCA_FAMILY_CATALOG,
    action_text_success,
    allowed_action_ids,
    allowed_family_ids,
    family_for_action,
    infer_action_id_from_text,
    infer_family_from_text,
)
from experiments.thesis_rebuild_20260315.shared.utils.io_utils import append_jsonl_row, write_json


DEFAULT_CONFIG_PATH = REBUILD_ROOT / "rq34" / "configs" / "rq3_local_probe_matrix_20260318.json"
DEFAULT_OUTPUT_DIR = REBUILD_ROOT / "rq34" / "analysis" / "rq3_local_probe_20260318"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--run-tag", type=str, default="rq3_local_probe_matrix_qwen35_9b_20260318")
    ap.add_argument("--datasets", type=str, default="HDFS,OpenStack,Hadoop")
    ap.add_argument("--noise-levels", type=str, default="")
    ap.add_argument("--modes", type=str, default="")
    ap.add_argument("--limit-cases-per-dataset", type=int, default=0)
    ap.add_argument("--max-llm-calls", type=int, default=0)
    return ap.parse_args()


def _artifact_paths(output_dir: Path, run_tag: str) -> Dict[str, Path]:
    return {
        "progress": output_dir / f"{run_tag}_progress.jsonl",
        "state": output_dir / f"{run_tag}_state.json",
        "summary": output_dir / f"{run_tag}_summary.json",
        "manifest": output_dir / f"{run_tag}_manifest.json",
    }


def _step_key(dataset: str, case_id: str, noise_key: str, mode_name: str) -> str:
    return f"{dataset}|{case_id}|{noise_key}|{mode_name}"


def _norm(text: str) -> str:
    return " ".join(str(text or "").strip().lower().split())


def _tokenize(text: str) -> List[str]:
    clean = []
    for raw in _norm(text).replace("/", " ").replace(":", " ").replace("-", " ").split():
        token = "".join(ch for ch in raw if ch.isalnum() or ch == "_")
        if len(token) >= 3:
            clean.append(token)
    return clean


def _overlap_score(text: str, meta: Mapping[str, object]) -> float:
    text_tokens = set(_tokenize(text))
    if not text_tokens:
        return 0.0
    desc_tokens = set(_tokenize(str(meta.get("description", ""))))
    overlap = len(text_tokens & desc_tokens)
    group_hits = 0
    for group in meta.get("keyword_groups", []):
        if any(_norm(tok) in _norm(text) for tok in group):
            group_hits += 1
    return overlap + 2.5 * group_hits


def _description_overlap_predict(dataset: str, text: str) -> Tuple[str, str, float]:
    best_action = ""
    best_score = -1.0
    for action_id, meta in ACTION_CATALOG.get(dataset, {}).items():
        score = _overlap_score(text, meta)
        if score > best_score:
            best_score = score
            best_action = action_id
    family_id = family_for_action(dataset, best_action)
    return family_id, best_action, round(max(0.0, best_score), 4)


def _catalog_infer_predict(dataset: str, text: str) -> Tuple[str, str]:
    action_id = infer_action_id_from_text(dataset, text)
    family_id = family_for_action(dataset, action_id)
    if not family_id:
        family_id = infer_family_from_text(dataset, text)
    return family_id, action_id


def _evidence_text(view: Mapping[str, object], evidence: str) -> str:
    selected_alert = str(view.get("selected_alert", "") or "").strip()
    context_text = str(view.get("context_text", "") or "").strip()
    if evidence == "alert_only":
        return selected_alert
    if evidence == "context_only":
        return context_text
    return "\n".join(part for part in (selected_alert, context_text) if part).strip()


def _format_refs(refs: Sequence[Mapping[str, object]]) -> str:
    if not refs:
        return "No historical references retrieved."
    lines: List[str] = []
    for idx, ref in enumerate(refs, start=1):
        lines.append(f"[{idx}] {str(ref.get('text', '')).strip()}")
    return "\n".join(lines)


def _closed_ids_prompt(dataset: str, evidence: str, case: Mapping[str, object], view: Mapping[str, object]) -> str:
    selected_alert = str(view.get("selected_alert", "") or "")
    context_text = str(view.get("context_text", "") or "")
    families = ", ".join(allowed_family_ids(dataset))
    actions = ", ".join(allowed_action_ids(dataset))
    blocks: List[str] = [f"Dataset: {dataset}"]
    if evidence != "context_only":
        blocks.append(f"Selected alert:\n{selected_alert}")
    if evidence != "alert_only":
        blocks.append(f"Incident context:\n{context_text}")
    return (
        "You are diagnosing an incident from logs.\n"
        "Use only the evidence shown below.\n"
        "Do not assume any hidden graph, memory, or remediation hint.\n"
        f"Allowed root-cause family IDs: {families}\n"
        f"Allowed remediation action IDs: {actions}\n"
        "Return only one raw JSON object with keys "
        "`root_cause_family`, `action_id`, and `repair_action`.\n"
        "Choose exactly one family ID and one action ID from the allowed lists.\n"
        "Make `repair_action` a short evidence-grounded sentence.\n\n"
        + "\n\n".join(blocks)
    )


def _open_text_prompt(dataset: str, evidence: str, view: Mapping[str, object]) -> str:
    selected_alert = str(view.get("selected_alert", "") or "")
    context_text = str(view.get("context_text", "") or "")
    blocks: List[str] = [f"Dataset: {dataset}"]
    if evidence != "context_only":
        blocks.append(f"Selected alert:\n{selected_alert}")
    if evidence != "alert_only":
        blocks.append(f"Incident context:\n{context_text}")
    return (
        "You are red-teaming a benchmark for log-based incident diagnosis.\n"
        "Use only the provided logs.\n"
        "Do not invent hidden symptoms, graph hints, prior incidents, or benchmark labels.\n"
        "Return only one raw JSON object with keys `diagnosis`, `repair_action`, and `evidence_phrases`.\n"
        "`diagnosis` should be a concise plain-language root-cause statement.\n"
        "`repair_action` should be a concise plain-language remediation.\n"
        "`evidence_phrases` should be a short list of exact phrases copied from the logs.\n\n"
        + "\n\n".join(blocks)
    )


def _call_ollama(
    *,
    url: str,
    model: str,
    prompt: str,
    temperature: float,
    num_predict: int,
) -> Dict[str, object]:
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


def _extract_open_prediction(dataset: str, parsed: Mapping[str, object], raw_response: str) -> Tuple[str, str, str]:
    diagnosis = str(parsed.get("diagnosis", "") or "")
    repair_action = str(parsed.get("repair_action", "") or "")
    combined = "\n".join(part for part in (diagnosis, repair_action, raw_response) if part).strip()
    pred_action = infer_action_id_from_text(dataset, combined)
    pred_family = family_for_action(dataset, pred_action)
    if not pred_family:
        pred_family = infer_family_from_text(dataset, combined)
    return pred_family, pred_action, repair_action or diagnosis or raw_response


def _load_progress(progress_path: Path) -> set[str]:
    completed: set[str] = set()
    if not progress_path.exists():
        return completed
    for line in progress_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        completed.add(_step_key(str(row["dataset"]), str(row["case_id"]), f"{float(row['noise']):.1f}", str(row["mode"])))
    return completed


def _write_state(
    paths: Mapping[str, Path],
    *,
    run_tag: str,
    completed_steps: int,
    total_steps: int,
    llm_calls: int,
) -> None:
    write_json(
        paths["state"],
        {
            "run_tag": run_tag,
            "completed_steps": completed_steps,
            "total_steps": total_steps,
            "llm_calls": llm_calls,
            "progress_path": str(paths["progress"]),
            "summary_path": str(paths["summary"]),
        },
    )


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    package = json.loads(Path(config["benchmark_path"]).read_text(encoding="utf-8"))

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = _artifact_paths(output_dir, args.run_tag)

    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    noise_keys = (
        [f"{float(x.strip()):.1f}" for x in args.noise_levels.split(",") if x.strip()]
        if args.noise_levels.strip()
        else [f"{float(x):.1f}" for x in package["noise_levels"]]
    )
    selected_mode_names = {x.strip() for x in args.modes.split(",") if x.strip()}
    mode_specs = [
        mode
        for mode in config["modes"]
        if not selected_mode_names or str(mode["name"]) in selected_mode_names
    ]

    cases: List[Mapping[str, object]] = []
    per_dataset_seen: Dict[str, int] = defaultdict(int)
    for case in package["cases"]:
        dataset = str(case["dataset"])
        if dataset not in datasets:
            continue
        if args.limit_cases_per_dataset > 0 and per_dataset_seen[dataset] >= args.limit_cases_per_dataset:
            continue
        per_dataset_seen[dataset] += 1
        cases.append(case)

    total_steps = len(cases) * len(noise_keys) * len(mode_specs)
    completed = _load_progress(paths["progress"])
    llm_calls = 0
    if paths["state"].exists():
        try:
            llm_calls = int(json.loads(paths["state"].read_text(encoding="utf-8")).get("llm_calls", 0))
        except Exception:
            llm_calls = 0

    write_json(
        paths["manifest"],
        {
            "run_tag": args.run_tag,
            "config_path": str(args.config),
            "benchmark_path": str(config["benchmark_path"]),
            "datasets": datasets,
            "noise_keys": noise_keys,
            "mode_names": [str(mode["name"]) for mode in mode_specs],
            "case_count": len(cases),
            "model": config["model"],
        },
    )

    pbar = tqdm(total=total_steps, desc="RQ3 local probe", unit="step", initial=len(completed))
    budget_exhausted = False

    for case in cases:
        dataset = str(case["dataset"])
        case_id = str(case["case_id"])
        gt_family = str(case["gt_family_id"])
        gt_action = str(case["gt_action_id"])
        for noise_key in noise_keys:
            view = dict(case["noise_views"][noise_key])
            for mode in mode_specs:
                mode_name = str(mode["name"])
                step = _step_key(dataset, case_id, noise_key, mode_name)
                if step in completed:
                    continue

                row: Dict[str, object] = {
                    "dataset": dataset,
                    "case_id": case_id,
                    "noise": float(noise_key),
                    "mode": mode_name,
                    "kind": str(mode["kind"]),
                    "gt_family_id": gt_family,
                    "gt_action_id": gt_action,
                }

                if str(mode["kind"]) == "heuristic":
                    evidence = _evidence_text(view, str(mode["evidence"]))
                    mapping = str(mode["mapping"])
                    if mapping == "catalog_infer":
                        pred_family, pred_action = _catalog_infer_predict(dataset, evidence)
                        extra_score = None
                    elif mapping == "description_overlap":
                        pred_family, pred_action, extra_score = _description_overlap_predict(dataset, evidence)
                    else:
                        raise ValueError(f"Unsupported heuristic mapping: {mapping}")
                    repair_action = ""
                    raw_response = ""
                    parsed = {}
                    row["probe_score"] = extra_score
                    row["evidence_preview"] = evidence[:800]
                else:
                    if args.max_llm_calls > 0 and llm_calls >= args.max_llm_calls:
                        budget_exhausted = True
                        break
                    prompt_style = str(mode["prompt_style"])
                    evidence_mode = str(mode["evidence"])
                    if prompt_style == "runner_v2":
                        method = str(mode["method"])
                        prompt = runner_v2._build_prompt(method, case, view)
                    elif prompt_style == "closed_ids":
                        prompt = _closed_ids_prompt(dataset, evidence_mode, case, view)
                    elif prompt_style == "open_text":
                        prompt = _open_text_prompt(dataset, evidence_mode, view)
                    else:
                        raise ValueError(f"Unsupported prompt style: {prompt_style}")
                    result = _call_ollama(
                        url=str(config["ollama_url"]),
                        model=str(config["model"]),
                        prompt=prompt,
                        temperature=float(config.get("temperature", 0)),
                        num_predict=int(config.get("num_predict", 180)),
                    )
                    llm_calls += 1
                    raw_response = str(result["raw_response"])
                    parsed = dict(result["parsed"])
                    if prompt_style in ("runner_v2", "closed_ids"):
                        pred_family, pred_action, repair_action = runner_v2._extract_structured_output(
                            dataset,
                            raw_response,
                            allowed_family_ids(dataset),
                            allowed_action_ids(dataset),
                        )
                    else:
                        pred_family, pred_action, repair_action = _extract_open_prediction(
                            dataset,
                            parsed,
                            raw_response,
                        )
                    row["prompt_preview"] = prompt[:2400]
                    row["response_preview"] = raw_response[:1600]
                    row["ollama_meta"] = result["meta"]

                row["pred_family_id"] = pred_family
                row["pred_action_id"] = pred_action
                row["repair_action"] = repair_action
                row["rca_success"] = bool(pred_family and pred_family == gt_family)
                row["action_success"] = bool(pred_action and pred_action == gt_action)
                row["action_text_success"] = bool(action_text_success(dataset, gt_action, repair_action))
                row["selected_alert"] = str(view.get("selected_alert", ""))[:1200]
                row["context_text"] = str(view.get("context_text", ""))[:1600]
                row["observed_template"] = str(view.get("observed_template", ""))
                row["graph_summary"] = str(view.get("graph_summary", ""))
                row["agent_reference_ids"] = [ref.get("reference_id") for ref in view.get("agent_references", [])]
                row["rag_reference_ids"] = [ref.get("reference_id") for ref in view.get("rag_references", [])]

                append_jsonl_row(paths["progress"], row)
                completed.add(step)
                pbar.update(1)
                _write_state(
                    paths,
                    run_tag=args.run_tag,
                    completed_steps=len(completed),
                    total_steps=total_steps,
                    llm_calls=llm_calls,
                )
            if budget_exhausted:
                break
        if budget_exhausted:
            break

    pbar.close()
    rows = []
    if paths["progress"].exists():
        rows = [json.loads(line) for line in paths["progress"].read_text(encoding="utf-8").splitlines() if line.strip()]
    by_mode: Dict[str, Dict[str, float]] = {}
    for mode_name in sorted({str(row["mode"]) for row in rows}):
        part = [row for row in rows if str(row["mode"]) == mode_name]
        total = len(part)
        by_mode[mode_name] = {
            "rows": total,
            "family_accuracy": round(sum(int(bool(row["rca_success"])) for row in part) / max(1, total), 4),
            "action_accuracy": round(sum(int(bool(row["action_success"])) for row in part) / max(1, total), 4),
            "action_text_success_rate": round(sum(int(bool(row["action_text_success"])) for row in part) / max(1, total), 4),
        }
    write_json(
        paths["summary"],
        {
            "run_tag": args.run_tag,
            "model": config["model"],
            "rows": len(rows),
            "llm_calls": llm_calls,
            "budget_exhausted": budget_exhausted,
            "by_mode": by_mode,
            "progress_path": str(paths["progress"]),
        },
    )
    _write_state(
        paths,
        run_tag=args.run_tag,
        completed_steps=len(completed),
        total_steps=total_steps,
        llm_calls=llm_calls,
    )


if __name__ == "__main__":
    main()
