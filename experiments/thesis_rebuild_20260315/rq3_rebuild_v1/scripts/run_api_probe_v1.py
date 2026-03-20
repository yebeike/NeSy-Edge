from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Mapping

from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
REBUILD_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = REBUILD_ROOT.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.rq3.agent import _get_deepseek_api_key
from experiments.thesis_rebuild_20260315.shared.utils.io_utils import append_jsonl_row, write_json
from experiments.thesis_rebuild_20260315.rq3_rebuild_v1.scripts import rebuild_utils_v1 as utils
from experiments.thesis_rebuild_20260315.rq3_rebuild_v1.scripts.run_local_probe_v1 import (
    DEFAULT_BENCHMARK_PATH,
    artifact_paths,
    build_prompt,
    extract_open_prediction,
    format_duration,
    load_progress,
    step_key,
    summarize_rows,
)


DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"
DEFAULT_OUTPUT_DIR = REBUILD_ROOT / "analysis" / "api_probe_v1_20260320"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark-path", type=Path, default=DEFAULT_BENCHMARK_PATH)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--run-tag", type=str, default="rq3_api_probe_v1_20260320")
    ap.add_argument("--datasets", type=str, default="HDFS,OpenStack,Hadoop")
    ap.add_argument("--modes", type=str, default="vanilla_open,rag_open,agent_open")
    ap.add_argument("--noise-levels", type=str, default="")
    ap.add_argument("--model", type=str, default="deepseek-chat")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-output-tokens", type=int, default=220)
    ap.add_argument("--max-api-calls", type=int, default=0)
    ap.add_argument("--max-retries", type=int, default=6)
    return ap.parse_args()


def estimate_cost_cny(prompt_tokens: int, completion_tokens: int) -> float:
    return (prompt_tokens * 2.0 + completion_tokens * 3.0) / 1_000_000.0


def call_deepseek_chat(
    *,
    api_key: str,
    model: str,
    prompt: str,
    temperature: float,
    max_output_tokens: int,
    max_retries: int,
) -> Dict[str, object]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_output_tokens,
        "temperature": temperature,
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    backoff = 2.0
    last_error: Exception | None = None
    for _ in range(max_retries):
        req = urllib.request.Request(
            DEEPSEEK_URL,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                raw = json.loads(resp.read().decode("utf-8"))
            choice = (raw.get("choices") or [None])[0]
            message = choice.get("message", {}) if isinstance(choice, dict) else {}
            usage = raw.get("usage") or {}
            return {
                "raw_response": str(message.get("content", "") or "").strip(),
                "usage": {
                    "prompt_tokens": int(usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0),
                    "completion_tokens": int(usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0),
                    "total_tokens": int(usage.get("total_tokens", 0) or 0),
                },
            }
        except urllib.error.HTTPError as exc:
            body_text = exc.read().decode("utf-8", errors="ignore") if exc.fp else ""
            last_error = RuntimeError(f"DeepSeek HTTP {exc.code}: {body_text[:300]}")
            if exc.code in {429, 500, 502, 503, 504}:
                time.sleep(backoff)
                backoff = min(backoff * 2.0, 60.0)
                continue
            raise last_error
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(backoff)
            backoff = min(backoff * 2.0, 60.0)
    raise RuntimeError("DeepSeek API failed after retries") from last_error


def write_state(
    path: Path,
    *,
    run_tag: str,
    completed_steps: int,
    total_steps: int,
    api_calls: int,
    prompt_tokens: int,
    completion_tokens: int,
    elapsed_seconds: float,
    eta_seconds: float,
    progress_path: Path,
    summary_path: Path,
) -> None:
    steps_per_second = completed_steps / elapsed_seconds if elapsed_seconds > 0 else 0.0
    write_json(
        path,
        {
            "run_tag": run_tag,
            "completed_steps": completed_steps,
            "total_steps": total_steps,
            "api_calls": api_calls,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "estimated_cost_cny": round(estimate_cost_cny(prompt_tokens, completion_tokens), 6),
            "elapsed_seconds": round(elapsed_seconds, 2),
            "estimated_remaining_seconds": round(eta_seconds, 2),
            "steps_per_second": round(steps_per_second, 4),
            "progress_path": str(progress_path),
            "summary_path": str(summary_path),
        },
    )


def main() -> None:
    args = parse_args()
    api_key = _get_deepseek_api_key()
    if not api_key:
        raise RuntimeError("No DeepSeek API key found in env or .env")

    package = utils.load_json(args.benchmark_path)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = artifact_paths(output_dir, args.run_tag)

    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    selected_modes = {x.strip() for x in args.modes.split(",") if x.strip()}
    noise_keys = (
        [f"{float(x.strip()):.1f}" for x in args.noise_levels.split(",") if x.strip()]
        if args.noise_levels.strip()
        else [f"{float(x):.1f}" for x in package["noise_levels"]]
    )
    cases = [case for case in package["cases"] if str(case["dataset"]) in datasets]
    completed = load_progress(paths["progress"])

    rows_so_far: List[Mapping[str, object]] = []
    prompt_tokens = 0
    completion_tokens = 0
    api_calls = 0
    if paths["progress"].exists():
        rows_so_far = [json.loads(line) for line in paths["progress"].read_text(encoding="utf-8").splitlines() if line.strip()]
        for row in rows_so_far:
            usage = dict(row.get("api_usage", {}) or {})
            prompt_tokens += int(usage.get("prompt_tokens", 0) or 0)
            completion_tokens += int(usage.get("completion_tokens", 0) or 0)
            api_calls += int(bool(row.get("api_call", False)))

    total_steps = len(cases) * len(noise_keys) * len(selected_modes)
    write_json(
        paths["manifest"],
        {
            "run_tag": args.run_tag,
            "benchmark_path": str(args.benchmark_path),
            "datasets": datasets,
            "noise_keys": noise_keys,
            "mode_names": sorted(selected_modes),
            "case_ids": [f"{case['dataset']}:{case['case_id']}" for case in cases],
            "model": args.model,
            "temperature": args.temperature,
            "max_output_tokens": args.max_output_tokens,
        },
    )

    run_started_at = time.time()
    pbar = tqdm(total=total_steps, desc="RQ3 rebuild api probe", unit="step", initial=len(completed))
    budget_exhausted = False
    for case in cases:
        dataset = str(case["dataset"])
        case_id = str(case["case_id"])
        gt_family = str(case["gt_family_id"])
        gt_action = str(case["gt_action_id"])
        for noise_key in noise_keys:
            view = dict(case["noise_views"][noise_key])
            for mode_name in sorted(selected_modes):
                step = step_key(dataset, case_id, noise_key, mode_name)
                if step in completed:
                    continue
                if args.max_api_calls > 0 and api_calls >= args.max_api_calls:
                    budget_exhausted = True
                    break
                prompt = build_prompt(mode_name, dataset, view)
                result = call_deepseek_chat(
                    api_key=api_key,
                    model=args.model,
                    prompt=prompt,
                    temperature=args.temperature,
                    max_output_tokens=args.max_output_tokens,
                    max_retries=args.max_retries,
                )
                usage = dict(result["usage"])
                api_calls += 1
                prompt_tokens += int(usage.get("prompt_tokens", 0) or 0)
                completion_tokens += int(usage.get("completion_tokens", 0) or 0)
                prediction = extract_open_prediction(dataset, {}, str(result["raw_response"]))
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
                    "prompt_preview": prompt[:2800],
                    "response_preview": str(result["raw_response"])[:1800],
                    "api_call": True,
                    "api_model": args.model,
                    "api_temperature": args.temperature,
                    "api_usage": usage,
                    "diagnosis": prediction["diagnosis"],
                    "repair_action": prediction["repair_action"],
                    "pred_family_id": prediction["pred_family_id"],
                    "pred_action_id": prediction["pred_action_id"],
                    "selected_alert": str(view.get("selected_alert", ""))[:1200],
                    "context_text": str(view.get("context_text", ""))[:1600],
                    "graph_summary_lines": list((view.get("graph_evidence", {}) or {}).get("summary_lines", [])),
                    "reference_ids": [ref.get("reference_id") for ref in view.get("rag_references", [])],
                }
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
                    f"done={completed_steps}/{total_steps} api={api_calls} cost≈¥{estimate_cost_cny(prompt_tokens, completion_tokens):.3f} eta={format_duration(eta_seconds)}"
                )
                write_state(
                    paths["state"],
                    run_tag=args.run_tag,
                    completed_steps=completed_steps,
                    total_steps=total_steps,
                    api_calls=api_calls,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    elapsed_seconds=elapsed_seconds,
                    eta_seconds=eta_seconds,
                    progress_path=paths["progress"],
                    summary_path=paths["summary"],
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
            "model": args.model,
            "temperature": args.temperature,
            "rows": len(rows),
            "api_calls": api_calls,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "estimated_cost_cny": round(estimate_cost_cny(prompt_tokens, completion_tokens), 6),
            "budget_exhausted": budget_exhausted,
            "progress_path": str(paths["progress"]),
        }
    )
    write_json(paths["summary"], summary)
    write_state(
        paths["state"],
        run_tag=args.run_tag,
        completed_steps=len(completed),
        total_steps=total_steps,
        api_calls=api_calls,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        elapsed_seconds=time.time() - run_started_at,
        eta_seconds=0.0,
        progress_path=paths["progress"],
        summary_path=paths["summary"],
    )


if __name__ == "__main__":
    main()
