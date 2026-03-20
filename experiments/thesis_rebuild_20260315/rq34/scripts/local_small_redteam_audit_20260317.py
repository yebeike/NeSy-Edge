from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path
from typing import Dict, Iterable, List


DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
FAMILY_OPTIONS = {
    "HDFS": [
        "HDFS_TRANSFER_LINK_FAILURE",
        "HDFS_PIPELINE_FAILURE",
        "HDFS_STORAGE_METADATA_PRESSURE",
    ],
    "OpenStack": [
        "OPENSTACK_REPAIR_BASE_IMAGE_CHAIN",
        "OPENSTACK_REBUILD_ON_COMPATIBLE_HOST",
        "OPENSTACK_RESYNC_POWER_STATE_AND_REBUILD",
        "OPENSTACK_RESYNC_INSTANCE_INVENTORY",
        "OPENSTACK_SCALE_METADATA_SERVICE",
    ],
    "Hadoop": [
        "HADOOP_RESTORE_NETWORK_AND_RETRY",
        "HADOOP_ISOLATE_NODE_AND_RESCHEDULE",
        "HADOOP_FREE_DISK_AND_RETRY",
    ],
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--progress", required=True, help="Path to rq34 progress.jsonl")
    ap.add_argument("--dataset", required=True, choices=sorted(FAMILY_OPTIONS))
    ap.add_argument("--noise", type=float, default=1.0)
    ap.add_argument("--method", default="vanilla")
    ap.add_argument("--model", default="qwen3.5:9b")
    ap.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    ap.add_argument("--limit", type=int, default=0, help="Optional cap on cases audited")
    ap.add_argument("--out", required=True, help="Output jsonl path")
    return ap.parse_args()


def _iter_rows(path: Path) -> Iterable[Dict[str, object]]:
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        yield json.loads(line)


def _load_completed(out_path: Path) -> set[str]:
    if not out_path.exists():
        return set()
    completed: set[str] = set()
    for row in _iter_rows(out_path):
        completed.add(str(row.get("case_id", "")))
    return completed


def _build_prompt(dataset: str, family_options: List[str], row: Dict[str, object]) -> str:
    families = ", ".join(family_options)
    selected_alert = str(row.get("noisy_selected_alert", "") or "")
    context_text = str(row.get("context_text", "") or "")
    return (
        f"You are auditing whether an {dataset} benchmark case is too easy at family level.\n"
        f"Choose exactly one family from [{families}].\n"
        "Return ONLY a JSON object with keys family, confidence, shortcut_signals, brief_reason.\n"
        "confidence must be a number between 0 and 1.\n"
        "shortcut_signals must be a short list of literal phrases copied from the logs when present.\n"
        "Use only the selected alert and the provided log window.\n\n"
        f"Selected alert:\n{selected_alert}\n\n"
        f"Log window:\n{context_text}\n"
    )


def _call_ollama(url: str, model: str, prompt: str) -> Dict[str, object]:
    payload = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "think": False,
            "options": {
                "temperature": 0,
                "num_predict": 160,
            },
        }
    ).encode("utf-8")
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as resp:
        obj = json.loads(resp.read().decode("utf-8"))
    response_text = str(obj.get("response", "") or "").strip()
    parsed = json.loads(response_text) if response_text else {}
    parsed["_ollama_meta"] = {
        "total_duration": obj.get("total_duration"),
        "eval_count": obj.get("eval_count"),
        "prompt_eval_count": obj.get("prompt_eval_count"),
    }
    return parsed


def main() -> None:
    args = parse_args()
    progress_path = Path(args.progress)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    completed = _load_completed(out_path)
    rows = [
        row
        for row in _iter_rows(progress_path)
        if str(row.get("dataset")) == args.dataset
        and str(row.get("method")) == args.method
        and float(row.get("noise")) == float(args.noise)
    ]
    if args.limit > 0:
        rows = rows[: args.limit]

    family_options = FAMILY_OPTIONS[args.dataset]
    with out_path.open("a", encoding="utf-8") as fh:
        for row in rows:
            case_id = str(row.get("case_id", ""))
            if not case_id or case_id in completed:
                continue
            prompt = _build_prompt(args.dataset, family_options, row)
            audit = _call_ollama(args.ollama_url, args.model, prompt)
            record = {
                "dataset": args.dataset,
                "method": args.method,
                "noise": float(args.noise),
                "case_id": case_id,
                "gt_family": row.get("gt_label"),
                "gt_action_id": row.get("gt_action_id"),
                "selected_alert": row.get("noisy_selected_alert"),
                "audit_family": audit.get("family", ""),
                "audit_confidence": audit.get("confidence"),
                "shortcut_signals": audit.get("shortcut_signals", []),
                "brief_reason": audit.get("brief_reason", ""),
                "audit_meta": audit.get("_ollama_meta", {}),
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            fh.flush()


if __name__ == "__main__":
    main()
