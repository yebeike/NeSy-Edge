from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.thesis_rebuild_20260315.shared.evaluators.normalized_pa import normalize_template

NOISE_LEVELS = ["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"]


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest-name", required=True)
    ap.add_argument("--rows-path", default="")
    ap.add_argument("--json-out", default="")
    ap.add_argument("--md-out", default="")
    return ap.parse_args()


def _load_manifest(name: str) -> dict:
    path = _PROJECT_ROOT / "experiments" / "thesis_rebuild_20260315" / "manifests" / name
    return json.loads(path.read_text(encoding="utf-8"))


def _read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _split_metrics(meta: dict) -> dict:
    refs = meta["reference_cases"]
    eval_cases = meta["eval_cases"]
    ref_case_ids = {row["case_id"] for row in refs}
    eval_case_ids = {row["case_id"] for row in eval_cases}
    ref_raw = {row["raw_alert"] for row in refs}
    eval_raw = {row["raw_alert"] for row in eval_cases}
    ref_clean = {row["clean_alert"] for row in refs}
    eval_clean = {row["clean_alert"] for row in eval_cases}
    ref_tpl = {normalize_template(row["gt_template"]) for row in refs}
    eval_tpl = {normalize_template(row["gt_template"]) for row in eval_cases}
    eval_clean_counter = Counter(row["clean_alert"] for row in eval_cases)
    top_dups = [
        {"count": count, "clean_alert": text}
        for text, count in eval_clean_counter.most_common(10)
        if count > 1
    ]
    return {
        "reference_count": len(refs),
        "eval_count": len(eval_cases),
        "case_id_overlap": len(ref_case_ids & eval_case_ids),
        "raw_overlap": len(ref_raw & eval_raw),
        "clean_overlap": len(ref_clean & eval_clean),
        "template_overlap": len(ref_tpl & eval_tpl),
        "ref_unique_clean": len(ref_clean),
        "eval_unique_clean": len(eval_clean),
        "eval_unique_clean_ratio": round(len(eval_clean) / max(len(eval_cases), 1), 4),
        "ref_unique_templates": len(ref_tpl),
        "eval_unique_templates": len(eval_tpl),
        "eval_duplicate_clean_cases": len(eval_cases) - len(eval_clean),
        "top_eval_clean_duplicates": top_dups,
    }


def _rows_metrics(rows: list[dict]) -> tuple[dict, list[str]]:
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["method"], f"{float(row['noise']):.1f}")].append(row)

    result: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    flags: list[str] = []
    per_method_pa: dict[tuple[str, str], list[tuple[str, float]]] = defaultdict(list)

    for (dataset, method, noise), part in sorted(grouped.items()):
        pa_vals = [float(row["pa_hit"]) for row in part]
        lat_vals = [float(row["latency_ms"]) for row in part]
        pa = round(sum(pa_vals) / len(pa_vals), 4)
        lat = round(sum(lat_vals) / len(lat_vals), 3)
        result[dataset][f"{method}:{noise}"] = {
            "cases": len(part),
            "pa": pa,
            "latency_ms": lat,
            "median_latency_ms": round(statistics.median(lat_vals), 3),
        }
        per_method_pa[(dataset, method)].append((noise, pa))

    for (dataset, method), seq in sorted(per_method_pa.items()):
        seq.sort(key=lambda item: float(item[0]))
        pa_values = [pa for _, pa in seq]
        if method == "Qwen" and len(pa_values) >= 3 and len(set(pa_values)) == 1:
            flags.append(f"{dataset}: {method} PA is exactly flat across noise levels {','.join(n for n, _ in seq)}")
    for dataset in sorted({dataset for dataset, _, _ in grouped}):
        curves = {}
        for method in ("Drain", "NuSy", "Qwen"):
            seq = sorted(per_method_pa.get((dataset, method), []), key=lambda item: float(item[0]))
            if len(seq) == len(NOISE_LEVELS):
                curves[method] = tuple(pa for _, pa in seq)
        for left, right in (("Drain", "NuSy"), ("Drain", "Qwen"), ("NuSy", "Qwen")):
            if left in curves and right in curves and curves[left] == curves[right]:
                flags.append(f"{dataset}: {left} and {right} PA curves are exactly identical")

    return result, flags


def _to_markdown(payload: dict) -> str:
    lines = [
        f"# RQ1 Artifact Audit: {payload['manifest']}",
        "",
        f"- Sampling mode: `{payload.get('sampling_mode', '')}`",
        "",
        "## Split audit",
        "",
    ]
    for dataset, meta in payload["datasets"].items():
        split = meta["split"]
        lines.extend(
            [
                f"### {dataset}",
                "",
                f"- refs/eval: `{split['reference_count']}` / `{split['eval_count']}`",
                f"- overlaps: `case_id={split['case_id_overlap']}`, `raw={split['raw_overlap']}`, `clean={split['clean_overlap']}`, `template={split['template_overlap']}`",
                f"- eval unique clean: `{split['eval_unique_clean']}` / `{split['eval_count']}` (`{split['eval_unique_clean_ratio']}`)",
                f"- eval unique templates: `{split['eval_unique_templates']}`",
                "",
            ]
        )
        if split["top_eval_clean_duplicates"]:
            lines.append("| Duplicate Clean Alert | Count |")
            lines.append("|---|---:|")
            for row in split["top_eval_clean_duplicates"][:5]:
                lines.append(f"| {row['clean_alert']} | {row['count']} |")
            lines.append("")

    if payload.get("rows_summary"):
        lines.extend(["## Result audit", ""])
        for dataset, meta in sorted(payload["rows_summary"].items()):
            lines.append(f"### {dataset}")
            for key, row in sorted(meta.items(), key=lambda item: (item[0].split(':')[0], float(item[0].split(':')[1]))):
                lines.append(f"- {key}: n={row['cases']} pa={row['pa']} lat={row['latency_ms']} ms")
            lines.append("")

    lines.extend(["## Flags", ""])
    if payload["flags"]:
        lines.extend(f"- {flag}" for flag in payload["flags"])
    else:
        lines.append("- None")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = _parse_args()
    manifest = _load_manifest(args.manifest_name)
    payload = {
        "manifest": args.manifest_name,
        "sampling_mode": manifest.get("sampling_mode", ""),
        "datasets": {},
        "rows_summary": {},
        "flags": [],
    }

    for dataset, meta in manifest["datasets"].items():
        payload["datasets"][dataset] = {"split": _split_metrics(meta)}

    rows_path = Path(args.rows_path) if args.rows_path else None
    if rows_path:
        rows = _read_rows(rows_path)
        rows_summary, row_flags = _rows_metrics(rows)
        payload["rows_summary"] = rows_summary
        payload["flags"].extend(row_flags)
        expected_rows = sum(len(meta["eval_cases"]) * len(NOISE_LEVELS) * 3 for meta in manifest["datasets"].values())
        if len(rows) != expected_rows:
            payload["flags"].append(f"Run incomplete: final rows {len(rows)} != expected {expected_rows}")
        grouped = defaultdict(int)
        for row in rows:
            grouped[(row["dataset"], f"{float(row['noise']):.1f}", row["method"])] += 1
        for dataset, meta in manifest["datasets"].items():
            expected_cases = len(meta["eval_cases"])
            for noise in NOISE_LEVELS:
                for method in ("Drain", "NuSy", "Qwen"):
                    actual = grouped[(dataset, noise, method)]
                    if actual != expected_cases:
                        payload["flags"].append(
                            f"{dataset}: incomplete group noise={noise} method={method} rows={actual} expected={expected_cases}"
                        )

    for dataset, meta in payload["datasets"].items():
        split = meta["split"]
        if split["case_id_overlap"] or split["raw_overlap"] or split["clean_overlap"]:
            payload["flags"].append(f"{dataset}: ref/eval overlap detected in strict audit")
        if split["eval_unique_clean_ratio"] < 0.9:
            payload["flags"].append(f"{dataset}: eval unique-clean ratio below 0.9")

    if args.json_out:
        path = Path(args.json_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if args.md_out:
        path = Path(args.md_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_to_markdown(payload), encoding="utf-8")

    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
