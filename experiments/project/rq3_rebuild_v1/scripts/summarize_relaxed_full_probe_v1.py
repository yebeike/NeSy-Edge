from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Mapping


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", type=Path, required=True)
    ap.add_argument("--progress", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    return ap.parse_args()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def metric(part: List[Mapping[str, object]]) -> Dict[str, float]:
    total = len(part)
    return {
        "rows": total,
        "family_accuracy": round(sum(int(bool(row.get("rca_success"))) for row in part) / max(1, total), 4),
        "action_accuracy": round(sum(int(bool(row.get("action_success"))) for row in part) / max(1, total), 4),
        "action_text_success_rate": round(sum(int(bool(row.get("action_text_success"))) for row in part) / max(1, total), 4),
        "e2e_accuracy": round(sum(int(bool(row.get("e2e_success"))) for row in part) / max(1, total), 4),
        "rca_e2e_gap_rows": int(
            sum(int(bool(row.get("rca_success")) and not bool(row.get("e2e_success"))) for row in part)
        ),
    }


def summarize_rows(rows: List[Mapping[str, object]]) -> Dict[str, object]:
    by_mode = {}
    by_mode_dataset = defaultdict(dict)
    by_mode_dataset_noise = defaultdict(lambda: defaultdict(dict))
    for mode in sorted({str(row["mode"]) for row in rows}):
        part = [row for row in rows if str(row["mode"]) == mode]
        by_mode[mode] = metric(part)
        for dataset in sorted({str(row["dataset"]) for row in part}):
            ds_part = [row for row in part if str(row["dataset"]) == dataset]
            by_mode_dataset[mode][dataset] = metric(ds_part)
            for noise_key in sorted({f"{float(row['noise']):.1f}" for row in ds_part}):
                noise_part = [row for row in ds_part if f"{float(row['noise']):.1f}" == noise_key]
                by_mode_dataset_noise[mode][dataset][noise_key] = metric(noise_part)
    return {
        "by_mode": by_mode,
        "by_mode_dataset": by_mode_dataset,
        "by_mode_dataset_noise": by_mode_dataset_noise,
    }


def build_case_meta(spec: Mapping[str, object]) -> Dict[tuple[str, str], Dict[str, object]]:
    out: Dict[tuple[str, str], Dict[str, object]] = {}
    for dataset, items in spec["datasets"].items():
        for item in items:
            case_id = str(item.get("eval_case_id", "") or item["case_id"])
            out[(dataset, case_id)] = dict(item)
    return out


def subset_rows(rows: List[Mapping[str, object]], spec_meta: Mapping[tuple[str, str], Mapping[str, object]]) -> Dict[str, List[Dict[str, object]]]:
    annotated: List[Dict[str, object]] = []
    for row in rows:
        key = (str(row["dataset"]), str(row["case_id"]))
        meta = dict(spec_meta.get(key, {}))
        merged = dict(row)
        merged.update(
            {
                "quality_tier": str(meta.get("quality_tier", row.get("quality_tier", "")) or ""),
                "selection_bucket": str(meta.get("selection_bucket", row.get("selection_bucket", "")) or ""),
                "toxicity_flags": list(meta.get("toxicity_flags", row.get("toxicity_flags", [])) or []),
                "base_incident_id": str(meta.get("base_incident_id", row.get("base_incident_id", "")) or ""),
                "reanchor_group_id": str(meta.get("reanchor_group_id", row.get("reanchor_group_id", "")) or ""),
                "is_duplicate_reanchor": bool(meta.get("is_duplicate_reanchor", row.get("is_duplicate_reanchor", False))),
            }
        )
        annotated.append(merged)

    return {
        "relaxed_full": [row for row in annotated if str(row.get("quality_tier", "")) != "excluded"],
        "core_subset": [
            row
            for row in annotated
            if str(row.get("quality_tier", "")) in {"core_hard", "core_usable"}
            and not bool(row.get("is_duplicate_reanchor", False))
        ],
        "dedup_relaxed": [
            row
            for row in annotated
            if str(row.get("quality_tier", "")) != "excluded" and not bool(row.get("is_duplicate_reanchor", False))
        ],
    }


def subset_case_inventory(spec: Mapping[str, object], subset_name: str) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for dataset, items in spec["datasets"].items():
        kept = []
        for item in items:
            quality = str(item.get("quality_tier", "") or "")
            duplicate = bool(item.get("is_duplicate_reanchor", False))
            if subset_name == "relaxed_full" and quality == "excluded":
                continue
            if subset_name == "core_subset" and not (quality in {"core_hard", "core_usable"} and not duplicate):
                continue
            if subset_name == "dedup_relaxed" and (quality == "excluded" or duplicate):
                continue
            kept.append(item)
        out[dataset] = {
            "cases": len(kept),
            "quality_tier_counts": dict(Counter(str(item.get("quality_tier", "")) for item in kept)),
            "selection_bucket_counts": dict(Counter(str(item.get("selection_bucket", "")) for item in kept)),
            "action_counts": dict(Counter(str(item.get("gt_action_id", "")) for item in kept)),
            "duplicate_reanchor_cases": int(sum(int(bool(item.get("is_duplicate_reanchor", False))) for item in kept)),
        }
    return out


def main() -> None:
    args = parse_args()
    spec = load_json(args.spec)
    rows = [json.loads(line) for line in args.progress.read_text(encoding="utf-8").splitlines() if line.strip()]
    spec_meta = build_case_meta(spec)
    subsets = subset_rows(rows, spec_meta)

    payload = {
        "benchmark_id": str(spec["benchmark_id"]),
        "progress_path": str(args.progress),
        "spec_path": str(args.spec),
        "subset_summaries": {},
    }
    for name, part in subsets.items():
        payload["subset_summaries"][name] = {
            "rows": len(part),
            "case_inventory": subset_case_inventory(spec, name),
            **summarize_rows(part),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(args.output)
    print(json.dumps({k: v["rows"] for k, v in payload["subset_summaries"].items()}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
