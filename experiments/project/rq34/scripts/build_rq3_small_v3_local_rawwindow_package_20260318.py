from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
REBUILD_ROOT = SCRIPT_DIR.parents[1]
PROJECT_ROOT = REBUILD_ROOT.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.thesis_rebuild_20260315.rq34.scripts import build_rq3_small_v2_benchmark_20260318 as builder_v2


SPEC_PATH = (
    REBUILD_ROOT
    / "rq34"
    / "analysis"
    / "rq3_small_v3_diagnostic_slice_20260318"
    / "rq3_small_v3_diagnostic_slice_spec_20260318.json"
)
OUTPUT_DIR = REBUILD_ROOT / "rq34" / "analysis" / "rq3_small_v3_diagnostic_slice_20260318"
WINDOW_RULES: Dict[str, Tuple[int, int]] = {
    "HDFS": (2, 1),
    "OpenStack": (2, 2),
    "Hadoop": (2, 2),
}


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _norm(text: str) -> str:
    return " ".join(str(text or "").split())


def _extract_raw_window_context(dataset: str, raw_log: str, selected_alert: str) -> str:
    before, after = WINDOW_RULES[dataset]
    lines = [line.strip() for line in str(raw_log or "").splitlines() if line.strip()]
    if not lines:
        return ""
    wanted = _norm(selected_alert)
    hit_index = -1
    for idx, line in enumerate(lines):
        if wanted == _norm(line) or wanted in _norm(line):
            hit_index = idx
            break
    if hit_index < 0:
        raise ValueError(f"Failed to locate raw-window context anchor for {dataset}: {selected_alert}")

    chosen: List[str] = []
    for idx in range(max(0, hit_index - before), min(len(lines), hit_index + after + 1)):
        if idx == hit_index:
            continue
        chosen.append(lines[idx])

    return builder_v2._compact_lines(chosen, builder_v2.DATASET_CONTEXT_MAX_CHARS[dataset])


def build_package() -> Dict[str, object]:
    spec = _load_json(SPEC_PATH)
    legacy = builder_v2._load_legacy_module()
    old_runner = builder_v2._load_old_runner_module()
    pool_rows = builder_v2._load_pool_rows()
    injector = legacy.NoiseInjector(seed=2026)
    injector_hadoop = legacy.HadoopNoiseInjector(seed=2026)

    cases: List[Dict[str, object]] = []
    dataset_case_counts: Dict[str, int] = {}

    for dataset, section in spec["datasets"].items():
        approved_cases = list(section.get("approved_cases", []))
        dataset_case_counts[dataset] = len(approved_cases)
        for item in approved_cases:
            source = str(item["source"])
            case_id = str(item["case_id"])
            gt_action_id = str(item["gt_action_id"])
            case_row = pool_rows[(source, case_id)]
            raw_log = str(case_row.get("raw_log", "") or "")
            selected_alert = builder_v2._find_alert_line(
                raw_log,
                alert_match=str(item["alert_match"]),
                occurrence=int(item.get("alert_occurrence", 1) or 1),
            )
            shared_context = _extract_raw_window_context(dataset, raw_log, selected_alert)

            noise_views: Dict[str, Dict[str, object]] = {}
            for noise in spec["noise_levels"]:
                noise_value = float(noise)
                noise_key = old_runner._noise_key(noise_value)
                if noise_value == 0.0:
                    noisy_alert = selected_alert
                    noisy_context = shared_context
                else:
                    noisy_alert = old_runner._inject_noise_line(
                        legacy,
                        selected_alert,
                        dataset,
                        noise_value,
                        role="selected_alert",
                    )
                    noisy_alert = builder_v2._sanitize_noisy_alert(
                        dataset,
                        selected_alert,
                        noisy_alert,
                        gt_action_id,
                        noise_value,
                    )
                    noisy_context_raw = old_runner._inject_noise_preserve_context(
                        legacy,
                        shared_context,
                        dataset,
                        injector,
                        injector_hadoop,
                        noise_value,
                    )
                    noisy_context = builder_v2._compact_lines(
                        [line for line in str(noisy_context_raw or "").splitlines() if line.strip()],
                        builder_v2.DATASET_CONTEXT_MAX_CHARS[dataset],
                    )

                noise_views[noise_key] = {
                    "noise": noise_value,
                    "selected_alert": noisy_alert,
                    "context_text": noisy_context,
                    "observed_template": "",
                    "graph_summary": "",
                    "symbolic_family_clue": "",
                    "agent_references": [],
                    "rag_references": [],
                }

            cases.append(
                {
                    "dataset": dataset,
                    "case_id": case_id,
                    "source": source,
                    "gt_family_id": str(item["gt_family_id"]),
                    "gt_action_id": gt_action_id,
                    "selected_alert_clean": selected_alert,
                    "shared_context_clean": shared_context,
                    "eligibility_note": str(item["eligibility_note"]) + " Raw-window context excludes the selected alert line.",
                    "raw_log": raw_log,
                    "noise_views": noise_views,
                }
            )

    package = {
        "benchmark_id": str(spec["benchmark_id"]) + "_rawwindow",
        "benchmark_kind": "local_diagnostic_slice_rawwindow",
        "purpose": str(spec["purpose"]) + " Raw-window variant excludes the selected alert from context.",
        "noise_levels": [float(x) for x in spec["noise_levels"]],
        "spec_path": str(SPEC_PATH),
        "formal_small_ready": False,
        "paid_api_allowed": False,
        "cases": cases,
        "reference_bank": {},
    }
    summary = {
        "benchmark_id": package["benchmark_id"],
        "benchmark_kind": package["benchmark_kind"],
        "dataset_case_counts": dataset_case_counts,
        "case_count": len(cases),
        "noise_levels": [float(x) for x in spec["noise_levels"]],
        "package_path": str(OUTPUT_DIR / "rq3_small_v3_local_rawwindow_package_20260318.json"),
    }
    return {"package": package, "summary": summary}


def main() -> None:
    payload = build_package()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    package_path = OUTPUT_DIR / "rq3_small_v3_local_rawwindow_package_20260318.json"
    summary_path = OUTPUT_DIR / "rq3_small_v3_local_rawwindow_summary_20260318.json"
    package_path.write_text(json.dumps(payload["package"], indent=2, ensure_ascii=False), encoding="utf-8")
    summary_path.write_text(json.dumps(payload["summary"], indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
