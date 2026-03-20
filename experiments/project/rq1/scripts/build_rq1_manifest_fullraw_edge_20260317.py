from __future__ import annotations

import argparse
from collections import defaultdict
import json
import random
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parents[3]
_REBUILD_ROOT = _SCRIPT_DIR.parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.thesis_rebuild_20260315.rq1.components.edge_protocol_20260317 import (
    _sample_cases_covered_sanity,
    _sample_cases_natural_split,
    preset_manifest_name,
    save_manifest,
)
from experiments.thesis_rebuild_20260315.shared.case_builders.rq1_case_pool import RQ1Case
from experiments.thesis_rebuild_20260315.shared.evaluators.normalized_pa import normalize_template


FULLRAW_PRESETS = {
    "small": {"eval": {"HDFS": 60, "OpenStack": 60, "Hadoop": 60}, "refs": {"HDFS": 96, "OpenStack": 128, "Hadoop": 96}},
    "mid": {"eval": {"HDFS": 300, "OpenStack": 300, "Hadoop": 300}, "refs": {"HDFS": 96, "OpenStack": 128, "Hadoop": 96}},
    "full": {
        "eval": {"HDFS": 2000, "OpenStack": 1600, "Hadoop": 1400},
        "refs": {"HDFS": 160, "OpenStack": 192, "Hadoop": 160},
    },
}

DEFAULT_POOLS = {
    "HDFS": _REBUILD_ROOT / "rq1" / "artifacts" / "rq1_hdfs_fullraw_pool_20260317_v2.json",
    "OpenStack": _REBUILD_ROOT / "rq1" / "artifacts" / "rq1_openstack_fullraw_pool_20260317.json",
    "Hadoop": _REBUILD_ROOT / "rq1" / "artifacts" / "rq1_hadoop_fullraw_pool_20260317.json",
}


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=sorted(FULLRAW_PRESETS), default="mid")
    ap.add_argument("--suffix", type=str, default="fullraw_v1")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--manifest-name", type=str, default="")
    ap.add_argument("--datasets", type=str, default="HDFS,OpenStack,Hadoop")
    ap.add_argument("--sampling-mode", choices=("natural", "unique_clean"), default="natural")
    return ap.parse_args()


def _selected_datasets(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _load_pool_cases(path: Path, dataset: str) -> tuple[list[RQ1Case], dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    meta = payload["datasets"][dataset]
    cases = [RQ1Case(**row) for row in meta["cases"]]
    return cases, meta


def _split_audit(refs: list[RQ1Case], eval_cases: list[RQ1Case]) -> dict:
    ref_case_ids = {case.case_id for case in refs}
    eval_case_ids = {case.case_id for case in eval_cases}
    ref_raw = {case.raw_alert for case in refs}
    eval_raw = {case.raw_alert for case in eval_cases}
    ref_clean = {case.clean_alert for case in refs}
    eval_clean = {case.clean_alert for case in eval_cases}
    ref_tpl = {normalize_template(case.gt_template) for case in refs}
    eval_tpl = {normalize_template(case.gt_template) for case in eval_cases}
    return {
        "case_id_overlap": len(ref_case_ids & eval_case_ids),
        "raw_overlap": len(ref_raw & eval_raw),
        "clean_overlap": len(ref_clean & eval_clean),
        "template_overlap": len(ref_tpl & eval_tpl),
        "ref_unique_clean": len(ref_clean),
        "eval_unique_clean": len(eval_clean),
        "ref_unique_templates": len(ref_tpl),
        "eval_unique_templates": len(eval_tpl),
    }


def _sample_cases_unique_clean(
    cases: list[RQ1Case],
    desired_eval: int,
    desired_refs: int,
    seed: int,
) -> tuple[list[RQ1Case], list[RQ1Case], dict]:
    rng = random.Random(seed)
    by_template: dict[str, dict[str, list[RQ1Case]]] = defaultdict(lambda: defaultdict(list))
    for case in cases:
        by_template[normalize_template(case.gt_template)][case.clean_alert].append(case)

    grouped: dict[str, list[list[RQ1Case]]] = {}
    for template, clean_map in by_template.items():
        groups = list(clean_map.values())
        for group in groups:
            rng.shuffle(group)
        rng.shuffle(groups)
        grouped[template] = groups

    refs: list[RQ1Case] = []
    template_order = sorted(grouped, key=lambda item: len(grouped[item]), reverse=True)
    for template in template_order:
        if len(refs) >= desired_refs:
            break
        groups = grouped[template]
        if not groups:
            continue
        refs.append(groups.pop()[0])

    while len(refs) < desired_refs:
        candidates = [template for template, groups in grouped.items() if groups]
        if not candidates:
            break
        template = max(candidates, key=lambda item: len(grouped[item]))
        refs.append(grouped[template].pop()[0])

    eval_cases: list[RQ1Case] = []
    chosen_ids = {case.case_id for case in refs}
    active_templates = [template for template, groups in grouped.items() if groups]
    active_templates.sort(key=lambda item: len(grouped[item]), reverse=True)

    for template in active_templates:
        if len(eval_cases) >= desired_eval:
            break
        groups = grouped[template]
        if not groups:
            continue
        picked = groups.pop()[0]
        if picked.case_id in chosen_ids:
            continue
        eval_cases.append(picked)
        chosen_ids.add(picked.case_id)

    eval_candidates: list[RQ1Case] = []
    for groups in grouped.values():
        for group in groups:
            if not group:
                continue
            picked = group[0]
            if picked.case_id not in chosen_ids:
                eval_candidates.append(picked)
    rng.shuffle(eval_candidates)
    for case in eval_candidates:
        if len(eval_cases) >= desired_eval:
            break
        eval_cases.append(case)
        chosen_ids.add(case.case_id)
    return refs, eval_cases, _split_audit(refs, eval_cases)


def _sample_cases(
    cases: list[RQ1Case],
    desired_eval: int,
    desired_refs: int,
    seed: int,
    dataset: str,
    sampling_mode: str,
) -> tuple[list[RQ1Case], list[RQ1Case], dict]:
    desired_eval = min(desired_eval, len(cases))
    desired_refs = min(desired_refs, max(len(cases) // 5, 1))
    if desired_eval <= 40:
        refs, eval_cases = _sample_cases_covered_sanity(
            cases=cases,
            desired_eval=desired_eval,
            desired_refs=desired_refs,
            seed=seed,
            dataset=dataset,
        )
        return refs, eval_cases, _split_audit(refs, eval_cases)
    if sampling_mode == "unique_clean":
        return _sample_cases_unique_clean(
            cases=cases,
            desired_eval=desired_eval,
            desired_refs=desired_refs,
            seed=seed,
        )
    refs, eval_cases = _sample_cases_natural_split(
        cases=cases,
        desired_eval=desired_eval,
        desired_refs=desired_refs,
        seed=seed,
    )
    return refs, eval_cases, _split_audit(refs, eval_cases)


def main() -> str:
    args = _parse_args()
    spec = FULLRAW_PRESETS[args.preset]
    selected = set(_selected_datasets(args.datasets))
    manifest = {
        "seed": args.seed,
        "protocol": f"rq1_edge_fullraw_{args.preset}_20260317",
        "sampling_mode": args.sampling_mode,
        "datasets": {},
    }
    compact = {}

    for dataset, path in DEFAULT_POOLS.items():
        if dataset not in selected:
            continue
        cases, pool_meta = _load_pool_cases(path, dataset)
        refs, eval_cases, audit = _sample_cases(
            cases=cases,
            desired_eval=spec["eval"][dataset],
            desired_refs=spec["refs"][dataset],
            seed=args.seed,
            dataset=dataset,
            sampling_mode=args.sampling_mode,
        )
        manifest["datasets"][dataset] = {
            "pool_size": len(cases),
            "pool_completed": bool(pool_meta.get("completed", False)),
            "pool_coverage": pool_meta.get("coverage"),
            "sampling_mode": args.sampling_mode,
            "split_audit": audit,
            "reference_count": len(refs),
            "eval_count": len(eval_cases),
            "reference_cases": [case.__dict__ for case in refs],
            "eval_cases": [case.__dict__ for case in eval_cases],
        }
        compact[dataset] = {
            "pool_size": len(cases),
            "pool_completed": bool(pool_meta.get("completed", False)),
            "pool_coverage": pool_meta.get("coverage"),
            "sampling_mode": args.sampling_mode,
            "reference_count": len(refs),
            "eval_count": len(eval_cases),
            "split_audit": audit,
        }

    name = args.manifest_name or preset_manifest_name(args.preset, args.suffix)
    out_path = save_manifest(manifest, name)
    print(json.dumps({"preset": args.preset, "manifest": name, "datasets": compact}, indent=2))
    print(f"[Saved] {out_path}")
    return str(out_path)


if __name__ == "__main__":
    main()
