from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

_SCRIPT_DIR = Path(__file__).resolve().parent
_AUDIT_ROOT = _SCRIPT_DIR.parent
_PROJECT_ROOT = _AUDIT_ROOT.parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_PROJECT_ROOT / "experiments" / "rq123_e2e") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "experiments" / "rq123_e2e"))

from build_e2e_scaled_benchmark import _extract_target_line_and_template  # type: ignore
from rq2_fullcase_audit_common_20260318 import (
    AUDIT_BENCHMARK_EVAL_PATH,
    AUDIT_BENCHMARK_FULL_PATH,
    RAW_OPENSTACK_2,
    REPORTS_DIR,
    SOURCE_BENCH_PATH,
    best_graph_template,
    dataset_to_domain,
    domain_template_pool,
    ensure_dirs,
    family_of,
    iter_with_progress,
    last_prior_by_family,
    load_json,
    manual_prior_pair_overlap,
    map_prior_templates_to_graph_candidates,
    pick_root_graph_from_prior_templates,
    write_json,
)


OPENSTACK_WINDOW_RADIUS = 18
OPENSTACK_CASE_SPECS: List[Dict[str, object]] = [
    {
        "name": "unknown_base",
        "keyword": "unknown base file:",
        "count": 10,
        "effect_template": "Unknown base file: <*>",
        "root_template": "Removable base files: <*>",
    },
    {
        "name": "base_too_young",
        "keyword": "base or swap file too young to remove:",
        "count": 15,
        "effect_template": "Base or swap file too young to remove: <*>",
        "root_template": "Removable base files: <*>",
    },
    {
        "name": "removable_base",
        "keyword": "removable base files:",
        "count": 10,
        "effect_template": "Removable base files: <*>",
        "root_template": "Active base files: <*>",
    },
    {
        "name": "pending_task",
        "keyword": "pending task",
        "count": 8,
        "effect_template": "nova-compute.log.<*>.<*>-<*>-<*>_<*>:<*>:<*>-<*>-<*>:<*>:<*>.<*> INFO nova.compute.manager [-] [instance: <*>] During sync_power_state the instance has a pending task <*> Skip.",
        "root_template": "While synchronizing instance power states, found <*> instances in the database and <*> instances on the hypervisor.",
    },
    {
        "name": "instance_sync_recreated",
        "keyword": "re-created its instancelist",
        "count": 5,
        "effect_template": "The instance sync for host 'cp-<*>.slowvm<*>.tcloud-pg<*>.utah.cloudlab.us' did not match. Re-created its InstanceList.",
        "root_template": "Successfully synced instances from host 'cp-<*>.slowvm<*>.tcloud-pg<*>.utah.cloudlab.us'.",
    },
    {
        "name": "token_validation",
        "keyword": "bad response code while validating token",
        "count": 2,
        "effect_template": "Bad response code while validating token: <*>",
        "root_template": "Identity response: <!DOCTYPE HTML PUBLIC \"-<*> HTML <*>.<*><*>\">",
    },
]


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    return ap.parse_args()


def _window_lines(lines: List[str], idx: int, radius: int = OPENSTACK_WINDOW_RADIUS) -> List[str]:
    lo = max(0, idx - radius)
    hi = min(len(lines), idx + radius + 1)
    return [line for line in lines[lo:hi] if line.strip()]


def _extract_target_and_prior_lines(case: Dict[str, object], dataset: str) -> Tuple[str, List[str], str]:
    lines = [str(x) for x in (case.get("raw_log_window") or []) if str(x).strip()]
    if not lines:
        raw = str(case.get("raw_log", "") or "")
        lines = [x for x in raw.splitlines() if x.strip()]

    hit_idx = case.get("audit_hit_local_index")
    if hit_idx is not None:
        idx = int(hit_idx)
        if 0 <= idx < len(lines):
            target_line = lines[idx]
            _, target_tpl = _extract_target_line_and_template([target_line], dataset)
            return target_line, lines[:idx], target_tpl

    target_line, target_tpl = _extract_target_line_and_template(lines, dataset)
    prior_lines: List[str] = []
    found_target = False
    for line in reversed(lines):
        if line == target_line and not found_target:
            found_target = True
            continue
        if found_target:
            prior_lines.append(line)
    prior_lines.reverse()
    if not found_target:
        prior_lines = lines[:-1]
    return target_line, prior_lines, target_tpl


def _parsed_templates_from_lines(lines: List[str], dataset: str) -> List[str]:
    parsed: List[str] = []
    for line in lines:
        _, tpl = _extract_target_line_and_template([line], dataset)
        tpl = str(tpl or "").strip()
        if tpl:
            parsed.append(tpl)
    return parsed


def _openstack_raw_files() -> List[Path]:
    return [RAW_OPENSTACK_2 / name for name in sorted(os.listdir(RAW_OPENSTACK_2)) if name.endswith(".log")]


def _collect_openstack_keyword_hits(keyword: str) -> List[Tuple[Path, int, List[str]]]:
    hits: List[Tuple[Path, int, List[str]]] = []
    keyword_lower = keyword.lower()
    for path in _openstack_raw_files():
        lines = path.read_text(encoding="latin-1", errors="ignore").splitlines()
        for idx, line in enumerate(lines):
            if keyword_lower in line.lower():
                hits.append((path, idx, lines))
    return hits


def _select_spaced_hits(
    hits: List[Tuple[Path, int, List[str]]],
    count: int,
    min_gap: int = 12,
) -> List[Tuple[Path, int, List[str]]]:
    if len(hits) <= count:
        return hits
    chosen: List[Tuple[Path, int, List[str]]] = []
    seen_by_file: Dict[Path, List[int]] = {}
    for rank in range(count):
        pick_idx = round(rank * (len(hits) - 1) / max(count - 1, 1))
        candidate = hits[pick_idx]
        path, idx, _ = candidate
        taken = seen_by_file.setdefault(path, [])
        if all(abs(idx - prev) >= min_gap for prev in taken):
            chosen.append(candidate)
            taken.append(idx)
    if len(chosen) == count:
        return chosen
    for candidate in hits:
        if len(chosen) >= count:
            break
        path, idx, _ = candidate
        taken = seen_by_file.setdefault(path, [])
        if all(abs(idx - prev) >= min_gap for prev in taken):
            chosen.append(candidate)
            taken.append(idx)
    return chosen[:count]


def _build_openstack_rebalanced_cases() -> List[Dict[str, object]]:
    cases: List[Dict[str, object]] = []
    case_counter = 0
    for spec in OPENSTACK_CASE_SPECS:
        keyword = str(spec["keyword"])
        raw_hits = _collect_openstack_keyword_hits(keyword)
        if len(raw_hits) < int(spec["count"]):
            raise RuntimeError(
                f"OpenStack keyword '{keyword}' only has {len(raw_hits)} hits, below requested {spec['count']}."
            )
        selected = _select_spaced_hits(raw_hits, int(spec["count"]))
        if len(selected) < int(spec["count"]):
            raise RuntimeError(
                f"OpenStack keyword '{keyword}' only yielded {len(selected)} spaced hits, below requested {spec['count']}."
            )
        for path, idx, lines in selected:
            window_lines = _window_lines(lines, idx)
            window_lo = max(0, idx - OPENSTACK_WINDOW_RADIUS)
            local_idx = idx - window_lo
            hit_line = lines[idx]
            _, hit_template = _extract_target_line_and_template([hit_line], "OpenStack")
            cases.append(
                {
                    "case_id": f"openstack_audit_{case_counter:03d}",
                    "dataset": "OpenStack",
                    "raw_log": "\n".join(window_lines),
                    "raw_log_window": window_lines,
                    "source": f"openstack_raw_semantic_rebuild_20260318::{path.name}",
                    "ground_truth_template": str(spec["effect_template"]),
                    "ground_truth_root_cause_template": str(spec["root_template"]),
                    "audit_case_spec": str(spec["name"]),
                    "audit_hit_keyword": keyword,
                    "audit_hit_line": hit_line,
                    "audit_hit_template": str(hit_template or ""),
                    "audit_hit_local_index": local_idx,
                }
            )
            case_counter += 1
    if len(cases) != 50:
        raise RuntimeError(f"Expected 50 rebuilt OpenStack cases, got {len(cases)}.")
    return cases


def _derive_graph_labels(
    dataset: str,
    row: Dict[str, object],
    pool: List[str],
    use_label_hints: bool,
) -> Tuple[str, str, Dict[str, object]]:
    target_line, prior_lines, target_tpl = _extract_target_and_prior_lines(row, dataset)
    desired_root_family = family_of(dataset, str(row.get("ground_truth_root_cause_template", "") or ""))
    prior_root_line = last_prior_by_family(dataset, prior_lines, desired_root_family)
    prior_templates = _parsed_templates_from_lines(prior_lines, dataset)

    effect_queries = [target_tpl, target_line]
    if use_label_hints:
        effect_queries.insert(0, str(row.get("ground_truth_template", "") or ""))

    graph_effect = best_graph_template(dataset, effect_queries, pool)
    graph_root = pick_root_graph_from_prior_templates(
        dataset,
        prior_templates,
        pool,
        graph_effect=graph_effect,
        root_label_hint=str(row.get("ground_truth_root_cause_template", "") or "") if use_label_hints else "",
    )
    mapped_prior_candidates = map_prior_templates_to_graph_candidates(dataset, prior_templates, pool)
    same_family_prior_candidates: List[str] = []
    seen_same_family = set()
    effect_family = family_of(dataset, graph_effect)
    for candidate in mapped_prior_candidates:
        if family_of(dataset, candidate) != effect_family:
            continue
        if candidate == graph_effect:
            continue
        if candidate in seen_same_family:
            continue
        seen_same_family.add(candidate)
        same_family_prior_candidates.append(candidate)

    if dataset == "OpenStack":
        if len(same_family_prior_candidates) == 1:
            graph_root = same_family_prior_candidates[0]
        else:
            graph_root = ""

    if not graph_root and prior_root_line:
        if dataset != "OpenStack":
            graph_root = best_graph_template(dataset, [prior_root_line], pool, desired_family=desired_root_family)
    return graph_effect, graph_root, {
        "audit_target_line": target_line,
        "audit_target_template": target_tpl,
        "audit_prior_lines": prior_lines,
        "audit_prior_templates": prior_templates,
        "audit_prior_root_line": prior_root_line,
        "audit_mapped_prior_graph_candidates": mapped_prior_candidates,
        "audit_same_family_prior_graph_candidates": same_family_prior_candidates,
    }


def main() -> None:
    args = _parse_args()
    ensure_dirs()
    report_path = REPORTS_DIR / "rq2_fullcase_audit_benchmark_20260318.md"

    if AUDIT_BENCHMARK_FULL_PATH.exists() and AUDIT_BENCHMARK_EVAL_PATH.exists() and report_path.exists() and not args.force:
        print(f"[*] Reusing audit benchmark: {AUDIT_BENCHMARK_FULL_PATH}")
        print(f"[*] Reusing evaluable audit benchmark: {AUDIT_BENCHMARK_EVAL_PATH}")
        print(f"[*] Reusing audit benchmark report: {report_path}")
        return

    print("[1/3] Loading source benchmark and template pools. Expected: <10s")
    source_rows = load_json(SOURCE_BENCH_PATH)
    template_pool = domain_template_pool()
    print("[2/3] Rebuilding OpenStack raw-window cases. Expected: 10-20s")
    openstack_rows = _build_openstack_rebalanced_cases()

    audit_rows: List[Dict[str, object]] = []
    counts_total = Counter()
    counts_overlap = Counter()
    counts_missing = Counter()
    effect_stats = Counter()
    root_stats = Counter()

    hdfs_source_rows = [row for row in source_rows if str(row.get("dataset", "") or "") == "HDFS"]
    print(f"[3/3] Deriving audit graph labels for HDFS ({len(hdfs_source_rows)} cases) and OpenStack ({len(openstack_rows)} cases).")
    for row in iter_with_progress(hdfs_source_rows, "HDFS audit benchmark"):
        row2 = dict(row)
        row2["ground_truth_template_label"] = str(row.get("ground_truth_template", "") or "")
        row2["ground_truth_root_cause_label"] = str(row.get("ground_truth_root_cause_template", "") or "")
        graph_effect, graph_root, audit_info = _derive_graph_labels("HDFS", row2, template_pool["HDFS"], use_label_hints=True)
        row2["ground_truth_template_graph"] = graph_effect
        row2["ground_truth_root_cause_template_graph"] = graph_root
        row2.update(audit_info)
        row2["manual_prior_pair_overlap"] = manual_prior_pair_overlap("HDFS", graph_root, graph_effect)
        row2["audit_exclusion_reason"] = "manual_prior_pair_overlap" if row2["manual_prior_pair_overlap"] else ""
        counts_total["HDFS"] += 1
        if row2["manual_prior_pair_overlap"]:
            counts_overlap["HDFS"] += 1
        if not graph_effect or not graph_root:
            counts_missing["HDFS"] += 1
        if graph_effect:
            effect_stats[("HDFS", graph_effect)] += 1
        if graph_root:
            root_stats[("HDFS", graph_root)] += 1
        audit_rows.append(row2)

    for row in iter_with_progress(openstack_rows, "OpenStack audit benchmark"):
        row2 = dict(row)
        row2["ground_truth_template_label"] = str(row.get("ground_truth_template", "") or "")
        row2["ground_truth_root_cause_label"] = str(row.get("ground_truth_root_cause_template", "") or "")
        graph_effect, graph_root, audit_info = _derive_graph_labels("OpenStack", row2, template_pool["OpenStack"], use_label_hints=False)
        row2["ground_truth_template_graph"] = graph_effect
        row2["ground_truth_root_cause_template_graph"] = graph_root
        row2.update(audit_info)
        row2["manual_prior_pair_overlap"] = manual_prior_pair_overlap("OpenStack", graph_root, graph_effect)
        row2["audit_exclusion_reason"] = "manual_prior_pair_overlap" if row2["manual_prior_pair_overlap"] else ""
        counts_total["OpenStack"] += 1
        if row2["manual_prior_pair_overlap"]:
            counts_overlap["OpenStack"] += 1
        if not graph_effect or not graph_root:
            counts_missing["OpenStack"] += 1
        if graph_effect:
            effect_stats[("OpenStack", graph_effect)] += 1
        if graph_root:
            root_stats[("OpenStack", graph_root)] += 1
        audit_rows.append(row2)

    evaluable_rows = [
        row
        for row in audit_rows
        if str(row.get("dataset", "") or "") in {"HDFS", "OpenStack"}
        and str(row.get("ground_truth_template_graph", "") or "").strip()
        and str(row.get("ground_truth_root_cause_template_graph", "") or "").strip()
        and not bool(row.get("manual_prior_pair_overlap", False))
    ]
    eval_counts = Counter(str(row.get("dataset", "") or "") for row in evaluable_rows)

    write_json(AUDIT_BENCHMARK_FULL_PATH, audit_rows)
    write_json(AUDIT_BENCHMARK_EVAL_PATH, evaluable_rows)

    report_lines = [
        "# RQ2 Fullcase Audit Benchmark (2026-03-18)",
        "",
        f"- Source benchmark: `{SOURCE_BENCH_PATH}`",
        f"- Full audit benchmark: `{AUDIT_BENCHMARK_FULL_PATH}`",
        f"- Evaluable audit benchmark: `{AUDIT_BENCHMARK_EVAL_PATH}`",
        "- Scope: `HDFS` and `OpenStack` only",
        "- HDFS graph-space mapping uses generic template-pool matching and does not call the old root priority patch.",
        "- OpenStack graph-space mapping is re-derived from parsed target/prior lines, not copied from hand-written case spec strings.",
        "- OpenStack roots are only kept when the prior window yields exactly one non-effect same-family graph candidate; otherwise the case is left non-evaluable.",
        "",
        "## Counts",
        "",
        "| Dataset | Total | Overlap Excluded | Missing Graph Label | Evaluable |",
        "|---|---:|---:|---:|---:|",
        f"| HDFS | {counts_total['HDFS']} | {counts_overlap['HDFS']} | {counts_missing['HDFS']} | {eval_counts['HDFS']} |",
        f"| OpenStack | {counts_total['OpenStack']} | {counts_overlap['OpenStack']} | {counts_missing['OpenStack']} | {eval_counts['OpenStack']} |",
        "",
    ]
    for dataset in ("HDFS", "OpenStack"):
        report_lines.extend(
            [
                f"## {dataset} graph-space effects",
                "",
            ]
        )
        for (ds, effect), count in sorted(effect_stats.items(), key=lambda kv: (kv[0][0], -kv[1], kv[0][1])):
            if ds != dataset:
                continue
            report_lines.append(f"- `{effect}`: `{count}`")
        report_lines.extend(
            [
                "",
                f"## {dataset} graph-space roots",
                "",
            ]
        )
        for (ds, root), count in sorted(root_stats.items(), key=lambda kv: (kv[0][0], -kv[1], kv[0][1])):
            if ds != dataset:
                continue
            report_lines.append(f"- `{root}`: `{count}`")
        report_lines.append("")
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(f"[Saved] {AUDIT_BENCHMARK_FULL_PATH}")
    print(f"[Saved] {AUDIT_BENCHMARK_EVAL_PATH}")
    print(f"[Saved] {report_path}")


if __name__ == "__main__":
    main()
