from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

_SCRIPT_DIR = Path(__file__).resolve().parent
_FULLCASE_ROOT = _SCRIPT_DIR.parent
_PROJECT_ROOT = _FULLCASE_ROOT.parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_PROJECT_ROOT / "experiments" / "rq123_e2e") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "experiments" / "rq123_e2e"))

from build_e2e_scaled_benchmark import _extract_target_line_and_template  # type: ignore
from rq2_fullcase_common_20260316 import (
    BENCH_V2_PATH,
    REPORTS_DIR,
    RQ2_FULLCASE_BENCH_PATH,
    canonical_tokens,
    ensure_dirs,
    exact_relaxed_match,
    family_of,
    load_hadoop_timeseries,
    load_hdfs_timeseries,
    load_openstack_semantic_timeseries,
)


RAW_OPENSTACK_2 = _PROJECT_ROOT / "data" / "raw" / "OpenStack_2"
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


def _patch_hdfs_root(case: Dict[str, object]) -> str:
    effect = str(case.get("ground_truth_template", "") or "")
    raw = str(case.get("raw_log", "") or "")
    lines = [x for x in raw.splitlines() if x.strip()]
    parsed: List[str] = []
    for line in lines:
        _, tpl = _extract_target_line_and_template([line], "HDFS")
        if tpl:
            parsed.append(tpl)

    last_idx = -1
    for idx, tpl in enumerate(parsed):
        if tpl == effect:
            last_idx = idx
    prior = parsed[:last_idx] if last_idx >= 0 else parsed[:-1]

    priority_by_effect = [
        (
            "Got exception while serving",
            [
                "PacketResponder <*> for block blk_<*> terminating",
                "Received block blk_<*> of size <*> from /<*>",
                "Receiving block blk_<*> src: /<*>:<*> dest: /<*>:<*>",
                "BLOCK* NameSystem.addStoredBlock: blockMap updated: <*>:<*> is added to blk_<*> size <*>",
                "BLOCK* NameSystem.allocateBlock: <*> blk_<*>",
            ],
        ),
        (
            "allocateBlock",
            [
                "Receiving block blk_<*> src: /<*>:<*> dest: /<*>:<*>",
                "Received block blk_<*> of size <*> from /<*>",
                "PacketResponder <*> for block blk_<*> terminating",
            ],
        ),
        (
            "Unexpected error trying to delete block",
            [
                "Deleting block blk_<*> file <*>blk_<*>",
                "BLOCK* NameSystem.delete: blk_<*> is added to invalidSet of <*>:<*>",
            ],
        ),
        (
            "writeBlock",
            [
                "Exception in receiveBlock for block blk_<*> java.io.IOException: Connection reset by peer",
                "PacketResponder blk_<*> Exception java.io.EOFException",
                "PacketResponder <*> for block blk_<*> terminating",
            ],
        ),
    ]

    target_priority: List[str] = []
    for key, options in priority_by_effect:
        if key.lower() in effect.lower():
            target_priority = options
            break

    if not target_priority:
        return str(case.get("ground_truth_root_cause_template", "") or "")

    seen = set(prior)
    for candidate in target_priority:
        if candidate in seen:
            return candidate
    return str(case.get("ground_truth_root_cause_template", "") or "")


def _domain_template_pool() -> Dict[str, List[str]]:
    _, hdfs_map = load_hdfs_timeseries()
    _, openstack_map = load_openstack_semantic_timeseries()
    _, hadoop_map = load_hadoop_timeseries()
    return {
        "HDFS": list(hdfs_map.values()),
        "OpenStack": list(openstack_map.values()),
        "Hadoop": list(hadoop_map.values()),
    }


def _select_representative_hadoop_root(pool: List[str], desired_family: str) -> str:
    preferred_substrings = {
        "HADOOP_MACHINE_DOWN": [
            "Task cleanup failed for attempt",
            "Container released on a *lost* node",
            "Last retry, killing",
            "failures on node",
        ],
        "HADOOP_NETWORK_DISCONNECTION": [
            "Retrying connect to server:",
            "Communication exception:",
            "Failed to connect to /",
            "Address change detected.",
        ],
        "HADOOP_DISK_FULL": [
            "Exception in createBlockOutputStream",
            "Could not find any valid local directory for output",
            "Could not delete hdfs:",
            "Shuffle failed : local error on this node",
        ],
    }
    for needle in preferred_substrings.get(desired_family, []):
        for tpl in pool:
            if desired_family == family_of("Hadoop", tpl) and needle.lower() in tpl.lower():
                return tpl
    for tpl in pool:
        if desired_family == family_of("Hadoop", tpl):
            return tpl
    return ""


def _best_graph_template(
    dataset: str,
    query_texts: List[str],
    pool: List[str],
    desired_family: str = "",
) -> str:
    queries = [str(x or "").strip() for x in query_texts if str(x or "").strip()]
    if not queries:
        return ""
    other_family = {"HDFS": "HDFS_OTHER", "OpenStack": "OS_OTHER", "Hadoop": "HADOOP_UNKNOWN"}[dataset]
    query_token_sets = [set(canonical_tokens(q)) for q in queries]

    best_tpl = ""
    best_score = -1.0
    for tpl in pool:
        fam = family_of(dataset, tpl)
        score = 0.0
        for q, toks in zip(queries, query_token_sets):
            if exact_relaxed_match(tpl, q):
                score = max(score, 1000.0)
            if desired_family and fam == desired_family and fam != other_family:
                score += 120.0
            qfam = family_of(dataset, q)
            if qfam == fam and fam != other_family:
                score += 80.0
            overlap = len(set(canonical_tokens(tpl)) & toks)
            score += overlap * 4.0
            if q.lower() in tpl.lower() or tpl.lower() in q.lower():
                score += 20.0
        if score > best_score:
            best_score = score
            best_tpl = tpl
    return best_tpl


def _extract_target_and_prior_lines(case: Dict[str, object], dataset: str) -> Tuple[str, List[str], str]:
    raw = str(case.get("raw_log", "") or "")
    lines = [x for x in raw.splitlines() if x.strip()]
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


def _last_prior_by_family(dataset: str, prior_lines: List[str], desired_family: str) -> str:
    for line in reversed(prior_lines):
        if family_of(dataset, line) == desired_family:
            return line
    return ""


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


def _window_text(lines: List[str], idx: int, radius: int = OPENSTACK_WINDOW_RADIUS) -> str:
    lo = max(0, idx - radius)
    hi = min(len(lines), idx + radius + 1)
    return "\n".join(line for line in lines[lo:hi] if line.strip())


def _build_openstack_rebalanced_cases() -> List[Dict[str, object]]:
    cases: List[Dict[str, object]] = []
    case_counter = 0
    spec_counts: Counter[str] = Counter()
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
            cases.append(
                {
                    "case_id": f"openstack_rebuild_{case_counter:03d}",
                    "dataset": "OpenStack",
                    "raw_log": _window_text(lines, idx),
                    "source": f"openstack_raw_semantic_rebuild_20260317::{path.name}",
                    "ground_truth_template": str(spec["effect_template"]),
                    "ground_truth_root_cause_template": str(spec["root_template"]),
                }
            )
            case_counter += 1
            spec_counts[str(spec["name"])] += 1
    if len(cases) != 50:
        raise RuntimeError(f"Expected 50 rebuilt OpenStack cases, got {len(cases)}.")
    return cases


def main() -> None:
    args = _parse_args()
    ensure_dirs()
    report_path = REPORTS_DIR / "rq2_fullcase_benchmark_patch_20260316.md"

    if RQ2_FULLCASE_BENCH_PATH.exists() and report_path.exists() and not args.force:
        print(f"[*] Reusing patched benchmark: {RQ2_FULLCASE_BENCH_PATH}")
        print(f"[*] Reusing benchmark patch report: {report_path}")
        return

    rows = json.loads(BENCH_V2_PATH.read_text(encoding="utf-8"))
    template_pool = _domain_template_pool()
    openstack_rebuilt = _build_openstack_rebalanced_cases()
    openstack_by_case = {str(row["case_id"]): row for row in openstack_rebuilt}
    patched: List[Dict[str, object]] = []
    stats_before = Counter()
    stats_after = Counter()
    graph_effect_stats = Counter()
    graph_root_stats = Counter()
    changed = 0
    openstack_replaced = 0

    for row in rows:
        dataset = str(row.get("dataset", "") or "")
        if dataset == "OpenStack":
            continue
        row2 = dict(row)
        gt_root = str(row.get("ground_truth_root_cause_template", "") or "")
        stats_before[(dataset, gt_root)] += 1
        if dataset == "HDFS":
            patched_root = _patch_hdfs_root(row2)
            if patched_root and patched_root != gt_root:
                row2["ground_truth_root_cause_template"] = patched_root
                changed += 1
        row2["ground_truth_template_label"] = str(row.get("ground_truth_template", "") or "")
        row2["ground_truth_root_cause_label"] = str(row.get("ground_truth_root_cause_template", "") or "")

        target_line, prior_lines, target_tpl = _extract_target_and_prior_lines(row2, dataset)
        desired_root_family = family_of(dataset, str(row2.get("ground_truth_root_cause_template", "") or ""))
        prior_root_line = _last_prior_by_family(dataset, prior_lines, desired_root_family)

        graph_effect = _best_graph_template(
            dataset,
            [str(row2.get("ground_truth_template", "") or ""), target_tpl, target_line],
            template_pool[dataset],
        )
        if dataset == "Hadoop" and desired_root_family in {
            "HADOOP_MACHINE_DOWN",
            "HADOOP_NETWORK_DISCONNECTION",
            "HADOOP_DISK_FULL",
        }:
            graph_root = _select_representative_hadoop_root(template_pool[dataset], desired_root_family)
        else:
            graph_root = _best_graph_template(
                dataset,
                [
                    str(row2.get("ground_truth_root_cause_template", "") or ""),
                    prior_root_line,
                ],
                template_pool[dataset],
                desired_family=desired_root_family,
            )
        if graph_effect:
            row2["ground_truth_template_graph"] = graph_effect
            graph_effect_stats[(dataset, graph_effect)] += 1
        if graph_root:
            row2["ground_truth_root_cause_template_graph"] = graph_root
            graph_root_stats[(dataset, graph_root)] += 1
        stats_after[(dataset, str(row2.get("ground_truth_root_cause_template", "") or ""))] += 1
        patched.append(row2)

    for row in openstack_rebuilt:
        dataset = "OpenStack"
        row2 = dict(row)
        gt_root = str(row2.get("ground_truth_root_cause_template", "") or "")
        stats_before[(dataset, gt_root)] += 1
        row2["ground_truth_template_label"] = str(row2.get("ground_truth_template", "") or "")
        row2["ground_truth_root_cause_label"] = str(row2.get("ground_truth_root_cause_template", "") or "")
        row2["ground_truth_template_graph"] = str(row2.get("ground_truth_template", "") or "")
        row2["ground_truth_root_cause_template_graph"] = str(row2.get("ground_truth_root_cause_template", "") or "")
        graph_effect_stats[(dataset, str(row2["ground_truth_template_graph"]))] += 1
        graph_root_stats[(dataset, str(row2["ground_truth_root_cause_template_graph"]))] += 1
        stats_after[(dataset, gt_root)] += 1
        patched.append(row2)
        openstack_replaced += 1

    RQ2_FULLCASE_BENCH_PATH.write_text(
        json.dumps(patched, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    report_lines = [
        "# RQ2 Fullcase Benchmark Patch (2026-03-16)",
        "",
        f"- Source benchmark: `{BENCH_V2_PATH}`",
        f"- Patched benchmark: `{RQ2_FULLCASE_BENCH_PATH}`",
        f"- HDFS roots changed: `{changed}`",
        f"- OpenStack rebuilt cases: `{openstack_replaced}`",
        "",
        "## HDFS root distribution after patch",
        "",
    ]
    for (dataset, root), count in sorted(stats_after.items(), key=lambda kv: (kv[0][0], -kv[1], kv[0][1])):
        if dataset != "HDFS":
            continue
        report_lines.append(f"- `{root or '<empty>'}`: `{count}`")

    for dataset in ("OpenStack", "Hadoop"):
        report_lines.extend(
            [
                "",
                f"## {dataset} graph-space effect templates",
                "",
            ]
        )
        for (ds, effect), count in sorted(graph_effect_stats.items(), key=lambda kv: (kv[0][0], -kv[1], kv[0][1])):
            if ds != dataset:
                continue
            report_lines.append(f"- `{effect}`: `{count}`")
        report_lines.extend(
            [
                "",
                f"## {dataset} graph-space root templates",
                "",
            ]
        )
        for (ds, root), count in sorted(graph_root_stats.items(), key=lambda kv: (kv[0][0], -kv[1], kv[0][1])):
            if ds != dataset:
                continue
            report_lines.append(f"- `{root}`: `{count}`")

    report_lines.extend(
        [
            "",
            "## OpenStack rebuilt case specs",
            "",
        ]
    )
    for spec in OPENSTACK_CASE_SPECS:
        report_lines.append(
            f"- `{spec['name']}`: keyword=`{spec['keyword']}` count=`{spec['count']}` "
            f"effect=`{spec['effect_template']}` root=`{spec['root_template']}`"
        )

    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(f"[Saved] {RQ2_FULLCASE_BENCH_PATH}")
    print(f"[Saved] {report_path}")


if __name__ == "__main__":
    main()
