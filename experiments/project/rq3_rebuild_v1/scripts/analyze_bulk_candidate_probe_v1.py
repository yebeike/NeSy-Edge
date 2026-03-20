from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Tuple


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--progress", type=Path, required=True)
    ap.add_argument("--spec", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--target-noise", type=float, default=1.0)
    ap.add_argument("--policy", type=str, default="strict", choices=["strict", "calibrated", "mid_relaxed"])
    return ap.parse_args()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def as_bool(value: object) -> int:
    return int(bool(value))


def mode_metric(metrics: Mapping[str, Mapping[str, object]], mode: str, key: str) -> int | None:
    if mode not in metrics:
        return None
    return as_bool(metrics.get(mode, {}).get(key))


def case_score_strict(metrics: Mapping[str, Mapping[str, object]]) -> Tuple[int, List[str]]:
    reasons: List[str] = []
    score = 0

    heuristic_alert_e2e = mode_metric(metrics, "heuristic_alert", "e2e_success")
    open_alert_only_e2e = mode_metric(metrics, "open_alert_only", "e2e_success")
    open_alert_only_rca = mode_metric(metrics, "open_alert_only", "rca_success")
    vanilla_e2e = mode_metric(metrics, "vanilla_open", "e2e_success")
    rag_e2e = mode_metric(metrics, "rag_open", "e2e_success")
    agent_e2e = mode_metric(metrics, "agent_open", "e2e_success")
    rag_rca = mode_metric(metrics, "rag_open", "rca_success")
    agent_rca = mode_metric(metrics, "agent_open", "rca_success")

    if heuristic_alert_e2e is not None and heuristic_alert_e2e:
        score -= 4
        reasons.append("heuristic_alert_solves_case")
    elif heuristic_alert_e2e is not None:
        score += 2
        reasons.append("heuristic_alert_not_direct")

    if open_alert_only_e2e is not None and open_alert_only_e2e:
        score -= 3
        reasons.append("open_alert_only_solves_case")
    elif open_alert_only_e2e is not None:
        score += 2
        reasons.append("open_alert_only_not_e2e")

    if open_alert_only_rca is not None and open_alert_only_e2e is not None and open_alert_only_rca and not open_alert_only_e2e:
        score += 1
        reasons.append("open_alert_only_rca_without_e2e")

    if rag_e2e is not None and vanilla_e2e is not None and rag_e2e > vanilla_e2e:
        score += 2
        reasons.append("rag_beats_vanilla")
    elif rag_e2e is not None and vanilla_e2e is not None and rag_e2e < vanilla_e2e:
        score -= 2
        reasons.append("rag_worse_than_vanilla")

    if agent_e2e is not None and rag_e2e is not None and agent_e2e > rag_e2e:
        score += 3
        reasons.append("agent_beats_rag")
    elif agent_e2e is not None and rag_e2e is not None and agent_e2e < rag_e2e:
        score -= 2
        reasons.append("agent_worse_than_rag")

    if agent_rca is not None and agent_e2e is not None and agent_rca and not agent_e2e:
        score += 1
        reasons.append("agent_rca_e2e_decoupled")
    if rag_rca is not None and rag_e2e is not None and rag_rca and not rag_e2e:
        score += 1
        reasons.append("rag_rca_e2e_decoupled")

    llm_values = [value for value in (vanilla_e2e, rag_e2e, agent_e2e) if value is not None]
    if len(llm_values) == 3 and vanilla_e2e == rag_e2e == agent_e2e:
        score -= 2
        reasons.append("all_llm_e2e_flat")
    if llm_values and max(llm_values) == 0:
        score -= 3
        reasons.append("all_llm_e2e_zero")
    elif llm_values:
        score += 1
        reasons.append("some_llm_support_present")

    return score, reasons


def case_score_calibrated(metrics: Mapping[str, Mapping[str, object]]) -> Tuple[int, List[str]]:
    reasons: List[str] = []
    score = 0

    heuristic_alert_e2e = mode_metric(metrics, "heuristic_alert", "e2e_success")
    open_alert_only_e2e = mode_metric(metrics, "open_alert_only", "e2e_success")
    vanilla_e2e = mode_metric(metrics, "vanilla_open", "e2e_success")
    rag_e2e = mode_metric(metrics, "rag_open", "e2e_success")
    agent_e2e = mode_metric(metrics, "agent_open", "e2e_success")
    rag_rca = mode_metric(metrics, "rag_open", "rca_success")
    agent_rca = mode_metric(metrics, "agent_open", "rca_success")

    if heuristic_alert_e2e is not None and heuristic_alert_e2e:
        score -= 1
        reasons.append("heuristic_alert_easy")
    elif heuristic_alert_e2e is not None:
        score += 2
        reasons.append("heuristic_alert_not_direct")

    if open_alert_only_e2e is not None and open_alert_only_e2e:
        score -= 4
        reasons.append("open_alert_only_solves_case")
    elif open_alert_only_e2e is not None:
        score += 2
        reasons.append("open_alert_only_not_e2e")

    if rag_e2e is not None and vanilla_e2e is not None and rag_e2e > vanilla_e2e:
        score += 2
        reasons.append("rag_beats_vanilla")
    elif rag_e2e is not None and vanilla_e2e is not None and rag_e2e < vanilla_e2e:
        score -= 1
        reasons.append("rag_worse_than_vanilla")

    if agent_e2e is not None and rag_e2e is not None and agent_e2e > rag_e2e:
        score += 2
        reasons.append("agent_beats_rag")
    elif agent_e2e is not None and rag_e2e is not None and agent_e2e < rag_e2e:
        score -= 1
        reasons.append("agent_worse_than_rag")

    if agent_rca is not None and agent_e2e is not None and agent_rca and not agent_e2e:
        score += 1
        reasons.append("agent_rca_e2e_decoupled")
    if rag_rca is not None and rag_e2e is not None and rag_rca and not rag_e2e:
        score += 1
        reasons.append("rag_rca_e2e_decoupled")

    llm_values = [value for value in (vanilla_e2e, rag_e2e, agent_e2e) if value is not None]
    if llm_values and max(llm_values) == 0:
        score -= 3
        reasons.append("all_llm_e2e_zero")
    elif llm_values:
        score += 1
        reasons.append("some_llm_support_present")

    if len(llm_values) == 3 and vanilla_e2e == rag_e2e == agent_e2e == 1:
        score -= 1
        reasons.append("all_llm_e2e_flat_positive")

    return score, reasons


def case_score_mid_relaxed(metrics: Mapping[str, Mapping[str, object]]) -> Tuple[int, List[str]]:
    reasons: List[str] = []
    score = 0

    heuristic_alert_e2e = mode_metric(metrics, "heuristic_alert", "e2e_success")
    open_alert_only_e2e = mode_metric(metrics, "open_alert_only", "e2e_success")
    vanilla_e2e = mode_metric(metrics, "vanilla_open", "e2e_success")
    rag_e2e = mode_metric(metrics, "rag_open", "e2e_success")
    agent_e2e = mode_metric(metrics, "agent_open", "e2e_success")
    rag_rca = mode_metric(metrics, "rag_open", "rca_success")
    agent_rca = mode_metric(metrics, "agent_open", "rca_success")

    if heuristic_alert_e2e is not None and not heuristic_alert_e2e:
        score += 1
        reasons.append("heuristic_alert_not_direct")
    elif heuristic_alert_e2e:
        reasons.append("heuristic_alert_easy")

    if open_alert_only_e2e is not None and open_alert_only_e2e:
        score -= 2
        reasons.append("open_alert_only_easy")
    elif open_alert_only_e2e is not None:
        score += 2
        reasons.append("open_alert_only_not_e2e")

    if rag_e2e is not None and vanilla_e2e is not None and rag_e2e > vanilla_e2e:
        score += 1
        reasons.append("rag_beats_vanilla")
    elif rag_e2e is not None and vanilla_e2e is not None and rag_e2e < vanilla_e2e:
        reasons.append("rag_worse_than_vanilla")

    if agent_e2e is not None and rag_e2e is not None and agent_e2e > rag_e2e:
        score += 1
        reasons.append("agent_beats_rag")
    elif agent_e2e is not None and rag_e2e is not None and agent_e2e < rag_e2e:
        reasons.append("agent_worse_than_rag")

    if agent_rca is not None and agent_e2e is not None and agent_rca and not agent_e2e:
        score += 1
        reasons.append("agent_rca_e2e_decoupled")
    if rag_rca is not None and rag_e2e is not None and rag_rca and not rag_e2e:
        score += 1
        reasons.append("rag_rca_e2e_decoupled")

    llm_values = [value for value in (vanilla_e2e, rag_e2e, agent_e2e) if value is not None]
    if llm_values and max(llm_values) == 0:
        score -= 2
        reasons.append("all_llm_e2e_zero")
    elif llm_values:
        score += 1
        reasons.append("some_llm_support_present")

    if len(llm_values) == 3 and vanilla_e2e == rag_e2e == agent_e2e == 1:
        score -= 2
        reasons.append("all_llm_e2e_flat_positive")

    return score, reasons


def case_score(metrics: Mapping[str, Mapping[str, object]], policy: str) -> Tuple[int, List[str]]:
    if policy == "calibrated":
        return case_score_calibrated(metrics)
    if policy == "mid_relaxed":
        return case_score_mid_relaxed(metrics)
    return case_score_strict(metrics)


def verdict_for_score(score: int, metrics: Mapping[str, Mapping[str, object]], policy: str) -> str:
    required_modes = {
        "heuristic_alert",
        "heuristic_context",
        "open_alert_only",
        "vanilla_open",
        "rag_open",
        "agent_open",
    }
    if not required_modes.issubset(set(metrics.keys())):
        return "incomplete"
    heuristic_alert_e2e = mode_metric(metrics, "heuristic_alert", "e2e_success")
    open_alert_only_e2e = mode_metric(metrics, "open_alert_only", "e2e_success")
    vanilla_e2e = mode_metric(metrics, "vanilla_open", "e2e_success")
    agent_e2e = mode_metric(metrics, "agent_open", "e2e_success")
    rag_e2e = mode_metric(metrics, "rag_open", "e2e_success")
    if policy == "strict":
        if heuristic_alert_e2e or open_alert_only_e2e:
            return "reject_direct"
        if agent_e2e is not None and agent_e2e and score >= 5:
            return "shortlist"
        if ((agent_e2e is not None and agent_e2e) or (rag_e2e is not None and rag_e2e)) and score >= 2:
            return "provisional"
        return "reject"
    if policy == "calibrated":
        if open_alert_only_e2e:
            return "reject_direct"
        if heuristic_alert_e2e and vanilla_e2e and rag_e2e and agent_e2e:
            return "reject_direct"
        if ((agent_e2e is not None and agent_e2e) or (rag_e2e is not None and rag_e2e)) and score >= 4:
            return "shortlist"
        if ((agent_e2e is not None and agent_e2e) or (rag_e2e is not None and rag_e2e) or (vanilla_e2e is not None and vanilla_e2e)) and score >= 1:
            return "provisional"
        return "reject"
    if open_alert_only_e2e and vanilla_e2e and agent_e2e:
        return "reject_direct"
    if heuristic_alert_e2e and open_alert_only_e2e and vanilla_e2e and rag_e2e and agent_e2e:
        return "reject_direct"
    if ((agent_e2e is not None and agent_e2e) or (rag_e2e is not None and rag_e2e) or (vanilla_e2e is not None and vanilla_e2e)) and score >= 2:
        return "shortlist"
    if (
        (agent_e2e is not None and agent_e2e)
        or (rag_e2e is not None and rag_e2e)
        or (vanilla_e2e is not None and vanilla_e2e)
        or any(
            mode_metric(metrics, mode, "rca_success") and not mode_metric(metrics, mode, "e2e_success")
            for mode in ("vanilla_open", "rag_open", "agent_open")
        )
    ) and score >= 0:
        return "provisional"
    return "reject"


def main() -> None:
    args = parse_args()
    rows = [json.loads(line) for line in args.progress.read_text(encoding="utf-8").splitlines() if line.strip()]
    spec = load_json(args.spec)
    noise_key = f"{float(args.target_noise):.1f}"
    rows = [row for row in rows if f"{float(row['noise']):.1f}" == noise_key]

    spec_meta: Dict[Tuple[str, str, str], Mapping[str, object]] = {}
    for dataset, items in spec["datasets"].items():
        for item in items:
            eval_case_id = str(item.get("eval_case_id", "") or item["case_id"])
            spec_meta[(str(dataset), str(item["source"]), eval_case_id)] = item

    by_case: Dict[Tuple[str, str], Dict[str, Mapping[str, object]]] = defaultdict(dict)
    source_by_case: Dict[Tuple[str, str], str] = {}
    gt_action_by_case: Dict[Tuple[str, str], str] = {}
    for row in rows:
        key = (str(row["dataset"]), str(row["case_id"]))
        by_case[key][str(row["mode"])] = row
        gt_action_by_case[key] = str(row["gt_action_id"])

    ranked: List[Dict[str, object]] = []
    for (dataset, case_id), metrics in sorted(by_case.items()):
        source = ""
        item = None
        for (ds, maybe_source, cid), meta in spec_meta.items():
            if ds == dataset and cid == case_id:
                source = maybe_source
                item = meta
                break
        score, reasons = case_score(metrics, args.policy)
        verdict = verdict_for_score(score, metrics, args.policy)
        ranked.append(
            {
                "dataset": dataset,
                "case_id": case_id,
                "eval_case_id": str((item or {}).get("eval_case_id", case_id)),
                "source": source,
                "gt_action_id": gt_action_by_case[(dataset, case_id)],
                "score": score,
                "verdict": verdict,
                "complete_case": verdict != "incomplete",
                "reasons": reasons,
                "selection_score": float((item or {}).get("selection_score", 0.0) or 0.0),
                "difficulty_score": float((item or {}).get("difficulty_score", 0.0) or 0.0),
                "weak_mainline_alert": bool((item or {}).get("weak_mainline_alert", False)),
                "selected_alert_flags": dict((item or {}).get("selected_alert_flags", {}) or {}),
                "eligibility_note": str((item or {}).get("eligibility_note", "")),
                "modes": {
                    mode: {
                        "rca_success": as_bool(row.get("rca_success")),
                        "e2e_success": as_bool(row.get("e2e_success")),
                        "pred_family_id": str(row.get("pred_family_id", "") or ""),
                        "pred_action_id": str(row.get("pred_action_id", "") or ""),
                    }
                    for mode, row in metrics.items()
                },
            }
        )

    ranked.sort(
        key=lambda row: (
            {"shortlist": 0, "provisional": 1, "reject": 2, "reject_direct": 3, "incomplete": 4}.get(str(row["verdict"]), 9),
            -int(row["score"]),
            str(row["gt_action_id"]),
            -float(row["selection_score"]),
            str(row["source"]),
            str(row["case_id"]),
        )
    )

    by_action: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    verdict_counts: Dict[str, int] = defaultdict(int)
    for row in ranked:
        by_action[str(row["gt_action_id"])].append(row)
        verdict_counts[str(row["verdict"])] += 1

    shortlist_by_action = {
        action_id: rows[:5]
        for action_id, rows in sorted(by_action.items())
    }
    payload = {
        "policy": args.policy,
        "target_noise": float(args.target_noise),
        "rows": len(ranked),
        "verdict_counts": dict(sorted(verdict_counts.items())),
        "ranked_cases": ranked,
        "shortlist_by_action": shortlist_by_action,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(args.output)
    print(json.dumps(payload["verdict_counts"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
