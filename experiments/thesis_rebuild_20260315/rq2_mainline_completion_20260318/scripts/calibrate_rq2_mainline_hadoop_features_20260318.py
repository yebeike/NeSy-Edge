from __future__ import annotations

import argparse

from rq2_mainline_completion_common_20260318 import (
    HADOOP_CALIBRATION_PATH,
    HADOOP_CAP_CANDIDATES,
    HADOOP_ORIGINAL_TIMEOUT_SEC,
    HADOOP_PC_TIMEOUT_SEC,
    REPORTS_DIR,
    Heartbeat,
    ensure_dirs,
    prepare_feature_space,
    run_hadoop_measurement_subprocess,
    write_json,
)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    ensure_dirs()
    report_path = REPORTS_DIR / "rq2_mainline_hadoop_calibration_20260318.md"
    if HADOOP_CALIBRATION_PATH.exists() and report_path.exists() and not args.force:
        print(f"[*] Reusing Hadoop calibration artifact: {HADOOP_CALIBRATION_PATH}")
        print(f"[*] Reusing calibration report: {report_path}")
        return

    print("[1/2] Calibrating Hadoop shared feature cap. Expected: 1-15 min")
    trials = []
    chosen_cap = HADOOP_CAP_CANDIDATES[-1]
    chosen_columns = []
    for idx, cap in enumerate(HADOOP_CAP_CANDIDATES, start=1):
        print(
            f"[1/2.{idx}] Testing Hadoop cap={cap} with shared feature space. "
            f"Timeouts: original {HADOOP_ORIGINAL_TIMEOUT_SEC}s, pc {HADOOP_PC_TIMEOUT_SEC}s"
        )
        with Heartbeat(f"Hadoop/original_dynotears cap={cap}", interval_sec=30, remaining="pc_cpdag_hypothesis"):
            original = run_hadoop_measurement_subprocess("original_dynotears", cap, HADOOP_ORIGINAL_TIMEOUT_SEC)
        with Heartbeat(f"Hadoop/pc_cpdag_hypothesis cap={cap}", interval_sec=30, remaining="trial decision"):
            pc = run_hadoop_measurement_subprocess("pc_cpdag_hypothesis", cap, HADOOP_PC_TIMEOUT_SEC)
        accepted = not bool(original["timed_out"]) and not bool(pc["timed_out"])
        trial = {
            "cap": cap,
            "original_dynotears": original,
            "pc_cpdag_hypothesis": pc,
            "accepted": accepted,
        }
        trials.append(trial)
        if accepted:
            chosen_cap = cap
            break

    df, _, profile = prepare_feature_space("Hadoop", max_cols=chosen_cap)
    chosen_columns = list(profile["column_ids"])
    payload = {
        "dataset": "Hadoop",
        "candidate_caps": HADOOP_CAP_CANDIDATES,
        "timeouts_sec": {
            "original_dynotears": HADOOP_ORIGINAL_TIMEOUT_SEC,
            "pc_cpdag_hypothesis": HADOOP_PC_TIMEOUT_SEC,
        },
        "trials": trials,
        "chosen_cap": chosen_cap,
        "column_ids": chosen_columns,
        "selected_columns": int(profile["selected_columns"]),
    }

    print("[2/2] Writing calibration artifact and report. Expected: <5s")
    write_json(HADOOP_CALIBRATION_PATH, payload)
    lines = [
        "# RQ2 Mainline Hadoop Feature Calibration (2026-03-18)",
        "",
        f"- Chosen cap: `{chosen_cap}`",
        f"- Selected columns: `{len(chosen_columns)}`",
        "- The chosen cap is the highest candidate that finished both original_dynotears and pc_cpdag_hypothesis within the configured timeouts.",
        "",
        "## Trials",
        "",
        "| Cap | Original Timed Out | Original Elapsed | PC Timed Out | PC Elapsed | Accepted |",
        "|---:|---|---:|---|---:|---|",
    ]
    for trial in trials:
        orig = trial["original_dynotears"]
        pc = trial["pc_cpdag_hypothesis"]
        lines.append(
            f"| {trial['cap']} | {orig['timed_out']} | {orig['elapsed_sec']} | {pc['timed_out']} | {pc['elapsed_sec']} | {trial['accepted']} |"
        )

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[Saved] {HADOOP_CALIBRATION_PATH}")
    print(f"[Saved] {report_path}")


if __name__ == "__main__":
    main()
