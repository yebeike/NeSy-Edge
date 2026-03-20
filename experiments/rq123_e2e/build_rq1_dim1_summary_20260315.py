"""
Build the thesis-facing RQ1 Dim1 summary table.

Sources:
- HDFS/OpenStack PA from results/rq123_e2e/stage2_midscale_dim1_20260311.csv
- Hadoop PA from results/rq1_hadoop/rq1_hadoop_smallbatch_dim1_20260311.csv
- Optional latency merge from results/rq123_e2e/stage2_dim1_latency_probe_20260315.csv
"""

import os
from typing import List

import pandas as pd


ROOT = "/Users/peihanye/Desktop/Projects/NuSy-Edge"
MID_CSV = os.path.join(ROOT, "results", "rq123_e2e", "stage2_midscale_dim1_20260311.csv")
HADOOP_CSV = os.path.join(ROOT, "results", "rq1_hadoop", "rq1_hadoop_smallbatch_dim1_20260311.csv")
LAT_CSV = os.path.join(ROOT, "results", "rq123_e2e", "stage2_dim1_latency_probe_20260315.csv")
OUT_DIR = os.path.join(ROOT, "results", "thesis")
OUT_CSV = os.path.join(OUT_DIR, "rq1_dim1_summary_20260315.csv")


METHOD_MAP = {
    "pa_nusy": "NuSy",
    "pa_drain": "Drain",
    "pa_qwen": "Qwen",
}


def _from_midscale(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    out: List[dict] = []
    for dataset in ["HDFS", "OpenStack"]:
        sub = df[df["dataset"] == dataset]
        if sub.empty:
            continue
        grouped = sub.groupby("noise")[["pa_nusy", "pa_drain", "pa_qwen"]].mean().reset_index()
        for _, row in grouped.iterrows():
            for col, method in METHOD_MAP.items():
                out.append(
                    {
                        "dataset": dataset,
                        "noise": float(row["noise"]),
                        "method": method,
                        "pa": round(float(row[col]), 3),
                    }
                )
    return pd.DataFrame(out)


def _from_hadoop(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    grouped = df.groupby("noise")[["pa_nusy", "pa_drain", "pa_qwen"]].mean().reset_index()
    out: List[dict] = []
    for _, row in grouped.iterrows():
        for col, method in METHOD_MAP.items():
            out.append(
                {
                    "dataset": "Hadoop",
                    "noise": float(row["noise"]),
                    "method": method,
                    "pa": round(float(row[col]), 3),
                }
            )
    return pd.DataFrame(out)


def main() -> str:
    os.makedirs(OUT_DIR, exist_ok=True)

    parts = [_from_midscale(MID_CSV), _from_hadoop(HADOOP_CSV)]
    summary = pd.concat(parts, ignore_index=True)

    if os.path.exists(LAT_CSV):
        lat = pd.read_csv(LAT_CSV)
        lat = (
            lat.groupby(["dataset", "noise", "method"])["latency_ms"]
            .mean()
            .reset_index()
        )
        lat["latency_ms"] = lat["latency_ms"].round(3)
        summary = summary.merge(lat, on=["dataset", "noise", "method"], how="left")

    summary = summary.sort_values(["dataset", "noise", "method"]).reset_index(drop=True)
    summary.to_csv(OUT_CSV, index=False)

    print(summary.to_string(index=False))
    print(f"\n[Saved] {OUT_CSV}")
    return OUT_CSV


if __name__ == "__main__":
    main()
