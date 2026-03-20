import os
import sys
import subprocess
import statistics
import pandas as pd


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
STEP2_SCRIPT = os.path.join(PROJECT_ROOT, "experiments", "run_rq2_step2_causal_analysis.py")


def run_once(seed: int):
    env = os.environ.copy()
    env["RQ2_HDFS_SEED"] = str(seed)
    env["PYTHONPATH"] = PROJECT_ROOT
    res = subprocess.run(
        [sys.executable, STEP2_SCRIPT],
        cwd=PROJECT_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=600,
    )
    if res.returncode != 0:
        print(f"[WARN] Seed {seed} run failed: {res.stderr[:200] or res.stdout[:200]}")
        return None
    summary_path = os.path.join(RESULTS_DIR, "rq2_hdfs_step2_summary.csv")
    if not os.path.exists(summary_path):
        print(f"[WARN] Seed {seed}: summary CSV not found at {summary_path}")
        return None
    df = pd.read_csv(summary_path)
    return df


def main():
    seeds = list(range(42, 52))  # 10 seeds: 42..51
    print(f"[INFO] Monte Carlo HDFS Step2 over seeds: {seeds}")

    # Collect metrics per (algorithm, type)
    per_alg = {}  # (alg, type) -> list of dicts
    nu_rank_list = []
    pe_rank_list = []
    fp_pearson_list = []
    fp_lagged_list = []
    fp_nusy_list = []
    conf_pairs_list = []

    for s in seeds:
        print(f"\n[INFO] Running HDFS Step2 with seed={s} ...")
        df = run_once(s)
        if df is None:
            continue

        # record per-alg metrics
        for _, row in df.iterrows():
            key = (row["algorithm"], row["type"])
            per_alg.setdefault(key, []).append(
                {
                    "recall": float(row["recall"]),
                    "precision": float(row["precision"]),
                    "f1": float(row["f1"]),
                    "edges": float(row["edges"]),
                }
            )

        # only need one row to read ranks / FP（各行相同）
        r0 = df.iloc[0]
        nu_rank_list.append(float(r0.get("nusy_intra_rank", -1)))
        pe_rank_list.append(float(r0.get("pearson_intra_rank", -1)))
        fp_pearson_list.append(float(r0.get("pearson_fp_conf", -1)))
        fp_lagged_list.append(float(r0.get("lagged_fp_conf", -1)))
        fp_nusy_list.append(float(r0.get("nusy_fp_conf", -1)))
        conf_pairs_list.append(float(r0.get("conf_pairs", -1)))

    if not per_alg:
        print("[ERROR] No successful runs; aborting MC summary.")
        return

    rows = []
    print("\n=== HDFS Step2 Monte Carlo Summary (10 seeds) ===")
    print(f"{'Algorithm':<15} | {'Type':<8} | {'F1_mean':<8} {'F1_std':<8} | {'Edges_mean':<10}")
    print("-" * 70)

    for (alg, t), lst in per_alg.items():
        f1s = [x["f1"] for x in lst]
        edges = [x["edges"] for x in lst]
        f1_mean = statistics.fmean(f1s)
        f1_std = statistics.pstdev(f1s) if len(f1s) > 1 else 0.0
        e_mean = statistics.fmean(edges)
        print(f"{alg:<15} | {t:<8} | {f1_mean:<8.4f} {f1_std:<8.4f} | {e_mean:<10.2f}")
        rows.append(
            {
                "algorithm": alg,
                "type": t,
                "f1_mean": f1_mean,
                "f1_std": f1_std,
                "edges_mean": e_mean,
            }
        )

    # Confounder & rank averages
    def m_and_s(vs):
        """Return (mean, std) for a list of floats, std as population std."""
        if not vs:
            return 0.0, 0.0
        m = statistics.fmean(vs)
        if len(vs) <= 1:
            return m, 0.0
        var = sum((x - m) ** 2 for x in vs) / len(vs)
        return m, var ** 0.5

    # 平均 Rank
    nu_rank_mean, nu_rank_std = m_and_s(nu_rank_list)
    pe_rank_mean, pe_rank_std = m_and_s(pe_rank_list)
    # 平均 FP 率（按对数归一）
    fp_pairs = [c if c > 0 else 1.0 for c in conf_pairs_list]
    pearson_fp_rate = [fp_pearson_list[i] / fp_pairs[i] for i in range(len(fp_pairs))]
    lagged_fp_rate = [fp_lagged_list[i] / fp_pairs[i] for i in range(len(fp_pairs))]
    nusy_fp_rate = [fp_nusy_list[i] / fp_pairs[i] for i in range(len(fp_pairs))]
    pearson_fp_mean, pearson_fp_std = m_and_s(pearson_fp_rate)
    lagged_fp_mean, lagged_fp_std = m_and_s(lagged_fp_rate)
    nusy_fp_mean, nusy_fp_std = m_and_s(nusy_fp_rate)

    print("\n--- Intra Rank (Monte Carlo) ---")
    print(f"NuSy-Edge Rank: mean={nu_rank_mean:.2f}, std={nu_rank_std:.2f}")
    print(f"Pearson Rank  : mean={pe_rank_mean:.2f}, std={pe_rank_std:.2f}")

    print("\n--- Confounder FP Rate (Monte Carlo, per pair) ---")
    print(f"Pearson FP rate  : mean={pearson_fp_mean:.2f}, std={pearson_fp_std:.2f}")
    print(f"Lagged FP rate   : mean={lagged_fp_mean:.2f}, std={lagged_fp_std:.2f}")
    print(f"NuSy-Edge FP rate: mean={nusy_fp_mean:.2f}, std={nusy_fp_std:.2f}")

    # 写 MC 汇总 CSV
    mc_path = os.path.join(RESULTS_DIR, "rq2_hdfs_step2_mc_summary.csv")
    extra = pd.DataFrame(
        [
            {
                "metric": "NuSy_intra_rank",
                "mean": nu_rank_mean,
                "std": nu_rank_std,
            },
            {
                "metric": "Pearson_intra_rank",
                "mean": pe_rank_mean,
                "std": pe_rank_std,
            },
            {
                "metric": "Pearson_conf_fp_rate",
                "mean": pearson_fp_mean,
                "std": pearson_fp_std,
            },
            {
                "metric": "Lagged_conf_fp_rate",
                "mean": lagged_fp_mean,
                "std": lagged_fp_std,
            },
            {
                "metric": "NuSy_conf_fp_rate",
                "mean": nusy_fp_mean,
                "std": nusy_fp_std,
            },
        ]
    )
    out_df = pd.DataFrame(rows)
    out_df.to_csv(mc_path, index=False)
    # 附加 extra 信息作为单独文件
    extra_path = os.path.join(RESULTS_DIR, "rq2_hdfs_step2_mc_extra.csv")
    extra.to_csv(extra_path, index=False)

    print(f"\n[INFO] HDFS Step2 MC summary written to {mc_path}")
    print(f"[INFO] HDFS Step2 MC extra metrics written to {extra_path}")


if __name__ == "__main__":
    main()

