import os
import sys
import subprocess
import statistics
import pandas as pd


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
STEP2_SCRIPT = os.path.join(PROJECT_ROOT, "experiments", "run_rq2_step2_causal_analysis.py")


def run_once(seed: int, lambda_w: float, w_th: float):
    env = os.environ.copy()
    env["RQ2_HDFS_SEED"] = str(seed)
    env["RQ2_HDFS_LAMBDA_W"] = str(lambda_w)
    env["RQ2_HDFS_W_THRESHOLD"] = str(w_th)
    # For hyperparameter search we cut Monte Carlo runs to 5 to keep
    # total runtime manageable; final reporting will still use 10 runs.
    env["RQ2_HDFS_NUM_RUNS"] = "5"
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
        print(f"[WARN] Seed {seed}, (lambda_w={lambda_w}, th={w_th}) failed: {res.stderr[:200] or res.stdout[:200]}")
        return None

    summary_path = os.path.join(RESULTS_DIR, "rq2_hdfs_step2_summary.csv")
    if not os.path.exists(summary_path):
        print(f"[WARN] Seed {seed}: summary CSV not found at {summary_path}")
        return None
    return pd.read_csv(summary_path)


def mean_std(vs):
    if not vs:
        return 0.0, 0.0
    m = statistics.fmean(vs)
    if len(vs) <= 1:
        return m, 0.0
    var = sum((x - m) ** 2 for x in vs) / len(vs)
    return m, var ** 0.5


def main():
    # Grid for lambda_w and W_THRESHOLD (HDFS)
    # NOTE: user-specified search space focusing on smaller regularisation
    lambdas = [0.001, 0.005, 0.01]
    thresholds = [0.05, 0.10, 0.20]
    # Each config: a few different random seeds (kept small for efficiency)
    seeds = list(range(9500, 9503))

    grid_rows = []

    print(f"[INFO] HDFS Step2 Grid Search over lambda_w={lambdas}, W_THRESHOLD={thresholds}, seeds={seeds}")

    for lw in lambdas:
        for th in thresholds:
            print(f"\n=== Config: lambda_w={lw}, W_THRESHOLD={th} ===")
            hits = 0
            ranks_hit = []
            nusy_fp_rates = []
            pearson_fp_rates = []
            all_f1 = []

            for s in seeds:
                print(f"[INFO]  - seed={s}")
                df = run_once(s, lw, th)
                if df is None:
                    continue

                # NuSy-Edge Intra row
                row_nu = df[(df["algorithm"] == "NuSy-Edge") & (df["type"] == "Intra")]
                if row_nu.empty:
                    print(f"[WARN]   seed={s}: no NuSy-Edge Intra row")
                    continue
                r = row_nu.iloc[0]

                f1 = float(r["f1"])
                rec = float(r["recall"])
                rank = float(r.get("nusy_intra_rank", -1))
                all_f1.append(f1)

                # Hit 定义：有召回且 F1>0
                hit = (rec > 0.0) and (f1 > 0.0)
                if hit:
                    hits += 1
                    ranks_hit.append(rank)

                # Confounder FP rate（按对数归一）
                conf_pairs = float(r.get("conf_pairs", 0.0))
                if conf_pairs <= 0:
                    continue
                pearson_fp = float(r.get("pearson_fp_conf", 0.0))
                nusy_fp = float(r.get("nusy_fp_conf", 0.0))
                pearson_fp_rates.append(pearson_fp / conf_pairs)
                nusy_fp_rates.append(nusy_fp / conf_pairs)

            if not all_f1:
                print("[WARN]   no valid runs for this config, skipping")
                continue

            hit_rate = hits / len(seeds)
            rank_mean = statistics.fmean(ranks_hit) if ranks_hit else float("inf")
            rank_std = (sum((x - rank_mean) ** 2 for x in ranks_hit) / len(ranks_hit)) ** 0.5 if len(ranks_hit) > 1 else 0.0
            f1_mean, f1_std = mean_std(all_f1)
            nusy_fp_mean, nusy_fp_std = mean_std(nusy_fp_rates)
            pearson_fp_mean, pearson_fp_std = mean_std(pearson_fp_rates)

            print(
                f"[RESULT] lambda_w={lw}, th={th} | "
                f"HitRate={hits}/{len(seeds)}={hit_rate:.2f}, "
                f"MeanRank(Hits)={rank_mean if ranks_hit else float('nan'):.2f}, "
                f"F1_Intra_mean={f1_mean:.4f}, "
                f"NuSy_FP_rate_mean={nusy_fp_mean:.2f}"
            )

            grid_rows.append(
                {
                    "lambda_w": lw,
                    "w_threshold": th,
                    "hit_count": hits,
                    "hit_rate": hit_rate,
                    "mean_rank_hits": rank_mean,
                    "rank_hits_std": rank_std,
                    "f1_intra_mean": f1_mean,
                    "f1_intra_std": f1_std,
                    "nusy_fp_rate_mean": nusy_fp_mean,
                    "nusy_fp_rate_std": nusy_fp_std,
                    "pearson_fp_rate_mean": pearson_fp_mean,
                    "pearson_fp_rate_std": pearson_fp_std,
                }
            )

    if not grid_rows:
        print("[ERROR] No configs produced results; aborting")
        return

    # 写 Grid Search 结果
    os.makedirs(RESULTS_DIR, exist_ok=True)
    grid_path = os.path.join(RESULTS_DIR, "rq2_hdfs_step2_grid.csv")
    grid_df = pd.DataFrame(grid_rows)
    grid_df.to_csv(grid_path, index=False)
    print(f"\n[INFO] HDFS Step2 grid search summary written to {grid_path}")

    # Select best config:
    #   1) Maximise NuSy-Edge Intra F1 (primary)
    #   2) Then minimise MeanRank(Hits)
    #   3) Then minimise NuSy FP rate
    def sort_key(row):
        return (
            -row["f1_intra_mean"],
            row["mean_rank_hits"],
            row["nusy_fp_rate_mean"],
        )

    best = sorted(grid_rows, key=sort_key)[0]
    print("\n=== Best Config (Grid Search) ===")
    print(
        f"lambda_w={best['lambda_w']}, W_THRESHOLD={best['w_threshold']} | "
        f"HitRate={best['hit_count']}/10={best['hit_rate']:.2f}, "
        f"MeanRank(Hits)={best['mean_rank_hits']:.2f}, "
        f"NuSy_FP_rate_mean={best['nusy_fp_rate_mean']:.2f}"
    )

    # 把推荐配置写到单独文件，方便人工或后续脚本更新主脚本默认值
    best_path = os.path.join(RESULTS_DIR, "rq2_hdfs_step2_grid_best.json")
    pd.DataFrame([best]).to_json(best_path, orient="records", indent=2)
    print(f"[INFO] Best config written to {best_path}")


if __name__ == "__main__":
    main()

