"""
RQ1 结果汇总脚本 (NeSy-Edge 硕士论文用)
读取 results/rq1_benchmark_mini_N*.csv，生成可放入论文的表格与简要分析，输出到 results/。
"""
import os
import sys
import glob
import pandas as pd

def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)
    results_dir = "results"
    
    # 查找最新的 benchmark CSV
    pattern = os.path.join(results_dir, "rq1_benchmark_mini_N*.csv")
    files = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    if not files:
        print("No rq1_benchmark_mini_N*.csv found in results/")
        return
    
    path = files[0]
    df = pd.read_csv(path)
    n = path.split("N")[-1].replace(".csv", "")
    
    out_lines = []
    out_lines.append("=" * 70)
    out_lines.append("RQ1 Result Summary (NeSy-Edge Thesis)")
    out_lines.append(f"Source: {path}")
    out_lines.append("=" * 70)
    
    # 1) 按 Dataset × Noise 的 PA 对比表 (适合论文表格)
    pivot_pa = df.pivot_table(index=["Dataset", "Noise_Rate"], columns="Method", values="PA")
    out_lines.append("\n[Table: PA by Dataset and Noise Rate]")
    out_lines.append(pivot_pa.round(3).to_string())
    
    pivot_ga = df.pivot_table(index=["Dataset", "Noise_Rate"], columns="Method", values="GA")
    out_lines.append("\n[Table: GA by Dataset and Noise Rate]")
    out_lines.append(pivot_ga.round(3).to_string())
    
    # 2) 全局平均 (各方法)
    global_avg = df.groupby("Method")[["PA", "GA", "Latency", "Tokens", "F1_Token"]].mean()
    global_avg = global_avg.sort_values("PA", ascending=False)
    out_lines.append("\n[Table: Global Average per Method]")
    out_lines.append(global_avg.round(4).to_string())
    
    # 3) 简要结论
    out_lines.append("\n[Brief Analysis]")
    nusy_pa = df[df["Method"] == "NuSy-Edge"]["PA"].mean()
    drain_pa = df[df["Method"] == "Drain"]["PA"].mean()
    vanilla_pa = df[df["Method"] == "Vanilla"]["PA"].mean()
    out_lines.append(f"  NuSy-Edge avg PA: {nusy_pa:.3f}")
    out_lines.append(f"  Drain avg PA:     {drain_pa:.3f}")
    out_lines.append(f"  Vanilla avg PA:   {vanilla_pa:.3f}")
    if nusy_pa >= drain_pa and nusy_pa > vanilla_pa:
        out_lines.append("  -> NuSy-Edge achieves best or tied best PA on average.")
    else:
        out_lines.append("  -> Check per-dataset/noise breakdown for robustness under noise.")
    
    # 4) 高噪音 (0.6, 0.8, 1.0) 下各方法平均 PA
    high_noise = df[df["Noise_Rate"] >= 0.6]
    if not high_noise.empty:
        high_avg = high_noise.groupby("Method")["PA"].mean().sort_values(ascending=False)
        out_lines.append("\n[High Noise (>=0.6) Average PA]")
        out_lines.append(high_avg.round(3).to_string())
    
    text = "\n".join(out_lines)
    print(text)
    
    # 写入 results
    summary_path = os.path.join(results_dir, f"rq1_summary_N{n}.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"\nSummary written to {summary_path}")

if __name__ == "__main__":
    main()
