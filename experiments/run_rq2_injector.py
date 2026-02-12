import pandas as pd
import numpy as np
import networkx as nx
import pickle

# ================= 配置区 =================
INPUT_MATRIX = '../data/rq2_background_matrix.csv'
OUTPUT_INJECTED = '../data/rq2_injected_data.csv'
OUTPUT_GT_GRAPH = '../data/rq2_ground_truth.gpickle' # 保存 NetworkX 图对象

# 定义注入规则： (Source, Target, Lag, Coeff)
# Lag=0: 同一时刻因果; Lag=1: 下一时刻因果
INJECTION_RULES = [
    # 场景 1: 链式滞后传播 (E5 -> E11 -> E26)
    ('E5', 'E11', 1, 0.8),   # E5(t-1) 导致 E11(t)
    ('E11', 'E26', 1, 0.7),  # E11(t-1) 导致 E26(t)
    
    # 场景 2: 同期强相关 (E9 -> E3)
    ('E9', 'E3', 0, 0.9),    # E9(t) 导致 E3(t) 瞬间发生
]

def inject_causality():
    print("Loading background matrix...")
    df = pd.read_csv(INPUT_MATRIX, index_col='Timestamp')
    
    # 确保所有涉及的列都存在
    all_events = set(src for src, _, _, _ in INJECTION_RULES) | \
                 set(tgt for _, tgt, _, _ in INJECTION_RULES)
    
    for col in all_events:
        if col not in df.columns:
            print(f"Warning: {col} not in logs, creating zero column.")
            df[col] = 0.0

    # 创建 Ground Truth 图
    G_gt = nx.DiGraph()
    # 添加所有节点以确保维度对齐
    for col in df.columns:
        G_gt.add_node(col)

    print("Injecting Causal Patterns...")
    # 归一化数据，方便控制注入强度
    df = (df - df.mean()) / (df.std() + 1e-5)
    
    injected_df = df.copy()
    
    for src, tgt, lag, coeff in INJECTION_RULES:
        # 添加边到真值图
        G_gt.add_edge(src, tgt, lag=lag)
        
        # 修改数据
        noise = np.random.normal(0, 0.1, size=len(df))
        if lag == 0:
            injected_df[tgt] += coeff * injected_df[src] + noise
        else:
            # Shift 数据模拟滞后: t时刻的 tgt 受 t-lag 时刻的 src 影响
            # shift(1) 意味着把上一行的数据移到现在这行
            shifted_src = injected_df[src].shift(lag).fillna(0)
            injected_df[tgt] += coeff * shifted_src + noise
            
    print("Injection complete.")
    injected_df.to_csv(OUTPUT_INJECTED)
    nx.write_gpickle(G_gt, OUTPUT_GT_GRAPH)
    print(f"Saved injected data and Ground Truth graph.")
    print(f"Ground Truth Edges: {G_gt.edges(data=True)}")

if __name__ == "__main__":
    inject_causality()