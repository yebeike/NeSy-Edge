import sys
import os
import pandas as pd
import numpy as np

# 路径黑魔法
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(os.path.join(project_root, "cloud_node"))

from causal_engine import CausalEngine

def run_test():
    print("🚀 Initializing Causal Engine...")
    engine = CausalEngine()
    
    # 1. 构造模拟数据 (Simulated Data)
    # 场景：DB_FAIL (原因) -> API_ERROR (结果)
    # 我们造一个很强的相关性：DB_FAIL 发生后，API_ERROR 必定发生
    print("📊 Generating Synthetic Data...")
    np.random.seed(42)
    
    # 100 个时间步
    n_steps = 100
    db_fails = np.random.binomial(1, 0.2, n_steps) # 20% 概率发生
    
    # API Error 滞后 1 秒发生 (模拟 t-1 -> t)
    # Shift array: api_errors[t] 取决于 db_fails[t-1]
    api_errors = np.roll(db_fails, 1) 
    api_errors[0] = 0 # 修正第一个点
    
    # 加一点噪声
    noise = np.random.binomial(1, 0.05, n_steps)
    api_errors = np.bitwise_or(api_errors, noise)
    
    df = pd.DataFrame({
        "ERR_DB_FAIL": db_fails,
        "ERR_API_500": api_errors
    })
    
    print(df.head())
    
    # 2. 运行学习 (包含 LLM 约束生成)
    print("\n🧠 Learning Structure (with LLM Constraints)...")
    sm = engine.learn_structure(df)
    
    if sm:
        print("✅ Graph Learned Successfully!")
        print(f"Edges: {sm.edges}")
        
        # 3. 根因推理测试
        print("\n🔍 Root Cause Analysis for 'ERR_API_500':")
        causes = engine.find_root_cause("ERR_API_500_lag0", None) # lag0 表示当前时刻
        # 注意：DYNOTEARS 输出的节点名可能带 _lag0 或 _lag1 后缀
        # 如果上面没找到，尝试找不带后缀的
        if not causes:
             causes = engine.find_root_cause("ERR_API_500", None)
             
        print(f"Potential Causes: {causes}")
        
        # 验证核心逻辑
        # 我们期望看到 ERR_DB_FAIL (lag1 或 lag0) 在原因列表里
        found = False
        for cause, weight in causes:
            if "ERR_DB_FAIL" in cause:
                found = True
                break
        
        if found:
            print("✅ SUCCESS: Correctly identified DB_FAIL as a cause!")
        else:
            print("⚠️ WARNING: DB_FAIL not found in top causes. Adjust lambda/threshold.")
            
    else:
        print("❌ Structure Learning Failed.")

if __name__ == "__main__":
    run_test()