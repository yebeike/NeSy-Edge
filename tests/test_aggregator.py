import sys
import os
import pandas as pd
from datetime import datetime, timedelta

# 路径黑魔法 (确保能 import cloud_node)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(os.path.join(project_root, "cloud_node"))

from aggregator import LogAggregator

def run_test():
    print("🚀 Testing Log Aggregator...")
    agg = LogAggregator(window_size_seconds=10)
    
    # 1. 模拟输入数据
    # 假设 10:00:00 ~ 10:00:09 发生了一次 DB 错误
    base_time = datetime(2025, 12, 7, 10, 0, 0)
    
    logs = [
        (base_time + timedelta(seconds=1), "Connection refused from <*>"),
        (base_time + timedelta(seconds=2), "Connection refused from <*>"), # 同一窗口发生两次
        (base_time + timedelta(seconds=11), "Authentication failure for user <*>"), # 下一个窗口
        (base_time + timedelta(seconds=15), "Unknown garbage log"), # 应该被忽略
        (base_time + timedelta(seconds=21), "Connection refused from <*>") # 第三个窗口
    ]
    
    print(f"📥 Injecting {len(logs)} mock logs...")
    for t, content in logs:
        agg.add_log(content, timestamp=t)
        
    # 2. 获取 DataFrame
    df = agg.get_dataframe()
    
    print("\n📊 Generated DataFrame (Time Series):")
    print(df)
    
    # 3. 验证逻辑
    print("\n🔍 Verification:")
    
    # 验证窗口 1 (10:00:00) 是否有 2 个 DB 错误
    row_0 = df.iloc[0]
    if row_0.get("ERR_DB_CONN") == 2:
        print("✅ Window 1 (DB Error count = 2): PASS")
    else:
        print(f"❌ Window 1 Failed: {row_0.get('ERR_DB_CONN')}")

    # 验证窗口 2 (10:00:10) 是否有 1 个 Auth 错误
    row_1 = df.iloc[1]
    if row_1.get("ERR_AUTH_FAIL") == 1:
        print("✅ Window 2 (Auth Error count = 1): PASS")
    else:
        print(f"❌ Window 2 Failed: {row_1.get('ERR_AUTH_FAIL')}")
        
    print("\n🎉 Aggregator is ready for CausalNex!")

if __name__ == "__main__":
    run_test()