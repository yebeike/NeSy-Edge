import pandas as pd
import re
import numpy as np
import os
import sys

# 路径修复
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config import OPENSTACK_LOG_PATH, PROCESSED_DATA_DIR

def convert_logs_to_timeseries():
    print(f"🔄 Converting Logs from {OPENSTACK_LOG_PATH}...")
    
    data = []
    with open(OPENSTACK_LOG_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            # 提取时间: 2017-05-16 00:00:00.008
            # 提取组件: nova.osapi_compute.wsgi.server
            # 提取级别: INFO / ERROR
            
            # 正则提取时间
            time_match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
            if not time_match:
                continue
                
            timestamp = time_match.group(1)
            
            parts = line.split()
            if len(parts) < 6:
                continue
                
            level = parts[5] # INFO
            component_full = parts[6] # nova.osapi_compute...
            
            # 简化组件名: nova, neutron, cinder, keystone
            if "nova" in component_full:
                component = "nova"
            elif "neutron" in component_full:
                component = "neutron"
            elif "cinder" in component_full:
                component = "cinder"
            elif "keystone" in component_full:
                component = "keystone"
            elif "glance" in component_full:
                component = "glance"
            else:
                component = "other"
                
            data.append({
                "timestamp": timestamp,
                "metric": f"{component}_{level}" # e.g., nova_INFO
            })
            
    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # 按秒聚合 (因为 2k 条日志时间跨度很短，按分钟聚合可能只有1个点)
    df_grouped = df.groupby(['timestamp', 'metric']).size().unstack(level=1, fill_value=0)
    
    # 重采样为 1秒 间隔，填充0
    df_resampled = df_grouped.resample('1s').sum().fillna(0)
    
    print(f"   Original Data Shape: {df_resampled.shape} (Time x Metrics)")
    
    # ⚠️ 关键处理：数据增强
    # 如果数据行数少于 100 行（DYNOTEARS 需要足够样本），我们进行 "Tile" (平铺复制) 并加入噪声
    # 模拟长时间运行的系统
    if len(df_resampled) < 200:
        print("   ⚠️ Data too short for causal discovery. Augmenting data...")
        n_repeats = 200 // len(df_resampled) + 2
        
        # 复制数据
        augmented_data = np.tile(df_resampled.values, (n_repeats, 1))
        
        # 加入随机高斯噪声 (模拟真实波动)
        noise = np.random.normal(0, 0.5, augmented_data.shape)
        augmented_data = np.abs(augmented_data + noise) # 保证非负
        
        # 截取前 500 个点
        augmented_data = augmented_data[:500]
        
        # 重建 DataFrame
        df_final = pd.DataFrame(augmented_data, columns=df_resampled.columns)
    else:
        df_final = df_resampled

    # 移除全为0的列 (没有变化的指标无法计算因果)
    df_final = df_final.loc[:, (df_final != 0).any(axis=0)]
    
    output_path = PROCESSED_DATA_DIR / "openstack_metrics.csv"
    df_final.to_csv(output_path, index=False)
    print(f"✅ Saved Time-Series Metrics to {output_path}")
    print(f"   Final Shape: {df_final.shape}")
    print(f"   Metrics: {list(df_final.columns)}")

if __name__ == "__main__":
    convert_logs_to_timeseries()