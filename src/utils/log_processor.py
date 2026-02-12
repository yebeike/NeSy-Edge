import pandas as pd
import numpy as np
import re
from datetime import datetime
from src.config import OPENSTACK_LOG_PATH

class LogProcessor:
    def __init__(self, log_path=OPENSTACK_LOG_PATH):
        self.log_path = log_path

    def _parse_line(self, line):
        """
        OpenStack Format: 
        LogFileName Date Time PID Level Component Content
        """
        try:
            parts = line.split()
            if len(parts) < 6: return None
            
            # 1. 提取时间
            date_str = parts[1]
            time_str = parts[2]
            dt = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S.%f")
            
            # 2. 提取细粒度组件 (Fine-grained Component)
            # Raw Component 可能是: nova.osapi_compute.wsgi.server
            # 我们提取前两段: nova.osapi_compute
            full_comp = parts[5]
            sub_parts = full_comp.split('.')
            
            if len(sub_parts) >= 2:
                # 提取如 'nova.virt', 'nova.api', 'oslo_messaging'
                # 这种粒度足以构建拓扑图
                component = f"{sub_parts[0]}.{sub_parts[1]}"
            else:
                component = sub_parts[0]
            
            # 过滤掉无关的 python 库日志 (如 requests, urllib)
            if component in ['python.requests', 'urllib3.connectionpool']:
                return None

            # 3. 权重 (Error 依然加权)
            level = parts[4]
            weight = 10.0 if level == "ERROR" else 1.0 
            
            return {"timestamp": dt, "component": component, "weight": weight}
            
        except Exception:
            return None

    def get_time_series(self, window_size='1s', augment=True):
        print(f"🔄 Processing logs from {self.log_path}...")
        
        with open(self.log_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        parsed_data = [self._parse_line(line) for line in lines]
        parsed_data = [d for d in parsed_data if d is not None]
        
        df = pd.DataFrame(parsed_data)
        
        # 统计组件分布，去掉极其稀疏的组件 (防止噪音)
        comp_counts = df['component'].value_counts()
        # 只保留出现次数最多的 Top 5-8 个组件，保证图的可读性
        top_components = comp_counts.head(8).index.tolist()
        df = df[df['component'].isin(top_components)]
        
        print(f"   Identified Components: {top_components}")
        
        df.set_index('timestamp', inplace=True)
        
        # 聚合
        df_agg = df.pivot_table(
            index='timestamp', 
            columns='component', 
            values='weight', 
            aggfunc='sum'
        ).resample(window_size).sum().fillna(0)
        
        # 数据增强 (同前，为了算法收敛)
        if augment and len(df_agg) < 1000:
            print(f"   Augmenting data from {len(df_agg)} to 1000 steps...")
            original_data = df_agg.values
            target_len = 1000
            repeats = target_len // len(df_agg) + 1
            
            # 使用 Tile 重复 + 随机噪声 (Noise Injection)
            # 噪声是为了防止矩阵奇异 (Singular Matrix)
            augmented = np.tile(original_data, (repeats, 1))[:target_len]
            noise = np.random.normal(0, 0.01, augmented.shape) 
            augmented += np.abs(noise)
            
            df_agg = pd.DataFrame(augmented, columns=df_agg.columns)
            
        # 标准化
        df_norm = (df_agg - df_agg.mean()) / (df_agg.std() + 1e-8)
        
        print(f"✅ Time-series ready: {df_norm.shape} (Time x Components)")
        return df_norm