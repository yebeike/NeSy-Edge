import pandas as pd
import logging
from datetime import datetime, timedelta

logger = logging.getLogger("NuSy-Aggregator")

class LogAggregator:
    def __init__(self, window_size_seconds=10):
        self.window_size = window_size_seconds
        # 原始数据缓存: List of dict {'timestamp': dt, 'event': 'ERR_DB'}
        self.raw_events = []
        
        # 定义模板到变量的映射规则 (Mapping Rules)
        # 在真实系统中，这可以由 LLM 自动生成，MVP 阶段我们先手写核心的几个
        self.event_mapping = {
            "connection refused": "ERR_DB_CONN",
            "connection failed": "ERR_DB_CONN",
            "authentication failure": "ERR_AUTH_FAIL",
            "check pass": "INFO_CHECK_PASS",
            "small model speed test": "TEST_SIGNAL",
            "exited abnormally": "ERR_CRASH"
        }
        
        # 已知的变量列表 (用于生成 DataFrame 列)
        self.known_variables = list(set(self.event_mapping.values()))

    def add_log(self, template, timestamp=None):
        """
        接收 LILAC 解析出的模板，存入缓存
        """
        if timestamp is None:
            timestamp = datetime.now()
            
        # 1. 映射模板到变量名
        event_var = self._map_template_to_variable(template)
        if not event_var:
            return # 没用的日志直接丢弃
            
        # 2. 存入缓存
        self.raw_events.append({
            "timestamp": timestamp,
            "variable": event_var
        })
        
        # 保持缓存不过大 (只留最近 1 小时)
        if len(self.raw_events) > 10000:
            self.raw_events = self.raw_events[-5000:]

    def _map_template_to_variable(self, template):
        """
        简单关键词匹配，把长模板变成短变量名
        """
        template_lower = template.lower()
        for keyword, var_name in self.event_mapping.items():
            if keyword in template_lower:
                return var_name
        return None # 未知类型的日志忽略

    def get_dataframe(self, limit_windows=60):
        """
        核心功能：生成 CausalNex 所需的 Pandas DataFrame
        index: 时间窗口
        columns: 变量名 (ERR_DB_CONN, etc.)
        values: 计数
        """
        if not self.raw_events:
            return pd.DataFrame(columns=self.known_variables)

        # 转为 DataFrame
        df_raw = pd.DataFrame(self.raw_events)
        
        # 设置时间索引
        df_raw.set_index("timestamp", inplace=True)
        
        # 按窗口聚合 (Resample) 并计数 (Count)
        # rule=f'{self.window_size}S' 表示按 N 秒聚合
        df_agg = df_raw.groupby('variable').resample(f'{self.window_size}S').size()
        
        # 此时 df_agg 是多级索引 (variable, timestamp)，我们需要把它变成宽表 (Pivot)
        # unstack(0) 把 variable 这一层索引变成列
        df_matrix = df_agg.unstack(level=0, fill_value=0)
        
        # 补齐可能缺失的列 (确保结构稳定)
        for col in self.known_variables:
            if col not in df_matrix.columns:
                df_matrix[col] = 0
                
        # 填补时间空洞 (Fill gaps with 0)
        df_matrix = df_matrix.asfreq(f'{self.window_size}S', fill_value=0)
        
        # 只返回最近的 N 个窗口
        return df_matrix.tail(limit_windows)