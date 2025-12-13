import pandas as pd
import logging
from datetime import datetime

logger = logging.getLogger("NuSy-Aggregator")

class LogAggregator:
    def __init__(self, window_size_seconds=10):
        self.window_size = window_size_seconds
        self.raw_events = []
        
        self.event_mapping = {
            "connection refused": "ERR_DB_CONN",
            "connection failed": "ERR_DB_CONN",
            "authentication failure": "ERR_AUTH_FAIL",
            "check pass": "INFO_CHECK_PASS",
            "small model speed test": "TEST_SIGNAL",
            "exited abnormally": "ERR_CRASH",
            "bad gateway": "ERR_NGINX_502" # 补上 Nginx 502 的映射
        }
        
        self.known_variables = list(set(self.event_mapping.values()))

    def add_log(self, template, timestamp=None):
        if timestamp is None:
            timestamp = datetime.now()
            
        event_var = self._map_template_to_variable(template)
        if not event_var:
            return 
            
        self.raw_events.append({
            "timestamp": timestamp,
            "variable": event_var
        })
        
        if len(self.raw_events) > 10000:
            self.raw_events = self.raw_events[-5000:]

    def _map_template_to_variable(self, template):
        template_lower = template.lower()
        for keyword, var_name in self.event_mapping.items():
            if keyword in template_lower:
                return var_name
        return None 

    def get_dataframe(self, limit_windows=60):
        """
        优雅重构版：使用 pd.Grouper 和 reindex 保证结构绝对稳定
        """
        # 1. 边界情况：无数据
        if not self.raw_events:
            return pd.DataFrame(columns=self.known_variables)

        df = pd.DataFrame(self.raw_events)
        
        # 确保时间列是 datetime 类型
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # 2. 核心聚合 (The Elegant Way)
        # 按照 (时间窗口, 变量名) 进行分组统计
        # 结果必定是一个 Series，Index 为 MultiIndex(timestamp, variable)
        counts = df.groupby([
            pd.Grouper(key='timestamp', freq=f'{self.window_size}S'), 
            'variable'
        ]).size()
        
        # 3. 维度转换 (Pivot)
        # 将 variable 层级转为列。fill_value=0 填充那些没发生的事件
        df_matrix = counts.unstack(level='variable', fill_value=0)
        
        # 4. 结构对齐 (Structural Alignment)
        # 使用 reindex 强制对齐列名。如果某些已知变量没出现，自动补0。
        # 这比手动 for 循环加 if 判断要快且安全得多。
        df_matrix = df_matrix.reindex(columns=self.known_variables, fill_value=0)
        
        # 5. 时间轴补全
        # 确保时间轴是连续的，中间没有断档
        df_matrix = df_matrix.asfreq(f'{self.window_size}S', fill_value=0)
        
        return df_matrix.tail(limit_windows)