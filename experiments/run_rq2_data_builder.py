import pandas as pd
import re
import os
from datetime import datetime
from collections import defaultdict

# ================= 配置区 (Robust Path Fix) =================
# 获取当前脚本所在的绝对路径 (.../NuSy-Edge/experiments)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录 (.../NuSy-Edge)
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

# 调试模式：设置为 200000 快速验证；设置为 None 跑全量
DEBUG_LIMIT = 200000 
# DEBUG_LIMIT = None 

# 拼接绝对路径 (根据你的截图，templates可能在 preprocessed 文件夹里，请注意检查)
# 假设你的目录结构是 data/HDFS_v1/HDFS.log
LOG_FILE = os.path.join(PROJECT_ROOT, 'data', 'HDFS_v1', 'HDFS.log')

# 注意：根据你的截图，模板文件好像在 preprocessed 子文件夹里？
# 如果报错，请尝试下一行注释掉的路径：
# TEMPLATE_FILE = os.path.join(PROJECT_ROOT, 'data', 'HDFS_v1', 'preprocessed', 'HDFS.log_templates.csv')
TEMPLATE_FILE = os.path.join(PROJECT_ROOT, 'data', 'raw', 'HDFS_v1', 'HDFS.log_templates.csv')

OUTPUT_MATRIX = os.path.join(PROJECT_ROOT, 'data', 'raw', 'rq2_background_matrix.csv')

# 打印路径检查 (如果不确定路径对不对，运行后看打印出来的路径是否存在)
print(f"Log Path: {LOG_FILE}")
print(f"Template Path: {TEMPLATE_FILE}")
# ==========================================================

def generate_matchers(template_path):
    """将模板转换为编译好的正则表达式，提升匹配速度"""
    df = pd.read_csv(template_path)
    matchers = []
    print(f"Loading {len(df)} templates...")
    for idx, row in df.iterrows():
        eid = row['EventId']
        # 将模板中的 [*] 替换为正则通配符 .*?
        # 并转义其他特殊字符
        pattern_str = re.escape(row['EventTemplate']).replace(r'\[\*\]', r'.*?')
        # 加上起止符优化匹配精度
        pattern = re.compile(pattern_str)
        matchers.append((eid, pattern))
    return matchers

def parse_hdfs_log():
    """流式读取日志，极速匹配"""
    matchers = generate_matchers(TEMPLATE_FILE)
    
    # 存储结果: timestamp -> {eid: count}
    time_series_data = defaultdict(lambda: defaultdict(int))
    
    print(f"Starting Flash Matching on {LOG_FILE}...")
    print(f"Debug Mode: {'ON (Limit ' + str(DEBUG_LIMIT) + ')' if DEBUG_LIMIT else 'OFF (Full Data)'}")

    with open(LOG_FILE, 'r', encoding='utf-8', errors='ignore') as f:
        for i, line in enumerate(f):
            if DEBUG_LIMIT and i >= DEBUG_LIMIT:
                break
            
            # HDFS Log 格式: 081109 203518 143 INFO ...
            # 提取前两个字段作为时间
            parts = line.split(maxsplit=2)
            if len(parts) < 3: continue
            
            date_str, time_str = parts[0], parts[1]
            content = parts[2]
            
            # 匹配 EventID
            matched_eid = None
            for eid, pattern in matchers:
                if pattern.search(content):
                    matched_eid = eid
                    break
            
            if matched_eid:
                # 转换时间戳 (聚合到分钟级)
                # 格式: 081109 203518 -> 2008-11-09 20:35:00
                ts_str = f"20{date_str} {time_str}"
                try:
                    dt = datetime.strptime(ts_str, "%Y%m%d %H%M%S")
                    # Round to minute
                    dt_minute = dt.replace(second=0) 
                    time_series_data[dt_minute][matched_eid] += 1
                except ValueError:
                    continue

            if i % 50000 == 0:
                print(f"Processed {i} lines...", end='\r')

    print(f"\nProcessing complete. Aggregating to matrix...")
    
    # 转为 DataFrame
    df = pd.DataFrame.from_dict(time_series_data, orient='index')
    df.fillna(0, inplace=True)
    df.sort_index(inplace=True)
    df.index.name = 'Timestamp'
    
    print(f"Matrix Shape: {df.shape}")
    df.to_csv(OUTPUT_MATRIX)
    print(f"Saved to {OUTPUT_MATRIX}")

if __name__ == "__main__":
    parse_hdfs_log()