# experiments/debug_drain_alignment.py
import sys
import os
import pandas as pd
from colorama import Fore, Style, init

# 路径黑魔法
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.perception.drain_parser import DrainParser
from src.utils.data_loader import DataLoader
from src.utils.metrics import MetricsCalculator

init(autoreset=True)

def visualize_diff(str1, str2):
    """辅助函数：高亮显示差异"""
    # 简单归一化对比
    norm1 = MetricsCalculator.normalize_template(str1)
    norm2 = MetricsCalculator.normalize_template(str2)
    
    if norm1 == norm2:
        return f"{Fore.GREEN}MATCH{Style.RESET_ALL}"
    else:
        return f"{Fore.RED}MISMATCH{Style.RESET_ALL}"

def debug_dataset(dataset_name, limit=5):
    print(f"\n{'='*20} Debugging {dataset_name} {'='*20}")
    
    loader = DataLoader()
    if dataset_name == "HDFS":
        logs, gt_df = loader.get_hdfs_test_data()
        # 针对 HDFS 特殊的 Header 预处理模拟
        logs = [l.split(': ', 1)[1].strip() if ': ' in l else l for l in logs]
    else:
        logs, gt_df = loader.get_openstack_test_data()
        # 针对 OpenStack 特殊的 Header 预处理模拟
        import re
        logs = [re.search(r'\[req-[^\]]+\]\s+(.*)', l).group(1).strip() if re.search(r'\[req-[^\]]+\]', l) else l for l in logs]

    gt_templates = gt_df['EventTemplate'].tolist()
    
    drain = DrainParser()
    
    # 预热 Drain (让它建立树)
    print("🤖 Training Drain...")
    for log in logs[:100]: 
        drain.parse(log)
        
    print(f"🔍 Checking first {limit} samples...\n")
    
    for i in range(limit):
        raw_log = logs[i]
        gt = gt_templates[i]
        pred = drain.parse(raw_log)
        
        status = visualize_diff(pred, gt)
        
        print(f"[{i}] Status: {status}")
        # print(f"    Raw : {raw_log[:100]}...") # 太长可以注释掉
        print(f"    GT  : {gt}")
        print(f"    Pred: {pred}")
        
        # 显示 MetricsCalculator 看到的归一化结果，帮助判断是标点问题还是单词问题
        if "MISMATCH" in status:
            print(f"    [Norm GT  ]: {MetricsCalculator.normalize_template(gt)}")
            print(f"    [Norm Pred]: {MetricsCalculator.normalize_template(pred)}")
        print("-" * 50)

if __name__ == "__main__":
    # debug_dataset("HDFS", limit=5)
    debug_dataset("OpenStack", limit=5)
    debug_dataset("HDFS", limit=5)