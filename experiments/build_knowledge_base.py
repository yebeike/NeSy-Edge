import sys
import os
import pandas as pd

# 路径修正
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.data_loader import DataLoader
from src.system.knowledge_base import KnowledgeBase

def build_kb():
    print("🚀 Building NuSy-Edge Knowledge Base (RAG)...")
    
    # 1. 初始化
    loader = DataLoader()
    kb = KnowledgeBase()
    
    # 为了保证实验纯净，先清空旧数据
    try:
        kb.clear()
        kb = KnowledgeBase() # Re-init
    except:
        pass

    # 2. 注入 OpenStack 知识
    print("\n--- Processing OpenStack ---")
    logs, gt_df = loader.get_openstack_test_data()
    templates = gt_df['EventTemplate'].tolist()
    
    # 关键：只用前 50% 构建知识库 (Train Set)
    # 后 50% 留着做测试 (Test Set)
    split_idx = int(len(logs) * 0.5)
    train_logs = logs[:split_idx]
    train_templates = templates[:split_idx]
    
    kb.add_knowledge(train_logs, train_templates, "OpenStack")
    print(f"✅ Indexed {len(train_logs)} OpenStack logs.")

    # 3. 注入 HDFS 知识
    print("\n--- Processing HDFS ---")
    logs, gt_df = loader.get_hdfs_test_data()
    templates = gt_df['EventTemplate'].tolist()
    
    split_idx = int(len(logs) * 0.5)
    train_logs = logs[:split_idx]
    train_templates = templates[:split_idx]
    
    kb.add_knowledge(train_logs, train_templates, "HDFS")
    print(f"✅ Indexed {len(train_logs)} HDFS logs.")

    print("\n🎉 Knowledge Base built successfully!")
    print(f"💾 Data stored in: {kb.persist_path}")

if __name__ == "__main__":
    build_kb()