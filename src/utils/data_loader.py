import pandas as pd
import os

class DataLoader:
    def __init__(self, base_dir="data/raw"):
        # 自动定位到项目根目录
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.base_dir = os.path.join(project_root, "data", "raw")

    def get_openstack_test_data(self):
        log_path = os.path.join(self.base_dir, "OpenStack", "OpenStack_2k.log")
        gt_path = os.path.join(self.base_dir, "OpenStack", "OpenStack_2k.log_structured.csv")
        return self._load_data(log_path, gt_path)

    def get_hdfs_test_data(self):
        log_path = os.path.join(self.base_dir, "HDFS", "HDFS_2k.log")
        gt_path = os.path.join(self.base_dir, "HDFS", "HDFS_2k.log_structured.csv")
        return self._load_data(log_path, gt_path)

    def _load_data(self, log_path, gt_path):
        if not os.path.exists(log_path) or not os.path.exists(gt_path):
            raise FileNotFoundError(f"Data not found: {log_path} or {gt_path}")
            
        print(f"   [Loader] Loading logs from {os.path.basename(log_path)}...")
        with open(log_path, 'r', encoding='utf-8') as f:
            raw_logs = [x.strip() for x in f.readlines()]

        gt_df = pd.read_csv(gt_path)
        
        # 简单对齐检查
        min_len = min(len(raw_logs), len(gt_df))
        return raw_logs[:min_len], gt_df.iloc[:min_len]