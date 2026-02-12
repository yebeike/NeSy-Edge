import os
import re
import hashlib
import json
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler

# ==========================================
# CONFIGURATION
# ==========================================
LOG_DIR = "data/raw/OpenStack_2"
RAW_MATRIX = "data/processed/openstack_timeseries.csv"
REFINED_MATRIX = "data/processed/openstack_refined_ts.csv"
MAPPING_FILE = "data/processed/openstack_id_map.json"
WINDOW = "1min"

class OSGranularEngine:
    def __init__(self):
        self.regex_list = [
            (r'([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})', '<UUID>'),
            (r'(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})', '<IP>'),
            (r'(0x[0-9a-fA-F]+)', '<HEX>'),
            (r'(\d+)', '<NUM>'),
            (r'(req-[0-9a-fA-F-]+)', '<REQ_ID>')
        ]
        self.id_to_template = {}

    def get_id(self, comp, lvl, content):
        for p, r in self.regex_list:
            content = re.sub(p, r, content)
        template = content.strip()
        fp = f"{comp}_{lvl}_{template}"
        eid = hashlib.md5(fp.encode()).hexdigest()[:12]
        self.id_to_template[eid] = template
        return eid

def main():
    engine = OSGranularEngine()
    all_events = []
    log_files = [f for f in os.listdir(LOG_DIR) if f.endswith('.log')]
    
    print(f"[*] Parsing logs and building identity map...")
    for f_name in log_files:
        comp = f_name.split('.')[0]
        with open(os.path.join(LOG_DIR, f_name), 'r', encoding='latin-1') as f:
            for line in f:
                ts_match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
                lvl_match = re.search(r' (INFO|WARNING|ERROR|DEBUG) ', line)
                if not ts_match or not lvl_match: continue
                lvl = lvl_match.group(1)
                content = line.split(lvl)[-1].strip()
                eid = engine.get_id(comp, lvl, content)
                all_events.append({'Timestamp': datetime.strptime(ts_match.group(1), "%Y-%m-%d %H:%M:%S"), 'EventID': eid})

    df = pd.DataFrame(all_events).set_index('Timestamp')
    matrix = df.groupby('EventID').resample(WINDOW).size().unstack(level=0).fillna(0)
    
    # 特征精炼：过滤低频 & 高相关
    matrix = matrix.loc[:, matrix.sum(axis=0) > 10]
    corr = matrix.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [c for c in upper.columns if any(upper[c] > 0.98)]
    matrix = matrix.drop(columns=to_drop)
    
    # 保留 Top 50 高变异维度
    cv = matrix.std() / (matrix.mean() + 1e-6)
    final_cols = cv.sort_values(ascending=False).head(50).index.tolist()
    matrix_final = matrix[final_cols]
    
    # 持久化：矩阵 + 映射表
    matrix_final.to_csv(REFINED_MATRIX)
    # 仅保存最终 50 维对应的映射，确保 100% 匹配
    final_mapping = {eid: engine.id_to_template[eid] for eid in final_cols}
    with open(MAPPING_FILE, 'w') as f:
        json.dump(final_mapping, f, indent=4)
    
    print(f"[SUCCESS] Refined Matrix (50D) and Sidecar Mapping saved.")

if __name__ == "__main__":
    main()