import os
import re
import sys
import time
import warnings
import pandas as pd
from datetime import datetime
from tqdm import tqdm

# Suppress warnings
warnings.filterwarnings("ignore")

# ==========================================
# CONFIGURATION
# ==========================================
# 第一次测试可以保持 True，确信没问题后改为 False 跑全量
# DEBUG_MODE = True 
DEBUG_MODE = False 
DEBUG_LIMIT = 200000

LOG_FILE = "data/raw/HDFS_v1/HDFS.log"
TEMPLATE_FILE = "data/raw/HDFS_v1/preprocessed/HDFS.log_templates.csv"
OUTPUT_DIR = "data/processed"
OUTPUT_FILE = f"{OUTPUT_DIR}/hdfs_timeseries.csv"
WINDOW_SIZE = '1min' 

# ==========================================
# PROCESSING CLASSES
# ==========================================
class FlashMatcher:
    def __init__(self, template_path):
        self.templates = self._load_templates(template_path)
        self.regex_map = self._compile_regex()

    def _load_templates(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Template file not found: {path}")
        df = pd.read_csv(path)
        return dict(zip(df['EventId'], df['EventTemplate']))

    def _compile_regex(self):
        regex_map = {}
        for eid, content in self.templates.items():
            pattern = re.escape(content).replace(r'\[\*\]', r'(.*?)')
            regex_map[eid] = re.compile(pattern)
        return regex_map

    def parse_log_file(self, log_path):
        print(f"[*] Starting Flash Matching on {log_path}...")
        parsed_events = []
        
        # Estimate total lines for tqdm (HDFS_v1 full size approx)
        total_lines = DEBUG_LIMIT if DEBUG_MODE else 11175629
        
        with open(log_path, 'r') as f:
            for idx, line in tqdm(enumerate(f), total=total_lines, unit="line", desc="Parsing"):
                if DEBUG_MODE and idx >= DEBUG_LIMIT:
                    break
                
                parts = line.strip().split()
                if len(parts) < 6: continue
                
                # Timestamp: 081109 203518
                try:
                    ts_str = f"{parts[0]} {parts[1]}"
                    dt = datetime.strptime(ts_str, "%y%m%d %H%M%S")
                except ValueError:
                    continue

                # Content extraction
                try:
                    content = " ".join(parts[5:])
                except:
                    continue

                for eid, regex in self.regex_map.items():
                    if regex.search(content):
                        parsed_events.append({'Timestamp': dt, 'EventId': eid})
                        break

        return pd.DataFrame(parsed_events)

def main():
    print(f"=== NeSy-Edge Step 1: Data Processing ===")
    print(f"Mode: {'DEBUG' if DEBUG_MODE else 'FULL PRODUCTION'}")
    
    if not os.path.exists(LOG_FILE):
        print(f"[Error] File not found: {LOG_FILE}")
        return

    # 1. Parsing
    matcher = FlashMatcher(TEMPLATE_FILE)
    df = matcher.parse_log_file(LOG_FILE)
    
    if df.empty:
        print("[Error] No events parsed.")
        return

    # 2. Aggregation
    print("[*] Aggregating to Time Series...")
    df.set_index('Timestamp', inplace=True)
    # Using size() to avoid IndexError
    ts_df = df.groupby('EventId').resample(WINDOW_SIZE).size().unstack(level=0).fillna(0)
    ts_df.sort_index(inplace=True)
    
    print(f"[+] Matrix Shape: {ts_df.shape}")
    
    # 3. Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ts_df.to_csv(OUTPUT_FILE)
    print(f"[+] Saved processed data to: {OUTPUT_FILE}")
    print("[Done] You can now run step2_causal_analysis.py multiple times.")

if __name__ == "__main__":
    main()