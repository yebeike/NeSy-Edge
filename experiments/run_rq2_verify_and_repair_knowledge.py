import json
import os
import re
import hashlib
import pandas as pd
import numpy as np

# ==========================================
# CONFIGURATION
# ==========================================
KNOWLEDGE_PATH = "data/processed/causal_knowledge.json"
OS_LOG_DIR = "data/raw/OpenStack_2"

class CausalKnowledgeRefiner:
    def __init__(self, json_path):
        with open(json_path, 'r') as f:
            self.kb = json.load(f)
        # 精准对齐预处理脚本的正则列表
        self.regex_list = [
            (r'([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})', '<UUID>'),
            (r'(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})', '<IP>'),
            (r'(0x[0-9a-fA-F]+)', '<HEX>'),
            (r'(\d+)', '<NUM>'),
            (r'(req-[0-9a-fA-F-]+)', '<REQ_ID>')
        ]
        self.os_map_repair = {}

    def _get_consistent_hash(self, comp, lvl, content):
        """完全同步预处理脚本的 EventID 生成逻辑"""
        for pattern, replacement in self.regex_list:
            content = re.sub(pattern, replacement, content)
        fp = f"{comp}_{lvl}_{content.strip()}"
        return hashlib.md5(fp.encode()).hexdigest()[:12], content.strip()

    def perform_os_repair(self):
        print("[*] Re-scanning OpenStack logs with synchronized Regex engine...")
        files = [f for f in os.listdir(OS_LOG_DIR) if f.endswith('.log')]
        for f_name in files:
            comp = f_name.split('.')[0]
            with open(os.path.join(OS_LOG_DIR, f_name), 'r', encoding='latin-1') as f:
                for line in f:
                    lvl_match = re.search(r' (INFO|WARNING|ERROR|DEBUG) ', line)
                    if not lvl_match: continue
                    lvl = lvl_match.group(1)
                    # 提取等级后的内容
                    content = line.split(lvl)[-1].strip()
                    eid, tmpl = self._get_consistent_hash(comp, lvl, content)
                    self.os_map_repair[eid] = tmpl

    def validate_and_fix(self):
        print("\n" + "="*60)
        print("SYMBOLIC KNOWLEDGE BASE REFINEMENT")
        print("="*60)
        
        # 统计初始状态
        initial_unknown = sum(1 for f in self.kb if f['source_template'] == "Unknown" or f['target_template'] == "Unknown")
        print(f"[*] Initial Unknown Templates: {initial_unknown}")
        
        if initial_unknown > 0:
            self.perform_os_repair()
            fixed = 0
            for fact in self.kb:
                if fact['domain'] == 'openstack':
                    if fact['source_template'] == "Unknown" and fact['source_id'] in self.os_map_repair:
                        fact['source_template'] = self.os_map_repair[fact['source_id']]
                        fixed += 1
                    if fact['target_template'] == "Unknown" and fact['target_id'] in self.os_map_repair:
                        fact['target_template'] = self.os_map_repair[fact['target_id']]
                        fixed += 1
            print(f"[+] Fixed {fixed} semantic gaps using synchronized mapping.")

        # 最终完整性检查
        final_unknown = sum(1 for f in self.kb if f['source_template'] == "Unknown" or f['target_template'] == "Unknown")
        health_score = (len(self.kb) - final_unknown) / len(self.kb)
        
        print(f"[1] Final Semantic Health: {health_score:.2%}")
        
        # 检查权重合理性 (RQ2 科学性检查)
        weights = [f['weight'] for f in self.kb]
        print(f"[2] Causal Strength Profile: Mean={np.mean(weights):.3f}, Max={np.max(weights):.3f}")

        # 保存修复后的结果
        with open(KNOWLEDGE_PATH, 'w') as f:
            json.dump(self.kb, f, indent=4)
        
        if final_unknown == 0:
            print("\n[SUCCESS] Knowledge base is now 100% semantic ready for RQ3.")
        else:
            print(f"\n[WARNING] Still {final_unknown} Unknowns. Check hash collisions.")

if __name__ == "__main__":
    if os.path.exists(KNOWLEDGE_PATH):
        refiner = CausalKnowledgeRefiner(KNOWLEDGE_PATH)
        refiner.validate_and_fix()
    else:
        print("[Error] Knowledge base not found.")