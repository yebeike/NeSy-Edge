import sys
import os
import re
from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig
from drain3.masking import MaskingInstruction

class DrainParser:
    def __init__(self):
        config = TemplateMinerConfig()
        config.profiling_enabled = False
        
        # --- 1. 参数调优 ---
        config.drain_sim_th = 0.5 
        config.drain_depth = 4    
        
        # --- 2. 关键修正：使用 MaskingInstruction 对象 ---
        config.masking_instructions = [
            MaskingInstruction(r"((?<=[^A-Za-z0-9])|^)(([0-9a-f]{2,}:){3,}([0-9a-f]{2,}))((?=[^A-Za-z0-9])|$)", "ID"), # IPv6
            MaskingInstruction(r"((?<=[^A-Za-z0-9])|^)(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})((?=[^A-Za-z0-9])|$)", "IP"), # IPv4
            MaskingInstruction(r"((?<=[^A-Za-z0-9])|^)([0-9a-f]{6,})((?=[^A-Za-z0-9])|$)", "SEQ"), # Hex Sequence
            MaskingInstruction(r"((?<=[^A-Za-z0-9])|^)(\d+)(?=[^A-Za-z0-9]|$)", "NUM"), # Numbers
            MaskingInstruction(r"blk_([-0-9]+)", "BLK_ID") # HDFS Block ID
        ]
        
        self.template_miner = TemplateMiner(config=config)

    def parse(self, log_content: str) -> str:
        if not log_content:
            return ""
            
        # Drain3 核心解析
        result = self.template_miner.add_log_message(log_content)
        template = result["template_mined"]
        
        # --- 3. 格式对齐 (Post-Processing) ---
        # 修复 Bug: Drain 通常会输出 <ID>, <NUM> 等带尖括号的格式
        # 之前的 replace("ID", "<*>") 会导致 <<*>>
        
        for mask in ["ID", "IP", "SEQ", "NUM", "BLK_ID"]:
             # 1. 尝试替换带尖括号的标签 <ID> -> <*>
             template = template.replace(f"<{mask}>", "<*>")
             # 2. 保底：如果Drain没加尖括号，也替换
             template = template.replace(mask, "<*>")
        
        # 3. 替换 Drain 默认生成的泛化符号
        template = template.replace("<:*:>", "<*>")
        
        # 4. [关键] 终极清洗：防止任何 <<*>> 或 <<<*>>> 产生
        while "<<*>>" in template:
            template = template.replace("<<*>>", "<*>")

        return " ".join(template.split())