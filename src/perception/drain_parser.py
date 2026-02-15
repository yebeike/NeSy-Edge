# ================================================================================
# --- 文件: src/perception/drain_parser.py ---
# ================================================================================

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
        
        # --- 参数调优 ---
        config.drain_sim_th = 0.5 
        config.drain_depth = 4     
        
        # --- 关键修正：Masking 列表 ---
        config.masking_instructions = [
            # 1. HDFS Block ID (保留，用于后续还原前缀)
            MaskingInstruction(r"(blk_[-0-9]+)", "BLK"), 
            
            # 2. [已删除] IP:Port 组合正则 
            # 原因: GT 希望是 "<*>:<*>" (分开的)，而原来的正则会合并成一个 "<*>" 导致 PA=0
            # MaskingInstruction(r"((\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}):(\d+))", "IP"),
            
            # 3. 标准 IP
            MaskingInstruction(r"(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})", "IP"),
            
            # 4. 文件路径 
            MaskingInstruction(r"(\/[\w\.\/-]+)", "PATH"),
            
            # 5. UUID / 16进制序列
            MaskingInstruction(r"([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})", "UUID"),
            MaskingInstruction(r"([0-9a-fA-F]{6,})", "SEQ"),
            
            # 6. 纯数字 (固定宽度 Look-behind 修复版)
            MaskingInstruction(r"(?<!\S)(\d+)(?=\s|$|\.|:)", "NUM"), 
            MaskingInstruction(r"(\d+)", "NUM") 
        ]
        
        self.template_miner = TemplateMiner(config=config)

    def parse(self, log_content: str) -> str:
        if not log_content:
            return ""
            
        try:
            result = self.template_miner.add_log_message(log_content)
            template = result["template_mined"]
        except Exception:
            return log_content

        # --- Post-Processing: 针对 GT 风格的精准对齐 ---
        
        # 1. [HDFS] 还原 Block ID 前缀
        # Drain 输出是 <BLK>，GT 需要 blk_<*>
        if "<BLK>" in template:
            template = template.replace("<BLK>", "blk_<*>")
        
        # 2. [OpenStack] 清洗 "HTTP" 协议残留
        # GT 通常忽略 "HTTP/1.1"，而 Drain 会保留 "HTTP" 并把 "/1.1" mask 掉
        # 变成 "HTTP<*>" 或 "HTTP <*>"，这里统一删掉
        template = re.sub(r'HTTP\S*', '', template, flags=re.IGNORECASE)

        # 3. 通用标签归一化
        # 处理带尖括号的标签 <IP>, <PATH>, <NUM> -> <*>
        template = re.sub(r'<[A-Z_]+>', '<*>', template)
        
        # 处理 Drain 可能没加尖括号的原始 Tag
        for tag in ["PATH", "UUID", "SEQ"]:
             # 注意：BLK 已经在上面处理过了，这里不处理
             if tag in template:
                 template = template.replace(tag, "<*>")

        # 4. Drain 默认泛化符
        template = template.replace("<:*:>", "<*>")
        
        # 5. 清理多余的 <*> (但在 HDFS 中，<*>:<*> 是合法的，不能无脑合并)
        # 策略：只合并连续的、中间没有标点的 <*>
        # 例如 "<*> <*>" -> "<*>"
        while "<*> <*>" in template:
            template = template.replace("<*> <*>", "<*>")
        
        # 注意：这里我们不再强制合并 <<*>> (如 blk_<*>) 或 <*><*>，以免误伤
        
        return " ".join(template.split())