# ================================================================================
# --- 源文件: src/system/edge_node.py ---
# ================================================================================

import time
import re
from difflib import SequenceMatcher
from src.utils.llm_client import LLMClient
from src.system.knowledge_base import KnowledgeBase

class NuSyEdgeNode:
    def __init__(self):
        print("🚀 [NuSy-Edge] Booting up...")
        self.llm = LLMClient(model_relative_path="models/qwen3-0.6b")
        try:
            self.kb = KnowledgeBase()
        except Exception as e:
            print(f"⚠️ [NuSy-Edge] Failed to load Knowledge Base: {e}")
            self.kb = None
        self.cache = {}

    def _compute_fingerprint(self, log_content):
        return re.sub(r'\d+', 'N', log_content)

    @staticmethod
    def preprocess_header(raw_log, dataset_type):
        """保持原有的 Header 清洗逻辑"""
        log_content = raw_log.strip()
        if dataset_type == "HDFS":
            match = re.search(r'\s(INFO|WARN|ERROR)\s+[^:]+:\s*(.*)', raw_log)
            if match:
                log_content = match.group(2).strip()
            else:
                parts = raw_log.split(': ', 1)
                if len(parts) > 1 and len(parts[0]) > 20: 
                     log_content = parts[1].strip()
        elif dataset_type == "OpenStack":
            match = re.search(r'\[req-[^\]]+\]\s+(.*)', raw_log)
            if match: log_content = match.group(1).strip()
        return log_content

    def _calculate_similarity(self, s1, s2):
        """计算两个字符串的相似度 (0.0 - 1.0)"""
        # 简单使用 SequenceMatcher，生产环境可用 Levenshtein 或 Cosine
        return SequenceMatcher(None, s1, s2).ratio()

    def parse_log_stream(self, raw_log, dataset_type):
        start_t = time.time()
        
        # A. 预处理
        content = NuSyEdgeNode.preprocess_header(raw_log, dataset_type)
        if not content: content = raw_log

        # B. L1 Cache
        fingerprint = self._compute_fingerprint(content)
        if fingerprint in self.cache:
            latency = (time.time() - start_t) * 1000
            return self.cache[fingerprint], latency, True 

        # C. L2 RAG + LLM (智能层)
        template = ""
        
        if self.kb:
            # 检索 Top-1
            rag_results = self.kb.search(content, dataset_type, top_k=1)
            
            if rag_results:
                best_match = rag_results[0]
                ref_log = best_match['raw_log']
                ref_template = best_match['template']
                
                # --- [关键策略] 符号直通 (Symbolic Shortcut) ---
                # 计算输入日志与检索到的参考日志的相似度
                # 注意：这里对比的是“去Header后的内容”，且忽略数字差异
                sim_score = self._calculate_similarity(
                    self._compute_fingerprint(content), 
                    self._compute_fingerprint(ref_log)
                )
                
                # 阈值判定：如果非常相似 (例如 > 0.6，因为 RAG 已经按向量筛选过了)
                # 或者是同一类日志 (Pattern 相似)，则直接信任参考模版
                # 0.6 是个经验值，对于 HDFS 这种只有变量不同的日志，相似度通常很高
                if sim_score > 0.6:
                    template = ref_template
                    # 记录到 Cache，下次直接命中
                    self.cache[fingerprint] = template
                    latency = (time.time() - start_t) * 1000
                    return template, latency, False # RAG Hit (算是 Miss Cache, 但没跑 LLM)

                # 如果相似度不够高，说明是新变体，交给 LLM 生成
                template = self.llm.parse_with_rag(content, ref_log, ref_template)
            else:
                # 没检索到（冷启动），裸跑 LLM (通常效果不好，但没办法)
                template = self.llm.parse_with_rag(content, "No reference", "No reference")
        else:
            template = self.llm.parse_with_rag(content, "No reference", "No reference")
        
        # D. 更新 Cache
        self.cache[fingerprint] = template
        latency = (time.time() - start_t) * 1000
        return template, latency, False