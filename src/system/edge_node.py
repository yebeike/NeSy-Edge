import time
import re
from src.utils.llm_client import LLMClient
from src.system.knowledge_base import KnowledgeBase

class NuSyEdgeNode:
    def __init__(self):
        """
        NuSy-Edge Node (Complete Architecture)
        L1: Memory Cache (Symbolic)
        L2: RAG + LLM (Neuro-Symbolic)
        """
        print("🚀 [NuSy-Edge] Booting up...")
        
        # 1. 加载模型
        self.llm = LLMClient(model_relative_path="models/qwen3-0.6b")
        
        # 2. 加载知识库 (RAG)
        try:
            self.kb = KnowledgeBase()
            print("📘 [NuSy-Edge] Knowledge Base connected.")
        except Exception as e:
            print(f"⚠️ [NuSy-Edge] Failed to load Knowledge Base: {e}")
            self.kb = None

        # 3. 初始化缓存
        self.cache = {}

    def _compute_fingerprint(self, log_content):
        """L1 指纹计算: 将所有数字替换为 N"""
        return re.sub(r'\d+', 'N', log_content)

    @staticmethod
    def preprocess_header(raw_log, dataset_type):
        """
        [关键修改] Header 剥离逻辑静态化，供外部 Benchmark 复用
        """
        log_content = raw_log.strip()
        if dataset_type == "HDFS":
            # 兼容 INFO/WARN, 取第一个 ': ' 之后的内容
            parts = raw_log.split(': ', 1)
            if len(parts) > 1: log_content = parts[1].strip()
        elif dataset_type == "OpenStack":
            # 取 [req-...] 之后的内容
            match = re.search(r'\[req-[^\]]+\]\s*(.*)', raw_log)
            if match: log_content = match.group(1).strip()
        return log_content

    def parse_log_stream(self, raw_log, dataset_type):
        """
        流式解析主逻辑
        """
        start_t = time.time()
        
        # A. 预处理 (调用静态方法)
        content = NuSyEdgeNode.preprocess_header(raw_log, dataset_type)
        if not content: content = raw_log # Fallback

        # B. L1 Cache (极速层)
        fingerprint = self._compute_fingerprint(content)
        if fingerprint in self.cache:
            latency = (time.time() - start_t) * 1000
            return self.cache[fingerprint], latency, True # True = Hit

        # C. L2 RAG + LLM (智能层) - Cache Miss!
        
        # C1. 检索 (RAG)
        ref_log = "No reference"
        ref_template = "No reference"
        
        if self.kb:
            rag_results = self.kb.search(content, dataset_type, top_k=1)
            if rag_results:
                ref_log = rag_results[0]['raw_log']
                ref_template = rag_results[0]['template']
        
        # C2. 推理 (LLM)
        template = self.llm.parse_with_rag(content, ref_log, ref_template)
        
        # D. 更新 L1 Cache
        self.cache[fingerprint] = template
        
        latency = (time.time() - start_t) * 1000
        return template, latency, False # False = Miss