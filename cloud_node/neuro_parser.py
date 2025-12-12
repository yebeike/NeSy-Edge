import time
import re
import chromadb
import uuid
import logging
import os
from langchain_ollama import ChatOllama
from config import LLM_MODEL, LLM_PARAMS

logger = logging.getLogger("LILAC-Parser")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

class NeuroParser:
    def __init__(self, collection_name="log_templates"):
        # ChromaDB 客户端
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = self.chroma_client.get_or_create_collection(name=collection_name)
        
        # System 2 (LLM)
        logger.info(f"🧠 Loading System 2 (LLM: {LLM_MODEL})...")
        self.llm = ChatOllama(
            model=LLM_MODEL,
            temperature=0.1, 
            num_predict=512, # 不需要太长，避免废话
            repeat_penalty=1.1
        )
        # 放宽阈值：允许更模糊的匹配
        self.threshold = 0.45

    def preprocess(self, log_line):
        """
        核心去噪：针对 Loghub Linux 格式优化
        Raw: Jun 14 15:16:01 combo sshd(pam_unix)[19939]: authentication failure...
        Target: authentication failure...
        """
        log_line = log_line.strip()
        
        # --- 新增：针对 Nginx / echo 注入的格式 ---
        # 格式: 2025/12/07 12:00:00 [error] ...
        # 策略: 找到 "] " (error 级别后面的括号)
        if "[error]" in log_line or "[warn]" in log_line:
            parts = log_line.split('] ', 1)
            if len(parts) > 1:
                return parts[1] # 返回 "] " 后面的内容
        # ---------------------------------------

        # 原有的 Linux 格式处理
        parts = log_line.split(']: ', 1)
        if len(parts) > 1:
            return parts[1]
            
        if len(log_line) > 20:
            header_candidate = log_line[:30]
            if ": " in header_candidate:
                pass
            match = re.search(r'\s(\w+|[\w\(\)\-]+)(\[\d+\])?:\s', log_line)
            if match:
                return log_line[match.end():]

        return log_line

    def parse(self, raw_log):
        # 1. 预处理
        clean_content = self.preprocess(raw_log)
        if not clean_content: return None

        # [DEBUG] 看看预处理切得对不对
        # print(f"DEBUG: Clean='{clean_content}'")

        start_time = time.time()
        
        # --- System 1: 向量检索 ---
        results = self.collection.query(
            query_texts=[clean_content],
            n_results=1
        )

        # 检查命中
        if results['ids'][0]:
            dist = results['distances'][0][0]
            # logger.info(f"Dist: {dist}") # 调试用
            if dist < self.threshold:
                template = results['documents'][0][0]
                duration = time.time() - start_time
                return {
                    "template": template,
                    "source": "cache",
                    "time": duration
                }

        # --- System 2: LLM ---
        logger.info(f"🐢 [Cache Miss] Asking LLM...")
        
        template = self._call_llm(clean_content)
        
        # [Fallback] 如果 LLM 返回空，或者完全没变，为了防止存入空数据，
        # 我们直接使用 clean_content 作为模板存进去。
        # 这样下一次同样的日志来，就能命中 System 1 了。
        if not template or len(template) < 5:
            template = clean_content
            # 把其中的数字替换掉，稍微做点通用化
            template = re.sub(r'\d+', '<*>', template)
            logger.warning(f"⚠️ LLM failed/empty. Using fallback template.")

        duration = time.time() - start_time
        
        # 存入库
        self.collection.add(
            documents=[template],
            metadatas=[{"raw": raw_log}],
            ids=[str(uuid.uuid4())]
        )
        
        logger.info(f"🧠 [Learned] '{template[:50]}...' (Time: {duration:.2f}s)")
        
        return {
            "template": template,
            "source": "llm",
            "time": duration
        }

    def _call_llm(self, content):
        # 极简 Prompt，减少 LLM 思考负担
        prompt = (
            f"Message: {content}\n\n"
            "Task: Replace variables (IPs, numbers, user names) with <*>. Keep static text.\n"
            "Output: ONLY the template string.\n"
        )
        try:
            resp = self.llm.invoke(prompt)
            return resp.content.strip().strip("'").strip('"')
        except Exception:
            return ""