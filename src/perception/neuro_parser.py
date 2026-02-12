import re
from src.utils.llm_client import LLMClient

class NeuroParser:
    def __init__(self, use_real_llm=True, model="qwen3:latest"):
        self.use_real_llm = use_real_llm
        if use_real_llm:
            self.llm = LLMClient(provider="ollama", model=model)

    def parse(self, log_content: str) -> str:
        """
        使用 LLM 解析日志模板。
        """
        if not self.use_real_llm:
            return log_content

        # 针对 Qwen/Llama 优化的 Prompt，要求只输出结果
        prompt = f"""
Task: Extract the static event template from the log.
Rules:
1. Replace specific variables (IPs, IDs, Numbers, Paths) with <*>.
2. Do NOT output explanations.
3. Output ONLY the template string.

Examples:
Log: "Accepted password for root from 119.137.62.142 port 49116 ssh2"
Template: "Accepted password for <*> from <*> port <*> ssh2"

Log: "instance: 54fadb41-2c4e Claim successful"
Template: "[instance: <*>] Claim successful"

Log: "{log_content}"
Template:
"""
        response = self.llm.generate(prompt)
        
        # 清洗结果：只取第一行有效文本，去除引号
        lines = [line.strip() for line in response.split('\n') if line.strip()]
        if not lines:
            return log_content
            
        # 有时候模型会重复 "Template: " 前缀，去掉它
        template = lines[0]
        if template.lower().startswith("template:"):
            template = template[9:].strip()
            
        return template.strip('"').strip("'")