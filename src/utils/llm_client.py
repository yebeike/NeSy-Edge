import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import re

class LLMClient:
    def __init__(self, model_relative_path="models/qwen3-0.6b"):
        """
        初始化本地 Qwen3-0.6B
        """
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.model_path = os.path.join(project_root, model_relative_path)
        
        # 硬件选择
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        if torch.cuda.is_available(): self.device = "cuda"
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype="auto", 
                device_map=self.device,
                trust_remote_code=True
            )
            self.model.eval()
        except Exception as e:
            raise RuntimeError(f"❌ Model Load Failed: {e}")

    def parse_with_rag(self, target_log, reference_log, reference_template):
        """
        [RAG 模式] 使用检索到的例子来引导模型
        """
        system_content = (
            "You are a Log Parser. \n"
            "Task: Transform the [Input Log] into an Event Template based on the [Reference].\n"
            "Rules:\n"
            "1. Replace dynamic variables (Numbers, IPs, BlockIDs, UUIDs) with <*>. \n"
            "2. Keep the static keywords exactly the same as the Reference.\n"
            "3. Output ONLY the template string."
        )

        user_content = (
            f"--- Reference ---\n"
            f"Log: {reference_log}\n"
            f"Template: {reference_template}\n\n"
            f"--- Input Log ---\n"
            f"{target_log}"
        )

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ]

        try:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except:
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                inputs.input_ids,
                max_new_tokens=64,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        gen_tokens = generated_ids[0][len(inputs.input_ids[0]):]
        response = self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

        return self._symbolic_cleanup(response)

    def _symbolic_cleanup(self, text):
        """正则清洗：修补 0.6B 模型的小毛病"""
        # 1. 去掉可能存在的 "Template:" 前缀
        text = re.sub(r'^(Template:|Output:|Result:)\s*', '', text, flags=re.IGNORECASE)
        
        # 2. 修复 blk_ 切分残留
        text = re.sub(r'blk_<\*>\s*\d+', 'blk_<*>', text)
        
        # 3. 强行把漏掉的长数字和IP替换掉 (兜底)
        text = re.sub(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(:\d+)?', '<*>', text)
        
        # 4. [关键新增] 修复双重尖括号 <<*>>
        text = text.replace("<<*>>", "<*>")
        
        return text.strip()