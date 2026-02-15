# ================================================================================
# --- 文件: src/utils/metrics.py ---
# ================================================================================

import pandas as pd
import re
from sklearn.metrics import adjusted_rand_score

class MetricsCalculator:
    @staticmethod
    def normalize_template(text):
        """
        [关键修复] 鲁棒性标准化
        忽略标点符号、大小写和空格差异，只关注 单词序列 和 变量位置。
        这能解决 "status: 200" vs "status=200" 导致的 PA=0 问题。
        """
        if not isinstance(text, str): return ""
        
        # 1. 转小写
        text = text.lower().strip()
        
        # 2. 统一变量占位符 (防止 <*>, <String>, <Num> 的差异)
        # 将所有 <...> 替换为统一的内部 Token，例如 __VAR__
        text = re.sub(r'<\*?>', ' __VAR__ ', text)
        text = re.sub(r'<[^>]+>', ' __VAR__ ', text) # 匹配 <String> 等其他写法
        
        # 3. [核心] 移除标点符号 (只保留字母、数字、下划线和我们的占位符)
        # 将非单词字符替换为空格
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # 4. 合并空格
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    @staticmethod
    def calculate_pa(pred_templates, gt_templates):
        """
        Parsing Accuracy (Exact Match with Robust Normalization)
        """
        correct = 0
        total = 0
        
        for pred, gt in zip(pred_templates, gt_templates):
            p_norm = MetricsCalculator.normalize_template(pred)
            g_norm = MetricsCalculator.normalize_template(gt)
            
            # Debug: 如果你想看具体为什么对不上，可以在这里 print
            # if p_norm != g_norm and total < 5:
            #     print(f"Mismatch:\n  P: {p_norm}\n  G: {g_norm}")
            
            if p_norm == g_norm:
                correct += 1
            total += 1
            
        return correct / total if total > 0 else 0.0

    @staticmethod
    def calculate_ga(pred_templates, gt_templates):
        """
        Grouping Accuracy (ARI)
        """
        return adjusted_rand_score(gt_templates, pred_templates)

    @staticmethod
    def estimate_tokens(text):
        if not text: return 0
        return int(len(str(text).split()) * 1.3)
    
    @staticmethod
    def calculate_token_f1(pred_templates, gt_templates):
        if not pred_templates or not gt_templates: return 0.0
        
        scores = []
        for pred, gt in zip(pred_templates, gt_templates):
            # 分词 (基于非字母数字切分)
            p_tokens = set(re.split(r'[^a-z0-9]+', str(pred).lower()))
            g_tokens = set(re.split(r'[^a-z0-9]+', str(gt).lower()))
            
            p_tokens.discard('')
            g_tokens.discard('')
            
            if not g_tokens: 
                scores.append(0.0)
                continue
                
            common = p_tokens.intersection(g_tokens)
            
            precision = len(common) / len(p_tokens) if p_tokens else 0
            recall = len(common) / len(g_tokens) if g_tokens else 0
            
            if precision + recall == 0:
                f1 = 0.0
            else:
                f1 = 2 * (precision * recall) / (precision + recall)
            
            scores.append(f1)
            
        return sum(scores) / len(scores)