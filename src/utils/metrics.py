import pandas as pd
import re
from sklearn.metrics import adjusted_rand_score

class MetricsCalculator:
    @staticmethod
    def normalize_template(text):
        """
        标准化模板：转小写，合并空格，统一变量符。
        """
        if not isinstance(text, str): return ""
        text = text.lower().strip()
        text = re.sub(r'\s+', ' ', text) # 合并多余空格
        text = re.sub(r'<\*?>', '<*>', text) # 统一占位符
        return text

    @staticmethod
    def calculate_pa(pred_templates, gt_templates):
        """
        Standard Robust PA: 忽略大小写和空格差异的 Exact Match。
        修复了之前过于激进(删除所有标点)的问题，因为输入修复后不需要那么激进。
        """
        correct = 0
        total = 0
        
        for pred, gt in zip(pred_templates, gt_templates):
            p_norm = MetricsCalculator.normalize_template(pred)
            g_norm = MetricsCalculator.normalize_template(gt)
            
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
        """
        Token-level F1 Score: 衡量语义保留程度
        """
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