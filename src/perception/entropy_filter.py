import math
from collections import Counter

class EntropyFilter:
    """
    基于字符分布的香农熵计算器 (Character-Level Shannon Entropy).
    用于区分自然语言模板 (Low Entropy) 和包含随机UUID/参数的复杂日志 (High Entropy).
    """
    
    def __init__(self, threshold: float = 4.5):
        # 经验值：普通英文文本熵约 3.5-4.5，随机字符串 > 4.5
        self.threshold = threshold

    def calculate_entropy(self, text: str) -> float:
        """计算字符串的字符级香农熵"""
        if not text:
            return 0.0
            
        # 统计字符频率
        counts = Counter(text)
        total_chars = len(text)
        
        entropy = 0.0
        for count in counts.values():
            p = count / total_chars
            if p > 0:
                entropy -= p * math.log2(p)
        
        return entropy

    def is_high_entropy(self, text: str) -> bool:
        """判断是否为高熵日志"""
        return self.calculate_entropy(text) > self.threshold