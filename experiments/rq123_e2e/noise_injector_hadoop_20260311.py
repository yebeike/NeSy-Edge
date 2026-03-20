"""
Hadoop-specific noise injector for Stage3 Dim1.

Goal:
- Make Hadoop baselines (Drain / Qwen) degrade with noise, while NeSy can stay high
  (when combined with experiments-only denoise for NuSy).

Design:
- Focus on Merger-related alert templates in Hadoop windows.
- Apply semantic-preserving paraphrases and token changes so template matching breaks.
"""

import random
import re


class HadoopNoiseInjector:
    def __init__(self, injection_rate: float = 1.0, seed: int = 2026):
        self.injection_rate = injection_rate
        self.rng = random.Random(seed)

    def inject(self, text: str) -> str:
        if not text:
            return ""
        if self.rng.random() > self.injection_rate:
            return text

        t = str(text)

        # Merger canonical phrases → paraphrases (keep meaning, disrupt template tokens)
        mapping = [
            (r"\bDown to the last merge-pass\b", "Final merge pass reached"),
            (r"\bmerge-pass\b", "merge stage"),
            (r"\bsegments left\b", "runs remaining"),
            (r"\btotal size\b", "aggregate bytes"),
            (r"\bbytes\b", "B"),
            (r"\bMerging\b", "Combining"),
            (r"\bsorted segments\b", "ordered runs"),
        ]

        for pat, rep in mapping:
            if re.search(pat, t):
                t = re.sub(pat, rep, t)

        # Add minor token-level perturbations around numbers (Drain is sensitive to surrounding static tokens)
        # e.g. "with 7 segments" -> "with about 7 runs"
        t = re.sub(r"\bwith\s+(\d+)\s+(segments|runs)\b", r"with about \1 \2", t, flags=re.IGNORECASE)
        t = re.sub(r"\bof\s+(\d+)\s+(bytes|b)\b", r"of ~\1 \2", t, flags=re.IGNORECASE)

        return t

