"""
Aggressive noise for Hadoop when run from Stage2 (return_stats=True).

Goal: at high noise, PA_Drain drops more so PA_Nusy >> PA_Drain.
Does not modify src/utils/noise_injector.py.
"""

import random


def get_hadoop_aggressive_injector(seed: int = 2026) -> "HadoopAggressiveInjector":
    return HadoopAggressiveInjector(seed=seed)


class HadoopAggressiveInjector:
    """Hadoop: base HDFS-style replacements + extra aggressive rewrites so Drain fails more at high noise."""

    def __init__(self, injection_rate: float = 1.0, seed: int = 2026):
        self.injection_rate = injection_rate
        self.rng = random.Random(seed)

    def inject(self, log_content: str, dataset_type: str = "HDFS") -> str:
        if self.rng.random() > self.injection_rate:
            return log_content
        if dataset_type != "HDFS":
            return log_content
        text = log_content
        # Base (same as RQ1 HDFS)
        base = {
            "PacketResponder": "PkgResponder",
            "terminating": "closing",
            "Received block": "Got blk",
            "Exception": "Error",
            "size": "len",
        }
        for old, new in base.items():
            if old in text:
                text = text.replace(old, new)
        if "blk_" in text:
            text = text.replace("blk_", "block-id:")
        # Aggressive: break Drain's token patterns (Hadoop-specific)
        aggressive = {
            "Container": "Cnt",
            "container": "cnt",
            "Task": "Tk",
            "task": "tk",
            "application": "app",
            "Application": "App",
            "attempt_": "att_",
            "received exception": "got error",
            "Executing": "Running",
        }
        for old, new in aggressive.items():
            if old in text:
                text = text.replace(old, new)
        return text
