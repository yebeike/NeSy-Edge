"""
RQ1-style noise injector for Stage2 HDFS mini-RQ1 path only.

Use this module when running HDFS Dim1 with mini-RQ1 data to match the trend
of visualize_rq1_complete.py (Drain degrades with noise, NeSy stays high).
Does NOT modify src/utils/noise_injector.py.

- HDFS: base synonym replacements + blk_ -> block-id only (no hard_replacements).
- OpenStack: same as main NoiseInjector (delegate or copy).
- Other: pass-through.
"""

import random
from typing import Optional

# Optional: if None, HDFS uses this module's logic; OpenStack can use main injector
_MAIN_INJECTOR = None


def get_rq1_injector(seed: int = 2026) -> "NoiseInjectorRQ1":
    return NoiseInjectorRQ1(seed=seed)


class NoiseInjectorRQ1:
    """RQ1-style injector: HDFS uses original/weaker rules (no hard_replacements)."""

    def __init__(self, injection_rate: float = 1.0, seed: int = 2026):
        self.injection_rate = injection_rate
        self.rng = random.Random(seed)

    def inject(self, log_content: str, dataset_type: str = "OpenStack") -> str:
        if self.rng.random() > self.injection_rate:
            return log_content
        if dataset_type == "OpenStack":
            return self._inject_openstack(log_content)
        if dataset_type == "HDFS":
            return self._inject_hdfs_rq1(log_content)
        return log_content

    def _inject_openstack(self, text: str) -> str:
        replacements = {
            "instance": "VM",
            "instances": "VMs",
            "server": "ComputeNode",
            "servers": "ComputeNodes",
            "GET": "FETCH",
            "POST": "SUBMIT",
            "status: ": "status=",
            "Unknown base file": "Unrecognized base resource",
            "While synchronizing instance power states": "While syncing VM power states",
        }
        for old, new in replacements.items():
            if old in text:
                text = text.replace(old, new)
        return text

    def _inject_hdfs_rq1(self, text: str) -> str:
        """Original RQ1 HDFS rules: base synonyms + blk_ -> block-id only."""
        base_replacements = {
            "PacketResponder": "PkgResponder",
            "terminating": "closing",
            "Received block": "Got blk",
            "Exception": "Error",
            "size": "len",
        }
        for old, new in base_replacements.items():
            if old in text:
                text = text.replace(old, new)
        if "blk_" in text:
            text = text.replace("blk_", "block-id:")
        return text
