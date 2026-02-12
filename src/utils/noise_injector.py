import random

class NoiseInjector:
    def __init__(self, injection_rate=1.0, seed=2026):
        self.injection_rate = injection_rate
        self.rng = random.Random(seed)

    def inject(self, log_content: str, dataset_type="OpenStack") -> str:
        # 如果随机数大于注入率，则保持原样 (Stable)
        if self.rng.random() > self.injection_rate:
            return log_content

        if dataset_type == "OpenStack":
            return self._inject_openstack(log_content)
        elif dataset_type == "HDFS":
            return self._inject_hdfs(log_content)
        return log_content

    def _inject_openstack(self, text):
        # 强制变异
        replacements = {
            "instance": "VM", "server": "ComputeNode",
            "GET": "FETCH", "status: ": "status="
        }
        for old, new in replacements.items():
            if old in text: text = text.replace(old, new)
        return text

    def _inject_hdfs(self, text):
        replacements = {
            "PacketResponder": "PkgResponder",
            "terminating": "closing",
            "Received block": "Got blk",
            "Exception": "Error",
            "size": "len"
        }
        for old, new in replacements.items():
            if old in text: text = text.replace(old, new)
            
        # 🔥 关键修改：如果是注入模式，且包含 blk_，强制改掉！
        # 之前是 50% 概率，导致 Vanilla 有一半机会捡漏
        if "blk_" in text:
            text = text.replace("blk_", "block-id:")
            
        return text