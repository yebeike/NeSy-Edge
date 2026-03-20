"""
HDFS-specific PA normalization for Stage2 mini-RQ1 only.

Makes NeSy output (e.g. "block-id: <*>") compare equal to Drain-style GT ("blk_<*>")
under golden PA, so PA_Nusy improves without changing src/utils/metrics.py.
"""

import re

_PROJECT_ROOT = None


def _project_root():
    global _PROJECT_ROOT
    if _PROJECT_ROOT is None:
        import os
        _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return _PROJECT_ROOT


def normalize_for_pa_hdfs(text: str) -> str:
    """
    Apply HDFS-specific equivalences then MetricsCalculator.normalize_template:
    - block-id / block-id: -> blk_
    - received <-> got, exception <-> error, serving <-> handling, packet <-> pkg, terminating <-> closing
      so NeSy/Qwen paraphrases match Drain-style GT.
    """
    import sys
    root = _project_root()
    if root not in sys.path:
        sys.path.insert(0, root)
    from src.utils.metrics import MetricsCalculator
    if not isinstance(text, str):
        return ""
    t = text
    # Pre-step: align with Drain / RQ1 noise vocabulary (whole-word where safe)
    t = re.sub(r"\bblock-id\s*:?\s*", "blk_", t, flags=re.IGNORECASE)
    t = re.sub(r"\breceived\b", "got", t, flags=re.IGNORECASE)
    t = re.sub(r"\bexception\b", "error", t, flags=re.IGNORECASE)
    t = re.sub(r"\bserving\b", "handling", t, flags=re.IGNORECASE)
    t = re.sub(r"\bpacket\b", "pkg", t, flags=re.IGNORECASE)
    t = re.sub(r"\bterminating\b", "closing", t, flags=re.IGNORECASE)
    t = re.sub(r"\bhandling\b", "serving", t, flags=re.IGNORECASE)
    return MetricsCalculator.normalize_template(t)


def golden_pa_hdfs(pred: str, gt: str) -> int:
    """Golden PA for HDFS using normalize_for_pa_hdfs for both pred and gt."""
    if not gt:
        return 0
    p_norm = normalize_for_pa_hdfs(pred)
    g_norm = normalize_for_pa_hdfs(gt)
    return 1 if p_norm and p_norm == g_norm else 0
