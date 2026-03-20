import collections
import json
import os
import re
import sys
from typing import Dict, List

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

DATA_PROCESSED = os.path.join(_PROJECT_ROOT, "data", "processed")
SRC_PATH = os.path.join(DATA_PROCESSED, "gt_causal_knowledge_dynotears.json")
OUT_PATH = os.path.join(DATA_PROCESSED, "gt_causal_knowledge_dynotears_rq2_pruned_20260314.json")


def _hdfs_family(text: str) -> str:
    t = str(text or "").lower()
    if "got exception while serving" in t:
        return "HDFS_GOT_EXCEPTION_SERVING"
    if "allocateblock" in t:
        return "HDFS_ALLOCATE_BLOCK"
    if "packetresponder" in t and "terminating" in t:
        return "HDFS_PACKETRESPONDER"
    if "exception in receiveblock" in t:
        return "HDFS_EXCEPTION_RECEIVEBLOCK"
    if "received block" in t:
        return "HDFS_RECEIVED_BLOCK"
    if "deleting block" in t or "unexpected error trying to delete block" in t:
        return "HDFS_DELETE_BLOCK"
    return "HDFS_OTHER"


def _openstack_effect_family(text: str) -> str:
    t = str(text or "").lower()
    if "unknown base file" in t:
        return "OS_UNKNOWN_BASE_FILE"
    if "sync_power_state" in t or "synchronizing instance power states" in t:
        return "OS_POWER_STATE_SYNC"
    if "instance sync for host" in t or "re-created its instancelist" in t:
        return "OS_SYNC_SUCCESS_ROOT"
    if "metadata" in t or "nova.metadata.wsgi.server" in t:
        return "OS_METADATA_SERVER"
    if "vcpu count" in t or "cpu affinity is not supported" in t:
        return "OS_VCPU_AFFINITY"
    return "OS_OTHER"


def _hadoop_family(text: str) -> str:
    t = str(text or "").lower()
    if "machine down" in t:
        return "HADOOP_MACHINE_DOWN"
    if "network disconnection" in t:
        return "HADOOP_NETWORK_DISCONNECTION"
    if "disk full" in t:
        return "HADOOP_DISK_FULL"
    if any(p in t for p in ["bad datanode", "failed to connect", "no route", "timed out", "connection", "connectexception"]):
        return "HADOOP_NETWORK_DISCONNECTION"
    if any(p in t for p in ["no space", "disk full", "shuffleerror", "error in shuffle", "could not delete hdfs", "diskerrorexception"]):
        return "HADOOP_DISK_FULL"
    if any(p in t for p in ["container killed", "applicationmaster", "nodemanager", "lost node", "last retry, killing"]):
        return "HADOOP_MACHINE_DOWN"
    return "HADOOP_UNKNOWN"


def _target_family(domain: str, text: str) -> str:
    if domain == "hdfs":
        return _hdfs_family(text)
    if domain == "openstack":
        return _openstack_effect_family(text)
    return _hadoop_family(text)


def _load_edges(path: str) -> List[Dict[str, object]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else []


def _prune_edges(edges: List[Dict[str, object]]) -> List[Dict[str, object]]:
    pruned: List[Dict[str, object]] = []
    for domain in ("hdfs", "openstack", "hadoop"):
        domain_edges = [e for e in edges if str(e.get("domain", "")).lower() == domain]
        benchmark_edges = [e for e in domain_edges if str(e.get("relation", "")) == "benchmark_prior"]
        pruned.extend(benchmark_edges)

        best_by_family: Dict[str, Dict[str, object]] = {}
        for edge in domain_edges:
            if str(edge.get("relation", "")) == "benchmark_prior":
                continue
            fam = _target_family(domain, str(edge.get("target_template", "") or ""))
            if fam.endswith("OTHER") or fam.endswith("UNKNOWN"):
                continue
            score = abs(float(edge.get("weight", 0.0) or 0.0))
            kept = best_by_family.get(fam)
            if kept is None or score > abs(float(kept.get("weight", 0.0) or 0.0)):
                best_by_family[fam] = edge
        pruned.extend(best_by_family.values())
    return pruned


def _domain_stats(edges: List[Dict[str, object]]) -> Dict[str, Dict[str, object]]:
    stats: Dict[str, Dict[str, object]] = {}
    for domain in ("hdfs", "openstack", "hadoop"):
        domain_edges = [e for e in edges if str(e.get("domain", "")).lower() == domain]
        rel_counts = collections.Counter(str(e.get("relation", "")) for e in domain_edges)
        stats[domain] = {"edges": len(domain_edges), "relations": dict(rel_counts)}
    return stats


def main() -> None:
    edges = _load_edges(SRC_PATH)
    pruned = _prune_edges(edges)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(pruned, f, indent=2, ensure_ascii=False)

    before = _domain_stats(edges)
    after = _domain_stats(pruned)
    print("[INFO] Saved pruned modified graph to:", OUT_PATH)
    for domain in ("hdfs", "openstack", "hadoop"):
        print(
            f"[INFO] {domain}: {before[domain]['edges']} -> {after[domain]['edges']} edges | "
            f"relations={after[domain]['relations']}"
        )


if __name__ == "__main__":
    main()
