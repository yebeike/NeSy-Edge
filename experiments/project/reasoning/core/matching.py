"""Project-owned text matching helpers for reasoning evaluation."""

from __future__ import annotations

import re


HDFS_EFFECT_PROXY_RULES = {
    "[*]Got exception while serving[*]to[*]": {
        "target_proxies": [
            ("[*]BLOCK* NameSystem[*]addStoredBlock: blockMap updated:[*]is added to[*]size[*]", 3),
            ("[*]BLOCK* NameSystem[*]allocateBlock:[*]", 4),
            ("[*]Received block[*]of size[*]from[*]", 5),
        ],
        "root_proxies": [
            ("[*]PacketResponder[*]for block[*]terminating[*]", 0),
            ("[*]Received block[*]of size[*]from[*]", 2),
            ("[*]Receiving block[*]src:[*]dest:[*]", 3),
        ],
    },
}

UNDIRECTED_RELATIONS = {"pearson_undirected", "pc_undirected", "pc_ambiguous", "pc_bidirected"}
OTHER_FAMILY = {"HDFS": "HDFS_OTHER", "OpenStack": "OS_OTHER", "Hadoop": "HADOOP_UNKNOWN"}


def _norm(text: str) -> str:
    return " ".join(str(text or "").strip().lower().split())


def exact_relaxed_match(a: str, b: str) -> bool:
    na = _norm(a)
    nb = _norm(b)
    if not na or not nb:
        return False
    if na == nb:
        return True
    pa = normalize_template(a)
    pb = normalize_template(b)
    return bool(pa and pb and pa == pb)


def normalize_template(text: str) -> str:
    value = str(text or "").lower()
    value = value.replace("[*]", "<*>")
    value = re.sub(r"<[^>]+>", "<*>", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def canonical_tokens(text: str) -> list[str]:
    value = str(text or "").lower()
    value = value.replace("[*]", " <*> ")
    value = re.sub(r"<[^>]+>", " <*> ", value)
    value = re.sub(r"blk[_-]?\s*\*?", " block ", value)
    value = value.replace("namesystem.allocateblock", " allocateblock ")
    value = value.replace("namesystem.addstoredblock", " addstoredblock ")
    value = value.replace("packetresponder", " packetresponder ")
    value = value.replace("sync_power_state", " sync power state ")
    value = value.replace("re-created its instancelist", " recreated instancelist ")
    value = re.sub(r"[^a-z0-9\s]+", " ", value)
    stop = {
        "*",
        "num",
        "ip",
        "uuid",
        "http",
        "status",
        "len",
        "time",
        "the",
        "a",
        "an",
        "to",
        "of",
        "for",
        "on",
        "in",
        "is",
        "it",
        "this",
        "that",
        "and",
        "or",
        "from",
        "by",
        "with",
        "its",
        "did",
        "not",
        "skip",
    }
    return [token for token in value.split() if token not in stop and len(token) > 1]


def fuzzy_match(a: str, b: str) -> bool:
    ta = set(canonical_tokens(a))
    tb = set(canonical_tokens(b))
    if not ta or not tb:
        return False
    overlap = len(ta & tb)
    return overlap >= max(2, min(len(ta), len(tb)) // 2)


def hdfs_family(text: str) -> str:
    value = str(text or "").lower()
    if "got exception while serving" in value:
        return "HDFS_GOT_EXCEPTION_SERVING"
    if "allocateblock" in value:
        return "HDFS_ALLOCATE_BLOCK"
    if "packetresponder" in value and "terminating" in value:
        return "HDFS_PACKETRESPONDER"
    if (
        "exception in receiveblock" in value
        or ("writeblock" in value and "connection reset" in value)
        or "exception writing block" in value
    ):
        return "HDFS_EXCEPTION_RECEIVEBLOCK"
    if "received block" in value or "receiving block" in value or "transmitted block" in value:
        return "HDFS_RECEIVED_BLOCK"
    if "deleting block" in value or "trying to delete block" in value or "volumemap" in value:
        return "HDFS_DELETE_BLOCK"
    return "HDFS_OTHER"


def openstack_effect_family(text: str) -> str:
    value = str(text or "").lower()
    if (
        "unknown base file" in value
        or "base or swap file" in value
        or "removable base files" in value
        or "active base files" in value
        or "removing base or swap file" in value
        or "image cache manager" in value
    ):
        return "OS_UNKNOWN_BASE_FILE"
    if "synchronizing instance power states" in value or "sync_power_state" in value or "pending task" in value:
        return "OS_POWER_STATE_SYNC"
    if "instance sync for host" in value or "re-created its instancelist" in value or "successfully synced instances from host" in value:
        return "OS_SYNC_SUCCESS_ROOT"
    if "metadata" in value or "validating token" in value or "identity response" in value:
        return "OS_METADATA_SERVER"
    if "vcpu count" in value or "cpu affinity is not supported" in value:
        return "OS_VCPU_AFFINITY"
    return "OS_OTHER"


def hadoop_family(text: str) -> str:
    value = str(text or "").lower()
    if (
        "machine down" in value
        or "(reset) equator" in value
        or "lost node" in value
        or "container released on a *lost* node" in value
        or "last retry, killing" in value
        or "failures on node" in value
        or "nodemanager" in value
        or "task cleanup failed" in value
        or "could not obtain bp-" in value
        or "no live nodes contain block" in value
        or "dfs choosedatanode" in value
    ):
        return "HADOOP_MACHINE_DOWN"
    if (
        "network disconnection" in value
        or "bad datanode" in value
        or "failed to connect" in value
        or "no route" in value
        or "timed out" in value
        or "connectexception" in value
        or "retrying connect to server" in value
        or "communication exception" in value
        or "error communicating with rm" in value
        or "could not contact rm" in value
        or "address change detected" in value
        or "failed to renew lease" in value
        or "failure sending status update" in value
        or "forcibly closed by the remote host" in value
        or "datastreamer exception" in value
    ):
        return "HADOOP_NETWORK_DISCONNECTION"
    if (
        "disk full" in value
        or "no space" in value
        or "shuffleerror" in value
        or "error in shuffle" in value
        or "could not delete hdfs" in value
        or "diskerrorexception" in value
        or "exception in createblockoutputstream" in value
        or "could not find any valid local directory" in value
        or "shuffle failed : local error on this node" in value
        or "reducetask metrics system shutdown complete" in value
        or ("task " in value and " done." in value)
    ):
        return "HADOOP_DISK_FULL"
    return "HADOOP_UNKNOWN"


def family_of(dataset: str, text: str) -> str:
    if dataset == "HDFS":
        return hdfs_family(text)
    if dataset == "OpenStack":
        return openstack_effect_family(text)
    if dataset == "Hadoop":
        return hadoop_family(text)
    return "UNKNOWN"


def effect_match_kind(dataset: str, pred_effect: str, gt_effect: str) -> str:
    if exact_relaxed_match(pred_effect, gt_effect):
        return "exact"
    pred_family = family_of(dataset, pred_effect)
    gt_family = family_of(dataset, gt_effect)
    if pred_family == gt_family and pred_family != OTHER_FAMILY[dataset]:
        return "family"
    if fuzzy_match(pred_effect, gt_effect):
        return "fuzzy"
    return "none"

