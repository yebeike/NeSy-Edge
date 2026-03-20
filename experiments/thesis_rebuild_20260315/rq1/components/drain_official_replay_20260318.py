from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, List


@dataclass
class _LogCluster:
    log_template: List[str]
    log_ids: List[int] = field(default_factory=list)


@dataclass
class _Node:
    childD: dict = field(default_factory=dict)
    depth: int = 0
    digitOrtoken: str | int | None = None


@dataclass
class OfficialDrainConfig:
    depth: int = 4
    st: float = 0.4
    max_child: int = 100


def _regex_sub(pattern: str, repl: str, text: str) -> str:
    return re.sub(pattern, repl, text, flags=re.IGNORECASE)


def _dataset_config(dataset: str) -> OfficialDrainConfig:
    if dataset == "HDFS":
        return OfficialDrainConfig(depth=4, st=0.5, max_child=100)
    if dataset == "Hadoop":
        return OfficialDrainConfig(depth=4, st=0.5, max_child=100)
    if dataset == "OpenStack":
        return OfficialDrainConfig(depth=6, st=0.95, max_child=100)
    return OfficialDrainConfig()


def _preprocess_patterns(dataset: str) -> list[tuple[str, str]]:
    if dataset == "HDFS":
        return [
            (r"blk_-?\d+", "blk_<*>"),
            (r"(\d+\.){3}\d+(:\d+)?", "<*>"),
        ]
    if dataset == "OpenStack":
        return [
            (r"((\d+\.){3}\d+,?)+", "<*>"),
            (r"/.+?\s", "<*> "),
            (r"\d+", "<*>"),
        ]
    if dataset == "Hadoop":
        return [
            (r"\bcontainer_\d+(?:_\d+){3}\b", "container_<*>"),
            (r"\bapplication_\d+(?:_\d+){2}\b", "application_<*>"),
            (r"\bappattempt_\d+(?:_\d+){3}\b", "appattempt_<*>"),
            (r"\bjob_\d+(?:_\d+){2}\b", "job_<*>"),
            (r"\btask_\d+(?:_\d+){2}_[mr]_\d+\b", "task_<*>"),
            (r"\battempt_\d+(?:_\d+){2}_[mr]_\d+(?:_\d+)?\b", "attempt_<*>"),
            (r"\bjvm_\d+(?:_\d+){2}_[mr]_\d+\b", "jvm_<*>"),
            (r"\bDFSClient_NONMAPREDUCE_-?\d+_\d+\b", "DFSClient_NONMAPREDUCE_<*>_<*>"),
            (r"\bCONTAINER_REMOTE_[A-Z_]+\b", "CONTAINER_REMOTE_<*>"),
            (
                r"\b(PendingReds|ScheduledMaps|ScheduledReds|AssignedMaps|AssignedReds|CompletedMaps|CompletedReds|ContAlloc|ContRel|HostLocal|RackLocal):-?\d+(?:\.\d+)?",
                r"\1:<*>",
            ),
            (r"\b(ask|release|newContainers|finishedContainers|resourceLimit)=-?\d+(?:\.\d+)?", r"\1=<*>"),
            (r"\[[A-Za-z0-9_.-]+:\d+\]", "[<*>:<*>]"),
            (r"(?:jar:file:)?/[A-Za-z0-9_./:-]+", "<*>"),
            (r"(\d+\.){3}\d+", "<*>"),
        ]
    return [(r"\d+", "<*>")]


class _OfficialDrainReplayParser:
    """
    A line-oriented adaptation of the original Drain parser logic.

    It keeps the prefix-tree matching strategy and token-template generalization
    behavior from the reference implementation, but operates on already-isolated
    log content lines instead of dataset files.
    """

    def __init__(self, dataset: str, config: OfficialDrainConfig | None = None):
        self.dataset = dataset
        self.config = config or _dataset_config(dataset)
        self.depth = max(1, int(self.config.depth) - 2)
        self.st = float(self.config.st)
        self.max_child = int(self.config.max_child)
        self.root = _Node()
        self.next_line_id = 1

    @staticmethod
    def _has_numbers(token: str) -> bool:
        return any(ch.isdigit() for ch in token)

    def preprocess(self, line: str) -> str:
        value = line or ""
        for pattern, repl in _preprocess_patterns(self.dataset):
            value = _regex_sub(pattern, repl, value)
        value = re.sub(r"\s+", " ", value).strip()
        return value

    def _postprocess(self, template: str, raw_log: str) -> str:
        value = " ".join((template or "").split())
        if self.dataset == "Hadoop" and value:
            return self._postprocess_hadoop(value, raw_log)
        if self.dataset != "OpenStack" or not value:
            return value

        lowered_raw = (raw_log or "").lower()

        if " status: " in value and " len: " in value and " time: " in value:
            method_match = re.search(r'"(GET|POST|DELETE)\s', raw_log, flags=re.IGNORECASE)
            if method_match:
                method = method_match.group(1).upper()
                return f'<*> "{method} <*>" status: <*> len: <*> time: <*>.<*>'

        if "lifecycle event" in lowered_raw:
            action_match = re.search(r"\bVM (Started|Stopped|Paused|Resumed)\b", raw_log, flags=re.IGNORECASE)
            if action_match:
                return f"[instance: <*>] VM {action_match.group(1).title()} (Lifecycle Event)"

        if "instance spawned successfully" in lowered_raw:
            return "[instance: <*>] Instance spawned successfully."
        if "instance destroyed successfully" in lowered_raw:
            return "[instance: <*>] Instance destroyed successfully."
        if "creating image" in lowered_raw:
            return "[instance: <*>] Creating image"
        if "terminating instance" in lowered_raw:
            return "[instance: <*>] Terminating instance"
        if "claim successful" in lowered_raw:
            return "[instance: <*>] Claim successful"

        if "defaulting to unlimited" in lowered_raw:
            if "disk limit" in lowered_raw:
                return "[instance: <*>] disk limit not specified, defaulting to unlimited"
            if "vcpu limit" in lowered_raw:
                return "[instance: <*>] vcpu limit not specified, defaulting to unlimited"

        if "took" in lowered_raw and "the instance on the hypervisor" in lowered_raw:
            action_match = re.search(r"\bto (spawn|destroy) the instance on the hypervisor\b", lowered_raw)
            if action_match:
                return f"[instance: <*>] Took <*>.<*> seconds to {action_match.group(1)} the instance on the hypervisor."

        if "total memory:" in lowered_raw:
            return "[instance: <*>] Total memory: <*> MB, used: <*>.<*> MB"
        if "total disk:" in lowered_raw:
            return "[instance: <*>] Total disk: <*> GB, used: <*>.<*> GB"
        if "total vcpu:" in lowered_raw:
            return "[instance: <*>] Total vcpu: <*> VCPU, used: <*>.<*> VCPU"
        if "memory limit:" in lowered_raw and "free:" in lowered_raw:
            return "[instance: <*>] memory limit: <*>.<*> MB, free: <*>.<*> MB"

        if lowered_raw.startswith("running instance usage audit for host "):
            return (
                "Running instance usage audit for host <*> from <*>-<*>-<*> <*>:<*>:<*> "
                "to <*>-<*>-<*> <*>:<*>:<*>. <*> instances."
            )

        if lowered_raw.startswith("final resource view: name="):
            return (
                "Final resource view: name=<*> phys_ram=<*> used_ram=<*> phys_disk=<*> "
                "used_disk=<*> total_vcpus=<*> used_vcpus=<*> pci_stats=[]"
            )

        return value

    def _postprocess_hadoop(self, value: str, raw_log: str) -> str:
        raw = " ".join((raw_log or "").split())

        if re.match(r"^Assigned container container_[^\s]+ to attempt_", raw):
            return "Assigned container container_<*> to attempt_<*>"

        if re.match(r"^Processing the event EventType:\s*CONTAINER_REMOTE_[A-Z_]+", raw):
            return "Processing the event EventType: CONTAINER_REMOTE_<*> for container container_<*> taskAttempt attempt_<*>"

        if re.match(r"^Received completed container container_[^\s]+$", raw):
            return "Received completed container container_<*>"

        if re.match(r"^Failed to renew lease for \[DFSClient_NONMAPREDUCE_", raw):
            return "Failed to renew lease for [DFSClient_NONMAPREDUCE_<*>_<*>] for <*> seconds. Will retry shortly ..."

        if re.match(r"^JVM with ID:\s*jvm_[^\s]+\s+given task:\s+\S+", raw):
            return "JVM with ID: jvm_<*> given task: <*>_<*>"

        if re.match(r"^JVM with ID\s*:\s*jvm_[^\s]+\s+asked for a task$", raw):
            if "ID :" in raw:
                return "JVM with ID : jvm_<*> asked for a task"
            return "JVM with ID: jvm_<*> asked for a task"

        if re.match(r"^TaskAttempt:\s*\[attempt_[^\]]+\]\s+using containerId:\s+\[container_[^\s]+ on NM:\s+\[[^\]]+\]$", raw):
            return "TaskAttempt: [attempt_<*>] using containerId: [container_<*> on NM: [<*>:<*>]"

        if raw.startswith("Before Scheduling:"):
            return (
                "Before Scheduling: PendingReds:<*> ScheduledMaps:<*> ScheduledReds:<*> "
                "AssignedMaps:<*> AssignedReds:<*> CompletedMaps:<*> CompletedReds:<*> "
                "ContAlloc:<*> ContRel:<*> HostLocal:<*> RackLocal:<*>"
            )

        if raw.startswith("After Scheduling:"):
            return (
                "After Scheduling: PendingReds:<*> ScheduledMaps:<*> ScheduledReds:<*> "
                "AssignedMaps:<*> AssignedReds:<*> CompletedMaps:<*> CompletedReds:<*> "
                "ContAlloc:<*> ContRel:<*> HostLocal:<*> RackLocal:<*>"
            )

        task_transition = re.match(
            r"^task_[^\s]+\s+Task Transitioned from\s+([A-Z]+)\s+to\s+([A-Z]+)$",
            raw,
        )
        if task_transition:
            return f"task_<*> Task Transitioned from {task_transition.group(1)} to {task_transition.group(2)}"

        if re.match(r"^getResources\(\) for application_[^:]+:", raw):
            return "getResources() for application_<*>: ask=<*> release=<*> newContainers=<*> finishedContainers=<*> resourceLimit=<*>"

        if re.match(r"^Retrying connect to server:\s+\S+$", raw):
            return "Retrying connect to server: <*>:<*>."

        if re.match(r"^Created MRAppMaster for application appattempt_[^\s]+$", raw):
            return "Created MRAppMaster for application appattempt_<*>"

        if re.match(r"^Started HttpServer2\\$SelectChannelConnectorWithSafeStartup@", raw):
            return "Started HttpServer2$SelectChannelConnectorWithSafeStartup@<*>:<*>"

        if re.match(r"^Adding job token for job_[^\s]+$", raw):
            return "Adding job token for job_<*>"

        if re.match(r"^Input size for job job_[^\s]+ is", raw):
            return "Input size for job job_<*> is <*>. Number of splits generated is <*>"

        job_transition = re.match(
            r"^(job_[^\s]+)Job Transitioned from\s+([A-Z]+)\s+to\s+([A-Z]+)$",
            raw,
        )
        if job_transition:
            return f"job_<*>Job Transitioned from {job_transition.group(2)} to {job_transition.group(3)}"

        if re.match(r"^JOB_CREATE job_[^\s]+$", raw):
            return "JOB_CREATE job_<*>"

        value = re.sub(r"\bcontainer\s+<\*>\b", "container_<*>", value)
        value = re.sub(r"\bjvm\s+<\*>\b", "jvm_<*>", value)
        value = re.sub(r"\btask\s+<\*>\b", "task_<*>", value)
        value = re.sub(r"\bjob\s+<\*>\b", "job_<*>", value)
        value = re.sub(r"\bapplication\s+<\*>\b", "application_<*>", value)
        value = re.sub(r"\bappattempt\s+<\*>\b", "appattempt_<*>", value)
        return value

    def _tree_search(self, rn: _Node, seq: list[str]) -> _LogCluster | None:
        seq_len = len(seq)
        if seq_len not in rn.childD:
            return None

        parentn = rn.childD[seq_len]
        current_depth = 1
        for token in seq:
            if current_depth >= self.depth or current_depth > seq_len:
                break
            if token in parentn.childD:
                parentn = parentn.childD[token]
            elif "<*>" in parentn.childD:
                parentn = parentn.childD["<*>"]
            else:
                return None
            current_depth += 1

        return self._fast_match(parentn.childD, seq)

    def _add_seq_to_prefix_tree(self, rn: _Node, log_clust: _LogCluster) -> None:
        seq_len = len(log_clust.log_template)
        if seq_len not in rn.childD:
            rn.childD[seq_len] = _Node(depth=1, digitOrtoken=seq_len)
        parentn = rn.childD[seq_len]

        current_depth = 1
        for token in log_clust.log_template:
            if current_depth >= self.depth or current_depth > seq_len:
                if len(parentn.childD) == 0:
                    parentn.childD = [log_clust]
                else:
                    parentn.childD.append(log_clust)
                break

            if token not in parentn.childD:
                if not self._has_numbers(token):
                    if "<*>" in parentn.childD:
                        if len(parentn.childD) < self.max_child:
                            parentn.childD[token] = _Node(depth=current_depth + 1, digitOrtoken=token)
                            parentn = parentn.childD[token]
                        else:
                            parentn = parentn.childD["<*>"]
                    else:
                        if len(parentn.childD) + 1 < self.max_child:
                            parentn.childD[token] = _Node(depth=current_depth + 1, digitOrtoken=token)
                            parentn = parentn.childD[token]
                        elif len(parentn.childD) + 1 == self.max_child:
                            parentn.childD["<*>"] = _Node(depth=current_depth + 1, digitOrtoken="<*>")
                            parentn = parentn.childD["<*>"]
                        else:
                            parentn = parentn.childD["<*>"]
                else:
                    if "<*>" not in parentn.childD:
                        parentn.childD["<*>"] = _Node(depth=current_depth + 1, digitOrtoken="<*>")
                    parentn = parentn.childD["<*>"]
            else:
                parentn = parentn.childD[token]

            current_depth += 1

    @staticmethod
    def _seq_dist(seq1: list[str], seq2: list[str]) -> tuple[float, int]:
        sim_tokens = 0
        num_of_par = 0
        for token1, token2 in zip(seq1, seq2):
            if token1 == "<*>":
                num_of_par += 1
                continue
            if token1 == token2:
                sim_tokens += 1
        return float(sim_tokens) / max(len(seq1), 1), num_of_par

    def _fast_match(self, log_clusts: list[_LogCluster], seq: list[str]) -> _LogCluster | None:
        max_sim = -1.0
        max_num_of_para = -1
        max_clust = None

        for log_clust in log_clusts:
            cur_sim, cur_num_of_para = self._seq_dist(log_clust.log_template, seq)
            if cur_sim > max_sim or (cur_sim == max_sim and cur_num_of_para > max_num_of_para):
                max_sim = cur_sim
                max_num_of_para = cur_num_of_para
                max_clust = log_clust

        if max_sim >= self.st:
            return max_clust
        return None

    @staticmethod
    def _get_template(seq1: list[str], seq2: list[str]) -> list[str]:
        return [word if word == seq2[i] else "<*>" for i, word in enumerate(seq1)]

    def add_log_message(self, log_content: str) -> str:
        content = self.preprocess(log_content)
        if not content:
            return ""
        tokens = content.split()
        match_cluster = self._tree_search(self.root, tokens)

        if match_cluster is None:
            new_cluster = _LogCluster(log_template=tokens, log_ids=[self.next_line_id])
            self._add_seq_to_prefix_tree(self.root, new_cluster)
            template = new_cluster.log_template
        else:
            new_template = self._get_template(tokens, match_cluster.log_template)
            match_cluster.log_ids.append(self.next_line_id)
            if new_template != match_cluster.log_template:
                match_cluster.log_template = new_template
            template = match_cluster.log_template

        self.next_line_id += 1
        return self._postprocess(" ".join(template), log_content)


class OfficialDrainBaseline:
    """
    Official-style Drain replay baseline for the thesis rebuild workspace.

    The parser is rebuilt per evaluated case and replays the fixed reference bank,
    so it stays comparable with the existing RQ1 split discipline and avoids
    leaking target cases across evaluations.
    """

    def __init__(self, reference_logs: Iterable[str], dataset: str = "", config: OfficialDrainConfig | None = None):
        self.reference_logs: List[str] = [item for item in reference_logs if isinstance(item, str) and item.strip()]
        self.dataset = dataset
        self.config = config or OfficialDrainConfig()

    def parse(self, target_log: str) -> str:
        parser = _OfficialDrainReplayParser(dataset=self.dataset, config=self.config)
        for ref in self.reference_logs:
            parser.add_log_message(ref)
        return parser.add_log_message(target_log)
