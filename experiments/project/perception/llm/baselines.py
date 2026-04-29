"""Project-owned baseline parsers for perception-layer protocol runs."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from experiments.project.perception.core.models import (
    PerceptionParseResult,
    PerceptionRouteDiagnostics,
)
from experiments.project.perception.core.preprocessing import prepare_runtime_alert
from experiments.project.perception.llm.local_llm import (
    ChatTemplateGenerator,
    LocalLlmGenerationConfig,
    TransformersChatTemplateGenerator,
    build_direct_messages,
    normalize_generated_template,
)
from experiments.project.perception.paths import ProjectPaths
from experiments.project.perception.retrieval.references import iter_reference_rows


@dataclass
class _LogCluster:
    log_template: list[str]
    log_ids: list[int] = field(default_factory=list)


@dataclass
class _Node:
    childD: dict = field(default_factory=dict)
    depth: int = 0
    digitOrtoken: str | int | None = None


@dataclass(frozen=True)
class OfficialDrainConfig:
    depth: int = 4
    st: float = 0.4
    max_child: int = 100


def _regex_sub(pattern: str, repl: str, text: str) -> str:
    return re.sub(pattern, repl, text, flags=re.IGNORECASE)


def _dataset_config(dataset: str) -> OfficialDrainConfig:
    if dataset == "hdfs":
        return OfficialDrainConfig(depth=4, st=0.5, max_child=100)
    if dataset == "hadoop":
        return OfficialDrainConfig(depth=4, st=0.5, max_child=100)
    if dataset == "openstack":
        return OfficialDrainConfig(depth=6, st=0.95, max_child=100)
    return OfficialDrainConfig()


def _normalize_hadoop_family(value: str, raw_log: str = "") -> str:
    lowered = value.lower()
    source = (raw_log or value).lower()
    if source.startswith("before scheduling: pendingreds:"):
        return (
            "Before Scheduling: PendingReds:<*> ScheduledMaps:<*> ScheduledReds:<*> "
            "AssignedMaps:<*> AssignedReds:<*> CompletedMaps:<*> CompletedReds:<*> "
            "ContAlloc:<*> ContRel:<*> HostLocal:<*> RackLocal:<*>"
        )
    if source.startswith("after scheduling: pendingreds:"):
        return (
            "After Scheduling: PendingReds:<*> ScheduledMaps:<*> ScheduledReds:<*> "
            "AssignedMaps:<*> AssignedReds:<*> CompletedMaps:<*> CompletedReds:0 "
            "ContAlloc:<*> ContRel:<*> HostLocal:<*> RackLocal:<*>"
        )
    if source.startswith("kind: yarn_am_rm_token"):
        return (
            "Kind: YARN_AM_RM_TOKEN, Service: , Ident: (appAttemptId { application_id "
            "{ id: <*> cluster_timestamp: <*> } attemptId: <*> } keyId: <*>)"
        )
    if source.startswith("job_create job_"):
        return "JOB_CREATE job_<*>"
    if "jvm with id :" in source and "asked for a task" in source:
        return "JVM with ID : jvm_<*> asked for a task"
    if "jvm with id:" in source and "given task:" in source:
        return "JVM with ID: jvm_<*> given task: <*>_<*>"
    if "failed to renew lease for" in source and "retry shortly" in source:
        return "Failed to renew lease for [DFSClient_NONMAPREDUCE_<*>_<*>] for <*> seconds.  Will retry shortly ..."
    if source.startswith("taskattempt: [attempt_") and "containerid" in source:
        return "TaskAttempt: [attempt_<*>] using containerId: [container_<*>_<*>_<*>_<*> on NM: [<*>:<*>]"
    if ("resourceslimit" in source or source.startswith("getresources() for application_")) and "newcontainers" in source and "finishedcontainers" in source:
        return (
            "getResources() for application_<*>: ask=<*> release= <*> newContainers=<*> "
            "finishedContainers=<*> resourcelimit=<memory:<*>, vCores:<*>> knownNMs=<*>"
        )
    if source.startswith("progress of taskattempt") or ("progress of" in source and re.search(r"\bis\s*:?", source)):
        return "Progress of TaskAttempt attempt_<*> is : <*>.<*>"
    if "assigned container" in source and ("attempt_" in source or "container_" in source):
        return "Assigned container container_<*> to attempt_<*>"
    if "received completed container" in source and "container_" in source:
        return "Received completed container container_<*>"
    if "container_remote_" in source and ("for container" in source or "container_" in source):
        return "Processing the event EventType: CONTAINER_REMOTE_<*> for container container_<*> taskAttempt attempt_<*>"
    if (
        "task state moved new->scheduled" in source
        or "task transitioned from new to scheduled" in source
        or "task transitioned from new->scheduled to running" in source
        or source == "transitioned from new to scheduled"
    ):
        return "task_<*> Task Transitioned from NEW to SCHEDULED"
    if "task transitioned from scheduled to running" in source:
        return "task_<*> Task Transitioned from SCHEDULED to RUNNING"
    if "task transitioned from running to succeeded" in source:
        return "task_<*> Task Transitioned from RUNNING to SUCCEEDED"
    if "retrying connect to server:" in source:
        return (
            "Retrying connect to server: <*>:<*>. Already tried <*> time(s); "
            "retry policy is RetryUpToMaximumCountWithFixedSleep(maxRetries=<*>, "
            "sleepTime=<*> MILLISECONDS)"
        )
    if "defaultspeculator.addspeculativeattempt" in source and "task_" in source:
        return "DefaultSpeculator.addSpeculativeAttempt -- we are speculating task_<*>"
    if "scheduling a redundant attempt for task task_" in source:
        return "Scheduling a redundant attempt for task task_<*>"
    if "starting socket reader #" in source and "for port" in source:
        return "Starting Socket Reader #<*> for port <*>"
    if "created mrappmaster for application appattempt_" in source:
        return "Created MRAppMaster for application appattempt_<*>"
    if "extract jar:file:" in source and " to " in source:
        return "Extract jar:file:<*> to <*>"
    if "started httpserver2$selectchannelconnectorwithsafestartup@" in source:
        return "Started HttpServer2$SelectChannelConnectorWithSafeStartup@<*>:<*>"
    if "event writer setup for jobid: job_" in source:
        return "Event Writer setup for JobId: job_<*>, File: hdfs://<*>"
    if "adding job token for job_" in source:
        return "Adding job token for job_<*> to jobTokenSecretManager"
    if "web app /mapreduce started at" in source:
        return "Web app /mapreduce started at <*>"
    if "not uberizing job_" in source and "because:" in source:
        return "Not uberizing job_<*> because: <*>"
    if "input size for job job_" in source and "number of splits =" in source:
        return "Input size for job job_<*> = <*>. Number of splits = <*>"
    if "number of reduces for job job_" in source:
        return "Number of reduces for job job_<*> = <*>"
    if "job transitioned from new to inited" in source:
        return "job_<*>Job Transitioned from NEW to INITED"
    if "mrappmaster launching normal, non-uberized, multi-container job job_" in source:
        return "MRAppMaster launching normal, non-uberized, multi-container job job_<*>."
    if "job transitioned from inited to setup" in source:
        return "job_<*>Job Transitioned from INITED to SETUP"
    if "job transitioned from setup to running" in source:
        return "job_<*>Job Transitioned from SETUP to RUNNING"
    if "the job-jar file on the remote fs is hdfs://" in source:
        return "The job-jar file on the remote FS is hdfs://<*>"
    return value


def _preprocess_patterns(dataset: str) -> list[tuple[str, str]]:
    if dataset == "hdfs":
        return [
            (r"blk_-?\d+", "blk_<*>"),
            (r"(\d+\.){3}\d+:\d+", "<*>:<*>"),
            (r"(\d+\.){3}\d+", "<*>"),
            (r"\b\d+\b", "<*>"),
        ]
    if dataset == "openstack":
        return [
            (r"((\d+\.){3}\d+,?)+", "<*>"),
            (r"/.+?\s", "<*> "),
            (r"\d+", "<*>"),
        ]
    if dataset == "hadoop":
        return [
            (r"\bcontainer_\d+(?:_\d+){3}\b", "container_<*>_<*>_<*>_<*>"),
            (r"\bapplication_\d+(?:_\d+){2}\b", "application_<*>"),
            (r"\bappattempt_\d+(?:_\d+){3}\b", "appattempt_<*>"),
            (r"\bjob_\d+(?:_\d+){2}\b", "job_<*>"),
            (r"\btask_\d+(?:_\d+){2}_[mr]_\d+\b", "task_<*>"),
            (r"\battempt_\d+(?:_\d+){2}_[mr]_\d+(?:_\d+)?\b", "attempt_<*>"),
            (r"\bjvm_\d+(?:_\d+){2}_[mr]_\d+\b", "jvm_<*>"),
            (r"\bDFSClient_NONMAPREDUCE_-?\d+_\d+\b", "DFSClient_NONMAPREDUCE_<*>_<*>"),
            (r"\bCONTAINER_REMOTE_[A-Z_]+\b", "CONTAINER_REMOTE_<*>"),
            (r"(\d+\.){3}\d+", "<*>"),
        ]
    return [(r"\d+", "<*>")]


class _OfficialDrainReplayParser:
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
        if not value:
            return value

        if self.dataset == "hdfs":
            lowered = value.lower()
            if "verification succeeded" in lowered:
                return "Verification succeeded for blk_<*>"
            if "packetresponder" in lowered:
                return "PacketResponder <*> for block blk_<*> terminating"
            if "receiving block" in lowered and "src:" in lowered:
                return "Receiving block blk_<*> src: /<*>:<*> dest: /<*>:<*>"
            if "received block" in lowered and "from /" in lowered:
                return "Received block blk_<*> of size <*> from /<*>"
            if "served block" in lowered and " to /" in lowered:
                return "<*>:<*> Served block blk_<*> to /<*>"
            if "starting thread to transfer block" in lowered and "blk_" in lowered:
                return "<*>:<*> Starting thread to transfer block blk_<*> to <*>:<*>"
            if " ask " in lowered and " replicate " in lowered and "blk_" in lowered:
                return "BLOCK* ask <*>:<*> to replicate blk_<*> to datanode(s) <*>:<*>"
            if "namesystem.allocateblock" in lowered:
                return "BLOCK* NameSystem.allocateBlock: /<*>/part-<*>. blk_<*>"
            if "namesystem.addstoredblock" in lowered and "added to" in lowered:
                return (
                    "BLOCK* NameSystem.addStoredBlock: blockMap updated: "
                    "<*>:<*> is added to blk_<*> size <*>"
                )
            return value

        if self.dataset != "openstack":
            if self.dataset == "hadoop":
                return _normalize_hadoop_family(value, raw_log)
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
        if "total disk:" in lowered_raw:
            return "[instance: <*>] Total disk: <*> GB, used: <*>.<*> GB"
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


def _normalize_direct_prediction(template: str, dataset_id: str) -> str:
    value = " ".join((template or "").split())
    if not value:
        return value

    dataset_key = dataset_id.lower()
    lowered = value.lower()

    if dataset_key == "hdfs":
        value = value.replace("blk<*>", "blk_<*>")
        value = re.sub(r"\bReceived block <\*><\*>\b", "Received block blk_<*>", value)
        value = re.sub(r"\bblock-id:?\s*<\*>\b", "blk_<*>", value, flags=re.IGNORECASE)
        lowered = value.lower()

        if "verification succeeded" in lowered:
            return "Verification succeeded for blk_<*>"
        if "packetresponder" in lowered and ("terminating" in lowered or "for block" in lowered):
            return "PacketResponder <*> for block blk_<*> terminating"
        if "receiving block" in lowered and "src:" in lowered:
            return "Receiving block blk_<*> src: /<*>:<*> dest: /<*>:<*>"
        if "received block" in lowered and "from /" in lowered:
            return "Received block blk_<*> of size <*> from /<*>"
        if "served block" in lowered and " to /" in lowered:
            return "<*>:<*> Served block blk_<*> to /<*>"
        if "starting thread to transfer block" in lowered and "blk_" in lowered:
            return "<*>:<*> Starting thread to transfer block blk_<*> to <*>:<*>"
        if " ask " in lowered and " replicate " in lowered and "blk_" in lowered:
            return "BLOCK* ask <*>:<*> to replicate blk_<*> to datanode(s) <*>:<*>"
        if "namesystem.allocateblock" in lowered:
            return "BLOCK* NameSystem.allocateBlock: /<*>/part-<*>. blk_<*>"
        if "namesystem.addstoredblock" in lowered and "added to" in lowered:
            return (
                "BLOCK* NameSystem.addStoredBlock: blockMap updated: "
                "<*>:<*> is added to blk_<*> size <*>"
            )
        return value

    if dataset_key == "openstack":
        method_match = re.search(r'"?(GET|POST|DELETE)\s+/[^"]+', value, flags=re.IGNORECASE)
        if method_match:
            method = method_match.group(1).upper()
            return f'<*> "{method} <*>" status: <*> len: <*> time: <*>.<*>'
        if "during sync_power_state" in lowered and "pending task" in lowered and "spawning" in lowered:
            return "[instance: <*>] During sync_power_state the instance has a pending task (spawning). Skip."
        if "instance spawned successfully" in lowered:
            return "[instance: <*>] Instance spawned successfully."
        if "instance destroyed successfully" in lowered:
            return "[instance: <*>] Instance destroyed successfully."
        if "creating image" in lowered:
            return "[instance: <*>] Creating image"
        if "claim successful" in lowered:
            return "[instance: <*>] Claim successful"
        if "network-vif-plugged" in lowered and "for instance" in lowered:
            return "Creating event network-vif-plugged:<*>-<*>-<*>-<*>-<*> for instance <*>"
        if "terminating instance" in lowered:
            return "[instance: <*>] Terminating instance"
        if "disk limit not specified, defaulting to unlimited" in lowered:
            return "[instance: <*>] disk limit not specified, defaulting to unlimited"
        if "vcpu limit not specified, defaulting to unlimited" in lowered:
            return "[instance: <*>] vcpu limit not specified, defaulting to unlimited"
        if "total vcpu:" in lowered and "used:" in lowered:
            return "[instance: <*>] Total vcpu: <*> VCPU, used: <*>.<*> VCPU"
        if "total disk:" in lowered and "used:" in lowered:
            return "[instance: <*>] Total disk: <*> GB, used: <*>.<*> GB"
        if "memory limit:" in lowered and "free:" in lowered:
            return "[instance: <*>] memory limit: <*>.<*> MB, free: <*>.<*> MB"
        return value

    if dataset_key == "hadoop":
        return _normalize_hadoop_family(value)

    return value


class DrainReplayProtocolParser:
    """Project-owned official-style Drain replay parser for protocol runs."""

    def __init__(
        self,
        manifest_id: str,
        *,
        paths: ProjectPaths | None = None,
    ) -> None:
        self.manifest_id = manifest_id
        self.paths = paths
        self.reference_logs_by_dataset: dict[str, list[str]] = {}
        for row in iter_reference_rows(manifest_id, paths=paths):
            clean_alert = prepare_runtime_alert(row.clean_alert, row.dataset_id)
            self.reference_logs_by_dataset.setdefault(row.dataset_id.lower(), []).append(
                clean_alert
            )

    def reset_cache(self) -> None:
        return None

    def parse(self, query_text: str, dataset_id: str) -> PerceptionParseResult:
        dataset_key = dataset_id.lower()
        parser = _OfficialDrainReplayParser(dataset_key)
        refs = self.reference_logs_by_dataset.get(dataset_key, [])
        for ref in refs:
            parser.add_log_message(ref)
        template = parser.add_log_message(query_text)
        return PerceptionParseResult(
            template=template,
            route="drain_replayed_refs",
            diagnostics=PerceptionRouteDiagnostics(
                route="drain_replayed_refs",
                query_text=query_text,
                query_chars=len(query_text),
                best_score=None,
                candidate_count=len(refs),
            ),
        )


class DirectLocalLlmParser:
    """Project-owned direct local-Qwen baseline for protocol runs."""

    def __init__(
        self,
        generator: ChatTemplateGenerator | None = None,
    ) -> None:
        self.generator = generator or TransformersChatTemplateGenerator(
            LocalLlmGenerationConfig()
        )

    def reset_cache(self) -> None:
        return None

    def warmup(self, dataset_id: str) -> None:
        if hasattr(self.generator, "warmup"):
            self.generator.warmup(dataset_id)

    def parse(self, query_text: str, dataset_id: str) -> PerceptionParseResult:
        messages = build_direct_messages(query_text, dataset_id)
        response, latency_ms = self.generator.generate(messages, dataset_id)
        template = _normalize_direct_prediction(
            normalize_generated_template(response),
            dataset_id,
        )
        return PerceptionParseResult(
            template=template,
            route="llm_direct",
            diagnostics=PerceptionRouteDiagnostics(
                route="llm_direct",
                query_text=query_text,
                query_chars=len(query_text),
                best_score=None,
                candidate_count=0,
                metadata={"llm_latency_ms": round(latency_ms, 3)},
            ),
        )
