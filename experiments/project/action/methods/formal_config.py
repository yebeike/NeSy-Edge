"""Runtime selections for action methods."""

from __future__ import annotations

from dataclasses import dataclass

@dataclass(frozen=True)
class FormalVanillaSpec:
    incident_style: str
    prompt_style: str
    noise_variant: str

@dataclass(frozen=True)
class FormalNeSySpec:
    incident_style: str
    retrieval_mode: str
    retrieved_style: str
    graph_pack: str
    graph_style: str
    policy: str
    noise_variant: str

@dataclass(frozen=True)
class FormalRagConfig:
    root_feature: str
    action_feature: str
    k: int
    action_strategy: str
    score_mode: str
    root_floor: float | None
    action_floor: float | None
    root_margin: float | None = None
    action_margin: float | None = None

FORMAL_NOISE_VARIANTS = {'HDFS': 'v1', 'Hadoop': 'v2', 'OpenStack': 'v1'}

FORMAL_VANILLA = {
    'HDFS': FormalVanillaSpec('hdfs_trace_digest', 'hdfs_root_then_action', 'v1'),
    'Hadoop': FormalVanillaSpec('hadoop_root_boundary_excerpt', 'hadoop_mid_boundary', 'v2'),
    'OpenStack': FormalVanillaSpec('openstack_anchor_sequence_excerpt', 'openstack_component_then_action', 'v1'),
}

FORMAL_NESY = {
    'HDFS': FormalNeSySpec('hdfs_family_rule_digest', 'local_combined', 'incident_head', 'all', 'bullets', 'hdfs_family_action', 'v1'),
    'Hadoop': FormalNeSySpec('hadoop_root_boundary_excerpt', 'rag_support', 'hadoop_workload_hint', 'phase_terminal', 'adaptive_compact', 'hadoop_machine_anchor_workload', 'v2'),
    'OpenStack': FormalNeSySpec('openstack_body_anchor_digest', 'local_combined', 'openstack_action_hint', 'severity', 'bullets', 'openstack_root_graph_action_hint', 'v1'),
}

FORMAL_RAG = {
    'HDFS': FormalRagConfig('full', 'full', 3, 'global_top1', 'raw', 0.05, 0.05),
    'Hadoop': FormalRagConfig('late', 'semantic', 5, 'filtered_vote', 'raw', 0.0, 0.0, 0.03, None),
    'OpenStack': FormalRagConfig('hybrid_event_bi_w4', 'hybrid_event_bi_w2', 3, 'filtered_vote', 'raw', 0.05, 0.0, None, None),
}

FORMAL_LOCAL_COMBINED = {
    'HDFS': {'raw': 'full', 'graph': 'suffix12', 'raw_weight': 0.5, 'graph_weight': 0.5, 'k': 5},
    'OpenStack': {'raw': 'token', 'graph': 'anchor_position_api', 'raw_weight': 0.5, 'graph_weight': 0.5, 'k': 5},
}

FORMAL_RAG_SUPPORT = {
    'HDFS': {'feature_name': 'full', 'k': 5},
    'Hadoop': {'feature_name': 'semantic', 'k': 5},
    'OpenStack': {'feature_name': 'token', 'k': 5},
}
