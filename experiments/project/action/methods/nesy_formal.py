"""Structured action method."""

from __future__ import annotations

from dataclasses import replace

from experiments.project.action.core.schema import ActionCase, ActionQuery, BenchmarkBundle, LabelEntry
from experiments.project.action.methods.formal_config import FORMAL_LOCAL_COMBINED, FORMAL_NESY, FORMAL_RAG_SUPPORT
from experiments.project.action.methods.formal_surfaces import render_graph_evidence, render_incident_surface, render_retrieved_case
from experiments.project.action.methods.postprocess import normalize_prediction
from experiments.project.action.methods.prompts import json_output_contract, render_labelbook
from experiments.project.action.methods.retrieval import RetrievedCase, topk_combined, topk_raw

def _build_messages(policy: str, query: ActionQuery, *, root_entries: list[LabelEntry], action_entries: list[LabelEntry], graph_text: str, retrieved_text: str) -> list[dict[str, str]]:
    if policy == 'hdfs_family_action':
        evidence_policy = (
            'Treat the graph-derived causal facts as higher-priority evidence. '
            'First use the HDFS family reference to choose the root family, then use the HDFS action tie-break rules below to choose the action inside that root family.'
        )
        extra_blocks = [
            'HDFS family reference',
            '\n'.join([
                '- hdfs_root_type_0: look for E6 and E16 together while E20 is absent, often with E2 and strong E21 closure.',
                '- hdfs_root_type_1: look for E27 together with E4/E3/E2 before the final E23/E21 closure.',
                '- hdfs_root_type_21: look for E20 with stronger E4/E3 activity, but without the distinctive E27 or E28 marker.',
                '- hdfs_root_type_4: look for the distinctive E28 marker, usually with E4 and E26 in the suffix.',
                '- hdfs_root_type_5: look for heavier E26 and E20 presence with E11/E9 echoes and weak or absent E4 activity.',
            ]),
            'HDFS action tie-break rules',
            '\n'.join([
                '- For hdfs_root_type_0: if a repeated E23 block appears before the final E21 closure, prefer the e23->e23 action. Use e21->e21 only when the repeated E21 block appears without a preceding repeated E23 block.',
                '- For hdfs_root_type_1: if the tail has strong E2 support, especially E2 x3 or an E4->E2 bridge before the E23/E21 closure, prefer the e2->e2 action over the generic e23 action.',
                '- For hdfs_root_type_4: if E4 dominates the tail, especially E4 x4 or repeated E4->E4 transitions, prefer the e4->e4 action.',
                '- For hdfs_root_type_4: if E23 dominates the tail and the bridge includes E2->E23 with weaker E4 support, prefer the e23->e23 action.',
                '- For hdfs_root_type_21 vs hdfs_root_type_5: prefer type 21 whenever E20 coexists with explicit E4 support. Use type 5 only when E20 is present, E4 is absent, and E26 plus E11/E9 echoes dominate.',
                '- Do not default to e23->e23 when a more discriminative within-root tail pattern is present.',
            ]),
        ]
    elif policy == 'openstack_root_graph_action_hint':
        evidence_policy = (
            'Use the OpenStack root-graph and action-hint rules below. '
            'Choose root_label from graph-derived component and server cues first. '
            'After the root_label is fixed, use retrieved action hints only for sync and detail decisions. '
            'Do not let retrieved action hints override the graph-based root decision.'
        )
        extra_blocks = [
            'OpenStack root-graph and action-hint reference',
            '\n'.join([
                '- Root step: read the graph-derived causal facts first. Use the first Component focus entry as the primary root cue.',
                '- Server override: if component:server or component:server_external_events is explicit in the graph-derived facts, prefer the server root unless instance_sync makes a sync imagecache path explicit.',
                '- Imagecache lock: if the first Component focus entry is imagecache, do not switch to compute_manager just because claim_successful or creating_image appears early. Keep imagecache unless a server override is explicit.',
                '- Compute-manager step: choose compute_manager only when the first Component focus entry is compute_manager and there is no explicit server override.',
                '- Action step: after the root_label is fixed, use retrieved action_hint lines only for sync and detail. If the retrieved action hints consistently show sync_detail_*, choose sync_* under the selected root. Otherwise choose no_sync_* under the selected root.',
                '- Detail step: use detail_high only when the retrieved action hints or graph API shape clearly indicate detail_high; otherwise keep detail_low.',
            ]),
        ]
    elif policy == 'hadoop_machine_anchor_workload':
        evidence_policy = (
            'Choose the root_label from the current incident evidence first. '
            'Use retrieved troubleshooting cases only as supporting examples after an incident-first hypothesis exists. '
            'Treat deadnodes, createblockoutputstream, remote block reader, and bad connect ack with firstbadlink as decisive machine_down cues, not network cues. '
            'Treat repeated returned-by-containermanager lines as supporting machine_down evidence when any explicit machine cue is present. '
            'Treat could not delete hdfs_path as weak aftermath only. '
            'Treat shuffling to disk, maxsingleshufflelimit, on disk map outputs, finalmerge, or merging as soft disk evidence only. '
            'Choose disk_full only for explicit not enough space on the disk or spill failed cues, or when repeated shuffle-pressure clearly dominates and machine cues are absent. '
            'Choose network_disconnection only when timed out or forcibly closed by remote host dominates and explicit machine_down or hard disk-full cues are absent. '
            'Do not let generic graph disk tokens override explicit machine_down cues. '
            'After the root_label is fixed, use explicit workload tokens from the current incident first. '
            'If the current incident only gives a generic workload alias such as batchjob or batch_job, use retrieved workload hints only to choose between pagerank and wordcount.'
        )
        extra_blocks = [
            'Hadoop machine-anchor and workload-hint reference',
            '\n'.join([
                '- Read the current incident evidence first and anchor on explicit machine_down cues before generic graph or retrieval hints.',
                '- Treat deadnodes, createblockoutputstream, remote block reader, and bad connect ack with firstbadlink as explicit machine_down cues, not network cues.',
                '- Treat repeated returned-by-containermanager lines as supporting machine_down evidence when any explicit machine cue is present.',
                '- Treat could not delete hdfs_path as weak aftermath only. It does not prove disk_full by itself.',
                '- Treat shuffling to disk, maxsingleshufflelimit, on disk map outputs, finalmerge, or merging as soft disk evidence only.',
                '- Choose disk_full only for explicit not-enough-space or spill-failed failures, or when repeated shuffle-pressure clearly dominates while machine cues are absent.',
                '- Choose network_disconnection only when timeout or forcibly-closed cues dominate and explicit machine_down or hard disk-full cues are absent.',
                '- After root_label is fixed, use explicit workload tokens from the current incident first. If only a generic batch alias remains, use retrieved workload hints to choose between pagerank and wordcount.',
            ]),
        ]
    else:
        raise ValueError(policy)

    system = (
        'You are an incident diagnosis assistant. '
        'Use only the current incident evidence, retrieved troubleshooting cases, and graph-derived causal facts explicitly listed below. '
        + evidence_policy
        + ' First choose root_label from the combined evidence, then choose action_label that is consistent with the selected root_label. '
        + json_output_contract(query.base_case.dataset)
    )
    user = '\n\n'.join([
        render_labelbook(root_entries, title='Allowed root labels'),
        render_labelbook(action_entries, title='Allowed action labels'),
        *extra_blocks,
        f'Current incident evidence\n{query.incident_text}',
        f'Graph-derived causal facts\n{graph_text}',
        f'Retrieved troubleshooting cases\n{retrieved_text}',
    ])
    return [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}]

class NeSyFormalRunner:
    method_name = 'nesy'

    def __init__(self, bundle: BenchmarkBundle) -> None:
        self._bundle = bundle

    def _retrieve(self, query: ActionQuery, *, support_cases: list[ActionCase]) -> list[RetrievedCase]:
        dataset = query.base_case.dataset
        spec = FORMAL_NESY[dataset]
        if spec.retrieval_mode == 'local_combined':
            config = FORMAL_LOCAL_COMBINED[dataset]
            return topk_combined(query, support_cases, raw_feature_name=config['raw'], graph_feature_name=config['graph'], raw_weight=float(config['raw_weight']), graph_weight=float(config['graph_weight']), k=int(config['k']))
        if spec.retrieval_mode == 'rag_support':
            config = FORMAL_RAG_SUPPORT[dataset]
            return topk_raw(query, support_cases, feature_name=config['feature_name'], k=int(config['k']))
        raise ValueError(spec.retrieval_mode)

    def prepare(self, query: ActionQuery, *, support_cases: list[ActionCase]) -> dict:
        dataset = query.base_case.dataset
        spec = FORMAL_NESY[dataset]
        labels = self._bundle.labelbook[dataset]
        retrieved = self._retrieve(query, support_cases=support_cases)
        incident_text = render_incident_surface(query, style=spec.incident_style)
        prompt_query = replace(query, incident_text=incident_text)
        graph_text = render_graph_evidence(query.base_case, graph_pack=spec.graph_pack, graph_style=spec.graph_style, noise_level=query.noise_level)
        retrieved_text = '\n'.join(render_retrieved_case(item, style=spec.retrieved_style) for item in retrieved)
        return {
            'method': self.method_name,
            'query_case_id': query.base_case.case_id,
            'messages': _build_messages(spec.policy, prompt_query, root_entries=labels['root'], action_entries=labels['action'], graph_text=graph_text, retrieved_text=retrieved_text),
            'metadata': {
                'dataset': dataset,
                'incident_text': incident_text,
                'graph_text': graph_text,
                'retrieved_text': retrieved_text,
                'retrieved_case_ids': [item.case.case_id for item in retrieved],
            },
        }

    def finalize_prediction(self, prediction: dict, *, incident_text: str | None) -> dict:
        return normalize_prediction(prediction, incident_text=incident_text)
