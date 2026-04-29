"""Vanilla action method."""

from __future__ import annotations

from experiments.project.action.core.schema import ActionQuery, BenchmarkBundle, LabelEntry
from experiments.project.action.methods.formal_config import FORMAL_VANILLA
from experiments.project.action.methods.formal_surfaces import render_incident_surface
from experiments.project.action.methods.postprocess import normalize_prediction
from experiments.project.action.methods.prompts import json_output_contract, render_labelbook

def _build_messages(prompt_style: str, query: ActionQuery, *, root_entries: list[LabelEntry], action_entries: list[LabelEntry]) -> list[dict[str, str]]:
    if prompt_style == 'hdfs_root_then_action':
        system = (
            'You are the Vanilla LLM baseline for incident diagnosis. '
            'Use only the current incident evidence. '
            'Do not map directly from one repeated tail token to a root label. '
            'Treat the root_label as a trace-family decision and the action_label as a root-conditioned repair decision. '
            + json_output_contract(query.base_case.dataset)
        )
        checklist = [
            '- Use the entire trace family to choose root_label.',
            '- Use the tail signature only after the root_label is chosen.',
            '- If several action labels share a similar tail signature, prefer the one that matches the selected root_label.',
        ]
    elif prompt_style == 'hadoop_mid_boundary':
        system = (
            'You are the Vanilla LLM baseline for incident diagnosis. '
            'Use only the current incident evidence. '
            'Choose root_label by following the ordered Hadoop machine-boundary rules exactly. '
            'Keep machine_down and network_disconnection separate, but use a middle-strength machine rule: stronger than soft boundary, weaker than machine fallback. '
            + json_output_contract(query.base_case.dataset)
        )
        checklist = [
            '1. If deadnodes, createblockoutputstream failure, or bad connect ack with firstbadlink is explicit, choose machine_down.',
            '2. If hard disk cues are explicit, choose disk_full.',
            '3. If hard cues are absent, repeated returned-by-containermanager lines can break ties toward machine_down only when weak machine cues are also present, such as add-to-deadnodes fallout or remote block reader failure.',
            '4. A single add-to-deadnodes or remote block reader mention is not enough by itself. Without repeated port-return support, compare it against disk and network evidence instead of locking machine_down.',
            '5. Treat could not delete hdfs_path as weak aftermath only. It never proves disk_full by itself.',
            '6. Choose disk_full when repeated shuffle-pressure cues clearly dominate and the machine rules above are not satisfied.',
            '7. Choose network_disconnection when timeout or forcibly-closed cues dominate and the machine rules above are not satisfied.',
            '8. If the evidence is mixed and no hard machine cue exists, prefer the softer non-machine explanation over machine_down.',
            '- After root_label is chosen, choose action_label inside the selected root family and copy the visible workload context exactly.',
        ]
    elif prompt_style == 'openstack_component_then_action':
        system = (
            'You are the Vanilla LLM baseline for incident diagnosis. '
            'Use only the current incident evidence. '
            'Choose the component-focused root_label first, then choose the action_label by sync flag and API-detail bucket. '
            'Do not jump directly from one cleanup token to the final action. '
            + json_output_contract(query.base_case.dataset)
        )
        checklist = [
            '- First identify the main component focus: imagecache, server, or compute_manager.',
            '- Then identify whether instance_sync is present or absent.',
            '- Then use API detail count or detail bucket to decide detail_high vs detail_low.',
            '- Keep the selected action inside the selected component-focused root family.',
            '- Generic image-cache cleanup lines are weaker than an explicit compute_manager or server focus.',
        ]
    else:
        raise ValueError(prompt_style)
    user = '\n\n'.join([
        render_labelbook(root_entries, title='Allowed root labels'),
        render_labelbook(action_entries, title='Allowed action labels'),
        'Decision checklist',
        '\n'.join(checklist),
        f'Current incident evidence\n{query.incident_text}',
    ])
    return [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}]

class VanillaFormalRunner:
    method_name = 'vanilla'

    def __init__(self, bundle: BenchmarkBundle) -> None:
        self._bundle = bundle

    def prepare(self, query: ActionQuery) -> dict:
        dataset = query.base_case.dataset
        spec = FORMAL_VANILLA[dataset]
        labels = self._bundle.labelbook[dataset]
        incident_text = render_incident_surface(query, style=spec.incident_style)
        prompt_query = ActionQuery(
            base_case=query.base_case,
            noise_level=query.noise_level,
            seed=query.seed,
            incident_text=incident_text,
            raw_features=query.raw_features,
            metadata=dict(query.metadata),
        )
        return {
            'method': self.method_name,
            'query_case_id': query.base_case.case_id,
            'messages': _build_messages(spec.prompt_style, prompt_query, root_entries=labels['root'], action_entries=labels['action']),
            'metadata': {'dataset': dataset, 'incident_text': incident_text, 'prompt_style': spec.prompt_style},
        }

    def finalize_prediction(self, prediction: dict, *, incident_text: str | None) -> dict:
        return normalize_prediction(prediction, incident_text=incident_text)
