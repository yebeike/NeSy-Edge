"""Evaluation helpers."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from experiments.project.action.core.schema import ActionCase

def evaluate_prediction(case: ActionCase, prediction: dict[str, Any] | None) -> dict[str, Any]:
    root_label = prediction.get('root_label') if prediction else None
    action_label = prediction.get('action_label') if prediction else None
    rca_correct = int(root_label == case.root_label) if root_label is not None else None
    e2e_correct = (
        int(root_label == case.root_label and action_label == case.action_label)
        if root_label is not None and action_label is not None
        else None
    )
    return {
        'predicted_root_label': root_label,
        'predicted_action_label': action_label,
        'rca_correct': rca_correct,
        'e2e_correct': e2e_correct,
    }

def aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped_rows: dict[tuple[str, str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped_rows[(row['method'], row['dataset'], float(row['noise_level']))].append(row)

    per_method_dataset_noise: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    totals: dict[str, dict[str, int]] = defaultdict(lambda: {'count': 0, 'rca_correct': 0, 'e2e_correct': 0})
    for (method, dataset, noise_level), items in sorted(grouped_rows.items()):
        rca_values = [item['evaluation']['rca_correct'] for item in items if item['evaluation']['rca_correct'] is not None]
        e2e_values = [item['evaluation']['e2e_correct'] for item in items if item['evaluation']['e2e_correct'] is not None]
        payload = {
            'count': len(items),
            'rca_correct': int(sum(rca_values)),
            'e2e_correct': int(sum(e2e_values)),
            'rca': round(sum(rca_values) / max(len(items), 1), 4),
            'e2e': round(sum(e2e_values) / max(len(items), 1), 4),
        }
        per_method_dataset_noise.setdefault(method, {}).setdefault(dataset, {})[f'{noise_level:.1f}'] = payload
        totals[method]['count'] += len(items)
        totals[method]['rca_correct'] += payload['rca_correct']
        totals[method]['e2e_correct'] += payload['e2e_correct']

    overall = {
        method: {
            'count': payload['count'],
            'rca_correct': payload['rca_correct'],
            'e2e_correct': payload['e2e_correct'],
            'rca': round(payload['rca_correct'] / max(payload['count'], 1), 4),
            'e2e': round(payload['e2e_correct'] / max(payload['count'], 1), 4),
        }
        for method, payload in sorted(totals.items())
    }
    return {
        'row_count': len(rows),
        'overall': overall,
        'per_method_dataset_noise': per_method_dataset_noise,
    }
