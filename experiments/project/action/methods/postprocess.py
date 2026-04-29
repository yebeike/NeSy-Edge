"""Prediction post-processing helpers."""

from __future__ import annotations

from typing import Any

def _hadoop_workload_signal_scores(incident_text: str | None) -> tuple[int, int, int, int]:
    text = (incident_text or '').lower()
    explicit_word = text.count('wordcount') + text.count('count_job')
    explicit_rank = text.count('pagerank') + text.count('rank_job')
    wordcount_markers = (
        'maxsingleshufflelimit',
        'shuffling to ',
        'could not delete hdfs_path',
        'on disk map outputs',
        'spill failed',
        'there is not enough space on the disk',
        'container released on a lost node',
        'diagnostics report',
    )
    pagerank_markers = (
        'socket reader',
        'communication ',
        'wrapexception',
        'destination host',
        'local host',
        'forcibly closed by the remote host',
        'no route to host',
        'address change detected',
        'remote host',
    )
    wordcount_score = sum(text.count(marker) for marker in wordcount_markers)
    pagerank_score = sum(text.count(marker) for marker in pagerank_markers)
    return explicit_word, explicit_rank, wordcount_score, pagerank_score

def _infer_hadoop_workload_suffix(incident_text: str | None) -> str:
    explicit_word, explicit_rank, wordcount_score, pagerank_score = _hadoop_workload_signal_scores(incident_text)
    if explicit_word > 0:
        return 'wordcount'
    if explicit_rank > 0:
        return 'pagerank'
    if wordcount_score > pagerank_score:
        return 'wordcount'
    return 'pagerank'

def normalize_prediction(prediction: dict[str, Any], *, incident_text: str | None) -> dict[str, Any]:
    action_value = prediction.get('action_label')
    if not isinstance(action_value, str):
        return prediction
    if not action_value.startswith('hadoop_root_') or '__repair__' not in action_value:
        return prediction
    root_prefix, suffix = action_value.split('__repair__', 1)
    suffix = suffix.strip().lower()
    if suffix in {'batch_job', 'count_job', 'rank_job'}:
        prediction['action_label'] = f'{root_prefix}__repair__{_infer_hadoop_workload_suffix(incident_text)}'
    return prediction
