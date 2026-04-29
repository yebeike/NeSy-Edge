"""Retrieval-based action method."""

from __future__ import annotations

import re
from collections import Counter, defaultdict

from experiments.project.action.core.schema import ActionCase, ActionQuery, BenchmarkBundle
from experiments.project.action.core.text import cosine_similarity
from experiments.project.action.methods.formal_config import FORMAL_RAG, FormalRagConfig

_TEXT_TOKEN_RE = re.compile(r'[A-Za-z0-9_<>:-]+')
_SENSITIVE_TEXT_TERMS = {'warning', 'event', 'notice', 'signal'}

def _dataset_cases(bundle: BenchmarkBundle) -> dict[str, list[ActionCase]]:
    payload: dict[str, list[ActionCase]] = defaultdict(list)
    for case in bundle.cases:
        payload[case.dataset].append(case)
    for dataset in payload:
        payload[dataset].sort(key=lambda item: item.case_id)
    return dict(payload)

def _text_tokens(text: str) -> list[str]:
    return [token.lower() for token in _TEXT_TOKEN_RE.findall(text)]

def _ngram_counter(tokens: list[str], n: int) -> Counter[str]:
    if len(tokens) < n:
        return Counter()
    return Counter(' '.join(tokens[index:index + n]) for index in range(len(tokens) - n + 1))

def _openstack_token_df(cases: list[ActionCase]) -> Counter[str]:
    df: Counter[str] = Counter()
    for case in cases:
        for token in case.raw_features['token']:
            df[token] += 1
    return df

def _openstack_text_df(cases: list[ActionCase]) -> Counter[str]:
    event_bi_df: Counter[str] = Counter()
    for case in cases:
        bigrams = _ngram_counter(_text_tokens(case.incident_text), 2)
        event_bi_df.update(gram for gram in bigrams if any(term in _SENSITIVE_TEXT_TERMS for term in gram.split()))
    return event_bi_df

def _feature_counter(query: ActionQuery | ActionCase, *, dataset: str, feature_name: str, token_df: Counter[str] | None, event_bi_df: Counter[str] | None) -> Counter[str]:
    raw_features = query.raw_features
    incident_text = query.incident_text
    if dataset != 'OpenStack':
        return Counter(raw_features[feature_name])
    if feature_name == 'token':
        return Counter(raw_features['token'])
    if feature_name.startswith('hybrid_event_bi_w'):
        if token_df is None or event_bi_df is None:
            raise ValueError('OpenStack lookup tables are required')
        weight = int(feature_name.rsplit('w', 1)[1])
        payload = Counter({token: count for token, count in raw_features['token'].items() if token_df.get(token, 0) <= 20})
        for gram, count in _ngram_counter(_text_tokens(incident_text), 2).items():
            if event_bi_df.get(gram, 0) <= 20 and any(term in _SENSITIVE_TEXT_TERMS for term in gram.split()):
                payload[gram] += count * weight
        return payload
    raise ValueError(feature_name)

def _score_value(raw_score: float, *, rank: int, score_mode: str) -> float:
    if score_mode == 'raw':
        return max(float(raw_score), 0.0)
    if score_mode == 'rank':
        return 1.0 / float(rank)
    raise ValueError(score_mode)

def _vote_label(items: list[tuple[float, ActionCase]], *, attr_name: str, score_mode: str, top1: bool, fallback_label: str | None) -> str | None:
    if not items:
        return fallback_label
    if top1:
        return getattr(items[0][1], attr_name)
    counter: Counter[str] = Counter()
    for rank, (score, case) in enumerate(items, start=1):
        counter[getattr(case, attr_name)] += _score_value(score, rank=rank, score_mode=score_mode)
    if not any(value > 0 for value in counter.values()):
        for _, case in items:
            counter[getattr(case, attr_name)] += 1.0
    ranked = sorted(counter.items(), key=lambda pair: (pair[1], pair[0]), reverse=True)
    return ranked[0][0] if ranked else fallback_label

def _top_margin(items: list[tuple[float, ActionCase]]) -> float:
    if len(items) < 2:
        return 1.0
    return max(float(items[0][0]) - float(items[1][0]), 0.0)

def _majority_labels(cases: list[ActionCase]) -> dict[str, object]:
    root_counts = Counter(case.root_label for case in cases)
    action_counts = Counter(case.action_label for case in cases)
    action_by_root: dict[str, Counter[str]] = defaultdict(Counter)
    for case in cases:
        action_by_root[case.root_label][case.action_label] += 1
    return {
        'root': root_counts.most_common(1)[0][0],
        'action': action_counts.most_common(1)[0][0],
        'action_by_root': {root: counts.most_common(1)[0][0] for root, counts in action_by_root.items()},
    }

def _predict(root_items: list[tuple[float, ActionCase]], action_items: list[tuple[float, ActionCase]], config: FormalRagConfig, priors: dict[str, object]) -> dict[str, str | None]:
    use_root_fallback = config.root_floor is not None and (not root_items or float(root_items[0][0]) < float(config.root_floor))
    if not use_root_fallback and config.root_margin is not None:
        use_root_fallback = _top_margin(root_items) < float(config.root_margin)
    if use_root_fallback:
        root_label = priors['root']
    else:
        root_label = _vote_label(root_items, attr_name='root_label', score_mode=config.score_mode, top1=False, fallback_label=priors['root'])
    if config.action_strategy.startswith('filtered'):
        candidate_items = [item for item in action_items if item[1].root_label == root_label] or action_items
    else:
        candidate_items = action_items
    action_fallback = priors['action_by_root'].get(root_label, priors['action'])
    use_action_fallback = config.action_floor is not None and (not candidate_items or float(candidate_items[0][0]) < float(config.action_floor))
    if not use_action_fallback and config.action_margin is not None:
        use_action_fallback = _top_margin(candidate_items) < float(config.action_margin)
    if use_action_fallback:
        action_label = action_fallback
    else:
        action_label = _vote_label(candidate_items, attr_name='action_label', score_mode=config.score_mode, top1=config.action_strategy.endswith('top1'), fallback_label=action_fallback)
    return {'root_label': root_label, 'action_label': action_label}

class RAGFormalRunner:
    method_name = 'rag'

    def __init__(self, bundle: BenchmarkBundle) -> None:
        self._bundle = bundle
        self._cases_by_dataset = _dataset_cases(bundle)
        self._priors = {dataset: _majority_labels(cases) for dataset, cases in self._cases_by_dataset.items()}
        self._openstack_token_df = _openstack_token_df(self._cases_by_dataset['OpenStack'])
        self._openstack_event_bi_df = _openstack_text_df(self._cases_by_dataset['OpenStack'])

    def _scored(self, query_feature: Counter[str], support_cases: list[ActionCase], *, dataset: str, feature_name: str) -> list[tuple[float, ActionCase]]:
        scored: list[tuple[float, ActionCase]] = []
        for support_case in support_cases:
            support_feature = _feature_counter(support_case, dataset=dataset, feature_name=feature_name, token_df=self._openstack_token_df if dataset == 'OpenStack' else None, event_bi_df=self._openstack_event_bi_df if dataset == 'OpenStack' else None)
            scored.append((cosine_similarity(query_feature, support_feature), support_case))
        scored.sort(key=lambda item: (item[0], item[1].case_id), reverse=True)
        return scored

    def predict(self, query: ActionQuery, *, support_cases: list[ActionCase]) -> dict:
        dataset = query.base_case.dataset
        config = FORMAL_RAG[dataset]
        root_feature = _feature_counter(query, dataset=dataset, feature_name=config.root_feature, token_df=self._openstack_token_df if dataset == 'OpenStack' else None, event_bi_df=self._openstack_event_bi_df if dataset == 'OpenStack' else None)
        action_feature = _feature_counter(query, dataset=dataset, feature_name=config.action_feature, token_df=self._openstack_token_df if dataset == 'OpenStack' else None, event_bi_df=self._openstack_event_bi_df if dataset == 'OpenStack' else None)
        root_items = self._scored(root_feature, support_cases, dataset=dataset, feature_name=config.root_feature)[: config.k]
        action_items = self._scored(action_feature, support_cases, dataset=dataset, feature_name=config.action_feature)[: config.k]
        prediction = _predict(root_items, action_items, config, self._priors[dataset])
        return {
            'prediction': prediction,
            'metadata': {
                'root_feature': config.root_feature,
                'action_feature': config.action_feature,
                'k': config.k,
                'retrieved_case_ids_root': [case.case_id for _, case in root_items],
                'retrieved_case_ids_action': [case.case_id for _, case in action_items],
            },
        }
