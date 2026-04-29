"""Evidence rendering helpers."""

from __future__ import annotations

import re
from collections import Counter

from experiments.project.action.core.noise_v1 import inject_text_noise
from experiments.project.action.core.schema import ActionCase, ActionQuery
from experiments.project.action.methods.retrieval import RetrievedCase

_TOKEN_RE = re.compile(r"[A-Za-z0-9_<>:-]+")
_GRAPH_PACKS = {
    'HDFS': {'all': ['suffix12', 'terminal_bigram', 'tail_position']},
    'Hadoop': {'phase_terminal': ['phase_terminal']},
    'OpenStack': {'severity': ['anchor_position_component_api_severity']},
}

def _line_after(text: str, prefix: str) -> str:
    for line in (text or '').splitlines():
        if line.startswith(prefix):
            return line[len(prefix) :].strip()
    return ''

def _int_after(text: str, prefix: str, *, default: int = 0) -> int:
    value = _line_after(text, prefix)
    match = re.search(r'\d+', value)
    if match:
        return int(match.group(0))
    return default

def _sequence_from_line(line: str) -> list[str]:
    if not line:
        return []
    return [part.strip() for part in line.split('->') if part.strip()]

def _alert_lines(query: ActionQuery) -> list[str]:
    lines = query.incident_text.splitlines()
    if 'Alert timeline:' not in lines:
        return [line[2:].strip() for line in lines if line.startswith('- ')]
    start = lines.index('Alert timeline:') + 1
    return [line[2:].strip() for line in lines[start:] if line.startswith('- ')]

def _top_terms(counter: Counter[str], *, n: int) -> str:
    if not counter:
        return 'none'
    return ', '.join(f'{token} x{count}' for token, count in counter.most_common(n))

def _pipe_suffix(text: str) -> str:
    parts = [part.strip() for part in (text or '').split('|')]
    if len(parts) <= 2:
        return text.strip()
    return ' | '.join(parts[2:]).strip()

def _openstack_body_lines(query: ActionQuery) -> list[str]:
    lines = query.incident_text.splitlines()
    if 'Bounded create-episode lines:' not in lines:
        return []
    start = lines.index('Bounded create-episode lines:') + 1
    payload = []
    for line in lines[start:]:
        if not line.strip():
            continue
        body = line.rsplit('] ', 1)[-1] if '] ' in line else line
        body = re.sub(r'\b[0-9a-f]{8}-[0-9a-f-]{27,36}\b', '<id>', body, flags=re.IGNORECASE)
        body = re.sub(r'\b\d+\.\d+\.\d+\.\d+\b', '<ip>', body)
        body = re.sub(r'\b\d+\b', '<num>', body)
        payload.append(body.strip())
    return payload

def _hdfs_rule_summary(events: list[str]) -> list[str]:
    counts = Counter(events)
    e4e3 = counts.get('E4', 0) + counts.get('E3', 0)
    e11e9 = counts.get('E11', 0) + counts.get('E9', 0)
    e20e26 = counts.get('E20', 0) + counts.get('E26', 0)
    tail = events[-10:] if len(events) >= 10 else events
    tail_text = ' -> '.join(tail)
    e23_block = 'E23 -> E23 -> E23' in tail_text
    e21_block = 'E21 -> E21 -> E21' in tail_text
    e23_before_e21 = e23_block and e21_block and tail_text.index('E23 -> E23 -> E23') < tail_text.index('E21 -> E21 -> E21')
    type0_gate = counts.get('E20', 0) == 0 and counts.get('E6', 0) > 0 and counts.get('E16', 0) > 0
    type5_gate = counts.get('E20', 0) > 0 and counts.get('E4', 0) == 0
    return [
        f"E27 marker explicit: {'yes' if counts.get('E27', 0) > 0 else 'no'}",
        f"E28 marker explicit: {'yes' if counts.get('E28', 0) > 0 else 'no'}",
        f"E20 explicit: {'yes' if counts.get('E20', 0) > 0 else 'no'}",
        f"E6+E16 pair active: {'yes' if counts.get('E6', 0) > 0 and counts.get('E16', 0) > 0 else 'no'}",
        f"Type0 gate active (E20 absent + E6/E16 present): {'yes' if type0_gate else 'no'}",
        f"Type5 gate active (E20 present + E4 absent): {'yes' if type5_gate else 'no'}",
        f"E20 with E4 support: {'yes' if counts.get('E20', 0) > 0 and counts.get('E4', 0) > 0 else 'no'}",
        f"E20 with E4/E3 support: {'yes' if counts.get('E20', 0) > 0 and e4e3 > 0 else 'no'}",
        f"E26/E20 dominates over E4/E3: {'yes' if e20e26 > e4e3 else 'no'}",
        f'E11/E9 echo pressure: {e11e9}',
        f"Repeated E23 block before final E21 closure: {'yes' if e23_before_e21 else 'no'}",
        f"Repeated E21 terminal block: {'yes' if e21_block else 'no'}",
    ]

def render_incident_surface(query: ActionQuery, *, style: str) -> str:
    if style == 'hdfs_trace_digest':
        prefix = _sequence_from_line(_line_after(query.incident_text, 'Prefix events:'))
        tail = _sequence_from_line(_line_after(query.incident_text, 'Tail events:'))
        tail_counter = query.raw_features.get('suffix12', Counter())
        event_count = _int_after(query.incident_text, 'Trace length:', default=query.base_case.metadata.get('event_count', 0))
        return '\n'.join([
            'HDFS incident digest',
            f'Trace length: {event_count}',
            f"Prefix chain: {' -> '.join(prefix[:8])}",
            f"Tail chain: {' -> '.join(tail[-8:])}",
            f'Tail-event mix: {_top_terms(tail_counter, n=6)}',
        ])
    if style == 'hdfs_family_rule_digest':
        full_events = _sequence_from_line(_line_after(query.incident_text, 'Full event trace:'))
        prefix = _sequence_from_line(_line_after(query.incident_text, 'Prefix events:'))
        tail = _sequence_from_line(_line_after(query.incident_text, 'Tail events:'))
        tail_counter = query.raw_features.get('suffix12', Counter())
        tail_transitions = [f'{left}->{right}' for left, right in zip(tail[:-1], tail[1:])]
        event_count = _int_after(query.incident_text, 'Trace length:', default=query.base_case.metadata.get('event_count', 0))
        window = ' -> '.join(full_events[max(0, len(full_events) - 12):])
        return '\n'.join([
            'HDFS family rule digest',
            f'Trace length: {event_count}',
            *_hdfs_rule_summary(full_events),
            f'Family cue window: {window}',
            f"Tail chain: {' -> '.join(tail[-10:])}",
            f'Tail transitions: {_top_terms(Counter(tail_transitions), n=6)}',
            f'Tail-event mix: {_top_terms(tail_counter, n=6)}',
            f'Trace-prefix mix: {_top_terms(query.raw_features.get("prefix12", Counter()), n=5)}',
        ])
    if style == 'hadoop_root_boundary_excerpt':
        workload = _line_after(query.incident_text, 'Workload context:') or query.base_case.metadata.get('workload', 'Unknown')
        alerts = _alert_lines(query)
        late_rare = query.raw_features.get('late_rare', Counter())
        tail_alerts = alerts[-4:] if len(alerts) >= 4 else alerts
        alert_count = _int_after(query.incident_text, 'Alert count:', default=query.base_case.metadata.get('alert_count', 0))

        def cue_block(patterns: list[str]) -> list[str]:
            lowered = [alert.lower() for alert in alerts]
            selected = [alerts[index] for index, alert in enumerate(lowered) if any(pattern in alert for pattern in patterns)]
            return selected[:6]

        machine = cue_block(['deadnodes', 'createblockoutputstream', 'firstbadlink', 'remote block reader', 'bad connect ack'])
        port_return = cue_block(['returned by containermanager'])
        force_close = cue_block(['forcibly closed', 'connection reset', 'connection refused'])
        timeout = cue_block(['timed out', 'timeout'])
        disk_hard = cue_block(['not enough space on the disk', 'there is not enough space on the disk', 'spill failed'])
        delete_path = cue_block(['could not delete hdfs_path'])
        pressure = cue_block(['shuffling to disk', 'maxsingleshufflelimit', 'on disk map outputs', 'finalmerge called', 'merging ', 'mergermanager memorylimit'])
        return '\n'.join([
            'Hadoop boundary cue excerpt',
            f'Workload context: {workload}',
            f'Alert count: {alert_count}',
            f'Late rare tokens: {_top_terms(late_rare, n=8)}',
            f'Machine hard cue count: {len(machine)}',
            f'Port-return cue count: {len(port_return)}',
            f'Force-close cue count: {len(force_close)}',
            f'Timeout cue count: {len(timeout)}',
            f'Hard disk cue count: {len(disk_hard)}',
            f'Delete-path cue count: {len(delete_path)}',
            f'Shuffle-pressure cue count: {len(pressure)}',
            'Machine hard cues:',
            *([f'- {alert}' for alert in machine[:4]] or ['- none']),
            'Port-return cues:',
            *([f'- {alert}' for alert in port_return[:4]] or ['- none']),
            'Network hard cues:',
            *([f'- {alert}' for alert in (force_close + timeout)[:4]] or ['- none']),
            'Hard disk-full cues:',
            *([f'- {alert}' for alert in disk_hard[:4]] or ['- none']),
            'Delete-path cues:',
            *([f'- {alert}' for alert in delete_path[:4]] or ['- none']),
            'Shuffle-pressure cues:',
            *([f'- {alert}' for alert in pressure[:4]] or ['- none']),
            'Tail alerts:',
            *[f'- {alert}' for alert in tail_alerts],
        ])
    if style == 'openstack_anchor_sequence_excerpt':
        anchor = query.raw_features.get('anchor', Counter())
        token = query.raw_features.get('token', Counter())
        detail = token.get('request', 0) + token.get('detail', 0) + token.get('servers', 0) + token.get('server', 0)
        body_lines = _openstack_body_lines(query)
        anchor_profile = [token_name for token_name, _ in anchor.most_common(8)]
        bounded = body_lines[:8] if len(body_lines) <= 12 else body_lines[:4] + body_lines[-4:]
        return '\n'.join([
            'OpenStack sequence excerpt',
            f'API detail score: {detail}',
            f"Anchor profile: {' -> '.join(anchor_profile)}",
            'Episode bodies:',
            *bounded,
        ])
    if style == 'openstack_body_anchor_digest':
        anchor = query.raw_features.get('anchor', Counter())
        token = query.raw_features.get('token', Counter())
        body_lines = _openstack_body_lines(query)
        anchor_profile = [token_name for token_name, _ in anchor.most_common(8)]
        return '\n'.join([
            'OpenStack body anchor digest',
            f"Anchor profile: {' -> '.join(anchor_profile)}",
            f'Anchor signals: {_top_terms(anchor, n=8)}',
            f'Message tokens: {_top_terms(token, n=10)}',
            'Episode bodies:',
            *body_lines[:6],
        ])
    raise ValueError(style)

def _fact_budget(noise_level: float, *, style: str) -> int:
    if style == 'bullets':
        return 999
    if style == 'adaptive_compact':
        return {0.0: 8, 0.2: 7, 0.4: 6, 0.6: 5, 0.8: 4, 1.0: 3}[noise_level]
    raise ValueError(style)

def render_graph_evidence(case: ActionCase, *, graph_pack: str, graph_style: str, noise_level: float) -> str:
    keys = _GRAPH_PACKS[case.dataset][graph_pack]
    facts: list[str] = []
    pre_noised = str(case.metadata.get('noise_variant') or '') == 'v2'
    for key in keys:
        for fact in case.graph_facts.get(key, []):
            noised_fact = fact if pre_noised else inject_text_noise(fact, dataset=case.dataset, noise_level=noise_level, seed=len(fact))
            facts.append(f'{key} | {noised_fact}')
    budget = _fact_budget(noise_level, style=graph_style)
    return '\n'.join(facts[:budget])

def render_retrieved_case(item: RetrievedCase, *, style: str) -> str:
    stripped = _pipe_suffix(item.case.support_summary)
    if style == 'incident_head':
        lines = item.case.incident_text.splitlines()[:6]
        return f'[Retrieved] score={item.score:.4f} title={item.case.title}\n' + '\n'.join(lines)
    if style == 'openstack_action_hint':
        action_label = item.case.action_label.removeprefix('os_episode_')
        for suffix in ('_imagecache', '_server', '_compute_manager'):
            if action_label.endswith(suffix):
                action_label = action_label[: -len(suffix)]
                break
        parts = [part.strip() for part in stripped.split('|')]
        detail_part = next((part for part in parts if part.startswith('detail=')), '')
        return f'[Retrieved] score={item.score:.4f} action_hint={action_label} | {detail_part}'.strip()
    if style == 'hadoop_workload_hint':
        workload_hint = item.case.action_label.rsplit('__repair__', 1)[-1]
        root_hint = item.case.root_label.removeprefix('hadoop_root_')
        return f'[Retrieved] score={item.score:.4f} root_hint={root_hint} | workload_hint={workload_hint}'
    raise ValueError(style)
