"""
Build E2E scaled benchmark with real sequential time-windows **and auto-annotated GT**.

- HDFS: 50 anomalous Block_ID groups; window = logs for that block ending at first ERROR/WARN.
  - We auto-assign `ground_truth_template` using Drain on the target alert line.
  - We auto-assign `ground_truth_root_cause_template` using keyword → canonical-pattern rules.
- OpenStack: 40–50 windows (2–3 minutes of logs leading up to each ERROR/WARN).
  - We auto-assign GT template/root cause via known ERROR patterns and SOP-style templates.
- Hadoop: ~44 abnormal application folders as natural sequential windows (full app log).
  - We map abnormal_label.txt paragraphs to macro-label GTs (Machine down / Network disconnection / Disk full).

Output: a new benchmark file `e2e_scaled_benchmark_v2.json` (the original v1 文件保留，不会覆盖旧数据)。
"""
import os
import re
import sys
import json
import csv
from typing import Dict, List, Tuple

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from experiments.rq123_e2e import hadoop_loader
from src.perception.drain_parser import DrainParser
from src.system.edge_node import NuSyEdgeNode

DATA_PROCESSED = os.path.join(_PROJECT_ROOT, "data", "processed")
RAW_ROOT = os.path.join(_PROJECT_ROOT, "data", "raw")
CAUSAL_KB_DYNOTEARS = os.path.join(DATA_PROCESSED, "gt_causal_knowledge_dynotears.json")

# HDFS paths
HDFS_V1 = os.path.join(RAW_ROOT, "HDFS_v1")
HDFS_LOG = os.path.join(HDFS_V1, "HDFS.log")
HDFS_ANOMALY_LABEL = os.path.join(HDFS_V1, "preprocessed", "anomaly_label.csv")
# OpenStack
OPENSTACK_2 = os.path.join(RAW_ROOT, "OpenStack_2")
OPENSTACK_ABNORMAL_LOG = os.path.join(OPENSTACK_2, "openstack_abnormal.log")
# Hadoop
HADOOP_ROOT = os.path.join(RAW_ROOT, "Hadoop")
HADOOP_LABEL_PATH = os.path.join(HADOOP_ROOT, "abnormal_label.txt")

# Targets
TARGET_HDFS = 50
TARGET_OPENSTACK = 50  # 40-50
TARGET_HADOOP = 44
BLK_RE = re.compile(r"blk_[-\d]+")

_DRAIN = DrainParser()


def _get_anomalous_block_ids(anomaly_label_path: str, limit: int) -> List[str]:
    """From anomaly_label.csv get BlockIds with Label=Anomaly, up to limit."""
    if not os.path.isfile(anomaly_label_path):
        return []
    ids: List[str] = []
    with open(anomaly_label_path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            if row.get("Label", "").strip().lower() != "anomaly":
                continue
            bid = (row.get("BlockId", "") or "").strip()
            if bid and BLK_RE.match(bid):
                ids.append(bid)
                if len(ids) >= limit:
                    break
    return ids


def _scan_hdfs_log_by_block(
    log_path: str,
    anomalous_ids: List[str],
) -> Dict[str, List[Tuple[int, str]]]:
    """
    Scan HDFS.log; for each line that contains a blk_ id, append (line_idx, line) to that block's list.
    Returns dict: block_id -> [(idx, line), ...] in file order.
    """
    block_lines: Dict[str, List[Tuple[int, str]]] = {bid: [] for bid in anomalous_ids}
    if not os.path.isfile(log_path):
        return block_lines
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for idx, raw in enumerate(f):
            line = raw.rstrip("\n")
            if not line.strip():
                continue
            for bid in anomalous_ids:
                if bid in line:
                    block_lines[bid].append((idx, line))
                    break
    return block_lines


def _extract_target_line_and_template(
    window_lines: List[str],
    dataset: str,
) -> Tuple[str, str]:
    """
    从一个时间窗口中抽取“目标告警行”以及其 Drain 模板：
      - 目标行优先选择最后一条包含 ERROR/WARN 的日志；
      - 若不存在，则退化为窗口中的最后一条非空行。
    """
    target_line = ""
    for line in reversed(window_lines):
        if not line.strip():
            continue
        if "ERROR" in line or "WARN" in line:
            target_line = line
            break
    if not target_line:
        for line in reversed(window_lines):
            if line.strip():
                target_line = line
                break
    if not target_line:
        return "", ""

    # 复用 NuSy-Edge 的 header 预处理逻辑做内容截断
    ds = "HDFS" if dataset.upper() == "HADOOP" else dataset
    clean = NuSyEdgeNode.preprocess_header(target_line, ds)
    if not clean:
        clean = target_line
    try:
        tpl = _DRAIN.parse(clean)
    except Exception:
        tpl = ""
    return target_line, tpl or ""


def _auto_gt_for_hdfs(window_lines: List[str]) -> Tuple[str, str]:
    """
    针对 HDFS 时间窗，通过关键字 + Drain 模板自动生成 GT：
      - gt_template: Drain 模板
      - gt_root: 使用关键字映射到与 SOP KB 对齐的 canonical 模板片段
    """
    target_line, tpl = _extract_target_line_and_template(window_lines, "HDFS")
    if not target_line:
        return "", "unknown"

    text = target_line
    # 这些模式与 repair_sop_kb.json 中 HDFS 部分保持语义一致
    mapping = [
        ("Got exception while serving", "[*]Got exception while serving[*]to[*]"),
        ("Exception in receiveBlock", "[*]Exception in receiveBlock for[*]"),
        ("PacketResponder", "[*]PacketResponder[*]for block[*]terminating[*]"),
        ("allocateBlock", "[*]BLOCK* NameSystem[*]allocateBlock:[*]"),
        ("addStoredBlock: blockMap updated", "[*]BLOCK* NameSystem[*]addStoredBlock: blockMap updated:[*]is added to[*]size[*]"),
        ("Receiving block", "[*]Receiving block[*]src:[*]dest:[*]"),
        ("Transmitted block", "[*]:Transmitted block[*]to[*]"),
        ("Verification succeeded for", "[*]Verification succeeded for[*]"),
        ("Deleting block", "[*]Deleting block[*]file[*]"),
    ]
    gt_root = "unknown"
    lower = text.lower()
    for key, canon in mapping:
        if key.lower() in lower:
            gt_root = canon
            break
    return tpl, gt_root


def _auto_gt_for_openstack(window_lines: List[str]) -> Tuple[str, str]:
    """
    针对 OpenStack 时间窗，根据典型 WARNING/ERROR 内容自动生成 GT：
      - 使用 Drain 模板作为 gt_template；
      - 利用关键字映射到 repair_sop_kb.json 中的 canonical 模板。
    """
    target_line, tpl = _extract_target_line_and_template(window_lines, "OpenStack")
    if not target_line:
        return "", "unknown"

    text = target_line
    mapping = [
        (
            "The instance sync for host",
            "The instance sync for host '<*>' did not match. Re-created its InstanceList.",
        ),
        (
            "During sync_power_state the instance has a pending task",
            "During sync_power_state the instance has a pending task (spawning). Skip.",
        ),
        (
            "While synchronizing instance power states",
            "During sync_power_state the instance has a pending task (spawning). Skip.",
        ),
        ("Terminating instance", "Terminating instance"),
        ("nova.metadata.wsgi.server", "nova.metadata.wsgi.server"),
        ("Unknown base file", "Unknown base file: <*>"),
        (
            "couldn't obtain the vcpu count from domain id",
            "couldn't obtain the vcpu count from domain id: <*>, exception: Requested operation is not valid: cpu affinity is not supported",
        ),
    ]
    gt_root = "unknown"
    lower = text.lower()
    for key, canon in mapping:
        if key.lower() in lower:
            gt_root = canon
            break
    return tpl, gt_root


def _window_for_block(
    lines_with_idx: List[Tuple[int, str]],
    end_at_error_warn: bool = True,
) -> List[str]:
    """
    Return list of log lines forming the window.
    If end_at_error_warn: from start up to and including first line containing ERROR or WARN; else all.
    """
    if not lines_with_idx:
        return []
    out: List[str] = []
    for _, line in lines_with_idx:
        out.append(line)
        if end_at_error_warn and ("ERROR" in line or "WARN" in line):
            break
    return out


def _build_hdfs_cases(limit: int = TARGET_HDFS) -> List[Dict[str, object]]:
    """Extract exactly `limit` anomalous Block_ID groups; window ends at target alert (ERROR/WARN)."""
    cases: List[Dict[str, object]] = []
    anomalous_ids = _get_anomalous_block_ids(HDFS_ANOMALY_LABEL, limit)
    if not anomalous_ids:
        return cases
    block_lines = _scan_hdfs_log_by_block(HDFS_LOG, anomalous_ids)
    for bid in anomalous_ids:
        lines_with_idx = block_lines.get(bid, [])
        window = _window_for_block(lines_with_idx, end_at_error_warn=True)
        if not window:
            continue
        gt_tpl, gt_root = _auto_gt_for_hdfs(window)
        raw_log = "\n".join(window)
        cases.append({
            "case_id": f"hdfs_blk_{bid}",
            "dataset": "HDFS",
            "raw_log": raw_log,
            "ground_truth_template": gt_tpl,
            "ground_truth_root_cause_template": gt_root,
            "source": "hdfs_time_window",
            "block_id": bid,
        })
        if len(cases) >= limit:
            break
    return cases


def _build_openstack_cases(
    limit: int = TARGET_OPENSTACK,
    lines_before: int = 180,
) -> List[Dict[str, object]]:
    """
    Find ERROR/WARN lines in openstack_abnormal.log; for each, take `lines_before` lines
    leading up to it as context (2-3 min of logs). Form one window per such anchor; cap at limit.
    """
    cases: List[Dict[str, object]] = []
    if not os.path.isfile(OPENSTACK_ABNORMAL_LOG):
        return cases
    with open(OPENSTACK_ABNORMAL_LOG, "r", encoding="utf-8", errors="ignore") as f:
        all_lines = [x.rstrip("\n") for x in f.readlines()]
    # Find indices where line contains ERROR or WARN
    anchor_indices: List[int] = []
    for i, line in enumerate(all_lines):
        if not line.strip():
            continue
        if "ERROR" in line or "WARN" in line:
            anchor_indices.append(i)
    # Build windows: [max(0, i - lines_before), i+1]
    seen_windows: set = set()
    for i in anchor_indices:
        start = max(0, i - lines_before)
        window_lines = all_lines[start : i + 1]
        key = (start, i)  # dedupe by (start, end)
        if key in seen_windows:
            continue
        seen_windows.add(key)
        gt_tpl, gt_root = _auto_gt_for_openstack(window_lines)
        raw_log = "\n".join(window_lines)
        cases.append({
            "case_id": f"openstack_win_{len(cases)}",
            "dataset": "OpenStack",
            "raw_log": raw_log,
            "ground_truth_template": gt_tpl,
            "ground_truth_root_cause_template": gt_root,
            "source": "openstack_time_window",
        })
        if len(cases) >= limit:
            break
    return cases


def _build_hadoop_cases(limit: int = TARGET_HADOOP) -> List[Dict[str, object]]:
    """Use ~limit abnormal application folders as natural sequential windows (full app log)."""
    if not (os.path.isdir(HADOOP_ROOT) and os.path.exists(HADOOP_LABEL_PATH)):
        return []
    rows = hadoop_loader.parse_abnormal_label_file(HADOOP_LABEL_PATH)
    cases: List[Dict[str, object]] = []
    for row in rows:
        if len(cases) >= limit:
            break
        label = (row.get("label") or "").lower()
        if label == "normal":
            continue
        app_id = row["application_id"]
        try:
            lines = hadoop_loader.iter_hadoop_application_logs(HADOOP_ROOT, app_id)
        except FileNotFoundError:
            continue
        if not lines:
            continue
        reason = (row.get("reason") or "").strip()
        if "machine down" in reason.lower():
            gt_root = "Machine down"
        elif "network disconnection" in reason.lower():
            gt_root = "Network disconnection"
        elif "disk full" in reason.lower():
            gt_root = "Disk full"
        else:
            gt_root = "unknown"
        raw_log = "\n".join(lines)
        cases.append({
            "case_id": f"hadoop_{app_id}",
            "dataset": "Hadoop",
            "raw_log": raw_log,
            "ground_truth_template": "",
            "ground_truth_root_cause_template": gt_root,
            "source": "hadoop_abnormal",
            "reason": reason,
        })
    return cases


def _norm_text(text: str) -> str:
    return " ".join(str(text or "").strip().lower().split())


def _load_causal_kb(path: str) -> List[Dict[str, object]]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _patch_hadoop_cases_with_template_gt(
    cases: List[Dict[str, object]],
    causal_kb_path: str = CAUSAL_KB_DYNOTEARS,
) -> List[Dict[str, object]]:
    """
    Rebuild Hadoop benchmark GT into the two-level form used by our later eval:
    - `ground_truth_template`: target/effect template extracted from the abnormal window
    - `ground_truth_root_cause_template`: strongest incoming source template from DYNOTEARS
    - `gt_action_label`: original coarse fault label from abnormal_label.txt
    """
    kb = _load_causal_kb(causal_kb_path)
    hadoop_edges = [e for e in kb if str(e.get("domain", "")).lower() == "hadoop"]
    by_target: Dict[str, List[Dict[str, object]]] = {}
    for edge in hadoop_edges:
        tgt = _norm_text(str(edge.get("target_template", "") or ""))
        if tgt:
            by_target.setdefault(tgt, []).append(edge)

    def best_graph_aligned_templates(lines: List[str]) -> Tuple[str, str]:
        best_tpl = ""
        best_root = ""
        best_weight = -1.0
        best_idx = -1
        for idx, line in enumerate(lines):
            if not line.strip():
                continue
            clean = NuSyEdgeNode.preprocess_header(line, "HDFS") or line
            try:
                tpl = _DRAIN.parse(clean)
            except Exception:
                tpl = ""
            key = _norm_text(tpl)
            if not key or key not in by_target:
                continue
            for edge in by_target[key]:
                src = str(edge.get("source_template", "") or "")
                if not src:
                    continue
                weight = abs(float(edge.get("weight", 0.0) or 0.0))
                if weight > best_weight or (weight == best_weight and idx > best_idx):
                    best_tpl = tpl
                    best_root = src
                    best_weight = weight
                    best_idx = idx
        return best_tpl, best_root

    patched: List[Dict[str, object]] = []
    for case in cases:
        if str(case.get("dataset", "")) != "Hadoop":
            patched.append(case)
            continue

        raw = str(case.get("raw_log", "") or "")
        lines = [x for x in raw.split("\n") if x.strip()]
        gt_tpl, best_root = best_graph_aligned_templates(lines)
        if not gt_tpl:
            _, gt_tpl = _extract_target_line_and_template(lines, "Hadoop")
        if gt_tpl:
            case["ground_truth_template"] = gt_tpl

        reason = str(case.get("reason", "") or "").strip()
        if reason:
            case["gt_action_label"] = reason

        if not best_root and gt_tpl:
            target_key = _norm_text(gt_tpl)
            best_source = ""
            best_weight = -1.0
            for edge in by_target.get(target_key, []):
                src = str(edge.get("source_template", "") or "")
                if not src:
                    continue
                weight = abs(float(edge.get("weight", 0.0) or 0.0))
                if weight > best_weight:
                    best_weight = weight
                    best_source = src
            best_root = best_source
        if best_root:
            case["ground_truth_root_cause_template"] = best_root
        patched.append(case)
    return patched


def _fallback_hdfs_from_rq3(limit: int) -> List[Dict[str, object]]:
    """When raw HDFS_v1 is missing or has no anomaly labels, take HDFS cases from rq3_test_set."""
    path = os.path.join(DATA_PROCESSED, "rq3_test_set.json")
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        all_cases = json.load(f)
    hdfs = [c for c in all_cases if str(c.get("dataset", "")).upper() == "HDFS"]
    out = []
    for c in hdfs[:limit]:
        d = dict(c)
        d.setdefault("source", "rq3_seed")
        out.append(d)
    return out


def build_e2e_scaled_benchmark(
    target_path: str = None,
    target_hdfs: int = TARGET_HDFS,
    target_openstack: int = TARGET_OPENSTACK,
    target_hadoop: int = TARGET_HADOOP,
) -> List[Dict[str, object]]:
    """
    Build E2E benchmark:
    - HDFS: 50 anomalous Block_ID groups (window ends at ERROR/WARN); fallback to rq3 HDFS if no raw.
    - OpenStack: 40-50 windows (2-3 min leading to ERROR/WARN).
    - Hadoop: ~44 abnormal app windows.
    """
    all_cases: List[Dict[str, object]] = []

    # HDFS
    hdfs_cases = _build_hdfs_cases(limit=target_hdfs)
    if len(hdfs_cases) < target_hdfs:
        fallback = _fallback_hdfs_from_rq3(target_hdfs - len(hdfs_cases))
        hdfs_cases.extend(fallback[: target_hdfs - len(hdfs_cases)])
    all_cases.extend(hdfs_cases)

    # OpenStack
    openstack_cases = _build_openstack_cases(limit=target_openstack)
    all_cases.extend(openstack_cases)

    # Hadoop
    hadoop_cases = _build_hadoop_cases(limit=target_hadoop)
    hadoop_cases = _patch_hadoop_cases_with_template_gt(hadoop_cases)
    all_cases.extend(hadoop_cases)

    if not target_path:
        # v2: 带自动 GT 标注的新基准文件，保留旧版 e2e_scaled_benchmark.json 不动
        target_path = os.path.join(DATA_PROCESSED, "e2e_scaled_benchmark_v2.json")
    os.makedirs(os.path.dirname(target_path) or ".", exist_ok=True)
    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(all_cases, f, indent=2, ensure_ascii=False)

    n_hdfs = len(hdfs_cases)
    n_os = len(openstack_cases)
    n_hadoop = len(hadoop_cases)
    print(f"[INFO] Built E2E benchmark: HDFS={n_hdfs}, OpenStack={n_os}, Hadoop={n_hadoop}, total={len(all_cases)}")
    print(f"[INFO] Written to: {target_path}")
    return all_cases


def main():
    build_e2e_scaled_benchmark()


if __name__ == "__main__":
    main()
