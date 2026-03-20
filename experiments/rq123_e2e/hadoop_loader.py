import os
from typing import Dict, List, Tuple


def parse_abnormal_label_file(label_path: str) -> List[Dict[str, str]]:
    """
    解析 Hadoop abnormal_label.txt 为一组 case。

    实际 LogHub Hadoop 文件是一个带有段落的文本，例如：

        The logs are generated from ...

        ### WordCount
        Normal:
        + application_1445087491445_0005
        ...
        Machine down:
        + application_1445087491445_0001
        ...

    我们按以下规则解析：
      - 以 "Normal:" 视为 label="normal"
      - 以 "<故障类型>:"（如 "Machine down:"、"Network disconnection:"、"Disk full:"）视为 label 的人类可读描述，
        并将 label 标准化为小写 + 下划线（例如 "machine_down"）
      - 以 "+ application_..." 开头的行提取 application_id，并绑定到当前 label / reason。

    返回值：
        [
            {
                "application_id": "...",
                "label": "normal" | "machine_down" | "network_disconnection" | "disk_full" | "unknown",
                "reason": 原始段落标题（例如 "Machine down"）
            },
            ...
        ]
    """
    cases: List[Dict[str, str]] = []
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"abnormal_label file not found: {label_path}")

    current_label = "unknown"
    current_reason = ""

    with open(label_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            # 跳过说明性文字与 Markdown 标题
            if line.startswith("The logs are generated"):
                continue
            if line.startswith("###"):
                # 例如 "### WordCount" / "### PageRank"
                continue

            # 段落标签，如 "Normal:" / "Machine down:" / "Network disconnection:"
            if line.endswith(":") and not line.startswith("+"):
                header = line.rstrip(":").strip()
                header_lower = header.lower()
                if header_lower == "normal":
                    current_label = "normal"
                    current_reason = "Normal"
                else:
                    # 标准化为 machine_down / network_disconnection / disk_full 等
                    current_label = header_lower.replace(" ", "_")
                    current_reason = header
                continue

            # 实际的 application 行，以 "+ application_xxx" 开头
            if line.startswith("+"):
                app = line.lstrip("+").strip()
                if app.startswith("application_"):
                    cases.append(
                        {
                            "application_id": app,
                            "label": current_label or "unknown",
                            "reason": current_reason,
                        }
                    )
                continue

            # 其他行（例如空白或意外格式）直接忽略
            continue

    return cases


def iter_hadoop_application_logs(
    hadoop_root: str,
    application_id: str,
    log_suffixes: Tuple[str, ...] = (".log", ".out"),
) -> List[str]:
    """
    Given a Hadoop dataset root and an application_id, collect all log lines
    under that application's folder as one consecutive "fault window".

    Expected directory layout (generic Hadoop-on-YARN style):
        <hadoop_root>/
            <application_id>/
                stdout.log
                stderr.log
                syslog
                ...

    This function is intentionally conservative: it walks the application
    folder and concatenates all text files whose names end with one of the
    provided suffixes (default: .log / .out), in lexicographical filename
    order and then in-file line order.

    Returns:
        List of raw log lines (strings) for this application_id.
    """
    app_dir = os.path.join(hadoop_root, application_id)
    if not os.path.isdir(app_dir):
        raise FileNotFoundError(f"Hadoop application folder not found: {app_dir}")

    lines: List[str] = []
    entries = sorted(os.listdir(app_dir))
    for name in entries:
        if not name.lower().endswith(log_suffixes):
            continue
        path = os.path.join(app_dir, name)
        if not os.path.isfile(path):
            continue
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for raw in f:
                raw = raw.rstrip("\n")
                if raw.strip():
                    lines.append(raw)
    return lines


def load_hadoop_fault_windows(
    hadoop_root: str,
    abnormal_label_path: str,
    only_abnormal: bool = True,
) -> List[Dict[str, object]]:
    """
    High-level loader that turns Hadoop abnormal_label.txt into E2E "cases".

    Each abnormal application_id is treated as a natural, consecutive fault
    time window, without doing explicit timestamp windowing.

    Returns:
        List of dicts with at least:
            {
                "case_id": application_id,
                "application_id": application_id,
                "label": label,
                "reason": reason,
                "lines": [raw_log_1, raw_log_2, ...]
            }
    """
    label_rows = parse_abnormal_label_file(abnormal_label_path)
    cases: List[Dict[str, object]] = []
    for row in label_rows:
        app_id = row["application_id"]
        label = row.get("label", "").lower()
        if only_abnormal and label != "abnormal":
            continue
        try:
            logs = iter_hadoop_application_logs(hadoop_root, app_id)
        except FileNotFoundError:
            # If a specific application folder is missing, skip but keep loading others.
            continue
        cases.append(
            {
                "case_id": app_id,
                "application_id": app_id,
                "label": label,
                "reason": row.get("reason", ""),
                "lines": logs,
            }
        )
    return cases


__all__ = [
    "parse_abnormal_label_file",
    "iter_hadoop_application_logs",
    "load_hadoop_fault_windows",
]

