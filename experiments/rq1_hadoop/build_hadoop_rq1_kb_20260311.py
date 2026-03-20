"""
RQ1-Hadoop: Build Knowledge Base for NeSy-Edge parsing.

Design:
- Use data/raw/Hadoop/abnormal_label.txt to find applications labeled "normal".
- For each normal application, iterate its log lines and parse with Drain.
- Store (raw_log_line, drain_template) into the shared KnowledgeBase
  with dataset_type="Hadoop", so NuSyEdgeNode can reuse it via RAG.

Notes:
- This script is Stage0/Stage1 tooling: it only writes to the ChromaDB
  under data/chroma_db, and does NOT touch benchmark JSON or GT labels.
- It is safe to run multiple times; duplicates are acceptable because
  retriever uses similarity rather than exact ID semantics.
"""

import os
import sys
from typing import List, Dict, Tuple

from tqdm import tqdm

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from experiments.rq123_e2e import hadoop_loader  # type: ignore
from src.perception.drain_parser import DrainParser
from src.system.knowledge_base import KnowledgeBase
from src.system.edge_node import NuSyEdgeNode


def _load_normal_and_abnormal(
    abnormal_label_path: str,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    rows = hadoop_loader.parse_abnormal_label_file(abnormal_label_path)
    normals: List[Dict[str, str]] = []
    abnormals: List[Dict[str, str]] = []
    for r in rows:
        lbl = r.get("label", "").lower()
        if lbl == "normal":
            normals.append(r)
        else:
            abnormals.append(r)
    return normals, abnormals


def build_hadoop_kb(
    max_normal_apps: int = 20,
    max_abnormal_apps: int = 20,
    max_lines_per_app: int = 200,
) -> None:
    data_raw = os.path.join(_PROJECT_ROOT, "data", "raw")
    hadoop_root = os.path.join(data_raw, "Hadoop")
    abnormal_label_path = os.path.join(hadoop_root, "abnormal_label.txt")

    if not os.path.exists(hadoop_root) or not os.path.exists(abnormal_label_path):
        raise FileNotFoundError(f"Hadoop raw logs or abnormal_label.txt not found under {hadoop_root}")

    normals, abnormals = _load_normal_and_abnormal(abnormal_label_path)
    if not normals and not abnormals:
        print("[WARN] No Hadoop applications found in abnormal_label.txt")
        return

    normals = normals[:max_normal_apps]
    abnormals = abnormals[:max_abnormal_apps]

    kb = KnowledgeBase()
    drain = DrainParser()

    raw_logs_all: List[str] = []
    templates_all: List[str] = []

    print(
        f"[INFO] Selected {len(normals)} normal and {len(abnormals)} abnormal "
        "Hadoop applications for KB building."
    )

    # 1) ingest logs from normal applications
    for row in tqdm(normals, desc="KB from normal apps", unit="app"):
        app_id = row["application_id"]
        try:
            lines = hadoop_loader.iter_hadoop_application_logs(hadoop_root, app_id)
        except FileNotFoundError:
            continue
        if not lines:
            continue

        used = 0
        for raw in lines:
            if used >= max_lines_per_app:
                break
            clean = NuSyEdgeNode.preprocess_header(raw, "Hadoop") or raw
            try:
                tpl = drain.parse(clean)
            except Exception:
                continue
            if not tpl:
                continue
            raw_logs_all.append(clean)
            templates_all.append(tpl)
            used += 1

    # 2) ingest non-tail logs from abnormal applications (avoid直接用评估 alert)
    for row in tqdm(abnormals, desc="KB from abnormal (non-tail) logs", unit="app"):
        app_id = row["application_id"]
        try:
            lines = hadoop_loader.iter_hadoop_application_logs(hadoop_root, app_id)
        except FileNotFoundError:
            continue
        if not lines:
            continue
        # exclude最后一行（通常是我们在 benchmark 里用作 alert 的行）
        core_lines = lines[:-1] if len(lines) > 1 else []
        if not core_lines:
            continue
        used = 0
        for raw in core_lines:
            if used >= max_lines_per_app:
                break
            clean = NuSyEdgeNode.preprocess_header(raw, "Hadoop") or raw
            try:
                tpl = drain.parse(clean)
            except Exception:
                continue
            if not tpl:
                continue
            raw_logs_all.append(clean)
            templates_all.append(tpl)
            used += 1

    if not raw_logs_all:
        print("[WARN] No Hadoop log/template pairs collected; KB not updated.")
        return

    print(f"[INFO] Ingesting {len(raw_logs_all)} Hadoop log/template pairs into KB (dataset=Hadoop).")
    kb.add_knowledge(raw_logs_all, templates_all, dataset_type="Hadoop")
    print("[DONE] Hadoop RQ1 KB build completed.")


if __name__ == "__main__":
    build_hadoop_kb()

