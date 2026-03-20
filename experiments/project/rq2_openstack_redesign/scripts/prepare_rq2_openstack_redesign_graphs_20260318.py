from __future__ import annotations

import argparse
from pathlib import Path

from rq2_openstack_redesign_common_20260318 import (
    GRAPH_FILES,
    REPORTS_DIR,
    SOURCE_GRAPH_FILES,
    copy_graph_files,
    ensure_dirs,
)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    ensure_dirs()
    report_path = REPORTS_DIR / "rq2_openstack_redesign_graph_prep_20260318.md"
    if all(path.exists() for path in GRAPH_FILES.values()) and report_path.exists() and not args.force:
        print(f"[*] Reusing graph copies under {GRAPH_FILES['modified'].parent}")
        print(f"[*] Reusing graph prep report: {report_path}")
        return

    print("[1/2] Copying frozen graph artifacts from the audit fork. Expected: <5s")
    copied = copy_graph_files()

    print("[2/2] Writing graph prep report. Expected: <5s")
    lines = [
        "# RQ2 OpenStack Redesign Graph Prep (2026-03-18)",
        "",
        "- Graph semantics are reused unchanged from the current audit fork.",
        "- No old graph-building script was rerun in this redesign workspace.",
        "- HDFS remains frozen; OpenStack benchmark changes only affect evaluation rows.",
        "",
        "## Copied Files",
        "",
    ]
    for name, path in copied.items():
        lines.append(f"- `{name}`: `{path}`")
        lines.append(f"  source: `{SOURCE_GRAPH_FILES[name]}`")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    for name, path in copied.items():
        print(f"[Saved] {name}: {path}")
    print(f"[Saved] {report_path}")


if __name__ == "__main__":
    main()
