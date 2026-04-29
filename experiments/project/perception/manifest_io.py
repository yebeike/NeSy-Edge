"""Small manifest readers for the public perception package."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

from experiments.project.perception.paths import ManifestBundlePaths, ProjectPaths, project_paths
from experiments.project.shared.jsonio import iter_jsonl, read_json


def manifest_paths(
    manifest_id: str,
    *,
    paths: ProjectPaths | None = None,
) -> ManifestBundlePaths:
    """Resolve one manifest bundle."""
    resolved_paths = paths or project_paths()
    return resolved_paths.manifest_bundle(manifest_id)


def load_manifest(
    manifest_id: str,
    *,
    paths: ProjectPaths | None = None,
) -> dict[str, Any]:
    """Load raw manifest metadata."""
    bundle = manifest_paths(manifest_id, paths=paths)
    payload = read_json(bundle.manifest_path)
    if not isinstance(payload, dict):
        raise TypeError("manifest.json must contain an object payload")
    return dict(payload)


def iter_cases(
    manifest_id: str,
    *,
    paths: ProjectPaths | None = None,
) -> Iterator[dict[str, Any]]:
    """Iterate manifest case rows."""
    bundle = manifest_paths(manifest_id, paths=paths)
    for row in iter_jsonl(bundle.cases_path):
        if isinstance(row, Mapping):
            yield dict(row)


def iter_references(
    manifest_id: str,
    *,
    paths: ProjectPaths | None = None,
) -> Iterator[dict[str, Any]]:
    """Iterate manifest reference rows."""
    bundle = manifest_paths(manifest_id, paths=paths)
    for row in iter_jsonl(bundle.references_path):
        if isinstance(row, Mapping):
            yield dict(row)
