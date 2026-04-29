"""Reference-row helpers for perception retrieval."""

from __future__ import annotations

from typing import Iterator

from experiments.project.perception.core.models import ReferenceRow
from experiments.project.perception.manifest_io import iter_references
from experiments.project.perception.paths import ProjectPaths


def load_reference_rows(
    manifest_id: str, paths: ProjectPaths | None = None
) -> list[ReferenceRow]:
    """Load perception reference rows in stable manifest order."""
    return [ReferenceRow.from_dict(row) for row in iter_references(manifest_id, paths=paths)]


def iter_reference_rows(
    manifest_id: str, paths: ProjectPaths | None = None
) -> Iterator[ReferenceRow]:
    """Iterate perception reference rows in stable manifest order."""
    for row in iter_references(manifest_id, paths=paths):
        yield ReferenceRow.from_dict(row)
