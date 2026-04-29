"""Minimal path helpers for the public reasoning package."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    """Resolved public-package paths for optional reasoning assets."""

    repo_root: Path
    project_root: Path
    data_root: Path
    knowledge_root: Path

    @classmethod
    def discover(cls) -> "ProjectPaths":
        project_root = Path(__file__).resolve().parents[1]
        return cls.from_project_root(project_root)

    @classmethod
    def from_project_root(cls, project_root: Path) -> "ProjectPaths":
        repo_root = project_root.parents[1]
        data_root = project_root / "data"
        return cls(
            repo_root=repo_root,
            project_root=project_root,
            data_root=data_root,
            knowledge_root=data_root / "knowledge" / "reasoning",
        )

    def ensure_layout(self) -> None:
        for directory in (self.project_root, self.data_root, self.knowledge_root):
            directory.mkdir(parents=True, exist_ok=True)

    def reasoning_knowledge_path(self, asset_name: str) -> Path:
        return self.knowledge_root / asset_name


def project_paths() -> ProjectPaths:
    """Return discovered project paths."""
    return ProjectPaths.discover()
