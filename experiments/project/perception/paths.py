"""Minimal path helpers for the public perception package."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ManifestBundlePaths:
    """Resolved files for one perception manifest bundle."""

    manifest_dir: Path
    manifest_path: Path
    cases_path: Path
    references_path: Path
    audit_path: Path


@dataclass(frozen=True)
class EmbeddingBundlePaths:
    """Resolved files for one perception embedding bundle."""

    bundle_dir: Path
    metadata_path: Path
    reference_index_path: Path
    embeddings_path: Path


@dataclass(frozen=True)
class PerceptionProbeRunPaths:
    """Resolved files for one perception probe artifact bundle."""

    run_dir: Path
    spec_path: Path
    summary_path: Path
    rows_path: Path


@dataclass(frozen=True)
class PerceptionProtocolRunPaths:
    """Resolved files for one perception protocol artifact bundle."""

    run_dir: Path
    spec_path: Path
    summary_path: Path
    rows_path: Path


@dataclass(frozen=True)
class ProjectPaths:
    """Resolved public-package paths."""

    repo_root: Path
    project_root: Path
    data_root: Path
    manifests_root: Path
    artifacts_root: Path

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
            manifests_root=data_root / "manifests",
            artifacts_root=project_root / "artifacts",
        )

    def ensure_layout(self) -> None:
        """Create the small public-package directory layout when needed."""
        for directory in (
            self.project_root,
            self.data_root,
            self.manifests_root,
            self.artifacts_root,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def manifest_bundle(self, manifest_id: str) -> ManifestBundlePaths:
        """Resolve one manifest bundle."""
        manifest_dir = self.manifests_root / manifest_id
        return ManifestBundlePaths(
            manifest_dir=manifest_dir,
            manifest_path=manifest_dir / "manifest.json",
            cases_path=manifest_dir / "cases.jsonl",
            references_path=manifest_dir / "references.jsonl",
            audit_path=manifest_dir / "audit.json",
        )

    def perception_embedding_bundle(
        self,
        manifest_id: str,
        bundle_id: str,
    ) -> EmbeddingBundlePaths:
        """Resolve one perception embedding bundle."""
        bundle_dir = (
            self.artifacts_root
            / "manifests"
            / manifest_id
            / "perception_embeddings"
            / bundle_id
        )
        return EmbeddingBundlePaths(
            bundle_dir=bundle_dir,
            metadata_path=bundle_dir / "metadata.json",
            reference_index_path=bundle_dir / "reference_index.json",
            embeddings_path=bundle_dir / "embeddings.npz",
        )

    def perception_probe_run(
        self,
        manifest_id: str,
        run_id: str,
    ) -> PerceptionProbeRunPaths:
        """Resolve one perception probe artifact bundle."""
        run_dir = (
            self.artifacts_root
            / "manifests"
            / manifest_id
            / "perception_probe_runs"
            / run_id
        )
        return PerceptionProbeRunPaths(
            run_dir=run_dir,
            spec_path=run_dir / "spec.json",
            summary_path=run_dir / "summary.json",
            rows_path=run_dir / "rows.jsonl",
        )

    def perception_protocol_run(
        self,
        manifest_id: str,
        run_id: str,
    ) -> PerceptionProtocolRunPaths:
        """Resolve one perception protocol artifact bundle."""
        run_dir = (
            self.artifacts_root
            / "manifests"
            / manifest_id
            / "perception_protocol_runs"
            / run_id
        )
        return PerceptionProtocolRunPaths(
            run_dir=run_dir,
            spec_path=run_dir / "spec.json",
            summary_path=run_dir / "summary.json",
            rows_path=run_dir / "rows.jsonl",
        )


def project_paths() -> ProjectPaths:
    """Return discovered project paths."""
    return ProjectPaths.discover()
