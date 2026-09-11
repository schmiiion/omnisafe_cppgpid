"""Timestamped artifact paths for datasets, visualizations, and probe results."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path


VERSION_PATTERN = re.compile(r'^\d{8}T\d{6}\.\d{6}Z$')


def new_version_id() -> str:
    """Return a sortable, collision-resistant UTC timestamp."""
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')


def versioned_path(base_path: Path, version: str) -> Path:
    """Place a configured file below its timestamp directory."""
    if not VERSION_PATTERN.fullmatch(version):
        raise ValueError(f'Invalid artifact version: {version!r}.')
    return base_path.parent / version / base_path.name


def versioned_directory(base_directory: Path, version: str) -> Path:
    """Return a timestamped child directory below a configured artifact root."""
    if not VERSION_PATTERN.fullmatch(version):
        raise ValueError(f'Invalid artifact version: {version!r}.')
    return base_directory / version


def resolve_versioned_file(base_path: Path, version: str) -> tuple[Path, str | None]:
    """Resolve one explicit/latest timestamped file, with unversioned support."""
    if version == 'unversioned':
        return base_path, None
    if version != 'latest':
        return versioned_path(base_path, version), version
    if not base_path.parent.is_dir():
        raise FileNotFoundError(f'No versioned artifacts found under {base_path.parent}.')
    for directory in sorted(base_path.parent.iterdir(), reverse=True):
        if directory.is_dir() and VERSION_PATTERN.fullmatch(directory.name):
            candidate = directory / base_path.name
            if candidate.is_file():
                return candidate, directory.name
    raise FileNotFoundError(f'No versioned `{base_path.name}` found under {base_path.parent}.')


def resolve_dataset_version(
    dataset_base_path: Path,
    metadata_base_path: Path,
    version: str,
) -> tuple[Path, Path, str | None]:
    """Resolve an explicit/latest dataset version, with legacy unversioned support."""
    if version == 'unversioned':
        return dataset_base_path, metadata_base_path, None
    if version != 'latest':
        return (
            versioned_path(dataset_base_path, version),
            versioned_path(metadata_base_path, version),
            version,
        )

    if not dataset_base_path.parent.is_dir():
        raise FileNotFoundError(
            f'No versioned datasets found under {dataset_base_path.parent}.',
        )
    for directory in sorted(dataset_base_path.parent.iterdir(), reverse=True):
        if not directory.is_dir() or not VERSION_PATTERN.fullmatch(directory.name):
            continue
        candidate_dataset = directory / dataset_base_path.name
        candidate_metadata = metadata_base_path.parent / directory.name / metadata_base_path.name
        if candidate_dataset.is_file() and candidate_metadata.is_file():
            return candidate_dataset, candidate_metadata, directory.name
    raise FileNotFoundError(
        f'No complete versioned dataset and metadata pair found under {dataset_base_path.parent}.',
    )
