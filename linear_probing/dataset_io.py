"""Persistence and validation for reusable linear-probing datasets."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import tempfile
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np

from linear_probing.targets import TASK_TYPES


DATASET_SCHEMA_VERSION = 5
TARGET_SCHEMA_VERSION = 3


def file_sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file without loading it fully into memory."""
    digest = hashlib.sha256()
    with path.open('rb') as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def reproducibility_metadata(repository_root: Path) -> dict[str, Any]:
    """Capture code and dependency versions relevant to dataset generation."""
    packages = {}
    for package in ('numpy', 'torch', 'gymnasium', 'safety-gymnasium', 'mujoco'):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None

    commit = None
    dirty = None
    try:
        commit_result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        )
        status_result = subprocess.run(
            ['git', 'status', '--porcelain'],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        )
        commit = commit_result.stdout.strip()
        dirty = bool(status_result.stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        pass

    source_hashes = {
        str(path.relative_to(repository_root)): file_sha256(path)
        for path in sorted((repository_root / 'linear_probing').glob('*.py'))
    }
    return {
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'python': platform.python_version(),
        'platform': platform.platform(),
        'packages': packages,
        'git_commit': commit,
        'git_worktree_dirty': dirty,
        'source_sha256': source_hashes,
    }


def metadata_from_dataset(dataset: Any) -> dict[str, Any]:
    """Decode the metadata embedded in a dataset mapping or loaded NPZ file."""
    if 'metadata_json' not in dataset:
        raise ValueError('The dataset contains no embedded `metadata_json`.')
    value = np.asarray(dataset['metadata_json'])
    if value.shape != ():
        raise ValueError('Dataset `metadata_json` must be a scalar string.')
    metadata = json.loads(str(value.item()))
    if not isinstance(metadata, dict):
        raise ValueError('Dataset metadata must decode to a JSON object.')
    return metadata


def save_dataset_artifact(
    dataset_path: Path,
    metadata_path: Path,
    arrays: dict[str, np.ndarray],
) -> None:
    """Save compressed arrays and a human-readable, checksummed metadata sidecar."""
    if dataset_path.suffix != '.npz':
        raise ValueError('The configured dataset output path must end in `.npz`.')
    if dataset_path.resolve() == metadata_path.resolve():
        raise ValueError('Dataset and metadata paths must be different files.')
    metadata = metadata_from_dataset(arrays)
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_file = tempfile.NamedTemporaryFile(
        dir=dataset_path.parent,
        prefix=f'.{dataset_path.name}.',
        suffix='.tmp',
        delete=False,
    )
    metadata_file = tempfile.NamedTemporaryFile(
        dir=metadata_path.parent,
        prefix=f'.{metadata_path.name}.',
        suffix='.tmp',
        delete=False,
    )
    temporary_dataset = Path(dataset_file.name)
    temporary_metadata = Path(metadata_file.name)
    dataset_file.close()
    metadata_file.close()
    try:
        with temporary_dataset.open('wb') as output:
            np.savez_compressed(output, **arrays)
            output.flush()
            os.fsync(output.fileno())
        sidecar = {
            'metadata': metadata,
            'artifact': {
                'dataset_path': str(dataset_path.resolve()),
                'sha256': file_sha256(temporary_dataset),
            },
        }
        with temporary_metadata.open('w', encoding='utf-8') as output:
            json.dump(sidecar, output, indent=2, sort_keys=True)
            output.write('\n')
            output.flush()
            os.fsync(output.fileno())
        temporary_dataset.replace(dataset_path)
        temporary_metadata.replace(metadata_path)
    finally:
        temporary_dataset.unlink(missing_ok=True)
        temporary_metadata.unlink(missing_ok=True)


def validate_dataset_artifact(
    dataset_path: Path,
    metadata_path: Path,
    dataset: Any,
) -> dict[str, Any]:
    """Validate sidecar integrity and the stored array schema, then return metadata."""
    embedded = metadata_from_dataset(dataset)
    if embedded.get('schema_version') != DATASET_SCHEMA_VERSION:
        raise ValueError(
            'Unsupported dataset schema. Regenerate the dataset with the current code.',
        )
    if embedded.get('target_schema_version') != TARGET_SCHEMA_VERSION:
        raise ValueError(
            'Outdated target definitions. Regenerate the dataset with the current code.',
        )
    if embedded.get('observation_is_raw') is not True:
        raise ValueError('The probe dataset must contain raw, unnormalized observations.')
    if not isinstance(embedded.get('env_id'), str) or not embedded['env_id']:
        raise ValueError('Dataset metadata has no valid environment ID.')
    if embedded.get('target_types') != TASK_TYPES:
        raise ValueError('Dataset task definitions do not match the current probing code.')
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f'Dataset metadata sidecar not found: {metadata_path}. Regenerate the dataset.',
        )
    sidecar = json.loads(metadata_path.read_text(encoding='utf-8'))
    if not isinstance(sidecar, dict):
        raise ValueError('Dataset metadata sidecar must contain a JSON object.')
    if sidecar.get('metadata') != embedded:
        raise ValueError('Dataset sidecar metadata does not match the embedded metadata.')
    artifact = sidecar.get('artifact')
    if not isinstance(artifact, dict):
        raise ValueError('Dataset metadata sidecar has no artifact description.')
    expected_digest = artifact.get('sha256')
    if not expected_digest or file_sha256(dataset_path) != expected_digest:
        raise ValueError('Dataset checksum does not match its metadata sidecar.')

    array_schema = embedded.get('array_schema')
    if not isinstance(array_schema, dict) or not array_schema:
        raise ValueError('Dataset metadata contains no array schema.')
    stored_sample_count = embedded.get('num_samples')
    if (
        not isinstance(stored_sample_count, int)
        or isinstance(stored_sample_count, bool)
        or stored_sample_count <= 0
    ):
        raise ValueError('Dataset metadata has an invalid sample count.')
    sample_count = stored_sample_count
    required_arrays = {'observations', 'episode_ids', 'sources'} | {
        f'target_{task}' for task in embedded.get('target_types', {})
    }
    missing_arrays = sorted(required_arrays - set(array_schema))
    if missing_arrays:
        raise ValueError(f'Dataset array schema is missing required arrays: {missing_arrays}.')

    for name, specification in array_schema.items():
        if not isinstance(specification, dict):
            raise ValueError(f'Dataset array schema for `{name}` must be a mapping.')
        if name not in dataset:
            raise ValueError(f'Dataset array `{name}` is missing.')
        array = dataset[name]
        shape_mismatch = list(array.shape) != specification.get('shape')
        dtype_mismatch = str(array.dtype) != specification.get('dtype')
        if shape_mismatch or dtype_mismatch:
            raise ValueError(f'Dataset array `{name}` does not match its stored schema.')
        if array.ndim == 0 or len(array) != sample_count:
            raise ValueError(f'Dataset array `{name}` does not contain {sample_count} rows.')
    observations = np.asarray(dataset['observations'])
    if observations.ndim != 2 or not np.all(np.isfinite(observations)):
        raise ValueError('Dataset observations must be a finite two-dimensional array.')
    if list(observations.shape[1:]) != embedded.get('observation_shape'):
        raise ValueError('Dataset observation shape does not match its metadata.')
    episode_ids = np.asarray(dataset['episode_ids'])
    if episode_ids.shape != (sample_count,) or not np.issubdtype(
        episode_ids.dtype,
        np.integer,
    ):
        raise ValueError('Dataset episode IDs must be a one-dimensional per-sample array.')
    sources = np.asarray(dataset['sources'])
    if sources.shape != (sample_count,) or sources.dtype.kind not in {'U', 'S'}:
        raise ValueError('Dataset sources must be a one-dimensional per-sample array.')
    for task in TASK_TYPES:
        target = np.asarray(dataset[f'target_{task}'])
        if not np.issubdtype(target.dtype, np.number) or np.any(np.isinf(target)):
            raise ValueError(f'Dataset target `{task}` must be numeric and contain no infinities.')
    return embedded
