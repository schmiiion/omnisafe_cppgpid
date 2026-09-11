"""Fit linear probes on frozen actor encoders from saved representation snapshots."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from linear_probing.artifacts import new_version_id, resolve_dataset_version, versioned_path
from linear_probing.config import DEFAULT_CONFIG_PATH, load_config, resolve_path, task_switches
from linear_probing.dataset_io import file_sha256, validate_dataset_artifact
from linear_probing.snapshots import (
    build_actor,
    ensure_disjoint_snapshot_runs,
    load_snapshot,
    read_only_normalize,
    select_snapshots,
    snapshot_run_directory,
)
from linear_probing.targets import TASK_TYPES


RESULT_COLUMNS = (
    'dataset_version',
    'result_version',
    'dataset_path',
    'dataset_sha256',
    'dataset_schema_version',
    'target_schema_version',
    'dataset_seed',
    'dataset_environment',
    'probe_config_sha256',
    'probe_code_sha256',
    'split_seed',
    'test_fraction',
    'actual_test_fraction',
    'use_snapshot_normalizer',
    'device',
    'ridge_alpha',
    'logistic_c',
    'max_iterations',
    'embedding_batch_size',
    'algorithm',
    'environment',
    'training_seed',
    'run_directory',
    'snapshot_path',
    'snapshot_sha256',
    'snapshot_schema_version',
    'snapshot_config_sha256',
    'actor_class',
    'representation_module',
    'phase',
    'stage',
    'stage_iteration',
    'total_env_steps',
    'snapshot_observation_normalized',
    'normalizer_count',
    'task',
    'task_type',
    'target_dimension',
    'n_train',
    'n_test',
    'train_class_counts',
    'test_class_counts',
    'r2',
    'mae',
    'accuracy',
    'balanced_accuracy',
    'f1_macro',
    'roc_auc',
    'status',
)


def _json_sha256(value: Any) -> str:
    """Return a stable digest for JSON-compatible configuration data."""
    serialized = json.dumps(value, sort_keys=True, separators=(',', ':'), default=str)
    return hashlib.sha256(serialized.encode()).hexdigest()


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _probe_code_sha256() -> str:
    """Identify the code that reconstructs and probes representations."""
    digest = hashlib.sha256()
    directory = Path(__file__).resolve().parent
    for name in ('run_probes.py', 'snapshots.py', 'targets.py'):
        digest.update(name.encode())
        digest.update((directory / name).read_bytes())
    return digest.hexdigest()


def _positive_integer(config: dict[str, Any], name: str) -> int:
    value = config.get(name)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f'`probing.{name}` must be a positive integer.')
    return value


def validate_probe_config(config: dict[str, Any]) -> None:
    """Reject ambiguous or invalid probing hyperparameters before doing any work."""
    dataset = config.get('dataset')
    probing = config.get('probing')
    snapshots = config.get('snapshots')
    if not all(isinstance(section, dict) for section in (dataset, probing, snapshots)):
        raise ValueError(
            'The configuration requires `dataset`, `probing`, and `snapshots` mappings.',
        )
    dataset_version = dataset.get('version', 'latest')
    if not isinstance(dataset_version, str) or not dataset_version.strip():
        raise ValueError('`dataset.version` must be `latest`, `unversioned`, or a timestamp.')
    if not isinstance(probing.get('enabled', True), bool):
        raise ValueError('`probing.enabled` must be a boolean.')
    if not isinstance(snapshots.get('use_snapshot_normalizer'), bool):
        raise ValueError('`snapshots.use_snapshot_normalizer` must be a boolean.')
    if not isinstance(snapshots.get('device'), str) or not snapshots['device'].strip():
        raise ValueError('`snapshots.device` must be a non-empty string.')

    test_fraction = probing.get('test_fraction')
    if (
        not isinstance(test_fraction, (int, float))
        or isinstance(test_fraction, bool)
        or not np.isfinite(test_fraction)
        or not 0.0 < float(test_fraction) < 1.0
    ):
        raise ValueError('`probing.test_fraction` must lie strictly between 0 and 1.')
    for name, allow_zero in (('ridge_alpha', True), ('logistic_c', False)):
        value = probing.get(name)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not np.isfinite(value)
            or float(value) < 0.0
            or (not allow_zero and float(value) == 0.0)
        ):
            qualifier = 'non-negative' if allow_zero else 'positive'
            raise ValueError(f'`probing.{name}` must be a finite {qualifier} number.')
    _positive_integer(probing, 'max_iterations')
    _positive_integer(probing, 'embedding_batch_size')
    if not isinstance(probing.get('seed'), int) or isinstance(probing.get('seed'), bool):
        raise ValueError('`probing.seed` must be an integer.')


def _build_frozen_encoder(snapshot: dict[str, Any], device: str) -> torch.nn.Module:
    actor = build_actor(snapshot, device)
    module_name = snapshot['actor_spec']['representation_module']
    encoder = getattr(actor, module_name, None)
    if not isinstance(encoder, torch.nn.Module):
        raise ValueError(f'Actor has no representation module named `{module_name}`.')
    return encoder


def extract_embeddings(
    observations: np.ndarray,
    snapshot: dict[str, Any],
    device: str,
    batch_size: int,
    use_snapshot_normalizer: bool,
) -> tuple[np.ndarray, int | None]:
    """Run raw observations through one frozen, reconstructed encoder."""
    if observations.ndim < 2 or len(observations) == 0:
        raise ValueError('Probe observations must be a non-empty batched array.')
    if batch_size <= 0:
        raise ValueError('Embedding batch size must be positive.')
    encoder = _build_frozen_encoder(snapshot, device)
    expected_shape = tuple(snapshot['actor_spec']['observation_shape'])
    if tuple(observations.shape[1:]) != expected_shape:
        raise ValueError(
            f'Dataset observation shape {observations.shape[1:]} does not match {expected_shape}.',
        )

    batches = []
    normalizer_count = None
    with torch.no_grad():
        for start in range(0, len(observations), batch_size):
            batch = torch.as_tensor(
                observations[start : start + batch_size],
                dtype=torch.float32,
                device=device,
            )
            batch, normalizer_count = read_only_normalize(
                batch,
                snapshot,
                use_snapshot_normalizer,
            )
            batches.append(encoder(batch).cpu().numpy())
    embeddings = np.concatenate(batches)
    if embeddings.ndim != 2 or not np.all(np.isfinite(embeddings)):
        raise ValueError('The frozen encoder produced non-finite or non-matrix embeddings.')
    return embeddings, normalizer_count


def grouped_train_test_split(
    episode_ids: np.ndarray,
    test_fraction: float,
    seed: int,
    sources: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Split whole episodes independently within each collection source."""
    episode_ids = np.asarray(episode_ids)
    if episode_ids.ndim != 1 or len(episode_ids) == 0:
        raise ValueError('`episode_ids` must be a non-empty one-dimensional array.')
    if not np.issubdtype(episode_ids.dtype, np.integer):
        raise ValueError('`episode_ids` must contain integers.')
    if (
        not isinstance(test_fraction, (int, float))
        or isinstance(test_fraction, bool)
        or not np.isfinite(test_fraction)
        or not 0.0 < test_fraction < 1.0
    ):
        raise ValueError('Test fraction must lie strictly between 0 and 1.')
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError('Split seed must be an integer.')
    if sources is None:
        sources = np.full(len(episode_ids), 'all')
    sources = np.asarray(sources)
    if sources.ndim != 1 or len(sources) != len(episode_ids):
        raise ValueError('`sources` must be one-dimensional and align with `episode_ids`.')
    if any(not isinstance(source, str) or not source for source in sources.tolist()):
        raise ValueError('`sources` must contain non-empty strings.')

    group_sources: dict[Any, Any] = {}
    for group, source in zip(episode_ids.tolist(), sources.tolist()):
        if group in group_sources and group_sources[group] != source:
            raise ValueError(f'Episode {group!r} occurs under multiple collection sources.')
        group_sources[group] = source

    generator = np.random.default_rng(seed)
    test_mask = np.zeros(len(episode_ids), dtype=bool)
    for source in np.unique(sources):
        source_mask = sources == source
        groups = np.asarray(
            sorted(group for group, value in group_sources.items() if value == source),
        )
        if len(groups) < 2:
            raise ValueError(
                f'Collection source {source!r} needs at least two sampled episodes for a '
                'source-stratified grouped split. Increase the dataset size.',
            )
        generator.shuffle(groups)
        test_group_count = min(max(round(len(groups) * test_fraction), 1), len(groups) - 1)
        test_mask |= source_mask & np.isin(episode_ids, groups[:test_group_count])

    train_mask = ~test_mask
    if not train_mask.any() or not test_mask.any():
        raise ValueError('Source-stratified grouped split produced an empty partition.')
    return train_mask, test_mask


def _empty_metrics() -> dict[str, float | str]:
    return {
        'r2': np.nan,
        'mae': np.nan,
        'accuracy': np.nan,
        'balanced_accuracy': np.nan,
        'f1_macro': np.nan,
        'roc_auc': np.nan,
        'train_class_counts': '',
        'test_class_counts': '',
        'status': 'ok',
    }


def _pooled_r2(target: np.ndarray, prediction: np.ndarray) -> float:
    """Compute variance-weighted R2 without coercing constant targets to perfect scores."""
    target = np.asarray(target, dtype=np.float64).reshape(len(target), -1)
    prediction = np.asarray(prediction, dtype=np.float64).reshape(len(target), -1)
    denominator = float(np.square(target - target.mean(axis=0)).sum())
    if denominator <= 0.0:
        return np.nan
    return 1.0 - float(np.square(target - prediction).sum()) / denominator


def _has_variation(target: np.ndarray, every_output: bool = False) -> bool:
    target = np.asarray(target, dtype=np.float64).reshape(len(target), -1)
    variation = np.square(target - target.mean(axis=0)).sum(axis=0) > 0.0
    return bool(np.all(variation) if every_output else np.any(variation))


def _ridge_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    alpha: float,
) -> np.ndarray:
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    model.fit(x_train, y_train)
    return np.asarray(model.predict(x_test))


def fit_probe(
    embeddings: np.ndarray,
    target: np.ndarray,
    task_type: str,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    probing_config: dict[str, Any],
) -> dict[str, float | str | int]:
    """Fit one linear model and return its held-out metrics."""
    embeddings = np.asarray(embeddings)
    target = np.asarray(target)
    train_mask = np.asarray(train_mask)
    test_mask = np.asarray(test_mask)
    if embeddings.ndim != 2 or not np.all(np.isfinite(embeddings)):
        raise ValueError('Embeddings must be a finite two-dimensional array.')
    if target.ndim not in {1, 2} or len(target) != len(embeddings):
        raise ValueError('Targets must be one- or two-dimensional and align with embeddings.')
    if (
        train_mask.shape != (len(target),)
        or test_mask.shape != (len(target),)
        or train_mask.dtype != np.bool_
        or test_mask.dtype != np.bool_
        or np.any(train_mask & test_mask)
        or not np.all(train_mask | test_mask)
    ):
        raise ValueError('Train/test masks must be disjoint boolean partitions of the dataset.')

    target_dimension = int(np.prod(target.shape[1:])) if target.ndim > 1 else 1
    if task_type not in {'regression', 'circular_regression', 'classification'}:
        raise ValueError(f'Unknown probe task type: {task_type!r}.')
    if task_type in {'circular_regression', 'classification'} and target_dimension != 1:
        raise ValueError(f'{task_type.replace("_", " ").title()} targets must be scalar.')
    reported_dimension = 2 if task_type == 'circular_regression' else target_dimension
    finite = np.isfinite(target)
    if target.ndim > 1:
        finite = np.all(finite, axis=tuple(range(1, target.ndim)))
    train = train_mask & finite
    test = test_mask & finite
    metrics: dict[str, float | str | int] = _empty_metrics()
    metrics.update(
        {
            'target_dimension': reported_dimension,
            'n_train': int(train.sum()),
            'n_test': int(test.sum()),
        },
    )
    if train.sum() < 2 or test.sum() < 2:
        metrics['status'] = 'fewer than two finite train/test samples'
        return metrics

    x_train, x_test = embeddings[train], embeddings[test]
    y_train, y_test = target[train], target[test]
    if task_type == 'regression':
        if not _has_variation(y_train, every_output=True):
            metrics['status'] = 'constant training target'
            return metrics
        if not _has_variation(y_test, every_output=True):
            metrics['status'] = 'constant test target'
            return metrics
        prediction = _ridge_predict(
            x_train,
            y_train,
            x_test,
            float(probing_config['ridge_alpha']),
        )
        metrics['mae'] = float(mean_absolute_error(y_test, prediction))
        metrics['r2'] = _pooled_r2(y_test, prediction)
        return metrics

    if task_type == 'circular_regression':
        train_angles = y_train.reshape(-1)
        test_angles = y_test.reshape(-1)
        encoded_train = np.column_stack((np.sin(train_angles), np.cos(train_angles)))
        encoded_test = np.column_stack((np.sin(test_angles), np.cos(test_angles)))
        if not _has_variation(encoded_train):
            metrics['status'] = 'constant circular training target'
            return metrics
        if not _has_variation(encoded_test):
            metrics['status'] = 'constant circular test target'
            return metrics
        prediction = _ridge_predict(
            x_train,
            encoded_train,
            x_test,
            float(probing_config['ridge_alpha']),
        )
        predicted_angles = np.arctan2(prediction[:, 0], prediction[:, 1])
        angular_error = np.arctan2(
            np.sin(predicted_angles - test_angles),
            np.cos(predicted_angles - test_angles),
        )
        metrics['mae'] = float(np.mean(np.abs(angular_error)))
        metrics['r2'] = _pooled_r2(encoded_test, prediction)
        return metrics

    y_train = y_train.reshape(-1)
    y_test = y_test.reshape(-1)
    if not np.all(y_train == np.round(y_train)) or not np.all(y_test == np.round(y_test)):
        metrics['status'] = 'classification target contains non-integral labels'
        return metrics
    y_train = y_train.astype(np.int64)
    y_test = y_test.astype(np.int64)
    train_classes = np.unique(y_train)
    test_classes = np.unique(y_test)
    metrics['train_class_counts'] = json.dumps(
        {str(label): int(np.count_nonzero(y_train == label)) for label in train_classes},
        sort_keys=True,
    )
    metrics['test_class_counts'] = json.dumps(
        {str(label): int(np.count_nonzero(y_test == label)) for label in test_classes},
        sort_keys=True,
    )
    if len(train_classes) != 2:
        metrics['status'] = 'training split does not contain exactly two classes'
        return metrics
    if not np.array_equal(test_classes, train_classes):
        metrics['status'] = 'test split does not contain both training classes'
        return metrics
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=float(probing_config['logistic_c']),
            max_iter=int(probing_config['max_iterations']),
            class_weight='balanced',
            random_state=int(probing_config['seed']),
        ),
    )
    with warnings.catch_warnings(record=True) as fit_warnings:
        warnings.simplefilter('always', ConvergenceWarning)
        model.fit(x_train, y_train)
    prediction = model.predict(x_test)
    metrics['accuracy'] = float(accuracy_score(y_test, prediction))
    metrics['balanced_accuracy'] = float(balanced_accuracy_score(y_test, prediction))
    metrics['f1_macro'] = float(f1_score(y_test, prediction, average='macro', zero_division=0))
    probability = model.predict_proba(x_test)[:, 1]
    metrics['roc_auc'] = float(roc_auc_score(y_test, probability))
    if any(issubclass(item.category, ConvergenceWarning) for item in fit_warnings):
        metrics['status'] = 'fit did not converge'
    return metrics


def evaluate_snapshot(
    snapshot_path: Path,
    dataset: dict[str, np.ndarray],
    dataset_environment: str,
    config: dict[str, Any],
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    result_context: dict[str, Any] | None = None,
    snapshot_digest: str | None = None,
    collector_identities: set[tuple[str, str, int]] | None = None,
) -> list[dict[str, Any]]:
    """Evaluate every enabled task for a single actor snapshot."""
    snapshot_config = config['snapshots']
    probing_config = config['probing']
    snapshot = load_snapshot(snapshot_path, snapshot_config['device'])
    training_config = snapshot.get('config', {})
    if not isinstance(training_config, dict):
        raise ValueError(f'Snapshot has invalid training configuration: {snapshot_path}')
    snapshot_environment = training_config.get('env_id')
    identity_values = (
        training_config.get('algo'),
        snapshot_environment,
        training_config.get('seed'),
    )
    if (
        not all(isinstance(value, str) and value for value in identity_values[:2])
        or not isinstance(identity_values[2], int)
        or isinstance(identity_values[2], bool)
    ):
        raise ValueError(
            f'Snapshot lacks a valid algorithm/environment/seed identity: {snapshot_path}',
        )
    snapshot_identity = identity_values[0], identity_values[1], identity_values[2]
    if collector_identities and snapshot_identity in collector_identities:
        raise ValueError(
            'A probing snapshot has the same algorithm/environment/training-seed identity as a '
            f'collector policy: {snapshot_identity}. Use an independently seeded collector run.',
        )
    if snapshot_environment != dataset_environment:
        warnings.warn(
            f'Snapshot environment {snapshot_environment!r} differs from probe dataset '
            f'{dataset_environment!r}; continuing intentionally for cross-configuration probing.',
            stacklevel=2,
        )
    embeddings, normalizer_count = extract_embeddings(
        dataset['observations'],
        snapshot,
        snapshot_config['device'],
        int(probing_config['embedding_batch_size']),
        bool(snapshot_config['use_snapshot_normalizer']),
    )

    metadata = snapshot.get('metadata', {})
    common = {
        **(result_context or {}),
        'algorithm': training_config.get('algo', ''),
        'environment': snapshot_environment or '',
        'training_seed': training_config.get('seed', ''),
        'run_directory': str(snapshot_run_directory(snapshot_path)),
        'snapshot_path': str(snapshot_path),
        'snapshot_sha256': snapshot_digest or file_sha256(snapshot_path),
        'snapshot_schema_version': metadata.get('snapshot_schema_version', ''),
        'snapshot_config_sha256': _json_sha256(training_config),
        'actor_class': snapshot['actor_spec']['class_path'],
        'representation_module': snapshot['actor_spec']['representation_module'],
        'phase': metadata.get('phase', ''),
        'stage': metadata.get('stage', ''),
        'stage_iteration': metadata.get('stage_iteration', ''),
        'total_env_steps': metadata.get('total_env_steps', ''),
        'snapshot_observation_normalized': metadata.get('observation_normalized', ''),
        'normalizer_count': normalizer_count if normalizer_count is not None else '',
    }

    rows = []
    switches = task_switches(config)
    for task, task_type in TASK_TYPES.items():
        if not switches[task]:
            continue
        target_key = f'target_{task}'
        if target_key not in dataset:
            raise KeyError(f'Dataset does not contain `{target_key}`.')
        metrics = fit_probe(
            embeddings,
            dataset[target_key],
            task_type,
            train_mask,
            test_mask,
            probing_config,
        )
        rows.append({**common, 'task': task, 'task_type': task_type, **metrics})
    return rows


def main() -> None:
    """Evaluate configured actor snapshots and persist one CSV row per snapshot/task."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(DEFAULT_CONFIG_PATH))
    args = parser.parse_args()
    config = load_config(args.config)
    validate_probe_config(config)
    if not config['probing'].get('enabled', True):
        print('Linear probing is disabled by `probing.enabled`.')
        return

    switches = task_switches(config)
    enabled_tasks = [task for task in TASK_TYPES if switches[task]]
    if not enabled_tasks:
        raise ValueError('At least one linear-probing task must be enabled.')

    dataset_base_path = resolve_path(config['dataset']['output_path'], args.config)
    metadata_base_path = resolve_path(config['dataset']['metadata_path'], args.config)
    dataset_path, metadata_path, dataset_version = resolve_dataset_version(
        dataset_base_path,
        metadata_base_path,
        config['dataset'].get('version', 'latest'),
    )
    print(f'Dataset version: {dataset_version or "unversioned"}')
    snapshot_directory = resolve_path(config['snapshots']['probing_directory'], args.config)
    output_base_path = resolve_path(config['probing']['output_csv'], args.config)
    snapshot_paths = select_snapshots(
        sorted(snapshot_directory.glob(config['snapshots']['pattern'])),
        config['snapshots']['probing_selection'],
    )
    if not snapshot_paths:
        raise FileNotFoundError(
            'No probing snapshots matched the configured directory, pattern, and selection '
            f'under {snapshot_directory}.',
        )
    with np.load(dataset_path, allow_pickle=False) as dataset:
        dataset_metadata = validate_dataset_artifact(dataset_path, metadata_path, dataset)
        stored_version = dataset_metadata.get('artifact_version')
        if dataset_version is not None and stored_version != dataset_version:
            raise ValueError(
                f'Dataset folder version {dataset_version!r} does not match its embedded '
                f'version {stored_version!r}.',
            )
        configured_environment = config['dataset']['env_id']
        dataset_environment = dataset_metadata.get('env_id')
        if configured_environment != dataset_environment:
            raise ValueError(
                f'Configured dataset environment {configured_environment!r} does not match '
                f'artifact environment {dataset_environment!r}.',
            )
        required_arrays = ['observations', 'episode_ids', 'sources'] + [
            f'target_{task}' for task in enabled_tasks
        ]
        missing_arrays = [name for name in required_arrays if name not in dataset]
        if missing_arrays:
            raise ValueError(
                f'Dataset is missing required arrays {missing_arrays}. Regenerate the dataset.',
            )
        cached_dataset = {name: np.asarray(dataset[name]) for name in required_arrays}

    expected_sources = set(dataset_metadata.get('source_candidate_counts', {}))
    observed_sources = set(cached_dataset['sources'].tolist())
    if not expected_sources or observed_sources != expected_sources:
        raise ValueError(
            f'Dataset source support {sorted(observed_sources)} does not match expected '
            f'{sorted(expected_sources)}. Regenerate a larger dataset.',
        )

    probing_config = config['probing']
    train_mask, test_mask = grouped_train_test_split(
        cached_dataset['episode_ids'],
        float(probing_config['test_fraction']),
        int(probing_config['seed']),
        sources=cached_dataset['sources'],
    )

    sidecar = json.loads(metadata_path.read_text(encoding='utf-8'))
    dataset_digest = sidecar['artifact']['sha256']
    collection_policies = dataset_metadata.get('collection_policies', [])
    if not isinstance(collection_policies, list) or not collection_policies:
        raise ValueError(
            'The dataset has no collector-policy provenance. Regenerate it before probing.',
        )
    if not all(isinstance(item, dict) for item in collection_policies):
        raise ValueError('Collector-policy provenance entries must be mappings.')
    required_identity = ('path', 'sha256', 'algorithm', 'environment', 'training_seed')
    invalid_policies = [
        index
        for index, item in enumerate(collection_policies)
        if any(item.get(key) is None for key in required_identity)
        or not isinstance(item.get('path'), str)
        or not item['path'].strip()
        or not _is_sha256(item.get('sha256'))
        or not isinstance(item.get('algorithm'), str)
        or not item['algorithm'].strip()
        or not isinstance(item.get('environment'), str)
        or not item['environment'].strip()
        or not isinstance(item.get('training_seed'), int)
        or isinstance(item.get('training_seed'), bool)
    ]
    if invalid_policies:
        raise ValueError(
            f'Collector-policy provenance entries {invalid_policies} lack a valid '
            'path/SHA-256/algorithm/environment/seed identity.',
        )
    collection_paths = [Path(item['path']) for item in collection_policies]
    collection_digests = [item['sha256'] for item in collection_policies]
    if len(set(collection_digests)) != len(collection_digests):
        raise ValueError('Collector-policy provenance contains duplicate checkpoints.')
    collection_hashes = set(collection_digests)
    collector_identities = {
        (item['algorithm'], item['environment'], item['training_seed'])
        for item in collection_policies
    }
    snapshot_hashes = {path: file_sha256(path) for path in snapshot_paths}
    ensure_disjoint_snapshot_runs(
        collection_paths,
        snapshot_paths,
        collection_hashes=collection_hashes,
        probing_hashes=set(snapshot_hashes.values()),
    )

    result_version = new_version_id()
    output_path = versioned_path(output_base_path, result_version)
    result_context = {
        'dataset_version': dataset_version or 'unversioned',
        'result_version': result_version,
        'dataset_path': str(dataset_path.resolve()),
        'dataset_sha256': dataset_digest,
        'dataset_schema_version': dataset_metadata['schema_version'],
        'target_schema_version': dataset_metadata['target_schema_version'],
        'dataset_seed': dataset_metadata['seed'],
        'dataset_environment': dataset_environment,
        'probe_config_sha256': _json_sha256(config),
        'probe_code_sha256': _probe_code_sha256(),
        'split_seed': probing_config['seed'],
        'test_fraction': probing_config['test_fraction'],
        'actual_test_fraction': float(test_mask.mean()),
        'use_snapshot_normalizer': config['snapshots']['use_snapshot_normalizer'],
        'device': config['snapshots']['device'],
        'ridge_alpha': probing_config['ridge_alpha'],
        'logistic_c': probing_config['logistic_c'],
        'max_iterations': probing_config['max_iterations'],
        'embedding_batch_size': probing_config['embedding_batch_size'],
    }

    output_path.parent.mkdir(parents=True, exist_ok=False)
    partial_path = Path(f'{output_path}.partial')
    row_count = 0
    non_ok_statuses: dict[str, int] = {}
    with partial_path.open('w', newline='', encoding='utf-8') as output_file:
        writer = csv.DictWriter(output_file, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        for index, snapshot_path in enumerate(snapshot_paths, start=1):
            print(f'[{index}/{len(snapshot_paths)}] {snapshot_path}')
            rows = evaluate_snapshot(
                snapshot_path,
                cached_dataset,
                dataset_environment,
                config,
                train_mask,
                test_mask,
                result_context,
                snapshot_hashes[snapshot_path],
                collector_identities,
            )
            writer.writerows(rows)
            output_file.flush()
            row_count += len(rows)
            for row in rows:
                status = str(row['status'])
                if status != 'ok':
                    non_ok_statuses[status] = non_ok_statuses.get(status, 0) + 1
    partial_path.replace(output_path)
    print(f'Result version: {result_version}')
    print(f'Saved {row_count} probe results to {output_path}')
    if non_ok_statuses:
        print(f'WARNING: non-ok probe rows: {json.dumps(non_ok_statuses, sort_keys=True)}')


if __name__ == '__main__':
    main()
