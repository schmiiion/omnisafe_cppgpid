"""Reconstruct actors and preprocessing from representation snapshots."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import gymnasium
import numpy as np
import torch

from omnisafe.models.actor.actor_builder import ActorBuilder


STAGES = ('policy', 'auxiliary')


def snapshot_run_directory(path: Path) -> Path:
    """Return the training-run directory containing an actor snapshot."""
    path = path.expanduser().resolve()
    for parent in path.parents:
        if parent.name == 'representation_checkpoints':
            return parent.parent
    return path.parent


def ensure_disjoint_snapshot_runs(
    collection_paths: list[Path],
    probing_paths: list[Path],
    collection_hashes: set[str] | None = None,
    probing_hashes: set[str] | None = None,
) -> None:
    """Reject collection and probing snapshots from the same run or exact checkpoint."""
    collection_runs = {snapshot_run_directory(path) for path in collection_paths}
    probing_runs = {snapshot_run_directory(path) for path in probing_paths}
    overlap = sorted(collection_runs & probing_runs)
    if overlap:
        formatted = '\n'.join(f'- {path}' for path in overlap)
        raise ValueError(
            'Collection and probing snapshots must come from disjoint training runs. '
            f'Overlapping runs:\n{formatted}',
        )

    if collection_hashes is None:
        collection_hashes = {_snapshot_sha256(path) for path in collection_paths}
    if probing_hashes is None:
        probing_hashes = {_snapshot_sha256(path) for path in probing_paths}
    duplicate_hashes = collection_hashes & probing_hashes
    if duplicate_hashes:
        raise ValueError(
            'Collection and probing sets contain an identical actor snapshot '
            f'(SHA-256: {sorted(duplicate_hashes)[0]}).',
        )


def _snapshot_sha256(path: Path) -> str:
    """Hash a snapshot without loading it into memory."""
    digest = hashlib.sha256()
    with path.open('rb') as snapshot_file:
        for chunk in iter(lambda: snapshot_file.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot_coordinates(path: Path) -> tuple[int, str, int]:
    """Parse outer phase, training stage, and stage iteration from a snapshot path."""
    try:
        if not path.parent.name.startswith('phase-'):
            raise ValueError
        phase = int(path.parent.name[len('phase-') :])
        if path.name.startswith('policy-iteration-'):
            stage = 'policy'
            iteration = int(path.stem[len('policy-iteration-') :])
        elif path.name.startswith('aux-epoch-'):
            stage = 'auxiliary'
            iteration = int(path.stem[len('aux-epoch-') :])
        else:
            raise ValueError
    except ValueError as error:
        raise ValueError(f'Unrecognized actor snapshot path: {path}') from error
    return phase, stage, iteration


def _selection_values(value: Any, name: str) -> set[int] | str:
    """Normalize an ``all``/``latest``/integer-list selector."""
    if isinstance(value, str):
        if value not in {'all', 'latest'}:
            raise ValueError(f'`{name}` must be `all`, `latest`, or a list of integers.')
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return {value}
    if isinstance(value, list) and value and all(
        isinstance(item, int) and not isinstance(item, bool) for item in value
    ):
        return set(value)
    raise ValueError(f'`{name}` must be `all`, `latest`, or a list of integers.')


def select_snapshots(paths: list[Path], selection: dict[str, Any]) -> list[Path]:
    """Filter snapshots by phase, stage, and policy/auxiliary iteration selectors."""
    stages = selection.get('stages', list(STAGES))
    if isinstance(stages, str):
        stages = [stages]
    if not isinstance(stages, list) or not stages or not set(stages) <= set(STAGES):
        raise ValueError(f'`stages` must be a non-empty subset of {STAGES}.')

    all_records = [
        (path, snapshot_run_directory(path), *snapshot_coordinates(path)) for path in paths
    ]
    records = [record for record in all_records if record[3] in stages]

    phase_selection = _selection_values(selection.get('phases', 'all'), 'phases')
    if phase_selection == 'latest':
        latest_phase = {}
        for _, run, phase, _, _ in records:
            latest_phase[run] = max(phase, latest_phase.get(run, phase))
        records = [record for record in records if record[2] == latest_phase[record[1]]]
    elif phase_selection != 'all':
        missing = [
            (run, phase)
            for run in {record[1] for record in all_records}
            for phase in phase_selection
            if not any(record[1] == run and record[2] == phase for record in records)
        ]
        if missing:
            raise ValueError(f'Requested snapshot phases are missing: {missing}.')
        records = [record for record in records if record[2] in phase_selection]

    selected_records = []
    for stage, selector_name in (
        ('policy', 'policy_iterations'),
        ('auxiliary', 'auxiliary_epochs'),
    ):
        stage_records = [record for record in records if record[3] == stage]
        iteration_selection = _selection_values(
            selection.get(selector_name, 'all'),
            selector_name,
        )
        if iteration_selection == 'latest':
            latest_iteration = {}
            for _, run, phase, _, iteration in stage_records:
                key = run, phase
                latest_iteration[key] = max(iteration, latest_iteration.get(key, iteration))
            stage_records = [
                record
                for record in stage_records
                if record[4] == latest_iteration[(record[1], record[2])]
            ]
        elif iteration_selection != 'all':
            missing = [
                (run, phase, stage, iteration)
                for run, phase in {(record[1], record[2]) for record in records}
                for iteration in iteration_selection
                if not any(
                    record[1] == run
                    and record[2] == phase
                    and record[3] == stage
                    and record[4] == iteration
                    for record in stage_records
                )
            ]
            if stage in stages and missing:
                raise ValueError(f'Requested snapshot iterations are missing: {missing}.')
            stage_records = [
                record for record in stage_records if record[4] in iteration_selection
            ]
        selected_records.extend(stage_records)

    stage_order = {stage: index for index, stage in enumerate(STAGES)}
    selected_records.sort(
        key=lambda record: (str(record[1]), record[2], stage_order[record[3]], record[4]),
    )
    return [record[0] for record in selected_records]


def load_snapshot(path: Path, device: str) -> dict[str, Any]:
    """Load a trusted local actor snapshot across supported PyTorch versions."""
    try:
        snapshot = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        snapshot = torch.load(path, map_location=device)
    if not isinstance(snapshot, dict):
        raise ValueError(f'Actor snapshot must contain a mapping: {path}')
    metadata = snapshot.get('metadata')
    if not isinstance(metadata, dict):
        raise ValueError(f'Actor snapshot has invalid metadata: {path}')
    if metadata.get('snapshot_schema_version') != 1:
        raise ValueError(f'Unsupported actor snapshot schema: {path}')
    if path.parent.name.startswith('phase-'):
        phase, stage, iteration = snapshot_coordinates(path)
        embedded = (
            metadata.get('phase'),
            metadata.get('stage'),
            metadata.get('stage_iteration'),
        )
        if embedded != (phase, stage, iteration):
            raise ValueError(
                f'Snapshot path coordinates {(phase, stage, iteration)} do not match '
                f'embedded metadata {embedded}: {path}',
            )
    return snapshot


def build_actor(snapshot: dict[str, Any], device: str) -> torch.nn.Module:
    """Reconstruct the complete actor stored in a snapshot."""
    spec = snapshot.get('actor_spec')
    if not isinstance(spec, dict):
        raise ValueError('Snapshot has no `actor_spec`; create it with the current snapshot code.')

    observation_space = gymnasium.spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=tuple(spec['observation_shape']),
        dtype=np.float32,
    )
    action_space = gymnasium.spaces.Box(
        low=-1.0,
        high=1.0,
        shape=tuple(spec['action_shape']),
        dtype=np.float32,
    )
    actor = ActorBuilder(
        observation_space,
        action_space,
        list(spec['hidden_sizes']),
        activation=spec['activation'],
        weight_initialization_mode=spec['weight_initialization_mode'],
    ).build_actor(spec['actor_type'])
    class_path = f'{type(actor).__module__}.{type(actor).__qualname__}'
    if spec.get('class_path') != class_path:
        raise ValueError(
            f'Snapshot expects actor class {spec.get("class_path")}, reconstructed {class_path}.',
        )
    actor.load_state_dict(snapshot[spec['state_dict_key']])
    actor.to(device).eval()
    actor.requires_grad_(False)
    return actor


def read_only_normalize(
    observations: torch.Tensor,
    snapshot: dict[str, Any],
    enabled: bool = True,
) -> tuple[torch.Tensor, int | None]:
    """Apply saved observation statistics without updating them."""
    if not enabled:
        return observations, None

    spec = snapshot['actor_spec']
    key = spec.get('normalizer_state_dict_key')
    normalized_during_training = snapshot['metadata'].get('observation_normalized')
    if not isinstance(normalized_during_training, bool):
        raise ValueError('Snapshot has no valid observation-normalization flag.')
    if normalized_during_training is False:
        if key is not None:
            raise ValueError('Snapshot normalizer metadata and actor specification disagree.')
        return observations, None
    if not key or key not in snapshot:
        raise ValueError('Snapshot is missing its required observation-normalizer state.')

    state = snapshot[key]
    required = {'_count', '_mean', '_std', '_clip'}
    missing = sorted(required - set(state))
    if missing:
        raise ValueError(f'Observation-normalizer state is missing fields: {missing}.')
    count_tensor = torch.as_tensor(state['_count'])
    if count_tensor.numel() != 1:
        raise ValueError('Observation-normalizer count must be scalar.')
    count_value = count_tensor.item()
    try:
        count_float = float(count_value)
    except (TypeError, ValueError) as error:
        raise ValueError('Observation-normalizer count must be numeric.') from error
    if (
        isinstance(count_value, (bool, np.bool_))
        or not np.isfinite(count_float)
        or count_float < 0
        or not count_float.is_integer()
    ):
        raise ValueError('Observation-normalizer count must be a finite non-negative integer.')
    count = int(count_float)
    expected_shape = tuple(spec['observation_shape'])
    unbatched = tuple(observations.shape) == expected_shape
    batched = (
        observations.ndim == len(expected_shape) + 1
        and tuple(observations.shape[1:]) == expected_shape
    )
    if not (unbatched or batched):
        raise ValueError('Observation tensor shape does not match the actor observation shape.')
    mean = torch.as_tensor(
        state['_mean'],
        dtype=observations.dtype,
        device=observations.device,
    )
    std = torch.as_tensor(
        state['_std'],
        dtype=observations.dtype,
        device=observations.device,
    )
    clip = torch.as_tensor(
        state['_clip'],
        dtype=observations.dtype,
        device=observations.device,
    )
    if any(tuple(value.shape) != expected_shape for value in (mean, std, clip)):
        raise ValueError('Observation-normalizer shape does not match the actor observation shape.')
    if not all(torch.all(torch.isfinite(value)) for value in (mean, std, clip)):
        raise ValueError('Observation-normalizer statistics must be finite.')
    if not torch.all(clip > 0):
        raise ValueError('Observation-normalizer clip must be positive.')
    if not torch.all(std >= 0):
        raise ValueError('Observation-normalizer standard deviation cannot be negative.')
    if count <= 1:
        return observations, count
    if not torch.all(std > 0):
        raise ValueError('Observation-normalizer standard deviation must be positive.')
    return torch.clamp((observations - mean) / std, -clip, clip), count
