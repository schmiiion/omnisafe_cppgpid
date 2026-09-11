"""Configuration helpers shared by the linear-probing scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG_PATH = Path(__file__).with_name('linear_probing_config.yaml')


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a probing YAML configuration."""
    with Path(path).expanduser().open(encoding='utf-8') as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict):
        raise ValueError(f'Expected a mapping in {path}.')
    return config


def task_switches(config: dict[str, Any]) -> dict[str, bool]:
    """Validate grouped task switches and return them in canonical order."""
    from linear_probing.targets import TASK_GROUPS, TASK_TYPES

    task_config = config.get('tasks')
    if not isinstance(task_config, dict):
        raise ValueError('`tasks` must be a mapping.')

    # Continue to accept the original flat format, but never silently omit a task.
    if set(task_config) == set(TASK_TYPES):
        configured = task_config
    else:
        if set(task_config) != set(TASK_GROUPS):
            raise ValueError(f'`tasks` groups must be exactly {list(TASK_GROUPS)}.')
        configured = {}
        for group, difficulties in TASK_GROUPS.items():
            group_config = task_config[group]
            if not isinstance(group_config, dict) or set(group_config) != set(difficulties):
                raise ValueError(
                    f'`tasks.{group}` difficulties must be exactly {list(difficulties)}.',
                )
            for difficulty, expected_tasks in difficulties.items():
                switches = group_config[difficulty]
                if not isinstance(switches, dict) or set(switches) != set(expected_tasks):
                    raise ValueError(
                        f'`tasks.{group}.{difficulty}` tasks must be exactly '
                        f'{list(expected_tasks)}.',
                    )
                configured.update(switches)

    invalid = [task for task, enabled in configured.items() if not isinstance(enabled, bool)]
    if invalid:
        raise ValueError(f'Task switches must be boolean: {invalid}.')
    return {task: configured[task] for task in TASK_TYPES}


def resolve_path(path: str | Path, config_path: str | Path) -> Path:
    """Resolve paths relative to the repository root, not the current shell directory."""
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate

    repository_root = Path(__file__).resolve().parent.parent
    return repository_root / candidate
