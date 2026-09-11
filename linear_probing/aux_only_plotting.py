"""Plot absolute linear-probe performance across auxiliary epochs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use('Agg')

import matplotlib.pyplot as plt  # noqa: E402

from linear_probing.artifacts import (
    new_version_id,
    resolve_versioned_file,
    versioned_directory,
)
from linear_probing.config import DEFAULT_CONFIG_PATH, load_config, resolve_path
from linear_probing.dataset_io import file_sha256
from linear_probing.targets import TASK_TYPES


METRIC_BY_TYPE = {
    'regression': 'r2',
    'circular_regression': 'r2',
    'classification': 'balanced_accuracy',
}
METRIC_LABELS = {'r2': 'R²', 'balanced_accuracy': 'Balanced accuracy'}
AVERAGE_PANEL = '__average__'
OVERALL_COLOR = '#C44E52'
THIRD_STYLES = (
    ('First third', '#9ECAE1'),
    ('Middle third', '#4C72B0'),
    ('Last third', '#756BB1'),
)


def _integer(row: dict[str, str], column: str) -> int:
    try:
        value = int(row[column])
    except (KeyError, TypeError, ValueError) as error:
        message = f'Invalid `{column}` value in probe results: {row.get(column)!r}.'
        raise ValueError(message) from error
    if value < 0:
        raise ValueError(f'`{column}` cannot be negative: {value}.')
    return value


def _score(row: dict[str, str]) -> float:
    task_type = row['task_type']
    if task_type not in METRIC_BY_TYPE:
        raise ValueError(f'Unsupported probe task type: {task_type!r}.')
    raw_value = row.get(METRIC_BY_TYPE[task_type], '')
    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        return float('nan')
    return value if np.isfinite(value) else float('nan')


def load_stage_scores(
    results_path: Path,
    stage: str,
    include_initial: bool,
) -> tuple[
    list[str],
    list[int],
    list[int],
    list[str],
    dict[str, np.ndarray],
    dict[str, Any],
]:
    """Load one stage into task-specific run-by-phase-by-iteration matrices."""
    if stage not in {'policy', 'auxiliary'}:
        raise ValueError(f'Unsupported training stage: {stage!r}.')
    with results_path.open(newline='', encoding='utf-8') as results_file:
        reader = csv.DictReader(results_file)
        required = {
            'algorithm',
            'environment',
            'training_seed',
            'run_directory',
            'result_version',
            'phase',
            'stage',
            'stage_iteration',
            'task',
            'task_type',
            'r2',
            'balanced_accuracy',
        }
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f'Probe-results CSV is missing columns: {sorted(missing)}.')
        rows = [
            row
            for row in reader
            if row['stage'] == stage
            and (include_initial or _integer(row, 'stage_iteration') > 0)
        ]
    if not rows:
        qualifier = '' if include_initial else 'post-update '
        raise ValueError(f'The probe-results CSV contains no {qualifier}{stage} snapshots.')

    identity_columns = ('algorithm', 'environment')
    identity = {}
    for column in identity_columns:
        values = {row[column] for row in rows}
        if len(values) != 1:
            raise ValueError(
                f'Stage plotting expects one experiment, but `{column}` has '
                f'{len(values)} values.',
            )
        identity[column] = values.pop()
    result_versions = {row['result_version'] for row in rows}
    if len(result_versions) != 1:
        raise ValueError(f'{stage.title()} rows contain more than one stored result version.')
    identity['result_version'] = result_versions.pop()

    run_seeds: dict[str, int] = {}
    for row in rows:
        run_directory = row['run_directory']
        if not run_directory:
            raise ValueError('Probe results contain an empty run directory.')
        seed = _integer(row, 'training_seed')
        previous_seed = run_seeds.setdefault(run_directory, seed)
        if previous_seed != seed:
            raise ValueError(f'Run {run_directory!r} occurs with multiple training seeds.')
    runs = sorted(run_seeds, key=lambda run: (run_seeds[run], run))
    seeds = [run_seeds[run] for run in runs]
    if len(set(seeds)) != len(seeds):
        raise ValueError('Each probing run must use a distinct training seed.')
    identity['run_directories'] = runs
    identity['training_seeds'] = seeds
    identity['num_runs'] = len(runs)

    observed_tasks = {row['task'] for row in rows}
    unknown_tasks = observed_tasks - set(TASK_TYPES)
    if unknown_tasks:
        raise ValueError(f'Unknown probing tasks in results: {sorted(unknown_tasks)}.')
    tasks = [task for task in TASK_TYPES if task in observed_tasks]
    phases = sorted({_integer(row, 'phase') for row in rows})
    iterations = sorted({_integer(row, 'stage_iteration') for row in rows})
    first_iteration = 0 if include_initial else 1
    if iterations != list(range(first_iteration, iterations[-1] + 1)):
        raise ValueError(
            f'{stage.title()} iteration indices must be contiguous from {first_iteration}, '
            f'found {iterations}.',
        )

    run_index = {run: index for index, run in enumerate(runs)}
    phase_index = {phase: index for index, phase in enumerate(phases)}
    iteration_index = {iteration: index for index, iteration in enumerate(iterations)}
    matrices = {
        task: np.full(
            (len(runs), len(phases), len(iterations)),
            np.nan,
            dtype=np.float64,
        )
        for task in tasks
    }
    seen = set()
    for row in rows:
        phase = _integer(row, 'phase')
        epoch = _integer(row, 'stage_iteration')
        task = row['task']
        run = row['run_directory']
        key = run, phase, epoch, task
        if key in seen:
            raise ValueError(f'Duplicate probe result for phase/epoch/task {key}.')
        seen.add(key)
        expected_type = TASK_TYPES[task]
        if row['task_type'] != expected_type:
            raise ValueError(
                f'Task `{task}` has type {row["task_type"]!r}; expected {expected_type!r}.',
            )
        matrices[task][run_index[run], phase_index[phase], iteration_index[epoch]] = _score(row)

    expected = {
        (run, phase, epoch, task)
        for run in runs
        for phase in phases
        for epoch in iterations
        for task in tasks
    }
    missing_results = expected - seen
    if missing_results:
        examples = sorted(missing_results)[:5]
        raise ValueError(
            f'Probe results are incomplete ({len(missing_results)} missing); examples: {examples}.',
        )
    return runs, phases, iterations, tasks, matrices, identity


def load_auxiliary_scores(
    results_path: Path,
) -> tuple[
    list[str],
    list[int],
    list[int],
    list[str],
    dict[str, np.ndarray],
    dict[str, Any],
]:
    """Load completed auxiliary epochs, excluding the pre-auxiliary snapshot."""
    return load_stage_scores(results_path, stage='auxiliary', include_initial=False)


def _finite_mean(values: np.ndarray, axis: int) -> np.ndarray:
    valid = np.isfinite(values)
    count = valid.sum(axis=axis)
    total = np.where(valid, values, 0.0).sum(axis=axis)
    return np.divide(
        total,
        count,
        out=np.full_like(total, np.nan, dtype=np.float64),
        where=count > 0,
    )


def _aggregate(values: np.ndarray, axis: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Return the finite mean and sample SD along an arbitrary axis."""
    valid = np.isfinite(values)
    count = valid.sum(axis=axis)
    mean = _finite_mean(values, axis=axis)
    centered = np.where(valid, values - np.expand_dims(mean, axis), 0.0)
    squared_error = np.square(centered).sum(axis=axis)
    denominator = np.where(count > 1, count - 1, count)
    variance = np.divide(
        squared_error,
        denominator,
        out=np.full_like(mean, np.nan, dtype=np.float64),
        where=denominator > 0,
    )
    deviation = np.sqrt(variance)
    return mean, deviation


def split_phase_thirds(phases: list[int]) -> list[list[int]]:
    """Split chronological phases into three non-empty, near-equal groups."""
    if len(phases) < 3:
        raise ValueError('At least three auxiliary phases are required for third-wise aggregation.')
    return [chunk.tolist() for chunk in np.array_split(np.asarray(phases), 3)]


def _panel_y_limits(
    matrices: dict[str, np.ndarray],
    phase_average: np.ndarray,
) -> dict[str, tuple[float, float]]:
    """Choose one zoomed scale per task and reuse it across all phase figures."""
    limits = {}
    panel_values = {AVERAGE_PANEL: phase_average, **matrices}
    for panel, values in panel_values.items():
        phase_mean, phase_deviation = _aggregate(values, axis=0)
        candidates = [
            values.reshape(-1),
            (phase_mean - phase_deviation).reshape(-1),
            (phase_mean + phase_deviation).reshape(-1),
        ]
        if values.ndim >= 3:
            per_run_mean = _finite_mean(values, axis=1)
            overall_mean, overall_deviation = _aggregate(per_run_mean, axis=0)
            candidates.extend((overall_mean - overall_deviation, overall_mean + overall_deviation))
        candidates = np.concatenate(candidates)
        finite = candidates[np.isfinite(candidates)]
        if not finite.size:
            limits[panel] = (0.0, 1.0)
            continue
        lower, upper = float(np.min(finite)), float(np.max(finite))
        padding = max((upper - lower) * 0.08, abs(lower + upper) * 0.01, 0.01)
        lower, upper = lower - padding, upper + padding
        if panel != AVERAGE_PANEL and TASK_TYPES[panel] == 'classification':
            lower, upper = max(0.0, lower), min(1.0, upper)
        limits[panel] = (lower, upper)
    return limits


def _task_title(task: str) -> str:
    metric = METRIC_BY_TYPE[TASK_TYPES[task]]
    return f'{task.replace("_", " ").title()}\n{METRIC_LABELS[metric]}'


def _baseline(task: str | None, tasks: list[str]) -> float:
    if task is not None:
        return 0.5 if TASK_TYPES[task] == 'classification' else 0.0
    return float(np.mean([_baseline(item, tasks) for item in tasks]))


def _axes(count: int) -> tuple[Any, np.ndarray]:
    columns = min(4, count)
    rows = math.ceil(count / columns)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(4.5 * columns, 3.4 * rows),
        sharex=True,
        constrained_layout=True,
    )
    return figure, np.asarray(axes, dtype=object).reshape(-1)


def _style_axis(
    axis: Any,
    epochs: list[int],
    y_limits: tuple[float, float],
    baseline: float,
    y_label: str,
) -> None:
    if y_limits[0] <= baseline <= y_limits[1]:
        axis.axhline(baseline, color='0.55', linestyle='--', linewidth=1)
    axis.set_xticks(epochs)
    axis.set_xlim(epochs[0] - 0.25, epochs[-1] + 0.25)
    axis.set_ylim(*y_limits)
    axis.grid(alpha=0.25)
    axis.tick_params(axis='x', labelbottom=True)
    axis.set_xlabel('Auxiliary epoch')
    axis.set_ylabel(y_label)


def _y_label(task: str | None) -> str:
    if task is None:
        return 'Mean performance\n(R² / balanced accuracy)'
    return METRIC_LABELS[METRIC_BY_TYPE[TASK_TYPES[task]]]


def save_phase_figure(
    output_path: Path,
    phase: int,
    phase_row: int,
    epochs: list[int],
    tasks: list[str],
    matrices: dict[str, np.ndarray],
    phase_average: np.ndarray,
    y_limits: dict[str, tuple[float, float]],
    identity: dict[str, Any],
) -> None:
    """Save one figure containing the average and every task for one outer phase."""
    figure, axes = _axes(len(tasks) + 1)
    series = [
        ('Average across tasks\nR² / balanced accuracy', None, phase_average[:, phase_row]),
    ]
    series.extend((_task_title(task), task, matrices[task][:, phase_row]) for task in tasks)
    for axis, (title, task, values) in zip(axes, series):
        mean, deviation = _aggregate(values, axis=0)
        axis.fill_between(
            epochs,
            mean - deviation,
            mean + deviation,
            color='#4C72B0',
            alpha=0.15,
        )
        axis.plot(epochs, mean, marker='o', linewidth=2, color='#4C72B0')
        axis.set_title(title)
        panel = task or AVERAGE_PANEL
        _style_axis(
            axis,
            epochs,
            y_limits[panel],
            _baseline(task, tasks),
            _y_label(task),
        )
    for axis in axes[len(series) :]:
        axis.set_visible(False)
    figure.suptitle(
        f'Auxiliary phase {phase + 1} (stored phase {phase:03d}) — '
        f'{identity["algorithm"]}, mean ± SD over {identity["num_runs"]} runs',
        fontsize=15,
    )
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def save_aggregate_figure(
    output_path: Path,
    phases: list[int],
    epochs: list[int],
    tasks: list[str],
    matrices: dict[str, np.ndarray],
    phase_average: np.ndarray,
    y_limits: dict[str, tuple[float, float]],
    identity: dict[str, Any],
) -> None:
    """Save overall and chronological-third means over all outer phases."""
    figure, axes = _axes(len(tasks) + 1)
    series = [('Average across tasks\nR² / balanced accuracy', None, phase_average)]
    series.extend((_task_title(task), task, matrices[task]) for task in tasks)
    phase_to_row = {phase: row for row, phase in enumerate(phases)}
    phase_thirds = split_phase_thirds(phases)
    for axis, (title, task, values) in zip(axes, series):
        for (label, color), group in zip(THIRD_STYLES, phase_thirds):
            rows = [phase_to_row[phase] for phase in group]
            per_run_group_mean = _finite_mean(values[:, rows], axis=1)
            group_mean, _ = _aggregate(per_run_group_mean, axis=0)
            axis.plot(
                epochs,
                group_mean,
                marker='o',
                linewidth=1.8,
                color=color,
                label=f'{label} (n={len(group)})',
            )
        per_run_mean = _finite_mean(values, axis=1)
        mean, deviation = _aggregate(per_run_mean, axis=0)
        axis.fill_between(
            epochs,
            mean - deviation,
            mean + deviation,
            color=OVERALL_COLOR,
            alpha=0.12,
        )
        axis.plot(
            epochs,
            mean,
            marker='o',
            linewidth=2.4,
            color=OVERALL_COLOR,
            label='All phases (mean ± SD)',
        )
        axis.set_title(title)
        panel = task or AVERAGE_PANEL
        _style_axis(
            axis,
            epochs,
            y_limits[panel],
            _baseline(task, tasks),
            _y_label(task),
        )
    unused_axes = axes[len(series) :]
    for axis in unused_axes:
        axis.set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    if unused_axes.size:
        legend_axis = unused_axes[0]
        legend_axis.set_visible(True)
        legend_axis.axis('off')
        legend_axis.legend(handles, labels, loc='center', frameon=False, fontsize=10)
    else:
        axes[0].legend(frameon=False, fontsize=8)
    figure.suptitle(
        f'Auxiliary performance over {len(phases)} phases — '
        f'{identity["algorithm"]}, {identity["num_runs"]} runs',
        fontsize=15,
    )
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main() -> None:
    """Create phase-specific and aggregate auxiliary-only probe figures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(DEFAULT_CONFIG_PATH))
    args = parser.parse_args()
    config = load_config(args.config)
    plotting_config = config.get('aux_only_plotting')
    if not isinstance(plotting_config, dict):
        raise ValueError('The configuration requires an `aux_only_plotting` mapping.')
    results_version = plotting_config.get('results_version', 'latest')
    if not isinstance(results_version, str) or not results_version:
        raise ValueError('`aux_only_plotting.results_version` must be a non-empty string.')

    results_base_path = resolve_path(config['probing']['output_csv'], args.config)
    results_path, resolved_results_version = resolve_versioned_file(
        results_base_path,
        results_version,
    )
    runs, phases, epochs, tasks, matrices, identity = load_auxiliary_scores(results_path)
    if (
        resolved_results_version is not None
        and identity['result_version'] != resolved_results_version
    ):
        raise ValueError(
            f'Result folder version {resolved_results_version!r} does not match the CSV value '
            f'{identity["result_version"]!r}.',
        )

    task_values = np.stack([matrices[task] for task in tasks], axis=0)
    phase_average = _finite_mean(task_values, axis=0)
    phase_thirds = split_phase_thirds(phases)
    y_limits = _panel_y_limits(matrices, phase_average)

    plot_version = new_version_id()
    output_base = resolve_path(plotting_config['output_directory'], args.config)
    output_directory = versioned_directory(output_base, plot_version)
    output_directory.mkdir(parents=True, exist_ok=False)
    for phase_row, phase in enumerate(phases):
        save_phase_figure(
            output_directory / f'phase-{phase:03d}.png',
            phase,
            phase_row,
            epochs,
            tasks,
            matrices,
            phase_average,
            y_limits,
            identity,
        )
    save_aggregate_figure(
        output_directory / 'aggregate-all-phases.png',
        phases,
        epochs,
        tasks,
        matrices,
        phase_average,
        y_limits,
        identity,
    )
    metadata = {
        'plot_version': plot_version,
        'source_results_version': resolved_results_version or 'unversioned',
        'source_results_path': str(results_path.resolve()),
        'source_results_sha256': file_sha256(results_path),
        'phases': phases,
        'auxiliary_epochs': epochs,
        'tasks': tasks,
        'task_metrics': {task: METRIC_BY_TYPE[TASK_TYPES[task]] for task in tasks},
        'average_definition': 'Unweighted mean of task R² and balanced accuracy values.',
        'aggregate_definition': (
            'Phases are averaged within each run, then the mean and standard deviation are '
            'computed across runs.'
        ),
        'phase_thirds': {
            label.lower().replace(' ', '_'): group
            for (label, _), group in zip(THIRD_STYLES, phase_thirds)
        },
        'phase_third_definition': 'Chronological near-equal thirds; lines show group means.',
        'panel_y_limits': {panel: list(bounds) for panel, bounds in y_limits.items()},
        'y_axis_policy': 'Dynamic per task; fixed for that task across all phase figures.',
        'run_identity': identity,
        'num_runs': len(runs),
    }
    (output_directory / 'plot_metadata.json').write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )
    print(f'Plot version: {plot_version}')
    print(f'Source results: {results_path}')
    print(f'Saved {len(phases) + 1} figures to {output_directory}')


if __name__ == '__main__':
    main()
