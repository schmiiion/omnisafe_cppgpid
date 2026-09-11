"""Plot a continuous, cross-run average over all policy snapshots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use('Agg')

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

from linear_probing import aux_only_plotting as shared
from linear_probing.artifacts import (
    new_version_id,
    resolve_versioned_file,
    versioned_directory,
)
from linear_probing.config import DEFAULT_CONFIG_PATH, load_config, resolve_path
from linear_probing.dataset_io import file_sha256
from linear_probing.targets import TASK_TYPES


def flatten_policy_timeline(
    runs: list[str],
    phases: list[int],
    iterations: list[int],
    matrices: dict[str, np.ndarray],
) -> tuple[list[tuple[int, int]], dict[str, np.ndarray]]:
    """Flatten phase-major policy coordinates into one chronological snapshot axis."""
    coordinates = [(phase, iteration) for phase in phases for iteration in iterations]
    timeline = {}
    for task, values in matrices.items():
        expected_shape = (len(runs), len(phases), len(iterations))
        if values.shape != expected_shape:
            raise ValueError(
                f'Policy matrix for `{task}` has shape {values.shape}; expected {expected_shape}.',
            )
        timeline[task] = values.reshape(len(runs), len(coordinates))
    return coordinates, timeline


def _style_axis(
    axis: Any,
    snapshot_indices: np.ndarray,
    y_limits: tuple[float, float],
    task: str | None,
    tasks: list[str],
) -> None:
    baseline = shared._baseline(task, tasks)
    if y_limits[0] <= baseline <= y_limits[1]:
        axis.axhline(baseline, color='0.55', linestyle='--', linewidth=1)
    axis.set_xlim(snapshot_indices[0], snapshot_indices[-1])
    axis.set_ylim(*y_limits)
    axis.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
    axis.grid(alpha=0.25)
    axis.tick_params(axis='x', labelbottom=True)
    axis.set_xlabel('Policy snapshot')
    axis.set_ylabel(shared._y_label(task))


def save_policy_figure(
    output_path: Path,
    tasks: list[str],
    timeline: dict[str, np.ndarray],
    run_average: np.ndarray,
    y_limits: dict[str, tuple[float, float]],
    identity: dict[str, Any],
) -> None:
    """Save the task average and every task over the complete policy timeline."""
    figure, axes = shared._axes(len(tasks) + 1)
    snapshot_indices = np.arange(1, run_average.shape[1] + 1)
    series = [
        ('Average across tasks\nR² / balanced accuracy', None, run_average),
    ]
    series.extend((shared._task_title(task), task, timeline[task]) for task in tasks)
    for axis, (title, task, values) in zip(axes, series):
        mean, deviation = shared._aggregate(values, axis=0)
        axis.fill_between(
            snapshot_indices,
            mean - deviation,
            mean + deviation,
            color='#4C72B0',
            alpha=0.15,
        )
        axis.plot(snapshot_indices, mean, linewidth=1.8, color='#4C72B0')
        axis.set_title(title)
        panel = task or shared.AVERAGE_PANEL
        _style_axis(axis, snapshot_indices, y_limits[panel], task, tasks)
    for axis in axes[len(series) :]:
        axis.set_visible(False)
    figure.suptitle(
        f'Policy-snapshot performance — {identity["algorithm"]}, '
        f'mean ± SD over {identity["num_runs"]} runs',
        fontsize=15,
    )
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main() -> None:
    """Create one continuous plot over every policy snapshot."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(DEFAULT_CONFIG_PATH))
    args = parser.parse_args()
    config = load_config(args.config)
    plotting_config = config.get('policy_only_plotting')
    if not isinstance(plotting_config, dict):
        raise ValueError('The configuration requires a `policy_only_plotting` mapping.')
    results_version = plotting_config.get('results_version', 'latest')
    if not isinstance(results_version, str) or not results_version:
        raise ValueError('`policy_only_plotting.results_version` must be a non-empty string.')

    results_base_path = resolve_path(config['probing']['output_csv'], args.config)
    results_path, resolved_results_version = resolve_versioned_file(
        results_base_path,
        results_version,
    )
    runs, phases, iterations, tasks, matrices, identity = shared.load_stage_scores(
        results_path,
        stage='policy',
        include_initial=True,
    )
    if (
        resolved_results_version is not None
        and identity['result_version'] != resolved_results_version
    ):
        raise ValueError(
            f'Result folder version {resolved_results_version!r} does not match the CSV value '
            f'{identity["result_version"]!r}.',
        )

    coordinates, timeline = flatten_policy_timeline(runs, phases, iterations, matrices)
    task_values = np.stack([timeline[task] for task in tasks], axis=0)
    run_average = shared._finite_mean(task_values, axis=0)
    y_limits = shared._panel_y_limits(timeline, run_average)

    plot_version = new_version_id()
    output_base = resolve_path(plotting_config['output_directory'], args.config)
    output_directory = versioned_directory(output_base, plot_version)
    output_directory.mkdir(parents=True, exist_ok=False)
    output_path = output_directory / 'all-policy-snapshots.png'
    save_policy_figure(output_path, tasks, timeline, run_average, y_limits, identity)

    metadata = {
        'plot_version': plot_version,
        'source_results_version': resolved_results_version or 'unversioned',
        'source_results_path': str(results_path.resolve()),
        'source_results_sha256': file_sha256(results_path),
        'policy_snapshot_coordinates': [
            {'snapshot': index, 'phase': phase, 'policy_iteration': iteration}
            for index, (phase, iteration) in enumerate(coordinates, start=1)
        ],
        'tasks': tasks,
        'task_metrics': {
            task: shared.METRIC_BY_TYPE[TASK_TYPES[task]] for task in tasks
        },
        'aggregation': 'Mean and sample standard deviation across equivalent run checkpoints.',
        'panel_y_limits': {panel: list(bounds) for panel, bounds in y_limits.items()},
        'run_identity': identity,
    }
    (output_directory / 'plot_metadata.json').write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )
    print(f'Plot version: {plot_version}')
    print(f'Source results: {results_path}')
    print(f'Saved policy plot to {output_path}')


if __name__ == '__main__':
    main()
