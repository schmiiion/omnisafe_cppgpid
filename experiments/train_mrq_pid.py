"""Run an MR.Q-PID experiment from a small YAML overlay."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

import omnisafe
from omnisafe.algorithms import ALGORITHM2TYPE
from omnisafe.utils.config import check_all_configs, get_default_kwargs_yaml
from omnisafe.utils.tools import recursive_check_config


DEFAULT_CONFIG = Path(__file__).with_name('mrq_pid_safety_car_goal1_5m.yaml')
REPO_ROOT = Path(__file__).resolve().parents[1]


def load_experiment(path: Path) -> tuple[str, str, dict[str, Any]]:
    """Load an overlay and resolve its output directory outside the project."""
    with path.open(encoding='utf-8') as file:
        settings = yaml.safe_load(file)
    if not isinstance(settings, dict):
        raise ValueError('Experiment config must be a YAML mapping')
    algo = settings.pop('algo')
    env_id = settings.pop('env_id')
    log_dir = Path(settings['logger_cfgs']['log_dir']).expanduser()
    if not log_dir.is_absolute():
        log_dir = REPO_ROOT / log_dir
    settings['logger_cfgs']['log_dir'] = str(log_dir.resolve())
    return algo, env_id, settings


def check_experiment(algo: str, env_id: str, settings: dict[str, Any]) -> None:
    """Validate the merged settings without allocating models or starting training."""
    algo_type = ALGORITHM2TYPE[algo]
    cfg = get_default_kwargs_yaml(algo, env_id, algo_type)
    recursive_check_config(settings, cfg)
    cfg.recurisve_update(settings)
    cfg.recurisve_update({'algo': algo, 'env_id': env_id})
    check_all_configs(cfg, algo_type)
    assert cfg.train_cfgs.total_steps % cfg.algo_cfgs.steps_per_epoch == 0
    assert cfg.algo_cfgs.steps_per_epoch % cfg.train_cfgs.vector_env_nums == 0
    assert cfg.algo_cfgs.start_learning_steps < cfg.algo_cfgs.size
    print(f'{algo} on {env_id}: {cfg.train_cfgs.total_steps:,} environment steps')
    print(f'Worker seeds: {[cfg.seed + i for i in range(cfg.train_cfgs.vector_env_nums)]}')
    print(f'Log directory: {cfg.logger_cfgs.log_dir}')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--check', action='store_true', help='Validate without training')
    args = parser.parse_args()
    algo, env_id, settings = load_experiment(args.config)
    if args.check:
        check_experiment(algo, env_id, settings)
        return
    omnisafe.Agent(algo, env_id, custom_cfgs=settings).learn()


if __name__ == '__main__':
    main()
