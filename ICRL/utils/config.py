"""Load and validate the separate ICRL and OmniSafe configurations."""

from __future__ import annotations

import math
from pathlib import Path

import yaml

from omnisafe.algorithms import ALGORITHM2TYPE
from omnisafe.envs.core import support_envs
from omnisafe.utils.config import Config, check_all_configs


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ICRL_CONFIG = REPOSITORY_ROOT / 'ICRL/icrl_config.yaml'
DEFAULT_OMNISAFE_CONFIG = REPOSITORY_ROOT / 'omnisafe/configs/on-policy/PPOLag.yaml'


def _read_config(path: str | Path) -> Config:
    """Read a YAML mapping without changing the working directory."""
    with Path(path).expanduser().open(encoding='utf-8') as config_file:
        data = yaml.safe_load(config_file)
    if not isinstance(data, dict):
        raise ValueError(f'Expected a YAML mapping in {path}.')
    return Config.dict2config(data)


def load_icrl_config(path: str | Path = DEFAULT_ICRL_CONFIG) -> Config:
    """Load the task and outer-loop settings from the ICRL YAML."""
    cfgs = _read_config(path)
    allowed = {
        'icrl': {'env_id', 'iterations', 'inner_steps', 'sample_rollouts',
                 'expert_dataset', 'runs_dir', 'observation_mode'},
        'constraint': {'model', 'di_prior', 'mode', 'confidence', 'input_type',
                       'hidden_sizes', 'learning_rate', 'backward_iters', 'batch_size',
                       'regularizer_coeff', 'no_importance_sampling',
                       'per_step_importance_sampling', 'target_kl_old_new',
                       'target_kl_new_old', 'cn_eps', 'anneal_clr_by_factor'},
        'oracle': {'enabled'},
        'visualize_cost': {'enabled', 'trajectories_per_source', 'step_stride'},
        'visualize_rollouts': {'enabled', 'seeds', 'view', 'frame_stride', 'fps', 'size'},
    }
    unexpected = set(cfgs) - set(allowed)
    if unexpected:
        raise ValueError(
            f'Unexpected ICRL sections: {sorted(unexpected)}. '
            'Keep the task under icrl.env_id; algorithm, seed, device and '
            'inner-loop settings belong in the OmniSafe configuration.',
        )
    for section in ('icrl', 'constraint'):
        if not isinstance(cfgs.get(section), Config):
            raise ValueError(f'ICRL configuration requires a {section!r} mapping.')
    for section, keys in allowed.items():
        if section in cfgs:
            unexpected = set(cfgs[section]) - keys
            if unexpected:
                raise ValueError(f'Unsupported {section} settings: {sorted(unexpected)}.')
    if 'backward_iters' not in cfgs.constraint:
        raise ValueError('Set constraint.backward_iters to the number of fitting passes.')
    if cfgs.get('oracle', {}).get('enabled', False):
        raise ValueError('oracle.enabled is not implemented by the ICRL solver.')
    observation_mode = cfgs.icrl.get('observation_mode', 'standard')
    if observation_mode not in ('standard', 'point_xy'):
        raise ValueError('icrl.observation_mode must be standard or point_xy.')
    if observation_mode == 'point_xy' and cfgs.icrl.env_id != 'SafetyPointCircle1-v0':
        raise ValueError('icrl.observation_mode: point_xy requires SafetyPointCircle1-v0.')
    learning_rate = cfgs.constraint.get('learning_rate')
    if (
        isinstance(learning_rate, bool)
        or not isinstance(learning_rate, (int, float))
        or not math.isfinite(learning_rate)
        or learning_rate <= 0
    ):
        raise ValueError('constraint.learning_rate must be a finite positive number.')
    return cfgs


def configure_observation_mode(cfgs: Config, observation_mode: str) -> None:
    """Select the ICRL-only XY environment before OmniSafe builds its networks."""
    if observation_mode == 'point_xy':
        # Importing registers only this dedicated CMDP. Explicit class_name below
        # selects it for this run; normal OmniSafe runs keep their native CMDP.
        import ICRL.point_circle_xy_env  # noqa: F401

        if cfgs.get('env_cfgs') is None:
            cfgs.env_cfgs = Config()
        cfgs.env_cfgs.class_name = 'ICRLPointCircleXYEnv'
        cfgs.exp_name += '-point_xy'


def load_omnisafe_config(path: str | Path, env_id: str) -> Config:
    """Resolve an OmniSafe algorithm YAML for the task selected by ICRL."""
    path = Path(path).expanduser()
    data = _read_config(path)
    if not isinstance(data.get('defaults'), Config):
        raise ValueError('The OmniSafe algorithm YAML must contain a defaults mapping.')
    algo = path.stem
    if algo not in ALGORITHM2TYPE:
        raise ValueError(f'Unknown OmniSafe algorithm {algo!r}; use its algorithm YAML filename.')
    # Small task-specific OmniSafe YAMLs can inherit the standard algorithm
    # defaults without copying the entire algorithm configuration.
    base = _read_config(REPOSITORY_ROOT / data.extends) if data.get('extends') else None
    cfgs = Config.dict2config(base.defaults.todict() if base else data.defaults.todict())
    if base:
        cfgs.recurisve_update(data.defaults)
    if not isinstance(env_id, str) or env_id not in support_envs():
        raise ValueError('Set icrl.env_id to a registered environment in the ICRL YAML.')
    if base and base.get(env_id):
        cfgs.recurisve_update(base[env_id])
    env_overrides = data.get(env_id, {})
    if not isinstance(env_overrides, dict):
        raise ValueError(f'The {env_id!r} environment overrides must be a mapping.')
    cfgs.recurisve_update(env_overrides)
    cfgs.update({'algo': algo, 'env_id': env_id, 'exp_name': f'{algo}-{{{env_id}}}'})
    if 'total_steps' in cfgs.train_cfgs and 'steps_per_epoch' in cfgs.algo_cfgs:
        cfgs.train_cfgs.epochs = cfgs.train_cfgs.total_steps // cfgs.algo_cfgs.steps_per_epoch
    check_all_configs(cfgs, ALGORITHM2TYPE[algo])
    if cfgs.train_cfgs.parallel != 1:
        raise ValueError('The ICRL loop currently requires train_cfgs.parallel: 1.')
    if type(cfgs.train_cfgs.vector_env_nums) is not int or cfgs.train_cfgs.vector_env_nums < 1:
        raise ValueError('train_cfgs.vector_env_nums must be a positive integer.')
    if type(cfgs.train_cfgs.torch_threads) is not int or cfgs.train_cfgs.torch_threads < 1:
        raise ValueError('train_cfgs.torch_threads must be a positive integer.')
    if type(cfgs.seed) is not int or not 0 <= cfgs.seed < 2**32:
        raise ValueError('seed must be an integer in [0, 2**32).')
    return cfgs
