"""Configuration ownership and task-specific inner-loop settings."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from omnisafe.envs.core import ENV_REGISTRY
from ICRL.utils.config import (
    DEFAULT_ICRL_CONFIG,
    DEFAULT_OMNISAFE_CONFIG,
    configure_observation_mode,
    load_icrl_config,
    load_omnisafe_config,
)


class InitializationTests(unittest.TestCase):
    """Verify metadata ownership and configuration resolution."""

    def setUp(self) -> None:
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        self.directory = Path(temporary_directory.name)
        self.inner_path = self.directory / 'PPOLag.yaml'
        self.outer_path = self.directory / 'icrl_config.yaml'
        self.inner = yaml.safe_load(DEFAULT_OMNISAFE_CONFIG.read_text(encoding='utf-8'))
        self.inner['defaults']['train_cfgs']['torch_threads'] = 1
        self.outer = yaml.safe_load(DEFAULT_ICRL_CONFIG.read_text(encoding='utf-8'))
        self.outer['icrl']['env_id'] = 'Simple-v0'
        self.outer['constraint']['hidden_sizes'] = [8, 4]
        self.write_configs()

    def write_configs(self) -> None:
        self.inner_path.write_text(yaml.safe_dump(self.inner), encoding='utf-8')
        self.outer_path.write_text(yaml.safe_dump(self.outer), encoding='utf-8')

    def test_outer_task_selects_inner_environment_overrides(self) -> None:
        self.inner['Simple-v0'] = {
            'seed': 42,
            'train_cfgs': {'total_steps': 80000},
            'algo_cfgs': {'gamma': 0.95},
        }
        self.write_configs()
        task = load_icrl_config(self.outer_path).icrl.env_id
        cfgs = load_omnisafe_config(self.inner_path, task)
        self.assertEqual(cfgs.seed, 42)
        self.assertEqual(cfgs.algo, 'PPOLag')
        self.assertEqual(cfgs.env_id, 'Simple-v0')
        self.assertNotIn('env_id', self.inner['defaults'])
        self.assertEqual(cfgs.algo_cfgs.gamma, 0.95)
        self.assertEqual(cfgs.train_cfgs.device, 'cpu')
        self.assertEqual(cfgs.train_cfgs.epochs, 4)
        self.assertNotIn('icrl', cfgs)

    def test_rejects_unknown_outer_task(self) -> None:
        with self.assertRaisesRegex(ValueError, 'icrl.env_id'):
            load_omnisafe_config(self.inner_path, 'UnknownICRLTask-v0')

    def test_rejects_shared_metadata_in_outer_config(self) -> None:
        self.outer['seed'] = 17
        self.write_configs()
        with self.assertRaisesRegex(ValueError, 'OmniSafe configuration'):
            load_icrl_config(self.outer_path)

    def test_rejects_unused_constraint_setting(self) -> None:
        self.outer['constraint']['update_steps'] = 100
        self.write_configs()
        with self.assertRaisesRegex(ValueError, 'Unsupported constraint settings.*update_steps'):
            load_icrl_config(self.outer_path)

    def test_rejects_unimplemented_oracle(self) -> None:
        self.outer['oracle']['enabled'] = True
        self.write_configs()
        with self.assertRaisesRegex(ValueError, 'oracle.enabled is not implemented'):
            load_icrl_config(self.outer_path)

    def test_point_xy_is_explicit_and_does_not_replace_native_environment(self) -> None:
        self.outer['icrl']['env_id'] = 'SafetyPointCircle1-v0'
        self.write_configs()
        native_class = ENV_REGISTRY.get_class('SafetyPointCircle1-v0', None)
        standard = load_omnisafe_config(self.inner_path, 'SafetyPointCircle1-v0')
        configure_observation_mode(standard, load_icrl_config(self.outer_path).icrl.observation_mode)
        self.assertNotIn('class_name', standard.env_cfgs)

        self.outer['icrl']['observation_mode'] = 'point_xy'
        self.write_configs()
        xy = load_omnisafe_config(self.inner_path, 'SafetyPointCircle1-v0')
        configure_observation_mode(xy, load_icrl_config(self.outer_path).icrl.observation_mode)
        self.assertEqual(xy.env_cfgs.class_name, 'ICRLPointCircleXYEnv')
        self.assertEqual(ENV_REGISTRY.get_class('SafetyPointCircle1-v0', None), native_class)
        self.assertTrue(xy.exp_name.endswith('-point_xy'))

    def test_point_xy_rejects_other_task(self) -> None:
        self.outer['icrl']['observation_mode'] = 'point_xy'
        self.write_configs()
        with self.assertRaisesRegex(ValueError, 'requires SafetyPointCircle1-v0'):
            load_icrl_config(self.outer_path)


if __name__ == '__main__':
    unittest.main()
