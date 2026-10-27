"""Compatibility regression tests for the imported Liu et al. constraint model."""

from __future__ import annotations

import io
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ICRL.constraint_models.constraint_net import ConstraintNet
from ICRL.constraint_models.constraint_net.constraint_net import idx2vector


def legacy_network(input_dim: int, hidden_sizes: tuple[int, ...]) -> nn.Sequential:
    """The original SB3 Linear/ReLU construction, independent of OmniSafe's builder."""
    layers = []
    for hidden_dim in hidden_sizes:
        layers.extend([nn.Linear(input_dim, hidden_dim), nn.ReLU()])
        input_dim = hidden_dim
    layers.extend([nn.Linear(input_dim, 1), nn.Sigmoid()])
    return nn.Sequential(*layers)


class ConstraintNetTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(123)
        self.expert_obs = rng.normal(size=(6, 3)).astype(np.float32)
        self.expert_acs = rng.uniform(-1, 1, size=(6, 2)).astype(np.float32)
        self.nominal_obs = rng.normal(size=(6, 3)).astype(np.float32)
        self.nominal_acs = rng.uniform(-1, 1, size=(6, 2)).astype(np.float32)

    def model(self, **overrides) -> ConstraintNet:
        kwargs = dict(
            obs_dim=3, acs_dim=2, hidden_sizes=(8, 4), batch_size=None,
            lr_schedule=lambda progress: 0.003 * progress,
            expert_obs=self.expert_obs, expert_acs=self.expert_acs,
            is_discrete=False, device='cpu', log_file=io.StringIO(),
        )
        kwargs.update(overrides)
        return ConstraintNet(**kwargs)

    def assert_parameters_equal(self, left: nn.Module, right: nn.Module) -> None:
        self.assertEqual(list(left.state_dict()), list(right.state_dict()))
        for name, value in left.state_dict().items():
            torch.testing.assert_close(value, right.state_dict()[name], rtol=0, atol=0)

    def test_architecture_seed_initialization_and_checkpoint_keys_match_sb3(self) -> None:
        for hidden_sizes in ((), (8,), (8, 4)):
            with self.subTest(hidden_sizes=hidden_sizes):
                torch.manual_seed(42)
                original = legacy_network(5, hidden_sizes)
                original_rng = torch.get_rng_state().clone()
                torch.manual_seed(42)
                ported = self.model(hidden_sizes=hidden_sizes)
                self.assert_parameters_equal(original, ported.network)
                self.assertTrue(torch.equal(original_rng, torch.get_rng_state()))
                self.assertEqual(
                    [type(layer) for layer in original],
                    [type(layer) for layer in ported.network],
                )

    def test_loss_gradient_and_adam_updates_match_original_objectives(self) -> None:
        expert = torch.from_numpy(np.concatenate([self.expert_obs, self.expert_acs], axis=1))
        nominal = torch.from_numpy(np.concatenate([self.nominal_obs, self.nominal_acs], axis=1))
        for binary in (False, True):
            with self.subTest(binary=binary):
                torch.manual_seed(4)
                original = legacy_network(5, (8, 4))
                optimizer = torch.optim.Adam(original.parameters(), lr=0.0015, eps=1e-5)
                for _ in range(2):
                    expert_pred, nominal_pred = original(expert), original(nominal)
                    if binary:
                        loss = nn.functional.binary_cross_entropy(expert_pred, torch.ones_like(expert_pred))
                        loss += nn.functional.binary_cross_entropy(nominal_pred, torch.zeros_like(nominal_pred))
                    else:
                        loss = -(expert_pred + 1e-5).log().mean()
                        loss += (nominal_pred + 1e-5).log().mean()
                        loss += 0.2 * ((1 - expert_pred).mean() + (1 - nominal_pred).mean())
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                torch.manual_seed(4)
                ported = self.model(regularizer_coeff=0.2, train_gail_lambda=binary)
                metrics = ported.train_nn(
                    2, self.nominal_obs, self.nominal_acs, np.array([3, 3]),
                    current_progress_remaining=0.5,
                )
                self.assertEqual(metrics['backward/cn_loss'], loss.item())
                self.assert_parameters_equal(original, ported.network)
                for expected, actual in zip(original.parameters(), ported.network.parameters()):
                    torch.testing.assert_close(expected.grad, actual.grad, rtol=0, atol=0)
                self.assertEqual(ported.optimizer.defaults['eps'], 1e-5)

    def test_learning_rate_updates_every_optimizer_group(self) -> None:
        model = self.model()
        parameters = list(model.parameters())
        model.optimizer = torch.optim.Adam([
            {'params': parameters[:2], 'lr': 0.5},
            {'params': parameters[2:], 'lr': 0.1},
        ])
        model._update_learning_rate(0.25)
        self.assertEqual([group['lr'] for group in model.optimizer.param_groups], [0.00075] * 2)
        self.assertEqual(model.current_progress_remaining, 0.25)

    def test_per_step_importance_weights_pair_with_individual_predictions(self) -> None:
        model = self.model(no_importance_sampling=False, per_step_importance_sampling=True)
        old = torch.tensor([[.25], [.5], [.75]])
        current = torch.tensor([[.5], [.25], [.75]])
        weights, _, _ = model.compute_is_weights(old, current, np.array([1, 2]))
        self.assertEqual(weights.shape, (3,))

        # Compute the intended weighted average explicitly, one transition at a time.
        ratios = [(float(new) + model.eps) / (float(previous) + model.eps)
                  for new, previous in zip(current[:, 0], old[:, 0])]
        expected_weights = np.asarray(ratios) / np.mean(ratios)
        np.testing.assert_allclose(weights.numpy(), expected_weights, rtol=1e-6)
        predictions = torch.tensor([[.2], [.5], [.8]], requires_grad=True)
        indices = np.array([2, 0, 1])
        actual = (weights[indices][..., None] * (predictions[indices] + model.eps).log()).mean()
        expected = sum(float(expected_weights[i]) * (predictions[i, 0] + model.eps).log()
                       for i in range(3)) / 3
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(torch.autograd.grad(actual, predictions)[0],
                                   torch.autograd.grad(expected, predictions)[0])
        self.assertGreater(abs(actual.item() - (predictions + model.eps).log().mean().item()), .1)

    def test_long_trajectory_importance_ratios_stay_finite(self) -> None:
        model = self.model(no_importance_sampling=False, per_step_importance_sampling=True)
        short_old = torch.tensor([[0.4], [0.5], [0.6], [0.7]])
        short_new = torch.tensor([[0.5], [0.4], [0.7], [0.6]])
        _, short_old_new, short_new_old = model.compute_is_weights(
            short_old, short_new, np.array([2, 2]),
        )
        ratios = ((short_new + model.eps) / (short_old + model.eps)).numpy().ravel()
        products = np.array([ratios[:2].prod(), ratios[2:].prod()])
        expected_old_new = -np.log(products).mean()
        expected_new_old = ((products / products.mean() - 1) * np.log(products)).mean()
        self.assertAlmostEqual(short_old_new.item(), expected_old_new, places=5)
        self.assertAlmostEqual(short_new_old.item(), expected_new_old, places=5)

        old = torch.full((500, 1), 0.5)
        for probability in (0.4, 0.6):
            current = torch.full_like(old, probability)
            weights, kl_old_new, kl_new_old = model.compute_is_weights(
                old, current, np.array([500]),
            )
            expected = -500 * math.log((probability + model.eps) / (0.5 + model.eps))
            self.assertTrue(torch.isfinite(weights).all())
            self.assertTrue(torch.isfinite(kl_old_new))
            self.assertTrue(torch.isfinite(kl_new_old))
            self.assertAlmostEqual(kl_old_new.item(), expected, delta=0.002)
            self.assertAlmostEqual(kl_new_old.item(), 0, delta=0.002)
            torch.testing.assert_close(weights, torch.ones(500), rtol=1e-5, atol=1e-5)

        current = torch.cat((torch.full_like(old, 0.4), torch.full_like(old, 0.6)))
        old_two = old.repeat(2, 1)
        model.per_step_importance_sampling = False
        weights, kl_old_new, kl_new_old = model.compute_is_weights(
            old_two, current, np.array([500, 500]),
        )
        self.assertTrue(torch.isfinite(weights).all())
        self.assertTrue(torch.isfinite(kl_old_new))
        self.assertTrue(torch.isfinite(kl_new_old))
        self.assertAlmostEqual(weights[:500].mean().item(), 0, places=5)
        self.assertAlmostEqual(weights[500:].mean().item(), 2, places=5)
        self.assertGreater(kl_new_old.item(), 0)

        current[0] = float('nan')
        with self.assertRaisesRegex(FloatingPointError, 'importance log ratios'):
            model.compute_is_weights(old_two, current, np.array([500, 500]))

    def test_normalization_clipping_and_cost_api_are_preserved(self) -> None:
        model = self.model(
            initial_obs_mean=np.array([1.0, 2.0, 3.0]),
            initial_obs_var=np.array([4.0, 1.0, 9.0]),
            clip_obs=0.5, action_low=-0.25, action_high=0.25,
        )
        expected = np.concatenate([
            np.clip((self.expert_obs - model.current_obs_mean) /
                    np.sqrt(model.current_obs_var + model.eps), -0.5, 0.5),
            np.clip(self.expert_acs, -0.25, 0.25),
        ], axis=1).astype(np.float32)
        prepared = model.prepare_data(self.expert_obs, self.expert_acs)
        np.testing.assert_array_equal(prepared.numpy(), expected)
        feasibility = model(prepared)
        self.assertEqual(feasibility.shape, (6, 1))
        self.assertTrue(feasibility.requires_grad)
        cost = model.cost_function(self.expert_obs, self.expert_acs)
        self.assertIsInstance(cost, np.ndarray)
        np.testing.assert_array_equal(cost, 1 - feasibility.detach().numpy().squeeze(-1))

    def test_grid_encoding_and_discrete_actions_are_preserved(self) -> None:
        coordinates = np.array([[0.5, 1.5], [1.0, 0.0]])
        encoded = idx2vector(coordinates, height=2, width=3)
        expected = np.array([[0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0]], dtype=np.float64)
        np.testing.assert_array_equal(encoded, expected)
        self.assertEqual(encoded.dtype, np.float64)
        self.assertEqual(idx2vector([], 2, 3).shape, (0,))
        model = self.model(
            obs_dim=6, acs_dim=3, is_discrete=True, recon_obs=True,
            env_configs={'map_height': 2, 'map_width': 3},
        )
        prepared = model.prepare_data(coordinates, np.array([[1], [2]]))
        np.testing.assert_array_equal(
            prepared.numpy(), np.concatenate([expected, np.eye(3)[[1, 2]]], axis=1),
        )

    def test_legacy_checkpoint_load_restore_optimizer_and_inference_resave(self) -> None:
        model = self.model(
            initial_obs_mean=np.zeros(3), initial_obs_var=np.ones(3),
            action_low=np.array([-1, -1]), action_high=np.array([1, 1]),
        )
        model.train_nn(1, self.nominal_obs, self.nominal_acs, np.array([6]),
                       obs_mean=np.zeros(3), obs_var=np.ones(3))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'constraint.pt'
            model.save(path)
            expected_cost = model.cost_function(self.nominal_obs, self.nominal_acs)
            for device in ('cpu', 'auto', None):
                with self.subTest(device=device):
                    restored = ConstraintNet.load(path, device=device)
                    self.assertIsNone(restored.optimizer)
                    np.testing.assert_allclose(
                        restored.cost_function(self.nominal_obs, self.nominal_acs), expected_cost,
                        rtol=1e-6, atol=1e-7,
                    )
            restored.save(Path(directory) / 'inference.pt')
            ConstraintNet.load(Path(directory) / 'inference.pt', device='cpu')

            resumed = self.model()
            resumed._load(path)
            self.assert_parameters_equal(model.network, resumed.network)
            self.assertEqual(model.optimizer.param_groups[0]['lr'], resumed.optimizer.param_groups[0]['lr'])
            self.assertEqual(len(model.optimizer.state), len(resumed.optimizer.state))
            # A further identical update also checks restoration of Adam's moments.
            for instance in (model, resumed):
                instance.train_nn(1, self.nominal_obs, self.nominal_acs, np.array([6]),
                                  obs_mean=np.zeros(3), obs_var=np.ones(3))
            self.assert_parameters_equal(model.network, resumed.network)

    def test_import_succeeds_without_sb3_or_original_repository_utilities(self) -> None:
        code = '''
import builtins
original_import = builtins.__import__
def checked_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level == 0 and name.split('.')[0] in {'stable_baselines3', 'utils', 'constraint_models'}:
        raise ImportError('Legacy dependency requested: ' + name)
    return original_import(name, globals, locals, fromlist, level)
builtins.__import__ = checked_import
from ICRL.constraint_models.constraint_net import ConstraintNet, GailDiscriminator, VariationalConstraintNet
from ICRL.constraint_models.constraint_net.gail_net import GailCallback
from ICRL.gail_callback import GailRolloutBuffer
'''
        result = subprocess.run(
            [sys.executable, '-c', code], cwd=Path(__file__).resolve().parents[2],
            capture_output=True, text=True, timeout=60, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
