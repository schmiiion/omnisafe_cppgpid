"""Numerical and runtime checks for the GAIL and variational compatibility ports."""

import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import scipy.stats
import torch
from torch import nn

from ICRL.constraint_models.constraint_net import GailDiscriminator, VariationalConstraintNet
from ICRL.constraint_models.constraint_net.variational_constraint_net import (
    dirichlet_kl_divergence_loss,
)


def model(cls, **kwargs):
    rng = np.random.default_rng(12)
    options = dict(
        obs_dim=3, acs_dim=2, hidden_sizes=(8, 4), batch_size=None,
        lr_schedule=lambda progress: 0.003 * progress,
        expert_obs=rng.normal(size=(6, 3)).astype(np.float32),
        expert_acs=rng.normal(size=(6, 2)).astype(np.float32), is_discrete=False,
    )
    if cls is VariationalConstraintNet:
        options['log_file'] = io.StringIO()
    options.update(kwargs)
    return cls(**options)


def legacy_network(hidden_sizes, variational):
    layers, width = [], 5
    for hidden in hidden_sizes:
        layers.extend([nn.Linear(width, hidden), nn.ReLU()])
        width = hidden
    layers.extend([nn.Linear(width, 2 if variational else 1),
                   nn.Softplus() if variational else nn.Sigmoid()])
    return nn.Sequential(*layers)


class ImportedModelTests(unittest.TestCase):
    def assert_network_equal(self, expected, actual):
        self.assertEqual(list(expected.state_dict()), list(actual.state_dict()))
        for key, value in expected.state_dict().items():
            torch.testing.assert_close(value, actual.state_dict()[key], rtol=0, atol=0)

    def test_both_builders_preserve_architecture_weights_and_rng(self):
        for cls in (GailDiscriminator, VariationalConstraintNet):
            for hidden_sizes in ((), (8,), (8, 4)):
                with self.subTest(cls=cls, hidden_sizes=hidden_sizes):
                    torch.manual_seed(19)
                    original = legacy_network(hidden_sizes, cls is VariationalConstraintNet)
                    rng = torch.get_rng_state().clone()
                    torch.manual_seed(19)
                    ported = model(cls, hidden_sizes=hidden_sizes)
                    self.assert_network_equal(original, ported.network)
                    torch.testing.assert_close(rng, torch.get_rng_state(), rtol=0, atol=0)
                    self.assertEqual([type(layer) for layer in original],
                                     [type(layer) for layer in ported.network])

    def test_gail_bce_gradients_and_adam_updates(self):
        ported = model(GailDiscriminator)
        original = legacy_network((8, 4), False)
        original.load_state_dict(ported.network.state_dict())
        optimizer = torch.optim.Adam(original.parameters(), lr=0.0015, eps=1e-5)
        obs, acs = -ported.expert_obs, -ported.expert_acs
        # Preserve the source's feature selection, including its action indexing.
        expert = ported.prepare_expert_data(ported.expert_obs, ported.expert_acs)
        nominal, _ = ported.prepare_nominal_data(obs, acs)
        np.random.seed(3)
        for _ in range(2):
            indices = np.random.permutation(6)
            pn, pe = original(nominal[indices]), original(expert[indices])
            loss = nn.functional.binary_cross_entropy(pn, torch.zeros_like(pn))
            loss += nn.functional.binary_cross_entropy(pe, torch.ones_like(pe))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        np.random.seed(3)
        metrics = ported.train_nn(2, obs, acs, current_progress_remaining=0.5)
        self.assertEqual(metrics['discriminator/disc_loss'], loss.item())
        self.assert_network_equal(original, ported.network)
        for expected, actual in zip(original.parameters(), ported.parameters()):
            torch.testing.assert_close(expected.grad, actual.grad, rtol=0, atol=0)

    def test_gail_mode_switching_legacy_fit_freezing_and_reward(self):
        ported = model(GailDiscriminator, freeze_weights=True, num_spurious_features=2)
        self.assertIs(ported.eval(), ported)
        self.assertFalse(ported.training)
        self.assertIs(ported.train(), ported)
        self.assertTrue(ported.network.training)
        before = {k: v.clone() for k, v in ported.network.state_dict().items()}
        for fit in (
            lambda: ported.train(1, ported.expert_obs, ported.expert_acs),
            lambda: ported.train(iterations=1, nominal_obs=ported.expert_obs,
                                 nominal_acs=ported.expert_acs),
        ):
            self.assertIn('discriminator/disc_loss', fit())
        for key, value in before.items():
            torch.testing.assert_close(value, ported.network.state_dict()[key], rtol=0, atol=0)
        probability = ported.reward_function(ported.expert_obs, ported.expert_acs, apply_log=False)
        np.testing.assert_array_equal(
            ported.reward_function(ported.expert_obs, ported.expert_acs),
            np.log(probability + ported.eps),
        )
        np.testing.assert_array_equal(
            ported.cost_function(ported.expert_obs, ported.expert_acs), 1 - probability[:, 0],
        )

    def test_dirichlet_kl_values_and_gradients_match_pytorch_distribution(self):
        alpha = torch.tensor([[0.2, 1.1], [2.3, 3.7]], dtype=torch.float64, requires_grad=True)
        prior = torch.tensor([[1., 1.], [1.5, 0.7]], dtype=torch.float64)
        actual = dirichlet_kl_divergence_loss(alpha, prior)
        expected = torch.distributions.kl_divergence(
            torch.distributions.Dirichlet(alpha), torch.distributions.Dirichlet(prior),
        )
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(
            torch.autograd.grad(actual.sum(), alpha)[0],
            torch.autograd.grad(expected.sum(), alpha)[0], rtol=1e-12, atol=1e-12,
        )

    def test_variational_bce_fit_has_finite_metrics_and_updates_weights(self):
        ported = model(VariationalConstraintNet, train_gail_lambda=True)
        before = [p.detach().clone() for p in ported.parameters()]
        metrics = ported.train_nn(2, -ported.expert_obs, -ported.expert_acs, np.array([3, 3]))
        self.assertEqual(metrics['backward/regularizer_loss'], 0.)
        self.assertTrue(np.isfinite(metrics['backward/cn_loss']))
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, ported.parameters())))

    def test_variational_kl_fit_and_nonfinite_importance_guard(self):
        torch.manual_seed(5)
        ported = model(
            VariationalConstraintNet, batch_size=3, regularizer_coeff=0.5,
            no_importance_sampling=False, per_step_importance_sampling=True,
        )
        nominal_obs, nominal_acs = -ported.expert_obs, -ported.expert_acs
        before = [parameter.detach().clone() for parameter in ported.parameters()]
        metrics = ported.train_nn(2, nominal_obs, nominal_acs, np.array([3, 3]))
        self.assertTrue(np.isfinite(metrics['backward/cn_loss']))
        self.assertTrue(np.isfinite(metrics['backward/kl_old_new']))
        self.assertTrue(np.isfinite(metrics['backward/kl_new_old']))
        self.assertGreater(metrics['backward/regularizer_loss'], 0)
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, ported.parameters())))
        self.assertTrue(all(torch.isfinite(parameter).all() for parameter in ported.parameters()))

        before = [parameter.detach().clone() for parameter in ported.parameters()]
        invalid = (torch.ones(6), torch.tensor(float('nan')), torch.tensor(0.))
        with patch.object(ported, 'compute_is_weights', return_value=invalid):
            with self.assertRaisesRegex(FloatingPointError, 'importance diagnostics'):
                ported.train_nn(2, nominal_obs, nominal_acs, np.array([3, 3]))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(before, ported.parameters())))

    def test_variational_nonfinite_gradient_cannot_update_parameters(self):
        ported = model(VariationalConstraintNet, regularizer_coeff=0.5)
        before = [parameter.detach().clone() for parameter in ported.parameters()]
        parameter = next(ported.parameters())
        handle = parameter.register_hook(lambda grad: torch.full_like(grad, float('nan')))
        try:
            with self.assertRaisesRegex(FloatingPointError, 'gradient'):
                ported.train_nn(1, -ported.expert_obs, -ported.expert_acs, np.array([3, 3]))
        finally:
            handle.remove()
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(before, ported.parameters())))

    def test_variational_risk_modes_and_reparameterized_gradient(self):
        ported = model(VariationalConstraintNet, confidence=0.9)
        obs, acs = ported.expert_obs, ported.expert_acs
        inputs = ported.prepare_data(obs, acs)
        ab = ported.network(inputs).detach().numpy()
        a, b = ab[:, 0], ab[:, 1]
        np.testing.assert_array_equal(ported.cost_function(obs, acs, 'mean'), 1 - a / (a + b))
        np.testing.assert_array_equal(
            ported.cost_function(obs, acs, 'VaR'),
            1 - scipy.stats.beta.ppf(1 - ported.confidence, a, b),
        )
        np.random.seed(34)
        quantiles = scipy.stats.beta.ppf(1 - ported.confidence, a, b)
        expected_cvar = []
        for ai, bi, quantile in zip(a, b, quantiles):
            samples = scipy.stats.beta.rvs(ai, bi, size=[1000])
            expected_cvar.append(1 - samples[samples < quantile].mean())
        np.random.seed(34)
        np.testing.assert_array_equal(ported.cost_function(obs, acs, 'CVaR'), expected_cvar)
        torch.manual_seed(2)
        expected = ported(inputs)
        expected.sum().backward()
        self.assertGreater(sum(p.grad.abs().sum().item() for p in ported.parameters()), 0)
        torch.manual_seed(2)
        np.testing.assert_array_equal(ported.cost_function(obs, acs, 'sample'),
                                      1 - expected.detach().numpy()[:, 0])
        torch.manual_seed(2)
        np.testing.assert_array_equal(ported.cost_function(obs, acs, 'hard'),
                                      1 - expected.detach().numpy()[:, 0].round())

    def test_checkpoint_roundtrips_and_optimizer_resume(self):
        for cls, options in (
            (GailDiscriminator, {'num_spurious_features': 2, 'eps': 2e-5}),
            (VariationalConstraintNet, {'di_prior': [2., 3.], 'mode': 'VaR',
                                       'confidence': 0.9, 'eps': 2e-5}),
        ):
            with self.subTest(cls=cls), tempfile.TemporaryDirectory() as directory:
                ported = model(cls, **options)
                obs, acs = ported.expert_obs, ported.expert_acs
                def fit(instance):
                    torch.manual_seed(73)
                    np.random.seed(31)
                    if cls is GailDiscriminator:
                        instance.train_nn(1, obs, acs)
                    else:
                        instance.train_nn(1, obs, acs, np.array([6]))
                fit(ported)
                path = Path(directory) / 'model.pt'
                ported.save(path)
                restored = cls.load(path, device='cpu')
                self.assertIsNone(restored.optimizer)
                self.assert_network_equal(ported.network, restored.network)
                np.testing.assert_array_equal(ported.cost_function(obs, acs),
                                              restored.cost_function(obs, acs))
                if cls is VariationalConstraintNet:
                    self.assertEqual(restored.dir_prior, [2., 3.])
                    self.assertEqual(restored.mode, 'VaR')
                    self.assertEqual(restored.confidence, 0.9)
                restored.save(Path(directory) / 'inference.pt')
                cls.load(Path(directory) / 'inference.pt', device='auto')
                resumed = model(cls, **options)
                resumed._load(path)
                fit(ported)
                fit(resumed)
                self.assert_network_equal(ported.network, resumed.network)


if __name__ == '__main__':
    unittest.main()
