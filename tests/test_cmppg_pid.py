"""Focused checks for CMPPGPID's model windows and auxiliary gradients."""

from copy import deepcopy
from types import SimpleNamespace

import torch
from gymnasium.spaces import Box

from omnisafe.algorithms.on_policy.pid_lagrange.cmppg_pid import CMPPGPID
from omnisafe.common.buffer.vector_model_ppg_buffer import VectorModelPPGBuffer
from omnisafe.common.buffer.vector_onpolicy_buffer import VectorOnPolicyBuffer
from omnisafe.models.actor.gaussian_learning_model_ppg_actor import GaussianLearningModelPPGActor


def test_model_windows_stop_at_episode_and_rollout_boundaries() -> None:
    obs_space = Box(-1, 1, shape=(2,))
    act_space = Box(-1, 1, shape=(1,))
    policy_buffer = VectorOnPolicyBuffer(
        obs_space=obs_space,
        act_space=act_space,
        size=6,
        gamma=0.99,
        lam=0.95,
        lam_c=0.95,
        advantage_estimator='gae',
        penalty_coefficient=0.0,
        standardized_adv_r=False,
        standardized_adv_c=False,
        num_envs=2,
    )
    buffer = VectorModelPPGBuffer(
        policy_buffer=policy_buffer,
        obs_space=obs_space,
        act_space=act_space,
        rollout_size=6,
        N_pi=2,
        standardized_adv_r=False,
        standardized_adv_c=False,
        num_envs=2,
        model_horizon=2,
    )
    for phase in range(2):
        for step in range(6):
            reward = torch.tensor(
                [phase * 100 + step, phase * 100 + 10 + step],
                dtype=torch.float32,
            )
            buffer.store(
                obs=torch.tensor([[step, 0], [step, 1]], dtype=torch.float32),
                act=torch.zeros(2, 1),
                reward=reward,
                cost=reward + 1,
                value_r=torch.zeros(2),
                value_c=torch.zeros(2),
                logp=torch.zeros(2),
            )
            if step == 2:
                buffer.finish_path(idx=0)
            if step == 4:
                buffer.finish_path(idx=1)
        buffer.finish_path(idx=0)
        buffer.finish_path(idx=1)
        buffer.get()

    data = buffer.get_aux_data()
    assert not {'target_value_r', 'target_value_c'} & data.keys()
    assert 'mean_old' in data and 'std_old' in data
    assert 'critic_value_r' in data and 'critic_value_c' in data
    expected = [True, False, False, True, False, False, True, True, True, False, False, False]
    assert data['model_valid'].tolist() == expected * 2
    assert data['reward'][:6].tolist() == list(range(6))
    assert data['reward'][6:12].tolist() == list(range(10, 16))
    assert data['reward'][12:18].tolist() == list(range(100, 106))
    assert data['cost'][12:18].tolist() == list(range(101, 107))
    assert buffer._aux_buffer.ptr == 0


def test_action_conditioned_model_backpropagates_through_unroll() -> None:
    actor = GaussianLearningModelPPGActor(
        obs_space=Box(-1, 1, shape=(3,)),
        act_space=Box(-1, 1, shape=(2,)),
        hidden_sizes=[16, 16],
    )
    assert not hasattr(actor, 'aux_r_head')
    assert not hasattr(actor, 'aux_c_head')
    value_r, value_c = actor.predict_values(torch.randn(4, 3))
    assert value_r.shape == value_c.shape == (4,)
    latent = actor.shared_backbone(torch.randn(4, 3))
    for _ in range(3):
        latent, reward, cost = actor.predict_model(latent, torch.randn(4, 2))
        assert latent.shape == (4, 16)
        assert reward.shape == cost.shape == (4,)
    (
        latent.square().mean()
        + reward.square().mean()
        + cost.square().mean()
        + value_r.square().mean()
        + value_c.square().mean()
    ).backward()
    assert actor.shared_backbone[0].weight.grad is not None
    assert actor.next_latent_head.weight.grad is not None
    assert actor.reward_head.weight.grad is not None
    assert actor.cost_head.weight.grad is not None
    assert actor.value_reward_head.weight.grad is not None
    assert actor.value_cost_head.weight.grad is not None


def test_auxiliary_kl_uses_frozen_policy_and_adapts_multiplier() -> None:
    torch.manual_seed(0)
    actor = GaussianLearningModelPPGActor(
        obs_space=Box(-1, 1, shape=(3,)),
        act_space=Box(-1, 1, shape=(2,)),
        hidden_sizes=[16, 16],
    )
    obs = torch.randn(8, 3)
    data = {
        'obs': obs,
        'act': torch.zeros(8, 2),
        'reward': torch.zeros(8),
        'cost': torch.zeros(8),
        'model_valid': torch.zeros(8, dtype=torch.bool),
        'critic_value_r': torch.zeros(8),
        'critic_value_c': torch.zeros(8),
        'mean_old': torch.zeros(8, 2),
        'std_old': torch.zeros(8, 2),
    }
    cfg = SimpleNamespace(
        N_pi=1,
        number_aux_mb_per_Npi=1,
        aux_kl_target=1e-6,
        init_beta_clone=1.0,
        beta_lr=0.01,
        latent_loss_weight=0.0,
        reward_prediction_weight=0.0,
        cost_prediction_weight=0.0,
        reward_value_prediction_weight=0.0,
        cost_value_prediction_weight=0.0,
    )
    algo = CMPPGPID.__new__(CMPPGPID)
    algo._cfgs = SimpleNamespace(algo_cfgs=cfg)
    algo._device = torch.device('cpu')
    algo._buf = SimpleNamespace(get_aux_data=lambda: data)
    algo._actor_critic = SimpleNamespace(
        actor=actor,
        actor_optimizer=torch.optim.SGD(actor.parameters(), lr=0.05),
        reward_critic=lambda inputs: (torch.zeros(len(inputs), 1),),
        cost_critic=lambda inputs: (torch.zeros(len(inputs), 1),),
    )
    algo._target_backbone = deepcopy(actor.shared_backbone).requires_grad_(False)
    algo._aux_metrics = {}
    algo._init_aux_policy_clone()
    algo._compute_old_policy_snapshot()
    frozen_means = data['mean_old'].clone()

    with torch.no_grad():
        actor.shared_backbone[0].weight.add_(0.2)
    assert not torch.allclose(actor._distribution(obs).mean, frozen_means)
    weights_before = actor.shared_backbone[0].weight.detach().clone()

    algo._aux_update()

    assert algo._aux_metrics['Loss/AuxKL'][0] > cfg.aux_kl_target
    assert algo._aux_metrics['Metrics/BetaClone'][0] > cfg.init_beta_clone
    assert algo._aux_metrics['Loss/AuxWeightedKL'][0] > 0
    assert not torch.equal(actor.shared_backbone[0].weight, weights_before)
