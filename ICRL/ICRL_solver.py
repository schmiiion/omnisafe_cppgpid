"""Alternate native OmniSafe policy learning and Liu-style constraint fitting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from omnisafe.envs.core import make
from omnisafe.utils.tools import get_device, seed_all

from ICRL.constraint_models.constraint_net import ConstraintNet, VariationalConstraintNet
from ICRL.inner_loop import (
    close_learner,
    collect_trajectories,
    log_iteration,
    make_learner,
    mean_learned_episode_cost,
    reset_rollout_buffers,
)
from ICRL.utils.config import (
    DEFAULT_ICRL_CONFIG,
    DEFAULT_OMNISAFE_CONFIG,
    configure_observation_mode,
    load_icrl_config,
    load_omnisafe_config,
)
from ICRL.utils.data import load_expert_data
from ICRL.visualize_cost.plot_cost import WorldTrace, visualize_iteration
from ICRL.visualize_rollouts import record_live_iteration


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--icrl-config', type=Path, default=DEFAULT_ICRL_CONFIG)
    parser.add_argument('--omnisafe-config', type=Path, default=DEFAULT_OMNISAFE_CONFIG)
    args = parser.parse_args(argv)

    # 1. Task/outer settings from ICRL; inner settings from OmniSafe.
    outer = load_icrl_config(args.icrl_config)
    inner = load_omnisafe_config(args.omnisafe_config, outer.icrl.get('env_id'))
    observation_mode = outer.icrl.get('observation_mode', 'standard')
    configure_observation_mode(inner, observation_mode)
    visual_cfg = outer.get('visualize_cost')
    visualize = visual_cfg is not None and visual_cfg.get('enabled', False)
    rollout_cfg = outer.get('visualize_rollouts')
    record_rollouts = rollout_cfg is not None and rollout_cfg.get('enabled', False)
    device = get_device(inner.train_cfgs.device)
    torch.set_num_threads(inner.train_cfgs.torch_threads)
    seed_all(inner.seed)

    dataset = Path(outer.icrl.expert_dataset).expanduser()
    provenance = dataset.with_suffix('.json')
    expert_metadata = {}
    if provenance.exists():
        expert_metadata = json.loads(provenance.read_text())
        expert_task = expert_metadata.get(
            'dataset_env_id', expert_metadata.get('source_config', {}).get('env_id'),
        )
        if expert_task is not None and expert_task != inner.env_id:
            raise ValueError(f'Expert trajectories are from {expert_task}, but the inner task is {inner.env_id}.')
    if observation_mode == 'point_xy' and expert_metadata.get('observation_mode') != 'point_xy':
        raise ValueError(
            'point_xy requires expert trajectories exported from an XY-observing policy '
            'with observation_mode: point_xy in the adjacent JSON provenance file.',
        )
    expert_obs, expert_acs = load_expert_data(dataset)

    # Native learners consume complete rollout epochs; ICRL sets the phase budget.
    inner_epochs, remainder = divmod(outer.icrl.inner_steps, inner.algo_cfgs.steps_per_epoch)
    assert inner_epochs > 0 and remainder == 0, (
        'icrl.inner_steps must be a positive multiple of algo_cfgs.steps_per_epoch.'
    )
    inner.train_cfgs.epochs = outer.icrl.iterations * inner_epochs
    inner.train_cfgs.total_steps = outer.icrl.iterations * outer.icrl.inner_steps
    # Put native training files and outer-loop artifacts under one run root.
    runs_root = Path(outer.icrl.get('runs_dir', Path(__file__).resolve().parent / 'runs'))
    inner.logger_cfgs.log_dir = str(runs_root.expanduser().resolve())

    # 2. Environment and constraint model. This separate raw environment samples
    #    the updated policy; the inner solver constructs its own training environment.
    env = make(inner.env_id, num_envs=1, device=device, **(inner.get('env_cfgs') or {}))
    learner = None

    env.set_seed(inner.seed + 1)
    obs_dim, acs_dim = env.observation_space.shape[0], env.action_space.shape[0]
    if expert_obs.shape[1] != obs_dim or expert_acs.shape[1] != acs_dim:
        raise ValueError(
            f'Expert observation/action dimensions {expert_obs.shape[1], expert_acs.shape[1]} '
            f'do not match the environment {obs_dim, acs_dim}.',
        )
    constraint_cfg = outer.constraint
    if constraint_cfg.input_type not in ('state', 'state_action'):
        raise ValueError('constraint.input_type must be state or state_action.')
    model_class = {'standard': ConstraintNet,
                   'variational': VariationalConstraintNet}[constraint_cfg.get('model', 'standard')]
    variational_options = (
        dict(task='VICRL', di_prior=constraint_cfg.get('di_prior', [1, 1]),
             mode=constraint_cfg.get('mode', 'mean'),
             confidence=constraint_cfg.get('confidence', 0.5))
        if model_class is VariationalConstraintNet else {}
    )
    constraint = model_class(
        obs_dim=obs_dim, acs_dim=acs_dim,
        hidden_sizes=tuple(constraint_cfg.hidden_sizes),
        batch_size=constraint_cfg.batch_size,
        lr_schedule=lambda progress: constraint_cfg.learning_rate * (
            constraint_cfg.get('anneal_clr_by_factor', 1.0)
            ** (outer.icrl.iterations * (1 - progress))
        ),
        expert_obs=expert_obs, expert_acs=expert_acs, is_discrete=False,
        acs_select_dim=(-1,) if constraint_cfg.input_type == 'state' else None,
        regularizer_coeff=constraint_cfg.regularizer_coeff,
        no_importance_sampling=constraint_cfg.no_importance_sampling,
        per_step_importance_sampling=constraint_cfg.per_step_importance_sampling,
        target_kl_old_new=constraint_cfg.target_kl_old_new,
        target_kl_new_old=constraint_cfg.target_kl_new_old,
        eps=constraint_cfg.get('cn_eps', 1e-5),
        clip_obs=None, action_low=env.action_space.low, action_high=env.action_space.high,
        device=str(device),
        **variational_options,
    )
    learner = make_learner(inner, constraint)
    # OmniSafe chooses the experiment and seed/timestamp names; use that exact run.
    native_output = Path(learner.logger.log_dir)
    output = native_output / 'icrl'
    output.mkdir(parents=True, exist_ok=True)
    (output / 'icrl_config.yaml').write_text(yaml.safe_dump(outer.todict()), encoding='utf-8')
    (output / 'run_metadata.json').write_text(json.dumps({
        'environment': inner.env_id, 'algorithm': inner.algo, 'seed': inner.seed,
        'observation_mode': observation_mode,
        'constraint_model': constraint_cfg.get('model', 'standard'),
        'expert_dataset': str(dataset.resolve()),
        'icrl_config': str(args.icrl_config.expanduser().resolve()),
        'omnisafe_config': str(args.omnisafe_config.resolve()),
        'omnisafe_log_dir': str(native_output.resolve()),
        'visualize_cost': visual_cfg.todict() if visual_cfg is not None else {'enabled': False},
        'visualize_rollouts': rollout_cfg.todict() if rollout_cfg is not None else {'enabled': False},
    }, indent=2), encoding='utf-8')
    learner.logger.log(
        f'ICRL config: {args.icrl_config.expanduser().resolve()} '
        f'({outer.icrl.iterations} outer iterations, {outer.icrl.inner_steps} inner steps, '
        f'{constraint_cfg.backward_iters} constraint passes)'
    )
    sampling_steps = 0
    visualization_replay_steps = 0
    video_eval_steps = 0

    # OUTER LOOP: start a fresh training phase -> solve CRL -> sample -> fit.
    # Construct the learner only once; its actor, critics and optimizers
    # persist while the same constraint object is updated.
    for iteration in range(1, outer.icrl.iterations + 1):
        learner.logger.log(f'ICRL iteration {iteration}/{outer.icrl.iterations}')

        # 3. INNER LOOP: discard on-policy data; train for exactly the
        # configured number of epochs.
        reset_rollout_buffers(learner)
        learner.learn(epochs=inner_epochs, close=False)
        learner.logger.torch_save()

        # Score the saved policy under the constraint used to train it, before
        # the next constraint update changes this in-memory model.
        if record_rollouts:
            videos = record_live_iteration(
                learner, constraint, inner.env_id, inner.get('env_cfgs') or {},
                rollout_cfg.seeds, iteration, output / 'rollouts',
                view=rollout_cfg.view, frame_stride=rollout_cfg.frame_stride,
                fps=rollout_cfg.fps, size=rollout_cfg.size,
            )
            video_eval_steps += sum(video['steps'] for video in videos)

        # 4. Fresh trajectories from the updated policy in the separate raw
        # environment.
        trace = WorldTrace() if visualize else None
        obs, acs, lengths, metrics = collect_trajectories(
            learner, env, constraint, outer.icrl.sample_rollouts, recorder=trace,
        )
        sampling_steps += int(lengths.sum())

        # 5. Compute the imported ICRL loss and take its gradient steps.
        # The training environment holds this same model, so the new costs are
        # used automatically in the next inner phase. No true-cost labels enter.
        fit_metrics = constraint.train_nn(
            iterations=constraint_cfg.backward_iters,
            nominal_obs=obs, nominal_acs=acs, episode_lengths=lengths,
            current_progress_remaining=1 - (iteration - 1) / outer.icrl.iterations,
        )
        # Rescore the same trajectories after fitting: no new environment steps.
        # Their observed reward and true cost remain unchanged.
        metrics['ICRL/SampleLearnedCostAfter'] = mean_learned_episode_cost(
            constraint, obs, acs, lengths,
        )
        metrics['ICRL/ConstraintLearningRate'] = constraint.optimizer.param_groups[0]['lr']
        training_steps = learner.logger.current_epoch * inner.algo_cfgs.steps_per_epoch
        metrics.update({
            'ICRL/Iteration': iteration,
            'ICRL/PolicyEpoch': learner.logger.current_epoch,
            'ICRL/TrainingSteps': training_steps,
            'ICRL/SamplingSteps': sampling_steps,
            'ICRL/TotalInteractions': training_steps + sampling_steps,
            'ICRL/VideoEvalSteps': video_eval_steps,
        })
        metrics.update({f'ICRL/{key}': value for key, value in fit_metrics.items()})
        if visualize:
            visualization = visualize_iteration(
                constraint=constraint, env_id=inner.env_id,
                env_cfgs=inner.get('env_cfgs') or {},
                expert_dataset=dataset, expert_obs=expert_obs, expert_acs=expert_acs,
                learner_obs=obs, learner_acs=acs, learner_lengths=lengths,
                learner_trace=trace, iteration=iteration, output_dir=output,
                trajectories_per_source=visual_cfg.trajectories_per_source,
                step_stride=visual_cfg.step_stride, seed=inner.seed,
                device=device,
            )
            visualization_replay_steps += visualization['expert_replay_steps']
            metrics['ICRL/VisualizationReplaySteps'] = visualization_replay_steps
        metrics['ICRL/AllEnvironmentSteps'] = (
            training_steps + sampling_steps + visualization_replay_steps + video_eval_steps
        )
        log_iteration(learner, metrics, output)
    print(f'ICRL complete. Run: {native_output}')
    env.close()
    close_learner(learner)


if __name__ == '__main__':
    main()
