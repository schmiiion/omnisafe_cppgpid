# ICRL outer loop

From the repository root, run the configured Point Circle expert dataset with
the selected constraint model and standard CPPOPID inner hyperparameters:

```bash
conda activate omnisafe
python -u -m ICRL.ICRL_solver --icrl-config ICRL/icrl_config.yaml \
  --omnisafe-config omnisafe/configs/on-policy/CPPOPID.yaml
```

`ICRL_solver.py:main()` follows the sequence directly:

1. Load the two configurations and expert observations/actions.
2. Initialize the sampling environment, selected constraint model, and native OmniSafe learner.
3. Reset the training phase, train the same inner learner for `inner_steps` transitions,
   save its policy, and optionally film it on held-out seeds.
4. Collect `sample_rollouts` fresh, complete episodes with the updated policy.
5. Fit the constraint model against these trajectories and the fixed expert data;
   optionally plot its predicted costs on sampled expert/learner paths.
6. Repeat for `iterations`, retaining weights, optimizers, schedulers, normalizers and multiplier.

The cost wrapper holds the same model object, so an outer update immediately changes
which costs the next inner phase receives. By default, reward and true-cost
metrics come from the constraint-fitting rollouts; enabling rollout videos adds
separately counted diagnostic interactions.
The stopping rule is a fixed iteration count, not convergence detection.

## Configuration and data

- The native OmniSafe YAML owns the seed, device, vector environment count,
  simulator options, logger settings, cost budget and inner algorithm settings. Its
  filename selects the algorithm. Environment-specific blocks for the task
  selected in the ICRL YAML override its defaults. Use `--omnisafe-config PATH`
  to select it. For ICRL runs, `icrl.runs_dir` overrides the native logger root.
- `ICRL/icrl_config.yaml` selects the environment with `icrl.env_id` and owns
  outer iteration counts, the inner-step budget, sampling, expert data,
  constraint fitting, the shared run root, and optional cost/rollout visualization.
  Use `--icrl-config PATH` to select it. Each run saves the effective YAML and
  records the source path in `icrl/run_metadata.json`; unsupported settings fail at startup.
- The actual training horizon is `iterations * inner_steps` environment transitions,
  summed across vector environments. This overrides native `total_steps` for this
  entry point. `inner_steps` must be a positive multiple of the native
  `algo_cfgs.steps_per_epoch`, so the inner learner always consumes complete training
  epochs and the requested budget is exact. Extra sampling interactions are counted
  separately. The default config trains for 100,000 transitions per outer iteration
  (five epochs of 20,000), repeated 75 times, and samples 50 episodes per iteration.
- The current expert path selects 50 Point Circle demonstrations from CPPOPID's
  epoch-500 snapshot. Other studies must supply demonstrations from the same
  environment/observation/action convention as their inner solver. Matching dimensions
  alone do not establish compatibility. When the export's adjacent JSON exists, the
  solver checks its source environment ID. Relative dataset paths resolve against
  the working directory.
- `icrl.observation_mode: standard` uses the native 28-value PointCircle1
  observation. `point_xy` explicitly selects the ICRL-only environment that
  appends world `(x, y)` at indices 28 and 29. The inner policy, its critics,
  fresh learner trajectories, and the constraint network all receive the same
  30-value observation. The solver requires an XY expert export with matching
  provenance; an old 28-value checkpoint or dataset cannot serve as an XY expert.
- Supported data: an NPZ with `observations` (or `original_observations`) and
  `actions`, shaped `[transitions, features]` or `[episodes, steps, features]`; or a
  directory of trusted benchmark `.pkl`/`.pickle` episode dictionaries with those
  keys. These are pre-action observations and native actions. The loader returns
  two concatenated NumPy arrays; it does not require rewards or safety labels.

Configuration helpers live in `ICRL/utils/config.py`; data loading lives in
`ICRL/utils/data.py`. There is no Hydra dependency or extra dataset class.

### PointCircle1 expert with explicit position

The XY experiment has its own forward CRL entry point and small CPO and CPPOPID
configs. Set their training steps and checkpoint frequencies there, then run
from the repository root:

```bash
python -m ICRL.expert.train_point_xy_expert --config ICRL/expert/configs/CPO.yaml
python -m ICRL.expert.train_point_xy_expert --config ICRL/expert/configs/CPPOPID.yaml
```

These use native OmniSafe learning and save ordinary checkpoints under
`ICRL/expert/expert_weights/<algorithm>-PointXY-{SafetyPointCircle1-v0}/...`.
The cost budget is `algo_cfgs.cost_limit` for CPO and
`lagrange_cfgs.cost_limit` for CPPOPID; both dedicated configs set it to 2.
This entry point explicitly requests the dedicated observation environment
through `env_cfgs.class_name`; native OmniSafe commands and configs continue
to select the usual 28-value environment. The world coordinates are privileged
features supplied to the actor and critics, not true-cost labels.

After training, set `WEIGHTS_PATH` at the top of
`ICRL/expert/weights2trajectories.py` to the chosen XY run or checkpoint and
export the desired number of zero-cost episodes. Its NPZ contains 30-value
pre-action observations, and its adjacent JSON records `observation_mode:
point_xy`. Set `icrl.expert_dataset` to that NPZ and
`icrl.observation_mode: point_xy` in the ICRL YAML before starting the outer
loop. Its constraint model then sees the appended position directly; it still
learns feasibility from expert and learner behavior without ground-truth cost
targets. Compare with a `standard` run as a privileged-observation ablation.

## Generate expert data from saved weights

Edit the first two lines of `ICRL/expert/weights2trajectories.py`: the absolute
checkpoint path (`WEIGHTS_PATH`) and the accepted episode count (`NUM_TRAJECTORIES`).
The defaults select the supplied Circle checkpoint at epoch 500 and 50 zero-cost episodes.

```bash
conda activate omnisafe
python -m ICRL.expert.weights2trajectories
```

The script loads the run's `config.json` from the parent of `torch_save/`, builds
the actor and environment from that metadata, and restores actor weights and saved
observation-normalization statistics. It uses deterministic actions and freezes
those statistics. For a model-based SafeLOOP checkpoint, it uses the saved actor
alone: that checkpoint lacks the dynamics scaler needed to restore the original
planner. It keeps the source action repeat, records every physical transition in
raw observations, and marks the export `actor-only`. Attempts use the saved seed
plus the attempt index. One
environment collects complete episodes, irrespective of the training vector count.
Only episodes whose **ground-truth cost is exactly zero at every step** are kept.
An episode with any nonzero or nonfinite cost is discarded in full, including its
safe prefix. No reward filter is applied. Sampling continues with fresh seeds until
`NUM_TRAJECTORIES` accepted episodes have been collected. The default limit is
`10 * NUM_TRAJECTORIES` attempts, or `25 * NUM_TRAJECTORIES` for an actor-only
SafeLOOP export; `sample_trajectories(..., max_attempts=...)` can override it.
Reaching the limit without enough accepted episodes raises an error
and leaves any existing export untouched; no partial dataset is written.

Exports go into `ICRL/expert/expert_trajectories/`, beside this script. Each snapshot
gets a folder named `<algorithm>-<env_id>__<run-directory-name>__<checkpoint-stem>`,
so the algorithm, task, seed/run timestamp and checkpoint are visible in the path.
For the supplied weights, this is:

```text
expert_trajectories/
  CPPOPID-SafetyPointCircle0-v0__seed-000-2026-10-21-15-01-29__epoch-500/
    epoch-500-50-episodes.npz
    epoch-500-50-episodes.json
```

The JSON records the source checkpoint, metadata, filtering rule and attempted/rejected
episode counts. The NPZ contains raw pre-action
`observations`, bounded native `actions`,
`episode_lengths`, per-step rewards/true costs, episode returns/costs and seeds.
`load_expert_data()` reads the observations/actions directly; true costs are
diagnostics, not constraint-fitting labels. Ground-truth costs are used to select
zero-cost expert episodes at export time, not as targets in the ICRL loss.
Rerunning the same checkpoint/count overwrites that generated dataset.

Set `icrl.expert_dataset` to the printed NPZ path and `icrl.env_id` to the same
environment in the ICRL YAML. The solver's fallback when no `--omnisafe-config`
is supplied is PPOLag on the task selected by `icrl.env_id`.

## Native training and constraint fitting

`ICRL/inner_loop.py` contains the cost wrapper and small trajectory collector.
For on-policy algorithms, `PolicyGradient.learn(epochs=..., close=False)` runs more
epochs on the same learner and leaves its resources open. Ordinary `learn()` calls
retain their normal train-and-close behavior. Native update rules are used.

For on-policy learners, `reset_rollout_buffers()` clears the trajectory tensors,
buffer pointers and episode-cost logging window. Native `learn()` resets the
environment and episode accumulators at the start of every rollout epoch. It does
not recreate the environment, reseed it or reset enabled running normalizers.
Sampling likewise starts each episode with a reset.

For those learners, the actor, reward critic, cost critic, all their optimizer states, scheduler progress,
and Lagrange multiplier/optimizer stay in memory. The outer fit updates the same
constraint model and retains its Adam state. Keeping the Lagrange optimizer is an
explicit warm-start choice: its moments reflect previous cost functions, but its
new updates use only fresh episode costs. Resetting that optimizer is a possible
future ablation, not a mathematical requirement.

Learning-rate behavior remains controlled by the native OmniSafe configuration.
When `model_cfgs.linear_lr_decay` is enabled, the actor schedule spans the full run
without restarting at outer boundaries; set it to `false` for a constant actor
rate, as in Liu's Swimmer configuration. State retention does not imply that all
learning-rate hyperparameters reproduce Liu's experiments. The constraint rate
starts at `constraint.learning_rate` and follows `constraint.anneal_clr_by_factor`.

This prototype supports single-process algorithms inheriting that on-policy training
loop, such as PPOLag, TRPOLag, CPO, PPOPID and CPPOPID. SafeLOOP support has been
removed from ICRL. Custom training-loop overrides (including CPPGPID's phasic
loop) and off-policy solvers are rejected. Native vector training is retained;
fresh constraint-fitting episodes are collected in one separate environment.

The default config selects Liu's `VariationalConstraintNet`: a ReLU MLP predicts
the positive parameters of a Beta feasibility distribution. `mode: mean` uses
its deterministic mean for costs; training still uses the variational objective.
`di_prior: [1, 1]` sets the prior, and `regularizer_coeff` weights its KL penalty.
Set `constraint.model: standard` for the sigmoid `ConstraintNet` instead.
`input_type: state` ignores actions; `state_action` includes them.

The constraint sees raw observations, without policy normalization or observation
clipping, and clips native actions to their bounds in its existing preprocessing.
The wrapper calculates `1 - feasibility(s_t, a_t)` before the environment step,
before OmniSafe cost normalization. True costs are retained only for reporting;
they never label expert/learner pairs or train the critics/multiplier. For
on-policy learners, the sampler
uses the actor's observation statistics without updating them, then maps policy
actions through the same affine transformation as OmniSafe's `ActionScale`.

`constraint.backward_iters` controls fitting passes. `batch_size: null` uses all
expert and learner transitions in each pass; a number selects minibatches using
the imported sampler (bounded by the smaller dataset). The imported likelihood
loss and regularizer retain their formulas. Importance ratios and KL stopping
diagnostics use log-space trajectory sums to avoid underflow/overflow over long
episodes; non-finite diagnostics, VI losses or gradients now raise instead of
silently continuing. `cn_eps` is the
constraint model's numerical stabilizer, and `anneal_clr_by_factor` multiplies its
learning rate after each outer iteration, starting from `learning_rate` in the first.
Per-step importance
weights are flattened to one scalar per transition, fixing upstream broadcasting
that otherwise pairs every weight with every prediction. Their
values in the YAML are starting settings, not a tuned reproduction of a paper
experiment. Old episode-cost logging statistics are cleared between outer phases
so the multiplier does not average costs from different constraint models.

The native cost limit applies to the accumulated learned costs. It does not
calibrate those predictions to the environment's true safety budget automatically.
Oracle feedback and GAIL solver selection are not enabled by this entry point.

## Goal1 variational trial

`ICRL/configs/goal1/icrl_variational_safeloop.yaml` selects the 50 zero-cost
actor-only demonstrations exported from a SafeLOOP Goal1 checkpoint (epoch 25).
The filename identifies the demonstration source; the ICRL learner is now an
on-policy algorithm and needs no transition-model checkpoint. This outer config
uses the variational constraint model in Beta-mean mode, six outer iterations of
20,000 inner steps, three fresh learner episodes per iteration, and cost figures.

```bash
python -u -m ICRL.ICRL_solver \
  --icrl-config ICRL/configs/goal1/icrl_variational_safeloop.yaml \
  --omnisafe-config omnisafe/configs/on-policy/PPOLag.yaml
```

The demonstrations are **actor-only**, although the source checkpoint was trained
with a planner. Zero-cost filtering selects a subset of that actor's rollouts;
they do not reproduce the planner's behavior. The native on-policy cost limit is
applied to learned episode costs, so select it for this model's cost scale.

## Outputs and checks

One run lives under `icrl.runs_dir/<algorithm>-{<environment>}/<seed-run-name>/`.
OmniSafe chooses the experiment and seed-run names and writes its native files
there. Outer-loop files go in the run's `icrl/` subfolder. The default root is
`ICRL/runs`. `icrl/run_metadata.json` records the native run path, configurations,
expert dataset, algorithm, environment, seed and visualization settings.

- Native `progress.csv`, `torch_save/epoch-N.pt` and TensorBoard metrics describe inner training.
  `Metrics/EpCost` is the **learned** cost. Native TensorBoard steps remain inner
  epochs, with one continuous run across all outer iterations.
- Outer `icrl/icrl_progress.csv` and `ICRL/*` metrics in the same TensorBoard run record outer losses, fresh
  trajectory returns, true/learned episode costs, and cumulative training/sampling
  interactions. Each row/point represents one **outer iteration**. TensorBoard
  uses the outer iteration for `ICRL/*` and inner epochs for native metrics;
  `ICRL/PolicyEpoch` records the completed policy epoch.
  `ICRL/SampleReturn` and `ICRL/SampleTrueCost` describe the final inner policy's
  sampled episodes. `ICRL/SampleLearnedCostBefore` (formerly `SampleLearnedCost`)
  uses the constraint before fitting; `ICRL/SampleLearnedCostAfter` rescores the
  **same observations/actions** after fitting, without additional environment
  interactions. Both are mean undiscounted episode costs, not mean step costs.
  These are fitting trajectories, not a held-out constraint-validation set.
  Cost-fitting loss fields retain the imported model's final-minibatch/pass reporting semantics.
  `SampleViolationRate` means any positive true cost, not exceeding the cost budget.
- Iteration starts and a completion summary are printed to the console. Persistent
  metrics carry `ICRL/Iteration`; there is no additional text-log file.
- Native `config.json` contains the resolved OmniSafe configuration, including
  the shared run root; `icrl/icrl_config.yaml` records the ICRL settings.
- Native `torch_save/epoch-N.pt` uses actor/normalizer checkpoints. Critics and
  optimizer states persist in memory across iterations; these policy snapshots
  are not full training-resume checkpoints.
- The constraint model and its optimizer stay in memory and are updated after
  each outer iteration. New runs do not save `constraint-*.pt` snapshots. A
  policy saved after iteration k was trained with constraint k-1; the newly
  fitted constraint k is used in the next iteration. Cost estimates remain in
  `icrl_progress.csv`, TensorBoard, live rollout manifests and optional cost plots.

The four Circle1 ICRL runs from 2026-10-25 were merged into this layout. Their
historical cost-model snapshots are in
`ICRL/cost_snapshot_archive/<experiment>/<seed-run>/`, outside active `ICRL/runs`.
The native policy checkpoints, TensorBoard files and outer-loop logs are in the
same run directory. Standalone forward-CRL runs continue using OmniSafe's
normal `runs/` root.

For a run started before the outer-iteration TensorBoard axis was restored,
`python -m ICRL.utils.align_tensorboard <run-directory>` copies its `ICRL/*`
CSV metrics into `icrl/tb_iteration_axis` at iteration steps. This leaves the
original event file untouched; rerun the command to append later iterations.

To put a completed historical run on W&B, use
`python -m ICRL.utils.upload_wandb <run-directory>`. It uploads the native and
ICRL scalar histories as one run, maps `ICRL/*` to the outer-iteration axis,
and includes both saved configurations. `--dry-run` reports the intended run
ID and number of steps without uploading. Import each completed run once.

Set `visualize_cost.enabled: true` in `ICRL/icrl_config.yaml` to save a figure and JSON
after every cost-function update under the outer run's `visualize_cost/` directory.
`trajectories_per_source` randomly selects that many **whole episodes** from each of
the current learner sample and the fixed expert dataset. `step_stride: 1` evaluates
the fitted immediate cost of every recorded `(observation, action)` pair; larger
strides show fewer points. Each panel uses that episode's SafetyGoal layout and
world-coordinate path, with hazards and goal locations drawn behind the points.
Red intensity is divided by the **largest plotted cost in that iteration**; the
unnormalized maximum and every plotted step, position and cost are saved in the
iteration JSON. Colors are therefore comparable within one figure, while raw JSON
values are needed to compare iterations. These plots cover visited actions only,
not an exhaustive cost map or a state-only function when the model uses actions.

Visualization requires a SafetyGoal task and an expert NPZ with `episode_lengths`
and `episode_seeds`, as written by `weights2trajectories.py`. That NPZ stores
pre-action observations/actions, but no world positions. Plotting replays selected
expert episodes to recover those positions and checks that replayed observations
match the file. This diagnostic replay adds environment steps without affecting
fitting or inner training.
`ICRL/TotalInteractions` keeps its training-plus-learner-sampling meaning;
`ICRL/VisualizationReplaySteps` and `ICRL/AllEnvironmentSteps` expose the added
diagnostic steps when visualization is enabled. Learner positions are captured
during the existing sampling pass, so plotting adds no learner rollouts.

## Inspect policy behavior from above

`ICRL/visualize_rollouts.py` deploys policies on **new seeds of the same task**.
The default `--view map` renders an overhead animation: Circle1 includes its
constrained walls and target circle; Goal1 includes hazards and the current goal.
`--view fixedfar` uses Safety Gymnasium's external MuJoCo camera, and `--view both`
puts camera and map side by side. The latter two require a working MuJoCo RGB
renderer. Videos, `manifest.json` and a browsable `index.html` go to `rollouts/`.

The supplied Circle1 outer YAML has `visualize_rollouts.enabled: true` to film the **live**
inner policy on the configured fixed seeds after every inner phase, before fitting
the next cost model. These episodes are separate from constraint fitting;
`ICRL/VideoEvalSteps` and `ICRL/AllEnvironmentSteps` count their interactions.
The live video for iteration `i` scores actions with the in-memory constraint
used to train that policy, before fitting the next constraint.

The same one script works after a run. For example, set `RUN_DIR` to
`ICRL/runs/<experiment>/<seed-run>/icrl` and `DATASET` to its expert NPZ:

```bash
python -m ICRL.visualize_rollouts --source learner --run-dir "$RUN_DIR" \
  --iterations 1 2 --seeds 10001 10002 --view map
python -m ICRL.visualize_rollouts --source expert-policy --dataset "$DATASET" \
  --seeds 10001 10002 --output-dir "$RUN_DIR/rollouts" --view map
python -m ICRL.visualize_rollouts --source expert-trajectory --dataset "$DATASET" \
  --episodes 0 1 --output-dir "$RUN_DIR/rollouts" --view map
```

Omit `--iterations` to render all completed outer iterations, or use
`--episodes all` to replay every expert episode. Fresh policy deployments use the
requested reset seeds; exact expert replay instead uses the dataset's saved seed
**and actions**, checking saved observations and reward/cost before accepting the
video. The learner viewer restores the on-policy actor and observation statistics.
When a matching historical constraint snapshot is available, it also scores the
replay with that model. New runs have no saved constraint snapshots, so saved
learner replay is actor-only: it reports true reward and cost but cannot
reconstruct the old learned-cost predictions. Live run videos and cost plots
still record those predictions at the time of training.
A SafeLOOP expert export remains labelled **actor-only**;
playback of old SafeLOOP ICRL learner runs is no longer supported. Different task IDs require compatible policy inputs;
changing the seed is the supported new-instance comparison.

```bash
python -m unittest discover -s ICRL/tests -v
```

The integration tests execute real PPOLag updates and two outer iterations in a
small deterministic test environment, check exact step budgets, cost/coordinate
separation, persistent weights/optimizers/Lagrange state and all enabled normalizers,
environment/buffer resets, frozen sampling normalization, checkpoints and TensorBoard output.
They also compare staged and uninterrupted native training for exact parameter
agreement. These are implementation checks, not a MuJoCo performance study.

## Imported Liu et al. constraint model

`constraint_models/constraint_net/constraint_net.py` is a compatibility port of
the class from `Guiliang/ICRL-benchmarks-public`, used by the standard outer-loop setting. Import it with:

```python
from ICRL.constraint_models.constraint_net import ConstraintNet
```

| Original dependency or runtime issue | OmniSafe-compatible replacement |
| --- | --- |
| SB3 `create_mlp` | OmniSafe `build_mlp_network`, with ReLU hidden layers and a sigmoid output. Preserve the original Linear initialization and CPU RNG progression, including exact weights for a given seed and existing checkpoint keys. |
| SB3 `update_learning_rate` | Assign the callable schedule's value directly to every PyTorch optimizer parameter group, exactly as the old helper did. |
| Benchmark `utils.data_utils.idx2vector` | A small local NumPy helper preserving grid-coordinate rounding, flattened one-hot encoding, dtype and empty-input behavior. OmniSafe has no corresponding gridworld helper. |
| Checkpoint loading on current PyTorch | Explicit full-checkpoint loading for trusted files containing NumPy statistics, device mapping, and resolution of `auto` through the available CPU/CUDA device and OmniSafe's device helper. |
| Existing `_load()` typo and inference-model saving | Replace the undefined `dic` with `state_dict`; allow saving a model loaded without an optimizer. |

The constructor, normalization/clipping, feature selection, Adam epsilon,
progress-based learning-rate schedule, loss formulas, regularization, importance
sampling and training loops retain their original behavior except for the per-step
importance-weight shape correction described above. `forward()` returns
feasibility tensors; `cost_function()` accepts NumPy inputs and returns NumPy
costs equal to `1 - feasibility`. It does not use the earlier `CostFunction`'s
cost-returning `forward()` convention. No dependency installation is needed in
the existing `omnisafe` environment.

At the initial dependency-port stage, all 14 ICRL tests passed. A direct
comparison against the pre-port class also matched metrics, parameters and costs
exactly in 12 cases: flat/trajectory training, ICRL/BCE loss, and disabled/episode/
per-step importance sampling (before the subsequent shape correction). Flat comparisons used minibatches; trajectory
comparisons used complete trajectories. The tests additionally cover network
initialization/RNG equivalence, preprocessing, grid/discrete inputs, optimizer
restoration, checkpoint round trips, and import without legacy dependencies.
This checks the compatibility changes, not the correctness of every upstream
algorithmic edge case; no full ICRL experiment or GPU validation was run here.

## Imported GAIL and variational models

Both additional models are available without Stable-Baselines3 or the benchmark
repository on `PYTHONPATH`:

```python
from ICRL.constraint_models.constraint_net import (
    ConstraintNet,
    GailDiscriminator,
    VariationalConstraintNet,
)
```

Their constructor parameters and NumPy-based fitting/cost APIs are retained.
GAIL has `train_nn(...)` for fitting; its old `train(iterations, obs, actions, ...)`
call remains supported. Calling `train()` or `eval()` now switches PyTorch module
mode normally. VICRL retains `train_nn(...)` and `train_traj_nn(...)`, Beta
reparameterized sampling, the Dirichlet KL regularizer, importance sampling and
all five cost modes (`sample`, `mean`, `VaR`, `CVaR`, `hard`). `forward()` returns
feasibility/discriminator predictions, not costs. `cost_function()` retains its
NumPy API; the outer loop uses the additional tensor-valued `immediate_cost()`
interface for both real and imagined OmniSafe transitions.

| Original dependency or incompatibility | Replacement |
| --- | --- |
| SB3 `create_mlp` in both models | OmniSafe `build_mlp_network`, preserving hidden ReLU layers, output sigmoid for GAIL / two Softplus outputs for VICRL, checkpoint keys, seeded weights and CPU RNG progression. |
| SB3 learning-rate helper | Direct assignment to PyTorch optimizer parameter groups; VICRL inherits the already ported implementation. |
| Absolute benchmark `ConstraintNet` import | Import the existing port through `ICRL.constraint_models.constraint_net`. |
| Benchmark `idx2vector` | Reuse the local helper from the existing constraint-model port. |
| Benchmark `dirichlet_kl_divergence_loss` | Retain the same small analytic PyTorch implementation locally, including operation order. OmniSafe has no counterpart. Values and gradients are also checked against PyTorch's `Dirichlet` KL. |
| SB3 `BaseCallback`, rollout buffer and `logger.record` | Explicit `GailCallback.on_rollout_end(...)`, a native `VectorOnPolicyBuffer` subclass, and this repository's `Logger.write_scalars`. |
| Benchmark `del_and_make` | Standard-library `Path.mkdir`; plotting no longer deletes the existing discriminator directory at initialization. Existing filenames may still be overwritten. |
| GAIL fitting method shadows PyTorch `train` | A compatibility dispatcher preserves old fitting calls and supports normal PyTorch mode switching. |
| GAIL checkpoint typo, current PyTorch loading, inference resaving | Fix the undefined `dic`; map devices using OmniSafe's device helper; explicitly load full trusted checkpoints; allow absent optimizers. Preserve spurious-feature/grid settings and epsilon in new checkpoints. |
| VICRL checkpoint settings | Small checkpoint-extension hooks in `ConstraintNet` retain VICRL's prior, risk mode, confidence, grid settings and epsilon on inference reload. Existing checkpoint keys remain compatible. |
| VICRL BCE label device / zero-metric dtype | Use prediction-device labels and a floating zero regularizer. The latter prevents `mean()` failing on an integer tensor after fitting on current PyTorch; the loss value and gradients are unchanged. |

`load(...)` constructs an inference model without an optimizer. To resume fitting,
construct the model with the original training configuration, data, prior/mode
settings and schedule, then call `_load(path)` to restore weights and optimizer
state. Legacy checkpoints that did not save optional inference settings continue
to use their constructor defaults; absent metadata cannot be recovered.

### GAIL rollout integration

OmniSafe does not implement SB3's callback protocol. `GailCallback` therefore
requires an explicit call from the training driver, after rollout collection and
before `buffer.get()` / the policy update:

```python
from ICRL.gail_callback import GailCallback, GailRolloutBuffer

# Construct GailRolloutBuffer with the same arguments as VectorOnPolicyBuffer
# and pass it to the on-policy rollout collector from the start.
hook = GailCallback(discriminator, learn_cost=False, update_freq=1)

# After collection; raw arrays have shape [steps, num_envs, features].
metrics = hook.on_rollout_end(
    buffer=buffer,
    observations=raw_pre_action_observations,
    actions=raw_executed_actions,
    logger=logger,
    global_step=total_environment_steps,
)
# The native policy update can now consume buffer.get().
```

`GailRolloutBuffer` retains each environment's episode boundaries and the actual
reward/cost bootstrap values passed to `finish_path`. After reward relabeling it
calls native OmniSafe return/advantage computation for each recorded path. This
supports terminal episodes, time limits, epoch cutoffs, vector environments and
all four native advantage estimators. `get()` clears the path records. Duplicate
relabeling and unfinished rollouts are rejected before fitting the discriminator.

The input observations/actions must match the expert data's coordinates and any
supplied `true_cost_function`. Keep the raw observations during collection;
clipped observation normalization cannot be inverted exactly. Do not silently mix
native actions with OmniSafe's scaled policy actions. The hook flattens time and
environment dimensions together before calling the original 2D preprocessing.

As in the original callback, `learn_cost=False` replaces rewards with
`log(D + eps)`, while `learn_cost=True` **adds** that quantity to the buffer's
existing rewards. This second mode does not replace OmniSafe's `cost` signal or
install a learned constraint. Reward normalization, if enabled, determines the
units of the existing reward in this addition. The hook keeps cost targets intact
and logs ground-truth cost only as `eval/mean_cost`. Discriminator metrics go to
the enabled TensorBoard/W&B sinks through the existing OmniSafe logger.

The hook is for on-policy buffers; it is not an off-policy replay-buffer adapter.
The outer ICRL loop selects the cost model with `constraint.model: standard` or
`constraint.model: variational` in `ICRL/icrl_config.yaml`. Both expose
`immediate_cost(raw_observation, native_action)`: one scalar cost per leading
batch entry in `[0, 1]` (larger means less feasible), used for real training
steps and evaluation.
State-only models ignore the action. The variational model defaults to the
deterministic Beta mean in this loop; `di_prior`, `mode`, and `confidence` may be
set in the constraint section. The legacy NumPy `cost_function` and its risk
modes remain available, but `VaR` and particularly Monte Carlo `CVaR` can be
slow when evaluated at every environment step. The
`regularizer_coeff` parameter weights different penalties in the standard and
variational models, so it should be tuned separately. For variational fitting,
importance ratios now compare Beta means rather than independent random draws;
this removes spurious changes with fixed weights and deliberately differs from
the copied Liu routine. Per-transition importance weights are also kept one
dimensional so the weighted loss cannot broadcast to a quadratic matrix.

### Preserved upstream behavior to consider before experiments

This is a dependency/runtime port. The following algorithmic quirks are present
in the copied source and remain unchanged:

- GAIL's `_define_input_dims` does not offset action indices by the observation
  width. With default selection and `obs_dim >= acs_dim`, these indices select
  observation columns again, so actions are effectively ignored. This affects
  action-dependent constraints.
- GAIL's observation normalization and action clipping are commented out in its
  preprocessing. Expert and learner inputs must already share coordinates.
- VICRL's `train_traj_nn(..., train_gail_lambda=True)` branch calculates BCE loss
  but does not run the optimizer step because of the source's indentation.
  Ordinary VICRL trajectory fitting and flat BCE fitting do run updates.
- VICRL's VaR/CVaR modes read `self.confidence`; the `confidence` argument to
  `cost_function` is unused in the source. CVaR uses SciPy/NumPy Monte Carlo
  samples, and `hard` rounds a random Beta sample, not its mean.

These are not claims that the original algorithms are correct in every mode.
Changing these behaviors would be a separate algorithmic correction.

Port verification before outer-loop integration: 24 focused tests in the existing `omnisafe` CPU environment,
including both GAIL reward modes against fresh native OmniSafe buffers for all
four advantage estimators, real TensorBoard event output, checkpoint round trips,
optimizer resumption, KL values/gradients and imports with legacy packages blocked.
A separate direct comparison against the original classes previously checked 20
training configurations. That comparison predates the intentional log-space
importance-sampling correction, so exact parity no longer applies to that path.
The current ICRL suite also tests 500-step importance ratios and non-finite VI
diagnostics.
