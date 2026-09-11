# Frozen-encoder linear probing

This directory contains a two-step probing pipeline:

1. `create_dataset.py` collects one fixed set of raw Safety-Gymnasium observations. Privileged
   simulator geometry is used only to construct supervision labels and is also stored in the
   dataset for future target definitions. It generates an oversized candidate pool from random,
   evasive-controller, and converged-policy rollouts, then retains a uniform random subset.
2. `run_probes.py` reconstructs every configured actor snapshot, optionally applies that snapshot's
   observation normalizer without updating it, freezes the shared actor backbone, and fits a linear
   model for each enabled task.

Configure the environment, sample count, snapshot directories, and task switches in
`linear_probing_config.yaml`. The configured `agent_radius` is added to the hazard radius when
testing path clearance. The default value `0.1` matches the Point robot; it must be adapted for
other morphologies. Custom environment-construction arguments can be placed under `env_kwargs`.
Task switches control probe fitting and explanation panels; all labels remain available in the
reusable dataset.

The task switches are grouped as `agent_related`, `goal_related`, and `hazard_related`, then by
`easy`, `medium`, and `hard`. This fixed hierarchy also determines the order of the explanation
panels. Every task must appear exactly once; its boolean controls whether its probe and panel run.

## Candidate collection

`dataset.candidate_pool_multiplier` controls the size of the temporary candidate pool. A value of
`10` means that 10 times the requested `dataset.num_samples` are generated. Values between 5 and
20 are a useful default range. The pool is divided approximately equally among:

- uniformly random environment actions;
- a simple goal-seeking controller that turns toward a clear goal-length ray; and
- deterministic restored policies with configurable Gaussian action noise.

The final observations are sampled uniformly without replacement from the combined pool. The
collector selects the candidate indices before rollout instead of holding the full pool in memory;
this is equivalent to sampling after generation. Each retained row stores its source in `sources`.
For converged-policy rows, `policy_snapshots` also stores the exact snapshot path.

Set `snapshots.collection_directory` to held-out runs used only for dataset generation. Set
`snapshots.probing_directory` to the experimental runs whose frozen encoders should be evaluated.
The scripts reject any training run that appears in both groups. This check is repeated during
probing using the collector provenance embedded in the dataset, so changing the configuration after
dataset generation cannot accidentally introduce overlap.

Set `collection.converged_policy_paths` to an explicit list when particular held-out policies should
produce the data. When the list is empty, the collector discovers snapshots below
`snapshots.collection_directory` and applies `collection_selection`. The default chooses the latest
auxiliary epoch from the latest outer phase of every run. Probing independently applies
`probing_selection` to snapshots below `snapshots.probing_directory`. A collection policy is
reconstructed in full and its saved observation normalizer is applied read-only, so actions match
the training-time input preprocessing. Multiple restored policies are rotated between episodes.
`collection.action_noise_std` is applied in normalized action space to both the evasive and
restored-policy actions.

`snapshots.use_snapshot_normalizer` affects probe embeddings only. Collector policies always use
their own saved training-time normalizer; otherwise changing this analysis switch would also change
the observations in the dataset.

The built-in evasive controller assumes a Point/Car-style two-dimensional continuous action: one
forward component and one steering component. A different action interface requires a corresponding
controller implementation.

## Snapshot selection

Collection and probing have independent selectors. `phases`, `policy_iterations`, and
`auxiliary_epochs` accept `all`, `latest`, or an explicit integer list. `stages` accepts any
non-empty subset of `[policy, auxiliary]`. For example, this evaluates every auxiliary epoch from
outer phase 31 and no policy snapshots:

```yaml
snapshots:
  probing_selection:
    phases: [31]
    stages: [auxiliary]
    policy_iterations: all
    auxiliary_epochs: all
```

Explicit `collection.converged_policy_paths` are exact overrides and are not filtered by
`collection_selection`.

## Reusing a generated dataset

Every dataset build receives a sortable UTC timestamp such as
`20260910T143012.123456Z`. The configured filenames are treated as base names, producing:

```text
linear_probing/data/<dataset-version>/probe_dataset.npz
linear_probing/data/<dataset-version>/probe_dataset.metadata.json
linear_probing/visualizations/<dataset-version>/sample-*.png
```

The dataset and its explanation figures deliberately share one version. Dataset metadata records
that version and the exact visualization directory. It also contains array shapes and types, all
generation settings, target and unit information, source counts, collector snapshot paths and
checksums, dependency versions, and the repository revision. The JSON sidecar adds a checksum for
the NPZ itself.

`dataset.version: latest` makes `run_probes.py` select the newest complete dataset/metadata pair.
Replace it with the exact timestamp directory to reproduce an older experiment. The special value
`unversioned` supports a compatible legacy artifact stored directly at the configured paths.

Every probe invocation gets its own independent version and writes:

```text
linear_probing/results/<result-version>/linear_probe_results.csv
```

Both the concrete dataset version and result version are columns in every CSV row, so later runs do
not overwrite or obscure their provenance. The scripts print the chosen versions and full paths.

`run_probes.py` verifies the supported dataset and target schema versions, checksum,
embedded/sidecar agreement, array schema, sample count, and collector/probe run separation before
fitting anything. It never regenerates the dataset. A target-definition or schema update therefore
requires rebuilding the dataset; old artifacts are rejected instead of being interpreted with new
code. Set `probing.enabled: false` to make an accidental invocation exit without fitting probes.
Running only `create_dataset.py` always builds the dataset and configured explanation figures
without probing.

## Qualitative explanations

Set `visualization.num_examples` to a positive integer to select that many observations uniformly
without replacement from the dataset. Set it to `0` to disable rendering. Selection is
reproducible through `visualization.seed`.

For every selected observation, dataset generation saves one PNG under the current timestamped
child of `visualization.output_directory`.
The first panel is the actual Safety-Gymnasium RGB render from the configured camera. A wide task
area groups the remaining top-down panels into agent-, goal-, and hazard-related sections, ordered
from easy to hard and marked with difficulty badges. The panels include the circular actor's swept
direct-path corridor, critical clearance, evasive direction, planned safe path, and hazard
nearest-neighbour connections. Kinetic energy is printed in scientific notation and explicitly
refers to all selected agent DoFs. These panels explain the ground-truth label; they do not show a
probe prediction.

Rendering requires a working MuJoCo OpenGL backend. When generating data on a headless machine,
configure the appropriate MuJoCo rendering backend or set `num_examples: 0`.

Build the dataset from the repository root:

```bash
python -m linear_probing.create_dataset \
  --config linear_probing/linear_probing_config.yaml
```

To fit probes, first set `probing.enabled: true`, then run:

```bash
python -m linear_probing.run_probes \
  --config linear_probing/linear_probing_config.yaml
```

## Auxiliary-only plots

`aux_only_plotting.py` reads the configured probe-result version, keeps only auxiliary snapshots,
and excludes `aux-epoch-000` because it is the checkpoint immediately before auxiliary learning.
For every outer phase it creates one figure containing an across-task average followed by one
subplot per task. Equivalent phase/epoch coordinates from independent training runs are shown as
their mean and standard deviation. One additional figure shows the overall mean in red. For this
figure, phases are first averaged within each run, after which the mean and standard deviation are
computed across runs. It also splits the phases chronologically into near-equal thirds and shows
their hierarchically aggregated means in light blue, medium blue, and blue-violet. Regression and
circular-regression tasks use R²; classification tasks use balanced accuracy. The across-task
average is the unweighted mean of those absolute metrics.

Each task receives dynamic y-axis limits based on its values across the complete run. Those limits
are then held fixed for that task in every phase figure, improving visibility while preserving
phase-to-phase comparability. Every subplot labels its own metric. The output is stored in a new
timestamped child of `aux_only_plotting.output_directory`; `plot_metadata.json` records the source
result version, task metrics, phases, epochs, and per-task limits.

```bash
python -m linear_probing.aux_only_plotting \
  --config linear_probing/linear_probing_config.yaml
```

`policy_only_plotting.py` instead reads a policy-only result, orders all policy checkpoints by
outer phase and policy iteration, and flattens them into one continuous `Policy snapshot` axis. It
does not create phase-specific figures or phase thirds. Equivalent checkpoints are averaged across
training runs and shown with their sample standard deviation. The single output figure again
contains the across-task average followed by one subplot per task.

```bash
python -m linear_probing.policy_only_plotting \
  --config linear_probing/linear_probing_config.yaml
```

The dataset is split by complete episodes instead of individual observations, preventing the same
randomized layout from appearing on both sides. The split is also stratified by collection source
when enough episodes are available, so random, evasive, and restored-policy observations are not
accidentally confined to only one side. All snapshots use that same split. The result CSV contains
the phase, policy/auxiliary stage, stage iteration, total training steps, normalizer sample count,
task name, sample counts, and relevant held-out metrics.

Each source must contribute at least two sampled episodes. The configured `num_samples: 5000` is
intended for the real probing run; reduce it temporarily for a smoke test. Tasks with constant
targets, missing classes, or too few finite labels are retained in the CSV with a non-`ok` status
and `NaN` metrics rather than a misleading score. Classification rows include train/test class
counts. Regression R² is variance-weighted across output dimensions; angular MAE is in radians.

## Planner semantics

All hazard-path targets treat hazards as circles expanded by `agent_radius`. The direct-path tests
use the exact finite agent-to-goal segment. `minimum_evasive_turn` checks straight, goal-length rays
at discrete multiples of `avoidance_angle_step_degrees` up to
`maximum_avoidance_angle_degrees`. It returns the smallest clear magnitude; when left and right are
both clear at that magnitude, the side with more clearance wins. If no candidate is clear, the turn
and side labels are undefined (`NaN`).

`safe_path_detour_ratio` is `1` for an exact clear direct path. Otherwise it uses an approximate
eight-neighbour grid within `planner.bounds`; grid edges and the exact-point connectors are checked
against the expanded circular hazards. A smaller `grid_resolution` improves fidelity but costs more
dataset-generation time. Agent and goal positions must lie within the bounds. Occupied endpoints or
the absence of a grid route produce an undefined (`NaN`) target.

## Target definitions

- `nearest_hazard_distance`: centre-to-centre distance from actor to its nearest hazard.
- `goal_distance`: centre-to-centre planar distance from actor to goal.
- `signed_goal_angle`: signed shortest angle from the actor heading to the goal direction.
- `agent_velocity`: two-output reconstruction of the forward and lateral velocimeter components.
- `agent_kinetic_energy`: instantaneous agent-subtree kinetic energy computed as
  `0.5 * qvel.T @ M(q) @ qvel` from MuJoCo's generalized mass matrix.
- `signed_slip_angle`: signed shortest angle from agent heading to planar world velocity; undefined
  when planar speed is effectively zero.
- `direct_path_blocked`: whether any radius-expanded hazard intersects the actor-to-goal segment.
- `direct_path_hazard_count`: number of expanded hazards intersecting that segment.
- `direct_path_signed_clearance`: minimum centre-to-segment distance minus expanded hazard radius.
- `minimum_evasive_turn`: smallest signed deviation from the goal direction producing a clear
  goal-length segment.
- `best_avoidance_side`: left/right sign of that turn, evaluated only when the direct path is
  blocked.
- `safe_path_detour_ratio`: grid-planned collision-free distance divided by straight-line goal
  distance.
- `hazard_dispersion`: mean nearest-neighbour hazard distance, min-max normalized over the fixed
  dataset so the most clustered layout is `0` and the most dispersed layout is `1`.

`signed_goal_angle` and `signed_slip_angle` cross the `-π`/`π` boundary. Their probes therefore
fit sine/cosine components and report circular error rather than treating the two sides of that
boundary as far apart. Undefined angular targets are excluded from both fitting and evaluation.
