"""Plot auxiliary KL diagnostics for CarGoal1 and HalfCheetahVelocity runs."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ENV_RUNS = {
    "SafetyCarGoal1-v0": Path("runs/CPPGPID-{SafetyCarGoal1-v0}"),
    "SafetyHalfCheetahVelocity-v1": Path("runs/CPPGPID-{SafetyHalfCheetahVelocity-v1}"),
}
HUE_COL = "Active_Constraint"
ENV_COL = "Environment"
SEED_COL = "Seed"
RUN_COL = "Run"
BETA_CLONE_LABEL = r"$\beta_{\mathrm{clone}} = 1.0$"
DELTA_CLONE_LABEL = r"$\delta_{\mathrm{clone}} = 0.01$"
PREFERRED_HUE_ORDER = [BETA_CLONE_LABEL, DELTA_CLONE_LABEL]

# Fixed label-to-color mapping so labels keep the same color in every subplot.
LABEL_COLORS = {
    DELTA_CLONE_LABEL: "#008080",
    BETA_CLONE_LABEL: "#ff8c00",
}
FALLBACK_COLORS = ["#6a0dad", "#228b22", "#0000ff", "#ff1493"]


def load_config(config_path: Path) -> dict:
    """Load one experiment config."""
    with config_path.open("r", encoding="utf-8") as config_file:
        return json.load(config_file)


def label_from_config(config: dict) -> str:
    """Build the plotted label from an experiment config."""
    algo_cfgs = config.get("algo_cfgs", {})
    is_static = algo_cfgs.get("static_beta_clone", False)

    if is_static:
        hp_value = algo_cfgs.get("beta_clone", "unknown")
        return rf"$\beta_{{\mathrm{{clone}}}} = {hp_value}$"

    hp_value = algo_cfgs.get("aux_kl_target", "unknown")
    return rf"$\delta_{{\mathrm{{clone}}}} = {hp_value}$"


def seed_from_config(config: dict, seed_folder: Path) -> str:
    """Read the true experiment seed, falling back to the run folder if needed."""
    if "seed" in config:
        return str(config["seed"])

    # Fallback for old runs that may not have stored the seed in config.json.
    folder_parts = seed_folder.name.split("-")
    if len(folder_parts) >= 2 and folder_parts[0] == "seed":
        return folder_parts[1].lstrip("0") or "0"

    return seed_folder.name


def collect_env_data(env_name: str, root_dir: Path) -> list[pd.DataFrame]:
    """Collect all aux KL CSVs for one environment."""
    env_frames = []

    if not root_dir.exists():
        print(f"Skipping missing run directory: {root_dir}")
        return env_frames

    for seed_folder in sorted(root_dir.iterdir()):
        config_path = seed_folder / "config.json"
        csv_path = seed_folder / "aux_kl_inspection.csv"

        if not csv_path.exists() or not config_path.exists():
            continue

        config = load_config(config_path)
        label = label_from_config(config)
        seed = seed_from_config(config, seed_folder)
        data = pd.read_csv(csv_path)
        data = data.rename(columns={"kl_difference": "kl_divergence"})
        data["aux_epoch"] = pd.to_numeric(data["aux_epoch"])
        data["training_phase"] = pd.to_numeric(data["training_phase"])
        data[HUE_COL] = label
        data[ENV_COL] = env_name
        data[SEED_COL] = seed
        data[RUN_COL] = seed_folder.name
        env_frames.append(data)

    return env_frames


def collect_data() -> pd.DataFrame:
    """Collect all configured environment data into one dataframe."""
    data_frames = []
    for env_name, root_dir in ENV_RUNS.items():
        data_frames.extend(collect_env_data(env_name, root_dir))

    if not data_frames:
        raise FileNotFoundError("No aux_kl_inspection.csv files found for configured runs.")

    return pd.concat(data_frames, ignore_index=True)


def filter_common_seed_runs(master_df: pd.DataFrame) -> pd.DataFrame:
    """Keep only condition/seed pairs that exist for every plotted environment."""
    env_pair_sets = {}
    for env_name in ENV_RUNS:
        env_df = master_df[master_df[ENV_COL] == env_name]
        env_pairs = set(
            map(
                tuple,
                env_df[[HUE_COL, SEED_COL]]
                .drop_duplicates()
                .sort_values([HUE_COL, SEED_COL])
                .to_numpy(),
            ),
        )
        env_pair_sets[env_name] = env_pairs

    common_pairs = set.intersection(*env_pair_sets.values())
    if not common_pairs:
        available = {
            env_name: sorted(seed for _, seed in pairs)
            for env_name, pairs in env_pair_sets.items()
        }
        raise ValueError(
            "No common condition/seed pairs found across all environments. "
            f"Available seeds by environment: {available}"
        )

    common_pair_df = pd.DataFrame(sorted(common_pairs), columns=[HUE_COL, SEED_COL])
    filtered_df = master_df.merge(common_pair_df, on=[HUE_COL, SEED_COL], how="inner")

    run_counts = (
        filtered_df[[ENV_COL, HUE_COL, SEED_COL, RUN_COL]]
        .drop_duplicates()
        .groupby([ENV_COL, HUE_COL, SEED_COL])
        .size()
    )
    duplicate_runs = run_counts[run_counts > 1]
    if not duplicate_runs.empty:
        print(
            "Warning: multiple run folders share the same environment, condition, "
            "and seed. All matching runs are kept:"
        )
        print(duplicate_runs.to_string())

    used_seeds = sorted(
        filtered_df[SEED_COL].unique(),
        key=lambda seed: (0, int(seed)) if seed.isdigit() else (1, seed),
    )
    print(f"Filtering to seeds shared across environments and conditions: {used_seeds}")
    return filtered_df


def build_palette(labels: list[str]) -> dict[str, str]:
    """Build a stable color palette keyed by label."""
    palette = {}
    fallback_iter = iter(FALLBACK_COLORS)
    for label in labels:
        if label in LABEL_COLORS:
            palette[label] = LABEL_COLORS[label]
        else:
            palette[label] = next(fallback_iter, "#555555")
    return palette


def ordered_hue_labels(labels: list[str]) -> list[str]:
    """Order known labels consistently, then append any unknown labels."""
    label_set = set(labels)
    ordered = [label for label in PREFERRED_HUE_ORDER if label in label_set]
    ordered.extend(sorted(label_set.difference(ordered)))
    return ordered


def clear_legend_title(axis) -> None:
    """Keep the legend compact by showing only the plotted labels."""
    legend = axis.get_legend()
    if legend is not None:
        legend.set_title("")


def final_aux_epoch_rows(env_df: pd.DataFrame) -> pd.DataFrame:
    """Keep the final auxiliary epoch measurement for each phase and seed."""
    group_cols = [HUE_COL, SEED_COL, RUN_COL, "training_phase"]
    final_row_indices = env_df.groupby(group_cols)["aux_epoch"].idxmax()
    return env_df.loc[final_row_indices].sort_values(group_cols)


def plot_aux_kl(master_df: pd.DataFrame) -> None:
    """Plot intra-phase and inter-phase KL diagnostics for each environment."""
    sns.set_style("white")

    env_order = list(ENV_RUNS.keys())
    hue_order = ordered_hue_labels(list(master_df[HUE_COL].unique()))
    palette = build_palette(hue_order)

    plot_kwargs = {
        "y": "kl_divergence",
        "hue": HUE_COL,
        "hue_order": hue_order,
        "palette": palette,
        "estimator": "median",
        "errorbar": ("pi", 95),
    }

    for env_name in env_order:
        env_df = master_df[master_df[ENV_COL] == env_name]
        if env_df.empty:
            continue

        fig, axes = plt.subplots(
            nrows=1,
            ncols=2,
            figsize=(12, 4),
            sharex=False,
            sharey=False,
        )
        intra_axis, inter_axis = axes

        sns.lineplot(ax=intra_axis, data=env_df, x="aux_epoch", **plot_kwargs)
        clear_legend_title(intra_axis)
        intra_axis.set_xlabel("Auxiliary Epoch")
        intra_axis.set_ylabel("KL Divergence")
        intra_axis.grid(False)

        final_aux_df = final_aux_epoch_rows(env_df)
        sns.lineplot(ax=inter_axis, data=final_aux_df, x="training_phase", **plot_kwargs)
        inter_axis.set_xlabel("Training Phase")
        inter_axis.set_ylabel("KL Divergence")
        inter_axis.grid(False)

        # Keep only one legend per environment figure to avoid clutter.
        if inter_axis.get_legend() is not None:
            inter_axis.get_legend().remove()

        fig.tight_layout()

    plt.show()


def main() -> None:
    """Collect data and show plots."""
    master_df = filter_common_seed_runs(collect_data())
    print("Collected common-seed aux KL runs:")
    print(master_df.groupby([ENV_COL, HUE_COL])[SEED_COL].nunique())
    plot_aux_kl(master_df)


if __name__ == "__main__":
    main()
