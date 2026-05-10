import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json
import os
from pathlib import Path

# 1. Setup paths and parameters
root_dir = Path("runs/CPPGPID-{SafetyCarButton1-v0}")
target_hyperparam = "alpha_cost"
data_list = []
custom_colors = ["#FF0000", "#B22222", "#8B4513", "#5F9EA0", "#00CED1", "#00FFFF"]

# 2. Collect data
if root_dir.exists():
    for seed_folder in root_dir.iterdir():
        config_path = seed_folder / "config.json"
        csv_path = seed_folder / "aux_kl_inspection.csv"

        if config_path.exists() and csv_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
                if config.get("seed") == 1000:
                    continue
                hp_value = config.get("algo_cfgs", {}).get(target_hyperparam, "unknown")

            df = pd.read_csv(csv_path)
            df[target_hyperparam] = hp_value
            df['Seed'] = seed_folder.name
            data_list.append(df)

# Fallback: If running locally without the folder structure
if not data_list:
    print("Warning: Directory structure not found. Loading local aux_kl_inspection.csv if available.")
    df = pd.read_csv("aux_kl_inspection.csv")
    df[target_hyperparam] = "Baseline"
    df['Seed'] = "Seed_1"
    data_list.append(df)

# 3. Create Master DataFrame
master_df = pd.concat(data_list, ignore_index=True)

# 4. Plotting - 1x2 Grid
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
sns.set_style("whitegrid")
palette = custom_colors[:master_df[target_hyperparam].nunique()]

# --- Plot 1: Intra-Phase Dynamics (Epoch over Epoch) ---
sns.lineplot(
    data=master_df,
    x="aux_epoch",
    y="kl_difference",
    hue=target_hyperparam,
    palette=palette,
    estimator='median',
    errorbar=('pi', 95),
    ax=axes[0]
)
axes[0].set_title("Aggregated Intra-Phase KL Growth (Median across all phases)")
axes[0].set_xlabel("Auxiliary Epoch")
axes[0].set_ylabel("KL Divergence")

# --- Plot 2: Inter-Phase Evolution (Phase over Phase) ---
sns.lineplot(
    data=master_df,
    x="training_phase",
    y="kl_difference",
    hue=target_hyperparam,
    palette=palette,
    estimator='mean',
    errorbar='sd',
    ax=axes[1]
)
axes[1].set_title("Inter-Phase KL Evolution (Mean ± Std of all epochs per phase)")
axes[1].set_xlabel("Training Phase")
axes[1].set_ylabel("Mean KL Divergence")

plt.tight_layout()
plt.show()
