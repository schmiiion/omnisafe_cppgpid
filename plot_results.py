import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json
import os
from pathlib import Path

# 1. Setup paths and your specific color palette
root_dir = Path("runs/CPPGPID-{SafetyCarGoal1-v0}")
target_hyperparam = "alpha_cost"
data_list = []

# Colors approximated from your image (Red -> Cyan gradient)
custom_colors = ["#FF0000", "#B22222", "#8B4513", "#5F9EA0", "#00CED1", "#00FFFF"]

# 2. Collect data
it_over = root_dir.iterdir()
for seed_folder in it_over:
    config_path = seed_folder / "config.json"
    csv_path = seed_folder / "progress.csv"

    if config_path.exists() and csv_path.exists():
        # Load hyperparam
        with open(config_path, 'r') as f:
            config = json.load(f)
            if config.get("seed") == 1000:
                continue

            algo_cfgs = config.get("algo_cfgs", {})
            alpha_cost = algo_cfgs.get(target_hyperparam, "unknown")

            # Check algo_cfgs, with a fallback to root config, defaulting to False
            joint_aux_phase = algo_cfgs.get("joint_aux_phase", True)

            # Catch cases where JSON saved the boolean as a string (e.g., "True" or "False")
            if isinstance(joint_aux_phase, str):
                joint_aux_phase = joint_aux_phase.lower() in ['true', '1', 't', 'yes']

        # Determine the grouping label
        if not joint_aux_phase:
            group_label = "joint_aux_phase=False"
        else:
            # Force this to a string to prevent Seaborn from dropping mixed types
            group_label = f"{target_hyperparam}={alpha_cost}"

            # Load CSV
        df = pd.read_csv(csv_path)

        # Add metadata for Seaborn grouping
        df["Grouping_Key"] = group_label
        df['Seed'] = seed_folder.name

        data_list.append(df)

# 3. Create Master DataFrame
master_df = pd.concat(data_list, ignore_index=True)

# 4. Plotting
plt.figure(figsize=(10, 8))
sns.set_style("white")

unique_groups = master_df["Grouping_Key"].unique()
palette_dict = {}
color_idx = 0

for group in unique_groups:
    if group == "joint_aux_phase=False":
        palette_dict[group] = "green"
    else:
        palette_dict[group] = custom_colors[color_idx % len(custom_colors)]
        color_idx += 1

plot = sns.lineplot(
    data=master_df,
    x="TotalEnvSteps",
    y="Metrics/EpRet",
    hue="Grouping_Key",
    palette=palette_dict,
    estimator='median',
    errorbar=None
)

plt.title("Median Returns Grouped by joint_aux_phase & alpha_cost")
plt.xlabel("Steps")
plt.ylabel("Episode Return")
plt.legend(title="Groups")
plt.show()
