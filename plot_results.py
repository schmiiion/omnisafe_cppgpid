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
for seed_folder in root_dir.iterdir():
    config_path = seed_folder / "config.json"
    csv_path = seed_folder / "progress.csv"

    if config_path.exists() and csv_path.exists():
        # Load hyperparam
        with open(config_path, 'r') as f:
            config = json.load(f)
            if config.get("seed") == 1000:
                continue
            hp_value = config.get("algo_cfgs", {}).get(target_hyperparam, "unknown")

        # Load CSV
        df = pd.read_csv(csv_path)

        # Add metadata for Seaborn grouping
        df[target_hyperparam] = hp_value
        df['Seed'] = seed_folder.name

        data_list.append(df)

# 3. Create Master DataFrame
master_df = pd.concat(data_list, ignore_index=True)

# 4. Plotting
plt.figure(figsize=(10, 8))
sns.set_style("white")

plot = sns.lineplot(
    data=master_df,
    x="TotalEnvSteps",  # or "Step" depending on your CSV header
    y="Metrics/EpRet",  # Use the absolute Reward column name
    hue=target_hyperparam,
    palette=custom_colors[:master_df[target_hyperparam].nunique()],
    estimator='median',
    errorbar=None
    #errorbar=("pi", 95)  # Optional: shows 95% percentile spread like your reference
)

plt.title(f"Median grouped by {target_hyperparam} across n=4 seeds")
plt.xlabel("Steps")
plt.ylabel("Episode Return")
plt.legend(title=f"{target_hyperparam}")
plt.show()
