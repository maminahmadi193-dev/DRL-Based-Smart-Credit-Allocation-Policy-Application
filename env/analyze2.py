import pandas as pd
import numpy as np

PATH = r"C:\\Users\\m.ahmadi\\Desktop\\FinalProject\\models\\results\\reward_landscape.csv"


df = pd.read_csv(PATH)

actions = [
    "0",
    "100M",
    "200M",
    "300M",
    "400M",
    "500M",
]

reward_cols = [
    f"reward_{a}"
    for a in actions
]

# ------------------------------------------------------------
# Oracle reward
# ------------------------------------------------------------

oracle_reward = df[reward_cols].max(axis=1)

# ------------------------------------------------------------
# Baselines
# ------------------------------------------------------------

results = []

for action in actions:

    rewards = df[f"reward_{action}"]

    regret = (
        oracle_reward - rewards
    )

    results.append({
        "policy": f"always_{action}",
        "mean_reward":
            rewards.mean(),
        "mean_regret":
            regret.mean(),
        "median_regret":
            regret.median(),
        "exact_optimality":
            np.mean(regret == 0),
        "near_optimality":
            np.mean(regret <= 0.01),
    })


# ------------------------------------------------------------
# Oracle
# ------------------------------------------------------------

results.append({
    "policy": "oracle",
    "mean_reward":
        oracle_reward.mean(),
    "mean_regret":
        0.0,
    "median_regret":
        0.0,
    "exact_optimality":
        1.0,
    "near_optimality":
        1.0,
})


results_df = pd.DataFrame(results)

print(
    results_df.to_string(
        index=False
    )
)