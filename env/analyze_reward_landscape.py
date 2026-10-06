import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from loan_env import LoanEnv


# ============================================================
# Configuration
# ============================================================

PATH = r"C:\\Users\\m.ahmadi\\Desktop\\FinalProject\\notebooks\\df_cleaned.csv"

N_VALIDATION_CUSTOMERS = 300
SEED = 42

ACTIONS = np.array([
    0,
    100_000_000,
    200_000_000,
    300_000_000,
    400_000_000,
    500_000_000,
], dtype=np.float32)

ACTION_NAMES = [
    "0",
    "100M",
    "200M",
    "300M",
    "400M",
    "500M",
]

# ------------------------------------------------------------
# These define "practically equivalent" actions.
# Start with 0.01 because that was your previous criterion.
# ------------------------------------------------------------

TIE_TOL = 0.01


FEATURE_COLUMNS = [
    "MountlyIncome",
    "IncomeScore",
    "JobScore",
    "AssetScore",
    "TotalCustomerScore",
    "TotalFacilities",
    "DebtAmount",
    "OriginAmount",
    "LoanCounts",
    "DebtRatio",
    "DebtPerLoan",
    "Has_No_History",
    "Employed",
    "Score_Min",
    "Score_Max",
    "Risk_Min",
    "Risk_Max",
    "Risk",
]


# ============================================================
# Load data
# ============================================================

def load_data():

    df_raw = pd.read_csv(PATH)

    all_indices = np.arange(
        len(df_raw)
    )

    # --------------------------------------------------------
    # Same split as train.py / test.py
    # --------------------------------------------------------

    train_idx, temp_idx = train_test_split(
        all_indices,
        test_size=0.30,
        random_state=42,
        shuffle=True
    )

    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=2 / 3,
        random_state=42,
        shuffle=True
    )

    train_raw = (
        df_raw
        .iloc[train_idx]
        .reset_index(drop=True)
    )

    val_raw = (
        df_raw
        .iloc[val_idx]
        .reset_index(drop=True)
    )

    test_raw = (
        df_raw
        .iloc[test_idx]
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Scaler fit ONLY on train
    # --------------------------------------------------------

    train_features = train_raw.copy()

    train_features["Risk"] = (
        train_features["Risk_Min"]
        + train_features["Risk_Max"]
    ) / 2

    scaler = StandardScaler()

    scaler.fit(
        train_features[
            FEATURE_COLUMNS
        ]
    )

    return (
        train_raw,
        val_raw,
        test_raw,
        scaler
    )


# ============================================================
# Calculate expected reward for one customer and one action
# ============================================================

def calculate_reward(
    env,
    row,
    action_index
):

    customer = (
        env._row_to_customer_state(
            row
        )
    )

    env.current_customer = customer

    _, reward, _, _, info = (
        env.step(
            action_index
        )
    )

    return float(reward)


# ============================================================
# Main analysis
# ============================================================

def main():

    (
        _,
        val_raw,
        _,
        scaler
    ) = load_data()

    rng = np.random.default_rng(
        SEED
    )

    n_customers = min(
        N_VALIDATION_CUSTOMERS,
        len(val_raw)
    )

    indices = rng.choice(
        len(val_raw),
        size=n_customers,
        replace=False
    )

    val_env = LoanEnv(
        val_raw,
        scaler
    )

    # --------------------------------------------------------
    # reward_matrix[i, a]
    # --------------------------------------------------------

    reward_matrix = np.zeros(
        (
            n_customers,
            len(ACTIONS)
        ),
        dtype=np.float64
    )

    # --------------------------------------------------------
    # Evaluate ALL actions for ALL customers
    # --------------------------------------------------------

    for i, idx in enumerate(indices):

        row = val_raw.iloc[idx]

        for action_index in range(
            len(ACTIONS)
        ):

            reward_matrix[
                i,
                action_index
            ] = calculate_reward(
                env=val_env,
                row=row,
                action_index=action_index
            )

    # ========================================================
    # 1. Best action
    # ========================================================

    best_action_indices = np.argmax(
        reward_matrix,
        axis=1
    )

    best_rewards = np.max(
        reward_matrix,
        axis=1
    )

    # --------------------------------------------------------
    # Second best
    # --------------------------------------------------------

    sorted_rewards = np.sort(
        reward_matrix,
        axis=1
    )

    second_best_rewards = (
        sorted_rewards[:, -2]
    )

    best_second_gap = (
        best_rewards
        - second_best_rewards
    )

    # ========================================================
    # 2. Number of near-optimal actions
    # ========================================================

    regret_matrix = (
        best_rewards[:, None]
        - reward_matrix
    )

    near_optimal_count = np.sum(
        regret_matrix <= TIE_TOL,
        axis=1
    )

    # ========================================================
    # 3. Print summary
    # ========================================================

    print()
    print("=" * 70)
    print("REWARD LANDSCAPE ANALYSIS")
    print("=" * 70)

    print(
        f"Validation customers analyzed : "
        f"{n_customers:,}"
    )

    print(
        f"Tie tolerance                 : "
        f"{TIE_TOL}"
    )

    print()

    # --------------------------------------------------------
    # Action distribution of simulator-optimal policy
    # --------------------------------------------------------

    print("=" * 70)
    print("OPTIMAL ACTION DISTRIBUTION")
    print("=" * 70)

    action_counts = (
        pd.Series(
            best_action_indices
        )
        .value_counts()
        .sort_index()
    )

    for action_index in range(
        len(ACTIONS)
    ):

        count = int(
            action_counts.get(
                action_index,
                0
            )
        )

        percentage = (
            100.0
            * count
            / n_customers
        )

        print(
            f"{ACTION_NAMES[action_index]:>6} : "
            f"{count:>4} "
            f"({percentage:>6.2f}%)"
        )

    # ========================================================
    # 4. Best vs second-best gap
    # ========================================================

    print()
    print("=" * 70)
    print("BEST vs SECOND-BEST GAP")
    print("=" * 70)

    print(
        f"Mean   : "
        f"{np.mean(best_second_gap):.6f}"
    )

    print(
        f"Median : "
        f"{np.median(best_second_gap):.6f}"
    )

    print(
        f"P10    : "
        f"{np.percentile(best_second_gap, 10):.6f}"
    )

    print(
        f"P25    : "
        f"{np.percentile(best_second_gap, 25):.6f}"
    )

    print(
        f"P75    : "
        f"{np.percentile(best_second_gap, 75):.6f}"
    )

    print(
        f"P90    : "
        f"{np.percentile(best_second_gap, 90):.6f}"
    )

    # ========================================================
    # 5. Number of near-optimal actions
    # ========================================================

    print()
    print("=" * 70)
    print("NUMBER OF NEAR-OPTIMAL ACTIONS")
    print("=" * 70)

    counts = pd.Series(
        near_optimal_count
    ).value_counts().sort_index()

    for n_actions, count in counts.items():

        percentage = (
            100.0
            * count
            / n_customers
        )

        print(
            f"{n_actions} near-optimal actions : "
            f"{count:>4} "
            f"({percentage:>6.2f}%)"
        )

    # ========================================================
    # 6. Policy uniqueness
    # ========================================================

    unique_policy_rate = np.mean(
        near_optimal_count == 1
    )

    print()
    print("=" * 70)
    print("POLICY UNIQUENESS")
    print("=" * 70)

    print(
        f"Customers with exactly one "
        f"near-optimal action: "
        f"{unique_policy_rate:.2%}"
    )

    print(
        f"Customers with >=2 "
        f"near-optimal actions: "
        f"{1 - unique_policy_rate:.2%}"
    )

    # ========================================================
    # 7. Save raw reward matrix
    # ========================================================

    output = pd.DataFrame(
        reward_matrix,
        columns=[
            f"reward_{name}"
            for name in ACTION_NAMES
        ]
    )

    output["best_action"] = [
        ACTION_NAMES[i]
        for i in best_action_indices
    ]

    output["best_reward"] = (
        best_rewards
    )

    output["second_best_reward"] = (
        second_best_rewards
    )

    output["best_second_gap"] = (
        best_second_gap
    )

    output["near_optimal_count"] = (
        near_optimal_count
    )

    output["row_index"] = (
        indices
    )

    os.makedirs(
        "models/results",
        exist_ok=True
    )

    output_path = (
        "models/results/"
        "reward_landscape.csv"
    )

    output.to_csv(
        output_path,
        index=False
    )

    print()
    print("=" * 70)
    print("SAVED")
    print("=" * 70)

    print(
        output_path
    )


if __name__ == "__main__":
    main()