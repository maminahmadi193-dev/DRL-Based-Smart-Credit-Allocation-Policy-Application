import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from loan_env import LoanEnv


# ============================================================
# Configuration
# ============================================================

PATH = r"C:\\Users\\m.ahmadi\\Desktop\\FinalProject\\notebooks\\df_cleaned.csv"

N_CUSTOMERS = 10
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

    # Same split as train.py
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
    # Fit scaler ONLY on train
    # --------------------------------------------------------

    train_features = train_raw.copy()

    train_features["Risk"] = (
        train_features["Risk_Min"]
        + train_features["Risk_Max"]
    ) / 2

    scaler = StandardScaler()

    scaler.fit(
        train_features[FEATURE_COLUMNS]
    )

    return (
        train_raw,
        val_raw,
        test_raw,
        scaler
    )


# ============================================================
# Analyze one customer
# ============================================================

def analyze_customer(
    env,
    row,
    customer_number
):

    print()
    print("=" * 90)
    print(
        f"CUSTOMER {customer_number}"
    )
    print("=" * 90)

    print(
        f"Risk              : "
        f"{((row['Risk_Min'] + row['Risk_Max']) / 2):.6f}"
    )

    print(
        f"TotalFacilities   : "
        f"{row['TotalFacilities']}"
    )

    print(
        f"TotalCustomerScore: "
        f"{row['TotalCustomerScore']}"
    )

    print(
        f"Has_No_History    : "
        f"{bool(row['Has_No_History'])}"
    )

    print()

    header = (
        f"{'Action':>8} | "
        f"{'Social':>10} | "
        f"{'RiskPen':>10} | "
        f"{'PD':>10} | "
        f"{'ExpDefPen':>10} | "
        f"{'Reward':>10}"
    )

    print(header)
    print("-" * len(header))

    rows = []

    # --------------------------------------------------------
    # Evaluate every action
    # --------------------------------------------------------

    for action_index, action_name in enumerate(
        ACTION_NAMES
    ):

        # Fresh customer for every action
        customer = env._row_to_customer_state(
            row
        )

        # ----------------------------------------------------
        # Previous state
        # ----------------------------------------------------

        previous_customer = customer.copy()

        # ----------------------------------------------------
        # Run simulator
        # ----------------------------------------------------

        outcome = env.simulator.simulate(
            customer=customer,
            approved_unsecured_amount=
                ACTIONS[action_index]
        )

        # ----------------------------------------------------
        # Social reward
        # ----------------------------------------------------

        social_reward = (
            env.reward_engine._social_reward(
                previous_customer=
                    previous_customer,
                approved_unsecured_amount=
                    ACTIONS[action_index]
            )
        )

        # ----------------------------------------------------
        # Risk penalty
        # ----------------------------------------------------

        risk_penalty = (
            env.reward_engine._risk_penalty(
                previous_customer=
                    previous_customer,
                outcome=outcome
            )
        )

        # ----------------------------------------------------
        # Expected default penalty
        # ----------------------------------------------------

        expected_default_penalty = (
            env.reward_engine.default_cost
            * outcome.probability_of_default
        )

        # ----------------------------------------------------
        # Expected reward
        # ----------------------------------------------------

        expected_reward = (
            social_reward
            - risk_penalty
            - expected_default_penalty
        )

        rows.append({
            "action":
                action_name,

            "approved_amount":
                ACTIONS[action_index],

            "social_reward":
                social_reward,

            "risk_penalty":
                risk_penalty,

            "pd":
                outcome.probability_of_default,

            "expected_default_penalty":
                expected_default_penalty,

            "expected_reward":
                expected_reward,

            "old_risk":
                previous_customer.risk,

            "new_risk":
                outcome.next_state.risk,
        })

        print(
            f"{action_name:>8} | "
            f"{social_reward:10.6f} | "
            f"{risk_penalty:10.6f} | "
            f"{outcome.probability_of_default:10.6f} | "
            f"{expected_default_penalty:10.6f} | "
            f"{expected_reward:10.6f}"
        )

    # --------------------------------------------------------
    # Best action
    # --------------------------------------------------------

    rewards = np.array([
        r["expected_reward"]
        for r in rows
    ])

    best_index = int(
        np.argmax(rewards)
    )

    print()
    print(
        f"BEST ACTION: "
        f"{ACTION_NAMES[best_index]}"
    )

    print(
        f"BEST EXPECTED REWARD: "
        f"{rewards[best_index]:.6f}"
    )

    return rows


# ============================================================
# Main
# ============================================================

def main():

    (
        _,
        val_raw,
        _,
        scaler
    ) = load_data()

    env = LoanEnv(
        val_raw,
        scaler
    )

    rng = np.random.default_rng(
        SEED
    )

    n_customers = min(
        N_CUSTOMERS,
        len(val_raw)
    )

    selected_indices = rng.choice(
        len(val_raw),
        size=n_customers,
        replace=False
    )

    all_results = []

    for i, idx in enumerate(
        selected_indices,
        start=1
    ):

        row = val_raw.iloc[idx]

        customer_results = analyze_customer(
            env=env,
            row=row,
            customer_number=i
        )

        for result in customer_results:

            result["customer_index"] = (
                int(idx)
            )

            all_results.append(
                result
            )

    # ========================================================
    # Save results
    # ========================================================

    results_df = pd.DataFrame(
        all_results
    )

    output_path = (
        "models/results/"
        "reward_components_analysis.csv"
    )

    results_df.to_csv(
        output_path,
        index=False
    )

    print()
    print("=" * 90)
    print("RESULTS SAVED")
    print("=" * 90)
    print(output_path)


if __name__ == "__main__":
    main()