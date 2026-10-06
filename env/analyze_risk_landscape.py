import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from loan_env import LoanEnv


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
], dtype=np.float64)

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


def load_data():

    df_raw = pd.read_csv(PATH)

    all_indices = np.arange(
        len(df_raw)
    )

    train_idx, temp_idx = train_test_split(
        all_indices,
        test_size=0.30,
        random_state=42,
        shuffle=True
    )

    val_idx, _ = train_test_split(
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

    train_features = train_raw.copy()

    train_features["Risk"] = (
        train_features["Risk_Min"]
        + train_features["Risk_Max"]
    ) / 2

    scaler = StandardScaler()

    scaler.fit(
        train_features[FEATURE_COLUMNS]
    )

    return val_raw, scaler


def analyze_customer(
    env,
    row,
):

    customer = env._row_to_customer_state(
        row
    )

    previous = customer.copy()

    results = []

    for action_index, action_name in enumerate(
        ACTION_NAMES
    ):

        amount = ACTIONS[action_index]

        outcome = env.simulator.simulate(
            customer=previous.copy(),
            approved_unsecured_amount=amount
        )

        new_customer = outcome.next_state

        delta_debt_ratio = (
            new_customer.debt_ratio
            - previous.debt_ratio
        )

        if previous.debt_per_loan > 0:

            delta_debt_per_loan_relative = (
                (
                    new_customer.debt_per_loan
                    - previous.debt_per_loan
                )
                / previous.debt_per_loan
            )

        else:

            delta_debt_per_loan_relative = 0.0

        delta_risk = (
            new_customer.risk
            - previous.risk
        )

        risk_penalty = max(
            0.0,
            delta_risk
        )

        results.append({
            "action": action_name,
            "amount": amount,

            "old_risk":
                previous.risk,

            "new_risk":
                new_customer.risk,

            "delta_risk":
                delta_risk,

            "risk_penalty":
                risk_penalty,

            "old_debt_ratio":
                previous.debt_ratio,

            "new_debt_ratio":
                new_customer.debt_ratio,

            "delta_debt_ratio":
                delta_debt_ratio,

            "old_debt_per_loan":
                previous.debt_per_loan,

            "new_debt_per_loan":
                new_customer.debt_per_loan,

            "delta_debt_per_loan_relative":
                delta_debt_per_loan_relative,
        })

    return results


def main():

    val_raw, scaler = load_data()

    env = LoanEnv(
        val_raw,
        scaler
    )

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

    all_results = []

    for idx in indices:

        row = val_raw.iloc[idx]

        customer_results = analyze_customer(
            env,
            row
        )

        for result in customer_results:

            result["customer_index"] = int(
                idx
            )

            all_results.append(
                result
            )

    df = pd.DataFrame(
        all_results
    )

    # ========================================================
    # Summary
    # ========================================================

    print("=" * 70)
    print("RISK LANDSCAPE ANALYSIS")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Monotonicity of risk penalty
    # --------------------------------------------------------

    increasing_count = 0
    decreasing_count = 0
    non_monotonic_count = 0

    for customer_index, group in (
        df.groupby("customer_index")
    ):

        risks = (
            group
            .sort_values("amount")
            ["risk_penalty"]
            .to_numpy()
        )

        if np.all(
            np.diff(risks) >= -1e-12
        ):

            increasing_count += 1

        elif np.all(
            np.diff(risks) <= 1e-12
        ):

            decreasing_count += 1

        else:

            non_monotonic_count += 1

    print()
    print(
        f"Risk increasing with amount : "
        f"{increasing_count:,} "
        f"({increasing_count / n_customers:.2%})"
    )

    print(
        f"Risk decreasing with amount : "
        f"{decreasing_count:,} "
        f"({decreasing_count / n_customers:.2%})"
    )

    print(
        f"Non-monotonic               : "
        f"{non_monotonic_count:,} "
        f"({non_monotonic_count / n_customers:.2%})"
    )

    # --------------------------------------------------------
    # 2. Zero-origin customers
    # --------------------------------------------------------

    zero_origin_indices = []

    for idx in indices:

        row = val_raw.iloc[idx]

        if row["OriginAmount"] == 0:

            zero_origin_indices.append(
                idx
            )

    print()
    print(
        f"OriginAmount = 0 customers : "
        f"{len(zero_origin_indices):,} "
        f"({len(zero_origin_indices) / n_customers:.2%})"
    )

    # --------------------------------------------------------
    # 3. Correlations with amount
    # --------------------------------------------------------

    correlations = (
        df.groupby("customer_index")
        .apply(
            lambda g: pd.Series({
                "risk_penalty_corr":
                    np.corrcoef(
                        g["amount"],
                        g["risk_penalty"]
                    )[0, 1],

                "debt_ratio_corr":
                    np.corrcoef(
                        g["amount"],
                        g["delta_debt_ratio"]
                    )[0, 1],

                "debt_per_loan_corr":
                    np.corrcoef(
                        g["amount"],
                        g[
                            "delta_debt_per_loan_relative"
                        ]
                    )[0, 1],
            }),
            include_groups=False
        )
    )

    print()
    print("=" * 70)
    print("CORRELATION WITH LOAN AMOUNT")
    print("=" * 70)

    print(
        f"Mean risk-penalty correlation : "
        f"{correlations['risk_penalty_corr'].mean():.4f}"
    )

    print(
        f"Mean debt-ratio correlation   : "
        f"{correlations['debt_ratio_corr'].mean():.4f}"
    )

    print(
        f"Mean debt/loan correlation    : "
        f"{correlations['debt_per_loan_corr'].mean():.4f}"
    )

    # --------------------------------------------------------
    # 4. Save
    # --------------------------------------------------------

    output_path = (
        "models/results/"
        "risk_landscape.csv"
    )

    df.to_csv(
        output_path,
        index=False
    )

    print()
    print(
        f"Saved to: {output_path}"
    )


if __name__ == "__main__":
    main()