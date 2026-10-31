"""Diagnose immediate social reward by candidate loan amount.

Run from the repository root with the real dataset available:
    python scripts/analyze_immediate_reward_by_amount.py --split test
    python scripts/analyze_immediate_reward_by_amount.py --split full

This isolates RewardEngine._social_reward from future cash flows and from the
explicit risk/default penalties that are not currently wired into the
portfolio environment's allocation reward. It answers whether larger loans
actually produce a larger immediate social reward for the observed population.
"""

import _bootstrap  # noqa: F401

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from customer_state import CustomerState
from portfolio_experiment_utils import load_split_and_scaler, project_root
from reward_engine import RewardEngine


ACTION_AMOUNTS = (0, 100_000_000, 200_000_000, 300_000_000, 400_000_000, 500_000_000)
POSITIVE_AMOUNTS = ACTION_AMOUNTS[1:]
REWARD_MODE = "amount_sensitive"
REWARD_ENGINE = RewardEngine()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare current immediate social reward across loan amounts."
    )
    parser.add_argument("--data", default=None, help="Path to df_cleaned.csv.")
    parser.add_argument(
        "--split",
        choices=["full", "train", "validation", "test"],
        default="test",
        help="Population to analyze. Defaults to the held-out test split.",
    )
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        default="models/results/reward_diagnostics",
        help="Directory for applicant-level and summary CSV files.",
    )
    return parser.parse_args()


def row_to_customer(row):
    risk_min = float(row["Risk_Min"])
    risk_max = float(row["Risk_Max"])
    return CustomerState(
        monthly_income=float(row["MountlyIncome"]),
        income_score=float(row["IncomeScore"]),
        job_score=float(row["JobScore"]),
        asset_score=float(row["AssetScore"]),
        total_customer_score=float(row["TotalCustomerScore"]),
        total_facilities=float(row["TotalFacilities"]),
        debt=float(row["DebtAmount"]),
        origin_amount=float(row["OriginAmount"]),
        loan_count=int(row["LoanCounts"]),
        debt_ratio=float(row["DebtRatio"]),
        debt_per_loan=float(row["DebtPerLoan"]),
        has_no_history=bool(row["Has_No_History"]),
        employed=bool(row["Employed"]),
        score_min=float(row["Score_Min"]),
        score_max=float(row["Score_Max"]),
        risk_min=risk_min,
        risk_max=risk_max,
        risk=(risk_min + risk_max) / 2.0,
    )


def summarize_mode(frame, mode):
    reward_cols = [f"{mode}_reward_{amount}" for amount in POSITIVE_AMOUNTS]
    values = frame[reward_cols].to_numpy(dtype=float)
    amounts = np.asarray(POSITIVE_AMOUNTS, dtype=np.int64)
    max_values = values.max(axis=1)
    tolerance = 1e-10
    maximizing = values >= (max_values[:, None] - tolerance)

    # When several amounts yield the same maximum, select the smallest one.
    # This is an explicit resource-conserving tie-break, not the agent's policy.
    smallest_max_index = maximizing.argmax(axis=1)
    smallest_max_amount = amounts[smallest_max_index]
    reward_100 = frame[f"{mode}_reward_{100_000_000}"].to_numpy(dtype=float)
    larger_than_100 = values[:, 1:] > (reward_100[:, None] + tolerance)
    all_positive_equal = np.all(
        np.abs(values - values[:, [0]]) <= tolerance,
        axis=1,
    )

    rows = []
    for amount, column in zip(POSITIVE_AMOUNTS, reward_cols):
        reward = frame[column].to_numpy(dtype=float)
        rows.append({
            "reward_mode": mode,
            "action_amount": amount,
            "mean_immediate_social_reward": float(np.mean(reward)),
            "median_immediate_social_reward": float(np.median(reward)),
            "p25_immediate_social_reward": float(np.quantile(reward, 0.25)),
            "p75_immediate_social_reward": float(np.quantile(reward, 0.75)),
            "mean_increment_vs_100M": float(np.mean(reward - reward_100)),
            "share_customers_reward_strictly_higher_than_100M": float(
                np.mean(reward > reward_100 + tolerance)
            ),
            "share_customers_for_which_amount_is_a_maximizer": float(
                np.mean(maximizing[:, list(POSITIVE_AMOUNTS).index(amount)])
            ),
            "share_customers_for_which_amount_is_smallest_maximizer": float(
                np.mean(smallest_max_amount == amount)
            ),
        })

    diagnostics = {
        "mode": mode,
        "customers": int(len(frame)),
        "share_100M_is_a_maximizer_including_ties": float(
            np.mean(maximizing[:, 0])
        ),
        "share_100M_is_the_unique_maximizer": float(
            np.mean(np.sum(maximizing, axis=1) == 1)  # replaced below with amount-specific check
        ),
        "share_any_larger_amount_strictly_beats_100M": float(
            np.mean(larger_than_100.any(axis=1))
        ),
        "share_all_positive_amounts_tie": float(np.mean(all_positive_equal)),
        "smallest_maximizing_amount_mean": float(np.mean(smallest_max_amount)),
    }
    # The number of maximizers being one only indicates a unique maximizer;
    # explicitly check that the unique maximizing amount is 100M.
    diagnostics["share_100M_is_the_unique_maximizer"] = float(
        np.mean(maximizing[:, 0] & (np.sum(maximizing, axis=1) == 1))
    )
    return rows, diagnostics, smallest_max_amount


def main():
    args = parse_args()
    train_df, validation_df, test_df, _, split_info = load_split_and_scaler(
        data_path=args.data,
        split_seed=args.split_seed,
    )

    splits = {
        "train": train_df,
        "validation": validation_df,
        "test": test_df,
        "full": pd.concat([train_df, validation_df, test_df], ignore_index=True),
    }
    data = splits[args.split].reset_index(drop=True).copy()
    data["applicant_row_in_split"] = np.arange(len(data), dtype=int)

    print("=" * 78)
    print("IMMEDIATE REWARD BY LOAN AMOUNT")
    print("=" * 78)
    print(f"Dataset     : {split_info['dataset_path']}")
    print(f"Split       : {args.split}")
    print(f"Split seed  : {args.split_seed}")
    print(f"Rows        : {len(data):,}")
    print("Reward scope: immediate social reward only (no future cash flows, no risk/default penalty)")
    print("-" * 78)

    rewards = np.zeros((len(data), len(ACTION_AMOUNTS)), dtype=float)
    for row_idx, (_, row) in enumerate(data.iterrows()):
        customer = row_to_customer(row)
        for action_idx, amount in enumerate(ACTION_AMOUNTS):
            rewards[row_idx, action_idx] = REWARD_ENGINE._social_reward(
                previous_customer=customer,
                approved_unsecured_amount=amount,
            )

    for action_idx, amount in enumerate(ACTION_AMOUNTS):
        data[f"{REWARD_MODE}_reward_{amount}"] = rewards[:, action_idx]

    summary_rows, diagnostics, smallest_max = summarize_mode(data, REWARD_MODE)
    diagnostics_rows = [diagnostics]
    data[f"{REWARD_MODE}_smallest_maximizing_amount"] = smallest_max

    summary_df = pd.DataFrame(summary_rows)
    diagnostics_df = pd.DataFrame(diagnostics_rows)

    root = project_root()
    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = output_dir / f"immediate_reward_by_amount_{args.split}.csv"
    summary_path = output_dir / f"immediate_reward_by_amount_{args.split}_summary.csv"
    data.to_csv(detail_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    display_cols = [
        "reward_mode",
        "action_amount",
        "mean_immediate_social_reward",
        "median_immediate_social_reward",
        "mean_increment_vs_100M",
        "share_customers_reward_strictly_higher_than_100M",
        "share_customers_for_which_amount_is_smallest_maximizer",
    ]
    print("BY ACTION")
    print(summary_df[display_cols].to_string(
        index=False,
        formatters={
            "action_amount": lambda value: f"{value / 1_000_000:,.0f}M",
            "mean_immediate_social_reward": lambda value: f"{value:.6f}",
            "median_immediate_social_reward": lambda value: f"{value:.6f}",
            "mean_increment_vs_100M": lambda value: f"{value:+.6f}",
            "share_customers_reward_strictly_higher_than_100M": lambda value: f"{value:.2%}",
            "share_customers_for_which_amount_is_smallest_maximizer": lambda value: f"{value:.2%}",
        },
    ))
    print()
    print("POPULATION DIAGNOSTICS")
    for _, row in diagnostics_df.iterrows():
        print("Reward design: amount-sensitive")
        print(f"  100M is a maximizer (ties included)         : {row['share_100M_is_a_maximizer_including_ties']:.2%}")
        print(f"  100M is the unique maximizer                : {row['share_100M_is_the_unique_maximizer']:.2%}")
        print(f"  Some larger amount strictly beats 100M     : {row['share_any_larger_amount_strictly_beats_100M']:.2%}")
        print(f"  All positive amounts tie                    : {row['share_all_positive_amounts_tie']:.2%}")
        print(f"  Mean smallest reward-maximizing amount      : {row['smallest_maximizing_amount_mean'] / 1_000_000:,.1f}M")
    print()
    print("Saved:")
    print(f"  Applicant-level detail: {detail_path}")
    print(f"  Action summary        : {summary_path}")


if __name__ == "__main__":
    main()
