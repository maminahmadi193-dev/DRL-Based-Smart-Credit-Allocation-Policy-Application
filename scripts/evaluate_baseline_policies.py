"""Compare simple baseline policies in PortfolioLoanEnv.

Run from the repository root:
    python scripts/evaluate_baseline_policies.py

By default the script uses df_cleaned.csv if it finds it in common project
locations. If no dataset is found, it uses a deterministic synthetic dataset
so the script can still smoke-test the environment. Pass --require-real-data
to prevent that fallback.

Each policy is evaluated with identical environment seeds, so the sampled
applicant cohorts match between policies. The random policy uses a separate RNG
so it does not alter the environment's applicant sampling.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import _bootstrap  # noqa: F401
from portfolio_loan_env import PortfolioLoanEnv
from portfolio_experiment_utils import (
    calibrate_hazard_intercept_for_population,
    load_split_and_scaler,
)


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

# All fields needed by PortfolioLoanEnv to construct CustomerState.
REQUIRED_RAW_COLUMNS = [
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
]

POLICIES = (
    "no_loan",
    "fixed_100M",
    "highest_affordable",
    "random_affordable",
)

METRICS = (
    "total_reward",
    "social_reward",
    "risk_penalty",
    "default_penalty",
    "financial_pnl",
    "interest_collected",
    "net_default_loss",
    "expected_default_count",
    "recognized_default_count",
    "realized_default_rate",
    "contract_term_pd",
    "new_applicants_seen",
    "reapplications_seen",
    "customers_with_reapplications",
    "missed_installment_count",
    "delayed_payment_amount_scheduled",
    "delayed_payment_amount_collected",
    "pending_delayed_receivables",
    "expected_delayed_installments",
    "principal_collected",
    "applicants_seen",
    "loans_granted",
    "loan_approval_rate",
    "total_disbursed",
    "average_loan_size",
    "invalid_actions",
    "final_cash",
    "ending_active_contracts",
    "final_expected_outstanding_principal",
    "ending_month",
)


def make_synthetic_dataset(n: int = 500, seed: int = 42):
    """Create a reproducible dataset compatible with the portfolio environment."""
    rng = np.random.default_rng(seed)
    income = np.linspace(10_000_000, 80_000_000, n)
    debt = np.linspace(0, 50_000_000, n)
    loan_counts = np.arange(n) % 5
    origin_amount = np.where(
        loan_counts > 0, np.maximum(debt, 1.0) * 1.5, 0.0
    )
    debt_ratio = np.divide(
        debt,
        origin_amount,
        out=np.zeros(n, dtype=float),
        where=origin_amount > 0,
    )
    debt_per_loan = np.divide(
        debt,
        loan_counts,
        out=np.zeros(n, dtype=float),
        where=loan_counts > 0,
    )
    risk_min = np.linspace(10, 50, n)

    df = pd.DataFrame({
        "MountlyIncome": income,
        "IncomeScore": np.linspace(20, 90, n),
        "JobScore": np.linspace(30, 95, n) + rng.normal(0, 0.1, n),
        "AssetScore": np.linspace(10, 85, n),
        "TotalCustomerScore": np.linspace(20, 90, n),
        "TotalFacilities": np.linspace(0, 50_000_000, n),
        "DebtAmount": debt,
        "OriginAmount": origin_amount,
        "LoanCounts": loan_counts,
        "DebtRatio": debt_ratio,
        "DebtPerLoan": debt_per_loan,
        "Has_No_History": (np.arange(n) % 3 == 0).astype(int),
        "Employed": (np.arange(n) % 4 != 0).astype(int),
        "Score_Min": np.linspace(400, 600, n),
        "Score_Max": np.linspace(500, 700, n),
        "Risk_Min": risk_min,
        "Risk_Max": risk_min + 5,
    })
    df["Risk"] = (df["Risk_Min"] + df["Risk_Max"]) / 2.0
    return df


def find_default_dataset(project_root: Path) -> Path | None:
    env_path = os.environ.get("PORTFOLIO_BASELINE_DATA")
    candidates = []
    if env_path:
        candidates.append(Path(env_path).expanduser())
    candidates.extend([
        project_root / "df_cleaned.csv",
        project_root / "notebooks" / "df_cleaned.csv",
        project_root.parent / "notebooks" / "df_cleaned.csv",
        Path.cwd() / "df_cleaned.csv",
        Path.cwd() / "notebooks" / "df_cleaned.csv",
    ])
    seen = set()
    for path in candidates:
        try:
            resolved = path.resolve()
        except OSError:
            resolved = path
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.is_file():
            return resolved
    return None


def load_data(data_path: str | None, require_real_data: bool, project_root: Path):
    source_path = Path(data_path).expanduser() if data_path else None
    if source_path is None:
        source_path = find_default_dataset(project_root)
    elif not source_path.is_absolute():
        source_path = (Path.cwd() / source_path).resolve()

    if source_path is not None and source_path.is_file():
        df = pd.read_csv(source_path)
        missing = sorted(set(REQUIRED_RAW_COLUMNS) - set(df.columns))
        if missing:
            raise ValueError(
                f"Dataset {source_path} is missing required columns: {missing}"
            )
        # The environment defines customer risk as the midpoint of Risk_Min/Max.
        # Match that exact convention when fitting the observation scaler.
        for column in REQUIRED_RAW_COLUMNS:
            df[column] = pd.to_numeric(df[column], errors="coerce")
        df["Risk"] = (df["Risk_Min"] + df["Risk_Max"]) / 2.0
        df = df.replace([np.inf, -np.inf], np.nan)
        # Blank monthly-income cells in this dataset represent no recorded salary.
        df["MountlyIncome"] = df["MountlyIncome"].fillna(0.0)
        non_income_required = [
            column for column in REQUIRED_RAW_COLUMNS
            if column != "MountlyIncome"
        ]
        df = df.dropna(subset=non_income_required + ["Risk"]).reset_index(drop=True)
        if df.empty:
            raise ValueError(f"No complete usable rows found in {source_path}")
        scaler = StandardScaler().fit(df[FEATURE_COLUMNS])
        return df, scaler, f"real:{source_path}"

    if require_real_data:
        raise FileNotFoundError(
            "No df_cleaned.csv was found. Pass --data PATH or set "
            "PORTFOLIO_BASELINE_DATA to your dataset."
        )

    df = make_synthetic_dataset()
    scaler = StandardScaler().fit(df[FEATURE_COLUMNS])
    return df, scaler, "synthetic_demo_data"


def choose_action(policy: str, env: PortfolioLoanEnv, rng: np.random.Generator) -> int:
    """Return an action index that is affordable under the current budget."""
    if env.phase != "allocation":
        return 0

    if policy == "no_loan":
        return 0

    amounts = env.ACTION_AMOUNTS
    budget = env.portfolio.available_budget
    affordable = [
        idx for idx, amount in enumerate(amounts)
        if amount <= budget + env.portfolio.EPSILON
    ]

    # Action 0 is always affordable because the budget is non-negative.
    if policy == "fixed_100M":
        fixed_index = 1
        return fixed_index if fixed_index in affordable else 0
    if policy == "highest_affordable":
        return max(affordable)
    if policy == "random_affordable":
        return int(rng.choice(affordable))
    raise ValueError(f"Unknown policy: {policy}")


def evaluate_one_episode(policy: str, episode: int, seed: int, df: pd.DataFrame,
                         scaler: StandardScaler, args, data_source: str) -> dict:
    env = PortfolioLoanEnv(
        raw_df=df,
        scaler=scaler,
        initial_budget=args.initial_budget,
        monthly_capital_injection=args.monthly_capital_injection,
        applicants_per_month=args.applicants_per_month,
        min_applicants_per_month=args.min_applicants_per_month,
        max_applicants_per_month=args.max_applicants_per_month,
        reapplication_share=args.reapplication_share,
        reapplication_cooldown_months=args.reapplication_cooldown_months,
        origination_months=args.origination_months,
        loan_term_months=args.loan_term_months,
        annual_interest_rate=args.annual_interest_rate,
        default_collection_lag_months=args.default_collection_lag_months,
        default_dpd_threshold_days=args.default_dpd_threshold_days,
        hazard_slope=args.hazard_slope,
        hazard_center=args.hazard_center,
        max_hazard=args.max_hazard,
        hazard_intercept=args.hazard_intercept,
        social_reward_weight=args.social_reward_weight,
        financial_reward_weight=args.financial_reward_weight,
        financial_reward_scale=args.financial_reward_scale,
        invalid_action_penalty=args.invalid_action_penalty,
    )
    env.reset(seed=seed)
    # Keep policy randomness separate from the environment's seeded RNG.
    rng = np.random.default_rng(seed + 100_003)
    total_reward = 0.0
    social_reward = 0.0
    risk_penalty = 0.0
    default_penalty = 0.0
    financial_pnl = 0.0
    interest_collected = 0.0
    net_default_loss = 0.0
    principal_collected = 0.0
    invalid_actions = 0

    max_steps = (
        env.max_applicants_per_month * args.origination_months
        + args.origination_months
        + args.loan_term_months + env.portfolio.max_delay_months + 10
    )
    terminated = False
    truncated = False

    for _ in range(max_steps):
        action = choose_action(policy, env, rng)
        _, reward, terminated, truncated, info = env.step(action)

        total_reward += float(reward)
        social_reward += float(info.get("social_reward", 0.0))
        risk_penalty += float(info.get("risk_penalty", 0.0))
        default_penalty += float(info.get("default_penalty", 0.0))
        financial_pnl += float(info.get("financial_pnl", 0.0))
        interest_collected += float(info.get("interest_collected", 0.0))
        net_default_loss += float(info.get("net_default_loss", 0.0))
        principal_collected += float(info.get("principal_collected", 0.0))
        invalid_actions += int(bool(info.get("invalid_action", False)))

        if terminated or truncated:
            break
    else:
        env.close()
        raise RuntimeError(
            f"Episode exceeded max_steps={max_steps}; policy={policy}, seed={seed}"
        )

    portfolio = env.portfolio
    loans_granted = len(portfolio.contracts)
    applicants_seen = env.total_applicants_seen
    total_disbursed = float(portfolio.total_disbursed)

    result = {
        "policy": policy,
        "episode": episode,
        "seed": seed,
        "data_source": data_source,
        "dataset_rows": len(df),
        "reapplication_share": float(args.reapplication_share),
        "reapplication_cooldown_months": int(args.reapplication_cooldown_months),
        "total_reward": total_reward,
        "social_reward": social_reward,
        "risk_penalty": risk_penalty,
        "default_penalty": default_penalty,
        "financial_pnl": financial_pnl,
        "interest_collected": float(portfolio.cumulative_interest_collected),
        "net_default_loss": float(portfolio.cumulative_net_default_loss),
        "expected_default_count": float(portfolio.cumulative_expected_defaults),
        "recognized_default_count": float(portfolio.cumulative_recognized_defaults),
        "realized_default_rate": (
            float(portfolio.cumulative_recognized_defaults / loans_granted)
            if loans_granted else 0.0
        ),
        "contract_term_pd": (
            float(np.mean([contract.default_probability_over_term for contract in portfolio.contracts]))
            if portfolio.contracts else 0.0
        ),
        "missed_installment_count": float(portfolio.cumulative_missed_installments),
        "delayed_payment_amount_scheduled": float(portfolio.cumulative_delayed_payment_amount_scheduled),
        "delayed_payment_amount_collected": float(portfolio.cumulative_delayed_payment_amount_collected),
        "pending_delayed_receivables": float(portfolio.pending_delayed_receivables),
        "expected_delayed_installments": float(portfolio.cumulative_expected_delayed_installments),
        "principal_collected": principal_collected,
        "applicants_seen": int(applicants_seen),
        "new_applicants_seen": int(env.new_applicants_seen),
        "reapplications_seen": int(env.reapplications_seen),
        "customers_with_reapplications": int(len(env.customers_with_reapplications)),
        "loans_granted": loans_granted,
        "loan_approval_rate": loans_granted / applicants_seen if applicants_seen else 0.0,
        "total_disbursed": total_disbursed,
        "average_loan_size": total_disbursed / loans_granted if loans_granted else 0.0,
        "invalid_actions": invalid_actions,
        "final_cash": float(portfolio.available_budget),
        "ending_active_contracts": int(portfolio.active_contract_count),
        "final_expected_outstanding_principal": float(portfolio.expected_outstanding_principal),
        "ending_month": int(portfolio.month),
    }
    env.close()
    return result


def summarize(episodes_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for policy, group in episodes_df.groupby("policy", sort=False):
        row = {"policy": policy, "episodes": len(group)}
        for metric in METRICS:
            values = pd.to_numeric(group[metric], errors="coerce")
            row[f"mean_{metric}"] = float(values.mean())
            row[f"std_{metric}"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        # Keep the arithmetic mean of episode-level rates, but also report a
        # pooled rate computed from total recognized defaults / total loans.
        # These values differ when episodes have different numbers of loans.
        episode_rates = pd.to_numeric(group["realized_default_rate"], errors="coerce")
        mean_episode_rate = float(episode_rates.mean()) if len(episode_rates) else 0.0
        total_loans = float(pd.to_numeric(group["loans_granted"], errors="coerce").sum())
        total_defaults = float(pd.to_numeric(group["recognized_default_count"], errors="coerce").sum())
        row["mean_episode_realized_default_rate"] = mean_episode_rate
        row["pooled_realized_default_rate"] = (
            total_defaults / total_loans if total_loans > 0 else 0.0
        )
        # Retain the previous column name for compatibility with existing CSV consumers.
        row["mean_realized_default_rate"] = mean_episode_rate
        rows.append(row)
    return pd.DataFrame(rows)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare four simple baseline policies using PortfolioLoanEnv."
    )
    parser.add_argument("--data", default=None, help="Path to df_cleaned.csv (autodetected if omitted).")
    parser.add_argument("--require-real-data", action="store_true",
                        help="Exit instead of using synthetic data if df_cleaned.csv is not found.")
    parser.add_argument("--episodes", type=int, default=10, help="Number of matched-seed episodes per policy.")
    parser.add_argument("--seed", type=int, default=42, help="Base seed; all policies use matching episode seeds.")
    parser.add_argument(
        "--split",
        choices=["full", "validation", "test"],
        default="full",
        help="Evaluate on all usable rows (full) or the fixed train/validation/test split.",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="Seed used by the same 70/10/20 split as DQN training/evaluation.",
    )
    parser.add_argument("--initial-budget", type=float, default=1_000_000_000)
    parser.add_argument("--monthly-capital-injection", type=float, default=0.0)
    parser.add_argument("--applicants-per-month", type=int, default=None, help="Legacy fixed cohort size. When supplied alone, monthly cohort size is fixed.")
    parser.add_argument("--min-applicants-per-month", type=int, default=None, help="Minimum monthly cohort size (inclusive); default range is 30..70.")
    parser.add_argument("--max-applicants-per-month", type=int, default=None, help="Maximum monthly cohort size (inclusive).")
    parser.add_argument("--reapplication-share", type=float, default=0.10, help="Target share of monthly applicants who reapply after cooldown; a scenario assumption.")
    parser.add_argument("--reapplication-cooldown-months", type=int, default=3, help="Minimum months since the customer's previous application before reapplying.")
    parser.add_argument("--origination-months", type=int, default=12)
    parser.add_argument("--loan-term-months", type=int, choices=[12, 18, 24, 36], default=18)
    parser.add_argument("--annual-interest-rate", type=float, default=0.04, help="Nominal annual interest rate; converted to monthly rate internally.")
    parser.add_argument("--default-collection-lag-months", type=int, choices=range(4, 13), default=4)
    parser.add_argument("--default-dpd-threshold-days", type=int, default=90)
    parser.add_argument("--hazard-slope", type=float, default=0.08)
    parser.add_argument("--hazard-center", type=float, default=60.0)
    parser.add_argument("--max-hazard", type=float, default=0.005)
    parser.add_argument("--target-annual-default-rate", type=float, default=0.013)
    parser.add_argument("--reference-loan-amount", type=float, default=100_000_000)
    parser.add_argument("--social-reward-weight", type=float, default=1.0)
    parser.add_argument("--financial-reward-weight", type=float, default=1.0)
    parser.add_argument("--financial-reward-scale", type=float, default=100_000_000)
    parser.add_argument("--invalid-action-penalty", type=float, default=0.25)
    parser.add_argument("--output-dir", default=None, help="Output directory; defaults to models/results.")
    return parser.parse_args()


def main():
    args = parse_args()

    # Default to an illustrative random range; a legacy explicit fixed
    # count keeps old baseline commands reproducible.
    if args.min_applicants_per_month is None and args.max_applicants_per_month is None:
        if args.applicants_per_month is None:
            args.min_applicants_per_month, args.max_applicants_per_month = 30, 70
        else:
            args.min_applicants_per_month = args.applicants_per_month
            args.max_applicants_per_month = args.applicants_per_month
    elif args.min_applicants_per_month is None or args.max_applicants_per_month is None:
        raise ValueError("Set both --min-applicants-per-month and --max-applicants-per-month.")

    if args.applicants_per_month is None:
        args.applicants_per_month = (args.min_applicants_per_month + args.max_applicants_per_month) // 2

    if args.episodes < 1:
        raise ValueError("--episodes must be >= 1")
    if args.applicants_per_month < 1 or args.origination_months < 1 or args.loan_term_months < 1:
        raise ValueError("Applicant count and both month horizons must be >= 1")
    if args.min_applicants_per_month < 1:
        raise ValueError("--min-applicants-per-month must be >= 1")
    if args.max_applicants_per_month < args.min_applicants_per_month:
        raise ValueError("--max-applicants-per-month must be >= min-applicants-per-month")
    if not np.isfinite(args.reapplication_share) or not 0.0 <= args.reapplication_share <= 1.0:
        raise ValueError("--reapplication-share must be between 0 and 1.")
    if args.reapplication_cooldown_months < 1:
        raise ValueError("--reapplication-cooldown-months must be >= 1.")
    if not np.isfinite(args.annual_interest_rate) or args.annual_interest_rate < 0:
        raise ValueError("--annual-interest-rate must be finite and non-negative")
    if not 4 <= args.default_collection_lag_months <= 12:
        raise ValueError("--default-collection-lag-months must be between 4 and 12")
    if args.default_dpd_threshold_days < 1:
        raise ValueError("--default-dpd-threshold-days must be >= 1")
    if not np.isfinite(args.hazard_slope) or args.hazard_slope < 0:
        raise ValueError("--hazard-slope must be finite and non-negative")
    if not np.isfinite(args.hazard_center):
        raise ValueError("--hazard-center must be finite")
    if not np.isfinite(args.max_hazard) or args.max_hazard <= 0:
        raise ValueError("--max-hazard must be finite and positive for calibration.")
    if not np.isfinite(args.target_annual_default_rate) or not 0 < args.target_annual_default_rate < 1:
        raise ValueError("--target-annual-default-rate must be between 0 and 1.")
    max_attainable_pd = -np.expm1(-args.max_hazard * 12)
    if args.target_annual_default_rate >= max_attainable_pd:
        raise ValueError(
            "--target-annual-default-rate must be below maximum attainable 12-month PD "
            f"({max_attainable_pd:.6%})."
        )
    if not np.isfinite(args.reference_loan_amount) or args.reference_loan_amount <= 0:
        raise ValueError("--reference-loan-amount must be finite and positive.")

    project_root = Path(__file__).resolve().parents[1]
    if args.split == "full":
        df, scaler, data_source = load_data(
            args.data, args.require_real_data, project_root
        )
        calibration_population = df
        split_info = None
    else:
        # Use the identical deterministic split and training-fitted scaler as DQN.
        # Calibration is still performed on the full usable feature population,
        # matching train_portfolio_dqn.py; only the evaluated applicants are held out.
        train_raw, validation_raw, test_raw, scaler, split_info = load_split_and_scaler(
            data_path=args.data,
            split_seed=args.split_seed,
        )
        df = validation_raw if args.split == "validation" else test_raw
        calibration_population = pd.concat(
            [train_raw, validation_raw, test_raw], ignore_index=True
        )
        data_source = (
            f"real:{split_info['dataset_path']} "
            f"(split={args.split}, split_seed={args.split_seed})"
        )

    hazard_calibration = calibrate_hazard_intercept_for_population(
        raw_df=calibration_population,
        target_annual_default_rate=args.target_annual_default_rate,
        reference_loan_amount=args.reference_loan_amount,
        calibration_horizon_months=12,
        max_hazard=args.max_hazard,
        hazard_slope=args.hazard_slope,
        hazard_center=args.hazard_center,
    )
    args.hazard_intercept = float(hazard_calibration["hazard_intercept"])

    output_dir = Path(args.output_dir) if args.output_dir else project_root / "models" / "results"
    if not output_dir.is_absolute():
        output_dir = Path.cwd() / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 76)
    print("PORTFOLIO BASELINE POLICY COMPARISON")
    print("=" * 76)
    print(f"Data source              : {data_source}")
    print(f"Evaluation rows          : {len(df):,}")
    if args.split != "full":
        print(f"Evaluation split         : {args.split} (split seed={args.split_seed})")
    print(f"Episodes per policy      : {args.episodes}")
    print(f"Initial budget           : {args.initial_budget:,.0f}")
    print(f"Applicants/month range   : {args.min_applicants_per_month}..{args.max_applicants_per_month} (uniform, inclusive)")
    print(f"Origination months       : {args.origination_months}")
    print(f"Loan term (months)       : {args.loan_term_months}")
    print(f"Annual nominal rate       : {args.annual_interest_rate:.2%}")
    print("Interest charge months   : 1, 13, 25, ... (annual rate on outstanding principal)")
    print(f"Maximum monthly hazard   : {args.max_hazard:.6f}")
    print(f"Target annual default rate : {args.target_annual_default_rate:.4%}")
    print(f"Reference loan amount    : {args.reference_loan_amount:,.0f}")
    print(f"Calibrated reference annual PD  : {hazard_calibration['calibrated_reference_annual_default_rate']:.4%}")
    print(f"Calibrated hazard int.   : {args.hazard_intercept:.8f}")
    print(f"Default threshold        : >{args.default_dpd_threshold_days} days past due")
    print(f"Missed-payment collection: {args.default_collection_lag_months} months after due date")
    print("Payment behavior         : complete installment paid or missed")
    print("Permanent credit loss    : disabled (missed installments are eventually collected in full)")
    print(f"Social reward weight     : {args.social_reward_weight}")
    print("Reward design            : amount-sensitive access rewards")
    print(f"Financial reward weight  : {args.financial_reward_weight}")
    if args.annual_interest_rate == 0.0 and args.max_hazard == 0.0:
        print("NOTE: Interest and missed-installment events are both disabled for this run.")
    if data_source == "synthetic_demo_data":
        print("NOTE: Results use synthetic data, not the real cleaned dataset.")
    print("-" * 76)

    rows = []
    for episode in range(args.episodes):
        episode_seed = args.seed + episode
        for policy in POLICIES:
            print(f"Running episode {episode + 1}/{args.episodes} | "
                  f"seed={episode_seed} | policy={policy}")
            rows.append(evaluate_one_episode(
                policy=policy,
                episode=episode + 1,
                seed=episode_seed,
                df=df,
                scaler=scaler,
                args=args,
                data_source=data_source,
            ))

    episodes_df = pd.DataFrame(rows)
    summary_df = summarize(episodes_df)
    episode_path = output_dir / "baseline_policy_comparison_episodes.csv"
    summary_path = output_dir / "baseline_policy_comparison_summary.csv"
    config_path = output_dir / "baseline_policy_comparison_config.json"

    episodes_df.to_csv(episode_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    config = vars(args).copy()
    config.update({
        "data_source": data_source,
        "dataset_rows": len(df),
        "evaluation_split": args.split,
        "split_seed": args.split_seed if args.split != "full" else None,
        "split_info": split_info,
        "hazard_intercept": args.hazard_intercept,
        "hazard_calibration": hazard_calibration,
        "policies": list(POLICIES),
        "metric_definitions": {
            "total_reward": "Sum of the environment rewards, including scaled financial reward.",
            "social_reward": "Sum of the environment's reported social reward component.",
            "financial_pnl": "Interest collected minus net default loss, in currency units.",
            "net_default_loss": "Zero under the full-collection assumption; delayed receipts are tracked separately.",
            "realized_default_rate": "Per-episode recognized defaults divided by loans granted in that episode.",
            "mean_episode_realized_default_rate": "Arithmetic mean of episode-level realized default rates; each episode has equal weight.",
            "pooled_realized_default_rate": "Total recognized defaults divided by total loans granted across all episodes.",
            "final_cash": "Available cash after all outstanding loans have been serviced.",
        },
    })
    with config_path.open("w", encoding="utf-8") as file:
        json.dump(config, file, ensure_ascii=False, indent=2)

    display_columns = [
        "policy",
        "mean_total_reward",
        "mean_social_reward",
        "mean_financial_pnl",
        "mean_net_default_loss",
        "mean_recognized_default_count",
        "mean_episode_realized_default_rate",
        "pooled_realized_default_rate",
        "mean_contract_term_pd",
        "mean_missed_installment_count",
        "mean_delayed_payment_amount_scheduled",
        "mean_delayed_payment_amount_collected",
        "mean_pending_delayed_receivables",
        "mean_applicants_seen",
        "mean_loans_granted",
        "mean_total_disbursed",
        "mean_final_cash",
        "mean_ending_active_contracts",
        "mean_invalid_actions",
    ]
    print("\nSUMMARY (mean across matched-seed episodes)")
    print(summary_df[display_columns].to_string(index=False, float_format=lambda x: f"{x:,.3f}"))
    print("\nSaved:")
    print(f"  Per-episode results : {episode_path}")
    print(f"  Policy summary      : {summary_path}")
    print(f"  Run configuration   : {config_path}")
    print("\nInterpretation reminder: these are heuristic baselines, not an optimal policy.")


if __name__ == "__main__":
    main()
