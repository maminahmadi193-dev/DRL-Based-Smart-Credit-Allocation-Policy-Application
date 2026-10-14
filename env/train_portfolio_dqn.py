"""Train DQN on the portfolio-level credit-allocation environment.

Examples from the repository root:
    python env/train_portfolio_dqn.py --run-name portfolio_dqn_legacy --timesteps 5000
    python env/train_portfolio_dqn.py --run-name portfolio_dqn_amount_sensitive --timesteps 5000 --scale-access-rewards-by-coverage

Use the short runs above as pipeline smoke tests first. For a full run, increase
--timesteps after verifying the output and evaluation scripts.
"""

import argparse
import json
import math
from pathlib import Path

import pandas as pd
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback

from portfolio_experiment_utils import (
    calibrate_hazard_intercept_for_population,
    load_split_and_scaler,
    project_root,
)
from portfolio_loan_env import PortfolioLoanEnv


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a DQN agent using PortfolioLoanEnv."
    )
    parser.add_argument("--data", default=None, help="Path to df_cleaned.csv; auto-detected if omitted.")
    parser.add_argument("--run-name", default=None, help="Unique name for this model/run.")
    parser.add_argument(
        "--scale-access-rewards-by-coverage",
        action="store_true",
        help="Use the experimental, amount-sensitive social reward design.",
    )
    parser.add_argument("--timesteps", type=int, default=420_000)
    parser.add_argument("--seed", type=int, default=42, help="DQN and environment seed.")
    parser.add_argument("--split-seed", type=int, default=42, help="Seed for the fixed 70/10/20 data split.")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--buffer-size", type=int, default=420_000)
    parser.add_argument("--learning-starts", type=int, default=1_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--train-freq", type=int, default=1)
    parser.add_argument("--target-update-interval", type=int, default=500)
    parser.add_argument("--exploration-fraction", type=float, default=0.20)
    parser.add_argument("--exploration-final-eps", type=float, default=0.05)
    parser.add_argument("--checkpoint-freq", type=int, default=50_000)
    parser.add_argument("--initial-budget", type=float, default=1_000_000_000)
    parser.add_argument("--monthly-capital-injection", type=float, default=0.0)
    parser.add_argument("--applicants-per-month", type=int, default=None, help="Legacy fixed cohort size. When supplied alone, monthly cohort size is fixed.")
    parser.add_argument("--min-applicants-per-month", type=int, default=None, help="Minimum monthly cohort size (inclusive); default range is 30..70.")
    parser.add_argument("--max-applicants-per-month", type=int, default=None, help="Maximum monthly cohort size (inclusive).")
    parser.add_argument("--origination-months", type=int, default=12)
    parser.add_argument("--loan-term-months", type=int, choices=[12, 18, 24, 36], default=18, help="Fixed loan term used for all contracts in this run.")
    parser.add_argument("--annual-interest-rate", type=float, default=0.04, help="Nominal annual interest rate; converted to monthly rate internally.")
    parser.add_argument(
        "--default-collection-lag-months",
        type=int,
        choices=range(4, 13),
        default=4,
        help="Full collection lag for the one missed installment: 4 months base, 12 months stress.",
    )
    parser.add_argument(
        "--default-dpd-threshold-days",
        type=int,
        default=90,
        help="Recognize default only when a missed installment is more than this many days past due.",
    )
    parser.add_argument("--hazard-slope", type=float, default=0.08)
    parser.add_argument("--hazard-center", type=float, default=60.0)
    parser.add_argument(
        "--max-hazard",
        type=float,
        default=0.005,
        help="Maximum monthly hazard in the risk-to-hazard mapping; must be calibrated before predictive claims.",
    )
    parser.add_argument("--target-term-default-rate", type=float, default=0.013, help="Target cumulative 90+ DPD default probability over the loan term for reference exposure (default: 1.3%).")
    parser.add_argument("--reference-loan-amount", type=float, default=100_000_000, help="Reference loan amount for population calibration.")
    parser.add_argument("--social-reward-weight", type=float, default=1.0)
    parser.add_argument("--financial-reward-weight", type=float, default=1.0)
    parser.add_argument("--financial-reward-scale", type=float, default=100_000_000)
    parser.add_argument("--invalid-action-penalty", type=float, default=0.25)
    parser.add_argument("--output-dir", default="models/portfolio_dqn")
    return parser.parse_args()


def main():
    args = parse_args()

    # Default to an illustrative random range, but preserve the old
    # --applicants-per-month interface as a fixed-size override.
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

    if args.timesteps < 1:
        raise ValueError("--timesteps must be >= 1")
    if args.checkpoint_freq < 0:
        raise ValueError("--checkpoint-freq must be >= 0")
    if args.applicants_per_month < 1 or args.origination_months < 1 or args.loan_term_months < 1:
        raise ValueError("Applicant count and month horizons must be >= 1")
    if args.min_applicants_per_month < 1:
        raise ValueError("--min-applicants-per-month must be >= 1")
    if args.max_applicants_per_month < args.min_applicants_per_month:
        raise ValueError("--max-applicants-per-month must be >= min-applicants-per-month")
    if args.initial_budget <= 0 or args.financial_reward_scale <= 0:
        raise ValueError("Initial budget and financial reward scale must be positive.")
    if not math.isfinite(args.annual_interest_rate) or args.annual_interest_rate < 0:
        raise ValueError("--annual-interest-rate must be finite and non-negative")
    if args.default_dpd_threshold_days < 1:
        raise ValueError("--default-dpd-threshold-days must be >= 1")
    if not 4 <= args.default_collection_lag_months <= 12:
        raise ValueError("--default-collection-lag-months must be between 4 and 12")
    if not math.isfinite(args.hazard_slope) or args.hazard_slope < 0:
        raise ValueError("--hazard-slope must be finite and non-negative")
    if not math.isfinite(args.hazard_center):
        raise ValueError("--hazard-center must be finite")
    if not math.isfinite(args.max_hazard) or args.max_hazard <= 0:
        raise ValueError("--max-hazard must be finite and positive for calibration.")
    if not math.isfinite(args.target_term_default_rate) or not 0 < args.target_term_default_rate < 1:
        raise ValueError("--target-term-default-rate must be between 0 and 1.")
    max_attainable_pd = -math.expm1(-args.max_hazard * args.loan_term_months)
    if args.target_term_default_rate >= max_attainable_pd:
        raise ValueError(
            "--target-term-default-rate must be below the maximum term PD "
            f"({max_attainable_pd:.6%}) allowed by max-hazard and loan term."
        )
    if not math.isfinite(args.reference_loan_amount) or args.reference_loan_amount <= 0:
        raise ValueError("--reference-loan-amount must be finite and positive.")

    root = project_root()
    output_root = Path(args.output_dir).expanduser()
    if not output_root.is_absolute():
        output_root = root / output_root
    output_root.mkdir(parents=True, exist_ok=True)

    run_name = args.run_name
    if not run_name:
        mode_name = (
            "amount_sensitive"
            if args.scale_access_rewards_by_coverage
            else "legacy"
        )
        run_name = f"portfolio_dqn_{mode_name}"

    run_dir = output_root / run_name
    if run_dir.exists() and (run_dir / "model.zip").exists():
        raise FileExistsError(
            f"Model already exists at {run_dir / 'model.zip'}. "
            "Choose a new --run-name to avoid overwriting a previous run."
        )
    run_dir.mkdir(parents=True, exist_ok=True)

    train_raw, validation_raw, test_raw, scaler, split_info = load_split_and_scaler(
        data_path=args.data,
        split_seed=args.split_seed,
    )

    # Calibration uses only observed risk/exposure features, not default labels.
    # Combine the partitions to calibrate to the supplied bank-wide population
    # benchmark while keeping the same fitted parameters for every policy.
    calibration_population = pd.concat(
        [train_raw, validation_raw, test_raw],
        ignore_index=True,
    )
    hazard_calibration = calibrate_hazard_intercept_for_population(
        raw_df=calibration_population,
        target_term_default_rate=args.target_term_default_rate,
        reference_loan_amount=args.reference_loan_amount,
        loan_term_months=args.loan_term_months,
        max_hazard=args.max_hazard,
        hazard_slope=args.hazard_slope,
        hazard_center=args.hazard_center,
    )
    hazard_intercept = float(hazard_calibration["hazard_intercept"])

    env = PortfolioLoanEnv(
        raw_df=train_raw,
        scaler=scaler,
        initial_budget=args.initial_budget,
        monthly_capital_injection=args.monthly_capital_injection,
        applicants_per_month=args.applicants_per_month,
        min_applicants_per_month=args.min_applicants_per_month,
        max_applicants_per_month=args.max_applicants_per_month,
        origination_months=args.origination_months,
        loan_term_months=args.loan_term_months,
        annual_interest_rate=args.annual_interest_rate,
        default_collection_lag_months=args.default_collection_lag_months,
        default_dpd_threshold_days=args.default_dpd_threshold_days,
        hazard_slope=args.hazard_slope,
        hazard_center=args.hazard_center,
        max_hazard=args.max_hazard,
        hazard_intercept=hazard_intercept,
        social_reward_weight=args.social_reward_weight,
        financial_reward_weight=args.financial_reward_weight,
        financial_reward_scale=args.financial_reward_scale,
        invalid_action_penalty=args.invalid_action_penalty,
        scale_access_rewards_by_coverage=args.scale_access_rewards_by_coverage,
    )

    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        gamma=args.gamma,
        train_freq=args.train_freq,
        target_update_interval=args.target_update_interval,
        exploration_fraction=args.exploration_fraction,
        exploration_final_eps=args.exploration_final_eps,
        seed=args.seed,
        verbose=1,
    )

    callbacks = []
    if args.checkpoint_freq > 0:
        checkpoint_dir = run_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        callbacks.append(
            CheckpointCallback(
                save_freq=args.checkpoint_freq,
                save_path=str(checkpoint_dir),
                name_prefix=run_name,
            )
        )

    print("=" * 76)
    print("PORTFOLIO DQN TRAINING")
    print("=" * 76)
    print(f"Dataset                : {split_info['dataset_path']}")
    print(f"Rows (usable)          : {split_info['dataset_rows_used']:,}")
    print(f"Train rows             : {split_info['train_rows']:,}")
    print(f"Validation rows        : {split_info['validation_rows']:,}")
    print(f"Test rows              : {split_info['test_rows']:,}")
    print(f"Run name               : {run_name}")
    print(f"Reward design          : {'amount-sensitive' if args.scale_access_rewards_by_coverage else 'legacy'}")
    print(f"Training timesteps     : {args.timesteps:,}")
    print(f"Seed                   : {args.seed}")
    print(f"Initial budget         : {args.initial_budget:,.0f}")
    print(f"Applicants/month range : {args.min_applicants_per_month}..{args.max_applicants_per_month} (uniform, inclusive)")
    print(f"Origination months     : {args.origination_months}")
    print(f"Loan term              : {args.loan_term_months}")
    print(f"Annual nominal rate    : {args.annual_interest_rate:.2%}")
    print(f"Monthly rate (derived) : {args.annual_interest_rate / 12.0:.6%}")
    print(f"Monthly hazard maximum : {args.max_hazard:.6f}")
    print(f"Target term default rate: {args.target_term_default_rate:.4%}")
    print(f"Reference loan amount  : {args.reference_loan_amount:,.0f}")
    print(f"Calibrated term PD     : {hazard_calibration['calibrated_reference_term_default_rate']:.4%}")
    print(f"Calibrated hazard int. : {hazard_intercept:.8f}")
    print(f"DPD default threshold  : >{args.default_dpd_threshold_days} days")
    print(f"Missed-payment collect : {args.default_collection_lag_months} months after due date")
    print("Payment behavior       : each installment is paid in full or missed in full")
    print("Permanent credit loss  : disabled (the missed installment is eventually collected in full)")
    print("-" * 76)

    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=callbacks if callbacks else None,
            progress_bar=False,
        )
        model_path = run_dir / "model"
        model.save(str(model_path))

        config = vars(args).copy()
        config.update({
            "run_name": run_name,
            "model_path": str(model_path.with_suffix(".zip").resolve()),
            "reward_mode": (
                "amount_sensitive"
                if args.scale_access_rewards_by_coverage
                else "legacy"
            ),
            "environment_class": "PortfolioLoanEnv",
            "environment_model_version": 3,
            "payment_model": "stochastic_full_or_zero_installment",
            "default_definition": "a missed installment is defaulted after > default_dpd_threshold_days",
            "hazard_intercept": hazard_intercept,
            "hazard_calibration": hazard_calibration,
            **split_info,
        })
        config_path = run_dir / "config.json"
        with config_path.open("w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

        print()
        print("Training finished.")
        print(f"Model saved : {model_path.with_suffix('.zip')}")
        print(f"Config saved: {config_path}")
        print("No existing model was overwritten.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
