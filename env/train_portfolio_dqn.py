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

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback

from portfolio_experiment_utils import load_split_and_scaler, project_root
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
    parser.add_argument("--applicants-per-month", type=int, default=50)
    parser.add_argument("--origination-months", type=int, default=12)
    parser.add_argument("--loan-term-months", type=int, choices=[12, 18, 24, 36], default=18, help="Fixed loan term used for all contracts in this run.")
    parser.add_argument("--annual-interest-rate", type=float, default=0.04, help="Nominal annual interest rate; converted to monthly rate internally.")
    parser.add_argument("--recovery-rate", type=float, default=0.0)
    parser.add_argument("--social-reward-weight", type=float, default=1.0)
    parser.add_argument("--financial-reward-weight", type=float, default=1.0)
    parser.add_argument("--financial-reward-scale", type=float, default=100_000_000)
    parser.add_argument("--invalid-action-penalty", type=float, default=0.25)
    parser.add_argument("--output-dir", default="models/portfolio_dqn")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.timesteps < 1:
        raise ValueError("--timesteps must be >= 1")
    if args.checkpoint_freq < 0:
        raise ValueError("--checkpoint-freq must be >= 0")
    if args.applicants_per_month < 1 or args.origination_months < 1 or args.loan_term_months < 1:
        raise ValueError("Applicant count and month horizons must be >= 1")
    if args.initial_budget <= 0 or args.financial_reward_scale <= 0:
        raise ValueError("Initial budget and financial reward scale must be positive.")
    if not math.isfinite(args.annual_interest_rate) or args.annual_interest_rate < 0:
        raise ValueError("--annual-interest-rate must be finite and non-negative")
    if not 0.0 <= args.recovery_rate <= 1.0:
        raise ValueError("--recovery-rate must be between 0 and 1")

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

    env = PortfolioLoanEnv(
        raw_df=train_raw,
        scaler=scaler,
        initial_budget=args.initial_budget,
        monthly_capital_injection=args.monthly_capital_injection,
        applicants_per_month=args.applicants_per_month,
        origination_months=args.origination_months,
        loan_term_months=args.loan_term_months,
        annual_interest_rate=args.annual_interest_rate,
        recovery_rate=args.recovery_rate,
        social_reward_weight=args.social_reward_weight,
        financial_reward_weight=args.financial_reward_weight,
        financial_reward_scale=args.financial_reward_scale,
        invalid_action_penalty=args.invalid_action_penalty,
        scale_access_rewards_by_coverage=args.scale_access_rewards_by_coverage,
    )

    # The simulator defaults to a monthly hazard of 0.005. PortfolioLoanEnv
    # recreates the simulator on reset, so we keep this experiment on that
    # existing default instead of setting a transient per-reset value.
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
    print(f"Applicants/month       : {args.applicants_per_month}")
    print(f"Origination months     : {args.origination_months}")
    print(f"Loan term              : {args.loan_term_months}")
    print(f"Annual nominal rate    : {args.annual_interest_rate:.2%}")
    print(f"Monthly rate (derived) : {args.annual_interest_rate / 12.0:.6%}")
    print(f"Recovery rate          : {args.recovery_rate:.2%}")
    print("Max monthly hazard     : 0.005 (PortfolioLoanEnv/CustomerSimulator default)")
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
            "max_monthly_hazard": 0.005,
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
