"""Evaluate portfolio DQN models on the same held-out applicants and seeds.

Run from repository root after training both models:
    python env/evaluate_portfolio_dqn.py --run-names portfolio_dqn_legacy portfolio_dqn_amount_sensitive --episodes 20

The evaluator reports action distributions and computes social/total reward
under BOTH reward formulas for each policy, allowing fair comparison even
though the trained agents used different reward designs.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import DQN

from portfolio_experiment_utils import load_split_and_scaler, project_root
from portfolio_loan_env import PortfolioLoanEnv
from reward_engine import RewardEngine


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate one or more portfolio DQN models on common test episodes."
    )
    parser.add_argument(
        "--run-names",
        nargs="+",
        required=True,
        help="Model run folders under models/portfolio_dqn, as created by train_portfolio_dqn.py.",
    )
    parser.add_argument(
        "--checkpoint-paths",
        nargs="+",
        default=None,
        help="Optional model/checkpoint file for each run-name, in the same order.",
    )
    parser.add_argument("--data", default=None, help="Optional override for the dataset path.")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--eval-seed", type=int, default=10_000)
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument("--output-dir", default="models/results/portfolio_dqn")
    return parser.parse_args()


def evaluate_episode(run_name, model, config, raw_df, scaler, episode, seed):
    env = PortfolioLoanEnv(
        raw_df=raw_df,
        scaler=scaler,
        initial_budget=float(config["initial_budget"]),
        monthly_capital_injection=float(config["monthly_capital_injection"]),
        applicants_per_month=int(config.get("applicants_per_month", 50)),
        min_applicants_per_month=config.get("min_applicants_per_month"),
        max_applicants_per_month=config.get("max_applicants_per_month"),
        reapplication_share=float(config.get("reapplication_share", 0.10)),
        reapplication_cooldown_months=int(config.get("reapplication_cooldown_months", 3)),
        origination_months=int(config["origination_months"]),
        loan_term_months=int(config["loan_term_months"]),
        annual_interest_rate=float(config.get("annual_interest_rate", float(config.get("monthly_interest_rate", 0.0)) * 12.0)),
        default_collection_lag_months=int(config.get("default_collection_lag_months", 4)),
        default_dpd_threshold_days=int(config.get("default_dpd_threshold_days", 90)),
        hazard_slope=float(config.get("hazard_slope", 0.08)),
        hazard_center=float(config.get("hazard_center", 60.0)),
        max_hazard=float(config.get("max_hazard", 0.005)),
        hazard_intercept=float(config.get("hazard_intercept", 0.0)),
        social_reward_weight=float(config["social_reward_weight"]),
        financial_reward_weight=float(config["financial_reward_weight"]),
        financial_reward_scale=float(config["financial_reward_scale"]),
        invalid_action_penalty=float(config["invalid_action_penalty"]),
        scale_access_rewards_by_coverage=bool(config["scale_access_rewards_by_coverage"]),
    )

    legacy_engine = RewardEngine(scale_access_rewards_by_coverage=False)
    experimental_engine = RewardEngine(scale_access_rewards_by_coverage=True)

    obs, _ = env.reset(seed=seed)
    total_reward_as_trained = 0.0
    total_legacy_social = 0.0
    total_experimental_social = 0.0
    action_counts = np.zeros(len(env.ACTION_AMOUNTS), dtype=np.int64)
    approved_action_counts = np.zeros(len(env.ACTION_AMOUNTS), dtype=np.int64)
    invalid_actions = 0
    total_risk_penalty = 0.0
    total_recognized_default_penalty = 0.0

    max_steps = (
        env.max_applicants_per_month * env.origination_months
        + env.origination_months
        + env.loan_term_months + env.portfolio.max_delay_months + 10
    )

    terminated = False
    truncated = False
    for _ in range(max_steps):
        customer_before_action = (
            env.current_customer.copy()
            if env.phase == "allocation"
            else None
        )

        action, _ = model.predict(obs, deterministic=True)
        action = int(np.asarray(action).reshape(-1)[0])

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward_as_trained += float(reward)
        total_risk_penalty += float(info.get("risk_penalty", 0.0))
        total_recognized_default_penalty += float(info.get("default_penalty", 0.0))

        if "action_index" in info:
            action_index = int(info["action_index"])
            action_counts[action_index] += 1
            approved_amount = float(info.get("approved_amount", 0.0))
            if approved_amount > 0:
                approved_action_counts[action_index] += 1
                total_legacy_social += legacy_engine._social_reward(
                    customer_before_action, approved_amount
                )
                total_experimental_social += experimental_engine._social_reward(
                    customer_before_action, approved_amount
                )
            if bool(info.get("invalid_action", False)):
                invalid_actions += 1

        if terminated or truncated:
            break
    else:
        env.close()
        raise RuntimeError(
            f"Episode exceeded max_steps={max_steps}; model={run_name}, seed={seed}"
        )

    portfolio = env.portfolio
    applicants_seen = env.total_applicants_seen
    financial_pnl = float(portfolio.cumulative_interest_collected - portfolio.cumulative_net_default_loss)
    social_weight = float(config["social_reward_weight"])
    financial_weight = float(config["financial_reward_weight"])
    financial_scale = float(config["financial_reward_scale"])
    invalid_penalty = float(config["invalid_action_penalty"])

    # Both counterfactual social modes share the realized risk/default costs
    # and financial cashflows from this episode. This matches the new reward
    # structure without charging expected PD as an immediate default penalty.
    shared_non_social_reward = (
        financial_weight * financial_pnl / financial_scale
        - total_risk_penalty
        - total_recognized_default_penalty
        - invalid_penalty * invalid_actions
    )
    total_reward_legacy = (
        social_weight * total_legacy_social + shared_non_social_reward
    )
    total_reward_experimental = (
        social_weight * total_experimental_social + shared_non_social_reward
    )

    result = {
        "run_name": run_name,
        "reward_mode_used_for_training": config["reward_mode"],
        "episode": episode,
        "seed": seed,
        "split": config.get("evaluation_split", "test"),
        "reapplication_share": float(config.get("reapplication_share", 0.10)),
        "reapplication_cooldown_months": int(config.get("reapplication_cooldown_months", 3)),
        "total_reward_as_trained": total_reward_as_trained,
        "total_reward_under_legacy_formula": total_reward_legacy,
        "total_reward_under_amount_sensitive_formula": total_reward_experimental,
        "social_reward_legacy": total_legacy_social,
        "social_reward_amount_sensitive": total_experimental_social,
        "financial_pnl": financial_pnl,
        "total_risk_penalty": total_risk_penalty,
        "total_recognized_default_penalty": total_recognized_default_penalty,
        "interest_collected": float(portfolio.cumulative_interest_collected),
        "net_default_loss": float(portfolio.cumulative_net_default_loss),
        "expected_default_count": float(portfolio.cumulative_expected_defaults),
        "recognized_default_count": float(portfolio.cumulative_recognized_defaults),
        "realized_default_rate": (
            float(portfolio.cumulative_recognized_defaults / len(portfolio.contracts))
            if portfolio.contracts else 0.0
        ),
        "mean_contract_term_pd": (
            float(np.mean([contract.default_probability_over_term for contract in portfolio.contracts]))
            if portfolio.contracts else 0.0
        ),
        "missed_installment_count": float(portfolio.cumulative_missed_installments),
        "delayed_payment_amount_scheduled": float(portfolio.cumulative_delayed_payment_amount_scheduled),
        "delayed_payment_amount_collected": float(portfolio.cumulative_delayed_payment_amount_collected),
        "pending_delayed_receivables": float(portfolio.pending_delayed_receivables),
        "expected_delayed_installments": float(portfolio.cumulative_expected_delayed_installments),
        "principal_collected": float(portfolio.cumulative_principal_collected)
            if hasattr(portfolio, "cumulative_principal_collected") else np.nan,
        "applicants_seen": int(applicants_seen),
        "new_applicants_seen": int(env.new_applicants_seen),
        "reapplications_seen": int(env.reapplications_seen),
        "customers_with_reapplications": int(len(env.customers_with_reapplications)),
        "loans_granted": int(len(portfolio.contracts)),
        "loan_approval_rate": len(portfolio.contracts) / applicants_seen if applicants_seen else 0.0,
        "total_disbursed": float(portfolio.total_disbursed),
        "average_loan_size": float(portfolio.total_disbursed / len(portfolio.contracts))
            if portfolio.contracts else 0.0,
        "invalid_actions": int(invalid_actions),
        "final_cash": float(portfolio.available_budget),
        "ending_active_contracts": int(portfolio.active_contract_count),
        "final_expected_outstanding_principal": float(portfolio.expected_outstanding_principal),
        "ending_month": int(portfolio.month),
    }

    for action_index, amount in enumerate(env.ACTION_AMOUNTS):
        result[f"requested_action_{amount}_count"] = int(action_counts[action_index])
        result[f"requested_action_{amount}_share"] = (
            float(action_counts[action_index] / applicants_seen)
            if applicants_seen else 0.0
        )
        result[f"approved_action_{amount}_count"] = int(approved_action_counts[action_index])

    env.close()
    return result


def main():
    args = parse_args()
    if args.episodes < 1:
        raise ValueError("--episodes must be >= 1")

    root = project_root()
    model_root = root / "models" / "portfolio_dqn"

    checkpoint_paths = {}
    if args.checkpoint_paths is not None:
        if len(args.checkpoint_paths) != len(args.run_names):
            raise ValueError(
                "--checkpoint-paths must provide exactly one file per --run-names entry."
            )
        for run_name, supplied_path in zip(args.run_names, args.checkpoint_paths):
            checkpoint_path = Path(supplied_path).expanduser()
            if not checkpoint_path.is_absolute():
                checkpoint_path = root / checkpoint_path
            if not checkpoint_path.is_file():
                raise FileNotFoundError(
                    f"Checkpoint for {run_name} does not exist: {checkpoint_path}"
                )
            checkpoint_paths[run_name] = checkpoint_path

    run_configs = {}
    for run_name in args.run_names:
        run_dir = model_root / run_name
        model_path = checkpoint_paths.get(run_name, run_dir / "model.zip")
        config_path = run_dir / "config.json"
        if not model_path.is_file() or not config_path.is_file():
            raise FileNotFoundError(
                f"Missing model/checkpoint or config for {run_name}. "
                f"Expected {model_path} and {config_path}. Train this run first."
            )
        with config_path.open("r", encoding="utf-8") as f:
            config = json.load(f)
        if int(config.get("environment_model_version", 1)) < 7:
            raise ValueError(
                f"Model {run_name} predates the unique first-time applicant, "
                "configurable reapplication process, and applicant-history observation. "
                "Retrain it with env/train_portfolio_dqn.py before evaluation."
            )
        run_configs[run_name] = config

    first_config = run_configs[args.run_names[0]]
    split_seed = int(first_config["split_seed"])
    dataset_path = args.data or first_config["dataset_path"]
    train_raw, validation_raw, test_raw, scaler, split_info = load_split_and_scaler(
        data_path=dataset_path,
        split_seed=split_seed,
    )

    for run_name, config in run_configs.items():
        if int(config["split_seed"]) != split_seed:
            raise ValueError("Models use different data split seeds; evaluation is not matched.")
        if Path(config["dataset_path"]).resolve() != Path(split_info["dataset_path"]).resolve():
            raise ValueError(
                f"Model {run_name} was trained with a different dataset path. "
                "Use the same --data file for both experiments."
            )

    evaluation_raw = validation_raw if args.split == "validation" else test_raw
    episode_rows = []

    print("=" * 76)
    print("PORTFOLIO DQN EVALUATION")
    print("=" * 76)
    print(f"Dataset         : {split_info['dataset_path']}")
    print(f"Usable rows     : {split_info['dataset_rows_used']:,}")
    print(f"Evaluation split: {args.split} ({len(evaluation_raw):,} rows)")
    print(f"Episodes/model  : {args.episodes}")
    print(f"Matched seeds   : {args.eval_seed}..{args.eval_seed + args.episodes - 1}")
    print("-" * 76)

    for run_name in args.run_names:
        config = run_configs[run_name]
        model_path = checkpoint_paths.get(
            run_name,
            model_root / run_name / "model.zip",
        )
        model = DQN.load(str(model_path), device="auto")
        print(f"Evaluating {run_name} | trained reward mode={config['reward_mode']}")

        for episode in range(args.episodes):
            seed = args.eval_seed + episode
            result = evaluate_episode(
                run_name=run_name,
                model=model,
                config=config,
                raw_df=evaluation_raw,
                scaler=scaler,
                episode=episode + 1,
                seed=seed,
            )
            result["split"] = args.split
            result["model_path_used"] = str(model_path)
            episode_rows.append(result)
            print(
                f"  episode {episode + 1}/{args.episodes} | seed={seed} | "
                f"loans={result['loans_granted']} | "
                f"legacy reward={result['total_reward_under_legacy_formula']:.3f} | "
                f"experimental reward={result['total_reward_under_amount_sensitive_formula']:.3f}"
            )

    episodes_df = pd.DataFrame(episode_rows)
    numeric_metrics = [
        "total_reward_as_trained",
        "total_reward_under_legacy_formula",
        "total_reward_under_amount_sensitive_formula",
        "social_reward_legacy",
        "social_reward_amount_sensitive",
        "total_risk_penalty",
        "total_recognized_default_penalty",
        "financial_pnl",
        "interest_collected",
        "net_default_loss",
        "expected_default_count",
        "recognized_default_count",
        "missed_installment_count",
        "delayed_payment_amount_scheduled",
        "delayed_payment_amount_collected",
        "pending_delayed_receivables",
        "expected_delayed_installments",
        "applicants_seen",
        "new_applicants_seen",
        "reapplications_seen",
        "customers_with_reapplications",
        "loans_granted",
        "loan_approval_rate",
        "total_disbursed",
        "average_loan_size",
        "invalid_actions",
        "final_cash",
        "ending_active_contracts",
        "final_expected_outstanding_principal",
        "ending_month",
    ]
    summary_rows = []
    for run_name, group in episodes_df.groupby("run_name", sort=False):
        row = {"run_name": run_name, "reward_mode_used_for_training": group["reward_mode_used_for_training"].iloc[0], "episodes": len(group)}
        for metric in numeric_metrics:
            row[f"mean_{metric}"] = float(group[metric].mean())
            row[f"std_{metric}"] = float(group[metric].std(ddof=1)) if len(group) > 1 else 0.0
        for amount in PortfolioLoanEnv.ACTION_AMOUNTS:
            row[f"mean_requested_action_{amount}_share"] = float(
                group[f"requested_action_{amount}_share"].mean()
            )
            row[f"mean_approved_action_{amount}_count"] = float(
                group[f"approved_action_{amount}_count"].mean()
            )
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    episodes_path = output_dir / f"portfolio_dqn_eval_{args.split}_episodes.csv"
    summary_path = output_dir / f"portfolio_dqn_eval_{args.split}_summary.csv"
    episodes_df.to_csv(episodes_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    display_columns = [
        "run_name",
        "reward_mode_used_for_training",
        "mean_total_reward_under_legacy_formula",
        "mean_total_reward_under_amount_sensitive_formula",
        "mean_total_risk_penalty",
        "mean_total_recognized_default_penalty",
        "mean_financial_pnl",
        "mean_net_default_loss",
        "mean_recognized_default_count",
        "mean_missed_installment_count",
        "mean_delayed_payment_amount_scheduled",
        "mean_delayed_payment_amount_collected",
        "mean_pending_delayed_receivables",
        "mean_applicants_seen",
        "mean_new_applicants_seen",
        "mean_reapplications_seen",
        "mean_customers_with_reapplications",
        "mean_loans_granted",
        "mean_average_loan_size",
        "mean_final_cash",
    ]
    print()
    print("SUMMARY")
    print(summary_df[display_columns].to_string(index=False, float_format=lambda value: f"{value:,.3f}"))
    print("Saved:")
    print(f"  Episodes: {episodes_path}")
    print(f"  Summary : {summary_path}")
    print("Action-distribution columns are included in both outputs.")


if __name__ == "__main__":
    main()
