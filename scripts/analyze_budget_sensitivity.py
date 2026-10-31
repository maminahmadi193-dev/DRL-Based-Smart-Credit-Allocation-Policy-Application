"""Diagnose the portfolio DQN cash-budget threshold and policy sensitivity.

Run from the repository root. This utility does not train a model.

Modes
-----
trace:
    Run a frozen policy at a reference budget and compute, per episode, the
    minimum initial cash needed to fund the exact same observed action path.
sweep:
    Run the same frozen policy and common episode seeds across selected initial
    budgets. The budget-normalization scale in observations is held fixed, so
    the trained policy is evaluated on a comparable feature scale.

The trace threshold is conditional on the recorded action/repayment path. It is
not the threshold at which the policy would necessarily stop changing, because
a constrained budget can change observations, actions, contract repayments,
and later decisions.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import DQN

import _bootstrap  # noqa: F401
from portfolio_experiment_utils import load_split_and_scaler, project_root
from portfolio_loan_env import PortfolioLoanEnv


DEFAULT_RUN = (
    "portfolio_dqn_amount_sensitive_budget420B_seed42_100k_"
    "lr1e4_tgt1000_reapp3m_work"
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["trace", "sweep"], default="trace")
    parser.add_argument("--run-name", default=DEFAULT_RUN)
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--data", default="notebooks/df_cleaned.csv")
    parser.add_argument(
        "--split", choices=["validation", "test"], default="validation"
    )
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--eval-seed", type=int, default=50000)
    parser.add_argument(
        "--reference-budget",
        type=float,
        default=420_000_000_000.0,
        help="Actual cash used for the trace run; also the budget baseline in the threshold calculation.",
    )
    parser.add_argument(
        "--observation-scale-budget",
        type=float,
        default=None,
        help="Fixed denominator for budget-normalized observations. Defaults to the saved run's initial budget.",
    )
    parser.add_argument(
        "--budgets",
        nargs="+",
        type=float,
        default=None,
        help="Required in sweep mode; budgets in raw currency units, e.g. 35e9 87.5e9 140e9 175e9 210e9 420e9.",
    )
    parser.add_argument(
        "--output-dir",
        default="models/results/portfolio_dqn/budget_sensitivity",
    )
    return parser.parse_args()


def load_inputs(args):
    root = project_root()
    run_dir = root / "models" / "portfolio_dqn" / args.run_name
    config_path = run_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing run config: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        config = json.load(file)

    if int(config.get("environment_model_version", 1)) < PortfolioLoanEnv.ENVIRONMENT_MODEL_VERSION:
        raise ValueError(
            f"This diagnostic requires environment model version "
            f">= {PortfolioLoanEnv.ENVIRONMENT_MODEL_VERSION}; the checkpoint "
            "must use the synchronized customer-observation semantics."
        )

    if args.checkpoint_path:
        model_path = Path(args.checkpoint_path).expanduser()
        if not model_path.is_absolute():
            model_path = root / model_path
    else:
        model_path = (
            run_dir / "checkpoints" / f"{args.run_name}_100000_steps.zip"
        )
        if not model_path.is_file():
            model_path = run_dir / "model.zip"

    if not model_path.is_file():
        raise FileNotFoundError(f"Model/checkpoint not found: {model_path}")

    train, validation, test, scaler, split_info = load_split_and_scaler(
        data_path=args.data,
        split_seed=int(config["split_seed"]),
    )

    saved_dataset = Path(config["dataset_path"]).resolve()
    loaded_dataset = Path(split_info["dataset_path"]).resolve()
    if saved_dataset != loaded_dataset:
        raise ValueError(
            f"Dataset mismatch. Run config records {saved_dataset}, "
            f"but diagnostic loaded {loaded_dataset}."
        )

    evaluation_data = validation if args.split == "validation" else test
    model = DQN.load(str(model_path), device="auto")
    return root, config, model_path, model, evaluation_data, scaler, split_info


def make_env(config, raw_df, scaler, budget):
    """Build the evaluation environment from the saved run config."""
    return PortfolioLoanEnv(
        raw_df=raw_df,
        scaler=scaler,
        initial_budget=float(budget),
        monthly_capital_injection=float(
            config.get("monthly_capital_injection", 0.0)
        ),
        applicants_per_month=int(config.get("applicants_per_month", 50)),
        min_applicants_per_month=config.get("min_applicants_per_month"),
        max_applicants_per_month=config.get("max_applicants_per_month"),
        reapplication_share=float(config.get("reapplication_share", 0.10)),
        reapplication_cooldown_months=int(
            config.get("reapplication_cooldown_months", 3)
        ),
        origination_months=int(config["origination_months"]),
        loan_term_months=int(config["loan_term_months"]),
        annual_interest_rate=float(
            config.get(
                "annual_interest_rate",
                float(config.get("monthly_interest_rate", 0.0)) * 12.0,
            )
        ),
        default_collection_lag_months=int(
            config.get("default_collection_lag_months", 4)
        ),
        default_dpd_threshold_days=int(
            config.get("default_dpd_threshold_days", 90)
        ),
        hazard_slope=float(config.get("hazard_slope", 0.08)),
        hazard_center=float(config.get("hazard_center", 60.0)),
        max_hazard=float(config.get("max_hazard", 0.005)),
        hazard_intercept=float(config.get("hazard_intercept", 0.0)),
        social_reward_weight=float(config.get("social_reward_weight", 1.0)),
        financial_reward_weight=float(
            config.get("financial_reward_weight", 1.0)
        ),
        financial_reward_scale=float(
            config.get("financial_reward_scale", 100_000_000.0)
        ),
        invalid_action_penalty=float(
            config.get("invalid_action_penalty", 0.25)
        ),
        risk_weight=float(config.get("risk_weight", 1.0)),
        default_cost=float(config.get("default_cost", 10.0)),
    )


def run_episode(
    model,
    config,
    raw_df,
    scaler,
    budget,
    observation_scale_budget,
    episode,
    seed,
    run_name,
    split_name,
    reference_budget=None,
):
    env = make_env(config, raw_df, scaler, budget)
    obs, _ = env.reset(seed=seed)

    # PortfolioSimulator already holds the actual test budget. Keep the
    # normalization denominator seen by the frozen policy fixed at its
    # training/reference scale, then rebuild the initial observation.
    env.initial_budget = float(observation_scale_budget)
    obs = env._get_observation()

    total_reward = 0.0
    invalid_actions = 0
    decisions = []
    action_counts = np.zeros(len(env.ACTION_AMOUNTS), dtype=np.int64)
    max_steps = (
        env.max_applicants_per_month * env.origination_months
        + env.origination_months
        + env.loan_term_months
        + env.portfolio.max_delay_months
        + 10
    )

    for step_no in range(max_steps):
        phase_before = env.phase
        cash_before = float(env.portfolio.available_budget)
        month_before = int(env.portfolio.month)
        applicant_id = (
            env.current_row_index if phase_before == "allocation" else None
        )

        action, _ = model.predict(obs, deterministic=True)
        action = int(np.asarray(action).reshape(-1)[0])
        requested_amount = float(env.ACTION_AMOUNTS[action])

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)

        if phase_before == "allocation":
            action_counts[action] += 1
            is_invalid = bool(info.get("invalid_action", False))
            invalid_actions += int(is_invalid)
            row = {
                "run_name": run_name,
                "split": split_name,
                "budget": float(budget),
                "reference_budget": (
                    float(reference_budget)
                    if reference_budget is not None
                    else np.nan
                ),
                "observation_scale_budget": float(observation_scale_budget),
                "episode": int(episode),
                "seed": int(seed),
                "decision_step": int(step_no + 1),
                "month": month_before,
                "applicant_id": applicant_id,
                "cash_before": cash_before,
                "action_index": action,
                "requested_amount": requested_amount,
                "approved_amount": float(info.get("approved_amount", 0.0)),
                "invalid_action": is_invalid,
                "cash_after": float(env.portfolio.available_budget),
            }
            if reference_budget is not None and requested_amount > 0.0:
                # If the entire recorded cash-flow/action path were preserved,
                # starting cash would shift all cash balances one-for-one.
                row["minimum_initial_budget_for_this_decision"] = max(
                    0.0,
                    float(reference_budget)
                    + requested_amount
                    - cash_before,
                )
            else:
                row["minimum_initial_budget_for_this_decision"] = np.nan
            decisions.append(row)

        if terminated or truncated:
            break
    else:
        env.close()
        raise RuntimeError(
            f"Episode exceeded max_steps={max_steps}; run={run_name}, seed={seed}"
        )

    portfolio = env.portfolio
    applicants_seen = int(env.total_applicants_seen)
    loan_count = int(len(portfolio.contracts))
    disbursed = float(portfolio.total_disbursed)
    thresholds = [
        row["minimum_initial_budget_for_this_decision"]
        for row in decisions
        if np.isfinite(row["minimum_initial_budget_for_this_decision"])
    ]

    result = {
        "run_name": run_name,
        "split": split_name,
        "budget": float(budget),
        "reference_budget": (
            float(reference_budget) if reference_budget is not None else np.nan
        ),
        "observation_scale_budget": float(observation_scale_budget),
        "episode": int(episode),
        "seed": int(seed),
        "total_reward_as_trained": float(total_reward),
        "applicants_seen": applicants_seen,
        "loans_granted": loan_count,
        "loan_approval_rate": (
            loan_count / applicants_seen if applicants_seen else 0.0
        ),
        "invalid_actions": int(invalid_actions),
        "total_disbursed": disbursed,
        "average_loan_size": disbursed / loan_count if loan_count else 0.0,
        "final_cash": float(portfolio.available_budget),
        "ending_month": int(portfolio.month),
        "recognized_default_count": float(
            portfolio.cumulative_recognized_defaults
        ),
        "missed_installment_count": float(
            portfolio.cumulative_missed_installments
        ),
        "financial_pnl": float(
            portfolio.cumulative_interest_collected
            - portfolio.cumulative_net_default_loss
        ),
        "min_cash_before_decision": min(
            (row["cash_before"] for row in decisions), default=np.nan
        ),
        "allocation_decisions": len(decisions),
        "required_initial_budget_for_same_action_path": (
            max(thresholds) if thresholds else np.nan
        ),
    }

    for action_index, amount in enumerate(env.ACTION_AMOUNTS):
        result[f"requested_action_{amount}_count"] = int(
            action_counts[action_index]
        )
        result[f"requested_action_{amount}_share"] = (
            float(action_counts[action_index] / applicants_seen)
            if applicants_seen
            else 0.0
        )

    env.close()
    return result, decisions


def summarize_sweep(episodes_df):
    rows = []
    metrics = [
        "total_reward_as_trained",
        "loans_granted",
        "loan_approval_rate",
        "invalid_actions",
        "total_disbursed",
        "average_loan_size",
        "final_cash",
        "min_cash_before_decision",
    ]
    metrics.extend(
        f"requested_action_{amount}_share"
        for amount in (0, 100_000_000, 200_000_000, 300_000_000, 400_000_000, 500_000_000)
    )

    for budget, group in episodes_df.groupby("budget", sort=True):
        row = {"budget": float(budget), "episodes": int(len(group))}
        for metric in metrics:
            if metric not in group:
                continue
            values = pd.to_numeric(group[metric], errors="coerce")
            row[f"mean_{metric}"] = float(values.mean())
            if metric in {
                "total_reward_as_trained",
                "loans_granted",
                "invalid_actions",
                "total_disbursed",
            }:
                row[f"std_{metric}"] = (
                    float(values.std(ddof=1)) if len(values) > 1 else 0.0
                )
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    if args.episodes < 1:
        raise ValueError("--episodes must be >= 1")
    if args.mode == "trace" and (
        not np.isfinite(args.reference_budget) or args.reference_budget <= 0
    ):
        raise ValueError("--reference-budget must be finite and positive")
    if args.mode == "sweep" and not args.budgets:
        raise ValueError("--budgets is required in sweep mode")
    if args.mode == "sweep" and any(
        not np.isfinite(budget) or budget <= 0 for budget in args.budgets
    ):
        raise ValueError("Every budget must be finite and positive")
    if args.observation_scale_budget is not None and (
        not np.isfinite(args.observation_scale_budget)
        or args.observation_scale_budget <= 0
    ):
        raise ValueError("--observation-scale-budget must be finite and positive")

    (
        root,
        config,
        model_path,
        model,
        raw_df,
        scaler,
        split_info,
    ) = load_inputs(args)

    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    observation_scale_budget = (
        float(args.observation_scale_budget)
        if args.observation_scale_budget is not None
        else float(config["initial_budget"])
    )
    budgets = (
        [float(args.reference_budget)]
        if args.mode == "trace"
        else sorted(set(float(budget) for budget in args.budgets))
    )
    reference_budget = (
        float(args.reference_budget) if args.mode == "trace" else None
    )

    episode_rows = []
    all_decisions = []
    total_runs = len(budgets) * args.episodes

    print("=" * 78)
    print("PORTFOLIO DQN BUDGET SENSITIVITY DIAGNOSTIC")
    print(f"Mode                  : {args.mode}")
    print(f"Reference policy      : {args.run_name}")
    print(f"Checkpoint            : {model_path}")
    print(f"Split                 : {args.split} ({len(raw_df):,} rows)")
    print(f"Episodes per budget   : {args.episodes}")
    print(
        f"Matched episode seeds : {args.eval_seed}.."
        f"{args.eval_seed + args.episodes - 1}"
    )
    print(f"Budgets               : {budgets}")
    print(
        f"Observation scale     : {observation_scale_budget:,.0f}"
    )
    print(
        "Only initial cash changes in sweep mode; policy weights, "
        "data split, random seeds, and normalization scale stay fixed."
    )
    print("=" * 78)

    run_index = 0
    for budget in budgets:
        for episode_index in range(args.episodes):
            run_index += 1
            seed = args.eval_seed + episode_index
            episode_row, decisions = run_episode(
                model=model,
                config=config,
                raw_df=raw_df,
                scaler=scaler,
                budget=budget,
                observation_scale_budget=observation_scale_budget,
                episode=episode_index + 1,
                seed=seed,
                run_name=args.run_name,
                split_name=args.split,
                reference_budget=reference_budget,
            )
            episode_rows.append(episode_row)
            all_decisions.extend(decisions)

            if (
                episode_index == 0
                or (episode_index + 1) % 25 == 0
                or episode_index + 1 == args.episodes
            ):
                print(
                    f"budget={budget / 1e9:8.3f}B | "
                    f"episode={episode_index + 1:>3}/{args.episodes} "
                    f"| reward={episode_row['total_reward_as_trained']:.3f} "
                    f"| loans={episode_row['loans_granted']} "
                    f"| invalid={episode_row['invalid_actions']} "
                    f"| progress={run_index}/{total_runs}"
                )

    episodes_df = pd.DataFrame(episode_rows)
    decisions_df = pd.DataFrame(all_decisions)

    if args.mode == "trace":
        episodes_path = output_dir / "budget_path_episodes.csv"
        decisions_path = output_dir / "budget_path_decisions.csv"
        summary_path = output_dir / "budget_path_threshold_summary.csv"

        episodes_df.to_csv(episodes_path, index=False)
        decisions_df.to_csv(decisions_path, index=False)

        thresholds = episodes_df[
            "required_initial_budget_for_same_action_path"
        ].dropna()
        if thresholds.empty:
            raise RuntimeError("No valid episode thresholds were calculated.")

        summary = pd.DataFrame(
            [{
                "reference_budget": float(args.reference_budget),
                "episodes": int(len(thresholds)),
                "critical_budget_min": float(thresholds.min()),
                "critical_budget_p10": float(thresholds.quantile(0.10)),
                "critical_budget_p25": float(thresholds.quantile(0.25)),
                "critical_budget_p50": float(thresholds.quantile(0.50)),
                "critical_budget_p75": float(thresholds.quantile(0.75)),
                "critical_budget_p90": float(thresholds.quantile(0.90)),
                "critical_budget_p95": float(thresholds.quantile(0.95)),
                "critical_budget_max": float(thresholds.max()),
            }]
        )
        summary.to_csv(summary_path, index=False)

        display_summary = summary.copy()
        for column in display_summary.columns:
            if column.startswith("critical_budget_"):
                display_summary[column] = display_summary[column].map(
                    lambda value: f"{value / 1e9:.3f}B"
                )
        print("\nBudget needed to fund each same reference action path:")
        print(display_summary.to_string(index=False))
        print(f"\nSaved episode thresholds : {episodes_path}")
        print(f"Saved decision trace     : {decisions_path}")
        print(f"Saved threshold summary  : {summary_path}")
        print(
            "Note: thresholds apply to the frozen reference trajectories only; "
            "they are not by themselves the policy saturation threshold."
        )
    else:
        episodes_path = output_dir / "budget_sweep_episodes.csv"
        summary_path = output_dir / "budget_sweep_summary.csv"
        episodes_df.to_csv(episodes_path, index=False)
        summary = summarize_sweep(episodes_df)
        summary.to_csv(summary_path, index=False)

        display_columns = [
            "budget",
            "episodes",
            "mean_total_reward_as_trained",
            "std_total_reward_as_trained",
            "mean_loans_granted",
            "mean_loan_approval_rate",
            "mean_invalid_actions",
            "mean_total_disbursed",
            "mean_average_loan_size",
            "mean_final_cash",
            "mean_min_cash_before_decision",
        ]
        print("\nBudget sweep summary:")
        display = summary[
            [column for column in display_columns if column in summary.columns]
        ].copy()
        display["budget"] = display["budget"].map(
            lambda value: f"{value / 1e9:.3f}B"
        )
        print(display.to_string(index=False))
        print(f"\nSaved episode metrics: {episodes_path}")
        print(f"Saved summary        : {summary_path}")
        print(
            "The same frozen checkpoint was evaluated at every budget. "
            "The cash-normalization denominator was kept fixed."
        )


if __name__ == "__main__":
    main()
