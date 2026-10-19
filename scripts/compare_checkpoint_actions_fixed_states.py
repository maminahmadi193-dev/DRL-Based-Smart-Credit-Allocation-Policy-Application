"""Compare DQN checkpoint policies on the exact same observed states.

A reference checkpoint generates a fixed panel of validation observations.
Every selected checkpoint then predicts actions on those identical observations.
This avoids confusing changing episode trajectories with changes in the policy's
action choice on a common state panel.

This is a diagnostic, not a proof of convergence and not a full-state-space test.
The panel reflects states visited under the reference checkpoint's policy.
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import DQN

from portfolio_experiment_utils import load_split_and_scaler, project_root
from portfolio_loan_env import PortfolioLoanEnv


DEFAULT_CHECKPOINT_STEPS = [60_000, 70_000, 80_000, 85_000, 90_000, 95_000, 100_000]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare saved DQN checkpoint actions on a fixed panel of states "
            "collected from validation trajectories."
        )
    )
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--data", default=None)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--eval-seed", type=int, default=20_000)
    parser.add_argument(
        "--reference-steps",
        type=int,
        default=95_000,
        help="Checkpoint used to generate the fixed observation panel.",
    )
    parser.add_argument(
        "--checkpoint-steps",
        type=int,
        nargs="+",
        default=DEFAULT_CHECKPOINT_STEPS,
        help="Checkpoint steps to compare on the same observation panel.",
    )
    parser.add_argument(
        "--output-dir",
        default="models/results/portfolio_dqn/policy_stability_fixed_states",
    )
    return parser.parse_args()


def _make_env(config, raw_df, scaler):
    """Construct the exact configured environment used for this training run."""
    return PortfolioLoanEnv(
        raw_df=raw_df,
        scaler=scaler,
        initial_budget=float(config["initial_budget"]),
        monthly_capital_injection=float(config.get("monthly_capital_injection", 0.0)),
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
        default_dpd_threshold_days=int(config.get("default_dpd_threshold_days", 90)),
        hazard_slope=float(config.get("hazard_slope", 0.08)),
        hazard_center=float(config.get("hazard_center", 60.0)),
        max_hazard=float(config.get("max_hazard", 0.005)),
        hazard_intercept=float(config.get("hazard_intercept", 0.0)),
        social_reward_weight=float(config["social_reward_weight"]),
        financial_reward_weight=float(config["financial_reward_weight"]),
        financial_reward_scale=float(config["financial_reward_scale"]),
        invalid_action_penalty=float(config["invalid_action_penalty"]),
        scale_access_rewards_by_coverage=bool(
            config["scale_access_rewards_by_coverage"]
        ),
    )


def _checkpoint_map(checkpoint_dir: Path, run_name: str) -> dict[int, Path]:
    found = {}
    for path in checkpoint_dir.glob(f"{run_name}_*_steps.zip"):
        match = re.search(r"_(\d+)_steps\.zip$", path.name)
        if match:
            found[int(match.group(1))] = path
    return found


def collect_reference_observations(
    model: DQN,
    config: dict,
    raw_df: pd.DataFrame,
    scaler,
    episodes: int,
    eval_seed: int,
    reference_steps: int,
):
    """Collect allocation-phase observations from fixed seeded validation rollouts."""
    observations: list[np.ndarray] = []
    metadata: list[dict] = []
    action_amounts = np.asarray(PortfolioLoanEnv.ACTION_AMOUNTS, dtype=np.int64)

    env = _make_env(config, raw_df, scaler)
    try:
        for episode_index in range(1, episodes + 1):
            seed = eval_seed + episode_index - 1
            obs, _ = env.reset(seed=seed)
            max_steps = (
                env.max_applicants_per_month * env.origination_months
                + env.origination_months
                + env.loan_term_months
                + env.portfolio.max_delay_months
                + 10
            )
            decision_index = 0
            for step_index in range(max_steps):
                # Only allocation decisions are relevant; settlement transitions
                # are still simulated to keep the environment trajectory intact.
                in_allocation = env.phase == "allocation"
                if in_allocation:
                    observations.append(np.asarray(obs, dtype=np.float32).copy())
                    metadata.append({
                        "reference_policy_steps": reference_steps,
                        "episode": episode_index,
                        "seed": seed,
                        "decision_index_in_episode": decision_index,
                        "month": int(env.portfolio.month),
                        "applicant_position_in_cohort": int(
                            env.current_applicant_position
                        ),
                        "customer_row_index": int(env.current_row_index),
                        "is_reapplication": bool(env.current_is_reapplication),
                        "prior_application_count": int(
                            env.current_prior_application_count
                        ),
                        "available_budget_before_action": float(
                            env.portfolio.available_budget
                        ),
                    })
                    decision_index += 1

                action, _ = model.predict(obs, deterministic=True)
                action_index = int(np.asarray(action).reshape(-1)[0])
                if action_index < 0 or action_index >= len(action_amounts):
                    raise RuntimeError(
                        f"Model predicted invalid action index {action_index}."
                    )

                obs, _, terminated, truncated, _ = env.step(action_index)
                if terminated or truncated:
                    break
            else:
                raise RuntimeError(
                    f"Episode seed={seed} exceeded max_steps={max_steps} "
                    "while collecting reference states."
                )
    finally:
        env.close()

    if not observations:
        raise RuntimeError("No allocation observations were collected.")

    return np.stack(observations).astype(np.float32), pd.DataFrame(metadata)


def main():
    args = parse_args()
    if args.episodes < 1:
        raise ValueError("--episodes must be >= 1")
    if args.reference_steps not in args.checkpoint_steps:
        raise ValueError(
            "--reference-steps must also appear in --checkpoint-steps so its "
            "policy can be included in the comparison."
        )
    if len(set(args.checkpoint_steps)) != len(args.checkpoint_steps):
        raise ValueError("--checkpoint-steps must not contain duplicates.")

    root = project_root()
    run_dir = root / "models" / "portfolio_dqn" / args.run_name
    config_path = run_dir / "config.json"
    checkpoint_dir = run_dir / "checkpoints"

    if not config_path.is_file():
        raise FileNotFoundError(f"Missing run config: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    if int(config.get("environment_model_version", 1)) < 7:
        raise ValueError(
            "This diagnostic requires environment model version >= 7."
        )

    checkpoint_paths = _checkpoint_map(checkpoint_dir, args.run_name)
    missing_steps = [
        step for step in args.checkpoint_steps if step not in checkpoint_paths
    ]
    if missing_steps:
        raise FileNotFoundError(
            f"Missing checkpoint steps {missing_steps} in {checkpoint_dir}"
        )

    dataset_path = args.data or config.get("dataset_path")
    _, validation_raw, _, scaler, split_info = load_split_and_scaler(
        data_path=dataset_path,
        split_seed=int(config["split_seed"]),
    )
    if Path(config["dataset_path"]).resolve() != Path(
        split_info["dataset_path"]
    ).resolve():
        raise ValueError(
            "The supplied dataset path differs from the path stored in this run's config."
        )

    print("=" * 78)
    print("FIXED-STATE CHECKPOINT POLICY COMPARISON")
    print("=" * 78)
    print(f"Run                  : {args.run_name}")
    print(f"Dataset              : {split_info['dataset_path']}")
    print(f"Split                : validation ({len(validation_raw):,} rows)")
    print(f"Reference checkpoint : {args.reference_steps:,}")
    print(f"Rollout episodes     : {args.episodes}")
    print(f"Seeds                : {args.eval_seed}..{args.eval_seed + args.episodes - 1}")
    print(f"Compared checkpoints : {sorted(args.checkpoint_steps)}")
    print("-" * 78)
    print("Loading reference checkpoint and collecting fixed observations...")

    reference_model = DQN.load(
        str(checkpoint_paths[args.reference_steps]), device="auto"
    )
    observations, state_metadata = collect_reference_observations(
        model=reference_model,
        config=config,
        raw_df=validation_raw,
        scaler=scaler,
        episodes=args.episodes,
        eval_seed=args.eval_seed,
        reference_steps=args.reference_steps,
    )
    del reference_model

    action_amounts = np.asarray(PortfolioLoanEnv.ACTION_AMOUNTS, dtype=np.int64)
    action_matrix = {}
    summary_rows = []

    print(f"Collected allocation observations: {len(observations):,}")
    print("Predicting actions for all requested checkpoints on the SAME observations...")

    for steps in sorted(args.checkpoint_steps):
        model = DQN.load(str(checkpoint_paths[steps]), device="auto")
        actions, _ = model.predict(observations, deterministic=True)
        actions = np.asarray(actions).reshape(-1).astype(int)
        del model

        if len(actions) != len(observations):
            raise RuntimeError(
                f"Checkpoint {steps} returned {len(actions)} actions for "
                f"{len(observations)} observations."
            )
        if np.any(actions < 0) or np.any(actions >= len(action_amounts)):
            raise RuntimeError(f"Checkpoint {steps} returned an invalid action index.")

        action_matrix[steps] = actions
        row = {
            "checkpoint_steps": steps,
            "reference_checkpoint_steps": args.reference_steps,
            "reference_episodes": args.episodes,
            "fixed_observations": int(len(observations)),
            "exact_action_agreement_with_reference_pct": float(
                np.mean(actions == action_matrix[args.reference_steps]) * 100.0
            ) if args.reference_steps in action_matrix else np.nan,
            "mean_absolute_action_difference_million": np.nan,
        }
        for action_index, amount in enumerate(action_amounts):
            row[f"action_{amount}_share_pct"] = float(
                np.mean(actions == action_index) * 100.0
            )
        summary_rows.append(row)
        print(
            f"steps={steps:>8,} | "
            f"reference action agreement="
            f"{row['exact_action_agreement_with_reference_pct'] if np.isfinite(row['exact_action_agreement_with_reference_pct']) else float('nan'):.2f}%"
        )

    # Build state-by-state action records. Observations themselves are not saved;
    # the CSV contains state identifiers and each checkpoint's predicted action.
    per_state = state_metadata.copy()
    for steps in sorted(action_matrix):
        actions = action_matrix[steps]
        per_state[f"action_index_steps_{steps}"] = actions
        per_state[f"action_amount_steps_{steps}"] = action_amounts[actions]

    reference_actions = action_matrix[args.reference_steps]
    pairwise_rows = []
    for steps_a in sorted(action_matrix):
        for steps_b in sorted(action_matrix):
            actions_a = action_matrix[steps_a]
            actions_b = action_matrix[steps_b]
            pairwise_rows.append({
                "checkpoint_a_steps": steps_a,
                "checkpoint_b_steps": steps_b,
                "same_action_pct": float(np.mean(actions_a == actions_b) * 100.0),
                "mean_absolute_action_difference_million_toman": float(
                    np.mean(np.abs(action_amounts[actions_a] - action_amounts[actions_b]))
                    / 1_000_000.0
                ),
            })

    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_df = pd.DataFrame(summary_rows)
    reference_actions = action_matrix[args.reference_steps]
    summary_df["exact_action_agreement_with_reference_pct"] = summary_df[
        "checkpoint_steps"
    ].map({
        steps: float(np.mean(action_matrix[steps] == reference_actions) * 100.0)
        for steps in action_matrix
    })
    summary_df["mean_absolute_action_difference_million"] = summary_df[
        "checkpoint_steps"
    ].map({
        steps: float(
            np.mean(np.abs(action_amounts[action_matrix[steps]] - action_amounts[reference_actions]))
            / 1_000_000.0
        )
        for steps in action_matrix
    })

    summary_path = output_dir / f"{args.run_name}_fixed_state_action_summary.csv"
    pairwise_path = output_dir / f"{args.run_name}_fixed_state_pairwise_agreement.csv"
    per_state_path = output_dir / f"{args.run_name}_fixed_state_actions.csv"
    summary_df.sort_values("checkpoint_steps").to_csv(summary_path, index=False)
    pd.DataFrame(pairwise_rows).to_csv(pairwise_path, index=False)
    per_state.to_csv(per_state_path, index=False)

    print()
    print("ACTION DISTRIBUTION AND REFERENCE AGREEMENT")
    summary_columns = [
        "checkpoint_steps",
        "action_0_share_pct",
        "action_100000000_share_pct",
        "action_200000000_share_pct",
        "action_300000000_share_pct",
        "action_400000000_share_pct",
        "action_500000000_share_pct",
        "exact_action_agreement_with_reference_pct",
        "mean_absolute_action_difference_million",
    ]
    print(
        summary_df.sort_values("checkpoint_steps")[summary_columns].to_string(
            index=False, float_format=lambda x: f"{x:.2f}"
        )
    )
    print()
    print("Saved:")
    print(f"  Summary:          {summary_path}")
    print(f"  Pairwise agreement: {pairwise_path}")
    print(f"  Per-state actions: {per_state_path}")
    print(
        "Interpretation note: these are fixed observations visited by the reference "
        "policy on validation trajectories, not the full state space and not a proof "
        "of global convergence."
    )


if __name__ == "__main__":
    main()
