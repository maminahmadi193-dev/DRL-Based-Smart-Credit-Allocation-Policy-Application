"""Inspect Q-value margins and drift across saved DQN checkpoints.

All checkpoints are evaluated on a shared panel of validation observations
collected from trajectories of a chosen reference checkpoint. This diagnoses
whether action disagreements occur near Q-value ties or with larger margins.
It is not a proof of convergence and does not estimate Q* for the full MDP.
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from stable_baselines3 import DQN

from compare_checkpoint_actions_fixed_states import (
    _checkpoint_map,
    collect_reference_observations,
)
from portfolio_experiment_utils import load_split_and_scaler, project_root
from portfolio_loan_env import PortfolioLoanEnv


DEFAULT_CHECKPOINT_STEPS = [60_000, 70_000, 80_000, 85_000, 90_000, 95_000, 100_000]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare DQN Q-value margins and drift on fixed validation states."
    )
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--data", default=None)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--eval-seed", type=int, default=20_000)
    parser.add_argument("--reference-steps", type=int, default=95_000)
    parser.add_argument(
        "--checkpoint-steps",
        type=int,
        nargs="+",
        default=DEFAULT_CHECKPOINT_STEPS,
    )
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument(
        "--output-dir",
        default="models/results/portfolio_dqn/q_value_stability",
    )
    return parser.parse_args()


def q_values_for_observations(model: DQN, observations: np.ndarray, batch_size: int):
    """Return online-network Q-values without updating the model."""
    chunks = []
    model.policy.set_training_mode(False)
    with torch.no_grad():
        for start in range(0, len(observations), batch_size):
            batch = observations[start : start + batch_size]
            obs_tensor, _ = model.policy.obs_to_tensor(batch)
            q_tensor = model.q_net(obs_tensor)
            chunks.append(q_tensor.detach().cpu().numpy())
    if not chunks:
        raise ValueError("No observations were supplied.")
    return np.concatenate(chunks, axis=0)


def summarize_q(values: np.ndarray, steps: int, reference_values: np.ndarray,
                reference_actions: np.ndarray, action_amounts: np.ndarray):
    predicted_actions = np.argmax(values, axis=1)
    sorted_q = np.sort(values, axis=1)
    margins = sorted_q[:, -1] - sorted_q[:, -2]
    ref_sorted = np.sort(reference_values, axis=1)
    reference_margins = ref_sorted[:, -1] - ref_sorted[:, -2]
    disagreements = predicted_actions != reference_actions

    if np.any(disagreements):
        ref_disagreement_margin_mean = float(reference_margins[disagreements].mean())
        ref_disagreement_margin_median = float(np.median(reference_margins[disagreements]))
        current_disagreement_margin_mean = float(margins[disagreements].mean())
        current_disagreement_margin_median = float(np.median(margins[disagreements]))
    else:
        ref_disagreement_margin_mean = np.nan
        ref_disagreement_margin_median = np.nan
        current_disagreement_margin_mean = np.nan
        current_disagreement_margin_median = np.nan

    q_delta = values - reference_values
    row = {
        "checkpoint_steps": steps,
        "fixed_observations": int(len(values)),
        "action_agreement_with_reference_pct": float(np.mean(~disagreements) * 100.0),
        "mean_abs_action_difference_million_toman": float(
            np.mean(np.abs(action_amounts[predicted_actions] - action_amounts[reference_actions]))
            / 1_000_000.0
        ),
        "mean_top1_top2_q_margin": float(margins.mean()),
        "median_top1_top2_q_margin": float(np.median(margins)),
        "p10_top1_top2_q_margin": float(np.quantile(margins, 0.10)),
        "p90_top1_top2_q_margin": float(np.quantile(margins, 0.90)),
        "mean_reference_q_margin_on_disagreements": ref_disagreement_margin_mean,
        "median_reference_q_margin_on_disagreements": ref_disagreement_margin_median,
        "mean_current_q_margin_on_disagreements": current_disagreement_margin_mean,
        "median_current_q_margin_on_disagreements": current_disagreement_margin_median,
        "mean_abs_q_drift_from_reference": float(np.mean(np.abs(q_delta))),
        "q_rmse_from_reference": float(np.sqrt(np.mean(np.square(q_delta)))),
        "mean_max_q": float(np.max(values, axis=1).mean()),
    }
    for action_index, amount in enumerate(action_amounts):
        row[f"action_{amount}_share_pct"] = float(
            np.mean(predicted_actions == action_index) * 100.0
        )
    return row, predicted_actions, margins


def main():
    args = parse_args()
    if args.episodes < 1:
        raise ValueError("--episodes must be >= 1")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.reference_steps not in args.checkpoint_steps:
        raise ValueError("--reference-steps must also be listed in --checkpoint-steps.")
    if len(set(args.checkpoint_steps)) != len(args.checkpoint_steps):
        raise ValueError("--checkpoint-steps must not contain duplicates.")

    root = project_root()
    run_dir = root / "models" / "portfolio_dqn" / args.run_name
    config_path = run_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing run config: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    if int(config.get("environment_model_version", 1)) < PortfolioLoanEnv.ENVIRONMENT_MODEL_VERSION:
        raise ValueError(
            f"This diagnostic requires environment model version "
            f">= {PortfolioLoanEnv.ENVIRONMENT_MODEL_VERSION}; the checkpoint "
            "must use the synchronized customer-observation semantics."
        )

    checkpoint_map = _checkpoint_map(run_dir / "checkpoints", args.run_name)
    missing = [step for step in args.checkpoint_steps if step not in checkpoint_map]
    if missing:
        raise FileNotFoundError(f"Missing checkpoint steps: {missing}")

    dataset_path = args.data or config.get("dataset_path")
    _, validation_raw, _, scaler, split_info = load_split_and_scaler(
        data_path=dataset_path,
        split_seed=int(config["split_seed"]),
    )
    if Path(config["dataset_path"]).resolve() != Path(
        split_info["dataset_path"]
    ).resolve():
        raise ValueError("Dataset path does not match the run config.")

    print("=" * 78)
    print("DQN Q-VALUE AND ACTION-MARGIN STABILITY DIAGNOSTIC")
    print("=" * 78)
    print(f"Run                  : {args.run_name}")
    print(f"Validation rows      : {len(validation_raw):,}")
    print(f"Reference checkpoint : {args.reference_steps:,}")
    print(f"Rollout episodes     : {args.episodes}")
    print(f"Compared checkpoints : {sorted(args.checkpoint_steps)}")
    print("Collecting fixed observations...")

    reference_model = DQN.load(str(checkpoint_map[args.reference_steps]), device="auto")
    observations, metadata = collect_reference_observations(
        model=reference_model,
        config=config,
        raw_df=validation_raw,
        scaler=scaler,
        episodes=args.episodes,
        eval_seed=args.eval_seed,
        reference_steps=args.reference_steps,
    )
    reference_values = q_values_for_observations(
        reference_model, observations, args.batch_size
    )
    reference_actions = np.argmax(reference_values, axis=1)
    del reference_model

    action_amounts = np.asarray([0, 100_000_000, 200_000_000,
                                 300_000_000, 400_000_000, 500_000_000],
                                dtype=np.int64)
    summary_rows = []
    pairwise_rows = []
    q_data = metadata.copy()

    action_by_step = {}
    values_by_step = {}
    margins_by_step = {}

    for steps in sorted(args.checkpoint_steps):
        model = DQN.load(str(checkpoint_map[steps]), device="auto")
        values = q_values_for_observations(model, observations, args.batch_size)
        actions = np.argmax(values, axis=1)
        row, actions, margins = summarize_q(
            values=values,
            steps=steps,
            reference_values=reference_values,
            reference_actions=reference_actions,
            action_amounts=action_amounts,
        )
        row["reference_checkpoint_steps"] = args.reference_steps
        summary_rows.append(row)
        action_by_step[steps] = actions
        values_by_step[steps] = values
        margins_by_step[steps] = margins
        q_data[f"action_steps_{steps}"] = actions
        q_data[f"q_margin_steps_{steps}"] = margins
        for action_index, amount in enumerate(action_amounts):
            q_data[f"q_action_{amount}_steps_{steps}"] = values[:, action_index]
        print(
            f"steps={steps:>8,} | agreement={row['action_agreement_with_reference_pct']:.2f}% | "
            f"Q margin median={row['median_top1_top2_q_margin']:.4f} | "
            f"Q RMSE vs reference={row['q_rmse_from_reference']:.4f}"
        )
        del model

    for steps_a in sorted(values_by_step):
        for steps_b in sorted(values_by_step):
            values_a = values_by_step[steps_a]
            values_b = values_by_step[steps_b]
            actions_a = action_by_step[steps_a]
            actions_b = action_by_step[steps_b]
            disagreement = actions_a != actions_b
            margins_a = margins_by_step[steps_a]
            margins_b = margins_by_step[steps_b]
            pairwise_rows.append({
                "checkpoint_a_steps": steps_a,
                "checkpoint_b_steps": steps_b,
                "same_action_pct": float(np.mean(~disagreement) * 100.0),
                "mean_abs_action_difference_million_toman": float(
                    np.mean(np.abs(action_amounts[actions_a] - action_amounts[actions_b]))
                    / 1_000_000.0
                ),
                "mean_abs_q_difference_all_actions": float(
                    np.mean(np.abs(values_a - values_b))
                ),
                "q_rmse_all_actions": float(
                    np.sqrt(np.mean(np.square(values_a - values_b)))
                ),
                "mean_margin_a_on_disagreements": float(
                    margins_a[disagreement].mean()
                ) if np.any(disagreement) else np.nan,
                "mean_margin_b_on_disagreements": float(
                    margins_b[disagreement].mean()
                ) if np.any(disagreement) else np.nan,
            })

    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / f"{args.run_name}_q_stability_summary.csv"
    pairwise_path = output_dir / f"{args.run_name}_q_stability_pairwise.csv"
    per_state_path = output_dir / f"{args.run_name}_q_stability_per_state.csv"
    pd.DataFrame(summary_rows).sort_values("checkpoint_steps").to_csv(
        summary_path, index=False
    )
    pd.DataFrame(pairwise_rows).to_csv(pairwise_path, index=False)
    q_data.to_csv(per_state_path, index=False)

    print("\nSaved:")
    print(f"  Summary:   {summary_path}")
    print(f"  Pairwise:  {pairwise_path}")
    print(f"  Per-state: {per_state_path}")
    print(
        "\nInterpretation: low Q margins on disagreement states suggest competing "
        "actions may be near-ties; large margins and substantial Q drift suggest "
        "more consequential policy instability. This remains a diagnostic on a "
        "reference-policy validation panel, not a convergence proof."
    )


if __name__ == "__main__":
    main()
