"""Evaluate saved DQN checkpoints on matched validation episodes.

Train with --checkpoint-freq 5000 (or another suitable interval), then run:
    python env/evaluate_portfolio_dqn_checkpoints.py --run-name RUN_NAME

All checkpoints are assessed with the same validation rows and episode seeds.
This makes the resulting reward curve useful for examining learning stability
without rerunning multiple training seeds.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import DQN

from evaluate_portfolio_dqn import evaluate_episode
from portfolio_experiment_utils import load_split_and_scaler, project_root


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate checkpoints from one DQN training run on matched validation episodes."
    )
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--data", default=None)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--eval-seed", type=int, default=20_000)
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument(
        "--output-dir",
        default="models/results/portfolio_dqn/convergence",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.episodes < 1:
        raise ValueError("--episodes must be >= 1")

    root = project_root()
    run_dir = root / "models" / "portfolio_dqn" / args.run_name
    config_path = run_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing run config: {config_path}")

    with config_path.open("r", encoding="utf-8") as config_file:
        config = json.load(config_file)
    if int(config.get("environment_model_version", 1)) < 5:
        raise ValueError(
            "This run predates the configurable reapplication process. "
            "Retrain with the current env/train_portfolio_dqn.py first."
        )

    checkpoint_dir = (
        Path(args.checkpoint_dir).expanduser()
        if args.checkpoint_dir
        else run_dir / "checkpoints"
    )
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = root / checkpoint_dir
    checkpoints = list(checkpoint_dir.glob(f"{args.run_name}_*_steps.zip"))
    checkpoints = [
        (int(match.group(1)), path)
        for path in checkpoints
        if (match := re.search(r"_(\d+)_steps\.zip$", path.name))
    ]
    checkpoints.sort(key=lambda item: item[0])
    if not checkpoints:
        raise FileNotFoundError(
            f"No checkpoint files matching {args.run_name}_*_steps.zip in "
            f"{checkpoint_dir}. Retrain with --checkpoint-freq 5000 (or another "
            "interval) to inspect convergence."
        )

    dataset_path = args.data or config.get("dataset_path")
    train_raw, validation_raw, test_raw, scaler, split_info = load_split_and_scaler(
        data_path=dataset_path,
        split_seed=int(config["split_seed"]),
    )
    if Path(config["dataset_path"]).resolve() != Path(split_info["dataset_path"]).resolve():
        raise ValueError(
            "The supplied dataset differs from the dataset used for training."
        )
    evaluation_raw = validation_raw if args.split == "validation" else test_raw

    seeds = [args.eval_seed + i for i in range(args.episodes)]
    episode_rows = []
    summary_rows = []

    print("=" * 76)
    print("PORTFOLIO DQN CHECKPOINT / CONVERGENCE EVALUATION")
    print("=" * 76)
    print(f"Run name       : {args.run_name}")
    print(f"Dataset        : {split_info['dataset_path']}")
    print(f"Evaluation split: {args.split} ({len(evaluation_raw):,} rows)")
    print(f"Episodes/point : {args.episodes}")
    print(f"Matched seeds  : {seeds[0]}..{seeds[-1]}")
    print(f"Checkpoints    : {len(checkpoints)}")
    print("-" * 76)

    for checkpoint_steps, checkpoint_path in checkpoints:
        model = DQN.load(str(checkpoint_path), device="auto")
        results = []
        for episode_index, seed in enumerate(seeds, start=1):
            result = evaluate_episode(
                run_name=f"{args.run_name}@{checkpoint_steps}",
                model=model,
                config=config,
                raw_df=evaluation_raw,
                scaler=scaler,
                episode=episode_index,
                seed=seed,
            )
            result["checkpoint_steps"] = checkpoint_steps
            result["checkpoint_path"] = str(checkpoint_path)
            result["split"] = args.split
            episode_rows.append(result)
            results.append(result)

        group = pd.DataFrame(results)
        row = {
            "run_name": args.run_name,
            "checkpoint_steps": checkpoint_steps,
            "checkpoint_path": str(checkpoint_path),
            "split": args.split,
            "episodes": len(group),
        }
        for metric in (
            "total_reward_as_trained",
            "total_reward_under_legacy_formula",
            "total_reward_under_amount_sensitive_formula",
            "social_reward_amount_sensitive",
            "financial_pnl",
            "total_risk_penalty",
            "total_recognized_default_penalty",
            "recognized_default_count",
            "loans_granted",
            "loan_approval_rate",
            "average_loan_size",
            "invalid_actions",
            "reapplications_seen",
            "customers_with_reapplications",
            "final_cash",
        ):
            row[f"mean_{metric}"] = float(group[metric].mean())
            row[f"std_{metric}"] = (
                float(group[metric].std(ddof=1))
                if len(group) > 1 else 0.0
            )
        summary_rows.append(row)
        print(
            f"steps={checkpoint_steps:>8,} | "
            f"mean reward={row['mean_total_reward_as_trained']:.3f} "
            f"(std={row['std_total_reward_as_trained']:.3f}) | "
            f"loans={row['mean_loans_granted']:.2f} | "
            f"reapplications={row['mean_reapplications_seen']:.2f}"
        )

    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    episode_output = output_dir / f"{args.run_name}_{args.split}_checkpoint_episodes.csv"
    summary_output = output_dir / f"{args.run_name}_{args.split}_convergence.csv"
    pd.DataFrame(episode_rows).to_csv(episode_output, index=False)
    summary_df = pd.DataFrame(summary_rows).sort_values("checkpoint_steps")
    best_index = summary_df["mean_total_reward_as_trained"].idxmax()
    summary_df["best_by_validation_mean_reward"] = False
    summary_df.loc[best_index, "best_by_validation_mean_reward"] = True
    summary_df.to_csv(summary_output, index=False)

    best_row = summary_df.loc[best_index]
    print()
    print(
        "Highest mean reward on the selected split: "
        f"{int(best_row['checkpoint_steps']):,} steps "
        f"(mean={best_row['mean_total_reward_as_trained']:.3f})."
    )
    print("Use validation as the selection split; reserve test results for final reporting.")
    print("Saved:")
    print(f"  Episode details: {episode_output}")
    print(f"  Convergence:     {summary_output}")


if __name__ == "__main__":
    main()
