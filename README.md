# Smart Credit Allocation with Portfolio-Level Deep Reinforcement Learning

This repository contains an evolving research implementation for credit-allocation policies using a portfolio-level Gymnasium environment and Stable-Baselines3 DQN.

> **Development status:** the portfolio simulator and environment are the current implementation under active development. Older single-customer experiments and one-off diagnostic scripts are still present and are being catalogued before any move or deletion.

## Current portfolio-level implementation

The core pieces are:

- `env/customer_state.py` — persistent customer state.
- `env/customer_simulator.py` — customer-level feature/risk behavior.
- `env/reward_engine.py` — reward components.
- `env/portfolio_simulator.py` — shared cash, contracts, installments, missed payments, collections, and portfolio accounting.
- `env/portfolio_loan_env.py` — the Gymnasium environment for portfolio-level allocation.
- `env/portfolio_experiment_utils.py` — data splitting, feature scaling, and shared experiment utilities.
- `env/train_portfolio_dqn.py` — DQN training entry point.
- `env/evaluate_portfolio_dqn.py`, `env/evaluate_portfolio_dqn_checkpoints.py`, and `env/evaluate_baseline_policies.py` — evaluation entry points.
- `env/analyze_budget_sensitivity.py` — evaluates one frozen checkpoint across initial-budget scenarios; it does not train a model.

## Data

The real dataset is intentionally not stored in Git. Place the approved local dataset at:

```text
notebooks/df_cleaned.csv
```

Use the `--data` argument when running an experiment with a different local path. Do not commit customer-level data or generated model artifacts.

## Running the focused tests

Run these commands from the repository root in the project's Python environment:

```powershell
python env/test_reward_engine.py
python env/test_portfolio_simulator.py
python env/test_portfolio_loan_env.py
```

The focused simulator and environment tests passed on the work PC on **2026-10-19** (16 simulator tests and 22 environment tests). Gymnasium emitted warnings that the observation-space Box uses infinite bounds; the tests still passed. This warning remains an open engineering item and should be evaluated separately from the simulator/reward logic.

## Training and evaluation discipline

- Give each training run a unique, descriptive `--run-name`.
- Retain its `config.json`, checkpoints, logs, and evaluation summaries together in the local run directory.
- Use validation data for model selection and budget/hyperparameter tuning. Keep the test split out of tuning.
- For fair comparisons, record the code commit, dataset identity, split seed, training seed, environment settings, and evaluation seeds.
- When comparing different budget-specific policies, train separate models and evaluate each on a matched validation protocol.

Generated runs and result bundles under `models/portfolio_dqn/` and `models/results/` are local artifacts and are ignored by Git. Existing tracked legacy artifacts will be reviewed separately; this rule does not automatically untrack them.

## Repository organization

The current repository is being refactored in small, testable steps. See [the project structure and stabilization notes](docs/project-structure-and-stabilization.md) for the provisional file inventory, known issues, and migration plan. No core environment or reward behavior is changed by the initial documentation/ignore-file pass.
