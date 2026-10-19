# Project Structure and Stabilization Notes

**Audit snapshot:** branch `feature/portfolio-rl`, commit `80b3fa1` (2026-10-19).  
**Purpose:** make the working implementation reproducible before training budget-specific policies at 25B, 50B, 75B, and 100B.

## 1. Current state observed

- The work-PC working tree was aligned with `origin/feature/portfolio-rl`.
- The only untracked paths reported at the time of the audit were local run/output directories under `models/portfolio_dqn/` and `models/results/`.
- `python env/test_portfolio_simulator.py`: **16 tests passed**.
- `python env/test_portfolio_loan_env.py`: **22 tests passed**.
- Gymnasium's checker warned that the observation-space `Box` has infinite lower/upper bounds. The affected tests passed; the warning is noted, not silently suppressed or fixed here.
- The data file `notebooks/df_cleaned.csv` is local and is not expected in Git.

These results describe the reported work-PC run only; they are not a CI guarantee.

## 2. Provisional inventory

The groupings below are based on current usage and filenames. Before a file is moved or deleted, inspect its imports, callers, purpose, and whether it is needed to reproduce a previous result.

### Portfolio environment and shared logic — keep as core

- `customer_state.py`
- `customer_simulator.py`
- `reward_engine.py`
- `simulation_outcome.py`
- `portfolio_simulator.py`
- `portfolio_loan_env.py`
- `portfolio_experiment_utils.py`

### Main portfolio experiment entry points

- Training: `train_portfolio_dqn.py`
- Evaluation: `evaluate_portfolio_dqn.py`, `evaluate_portfolio_dqn_checkpoints.py`, `evaluate_baseline_policies.py`
- Cash-flow trace: `trace_portfolio_cashflows.py`
- Budget sensitivity: `analyze_budget_sensitivity.py`

### Tests

- Core tests: `test_reward_engine.py`, `test_portfolio_simulator.py`, `test_portfolio_loan_env.py`
- Other regression/research tests to retain while dependencies are checked: `test_baseline_policy_metrics.py`, `test_exact_reference_mdp.py`, `test_stability.py`
- `test.py` needs inspection to determine whether it is an old test module, an experiment, or both.

### Analysis and earlier experiments — classify before relocating

- `analyze2.py`
- `analyze_checkpoint_q_stability.py`
- `analyze_immediate_reward_by_amount.py`
- `analyze_reward_landscape.py`
- `analyze_risk_landscape.py`
- `analyze_social_reward_components.py`
- `compare_checkpoint_actions_fixed_states.py`
- `coverage_dqn.py`
- `exact_reference_mdp.py`
- `inspect_reward.py`
- `pd_calibration.py`
- `train.py`
- `train_policy_classifier.py`
- `train_q_regression.py`
- `loan_env.py` (legacy single-customer environment candidate)

These files are not marked for deletion. Some may retain important diagnostic logic or support results already cited in the research history.

## 3. Generated files and artifacts

The local `models/portfolio_dqn/` tree contains multiple training runs, checkpoints, and run configurations. The local `models/results/` tree contains budget sensitivity and other evaluation outputs. They should remain on the workstations, not be committed as a bulk directory.

At present, the repository also tracks a few older model and plot artifacts directly under `models/`. Do not delete or untrack these in the first pass. Later, decide explicitly which artifacts are required for reproducibility and either retain selected lightweight artifacts or move the record of the run to a small, version-controlled manifest.

## 4. Intended target layout

The refactor target is:

```text
project/
├── README.md
├── requirements.txt
├── docs/
│   ├── exact-reference-mdp.md
│   ├── stochastic_missed_payment_model.md
│   └── project-structure-and-stabilization.md
├── env/             # environment and simulator implementation
├── tests/           # automated tests
├── scripts/         # training, evaluation, and diagnostic entry points
├── notebooks/       # exploratory work; real dataset stays local
└── models/          # selected versioned artifacts only; generated runs ignored
```

This is a target, not the current on-disk layout. Moving modules can break imports and script execution paths, so perform moves in small batches and run regression tests after each batch.

## 5. Stabilization sequence

1. Protect local model runs and generated results from accidental Git staging.
2. Document the environment's current behavior and reward calculation, including the stochastic missed-payment/collection assumptions.
3. Inventory imports and command-line entry points; identify genuinely obsolete code only after inspecting usage.
4. Move tests to `tests/` and scripts to `scripts/` in separate commits, updating imports/instructions and running the test suite after each move.
5. Run all applicable deterministic tests and short synthetic-data smoke tests.
6. Capture a named baseline commit and configuration before any budget-specific training.
7. Train separate policies for the four budget levels and evaluate on the same validation protocol. Do not use the test split to select budgets or models.

## 6. Explicitly deferred items

- Do not alter the reward formula, hazard calibration, payment simulation, loan eligibility, or observation vector as part of a directory cleanup.
- Do not remove old scripts just because their names look exploratory.
- Do not delete checkpoints/results until their run provenance has been recorded and the user has chosen an archive/retention policy.
- The Gymnasium infinite-Box warning should be assessed against actual observation values before changing bounds; it is a separate code-quality change.
