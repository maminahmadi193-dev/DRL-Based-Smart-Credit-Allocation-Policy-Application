# Project Structure and Stabilization Notes

**Audit snapshot:** branch `feature/portfolio-rl`, commit `80b3fa1` (2026-10-19).  
**Purpose:** make the working implementation reproducible before training budget-specific policies at 25B, 50B, 75B, and 100B.

## 1. Current state observed

- The work-PC working tree was aligned with `origin/feature/portfolio-rl`.
- The only untracked paths reported at the time of the audit were local run/output directories under `models/portfolio_dqn/` and `models/results/`.
- `python tests/run_tests.py`: **53 tests passed** in the combined suite (verified on 2026-10-19).
- Gymnasium's checker warned that the observation-space `Box` has infinite lower/upper bounds. The affected tests passed; the warning is noted, not silently suppressed or fixed here.
- The data file `notebooks/df_cleaned.csv` is local and is not expected in Git.

These results describe the reported work-PC run only; they are not a CI guarantee.

## 2. Current inventory and remaining review

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

- Training: `scripts/train_portfolio_dqn.py`
- Evaluation: `scripts/evaluate_portfolio_dqn.py`, `scripts/evaluate_portfolio_dqn_checkpoints.py`, `scripts/evaluate_baseline_policies.py`
- Cash-flow trace: `scripts/trace_portfolio_cashflows.py`
- Budget sensitivity: `scripts/analyze_budget_sensitivity.py`

### Portfolio diagnostics - scripts

- `scripts/analyze_checkpoint_q_stability.py`
- `scripts/analyze_immediate_reward_by_amount.py`
- `scripts/analyze_social_reward_components.py`
- `scripts/compare_checkpoint_actions_fixed_states.py`

### Tests

- Core regression tests: `test_reward_engine.py`, `test_portfolio_simulator.py`, and `test_portfolio_loan_env.py` are in `tests/`. Run all tests with `python tests/run_tests.py`.
- The baseline-policy and exact-reference benchmark tests are also in `tests/`.
- `scripts/analyze_legacy_dqn_convergence.py` and `scripts/analyze_legacy_dqn_stability.py` are research/evaluation scripts, not automated unit tests. They may depend on the local dataset and legacy model checkpoints.

### Legacy single-customer diagnostic scripts in `scripts/`

- `scripts/analyze2.py` (offline reward/regret post-processing)
- `scripts/analyze_reward_landscape.py`
- `scripts/analyze_risk_landscape.py`
- `scripts/inspect_reward.py`

These scripts depend on the legacy single-customer environment or its result files. Their calculations and local data/result paths are intentionally unchanged. They are not targets for automated smoke execution.

### Legacy single-customer experiment entry points - in `scripts/`

- `scripts/train_legacy_dqn.py`
- `scripts/analyze_legacy_dqn_convergence.py`
- `scripts/analyze_legacy_dqn_stability.py`
- `scripts/train_legacy_policy_classifier.py`
- `scripts/train_legacy_q_regression.py`

These preserve earlier experiment workflows. Their training calculations, evaluation logic, dataset paths, and model/result output locations are intentionally unchanged. They are not part of the current portfolio training pipeline.

### Remaining analysis and support modules - keep in `env/` for now

- `coverage_dqn.py` (support module for the legacy DQN trainer)
- `exact_reference_mdp.py` (retained because the benchmark tests depend on it)
- `pd_calibration.py` (currently a placeholder)
- `loan_env.py` (legacy single-customer environment)

These files are not marked for deletion. Review their usage and historical role separately before considering any removal.

## 3. Portfolio reward design

The portfolio-level pipeline now has one fixed amount-sensitive social reward. `RewardEngine` and `PortfolioLoanEnv` do not expose a Legacy/experimental selector. Training writes `reward_mode: amount_sensitive`; evaluation and diagnostics use that same formula. The historical amount-independent access bonus is isolated in `LegacySingleCustomerRewardEngine` for the older single-customer `LoanEnv` and legacy classifier workflow. Environment model version 12 marks the reward-semantics change; pre-v12 portfolio checkpoints must be retrained before evaluation.

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

## 5. Stabilization status and remaining sequence

1. **Complete:** protect local model runs and generated results from accidental Git staging.
2. **Documented; verify at baseline freeze:** preserve the environment behavior, reward calculation, and stochastic missed-payment/collection assumptions.
3. **Complete for the main entry-point migration:** tests and experiment scripts have been inventoried; review remaining support-module dependencies before any further move or deletion.
4. **Complete:** move regression tests into `tests/` and experiment entry points and diagnostics into `scripts/`, with imports and command examples updated in separate commits.
5. **Passed for the current code snapshot:** 53 local regression tests pass, and GitHub Actions has passed compilation, regression, baseline-policy smoke, and cash-flow trace smoke stages.
6. **Next:** capture a named baseline commit and configuration before any budget-specific training.
7. **Pending:** train separate policies for the four budget levels and evaluate them on the same validation protocol. Do not use the test split to select budgets or models.

## 6. Explicitly deferred items

- Do not alter the reward formula, hazard calibration, payment simulation, loan eligibility, or observation vector as part of a directory cleanup.
- Do not remove old scripts just because their names look exploratory.
- Do not delete checkpoints/results until their run provenance has been recorded and the user has chosen an archive/retention policy.
- The Gymnasium infinite-Box warning should be assessed against actual observation values before changing bounds; it is a separate code-quality change.
