# Stochastic missed-installment model

## Purpose and scope

The portfolio simulator models a loan as a sequence of monthly installment due dates. On each due date, an installment is either paid in full or not paid at all. It is never split into fractional on-time and late shares.

This is a deliberately limited, reproducible experiment model. It does not estimate applicant-level default labels or fit a predictive PD model from observed repayment histories; it uses the aggregate rate only to calibrate the population level.

## Risk model and population calibration

After a loan is approved, the current customer-state and risk update is applied. The updated risk score is mapped to a monthly hazard with a logistic curve and a calibratable intercept:

`h_i = max_hazard * sigmoid(hazard_slope * (risk_i - hazard_center) + hazard_intercept)`

where `sigmoid(x) = 1 / (1 + exp(-x))`. The additive intercept changes the population-level level of risk while preserving the monotonic ordering of customers by risk. The hazard is bounded above by `max_hazard`.

The probability of a missed installment at a monthly checkpoint is:

`q_i = 1 - exp(-h_i)`

and the probability of at least one missed installment during a contractual term of `T` months is:

`PD_i(T) = 1 - exp(-h_i * T) = 1 - (1 - q_i)^T`.

### Current operational calibration

The bank-provided 1.3% figure is treated as the **annual population-level default rate**, and calibration therefore targets a 12-month horizon, independent of the chosen contractual term. For the reference exposure, the default is a hypothetical 100M loan; each row's post-loan risk is calculated, then `hazard_intercept` is solved so mean 12-month default probability is 1.3%.

For a longer or shorter contract, term probability is derived from the calibrated monthly hazard: `PD_i(T) = 1 - exp(-h_i * T)`. Consequently, an 18-month contract will generally have a higher cumulative term probability than its 12-month probability. The calibration does not need individual default labels and does not manufacture a label for any row. It matches the mean model-implied annual probability for the reference population, not the realized count in every episode. Different actions and selected customer mixes can move a portfolio's rate away from the benchmark; that is intentional.

The 100M reference amount is a modeling convention, not something implied by the bank-wide statistic. The curve slope, hazard ceiling, and post-loan risk relationship remain assumptions. A population annual rate is useful as a level calibration, but it does not validate applicant-level risk ordering or imply every product has identical risk. If 1.3% is a ratio of overdue balances rather than an annual customer/contract default rate, its denominator and event definition need to be checked before treating it as an annual PD.

## Missed installment and default recognition

The current simplified contract behavior allows at most one missed installment per loan. If one installment is missed:

1. The entire principal-and-interest amount for that installment is held as an overdue receivable. Nothing from that installment is counted as cash collected at the due date.
2. Later scheduled installments are assumed to be paid in full on their due dates.
3. Default is recognized at the first monthly checkpoint where the missed installment is more than `default_dpd_threshold_days` past due. The default definition defaults to strictly more than 90 calendar days.
4. The missed installment is collected in full after `default_collection_lag_months` from its due month. The base setting is 4 months; a stress test can use 12 months.
5. No permanent credit loss is modeled. Recognizing a default is a delinquency-status event, not a write-off; net default loss remains zero under the full-collection assumption.

Calendar checkpoints are currently represented by the first day of each simulated month. Therefore, the threshold is checked monthly, not daily. A four-month lag takes the receipt beyond 90 days past due at the checkpoint cadence.

## Randomness and comparison

`PortfolioLoanEnv.reset(seed=...)` seeds the payment-event generator independently of applicant/cohort sampling. Identical environment seeds therefore reuse the same sampled cohorts and the same repayment-event random stream, which supports matched-seed policy comparisons. Different episode seeds are needed to measure variation in outcomes.

## Assumptions and limitations

- Only the first missed installment is simulated for each loan; all later installments are assumed on time. This is a simplification, not a claim about real customer behavior.
- The contractual amortization schedule continues after a missed installment. Its overdue principal-and-interest amount is tracked separately until collected.
- The default definition uses 90+ days past due, while local past-due / overdue / doubtful classifications are separate concepts and are not implemented as status tiers here.
- The logistic hazard intercept is calibrated to a population-level 1.3% reference term PD using customer features only. The slope, hazard ceiling, post-loan risk relationship, and interpretation of the 1.3% benchmark remain assumptions; this is not applicant-level predictive validation.
- Interest accrues only through the currently specified contractual schedule; the delayed installment is not assessed additional penalty interest in this version.

## Configuration

The default settings are exposed by the training, baseline, and trace scripts:

- `--loan-term-months`: 12, 18, 24, or 36 months; one term per experiment.
- `--annual-interest-rate`: nominal annual rate, 4% by default.
- `--max-hazard`: maximum monthly hazard, 0.005 by default.
- `--target-term-default-rate`: calibrated mean cumulative reference term PD, 0.013 by default.
- `--reference-loan-amount`: exposure defining the calibration population, 100M by default.
- `--default-dpd-threshold-days`: 90 by default; recognition uses strictly greater than this threshold.
- `--default-collection-lag-months`: 4 by default, with 12 for the stress scenario.

The single-loan stress trace forces a missed installment using a very high hazard. Example commands from the repository root:

```powershell
python env/test_portfolio_simulator.py
python env/test_portfolio_loan_env.py
python env/test_reward_engine.py
python env/trace_portfolio_cashflows.py --data "C:\path\to\df_cleaned.csv" --stress-max-delay --output "models/results/portfolio_cashflow_stress.csv"
python env/train_portfolio_dqn.py --data "C:\path\to\df_cleaned.csv" --run-name portfolio_dqn_stochastic_payments_smoke --timesteps 5000 --seed 42
```

Models trained before environment model version 3 do not include the population-calibrated hazard intercept and must be retrained before comparison under these dynamics.
