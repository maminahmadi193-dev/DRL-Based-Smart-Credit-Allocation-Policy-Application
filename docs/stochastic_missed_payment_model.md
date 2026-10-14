# Stochastic missed-installment model

## Purpose and scope

The portfolio simulator models a loan as a sequence of monthly installment due dates. On each due date, an installment is either paid in full or not paid at all. It is never split into fractional on-time and late shares.

This is a deliberately limited, reproducible experiment model. It does not yet estimate risk parameters from observed repayment histories.

## Risk model

After a loan is approved, the existing customer-risk update is applied. The resulting risk score is mapped to a monthly hazard:

`h = max_hazard * sigmoid(hazard_slope * (risk - hazard_center))`

where `sigmoid(x) = 1 / (1 + exp(-x))`.

The event probability for a single installment month is:

`q = 1 - exp(-h)`

and the probability of at least one missed installment during a loan term of `T` months is:

`PD(T) = 1 - exp(-h * T) = 1 - (1 - q)^T`.

The term-specific PD is stored on each loan contract. Actual payments use a seeded random draw with probability `q`; this is not an expected-value split of cash flows. Here, term PD means the probability that an installment due within the contractual term is missed and eventually crosses the DPD threshold. For a missed final installment, default recognition can occur after contractual maturity during the settlement tail.

The environment's next-month expected-collections feature remains an expectation used for decision-making and forecasting. It can represent a probability-weighted expected value, but it is **not booked as a realized receipt**: actual cash collections are recorded only when the full installment is paid or when the full overdue installment is later collected.

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
- The risk-to-hazard mapping and its parameters are illustrative and not empirically calibrated. The model should not be described as having predictive PD until the mapping is calibrated and validated against historical DPD/default labels.
- Interest accrues only through the currently specified contractual schedule; the delayed installment is not assessed additional penalty interest in this version.

## Configuration

The default settings are exposed by the training, baseline, and trace scripts:

- `--loan-term-months`: 12, 18, 24, or 36 months; one term per experiment.
- `--annual-interest-rate`: nominal annual rate, 4% by default.
- `--max-hazard`: maximum monthly hazard, 0.005 by default.
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

Models trained before environment model version 2 used the old fractional-delay observation/reward dynamics and must be retrained before comparison under this model.
