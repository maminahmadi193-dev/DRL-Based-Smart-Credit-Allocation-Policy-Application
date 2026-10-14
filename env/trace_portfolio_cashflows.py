"""Trace one loan from origination through final delayed receipt.

Run from the repository root:
    python env/trace_portfolio_cashflows.py
    python env/trace_portfolio_cashflows.py --stress-max-delay
"""
import argparse
import math
from pathlib import Path

import pandas as pd

from portfolio_experiment_utils import (
    calibrate_hazard_intercept_for_population,
    load_split_and_scaler,
    project_root,
)
from portfolio_loan_env import PortfolioLoanEnv


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run one 100M loan and verify cash-flow reconciliation."
    )
    parser.add_argument("--data", default=None, help="Path to df_cleaned.csv.")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--loan-term-months", type=int, choices=[12, 18, 24, 36], default=18)
    parser.add_argument("--annual-interest-rate", type=float, default=0.04)
    parser.add_argument("--default-collection-lag-months", type=int, choices=range(4, 13), default=4)
    parser.add_argument("--default-dpd-threshold-days", type=int, default=90)
    parser.add_argument("--hazard-slope", type=float, default=0.08)
    parser.add_argument("--hazard-center", type=float, default=60.0)
    parser.add_argument("--max-hazard", type=float, default=0.005)
    parser.add_argument("--target-term-default-rate", type=float, default=0.013)
    parser.add_argument("--reference-loan-amount", type=float, default=100_000_000)
    parser.add_argument(
        "--stress-max-delay",
        action="store_true",
        help="Force a missed installment and collect it 12 months later.",
    )
    parser.add_argument(
        "--output",
        default="models/results/portfolio_cashflow_trace.csv",
        help="CSV output path. Existing files are never overwritten.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not math.isfinite(args.annual_interest_rate) or args.annual_interest_rate < 0:
        raise ValueError("--annual-interest-rate must be finite and non-negative.")
    if args.default_dpd_threshold_days < 1:
        raise ValueError("--default-dpd-threshold-days must be >= 1.")
    if not 4 <= args.default_collection_lag_months <= 12:
        raise ValueError("--default-collection-lag-months must be between 4 and 12.")
    if not math.isfinite(args.hazard_slope) or args.hazard_slope < 0:
        raise ValueError("--hazard-slope must be finite and non-negative.")
    if not math.isfinite(args.hazard_center):
        raise ValueError("--hazard-center must be finite.")
    if not math.isfinite(args.max_hazard) or args.max_hazard < 0:
        raise ValueError("--max-hazard must be finite and non-negative.")

    base_dir = project_root()
    output_path = Path(args.output).expanduser()
    if not output_path.is_absolute():
        output_path = (base_dir / output_path).resolve()
    if output_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing trace: {output_path}. "
            "Choose another --output path."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Use the held-out test rows for the single trace. The calibration itself
    # uses only population risk/exposure features, not applicant default labels.
    train_raw, validation_raw, test_raw, scaler, split_info = load_split_and_scaler(
        data_path=args.data,
        split_seed=42,
    )

    collection_lag = 12 if args.stress_max_delay else args.default_collection_lag_months
    maximum_hazard = 1_000.0 if args.stress_max_delay else args.max_hazard
    hazard_intercept = 0.0
    hazard_calibration = None
    if not args.stress_max_delay:
        calibration_population = pd.concat(
            [train_raw, validation_raw, test_raw],
            ignore_index=True,
        )
        hazard_calibration = calibrate_hazard_intercept_for_population(
            raw_df=calibration_population,
            target_term_default_rate=args.target_term_default_rate,
            reference_loan_amount=args.reference_loan_amount,
            loan_term_months=args.loan_term_months,
            max_hazard=maximum_hazard,
            hazard_slope=args.hazard_slope,
            hazard_center=args.hazard_center,
        )
        hazard_intercept = float(hazard_calibration["hazard_intercept"])

    env = PortfolioLoanEnv(
        raw_df=test_raw,
        scaler=scaler,
        initial_budget=100_000_000,
        monthly_capital_injection=0.0,
        applicants_per_month=1,
        min_applicants_per_month=1,
        max_applicants_per_month=1,
        origination_months=1,
        loan_term_months=args.loan_term_months,
        annual_interest_rate=args.annual_interest_rate,
        default_collection_lag_months=collection_lag,
        default_dpd_threshold_days=args.default_dpd_threshold_days,
        hazard_slope=args.hazard_slope,
        hazard_center=args.hazard_center,
        max_hazard=maximum_hazard,
        hazard_intercept=hazard_intercept,
        social_reward_weight=1.0,
        financial_reward_weight=1.0,
        financial_reward_scale=100_000_000,
        invalid_action_penalty=0.25,
    )

    history = []
    initial_cash = env.initial_budget
    requested_principal = 100_000_000.0
    try:
        observation, _ = env.reset(seed=args.seed)
        if args.stress_max_delay:
            # Test-only stress mode: a very high hazard forces the first
            # scheduled installment to be missed with probability effectively 1.
            env.current_customer.risk = 100.0

        terminated = False
        max_steps = (
            env.max_applicants_per_month * env.origination_months
            + env.loan_term_months
            + env.portfolio.max_delay_months
            + 5
        )

        for step_number in range(1, max_steps + 1):
            cash_before_step = env.portfolio.available_budget
            was_settlement = env.phase == "settlement"
            action = 0 if was_settlement else 1  # one 100M loan, then no-op settlement
            observation, reward, terminated, truncated, info = env.step(action)
            portfolio = env.portfolio
            monthly_principal = float(info.get("principal_collected", 0.0))
            monthly_interest = float(info.get("interest_collected", 0.0))
            approved = float(info.get("approved_amount", 0.0))
            monthly_result_cash_before = cash_before_step - approved
            monthly_result_cash_after = float(portfolio.available_budget)

            history.append({
                "step": step_number,
                "month_after_step": int(portfolio.month),
                "phase_after_step": env.phase,
                "action_ignored": bool(info.get("action_ignored", False)),
                "approved_amount": approved,
                "cash_before_monthly_settlement": monthly_result_cash_before,
                "cash_after_step": monthly_result_cash_after,
                "cumulative_disbursed": float(portfolio.total_disbursed),
                "principal_collected_this_month": monthly_principal,
                "interest_collected_this_month": monthly_interest,
                "late_amount_scheduled_this_month": float(
                    info.get("delayed_payment_amount_scheduled", 0.0)
                ),
                "late_amount_collected_this_month": float(
                    info.get("delayed_payment_amount_collected", 0.0)
                ),
                "pending_late_receivables": float(
                    portfolio.pending_delayed_receivables
                ),
                "active_contracts": int(portfolio.active_contract_count),
                "recognized_defaults_this_month": int(
                    info.get("recognized_default_count", 0)
                ),
                "cumulative_recognized_defaults": int(
                    portfolio.cumulative_recognized_defaults
                ),
                "cumulative_missed_installments": int(
                    portfolio.cumulative_missed_installments
                ),
                "cumulative_principal_collected": float(
                    portfolio.cumulative_principal_collected
                ),
                "cumulative_interest_collected": float(
                    portfolio.cumulative_interest_collected
                ),
                "cumulative_late_amount_scheduled": float(
                    portfolio.cumulative_delayed_payment_amount_scheduled
                ),
                "cumulative_late_amount_collected": float(
                    portfolio.cumulative_delayed_payment_amount_collected
                ),
                "expected_outstanding_principal_including_late_principal": float(
                    portfolio.expected_outstanding_principal
                ),
                "reward": float(reward),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
            })

            if terminated or truncated:
                break
        else:
            raise RuntimeError(
                f"Episode did not terminate within {max_steps} steps."
            )

        portfolio = env.portfolio
        df = pd.DataFrame(history)
        df.to_csv(output_path, index=False, encoding="utf-8-sig")

        expected_final_cash = (
            initial_cash
            - portfolio.total_disbursed
            + portfolio.cumulative_principal_collected
            + portfolio.cumulative_interest_collected
        )
        cash_error = portfolio.available_budget - expected_final_cash
        principal_error = (
            portfolio.cumulative_principal_collected - portfolio.total_disbursed
        )
        delayed_balance_error = (
            portfolio.cumulative_delayed_payment_amount_scheduled
            - portfolio.cumulative_delayed_payment_amount_collected
        )

        actual_contract = portfolio.contracts[0] if portfolio.contracts else None
        monthly_miss_probability = (
            actual_contract.monthly_miss_probability
            if actual_contract is not None
            else math.nan
        )

        print("=" * 72)
        print("SINGLE-LOAN CASH-FLOW TRACE")
        print("=" * 72)
        print(f"Dataset rows used       : {split_info['dataset_rows_used']:,}")
        print(f"Seed                    : {args.seed}")
        print(f"Loan term               : {args.loan_term_months} months")
        print(f"Annual nominal rate     : {args.annual_interest_rate:.2%}")
        print(f"Monthly nominal rate    : {args.annual_interest_rate / 12:.6%}")
        print(f"Forced missed-payment stress: {args.stress_max_delay}")
        print(f"Monthly miss probability: {monthly_miss_probability:.4%}")
        print(f"Maximum monthly hazard : {maximum_hazard:.6f}")
        print(f"Default DPD threshold  : >{args.default_dpd_threshold_days} days")
        print(f"Collection lag         : {collection_lag} months")
        print(f"Episode ended           : {terminated}")
        print(f"Final month             : {portfolio.month}")
        print(f"Loan principal          : {portfolio.total_disbursed:,.2f}")
        print(f"Principal collected     : {portfolio.cumulative_principal_collected:,.2f}")
        print(f"Interest collected      : {portfolio.cumulative_interest_collected:,.2f}")
        print(f"Late amount scheduled   : {portfolio.cumulative_delayed_payment_amount_scheduled:,.2f}")
        print(f"Late amount collected   : {portfolio.cumulative_delayed_payment_amount_collected:,.2f}")
        print(f"Missed installments     : {portfolio.cumulative_missed_installments}")
        print(f"Recognized defaults     : {portfolio.cumulative_recognized_defaults}")
        print(f"Late receivables left   : {portfolio.pending_delayed_receivables:,.8f}")
        print(f"Final cash              : {portfolio.available_budget:,.2f}")
        print(f"Expected final cash     : {expected_final_cash:,.2f}")
        print(f"Cash reconciliation err : {cash_error:,.8f}")
        print(f"Principal balance err   : {principal_error:,.8f}")
        print(f"Late balance err        : {delayed_balance_error:,.8f}")
        print(f"CSV trace               : {output_path}")
        print()
        print(df.to_string(index=False, float_format=lambda value: f"{value:,.2f}"))

        if not terminated:
            raise AssertionError("Expected the episode to terminate.")
        if portfolio.active_contract_count != 0:
            raise AssertionError("Active contracts remain after episode termination.")
        if portfolio.pending_delayed_receivables > 1e-5:
            raise AssertionError("Delayed receivables remain after episode termination.")
        if not math.isclose(
            portfolio.cumulative_principal_collected,
            portfolio.total_disbursed,
            rel_tol=1e-8,
            abs_tol=1e-4,
        ):
            raise AssertionError("Collected principal does not match total disbursement.")
        if not math.isclose(
            portfolio.cumulative_delayed_payment_amount_scheduled,
            portfolio.cumulative_delayed_payment_amount_collected,
            rel_tol=1e-8,
            abs_tol=1e-4,
        ):
            raise AssertionError("Scheduled late amounts do not match collected late amounts.")
        if not math.isclose(
            portfolio.available_budget,
            expected_final_cash,
            rel_tol=1e-8,
            abs_tol=1e-4,
        ):
            raise AssertionError("Final cash does not reconcile with the cash-flow identity.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
