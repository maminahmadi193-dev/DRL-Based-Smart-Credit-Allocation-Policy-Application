"""Trace one loan from origination through final delayed receipt.

Run from the repository root:
    python env/trace_portfolio_cashflows.py
    python env/trace_portfolio_cashflows.py --stress-max-delay
"""
import argparse
import math
from pathlib import Path

import pandas as pd

from portfolio_experiment_utils import load_split_and_scaler, project_root
from portfolio_loan_env import PortfolioLoanEnv


def parse_weights(value):
    try:
        weights = tuple(float(part.strip()) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Weights must be four comma-separated numbers for lags 1,2,3,12."
        ) from exc
    if len(weights) != 4 or any(not math.isfinite(x) or x < 0 for x in weights):
        raise argparse.ArgumentTypeError(
            "Provide four finite, non-negative weights for lags 1,2,3,12."
        )
    total = sum(weights)
    if total <= 0:
        raise argparse.ArgumentTypeError("At least one lag weight must be positive.")
    return tuple(x / total for x in weights)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run one 100M loan and verify cash-flow reconciliation."
    )
    parser.add_argument("--data", default=None, help="Path to df_cleaned.csv.")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--loan-term-months", type=int, choices=[12, 18, 24, 36], default=18)
    parser.add_argument("--annual-interest-rate", type=float, default=0.04)
    parser.add_argument(
        "--payment-delay-probability",
        type=float,
        default=0.10,
        help="Maximum expected late-payment share, scaled by post-loan risk / 100.",
    )
    parser.add_argument(
        "--delay-lag-weights",
        type=parse_weights,
        default=(0.25, 0.25, 0.25, 0.25),
        help="Normalized weights for delay lags 1,2,3,12 months.",
    )
    parser.add_argument(
        "--stress-max-delay",
        action="store_true",
        help="Use 100% maximum late share and set the test applicant risk to 100.",
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
    if not math.isfinite(args.payment_delay_probability) or not 0 <= args.payment_delay_probability <= 1:
        raise ValueError("--payment-delay-probability must be between 0 and 1.")

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

    # Use the held-out test rows for the applicant pool. The scaler was fitted
    # on training rows only by the shared data-preparation utility.
    _, _, test_raw, scaler, split_info = load_split_and_scaler(
        data_path=args.data,
        split_seed=42,
    )

    delay_share = 1.0 if args.stress_max_delay else args.payment_delay_probability
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
        payment_delay_probability=delay_share,
        delay_lag_weights=args.delay_lag_weights,
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
            # Test-only stress scenario: ensure the configured late share is reached.
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
        effective_late_share = (
            actual_contract.installment_delay_probability
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
        print(f"Stress max-delay mode   : {args.stress_max_delay}")
        print(f"Effective late share    : {effective_late_share:.2%}")
        print(f"Delay lag weights       : {args.delay_lag_weights}")
        print(f"Episode ended           : {terminated}")
        print(f"Final month             : {portfolio.month}")
        print(f"Loan principal          : {portfolio.total_disbursed:,.2f}")
        print(f"Principal collected     : {portfolio.cumulative_principal_collected:,.2f}")
        print(f"Interest collected      : {portfolio.cumulative_interest_collected:,.2f}")
        print(f"Late amount scheduled   : {portfolio.cumulative_delayed_payment_amount_scheduled:,.2f}")
        print(f"Late amount collected   : {portfolio.cumulative_delayed_payment_amount_collected:,.2f}")
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
