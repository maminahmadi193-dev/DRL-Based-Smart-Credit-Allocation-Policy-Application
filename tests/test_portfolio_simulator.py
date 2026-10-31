
import unittest

import numpy as np

from customer_state import CustomerState
from customer_simulator import CustomerSimulator
from portfolio_simulator import PortfolioSimulator


def make_customer():
    return CustomerState(
        monthly_income=100_000,
        income_score=50,
        job_score=50,
        asset_score=50,
        total_customer_score=50,
        total_facilities=0,
        debt=0,
        origin_amount=0,
        loan_count=0,
        debt_ratio=0,
        debt_per_loan=0,
        has_no_history=True,
        employed=True,
        score_min=500,
        score_max=600,
        risk_min=0,
        risk_max=10,
        risk=5,
    )


def make_portfolio(
    budget=1_000_000,
    term=3,
    annual_rate=0.0,
    collection_lag=4,
    max_hazard=0.0,
    seed=42,
):
    return PortfolioSimulator(
        initial_budget=budget,
        monthly_capital_injection=0,
        loan_term_months=term,
        annual_interest_rate=annual_rate,
        default_collection_lag_months=collection_lag,
        customer_simulator=CustomerSimulator(max_hazard=max_hazard),
        rng=np.random.default_rng(seed),
    )


class TestPortfolioSimulator(unittest.TestCase):

    def test_loan_grant_deducts_budget(self):
        portfolio = make_portfolio()
        portfolio.advance_month()

        contract = portfolio.grant_loan(
            make_customer(),
            300_000,
            customer_id=1,
        )

        self.assertIsNotNone(contract)
        self.assertEqual(portfolio.available_budget, 700_000)
        self.assertEqual(contract.origin_month, 1)
        self.assertEqual(len(portfolio.contracts), 1)

    def test_zero_amount_does_not_create_contract(self):
        portfolio = make_portfolio()
        portfolio.advance_month()

        contract = portfolio.grant_loan(
            make_customer(),
            0,
            customer_id=1,
        )

        self.assertIsNone(contract)
        self.assertEqual(portfolio.available_budget, 1_000_000)
        self.assertEqual(len(portfolio.contracts), 0)

    def test_insufficient_budget_rejects_loan(self):
        portfolio = make_portfolio(budget=200_000)
        portfolio.advance_month()

        with self.assertRaises(ValueError):
            portfolio.grant_loan(
                make_customer(),
                300_000,
                customer_id=1,
            )

        self.assertEqual(portfolio.available_budget, 200_000)
        self.assertEqual(len(portfolio.contracts), 0)
        self.assertEqual(portfolio.total_disbursed, 0)

    def test_principal_is_repaid_over_full_term(self):
        portfolio = make_portfolio(
            budget=1_000_000,
            term=3,
            annual_rate=0.0,
            max_hazard=0.0,
        )
        portfolio.advance_month()

        contract = portfolio.grant_loan(
            make_customer(),
            300_000,
            customer_id=1,
        )

        # A three-month contract has one interest-only month and two
        # equal principal installments.
        self.assertAlmostEqual(contract.principal_payment_amount, 150_000)
        self.assertEqual(contract.principal_payment_months, (2, 3))

        expected = [
            (0.0, 0.0, 300_000.0),
            (150_000.0, 0.0, 150_000.0),
            (150_000.0, 0.0, 0.0),
        ]
        for expected_principal, expected_interest, expected_balance in expected:
            result = portfolio.advance_month()
            self.assertAlmostEqual(result.principal_collected, expected_principal)
            self.assertAlmostEqual(result.interest_collected, expected_interest)
            self.assertAlmostEqual(
                contract.outstanding_principal,
                expected_balance,
                places=5,
            )

        self.assertEqual(contract.status, "matured")
        self.assertEqual(portfolio.active_contract_count, 0)
        self.assertAlmostEqual(portfolio.available_budget, 1_000_000, places=5)

    def test_missed_installment_is_full_or_zero_and_eventually_collected(self):
        portfolio = make_portfolio(
            budget=1_000_000,
            term=3,
            annual_rate=0.0,
            collection_lag=4,
            max_hazard=0.0,
        )
        portfolio.advance_month()
        contract = portfolio.grant_loan(
            make_customer(),
            300_000,
            customer_id=1,
        )

        # Payment 1 is an interest-only month. With a zero rate, no amount
        # is due and there is no monetary installment to miss.
        result = portfolio.advance_month()
        self.assertAlmostEqual(result.principal_collected, 0.0)
        self.assertAlmostEqual(result.interest_collected, 0.0)
        self.assertEqual(result.missed_installment_count, 0.0)

        # Force the next (principal) installment to be missed.
        contract.monthly_miss_probability = 1.0
        result = portfolio.advance_month()
        self.assertAlmostEqual(result.principal_collected, 0.0)
        self.assertAlmostEqual(result.delayed_payment_amount_scheduled, 150_000)
        self.assertEqual(result.missed_installment_count, 1.0)
        self.assertEqual(result.recognized_default_count, 0.0)
        self.assertAlmostEqual(result.pending_delayed_receivables, 150_000)
        self.assertAlmostEqual(portfolio.available_budget, 700_000)
        self.assertTrue(contract.missed_installment_occurred)
        self.assertFalse(contract.default_recognized)

        # The final scheduled principal installment is paid on time.
        result = portfolio.advance_month()
        self.assertAlmostEqual(result.principal_collected, 150_000)
        self.assertAlmostEqual(result.delayed_payment_amount_scheduled, 0.0)
        self.assertEqual(contract.status, "matured")
        self.assertFalse(portfolio.is_fully_settled)

        # The missed March installment is over 90 days past due in June.
        portfolio.advance_month()  # month 5
        self.assertFalse(contract.default_recognized)
        result = portfolio.advance_month()  # month 6
        self.assertTrue(contract.default_recognized)
        self.assertEqual(result.expected_default_count, 1.0)
        self.assertEqual(result.recognized_default_count, 1.0)
        self.assertAlmostEqual(result.principal_collected, 0.0)
        self.assertFalse(contract.late_payment_collected)
        self.assertAlmostEqual(portfolio.pending_delayed_receivables, 150_000)

        result = portfolio.advance_month()  # month 7: full delayed collection
        self.assertTrue(contract.late_payment_collected)
        self.assertTrue(portfolio.is_fully_settled)
        self.assertAlmostEqual(result.principal_collected, 150_000)
        self.assertAlmostEqual(portfolio.cumulative_principal_collected, 300_000)
        self.assertAlmostEqual(
            portfolio.cumulative_delayed_payment_amount_scheduled,
            portfolio.cumulative_delayed_payment_amount_collected,
        )
        self.assertAlmostEqual(portfolio.cumulative_net_default_loss, 0.0)
        self.assertAlmostEqual(portfolio.available_budget, 1_000_000)

    def test_debt_per_loan_never_becomes_negative_and_new_small_loan_does_not_lower_risk(self):
        simulator = CustomerSimulator()
        customer = make_customer()
        customer.debt = 200_000
        customer.origin_amount = 400_000
        customer.loan_count = 2
        customer.debt_ratio = 0.5
        customer.debt_per_loan = 100_000

        updated = simulator._update_customer_state(
            customer,
            approved_unsecured_amount=50_000,
        )
        updated_risk = simulator._update_risk(
            customer=updated,
            previous_customer=customer,
            approved_unsecured_amount=50_000,
        )

        self.assertGreaterEqual(updated.debt_per_loan, 0.0)
        self.assertGreaterEqual(updated.debt_ratio, 0.0)
        self.assertGreaterEqual(updated_risk, customer.risk)

    def test_customer_state_persists_across_repeat_loans_and_risk_falls_with_repayment(self):
        portfolio = make_portfolio(
            budget=1_000_000,
            term=3,
            annual_rate=0.0,
            max_hazard=0.005,
        )
        customer_id = 7
        initial = make_customer()
        portfolio.advance_month()
        first = portfolio.grant_loan(initial, 300_000, customer_id=customer_id)

        initial_risk = portfolio.get_customer_state(customer_id).risk
        self.assertGreater(initial_risk, initial.risk)
        first_hazard = first.monthly_hazard

        # The first payment month is interest-only; principal and risk remain unchanged.
        portfolio.advance_month()
        before_principal = portfolio.register_customer(customer_id, make_customer())
        self.assertAlmostEqual(before_principal.debt, 300_000)
        self.assertAlmostEqual(before_principal.risk, initial_risk)

        # The next payment reduces outstanding principal and customer risk.
        portfolio.advance_month()
        after_payment = portfolio.register_customer(customer_id, make_customer())
        self.assertAlmostEqual(after_payment.debt, 150_000)
        self.assertEqual(after_payment.loan_count, 1)
        self.assertEqual(after_payment.origin_amount, 300_000)
        self.assertGreaterEqual(after_payment.debt_per_loan, 0.0)
        self.assertLess(after_payment.risk, initial_risk)
        self.assertLess(first.monthly_hazard, first_hazard)

        # Reappearing source rows return the persistent state, not a fresh baseline.
        second = portfolio.grant_loan(after_payment, 100_000, customer_id=customer_id)
        after_second_loan = portfolio.get_customer_state(customer_id)
        self.assertEqual(len([
            loan for loan in portfolio.contracts
            if loan.customer_id == customer_id
        ]), 2)
        self.assertEqual(after_second_loan.loan_count, 2)
        self.assertEqual(after_second_loan.origin_amount, 400_000)
        self.assertAlmostEqual(after_second_loan.debt, 250_000)
        self.assertGreater(after_second_loan.risk, after_payment.risk)
        self.assertGreaterEqual(second.risk_loading, 0.0)

        # The first contract completes its final principal installment.
        risk_after_second_origination = after_second_loan.risk
        portfolio.advance_month()
        self.assertLess(
            portfolio.get_customer_state(customer_id).risk,
            risk_after_second_origination,
        )

    def test_missed_principal_remains_customer_debt_until_late_collection(self):
        portfolio = make_portfolio(
            budget=1_000_000,
            term=1,
            annual_rate=0.0,
            collection_lag=4,
            max_hazard=1_000.0,
        )
        customer_id = 9
        portfolio.advance_month()
        portfolio.grant_loan(make_customer(), 100_000, customer_id=customer_id)

        # Month 2 misses the only installment. The scheduled contract matures,
        # but the unpaid principal remains in the customer ledger as a receivable.
        portfolio.advance_month()
        customer = portfolio.get_customer_state(customer_id)
        self.assertAlmostEqual(customer.debt, 100_000)
        self.assertGreaterEqual(customer.debt_per_loan, 0.0)

        # By month 6 the delayed amount is collected and no simulated debt remains.
        for _ in range(4):
            portfolio.advance_month()
        customer = portfolio.get_customer_state(customer_id)
        self.assertAlmostEqual(customer.debt, 0.0)
        self.assertAlmostEqual(customer.debt_per_loan, 0.0)
        self.assertTrue(portfolio.contracts[0].default_recognized)
        self.assertTrue(portfolio.contracts[0].late_payment_collected)

    def test_twelve_month_collection_lag_extends_settlement_after_maturity(self):
        portfolio = make_portfolio(
            budget=1_000_000,
            term=1,
            annual_rate=0.0,
            collection_lag=12,
            max_hazard=1_000.0,
        )
        portfolio.advance_month()
        portfolio.grant_loan(make_customer(), 300_000, customer_id=1)

        result = portfolio.advance_month()
        self.assertEqual(portfolio.month, 2)
        self.assertEqual(portfolio.contracts[0].status, "matured")
        self.assertEqual(portfolio.max_delay_months, 12)
        self.assertFalse(portfolio.is_fully_settled)
        self.assertAlmostEqual(portfolio.pending_delayed_receivables, 300_000)

        for _ in range(12):
            result = portfolio.advance_month()

        self.assertEqual(portfolio.month, 14)
        self.assertTrue(portfolio.contracts[0].default_recognized)
        self.assertTrue(portfolio.is_fully_settled)
        self.assertAlmostEqual(result.principal_collected, 300_000)
        self.assertAlmostEqual(portfolio.available_budget, 1_000_000)

    def test_monthly_miss_probability_increases_with_risk(self):
        portfolio = make_portfolio(
            budget=1_000_000,
            term=3,
            max_hazard=0.005,
        )
        portfolio.advance_month()

        low_risk_customer = make_customer()
        low_risk_customer.risk = 20.0
        high_risk_customer = make_customer()
        high_risk_customer.risk = 80.0

        low = portfolio.grant_loan(low_risk_customer, 100_000, customer_id="low")
        high = portfolio.grant_loan(high_risk_customer, 100_000, customer_id="high")

        self.assertGreater(high.monthly_hazard, low.monthly_hazard)
        self.assertGreater(high.monthly_miss_probability, low.monthly_miss_probability)
        self.assertAlmostEqual(
            high.monthly_miss_probability,
            1.0 - np.exp(-high.monthly_hazard),
        )
        self.assertAlmostEqual(
            high.default_probability_over_term,
            1.0 - np.exp(-high.monthly_hazard * high.term_months),
        )

    def test_term_default_probability_matches_monthly_probability(self):
        simulator = CustomerSimulator(max_hazard=0.005)
        hazard = 0.002
        monthly_probability = simulator._calculate_monthly_event_probability(hazard)

        self.assertAlmostEqual(monthly_probability, 1.0 - np.exp(-hazard))
        previous_pd = 0.0
        for term in (12, 18, 24, 36):
            pd_over_term = simulator._calculate_default_probability(
                hazard,
                horizon_months=term,
            )
            self.assertAlmostEqual(
                pd_over_term,
                1.0 - (1.0 - monthly_probability) ** term,
                places=12,
            )
            self.assertGreater(pd_over_term, previous_pd)
            previous_pd = pd_over_term

        self.assertEqual(simulator._calculate_monthly_event_probability(0.0), 0.0)
        self.assertEqual(
            simulator._calculate_default_probability(0.0, horizon_months=36),
            0.0,
        )

    def test_hazard_intercept_calibrates_mean_term_pd_to_target(self):
        risks = np.linspace(0.0, 100.0, 101)
        target_rate = 0.013
        maximum_hazard = 0.005
        term_months = 18
        slope = 0.08
        center = 60.0

        intercept = CustomerSimulator.calibrate_hazard_intercept(
            reference_risks=risks,
            target_term_default_rate=target_rate,
            max_hazard=maximum_hazard,
            hazard_slope=slope,
            hazard_center=center,
            horizon_months=term_months,
        )
        z = slope * (risks - center) + intercept
        hazards = maximum_hazard / (1.0 + np.exp(-z))
        term_pds = -np.expm1(-hazards * term_months)

        self.assertAlmostEqual(float(np.mean(term_pds)), target_rate, places=12)
        self.assertTrue(np.all(np.diff(term_pds) >= 0.0))
        self.assertGreater(intercept, -60.0)
        self.assertLess(intercept, 60.0)

    def test_hazard_calibration_rejects_unattainable_target(self):
        with self.assertRaises(ValueError):
            CustomerSimulator.calibrate_hazard_intercept(
                reference_risks=np.array([20.0, 80.0]),
                target_term_default_rate=0.20,
                max_hazard=0.005,
                hazard_slope=0.08,
                hazard_center=60.0,
                horizon_months=18,
            )

    def test_payment_rng_makes_missed_installment_path_reproducible(self):
        outcomes = []
        for seed in (987, 987):
            portfolio = make_portfolio(
                budget=1_000_000,
                term=2,
                annual_rate=0.04,
                max_hazard=1000.0,
                seed=seed,
            )
            portfolio.advance_month()
            contract = portfolio.grant_loan(make_customer(), 200_000, customer_id=1)
            result = portfolio.advance_month()
            outcomes.append((
                contract.missed_installment_occurred,
                result.principal_collected,
                result.interest_collected,
                result.delayed_payment_amount_scheduled,
            ))

        self.assertEqual(outcomes[0], outcomes[1])
        self.assertTrue(outcomes[0][0])
        self.assertEqual(outcomes[0][1], 0.0)
        self.assertAlmostEqual(outcomes[0][2], 0.0)
        self.assertAlmostEqual(outcomes[0][3], 8_000.0)

    def test_recovery_rate_is_rejected_when_full_collection_is_assumed(self):
        with self.assertRaises(ValueError):
            PortfolioSimulator(
                initial_budget=1_000_000,
                recovery_rate=0.5,
            )

    def test_positive_interest_loan_amortizes_correctly(self):
        principal = 300_000
        term = 3
        initial_budget = 1_000_000

        portfolio = make_portfolio(
            budget=initial_budget,
            term=term,
            annual_rate=0.04,
            max_hazard=0.0,
        )
        portfolio.advance_month()
        contract = portfolio.grant_loan(
            make_customer(),
            principal,
            customer_id=1,
        )

        self.assertAlmostEqual(contract.annual_interest_rate, 0.04)
        self.assertAlmostEqual(contract.principal_payment_amount, 150_000)

        total_principal_collected = 0.0
        total_interest_collected = 0.0
        for _ in range(term):
            result = portfolio.advance_month()
            total_principal_collected += result.principal_collected
            total_interest_collected += result.interest_collected

        self.assertAlmostEqual(total_principal_collected, principal, places=5)
        self.assertAlmostEqual(total_interest_collected, 12_000, places=5)
        self.assertAlmostEqual(
            portfolio.available_budget,
            initial_budget + total_interest_collected,
            places=5,
        )
        self.assertEqual(contract.status, "matured")
        self.assertAlmostEqual(contract.outstanding_principal, 0, places=5)

    def test_twelve_eighteen_and_thirty_six_month_annual_interest_schedule(self):
        principal = 100_000_000
        annual_rate = 0.04
        schedules = {
            12: {1: 4_000_000.0},
            18: {1: 4_000_000.0, 13: 1_250_000.0},
            36: {
                1: 4_000_000.0,
                13: 2_666_666.6666666665,
                25: 1_333_333.3333333333,
            },
        }

        for term, expected_interest_by_payment in schedules.items():
            with self.subTest(term=term):
                portfolio = make_portfolio(
                    budget=principal,
                    term=term,
                    annual_rate=annual_rate,
                    max_hazard=0.0,
                )
                portfolio.advance_month()
                contract = portfolio.grant_loan(
                    make_customer(),
                    principal,
                    customer_id=term,
                )

                interest_months = set(expected_interest_by_payment)
                principal_months = [
                    month for month in range(1, term + 1)
                    if month not in interest_months
                ]
                expected_principal_installment = principal / len(principal_months)
                self.assertAlmostEqual(
                    contract.principal_payment_amount,
                    expected_principal_installment,
                    places=5,
                )

                total_principal = 0.0
                total_interest = 0.0
                for payment_number in range(1, term + 1):
                    expected_principal = (
                        0.0
                        if payment_number in interest_months
                        else expected_principal_installment
                    )
                    expected_interest = expected_interest_by_payment.get(
                        payment_number,
                        0.0,
                    )
                    expected_collection = expected_principal + expected_interest

                    # The observation forecast must match the next contractual
                    # collection when every installment is paid on time.
                    self.assertAlmostEqual(
                        portfolio.expected_cash_collections_next_month,
                        expected_collection,
                        delta=0.02,
                    )
                    cash_before = portfolio.available_budget
                    result = portfolio.advance_month()

                    self.assertAlmostEqual(
                        result.principal_collected,
                        expected_principal,
                        delta=0.02,
                    )
                    self.assertAlmostEqual(
                        result.interest_collected,
                        expected_interest,
                        delta=0.02,
                    )
                    # Principal affects liquidity; only realized interest is P&L.
                    self.assertAlmostEqual(
                        result.financial_pnl,
                        expected_interest,
                        delta=0.02,
                    )
                    self.assertAlmostEqual(
                        result.cash_after,
                        cash_before + expected_collection,
                        delta=0.02,
                    )
                    total_principal += result.principal_collected
                    total_interest += result.interest_collected

                self.assertAlmostEqual(total_principal, principal, delta=0.02)
                self.assertAlmostEqual(
                    total_interest,
                    sum(expected_interest_by_payment.values()),
                    delta=0.02,
                )
                self.assertAlmostEqual(
                    portfolio.available_budget,
                    principal + total_interest,
                    delta=0.02,
                )
                self.assertEqual(contract.status, "matured")


    def test_annual_interest_includes_uncollected_missed_principal(self):
        principal = 100_000_000
        principal_installment = principal / 16.0
        portfolio = make_portfolio(
            budget=principal,
            term=18,
            annual_rate=0.04,
            collection_lag=4,
            max_hazard=0.0,
        )
        portfolio.advance_month()
        contract = portfolio.grant_loan(
            make_customer(),
            principal,
            customer_id=18,
        )

        # Pay payment months 1..11 on time: one annual-interest payment and
        # ten principal installments have been collected.
        for _ in range(11):
            portfolio.advance_month()
        self.assertAlmostEqual(
            contract.outstanding_principal,
            principal - 10 * principal_installment,
            delta=0.02,
        )

        # Miss principal installment 12. It remains part of the customer's
        # outstanding exposure until the delayed collection is received.
        contract.monthly_miss_probability = 1.0
        missed = portfolio.advance_month()
        self.assertAlmostEqual(missed.principal_collected, 0.0, delta=0.02)
        self.assertAlmostEqual(
            missed.delayed_payment_amount_scheduled,
            principal_installment,
            delta=0.02,
        )
        self.assertAlmostEqual(
            portfolio.pending_delayed_principal,
            principal_installment,
            delta=0.02,
        )

        # Payment month 13 occurs before the four-month-late principal is
        # collected, so annual interest is based on both scheduled balance
        # and the still-unpaid principal installment.
        expected_annual_interest = (
            contract.outstanding_principal + principal_installment
        ) * 0.04
        self.assertAlmostEqual(
            portfolio.expected_cash_collections_next_month,
            expected_annual_interest,
            delta=0.02,
        )
        annual_interest = portfolio.advance_month()
        self.assertAlmostEqual(annual_interest.principal_collected, 0.0, delta=0.02)
        self.assertAlmostEqual(
            annual_interest.interest_collected,
            expected_annual_interest,
            delta=0.02,
        )
        self.assertAlmostEqual(
            annual_interest.financial_pnl,
            expected_annual_interest,
            delta=0.02,
        )

        # The late principal is due in the same month as scheduled installment
        # 16. The next-month forecast must include both cash receipts.
        portfolio.advance_month()  # Payment 14
        portfolio.advance_month()  # Payment 15
        expected_combined_receipt = 2.0 * principal_installment
        self.assertAlmostEqual(
            portfolio.expected_cash_collections_next_month,
            expected_combined_receipt,
            delta=0.02,
        )
        late_collection = portfolio.advance_month()  # Payment 16
        self.assertAlmostEqual(
            late_collection.principal_collected,
            expected_combined_receipt,
            delta=0.02,
        )
        self.assertAlmostEqual(
            late_collection.delayed_payment_amount_collected,
            principal_installment,
            delta=0.02,
        )
        self.assertTrue(contract.late_payment_collected)

        portfolio.advance_month()  # Payment 17
        portfolio.advance_month()  # Payment 18
        self.assertEqual(contract.status, "matured")
        self.assertTrue(portfolio.is_fully_settled)
        self.assertAlmostEqual(
            portfolio.cumulative_principal_collected,
            principal,
            delta=0.02,
        )
        self.assertAlmostEqual(
            portfolio.cumulative_interest_collected,
            4_000_000.0 + expected_annual_interest,
            delta=0.02,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
