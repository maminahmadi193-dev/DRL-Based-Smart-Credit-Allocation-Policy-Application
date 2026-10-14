
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
    rate=0.0,
    collection_lag=4,
    max_hazard=0.0,
    seed=42,
):
    return PortfolioSimulator(
        initial_budget=budget,
        monthly_capital_injection=0,
        loan_term_months=term,
        monthly_interest_rate=rate,
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
            rate=0,
            max_hazard=0.0,
        )
        portfolio.advance_month()

        contract = portfolio.grant_loan(
            make_customer(),
            300_000,
            customer_id=1,
        )

        # No repayment occurs at the moment of origination.
        self.assertEqual(portfolio.available_budget, 700_000)
        self.assertAlmostEqual(
            contract.monthly_payment,
            100_000,
        )

        expected_balances = [200_000, 100_000, 0]

        # First repayment is processed in month 2.
        for expected_balance in expected_balances:
            result = portfolio.advance_month()

            self.assertAlmostEqual(
                result.principal_collected,
                100_000,
                places=5,
            )

            self.assertAlmostEqual(
                contract.outstanding_principal,
                expected_balance,
                places=5,
            )

        self.assertEqual(contract.status, "matured")
        self.assertEqual(portfolio.active_contract_count, 0)
        self.assertAlmostEqual(
            portfolio.available_budget,
            1_000_000,
            places=5,
        )

    def test_missed_installment_is_full_or_zero_and_eventually_collected(self):
        portfolio = make_portfolio(
            budget=1_000_000,
            term=3,
            rate=0.0,
            collection_lag=4,
            max_hazard=1_000.0,  # Force the first eligible installment to be missed.
        )
        portfolio.advance_month()
        portfolio.grant_loan(make_customer(), 300_000, customer_id=1)

        # Month 2: the complete 100k installment is missed; zero is collected.
        result = portfolio.advance_month()
        self.assertAlmostEqual(result.principal_collected, 0.0)
        self.assertAlmostEqual(result.delayed_payment_amount_scheduled, 100_000)
        self.assertEqual(result.missed_installment_count, 1.0)
        self.assertEqual(result.recognized_default_count, 0.0)
        self.assertAlmostEqual(result.pending_delayed_receivables, 100_000)
        self.assertAlmostEqual(portfolio.available_budget, 700_000)
        contract = portfolio.contracts[0]
        self.assertTrue(contract.missed_installment_occurred)
        self.assertFalse(contract.default_recognized)

        # The one missed installment is a single event; the later installments
        # are paid in full on their contractual due dates.
        result = portfolio.advance_month()
        self.assertAlmostEqual(result.principal_collected, 100_000)
        self.assertAlmostEqual(result.delayed_payment_amount_scheduled, 0.0)
        result = portfolio.advance_month()
        self.assertAlmostEqual(result.principal_collected, 100_000)
        self.assertEqual(contract.status, "matured")
        self.assertFalse(portfolio.is_fully_settled)

        # At month 6, the installment is >90 DPD; default is recognized before
        # the same-month full collection of the overdue amount.
        portfolio.advance_month()
        self.assertEqual(portfolio.month, 5)
        self.assertFalse(contract.default_recognized)
        result = portfolio.advance_month()
        self.assertEqual(portfolio.month, 6)
        self.assertTrue(contract.default_recognized)
        self.assertEqual(result.expected_default_count, 1.0)
        self.assertEqual(result.recognized_default_count, 1.0)
        self.assertAlmostEqual(result.principal_collected, 100_000)
        self.assertTrue(contract.late_payment_collected)
        self.assertTrue(portfolio.is_fully_settled)
        self.assertAlmostEqual(portfolio.cumulative_principal_collected, 300_000)
        self.assertAlmostEqual(
            portfolio.cumulative_delayed_payment_amount_scheduled,
            portfolio.cumulative_delayed_payment_amount_collected,
        )
        self.assertAlmostEqual(portfolio.cumulative_net_default_loss, 0.0)
        self.assertAlmostEqual(portfolio.available_budget, 1_000_000)

    def test_twelve_month_collection_lag_extends_settlement_after_maturity(self):
        portfolio = make_portfolio(
            budget=1_000_000,
            term=1,
            rate=0.0,
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
                max_hazard=1000.0,
                seed=seed,
            )
            portfolio.advance_month()
            contract = portfolio.grant_loan(make_customer(), 200_000, customer_id=1)
            result = portfolio.advance_month()
            outcomes.append((
                contract.missed_installment_occurred,
                result.principal_collected,
                result.delayed_payment_amount_scheduled,
            ))

        self.assertEqual(outcomes[0], outcomes[1])
        self.assertTrue(outcomes[0][0])
        self.assertEqual(outcomes[0][1], 0.0)
        self.assertAlmostEqual(outcomes[0][2], 100_000.0)

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
            rate=0.01,
            max_hazard=0.0,
        )

        portfolio.advance_month()

        contract = portfolio.grant_loan(
            make_customer(),
            principal,
            customer_id=1,
        )

        self.assertGreater(
            contract.monthly_payment,
            principal / term,
        )

        total_principal_collected = 0.0
        total_interest_collected = 0.0

        for _ in range(term):
            result = portfolio.advance_month()

            total_principal_collected += (
                result.principal_collected
            )

            total_interest_collected += (
                result.interest_collected
            )

        # The entire principal must be repaid.
        self.assertAlmostEqual(
            total_principal_collected,
            principal,
            places=5,
        )

        # Positive interest must be collected.
        self.assertGreater(
            total_interest_collected,
            0,
        )

        # With zero default risk and no capital injections,
        # principal is fully returned and interest increases cash.
        self.assertAlmostEqual(
            portfolio.available_budget,
            initial_budget + total_interest_collected,
            places=5,
        )

        self.assertEqual(
            contract.status,
            "matured",
        )

        self.assertAlmostEqual(
            contract.outstanding_principal,
            0,
            places=5,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
