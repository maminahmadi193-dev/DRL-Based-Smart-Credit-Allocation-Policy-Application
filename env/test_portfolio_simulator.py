
import math
import unittest

from customer_state import CustomerState
from customer_simulator import CustomerSimulator
from portfolio_simulator import PortfolioSimulator


class FixedHazardSimulator(CustomerSimulator):
    """Deterministic hazard model for repeatable unit tests."""

    def __init__(self, hazard):
        super().__init__()
        self.fixed_hazard = hazard

    def _estimate_hazard(self, customer):
        return self.fixed_hazard


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
    hazard=0.0,
    recovery=0.0,
):
    return PortfolioSimulator(
        initial_budget=budget,
        monthly_capital_injection=0,
        loan_term_months=term,
        monthly_interest_rate=rate,
        recovery_rate=recovery,
        customer_simulator=FixedHazardSimulator(hazard),
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
            hazard=0,
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

    def test_expected_default_and_recovery(self):
        hazard = 0.1
        recovery_rate = 0.5

        portfolio = make_portfolio(
            budget=1_000_000,
            term=3,
            rate=0,
            hazard=hazard,
            recovery=recovery_rate,
        )
        portfolio.advance_month()

        portfolio.grant_loan(
            make_customer(),
            300_000,
            customer_id=1,
        )

        result = portfolio.advance_month()

        expected_default_probability = 1 - math.exp(-hazard)
        expected_defaulted_principal = (
            300_000 * expected_default_probability
        )
        expected_recovery = (
            expected_defaulted_principal * recovery_rate
        )
        expected_net_loss = (
            expected_defaulted_principal - expected_recovery
        )

        self.assertAlmostEqual(
            result.expected_default_count,
            expected_default_probability,
            places=7,
        )

        self.assertAlmostEqual(
            result.gross_defaulted_principal,
            expected_defaulted_principal,
            places=5,
        )

        self.assertAlmostEqual(
            result.recovery_collected,
            expected_recovery,
            places=5,
        )

        self.assertAlmostEqual(
            result.net_default_loss,
            expected_net_loss,
            places=5,
        )

        self.assertAlmostEqual(
            result.financial_pnl,
            -expected_net_loss,
            places=5,
        )

    
    def test_positive_interest_loan_amortizes_correctly(self):
        principal = 300_000
        term = 3
        initial_budget = 1_000_000

        portfolio = make_portfolio(
            budget=initial_budget,
            term=term,
            rate=0.01,
            hazard=0,
            recovery=0,
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
