
from dataclasses import dataclass
import math
from typing import Optional, Union

from customer_state import CustomerState
from customer_simulator import CustomerSimulator


CustomerId = Optional[Union[str, int]]


@dataclass
class LoanContract:
    """One loan, tracked using expected-value accounting."""

    loan_id: int
    customer_id: CustomerId
    origin_month: int

    principal: float
    outstanding_principal: float
    monthly_payment: float
    monthly_interest_rate: float
    monthly_hazard: float

    term_months: int
    remaining_months: int

    # Probability that the contract has not defaulted yet.
    survival_probability: float = 1.0

    status: str = "active"


@dataclass
class MonthlyPortfolioResult:
    """Expected portfolio cash flows and credit losses for one month."""

    month: int
    cash_before: float
    capital_injection: float

    principal_collected: float
    interest_collected: float
    recovery_collected: float

    expected_default_count: float
    gross_defaulted_principal: float
    net_default_loss: float

    financial_pnl: float
    cash_after: float

    active_contracts: int
    expected_active_contracts: float
    expected_outstanding_principal: float


class PortfolioSimulator:
    """
    Portfolio-level simulator.

    Time convention:
    - Call advance_month() to enter a new calendar month.
    - Grant loans during that month.
    - A loan originated in month t first becomes eligible
      for repayment processing when entering month t + 1.

    Defaults and collections are calculated in expectation.
    This class does not generate random default events.
    """

    EPSILON = 1e-6

    def __init__(
        self,
        initial_budget: float,
        monthly_capital_injection: float = 0.0,
        loan_term_months: int = 18,
        monthly_interest_rate: float = 0.0,
        recovery_rate: float = 0.0,
        customer_simulator: Optional[CustomerSimulator] = None,
    ):
        if not math.isfinite(initial_budget) or initial_budget < 0:
            raise ValueError("initial_budget must be finite and non-negative.")

        if (
            not math.isfinite(monthly_capital_injection)
            or monthly_capital_injection < 0
        ):
            raise ValueError(
                "monthly_capital_injection must be finite and non-negative."
            )

        if loan_term_months <= 0:
            raise ValueError("loan_term_months must be positive.")

        if (
            not math.isfinite(monthly_interest_rate)
            or monthly_interest_rate < 0
        ):
            raise ValueError(
                "monthly_interest_rate must be finite and non-negative."
            )

        if not 0.0 <= recovery_rate <= 1.0:
            raise ValueError("recovery_rate must be between 0 and 1.")

        self.initial_budget = float(initial_budget)
        self.cash_available = float(initial_budget)

        self.monthly_capital_injection = float(
            monthly_capital_injection
        )

        self.loan_term_months = int(loan_term_months)
        self.monthly_interest_rate = float(monthly_interest_rate)
        self.recovery_rate = float(recovery_rate)

        self.customer_simulator = (
            customer_simulator
            if customer_simulator is not None
            else CustomerSimulator()
        )

        self.month = 0
        self.next_loan_id = 1

        self.contracts: list[LoanContract] = []

        self.total_disbursed = 0.0
        self.cumulative_expected_defaults = 0.0
        self.cumulative_net_default_loss = 0.0
        self.cumulative_interest_collected = 0.0

    @property
    def available_budget(self) -> float:
        """Cash currently available for new loan originations."""
        return self.cash_available

    @property
    def active_contract_count(self) -> int:
        """Number of contracts with non-negligible survival probability."""
        return sum(
            contract.status == "active"
            and contract.remaining_months > 0
            and contract.survival_probability > 1e-12
            for contract in self.contracts
        )

    @property
    def expected_active_contracts(self) -> float:
        """Expected number of contracts that have not defaulted or matured."""
        return sum(
            contract.survival_probability
            for contract in self.contracts
            if contract.status == "active"
            and contract.remaining_months > 0
        )

    @property
    def expected_outstanding_principal(self) -> float:
        """Expected outstanding principal across the active portfolio."""
        return sum(
            contract.survival_probability
            * contract.outstanding_principal
            for contract in self.contracts
            if contract.status == "active"
            and contract.remaining_months > 0
        )

    def can_fund(self, amount: float) -> bool:
        """Check whether the current portfolio can fund an action."""
        amount = float(amount)

        if not math.isfinite(amount) or amount < 0:
            return False

        return amount <= self.cash_available + self.EPSILON

    def _calculate_monthly_payment(
        self,
        principal: float,
    ) -> float:
        """Fixed monthly instalment for a fully amortizing loan."""
        rate = self.monthly_interest_rate
        term = self.loan_term_months

        if rate == 0:
            return principal / term

        denominator = 1.0 - (1.0 + rate) ** (-term)

        return principal * rate / denominator

    def grant_loan(
        self,
        customer: CustomerState,
        approved_amount: float,
        customer_id: CustomerId = None,
    ) -> Optional[LoanContract]:
        """
        Originate a loan, deduct its principal from available cash,
        and register its repayment schedule.

        A zero-amount action does not create a contract.
        """
        amount = float(approved_amount)

        if not math.isfinite(amount) or amount < 0:
            raise ValueError(
                "approved_amount must be finite and non-negative."
            )

        if amount == 0:
            return None

        if self.month <= 0:
            raise RuntimeError(
                "Call advance_month() before granting loans."
            )

        if not self.can_fund(amount):
            raise ValueError(
                f"Insufficient budget: available={self.cash_available:,.2f}, "
                f"requested={amount:,.2f}."
            )

        previous_customer = customer.copy()

        # Reuse the existing project's state-transition and risk models.
        customer_after_loan = (
            self.customer_simulator._update_customer_state(
                customer=previous_customer,
                approved_unsecured_amount=amount,
            )
        )

        customer_after_loan.risk = (
            self.customer_simulator._update_risk(
                customer=customer_after_loan,
                previous_customer=previous_customer,
                approved_unsecured_amount=amount,
            )
        )

        # Convert the post-origination risk into a monthly hazard.
        # The hazard remains fixed for this first simulator version.
        monthly_hazard = max(
            0.0,
            float(
                self.customer_simulator._estimate_hazard(
                    customer_after_loan
                )
            ),
        )

        monthly_payment = self._calculate_monthly_payment(amount)

        contract = LoanContract(
            loan_id=self.next_loan_id,
            customer_id=customer_id,
            origin_month=self.month,
            principal=amount,
            outstanding_principal=amount,
            monthly_payment=monthly_payment,
            monthly_interest_rate=self.monthly_interest_rate,
            monthly_hazard=monthly_hazard,
            term_months=self.loan_term_months,
            remaining_months=self.loan_term_months,
        )

        self.next_loan_id += 1
        self.contracts.append(contract)

        self.cash_available = max(
            0.0,
            self.cash_available - amount,
        )

        self.total_disbursed += amount

        return contract

    def advance_month(
        self,
        capital_injection: Optional[float] = None,
    ) -> MonthlyPortfolioResult:
        """
        Move the calendar forward by one month.

        Existing loans are serviced in expectation:
        1. Calculate conditional default probability.
        2. Calculate expected defaults before that month's payment.
        3. Collect expected instalments from surviving contracts.
        4. Account for recoveries and remaining credit losses.
        """
        injection = (
            self.monthly_capital_injection
            if capital_injection is None
            else float(capital_injection)
        )

        if not math.isfinite(injection) or injection < 0:
            raise ValueError(
                "capital_injection must be finite and non-negative."
            )

        cash_before = self.cash_available

        self.month += 1
        self.cash_available += injection

        principal_collected = 0.0
        interest_collected = 0.0
        recovery_collected = 0.0

        expected_default_count = 0.0
        gross_defaulted_principal = 0.0

        for contract in self.contracts:
            if contract.status != "active":
                continue

            if contract.remaining_months <= 0:
                contract.status = "matured"
                continue

            # Do not collect a payment in the month of origination.
            if contract.origin_month >= self.month:
                continue

            balance = contract.outstanding_principal
            survival_before = contract.survival_probability

            # Conditional probability of default during this month.
            p_default_month = 1.0 - math.exp(
                -contract.monthly_hazard
            )
            p_default_month = min(
                1.0,
                max(0.0, p_default_month),
            )

            default_mass = survival_before * p_default_month
            survival_after = survival_before * (
                1.0 - p_default_month
            )

            default_balance = default_mass * balance

            recovery = default_balance * self.recovery_rate

            # Instalment breakdown conditional on being alive.
            interest_due = (
                balance * contract.monthly_interest_rate
            )

            if contract.remaining_months == 1:
                # Avoid a tiny residual balance caused by rounding.
                principal_due = balance
            else:
                principal_due = min(
                    balance,
                    max(
                        0.0,
                        contract.monthly_payment - interest_due,
                    ),
                )

            # Expected collections come only from contracts surviving
            # the current month's default risk.
            expected_principal = survival_after * principal_due
            expected_interest = survival_after * interest_due

            principal_collected += expected_principal
            interest_collected += expected_interest
            recovery_collected += recovery

            expected_default_count += default_mass
            gross_defaulted_principal += default_balance

            # Update available cash. Principal repayments replenish
            # liquidity but are not counted as accounting profit.
            self.cash_available += (
                expected_principal
                + expected_interest
                + recovery
            )

            contract.survival_probability = survival_after
            contract.outstanding_principal = max(
                0.0,
                balance - principal_due,
            )
            contract.remaining_months -= 1

            if contract.remaining_months == 0:
                contract.outstanding_principal = 0.0
                contract.survival_probability = 0.0
                contract.status = "matured"

            elif survival_after <= 1e-12:
                contract.outstanding_principal = 0.0
                contract.survival_probability = 0.0
                contract.remaining_months = 0
                contract.status = "depleted"

        net_default_loss = max(
            0.0,
            gross_defaulted_principal - recovery_collected,
        )

        # Accounting profit: interest earned minus net credit losses.
        # Principal repayments are capital recovery, not profit.
        financial_pnl = interest_collected - net_default_loss

        self.cumulative_expected_defaults += expected_default_count
        self.cumulative_net_default_loss += net_default_loss
        self.cumulative_interest_collected += interest_collected

        return MonthlyPortfolioResult(
            month=self.month,
            cash_before=cash_before,
            capital_injection=injection,
            principal_collected=principal_collected,
            interest_collected=interest_collected,
            recovery_collected=recovery_collected,
            expected_default_count=expected_default_count,
            gross_defaulted_principal=gross_defaulted_principal,
            net_default_loss=net_default_loss,
            financial_pnl=financial_pnl,
            cash_after=self.cash_available,
            active_contracts=self.active_contract_count,
            expected_active_contracts=self.expected_active_contracts,
            expected_outstanding_principal=self.expected_outstanding_principal,
        )
