
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
    installment_delay_probability: float

    term_months: int
    remaining_months: int

    # Kept for compatibility with the expected-value accounting model.
    # In the full-collection scenario it remains 1 until maturity.
    survival_probability: float = 1.0

    status: str = "active"


@dataclass
class MonthlyPortfolioResult:
    """Expected collections, delayed receivables, and cash flows for one month."""

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
    delayed_payment_amount_scheduled: float
    delayed_payment_amount_collected: float
    pending_delayed_receivables: float
    expected_delayed_installments: float


class PortfolioSimulator:
    """
    Portfolio-level simulator.

    Time convention:
    - Call advance_month() to enter a new calendar month.
    - Grant loans during that month.
    - A loan originated in month t first becomes eligible
      for repayment processing when entering month t + 1.

    All contractual installments are ultimately collected. A configurable
    fraction may be delayed by 1, 2, 3, or 12 months. Since this is an
    expected-value simulator, each installment is split deterministically
    between on-time and future collections. No permanent credit loss is modeled.
    """

    EPSILON = 1e-6

    def __init__(
        self,
        initial_budget: float,
        monthly_capital_injection: float = 0.0,
        loan_term_months: int = 18,
        monthly_interest_rate: float = 0.0,
        recovery_rate: float = 0.0,
        payment_delay_probability: float = 0.0,
        delay_lag_weights=(0.25, 0.25, 0.25, 0.25),
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
        if recovery_rate != 0.0:
            raise ValueError(
                "recovery_rate is not used when all delayed payments are "
                "assumed to be fully collected. Use payment_delay_probability "
                "to model timing delays instead."
            )
        if not math.isfinite(payment_delay_probability) or not 0.0 <= payment_delay_probability <= 1.0:
            raise ValueError("payment_delay_probability must be between 0 and 1.")
        if len(delay_lag_weights) != 4:
            raise ValueError("delay_lag_weights must contain four weights for lags 1, 2, 3, and 12 months.")
        delay_lag_weights = tuple(float(weight) for weight in delay_lag_weights)
        if any(not math.isfinite(weight) or weight < 0.0 for weight in delay_lag_weights):
            raise ValueError("delay_lag_weights must be finite and non-negative.")
        weight_sum = sum(delay_lag_weights)
        if payment_delay_probability > 0.0 and weight_sum <= 0.0:
            raise ValueError("At least one delay_lag_weight must be positive when payment delays are enabled.")
        if weight_sum > 0.0:
            delay_lag_weights = tuple(weight / weight_sum for weight in delay_lag_weights)

        self.initial_budget = float(initial_budget)
        self.cash_available = float(initial_budget)

        self.monthly_capital_injection = float(
            monthly_capital_injection
        )

        self.loan_term_months = int(loan_term_months)
        self.monthly_interest_rate = float(monthly_interest_rate)
        self.recovery_rate = 0.0  # Retained as a deprecated compatibility field.
        self.payment_delay_probability = float(payment_delay_probability)
        self.delay_lags_months = (1, 2, 3, 12)
        self.delay_lag_weights = delay_lag_weights
        self.max_delay_months = (
            max(
                lag
                for lag, weight in zip(self.delay_lags_months, self.delay_lag_weights)
                if weight > 0.0
            )
            if self.payment_delay_probability > 0.0
            else 0
        )
        self._delayed_receivables: dict[int, dict[str, float]] = {}

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
        self.cumulative_principal_collected = 0.0
        self.cumulative_interest_collected = 0.0
        self.cumulative_delayed_payment_amount_scheduled = 0.0
        self.cumulative_delayed_payment_amount_collected = 0.0
        self.cumulative_expected_delayed_installments = 0.0

    @property
    def available_budget(self) -> float:
        """Cash currently available for new loan originations."""
        return self.cash_available

    @property
    def active_contract_count(self) -> int:
        """Number of contracts whose contractual installments remain scheduled."""
        return sum(
            contract.status == "active" and contract.remaining_months > 0
            for contract in self.contracts
        )

    @property
    def pending_delayed_receivables(self) -> float:
        """Total expected late principal and interest not yet collected."""
        return sum(
            amount.get("principal", 0.0) + amount.get("interest", 0.0)
            for amount in self._delayed_receivables.values()
        )

    @property
    def pending_delayed_principal(self) -> float:
        """Principal portion of delayed collections still outstanding."""
        return sum(
            amount.get("principal", 0.0)
            for amount in self._delayed_receivables.values()
        )

    @property
    def is_fully_settled(self) -> bool:
        return (
            self.active_contract_count == 0
            and self.pending_delayed_receivables <= self.EPSILON
        )

    @property
    def delayed_receipts_by_month_ahead(self) -> tuple[float, ...]:
        """Expected late receipts due in each of the next 1..max-delay months."""
        return tuple(
            self._delayed_receivables.get(self.month + offset, {}).get("principal", 0.0)
            + self._delayed_receivables.get(self.month + offset, {}).get("interest", 0.0)
            for offset in range(1, self.max_delay_months + 1)
        )

    @property
    def expected_cash_collections_next_month(self) -> float:
        """Forecast next-month cash receipts from scheduled and delayed payments."""
        scheduled = 0.0
        for contract in self.contracts:
            if (
                contract.status != "active"
                or contract.remaining_months <= 0
                or contract.origin_month > self.month
            ):
                continue
            balance = contract.outstanding_principal
            interest_due = balance * contract.monthly_interest_rate
            if contract.remaining_months == 1:
                principal_due = balance
            else:
                principal_due = min(
                    balance,
                    max(0.0, contract.monthly_payment - interest_due),
                )
            on_time_share = 1.0 - contract.installment_delay_probability
            scheduled += (principal_due + interest_due) * on_time_share
        delayed = self._delayed_receivables.get(self.month + 1, {})
        return scheduled + delayed.get("principal", 0.0) + delayed.get("interest", 0.0)

    @property
    def expected_active_contracts(self) -> float:
        """Number of contracts with scheduled installments remaining."""
        return float(self.active_contract_count)

    @property
    def expected_outstanding_principal(self) -> float:
        """Scheduled outstanding principal plus principal in delayed receivables."""
        scheduled_principal = sum(
            contract.outstanding_principal
            for contract in self.contracts
            if contract.status == "active" and contract.remaining_months > 0
        )
        return scheduled_principal + self.pending_delayed_principal

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

        # Customer risk changes the expected share of the installment paid late.
        # This is a transparent linear proxy, not an empirically calibrated PD model.
        risk_factor = min(
            1.0,
            max(0.0, float(customer_after_loan.risk) / 100.0),
        )
        installment_delay_probability = (
            self.payment_delay_probability * risk_factor
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
            installment_delay_probability=installment_delay_probability,
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
        Advance one calendar month and process all cash expected in that month.

        A share of every contractual installment is collected on time. The
        delayed share is placed into future-month receivables, and every
        delayed amount is ultimately collected. No permanent credit loss is
        modeled in this version.
        """
        injection = (
            self.monthly_capital_injection
            if capital_injection is None
            else float(capital_injection)
        )
        if not math.isfinite(injection) or injection < 0:
            raise ValueError("capital_injection must be finite and non-negative.")

        cash_before = self.cash_available
        self.month += 1
        self.cash_available += injection

        principal_collected = 0.0
        interest_collected = 0.0
        delayed_payment_amount_collected = 0.0
        delayed_payment_amount_scheduled = 0.0
        expected_delayed_installments = 0.0

        # First collect delayed installments scheduled for this calendar month.
        overdue = self._delayed_receivables.pop(
            self.month,
            {"principal": 0.0, "interest": 0.0},
        )
        delayed_principal = overdue.get("principal", 0.0)
        delayed_interest = overdue.get("interest", 0.0)
        principal_collected += delayed_principal
        interest_collected += delayed_interest
        delayed_payment_amount_collected += delayed_principal + delayed_interest
        self.cash_available += delayed_principal + delayed_interest

        for contract in self.contracts:
            if contract.status != "active":
                continue

            if contract.remaining_months <= 0:
                contract.status = "matured"
                continue

            # A contract originated in month t is first due in month t + 1.
            if contract.origin_month >= self.month:
                continue

            balance = contract.outstanding_principal
            interest_due = balance * contract.monthly_interest_rate

            if contract.remaining_months == 1:
                principal_due = balance
            else:
                principal_due = min(
                    balance,
                    max(0.0, contract.monthly_payment - interest_due),
                )

            delay_probability = contract.installment_delay_probability
            on_time_share = 1.0 - delay_probability
            on_time_principal = principal_due * on_time_share
            on_time_interest = interest_due * on_time_share
            late_principal = principal_due * delay_probability
            late_interest = interest_due * delay_probability

            principal_collected += on_time_principal
            interest_collected += on_time_interest
            self.cash_available += on_time_principal + on_time_interest

            if delay_probability > 0.0:
                delayed_payment_amount_scheduled += late_principal + late_interest
                expected_delayed_installments += delay_probability
                for lag, weight in zip(self.delay_lags_months, self.delay_lag_weights):
                    if weight <= 0.0:
                        continue
                    collection_month = self.month + lag
                    bucket = self._delayed_receivables.setdefault(
                        collection_month,
                        {"principal": 0.0, "interest": 0.0},
                    )
                    bucket["principal"] += late_principal * weight
                    bucket["interest"] += late_interest * weight

            # The contractual amortization schedule advances on its due dates.
            # Any unpaid principal portion is tracked separately in the queue.
            contract.outstanding_principal = max(0.0, balance - principal_due)
            contract.remaining_months -= 1

            if contract.remaining_months == 0:
                contract.outstanding_principal = 0.0
                contract.survival_probability = 0.0
                contract.status = "matured"
            else:
                contract.survival_probability = 1.0

        # This scenario assumes full collection: no permanent default or recovery loss.
        expected_default_count = 0.0
        gross_defaulted_principal = 0.0
        recovery_collected = 0.0
        net_default_loss = 0.0
        financial_pnl = interest_collected

        self.cumulative_principal_collected += principal_collected
        self.cumulative_expected_defaults += expected_default_count
        self.cumulative_net_default_loss += net_default_loss
        self.cumulative_interest_collected += interest_collected
        self.cumulative_delayed_payment_amount_scheduled += delayed_payment_amount_scheduled
        self.cumulative_delayed_payment_amount_collected += delayed_payment_amount_collected
        self.cumulative_expected_delayed_installments += expected_delayed_installments

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
            delayed_payment_amount_scheduled=delayed_payment_amount_scheduled,
            delayed_payment_amount_collected=delayed_payment_amount_collected,
            pending_delayed_receivables=self.pending_delayed_receivables,
            expected_delayed_installments=expected_delayed_installments,
        )
