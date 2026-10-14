
from dataclasses import dataclass
from datetime import date
import math
from typing import Optional, Union

import numpy as np

from customer_state import CustomerState
from customer_simulator import CustomerSimulator


CustomerId = Optional[Union[str, int]]


@dataclass
class LoanContract:
    """One contract with a stochastic full-or-zero installment path."""

    loan_id: int
    customer_id: CustomerId
    origin_month: int

    principal: float
    outstanding_principal: float
    monthly_payment: float
    monthly_interest_rate: float
    monthly_hazard: float
    monthly_miss_probability: float
    default_probability_over_term: float

    term_months: int
    remaining_months: int

    # At most one default-causing missed installment is simulated per loan.
    missed_installment_occurred: bool = False
    missed_due_date: Optional[date] = None
    missed_principal: float = 0.0
    missed_interest: float = 0.0
    default_recognized: bool = False
    default_recognition_date: Optional[date] = None
    late_payment_collected: bool = False

    # Retained for compatibility with existing reporting code.
    survival_probability: float = 1.0

    status: str = "active"


@dataclass
class MonthlyPortfolioResult:
    """Realized monthly collections, default recognitions, and cash flows."""

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

    Each installment is either paid in full on its due date or not paid at all.
    A seeded random draw controls the first missed installment for each loan:
    after that first missed installment, later scheduled installments are paid
    on time. The missed installment is collected in full after a configurable
    lag (four months in the base case, twelve months in stress tests).

    Default is recognized at the first monthly checkpoint where days past due
    is strictly greater than 90. This is a stochastic path simulator, not
    expected-fraction accounting. No permanent credit loss is modeled.
    """

    EPSILON = 1e-6

    def __init__(
        self,
        initial_budget: float,
        monthly_capital_injection: float = 0.0,
        loan_term_months: int = 18,
        monthly_interest_rate: float = 0.0,
        recovery_rate: float = 0.0,
        default_collection_lag_months: int = 4,
        default_dpd_threshold_days: int = 90,
        customer_simulator: Optional[CustomerSimulator] = None,
        rng: Optional[np.random.Generator] = None,
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
                "recovery_rate is not used when all overdue installments are "
                "assumed to be fully collected. It must remain zero."
            )
        if (
            not isinstance(default_collection_lag_months, int)
            or default_collection_lag_months < 4
            or default_collection_lag_months > 12
        ):
            raise ValueError(
                "default_collection_lag_months must be an integer from 4 to 12; "
                "the lag must allow the installment to exceed the DPD threshold."
            )
        if (
            not isinstance(default_dpd_threshold_days, int)
            or default_dpd_threshold_days < 1
        ):
            raise ValueError("default_dpd_threshold_days must be a positive integer.")
        if default_collection_lag_months * 30 <= default_dpd_threshold_days:
            raise ValueError(
                "The configured collection lag must extend beyond the DPD threshold "
                "at the monthly checkpoint cadence."
            )

        self.initial_budget = float(initial_budget)
        self.cash_available = float(initial_budget)

        self.monthly_capital_injection = float(
            monthly_capital_injection
        )

        self.loan_term_months = int(loan_term_months)
        self.monthly_interest_rate = float(monthly_interest_rate)
        self.recovery_rate = 0.0  # Deprecated compatibility field; no permanent loss.
        self.default_collection_lag_months = int(default_collection_lag_months)
        self.default_dpd_threshold_days = int(default_dpd_threshold_days)
        self.max_delay_months = self.default_collection_lag_months
        self._delayed_receivables: dict[int, dict[str, object]] = {}
        self.rng = rng if rng is not None else np.random.default_rng()

        self.customer_simulator = (
            customer_simulator
            if customer_simulator is not None
            else CustomerSimulator()
        )

        self.month = 0
        self.next_loan_id = 1

        self.contracts: list[LoanContract] = []

        self.total_disbursed = 0.0
        self.cumulative_expected_defaults = 0.0  # Compatibility alias for old reports.
        self.cumulative_recognized_defaults = 0.0
        self.cumulative_net_default_loss = 0.0
        self.cumulative_principal_collected = 0.0
        self.cumulative_interest_collected = 0.0
        self.cumulative_delayed_payment_amount_scheduled = 0.0
        self.cumulative_delayed_payment_amount_collected = 0.0
        self.cumulative_expected_delayed_installments = 0.0  # Compatibility alias.
        self.cumulative_missed_installments = 0.0

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
        """Total principal and interest on missed installments not yet collected."""
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
            for offset in range(1, 13)
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
            expected_on_time_share = (
                1.0
                if contract.missed_installment_occurred
                else 1.0 - contract.monthly_miss_probability
            )
            scheduled += (principal_due + interest_due) * expected_on_time_share
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

        # Convert the updated customer risk to a monthly hazard and probability.
        # PD is specific to this experiment's contractual loan term.
        monthly_hazard = self.customer_simulator._estimate_hazard(
            customer_after_loan
        )
        monthly_miss_probability = (
            self.customer_simulator._calculate_monthly_event_probability(
                monthly_hazard
            )
        )
        default_probability_over_term = (
            self.customer_simulator._calculate_default_probability(
                monthly_hazard,
                horizon_months=self.loan_term_months,
            )
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
            monthly_miss_probability=monthly_miss_probability,
            default_probability_over_term=default_probability_over_term,
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

    @staticmethod
    def _month_date(month: int) -> date:
        """Map the simulated month counter to a real calendar-month checkpoint."""
        if month < 1:
            raise ValueError("month must be >= 1")
        year = 2001 + (month - 1) // 12
        month_of_year = (month - 1) % 12 + 1
        return date(year, month_of_year, 1)

    def advance_month(
        self,
        capital_injection: Optional[float] = None,
    ) -> MonthlyPortfolioResult:
        """Advance one monthly checkpoint and process full-or-zero installments."""
        injection = (
            self.monthly_capital_injection
            if capital_injection is None
            else float(capital_injection)
        )
        if not math.isfinite(injection) or injection < 0:
            raise ValueError("capital_injection must be finite and non-negative.")

        cash_before = self.cash_available
        self.month += 1
        current_date = self._month_date(self.month)
        self.cash_available += injection

        principal_collected = 0.0
        interest_collected = 0.0
        delayed_payment_amount_collected = 0.0
        delayed_payment_amount_scheduled = 0.0
        default_count = 0.0
        gross_defaulted_principal = 0.0
        delayed_installment_count = 0.0

        # Recognize default before collecting receipts scheduled this month.
        # The threshold is strictly more than 90 calendar days past due.
        for contract in self.contracts:
            if (
                contract.missed_installment_occurred
                and not contract.default_recognized
                and contract.missed_due_date is not None
                and (current_date - contract.missed_due_date).days > self.default_dpd_threshold_days
            ):
                contract.default_recognized = True
                contract.default_recognition_date = current_date
                default_count += 1.0
                gross_defaulted_principal += contract.missed_principal

        # Collect previously missed installments when their full collection lag ends.
        overdue = self._delayed_receivables.pop(
            self.month,
            {"principal": 0.0, "interest": 0.0, "items": []},
        )
        delayed_principal = float(overdue.get("principal", 0.0))
        delayed_interest = float(overdue.get("interest", 0.0))
        delayed_payment_amount_collected += delayed_principal + delayed_interest
        principal_collected += delayed_principal
        interest_collected += delayed_interest
        self.cash_available += delayed_principal + delayed_interest

        for item in overdue.get("items", []):
            contract_id = item.get("contract_id")
            for contract in self.contracts:
                if contract.loan_id == contract_id:
                    contract.late_payment_collected = True
                    break

        for contract in self.contracts:
            if contract.status != "active":
                continue

            if contract.remaining_months <= 0:
                contract.status = "matured"
                continue

            # A loan originated in month t is first due in month t + 1.
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

            installment_amount = principal_due + interest_due
            missed_this_installment = False

            # Each loan may have one default-causing missed installment.
            # An individual installment is either paid in full or not at all.
            if not contract.missed_installment_occurred:
                missed_this_installment = (
                    self.rng.random() < contract.monthly_miss_probability
                )

            if missed_this_installment:
                contract.missed_installment_occurred = True
                contract.missed_due_date = current_date
                contract.missed_principal = principal_due
                contract.missed_interest = interest_due

                collection_month = (
                    self.month + self.default_collection_lag_months
                )
                bucket = self._delayed_receivables.setdefault(
                    collection_month,
                    {"principal": 0.0, "interest": 0.0, "items": []},
                )
                bucket["principal"] = float(bucket.get("principal", 0.0)) + principal_due
                bucket["interest"] = float(bucket.get("interest", 0.0)) + interest_due
                bucket["items"].append({
                    "contract_id": contract.loan_id,
                    "principal": principal_due,
                    "interest": interest_due,
                    "due_date": current_date.isoformat(),
                    "collection_date": self._month_date(collection_month).isoformat(),
                })

                delayed_payment_amount_scheduled += installment_amount
                delayed_installment_count += 1.0
            else:
                # Full on-time collection; never collect a fractional installment.
                principal_collected += principal_due
                interest_collected += interest_due
                self.cash_available += installment_amount

            # Advance the contractual amortization schedule. A missed payment's
            # principal remains separately represented in pending receivables.
            contract.outstanding_principal = max(0.0, balance - principal_due)
            contract.remaining_months -= 1
            if contract.remaining_months == 0:
                contract.outstanding_principal = 0.0
                contract.survival_probability = 0.0
                contract.status = "matured"
            else:
                contract.survival_probability = 1.0

        # All missed installments are collected in full; no permanent loss.
        recovery_collected = 0.0
        net_default_loss = 0.0
        financial_pnl = interest_collected

        self.cumulative_principal_collected += principal_collected
        self.cumulative_expected_defaults += default_count
        self.cumulative_recognized_defaults += default_count
        self.cumulative_net_default_loss += net_default_loss
        self.cumulative_interest_collected += interest_collected
        self.cumulative_delayed_payment_amount_scheduled += delayed_payment_amount_scheduled
        self.cumulative_delayed_payment_amount_collected += delayed_payment_amount_collected
        self.cumulative_expected_delayed_installments += delayed_installment_count
        self.cumulative_missed_installments += delayed_installment_count

        return MonthlyPortfolioResult(
            month=self.month,
            cash_before=cash_before,
            capital_injection=injection,
            principal_collected=principal_collected,
            interest_collected=interest_collected,
            recovery_collected=recovery_collected,
            expected_default_count=default_count,
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
            expected_delayed_installments=delayed_installment_count,
        )
