from customer_state import CustomerState
from simulation_outcome import SimulationOutcome
import numpy as np

class RewardEngine:

    def __init__(
        self,
        history_reward=1.0,
        facility_weight=1.0,
        need_weight=1.0,
        risk_weight=1.0,
        default_cost=10.0,
        facility_decay=1e-9,
        need_decay=1e-9,
    ):

        if not np.isfinite(risk_weight) or risk_weight < 0.0:
            raise ValueError("risk_weight must be finite and non-negative.")
        if not np.isfinite(default_cost) or default_cost < 0.0:
            raise ValueError("default_cost must be finite and non-negative.")

        self.history_reward = history_reward
        self.facility_weight = facility_weight
        self.need_weight = need_weight
        self.risk_weight = float(risk_weight)
        self.default_cost = float(default_cost)

        self.facility_decay = facility_decay

        self.need_decay = need_decay


    def _estimated_need_amount(
        self,
        previous_customer: CustomerState,
    ) -> float:
        """Estimate required loan amount from the current need proxy."""

        score = np.clip(
            previous_customer.total_customer_score,
            0.0,
            100.0,
        )
        # Higher score -> lower estimated financial need.
        need_index = 1.0 - score / 100.0
        min_need = 100_000_000
        max_need = 500_000_000
        return min_need + need_index * (max_need - min_need)

    def _need_coverage(
        self,
        previous_customer: CustomerState,
        approved_unsecured_amount: float,
    ) -> float:
        """Return the fraction of estimated need covered, clipped to [0, 1]."""
        amount = max(0.0, float(approved_unsecured_amount))
        need_amount = self._estimated_need_amount(previous_customer)
        return float(min(amount / need_amount, 1.0))

    def _history_reward(
        self,
        previous_customer: CustomerState,
        coverage: float = 1.0,
    ) -> float:
        """Reward new-to-credit customers, optionally scaled by need coverage."""
        if previous_customer.has_no_history:
            return self.history_reward * float(np.clip(coverage, 0.0, 1.0))
        return 0.0

    def _facility_reward(
        self,
        previous_customer: CustomerState,
        coverage: float = 1.0,
    ) -> float:
        """
        Prioritize customers with a lower unsecured-credit eligibility limit.

        TotalFacilities represents the amount the existing bank credit-scoring
        system says the customer may qualify for; it is not an amount actually
        borrowed. This canonical portfolio reward scales with estimated-need
        coverage.
        """
        eligibility_limit = max(0.0, float(previous_customer.total_facilities))
        priority = 1.0 / (
            1.0 + self.facility_decay * eligibility_limit
        )
        return (
            self.facility_weight
            * priority
            * float(np.clip(coverage, 0.0, 1.0))
        )

    def _need_reward(
        self,
        previous_customer: CustomerState,
        approved_unsecured_amount: float,
    ) -> float:
        coverage = self._need_coverage(
            previous_customer,
            approved_unsecured_amount,
        )
        return self.need_weight * coverage

    def _social_reward(
        self,
        previous_customer: CustomerState,
        approved_unsecured_amount: float,
    ) -> float:
        if approved_unsecured_amount <= 0:
            return 0.0

        coverage = self._need_coverage(
            previous_customer,
            approved_unsecured_amount,
        )
        return (
            self._history_reward(previous_customer, coverage=coverage)
            + self._facility_reward(previous_customer, coverage=coverage)
            + self._need_reward(previous_customer, approved_unsecured_amount)
        )

    def risk_increase_penalty(self, risk_increase: float) -> float:
        """Penalty for a non-negative increase in risk at loan origination."""
        return self.risk_weight * max(0.0, float(risk_increase))

    def recognized_default_penalty(self, recognized_default_count: float) -> float:
        """Apply default_cost only to defaults newly recognized by the simulator."""
        count = float(recognized_default_count)
        if not np.isfinite(count) or count < 0.0:
            raise ValueError("recognized_default_count must be finite and non-negative.")
        return self.default_cost * count

    def _risk_penalty(
            self,
            previous_customer: CustomerState,
            outcome: SimulationOutcome
    ) -> float:
        delta_risk = max(
            0.0,
            outcome.next_state.risk - previous_customer.risk,
        )
        return self.risk_increase_penalty(delta_risk)

    def _default_penalty(
            self,
            outcome: SimulationOutcome
        ) -> float:
        """
        A sampled term-PD event at origination is not a recognized default.

        Portfolio-mode default_cost is charged only by the environment when
        PortfolioSimulator reports a newly recognized >DPD-threshold event.
        This compatibility method therefore intentionally applies no penalty.
        """
        return 0.0
    
    def _risk_cost(
            self,
            previous_customer: CustomerState,
            outcome: SimulationOutcome
    ) -> float:
        
        cost = (
            self._risk_penalty(previous_customer, outcome,) + 
            self._default_penalty(outcome,)
        )

        return cost
    
    def calculate(
            self,
            previous_customer: CustomerState,
            outcome: SimulationOutcome,
            approved_unsecured_amount

    ) -> float:
        
        reward = (
            self._social_reward(previous_customer, approved_unsecured_amount) -
            self._risk_cost(previous_customer, outcome,)
        )
        return reward

    def calculate_expected(
            self,
            previous_customer: CustomerState,
            outcome: SimulationOutcome,
            approved_unsecured_amount

    ) -> float:

        social_reward = self._social_reward(
        previous_customer,
        approved_unsecured_amount
    )

        risk_penalty = self._risk_penalty(
            previous_customer,
            outcome
        )

        # Do not turn a term-level probability into an immediate default
        # penalty. The portfolio environment applies default_cost only when
        # the loan simulator actually registers a default event.
        return social_reward - risk_penalty


class LegacySingleCustomerRewardEngine(RewardEngine):
    """Preserve the historical reward formula for the retired single-customer workflow.

    The portfolio-level environment must use RewardEngine, which is always
    amount-sensitive. This compatibility class is intentionally isolated from
    portfolio training and evaluation so old single-customer experiments keep
    their original amount-independent access bonuses.
    """

    def _social_reward(
        self,
        previous_customer: CustomerState,
        approved_unsecured_amount: float,
    ) -> float:
        if approved_unsecured_amount <= 0:
            return 0.0

        coverage = self._need_coverage(
            previous_customer,
            approved_unsecured_amount,
        )
        return (
            self._history_reward(previous_customer, coverage=1.0)
            + self._facility_reward(previous_customer, coverage=1.0)
            + self._need_reward(previous_customer, approved_unsecured_amount)
        )
