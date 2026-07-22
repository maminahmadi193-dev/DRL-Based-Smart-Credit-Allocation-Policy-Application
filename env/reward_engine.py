from .customer_state import CustomerState
from .simulation_outcome import SimulationOutcome


class RewardEngine:

    def __init__(
        self,
        history_reward=1.0,
        facility_weight=1.0,
        need_weight=1.0,
        risk_weight=1.0,
        default_cost=10.0,
        facility_decay=1e-9,
        need_decay=1e-9
    ):

        self.history_reward = history_reward

        self.facility_weight = facility_weight

        self.need_weight = need_weight

        self.risk_weight = risk_weight

        self.default_cost = default_cost

        self.facility_decay = facility_decay

        self.need_decay = need_decay


    def _history_reward(
        self,
        previous_customer: CustomerState,
    ) -> float:

        if previous_customer.has_no_history:
            return self.history_reward

        return 0.0
                
    def _facility_reward(
        self,
        previous_customer: CustomerState,
    ) -> float:
        """
        Reward customers with lower previous access
        to unsecured facilities.
        """

        reward = (
            1.0 /
            (
                1.0 +
                self.facility_decay *
                previous_customer.total_facilities
            )
        )

        return self.facility_weight * reward

    def _need_reward(
        self,
        previous_customer: CustomerState,
    ) -> float:
        """
        Estimate customer's financial need.

        TODO:
        Replace TotalCustomerScore with
        a dedicated Need Index.
        """

        reward = (
            1.0 /
            (
                1.0 +
                self.need_decay *
                previous_customer.total_customer_score
            )
        )

        return self.need_weight * reward


    def _social_reward(
        self,
        previous_customer: CustomerState,
        approved_unsecured_amount: float
    ) -> float:

        if approved_unsecured_amount <= 0 :
            return 0.0
    
        reward = (
            self._history_reward(previous_customer)
            + self._facility_reward(previous_customer)
            + self._need_reward(previous_customer)
        )

        return reward
    
    def _risk_penalty(
            self,
            previous_customer: CustomerState,
            outcome: SimulationOutcome
    ) -> float:
        
        delta_risk = max(0.0, 
            outcome.next_state.risk -
            previous_customer.risk
        )

        return self.risk_weight * delta_risk 
    
    def _default_penalty(
            self,
            outcome: SimulationOutcome
        ) -> float:
        if outcome.default_event:
            return self.default_cost
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