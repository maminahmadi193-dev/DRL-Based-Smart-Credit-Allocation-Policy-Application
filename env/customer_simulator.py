from .customer_state import CustomerState
from .simulation_outcome import SimulationOutcome
import numpy as np


class CustomerSimulator:

    def __init__(
        self,
        alpha=2.0,
        beta=1.0,
        hazard_slope=0.08,
        hazard_center=60,
        max_hazard = 0.005,
        horizon_months=18,
    ):
        self.alpha = alpha
        self.beta = beta

        self.hazard_slope = hazard_slope
        self.hazard_center = hazard_center
        self.max_hazard = max_hazard

        self.horizon_months = horizon_months

    # ======================================================
    # Deterministic State Updates
    # ======================================================

    def _update_customer_state(
        self,
        customer: CustomerState,
        approved_unsecured_amount: float,
    ) -> CustomerState:
        """
        Apply deterministic state transitions after
        approving an unsecured loan.
        """

        updated = customer.copy()

        if approved_unsecured_amount ==0 : 
            return updated

        # Total debt increases
        updated.debt += approved_unsecured_amount

        # Total originated loans increases
        updated.origin_amount += approved_unsecured_amount

        # One more active loan
        updated.loan_count += 1

        # Derived features
        updated.debt_ratio = (
            updated.debt /
            updated.origin_amount
            if updated.origin_amount > 0
            else 0.0
        )

        updated.debt_per_loan = (
            updated.debt /
            updated.loan_count
            if updated.loan_count > 0
            else 0.0
        )

        return updated
    
    # ======================================================
    # Behaviour Models
    # ======================================================

    def _update_risk(
        self,
        customer: CustomerState,
        previous_customer: CustomerState,
        approved_unsecured_amount: float
    ) -> float:
        """
        Estimate customer's new risk after
        state transition.
        """

        debt_ratio_change = (
            customer.debt_ratio -
            previous_customer.debt_ratio
        )

        if previous_customer.debt_per_loan > 0:

            debt_per_loan_change = (
                customer.debt_per_loan -
                previous_customer.debt_per_loan
            ) / previous_customer.debt_per_loan

        else:

            debt_per_loan_change = 0.0


        delta_risk = (
            + self.alpha * debt_ratio_change
            + self.beta * debt_per_loan_change
        )


        updated_risk = (
            previous_customer.risk
            + delta_risk 
        )


        updated_risk = np.clip(
            updated_risk,
            0.0,
            100.0
        )

        return updated_risk


    def _estimate_hazard(
        self,
        customer: CustomerState,
    ) -> float:
        
        risk = np.clip(customer.risk, 0, 100)

        z = self.hazard_slope * (
            risk - self.hazard_center
        )

        hazard_score = (1 / (1 + np.exp(-z)))

        hazard = self.max_hazard * hazard_score

        return float(hazard)


    def _calculate_default_probability(
        self,
        hazard: float,
    )  -> float :
        probability_of_default = 1 - np.exp(-hazard * self.horizon_months)

        return float(probability_of_default)

    # ======================================================
    # Simulation
    # ======================================================

    def simulate(
        self,
        customer: CustomerState,
        approved_unsecured_amount: float,
    ) -> SimulationOutcome:
        
        previous_customer = customer.copy()

        customer = self._update_customer_state(customer, approved_unsecured_amount)

        customer.risk = self._update_risk(customer, previous_customer, approved_unsecured_amount)

        hazard = self._estimate_hazard(customer)

        probability_of_default = self._calculate_default_probability(hazard)

        default_event = np.random.rand() < probability_of_default


        return SimulationOutcome(
            next_state=customer,
            probability_of_default=probability_of_default,
            default_event=default_event,
        )