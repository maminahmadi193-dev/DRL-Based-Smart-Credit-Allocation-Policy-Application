from customer_state import CustomerState
from simulation_outcome import SimulationOutcome
import numpy as np


class CustomerSimulator:

    def __init__(
        self,
        alpha=2.0,
        beta=1.0,
        gamma = 1.0,
        hazard_slope=0.08,
        hazard_center=60,
        max_hazard=0.005,
        hazard_intercept=0.0,
        horizon_months=18,
    ):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.hazard_slope = hazard_slope
        self.hazard_center = hazard_center
        self.max_hazard = float(max_hazard)
        self.hazard_intercept = float(hazard_intercept)
        if not np.isfinite(self.hazard_intercept):
            raise ValueError("hazard_intercept must be finite.")

        self.horizon_months = int(horizon_months)

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

        amount = float(approved_unsecured_amount)
        if not np.isfinite(amount) or amount < 0.0:
            raise ValueError("approved_unsecured_amount must be finite and non-negative.")

        updated = customer.copy()

        # Normalize the financial quantities before deriving ratios. This keeps
        # malformed/legacy negative values from propagating into the state.
        updated.debt = max(0.0, float(updated.debt))
        updated.origin_amount = max(0.0, float(updated.origin_amount))
        updated.loan_count = max(0, int(updated.loan_count))

        if amount > 0.0:
            updated.debt += amount
            updated.origin_amount += amount
            updated.loan_count += 1

        # Derived debt features are always non-negative. DebtPerLoan is based
        # on total current debt divided by the cumulative number of loans.
        updated.debt_ratio = (
            updated.debt / updated.origin_amount
            if updated.origin_amount > 0.0
            else 0.0
        )
        updated.debt_per_loan = (
            updated.debt / updated.loan_count
            if updated.loan_count > 0
            else 0.0
        )

        updated.debt_ratio = max(0.0, float(updated.debt_ratio))
        updated.debt_per_loan = max(0.0, float(updated.debt_per_loan))

        return updated

    # ======================================================
    # Calculate Capacity Pressure
    # ======================================================

    def _calculate_capacity_pressure(
        self,
        monthly_income: float,
        approved_unsecured_amount: float,
        ) -> float:

        if approved_unsecured_amount <= 0:
            return 0.0

        # No positive monthly income means the installment has no observed
        # income base to support it; treat capacity pressure as maximal.
        # The caller handles this as a modeling assumption for zero-income
        # applicants, not as a learned estimate from historical defaults.
        if monthly_income <= 0:
            return 1.0

        return (
            approved_unsecured_amount
            / (monthly_income + approved_unsecured_amount)
        )
        
    # ======================================================
    # Behaviour Models
    # ======================================================

    # def _update_risk(
    #     self,
    #     customer: CustomerState,
    #     previous_customer: CustomerState,
    #     approved_unsecured_amount: float
    # ) -> float:
    #     """
    #     Estimate customer's new risk after
    #     state transition.
    #     """

    #     debt_ratio_change = (
    #         customer.debt_ratio -
    #         previous_customer.debt_ratio
    #     )

    #     if previous_customer.debt_per_loan > 0:

    #         debt_per_loan_change = (
    #             customer.debt_per_loan -
    #             previous_customer.debt_per_loan
    #         ) / previous_customer.debt_per_loan

    #     else:

    #         debt_per_loan_change = 0.0


    #     delta_risk = (
    #         + self.alpha * debt_ratio_change
    #         + self.beta * debt_per_loan_change
    #     )


    #     updated_risk = (
    #         previous_customer.risk
    #         + delta_risk 
    #     )


    #     updated_risk = np.clip(
    #         updated_risk,
    #         0.0,
    #         100.0
    #     )

    #     return updated_risk

    def _update_risk(
        self,
        customer: CustomerState,
        previous_customer: CustomerState,
        approved_unsecured_amount: float
        ) -> float:
        """
        Estimate customer's new risk after state transition.
        """

        # --------------------------------------------------
        # 1. Debt ratio effect
        # --------------------------------------------------

        debt_ratio_change = max(
            0.0,
            float(customer.debt_ratio) - float(previous_customer.debt_ratio),
        )

        # --------------------------------------------------
        # 2. Debt per loan effect
        # --------------------------------------------------
        # A new loan must not reduce risk merely because the average debt per
        # loan decreases when another (smaller) loan is added. When the
        # previous average is zero, normalize by the largest available action.
        if previous_customer.debt_per_loan > 0.0:
            debt_per_loan_change = max(
                0.0,
                (
                    float(customer.debt_per_loan)
                    - float(previous_customer.debt_per_loan)
                ) / float(previous_customer.debt_per_loan),
            )
        else:
            debt_per_loan_change = min(
                max(float(customer.debt_per_loan), 0.0) / 500_000_000.0,
                1.0,
            )

        # --------------------------------------------------
        # 3. Repayment capacity pressure
        # --------------------------------------------------
        capacity_pressure = self._calculate_capacity_pressure(
            monthly_income=previous_customer.monthly_income,
            approved_unsecured_amount=approved_unsecured_amount,
        )

        # --------------------------------------------------
        # 4. Total risk change
        # --------------------------------------------------
        # Risk loading at origination is non-negative. Its dynamic decay is
        # handled by PortfolioSimulator as principal is repaid/collected.
        delta_risk = max(
            0.0,
            self.alpha * debt_ratio_change
            + self.beta * debt_per_loan_change
            + self.gamma * capacity_pressure,
        )

        # --------------------------------------------------
        # 5. Updated risk
        # --------------------------------------------------

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

        z = (
            self.hazard_slope * (risk - self.hazard_center)
            + self.hazard_intercept
        )

        # The additive intercept calibrates the population-level default rate
        # while preserving risk ordering and the configured hazard ceiling.
        z = float(np.clip(z, -60.0, 60.0))
        hazard_score = 1.0 / (1.0 + np.exp(-z))

        hazard = self.max_hazard * hazard_score

        return float(hazard)


    @staticmethod
    def calibrate_hazard_intercept(
        reference_risks,
        target_term_default_rate: float,
        max_hazard: float,
        hazard_slope: float,
        hazard_center: float,
        horizon_months: int,
    ) -> float:
        """
        Find a logistic intercept so mean term PD on a reference population
        matches the supplied target, without using individual default labels.

        reference_risks must represent the post-loan risk under a stated
        reference exposure (e.g. a hypothetical 100M loan). The returned
        intercept is a population calibration parameter, not an estimate of
        each applicant's observed default label.
        """
        risks = np.asarray(reference_risks, dtype=float).reshape(-1)
        if risks.size == 0 or not np.all(np.isfinite(risks)):
            raise ValueError("reference_risks must be a non-empty finite array.")
        if not np.isfinite(target_term_default_rate) or not 0.0 < target_term_default_rate < 1.0:
            raise ValueError("target_term_default_rate must be strictly between 0 and 1.")
        if not np.isfinite(max_hazard) or max_hazard <= 0.0:
            raise ValueError("max_hazard must be finite and positive for calibration.")
        if not np.isfinite(hazard_slope) or hazard_slope < 0.0:
            raise ValueError("hazard_slope must be finite and non-negative.")
        if not np.isfinite(hazard_center):
            raise ValueError("hazard_center must be finite.")
        if int(horizon_months) <= 0:
            raise ValueError("horizon_months must be positive.")

        maximum_attainable_pd = float(-np.expm1(-max_hazard * int(horizon_months)))
        if target_term_default_rate >= maximum_attainable_pd:
            raise ValueError(
                "Target term default rate must be below the maximum attainable "
                f"PD ({maximum_attainable_pd:.6%}) for max_hazard and loan term."
            )

        linear_predictor = hazard_slope * (
            np.clip(risks, 0.0, 100.0) - hazard_center
        )

        def mean_term_pd(intercept: float) -> float:
            z = np.clip(linear_predictor + intercept, -60.0, 60.0)
            sigmoid = 1.0 / (1.0 + np.exp(-z))
            hazards = max_hazard * sigmoid
            term_pds = -np.expm1(-hazards * int(horizon_months))
            return float(np.mean(term_pds))

        low, high = -60.0, 60.0
        for _ in range(100):
            midpoint = (low + high) / 2.0
            if mean_term_pd(midpoint) < target_term_default_rate:
                low = midpoint
            else:
                high = midpoint

        return float((low + high) / 2.0)

    def _calculate_monthly_event_probability(
        self,
        hazard: float,
    ) -> float:
        """Probability of an event during one month under a constant hazard."""
        hazard = float(hazard)
        if not np.isfinite(hazard) or hazard < 0.0:
            raise ValueError("hazard must be finite and non-negative.")
        return float(-np.expm1(-hazard))

    def _calculate_default_probability(
        self,
        hazard: float,
        horizon_months: int = None,
    ) -> float:
        """Cumulative probability of at least one event over the chosen term."""
        hazard = float(hazard)
        horizon = self.horizon_months if horizon_months is None else int(horizon_months)
        if not np.isfinite(hazard) or hazard < 0.0:
            raise ValueError("hazard must be finite and non-negative.")
        if horizon <= 0:
            raise ValueError("horizon_months must be positive.")
        return float(-np.expm1(-hazard * horizon))

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