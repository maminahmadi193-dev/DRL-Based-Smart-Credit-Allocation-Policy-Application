from dataclasses import dataclass
import copy
import numpy as np


@dataclass
class CustomerState:

    # Financial profile
    monthly_income: float

    income_score: float
    job_score: float
    asset_score: float
    total_customer_score: float

    # Banking information
    total_facilities: float
    debt: float
    origin_amount: float
    loan_count: int

    # Derived features
    debt_ratio: float
    debt_per_loan: float

    # Behaviour
    has_no_history: bool
    employed: bool

    # Credit score
    score_min: float
    score_max: float

    # Risk
    risk_min: float
    risk_max: float

    # Dynamic state variable
    risk: float

    def copy(self):
        return copy.deepcopy(self)
    def to_array(self):
        return np.array([
        self.monthly_income,

        self.income_score,
        self.job_score,
        self.asset_score,
        self.total_customer_score,

        self.total_facilities,
        self.debt,
        self.origin_amount,
        self.loan_count,

        self.debt_ratio,
        self.debt_per_loan,

        self.has_no_history,
        self.employed,

        self.score_min,
        self.score_max,

        self.risk_min,
        self.risk_max,

        self.risk
        ], dtype= np.float32)
    
    FEATURE_COUNT = 18