import gymnasium as gym
import numpy as np

from .customer_state import CustomerState
from .customer_simulator import CustomerSimulator
from .reward_engine import RewardEngine


class LoanEnv(gym.Env):

    def __init__(self, raw_df, scaler):

        super().__init__()

        self.raw_df = raw_df
        self.scaler = scaler

        self.simulator = CustomerSimulator()

        self.reward_engine = RewardEngine()

        self.current_customer = None

        self.current_index = None

        self.actions = np.array([
            0,
            100_000_000,
            200_000_000,
            300_000_000,
            400_000_000,
            500_000_000,
        ], dtype= np.float32)

        self.action_space = gym.spaces.Discrete(
            len(self.actions)
        )

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(CustomerState.FEATURE_COUNT,),
            dtype=np.float32,
        )

    def _get_scaled_observation(self):

        obs = self.current_customer.to_array()

        obs = self.scaler.transform(
            obs.reshape(1, -1)
        )[0]

        return obs.astype(np.float32)


    def _row_to_customer_state(self, row,) -> CustomerState:
        return CustomerState(
            monthly_income=row["MountlyIncome"],

            income_score=row["IncomeScore"],
            job_score=row["JobScore"],
            asset_score=row["AssetScore"],
            total_customer_score=row["TotalCustomerScore"],

            total_facilities=row["TotalFacilities"],

            debt=row["DebtAmount"],
            origin_amount=row["OriginAmount"],
            loan_count=int(row["LoanCounts"]),

            debt_ratio=row["DebtRatio"],
            debt_per_loan=row["DebtPerLoan"],

            has_no_history=bool(row["Has_No_History"]),
            employed=bool(row["Employed"]),

            score_min=row["Score_Min"],
            score_max=row["Score_Max"],

            risk_min=row["Risk_Min"],
            risk_max=row["Risk_Max"],

            risk=(
                row["Risk_Min"] +
                row["Risk_Max"]
            ) / 2,
        )

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        self.current_index = np.random.randint(len(self.raw_df))

        raw_row = self.raw_df.iloc[self.current_index]

        self.current_customer = self._row_to_customer_state(raw_row)

        return self._get_scaled_observation(), {}    
    
    def step(self, action):
        
        action = int(action)

        approved_amount = self.actions[action]

        previous_customer = self.current_customer.copy()

        outcome = self.simulator.simulate(
            customer= self.current_customer,
            approved_unsecured_amount= approved_amount
        )

        reward = self.reward_engine.calculate(
            previous_customer= previous_customer,
            outcome= outcome,
            approved_unsecured_amount= approved_amount
        )

        self.current_customer = outcome.next_state

        terminated = True

        truncated = False

        info = {
            "approved_amount": approved_amount,

            "probability_of_default":
                outcome.probability_of_default,

            "default_event":
                outcome.default_event,
        }

        return (
            self._get_scaled_observation(),

            reward,

            terminated,

            truncated,

            info,
        )

    def render(self):
        print(self.current_customer)