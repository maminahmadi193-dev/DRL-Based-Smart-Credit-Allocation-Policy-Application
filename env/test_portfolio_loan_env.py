
import unittest

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from gymnasium.utils.env_checker import check_env

from customer_state import CustomerState
from portfolio_loan_env import PortfolioLoanEnv


FEATURE_COLUMNS = [
    "MountlyIncome",
    "IncomeScore",
    "JobScore",
    "AssetScore",
    "TotalCustomerScore",
    "TotalFacilities",
    "DebtAmount",
    "OriginAmount",
    "LoanCounts",
    "DebtRatio",
    "DebtPerLoan",
    "Has_No_History",
    "Employed",
    "Score_Min",
    "Score_Max",
    "Risk_Min",
    "Risk_Max",
    "Risk",
]


def make_dataset(n=30):
    """Create a small deterministic customer dataset."""

    rng = np.random.default_rng(42)

    income = np.linspace(10_000_000, 80_000_000, n)
    debt = np.linspace(0, 50_000_000, n)
    loan_counts = np.arange(n) % 5

    origin_amount = np.where(
        loan_counts > 0,
        np.maximum(debt, 1.0) * 1.5,
        0.0,
    )

    debt_ratio = np.divide(
        debt,
        origin_amount,
        out=np.zeros(n, dtype=float),
        where=origin_amount > 0,
    )

    debt_per_loan = np.divide(
        debt,
        loan_counts,
        out=np.zeros(n, dtype=float),
        where=loan_counts > 0,
    )

    risk_min = np.linspace(10, 50, n)

    df = pd.DataFrame({
        "MountlyIncome": income,
        "IncomeScore": np.linspace(20, 90, n),
        "JobScore": np.linspace(30, 95, n),
        "AssetScore": np.linspace(10, 85, n),
        "TotalCustomerScore": np.linspace(20, 90, n),
        "TotalFacilities": np.linspace(0, 50_000_000, n),
        "DebtAmount": debt,
        "OriginAmount": origin_amount,
        "LoanCounts": loan_counts,
        "DebtRatio": debt_ratio,
        "DebtPerLoan": debt_per_loan,
        "Has_No_History": (np.arange(n) % 3 == 0).astype(int),
        "Employed": (np.arange(n) % 4 != 0).astype(int),
        "Score_Min": np.linspace(400, 600, n),
        "Score_Max": np.linspace(500, 700, n),
        "Risk_Min": risk_min,
        "Risk_Max": risk_min + 5,
    })

    df["Risk"] = (
        df["Risk_Min"] + df["Risk_Max"]
    ) / 2

    # Small variation makes the data suitable for scaler fitting.
    df["JobScore"] += rng.normal(0, 0.1, size=n)

    scaler = StandardScaler()
    scaler.fit(df[FEATURE_COLUMNS])

    return df, scaler


def make_env(**overrides):
    df, scaler = make_dataset()

    config = {
        "initial_budget": 100_000_000,
        "monthly_capital_injection": 0,
        "applicants_per_month": 1,
        "origination_months": 2,
        "loan_term_months": 2,
        "annual_interest_rate": 0.0,
        "payment_delay_probability": 0.0,
    }

    config.update(overrides)

    return PortfolioLoanEnv(
        raw_df=df,
        scaler=scaler,
        **config,
    )


class TestPortfolioLoanEnv(unittest.TestCase):

    def test_reset_returns_valid_observation(self):
        env = make_env()

        observation, info = env.reset(seed=42)

        expected_features = (
            CustomerState.FEATURE_COUNT
            + env.PORTFOLIO_FEATURE_COUNT
        )

        self.assertEqual(
            observation.shape,
            (expected_features,),
        )

        self.assertEqual(
            observation.dtype,
            np.float32,
        )

        self.assertTrue(
            env.observation_space.contains(observation)
        )

        self.assertEqual(info["month"], 1)
        self.assertEqual(info["phase"], "allocation")
        self.assertAlmostEqual(
            info["available_budget"],
            100_000_000,
        )

    def test_info_customer_index_matches_the_decision_customer(self):
        env = make_env(
            applicants_per_month=2,
            origination_months=1,
            loan_term_months=2,
        )
        env.reset(seed=42)
        expected_customer_index = env.current_row_index

        _, _, _, _, info = env.step(0)

        self.assertEqual(info["customer_index"], expected_customer_index)
        env.close()

    def test_budget_is_shared_between_applicants(self):
        env = make_env(
            initial_budget=100_000_000,
            applicants_per_month=2,
            origination_months=1,
            loan_term_months=2,
        )

        env.reset(seed=42)

        # The first applicant receives the full budget.
        _, _, terminated, _, first_info = env.step(1)

        self.assertFalse(terminated)
        self.assertFalse(first_info["invalid_action"])
        self.assertEqual(
            first_info["approved_amount"],
            100_000_000,
        )

        # The second applicant requests money that is no longer
        # available. The environment must not fund that request.
        _, _, terminated, _, second_info = env.step(1)

        self.assertTrue(second_info["invalid_action"])
        self.assertEqual(second_info["approved_amount"], 0)
        self.assertEqual(
            len(env.portfolio.contracts),
            1,
        )

        # One final allocation cohort has finished.
        self.assertEqual(env.portfolio.month, 2)
        self.assertEqual(env.phase, "settlement")
        self.assertFalse(terminated)

    def test_every_origination_month_is_processed(self):
        env = make_env(
            applicants_per_month=1,
            origination_months=3,
            loan_term_months=2,
        )

        env.reset(seed=42)

        # Zero-loan actions isolate the calendar logic
        # from loan cash flows.
        for expected_allocation_month in (1, 2):
            _, _, terminated, _, info = env.step(0)

            self.assertFalse(terminated)
            self.assertEqual(info["month"], expected_allocation_month + 1)
            self.assertEqual(env.phase, "allocation")

        _, _, terminated, _, info = env.step(0)

        self.assertTrue(terminated)
        self.assertEqual(info["month"], 4)
        self.assertEqual(env.phase, "settlement")

    def test_settlement_continues_until_contract_matures(self):
        env = make_env(
            initial_budget=100_000_000,
            applicants_per_month=1,
            origination_months=1,
            loan_term_months=2,
        )

        env.reset(seed=42)

        # Grant a single loan in the only origination month.
        _, _, terminated, _, info = env.step(1)

        self.assertFalse(terminated)
        self.assertEqual(info["approved_amount"], 100_000_000)
        self.assertEqual(env.phase, "settlement")
        self.assertEqual(env.portfolio.month, 2)
        self.assertGreater(
            env.portfolio.active_contract_count,
            0,
        )

        # During settlement the action is ignored, but the
        # outstanding contract must continue being serviced.
        _, _, terminated, _, info = env.step(5)

        self.assertTrue(info["action_ignored"])
        self.assertTrue(terminated)
        self.assertEqual(env.portfolio.month, 3)
        self.assertEqual(env.portfolio.active_contract_count, 0)
        self.assertEqual(
            env.portfolio.contracts[0].status,
            "matured",
        )

    def test_full_portfolio_episode_and_final_settlement(self):
        env = make_env(
            initial_budget=400_000_000,
            monthly_capital_injection=0,
            applicants_per_month=2,
            origination_months=2,
            loan_term_months=2,
            annual_interest_rate=0.0,
        )

        env.reset(seed=42)


        initial_budget = env.portfolio.available_budget

        # --------------------------------------------------
        # Month 1: two applicants receive 100M each.
        # --------------------------------------------------

        _, _, terminated, _, info = env.step(1)

        self.assertFalse(terminated)
        self.assertEqual(info["approved_amount"], 100_000_000)
        self.assertAlmostEqual(
            env.portfolio.available_budget,
            300_000_000,
        )

        _, _, terminated, _, info = env.step(1)

        self.assertFalse(terminated)
        self.assertEqual(env.portfolio.month, 2)
        self.assertEqual(env.phase, "allocation")

        # Month 1 loans have made their first instalments.
        self.assertAlmostEqual(
            env.portfolio.available_budget,
            300_000_000,
            places=4,
        )

        # --------------------------------------------------
        # Month 2: another two applicants receive 100M each.
        # --------------------------------------------------

        _, _, terminated, _, info = env.step(1)

        self.assertFalse(terminated)
        self.assertAlmostEqual(
            env.portfolio.available_budget,
            200_000_000,
        )

        _, _, terminated, _, info = env.step(1)

        self.assertFalse(terminated)
        self.assertEqual(env.portfolio.month, 3)
        self.assertEqual(env.phase, "settlement")

        # The first cohort has matured. The second cohort
        # still has its final instalment outstanding.
        self.assertEqual(
            env.portfolio.active_contract_count,
            2,
        )

        self.assertAlmostEqual(
            env.portfolio.available_budget,
            300_000_000,
            places=4,
        )

        # --------------------------------------------------
        # Settlement: service the remaining two contracts.
        # --------------------------------------------------

        _, _, terminated, _, info = env.step(5)

        self.assertTrue(terminated)
        self.assertTrue(info["action_ignored"])
        self.assertEqual(env.portfolio.month, 4)
        self.assertEqual(env.portfolio.active_contract_count, 0)

        self.assertEqual(
            len(env.portfolio.contracts),
            4,
        )

        self.assertAlmostEqual(
            env.portfolio.total_disbursed,
            400_000_000,
            places=4,
        )

        # With zero interest, zero default risk, and no
        # capital injections, the original cash is fully restored.
        self.assertAlmostEqual(
            env.portfolio.available_budget,
            initial_budget,
            places=4,
        )

        self.assertAlmostEqual(
            env.portfolio.cumulative_expected_defaults,
            0.0,
            places=8,
        )

        self.assertAlmostEqual(
            env.portfolio.cumulative_net_default_loss,
            0.0,
            places=4,
        )
        
    
    def test_random_monthly_cohort_sizes_are_seeded_and_within_bounds(self):
        def collect_sizes(seed):
            env = make_env(
                applicants_per_month=4,
                min_applicants_per_month=2,
                max_applicants_per_month=5,
                origination_months=6,
                loan_term_months=2,
            )
            initial_obs, _ = env.reset(seed=seed)
            self.assertEqual(
                initial_obs.shape,
                (CustomerState.FEATURE_COUNT + env.PORTFOLIO_FEATURE_COUNT,),
            )
            self.assertEqual(env.PORTFOLIO_FEATURE_COUNT, 16)
            self.assertAlmostEqual(
                float(initial_obs[CustomerState.FEATURE_COUNT + 3]),
                len(env.current_cohort_indices) / env.max_applicants_per_month,
            )
            sizes = []

            while env.phase == "allocation":
                cohort_size = len(env.current_cohort_indices)
                sizes.append(cohort_size)
                for _ in range(cohort_size):
                    _, _, _, _, _ = env.step(0)

            self.assertEqual(env.total_applicants_seen, sum(sizes))
            env.close()
            return sizes

        first_run = collect_sizes(seed=123)
        repeated_run = collect_sizes(seed=123)

        self.assertEqual(first_run, repeated_run)
        self.assertEqual(len(first_run), 6)
        self.assertTrue(all(2 <= size <= 5 for size in first_run))
        self.assertGreater(len(set(first_run)), 1)

    def test_gymnasium_checker_accepts_random_cohort_environment(self):
        env = make_env(
            applicants_per_month=4,
            min_applicants_per_month=2,
            max_applicants_per_month=5,
            origination_months=2,
            loan_term_months=2,
        )

        check_env(env, skip_render_check=True)
        env.close()

    def test_invalid_monthly_cohort_range_is_rejected(self):
        with self.assertRaises(ValueError):
            make_env(
                applicants_per_month=4,
                min_applicants_per_month=6,
                max_applicants_per_month=5,
            )

    def test_annual_nominal_rate_is_converted_to_monthly_rate(self):
        env = make_env(
            initial_budget=100_000_000,
            applicants_per_month=1,
            origination_months=1,
            loan_term_months=18,
            annual_interest_rate=0.04,
        )
        env.reset(seed=42)

        _, _, terminated, _, info = env.step(1)

        self.assertFalse(terminated)
        self.assertAlmostEqual(env.annual_interest_rate, 0.04)
        self.assertAlmostEqual(env.monthly_interest_rate, 0.04 / 12.0)
        self.assertAlmostEqual(
            env.portfolio.contracts[0].monthly_interest_rate,
            0.04 / 12.0,
        )
        self.assertGreater(info["interest_collected"], 0.0)

    def test_one_year_origination_then_settlement_to_longest_maturity(self):
        env = make_env(
            initial_budget=1_200_000_000,
            applicants_per_month=1,
            origination_months=12,
            loan_term_months=18,
            annual_interest_rate=0.0,
        )
        env.reset(seed=42)
        initial_cash = env.portfolio.available_budget

        # One new 100M loan in each of the 12 origination months.
        for _ in range(12):
            _, _, terminated, _, info = env.step(1)
            self.assertFalse(terminated)

        self.assertEqual(len(env.portfolio.contracts), 12)
        self.assertEqual(env.phase, "settlement")
        self.assertEqual(env.portfolio.month, 13)
        self.assertEqual(env.portfolio.contracts[0].remaining_months, 6)
        self.assertEqual(env.portfolio.contracts[-1].remaining_months, 17)

        # No new loans are issued in settlement. The last originated
        # contract is followed until its 18th payment is processed.
        for _ in range(17):
            _, _, terminated, _, info = env.step(0)

        self.assertTrue(terminated)
        self.assertEqual(env.portfolio.month, 30)
        self.assertEqual(env.portfolio.active_contract_count, 0)
        self.assertTrue(
            all(contract.status == "matured" for contract in env.portfolio.contracts)
        )
        self.assertAlmostEqual(
            env.portfolio.available_budget,
            initial_cash,
            places=4,
        )

    def test_random_cohorts_and_delay_schedule_share_one_observation(self):
        env = make_env(
            applicants_per_month=4,
            min_applicants_per_month=2,
            max_applicants_per_month=5,
            origination_months=2,
            loan_term_months=2,
            payment_delay_probability=0.1,
        )
        observation, _ = env.reset(seed=123)

        self.assertEqual(env.PORTFOLIO_FEATURE_COUNT, 28)
        self.assertEqual(
            observation.shape,
            (CustomerState.FEATURE_COUNT + env.PORTFOLIO_FEATURE_COUNT,),
        )
        self.assertTrue(env.observation_space.contains(observation))
        check_env(env, skip_render_check=True)

        # The last 12 entries of the portfolio feature block describe
        # expected receipts due 1..12 months from now; a new episode is empty.
        delay_start = CustomerState.FEATURE_COUNT + 12
        delay_schedule = observation[delay_start:delay_start + 12]
        self.assertEqual(len(delay_schedule), 12)
        np.testing.assert_allclose(delay_schedule, np.zeros(12), atol=1e-7)
        env.close()

    def test_integrated_interest_and_delayed_payment_accounting(self):
        env = make_env(
            initial_budget=100_000_000,
            monthly_capital_injection=0,
            applicants_per_month=1,
            origination_months=1,
            loan_term_months=2,
            annual_interest_rate=0.12,
            payment_delay_probability=1.0,
            delay_lag_weights=(1.0, 0.0, 0.0, 0.0),
        )

        observation, _ = env.reset(seed=42)
        self.assertEqual(
            observation.shape,
            (CustomerState.FEATURE_COUNT + env.PORTFOLIO_FEATURE_COUNT,),
        )
        self.assertEqual(env.PORTFOLIO_FEATURE_COUNT, 27)
        self.assertTrue(env.observation_space.contains(observation))
        initial_cash = env.portfolio.available_budget

        # Grant a loan. At the end of allocation month 1, the installment's
        # risk-scaled late share is queued for month 3.
        observation_1, reward_1, terminated, _, info_1 = env.step(1)

        self.assertFalse(terminated)
        self.assertTrue(
            env.observation_space.contains(observation_1)
        )
        delay_start = CustomerState.FEATURE_COUNT + 11
        delayed_schedule = observation_1[delay_start:delay_start + 12]
        self.assertEqual(len(delayed_schedule), 12)
        self.assertGreater(float(np.sum(delayed_schedule)), 0.0)
        self.assertAlmostEqual(
            float(np.sum(delayed_schedule)),
            info_1["pending_delayed_receivables"] / env.initial_budget,
            places=6,
        )
        self.assertTrue(np.isfinite(reward_1))
        self.assertEqual(env.phase, "settlement")
        self.assertGreater(info_1["interest_collected"], 0.0)
        self.assertGreater(info_1["delayed_payment_amount_scheduled"], 0.0)
        self.assertGreater(info_1["pending_delayed_receivables"], 0.0)
        self.assertAlmostEqual(info_1["net_default_loss"], 0.0)

        # Month 3 receives the first late installment and queues the final one.
        _, reward_2, terminated, _, info_2 = env.step(0)
        self.assertFalse(terminated)
        self.assertTrue(info_2["action_ignored"])
        self.assertTrue(np.isfinite(reward_2))
        self.assertGreater(info_2["delayed_payment_amount_collected"], 0.0)
        self.assertGreater(info_2["pending_delayed_receivables"], 0.0)

        # Month 4 receives the final installment; only now is the portfolio settled.
        _, reward_3, terminated, _, info_3 = env.step(0)
        self.assertTrue(terminated)
        self.assertTrue(info_3["action_ignored"])
        self.assertTrue(np.isfinite(reward_3))
        self.assertAlmostEqual(env.portfolio.cumulative_net_default_loss, 0.0)
        self.assertAlmostEqual(
            env.portfolio.cumulative_delayed_payment_amount_scheduled,
            env.portfolio.cumulative_delayed_payment_amount_collected,
        )
        self.assertAlmostEqual(
            env.portfolio.available_budget,
            initial_cash + env.portfolio.cumulative_interest_collected,
            delta=0.01,
        )
        env.close()

    def test_gymnasium_environment_checker(self):
        env = make_env(
            applicants_per_month=2,
            origination_months=2,
            loan_term_months=2,
        )

        # Checks the Gymnasium API and observation/action-space contract.
        check_env(env, skip_render_check=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
