
import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from gymnasium.utils.env_checker import check_env

from customer_state import CustomerState
from portfolio_loan_env import PortfolioLoanEnv
from customer_simulator import CustomerSimulator
from portfolio_experiment_utils import (
    calibrate_hazard_intercept_for_population,
    normalize_customer_financial_features,
    reference_risk_scores_after_loan,
)


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
        "max_hazard": 0.0,
    }

    config.update(overrides)

    return PortfolioLoanEnv(
        raw_df=df,
        scaler=scaler,
        **config,
    )


class TestPortfolioLoanEnv(unittest.TestCase):

    def test_population_calibration_matches_target_for_reference_loan(self):
        df, _ = make_dataset(n=101)
        calibration = calibrate_hazard_intercept_for_population(
            raw_df=df,
            target_annual_default_rate=0.013,
            reference_loan_amount=100_000_000,
            calibration_horizon_months=12,
            max_hazard=0.005,
            hazard_slope=0.08,
            hazard_center=60.0,
        )

        self.assertAlmostEqual(
            calibration["calibrated_reference_annual_default_rate"],
            0.013,
            places=12,
        )
        self.assertEqual(calibration["reference_population_rows"], len(df))
        self.assertEqual(calibration["calibration_horizon_months"], 12)


    def test_financial_feature_normalization_matches_runtime_customer_state(self):
        raw = pd.DataFrame({
            "DebtAmount": [-10.0, 30.0, 80.0],
            "OriginAmount": [0.0, 60.0, -5.0],
            "LoanCounts": [0, 2.9, -1],
            "DebtRatio": [99.0, -99.0, 77.0],
            "DebtPerLoan": [99.0, -99.0, 77.0],
            "TotalFacilities": [-1.0, 20.0, 30.0],
        })
        normalized = normalize_customer_financial_features(raw)

        np.testing.assert_allclose(
            normalized["DebtAmount"].to_numpy(),
            [0.0, 30.0, 80.0],
        )
        np.testing.assert_allclose(
            normalized["OriginAmount"].to_numpy(),
            [0.0, 60.0, 0.0],
        )
        np.testing.assert_array_equal(
            normalized["LoanCounts"].to_numpy(),
            [0, 2, 0],
        )
        np.testing.assert_allclose(
            normalized["DebtRatio"].to_numpy(),
            [0.0, 0.5, 0.0],
        )
        np.testing.assert_allclose(
            normalized["DebtPerLoan"].to_numpy(),
            [0.0, 15.0, 0.0],
        )
        np.testing.assert_allclose(
            normalized["TotalFacilities"].to_numpy(),
            [0.0, 20.0, 30.0],
        )

    def test_reference_risk_calibration_matches_runtime_risk_update(self):
        df, _ = make_dataset(n=31)
        # Include a first-time customer and rows whose raw derived ratios are
        # deliberately inconsistent with their underlying financial fields.
        df.loc[0, ["DebtAmount", "OriginAmount", "LoanCounts"]] = [0.0, 0.0, 0]
        df.loc[0, ["DebtRatio", "DebtPerLoan"]] = [0.9, 123.0]
        df.loc[1, ["DebtRatio", "DebtPerLoan"]] = [0.01, 1.0]

        amount = 100_000_000.0
        vectorized = reference_risk_scores_after_loan(
            df,
            reference_loan_amount=amount,
        )
        simulator = CustomerSimulator()
        runtime = []

        for _, row in df.iterrows():
            debt = max(0.0, float(row["DebtAmount"]))
            origin = max(0.0, float(row["OriginAmount"]))
            count = max(0, int(row["LoanCounts"]))
            customer = CustomerState(
                monthly_income=float(row["MountlyIncome"]),
                income_score=float(row["IncomeScore"]),
                job_score=float(row["JobScore"]),
                asset_score=float(row["AssetScore"]),
                total_customer_score=float(row["TotalCustomerScore"]),
                total_facilities=max(0.0, float(row["TotalFacilities"])),
                debt=debt,
                origin_amount=origin,
                loan_count=count,
                debt_ratio=debt / origin if origin > 0.0 else 0.0,
                debt_per_loan=debt / count if count > 0 else 0.0,
                has_no_history=bool(row["Has_No_History"]),
                employed=bool(row["Employed"]),
                score_min=float(row["Score_Min"]),
                score_max=float(row["Score_Max"]),
                risk_min=float(row["Risk_Min"]),
                risk_max=float(row["Risk_Max"]),
                risk=(float(row["Risk_Min"]) + float(row["Risk_Max"])) / 2.0,
            )
            updated = simulator._update_customer_state(customer, amount)
            updated_risk = simulator._update_risk(
                updated,
                customer,
                amount,
            )
            runtime.append(updated_risk)

        np.testing.assert_allclose(
            vectorized,
            np.asarray(runtime),
            rtol=0.0,
            atol=1e-12,
        )

    def test_environment_model_version_tracks_observation_and_reward_semantics(self):
        self.assertEqual(PortfolioLoanEnv.ENVIRONMENT_MODEL_VERSION, 12)
        self.assertEqual(len(PortfolioLoanEnv.DYNAMIC_CUSTOMER_FIELDS), 7)

    def test_portfolio_reward_uses_fixed_amount_sensitive_engine(self):
        import inspect
        from reward_engine import RewardEngine

        env = make_env()
        try:
            self.assertIsInstance(env.reward_engine, RewardEngine)
            self.assertNotIn(
                "scale_access_rewards_by_coverage",
                inspect.signature(RewardEngine.__init__).parameters,
            )
            customer = SimpleNamespace(
                total_customer_score=0.0,
                total_facilities=0.0,
                has_no_history=True,
            )
            self.assertAlmostEqual(
                env.reward_engine._social_reward(customer, 100_000_000),
                0.6,
            )
        finally:
            env.close()

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

    def test_observation_refreshes_dynamic_customer_features_from_portfolio(self):
        env = make_env(
            initial_budget=500_000_000,
            applicants_per_month=2,
            origination_months=1,
            loan_term_months=2,
            max_hazard=0.0,
        )
        env.reset(seed=42)

        customer_id = env.current_row_index
        baseline_customer = env.current_customer.copy()

        # Originate a loan directly in the portfolio while the environment's
        # local customer copy is still the pre-origination snapshot.
        env.portfolio.grant_loan(
            customer=baseline_customer,
            approved_amount=100_000_000,
            customer_id=customer_id,
        )
        latest_customer = env.portfolio.get_customer_state(customer_id)

        self.assertGreater(
            latest_customer.origin_amount,
            baseline_customer.origin_amount,
        )
        self.assertEqual(
            latest_customer.loan_count,
            baseline_customer.loan_count + 1,
        )

        expected_customer = baseline_customer.copy()
        for field_name in env.DYNAMIC_CUSTOMER_FIELDS:
            setattr(
                expected_customer,
                field_name,
                getattr(latest_customer, field_name),
            )

        expected_values = expected_customer.to_array()
        if hasattr(env.scaler, "feature_names_in_"):
            expected_input = pd.DataFrame(
                [expected_values],
                columns=env.scaler.feature_names_in_,
            )
        else:
            expected_input = expected_values.reshape(1, -1)
        expected_scaled = env.scaler.transform(expected_input)[0]

        observation = env._get_observation()
        np.testing.assert_allclose(
            observation[:CustomerState.FEATURE_COUNT],
            expected_scaled,
            rtol=0.0,
            atol=1e-6,
        )

        # The 11 static attributes must remain unchanged.
        static_fields = (
            "monthly_income",
            "income_score",
            "job_score",
            "asset_score",
            "total_customer_score",
            "total_facilities",
            "employed",
            "score_min",
            "score_max",
            "risk_min",
            "risk_max",
        )
        for field_name in static_fields:
            self.assertEqual(
                getattr(env.current_customer, field_name),
                getattr(baseline_customer, field_name),
            )

        # The local customer copy is synchronized with all 7 dynamic fields.
        for field_name in env.DYNAMIC_CUSTOMER_FIELDS:
            self.assertEqual(
                getattr(env.current_customer, field_name),
                getattr(latest_customer, field_name),
            )

        env.close()

    def test_no_applicant_observation_is_neutral_during_transition_and_settlement(self):
        env = make_env(
            initial_budget=100_000_000,
            applicants_per_month=1,
            origination_months=1,
            loan_term_months=2,
            annual_interest_rate=0.0,
            max_hazard=0.0,
        )
        env.reset(seed=42)

        transition_observation, _, terminated, _, _ = env.step(1)
        self.assertFalse(terminated)
        self.assertEqual(env.phase, "transition")
        self.assertIsNone(env.current_customer)
        self.assertIsNone(env.current_row_index)

        np.testing.assert_array_equal(
            transition_observation[:CustomerState.FEATURE_COUNT],
            np.zeros(CustomerState.FEATURE_COUNT, dtype=np.float32),
        )
        self.assertEqual(
            transition_observation[CustomerState.FEATURE_COUNT + 2],
            0.0,
        )
        self.assertEqual(
            transition_observation[CustomerState.FEATURE_COUNT + 5],
            0.0,
        )
        # Fixed-cohort layout omits cohort_size_ratio, so the transition
        # flag occupies portfolio feature index 6.
        self.assertEqual(
            transition_observation[CustomerState.FEATURE_COUNT + 6],
            1.0,
        )
        self.assertTrue(env.observation_space.contains(transition_observation))

        settlement_observation, _, terminated, _, info = env.step(0)
        self.assertFalse(terminated)
        self.assertEqual(env.phase, "settlement")
        self.assertTrue(info["action_ignored"])
        np.testing.assert_array_equal(
            settlement_observation[:CustomerState.FEATURE_COUNT],
            np.zeros(CustomerState.FEATURE_COUNT, dtype=np.float32),
        )
        self.assertEqual(
            settlement_observation[CustomerState.FEATURE_COUNT + 2],
            0.0,
        )
        self.assertEqual(
            settlement_observation[CustomerState.FEATURE_COUNT + 5],
            1.0,
        )
        self.assertEqual(
            settlement_observation[CustomerState.FEATURE_COUNT + 6],
            0.0,
        )
        self.assertTrue(env.observation_space.contains(settlement_observation))

        final_observation, _, terminated, _, _ = env.step(0)
        self.assertTrue(terminated)
        np.testing.assert_array_equal(
            final_observation[:CustomerState.FEATURE_COUNT],
            np.zeros(CustomerState.FEATURE_COUNT, dtype=np.float32),
        )
        self.assertTrue(env.observation_space.contains(final_observation))
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

        # Allocation is now complete for month 1, but time has not advanced.
        self.assertEqual(env.portfolio.month, 1)
        self.assertEqual(env.phase, "transition")
        self.assertFalse(terminated)

        # The independent transition step services loans and then starts
        # settlement because this was the final origination month.
        _, _, terminated, _, transition_info = env.step(0)
        self.assertTrue(transition_info["action_ignored"])
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
        for expected_month in (1, 2, 3):
            _, _, terminated, _, allocation_info = env.step(0)
            self.assertFalse(terminated)
            self.assertEqual(env.phase, "transition")
            self.assertEqual(env.portfolio.month, expected_month)

            _, _, terminated, _, transition_info = env.step(0)
            self.assertEqual(transition_info["month"], expected_month + 1)

            if expected_month < 3:
                self.assertFalse(terminated)
                self.assertEqual(env.phase, "allocation")
            else:
                self.assertTrue(terminated)
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
        self.assertEqual(env.phase, "transition")
        self.assertEqual(env.portfolio.month, 1)

        # First advance the clock and apply the first installment.
        _, _, terminated, _, transition_info = env.step(0)
        self.assertTrue(transition_info["action_ignored"])
        self.assertEqual(env.phase, "settlement")
        self.assertEqual(env.portfolio.month, 2)
        self.assertGreater(env.portfolio.active_contract_count, 0)

        # During settlement the action is ignored, but the remaining
        # installment must still be serviced.
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
        self.assertEqual(env.portfolio.month, 1)
        self.assertEqual(env.phase, "transition")

        # Monthly transition now collects the first installment separately.
        _, _, terminated, _, info = env.step(0)
        self.assertFalse(terminated)
        self.assertEqual(env.portfolio.month, 2)
        self.assertEqual(env.phase, "allocation")
        self.assertAlmostEqual(
            env.portfolio.available_budget,
            200_000_000,
            places=4,
        )

        # --------------------------------------------------
        # Month 2: another two applicants receive 100M each.
        # --------------------------------------------------

        _, _, terminated, _, info = env.step(1)

        self.assertFalse(terminated)
        self.assertAlmostEqual(
            env.portfolio.available_budget,
            100_000_000,
        )

        _, _, terminated, _, info = env.step(1)

        self.assertFalse(terminated)
        self.assertEqual(env.portfolio.month, 2)
        self.assertEqual(env.phase, "transition")

        _, _, terminated, _, info = env.step(0)
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
            200_000_000,
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
            self.assertEqual(env.PORTFOLIO_FEATURE_COUNT, 31)
            self.assertAlmostEqual(
                float(initial_obs[CustomerState.FEATURE_COUNT + 3]),
                len(env.current_cohort_indices) / env.max_applicants_per_month,
            )
            sizes = []

            while not env.episode_done:
                if env.phase == "allocation":
                    cohort_size = len(env.current_cohort_indices)
                    sizes.append(cohort_size)
                    for _ in range(cohort_size):
                        _, _, _, _, _ = env.step(0)
                    self.assertEqual(env.phase, "transition")

                elif env.phase == "transition":
                    _, _, terminated, _, _ = env.step(0)
                    if terminated:
                        break

                elif env.phase == "settlement":
                    _, _, terminated, _, _ = env.step(0)
                    if terminated:
                        break

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

    def test_annual_interest_is_charged_at_annual_due_months(self):
        env = make_env(
            initial_budget=100_000_000,
            applicants_per_month=1,
            origination_months=1,
            loan_term_months=18,
            annual_interest_rate=0.04,
        )
        env.reset(seed=42)

        _, _, terminated, _, _ = env.step(1)
        self.assertFalse(terminated)
        self.assertEqual(env.phase, "transition")

        # The first scheduled payment is due in the next calendar month:
        # annual interest is charged, while principal remains unchanged.
        _, _, terminated, _, first_payment = env.step(0)
        self.assertFalse(terminated)
        self.assertAlmostEqual(env.annual_interest_rate, 0.04)
        self.assertAlmostEqual(
            env.portfolio.contracts[0].annual_interest_rate,
            0.04,
        )
        self.assertAlmostEqual(first_payment["principal_collected"], 0.0)
        self.assertAlmostEqual(first_payment["interest_collected"], 4_000_000.0)
        self.assertAlmostEqual(
            env.portfolio.contracts[0].outstanding_principal,
            100_000_000.0,
        )

        # Principal is split across months 2..12 and 14..18.
        principal_installment = 100_000_000.0 / 16.0
        for _ in range(11):
            result = env.portfolio.advance_month()
            self.assertAlmostEqual(
                result.principal_collected,
                principal_installment,
                delta=0.02,
            )
            self.assertAlmostEqual(result.interest_collected, 0.0, delta=0.02)

        second_year_interest = env.portfolio.advance_month()
        self.assertAlmostEqual(second_year_interest.principal_collected, 0.0)
        self.assertAlmostEqual(
            second_year_interest.interest_collected,
            1_250_000.0,
            delta=0.02,
        )
        env.close()

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
        for month in range(1, 13):
            _, _, terminated, _, info = env.step(1)
            self.assertFalse(terminated)
            self.assertEqual(env.phase, "transition")
            self.assertEqual(env.portfolio.month, month)

            _, _, terminated, _, info = env.step(0)
            if month < 12:
                self.assertFalse(terminated)
                self.assertEqual(env.phase, "allocation")

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
            max_hazard=0.0,
        )
        observation, _ = env.reset(seed=123)

        self.assertEqual(env.PORTFOLIO_FEATURE_COUNT, 31)
        self.assertEqual(
            observation.shape,
            (CustomerState.FEATURE_COUNT + env.PORTFOLIO_FEATURE_COUNT,),
        )
        self.assertTrue(env.observation_space.contains(observation))
        check_env(env, skip_render_check=True)

        # The current portfolio summary has 13 non-delay entries in variable
        # cohort mode (including the transition flag), followed by 12 delay
        # forecasts and the 4 applicant-queue summary features.
        delay_start = CustomerState.FEATURE_COUNT + 15
        delay_schedule = observation[delay_start:delay_start + 12]
        self.assertEqual(len(delay_schedule), 12)
        np.testing.assert_allclose(delay_schedule, np.zeros(12), atol=1e-7)
        env.close()

    def test_integrated_interest_and_full_missed_installment_accounting(self):
        env = make_env(
            initial_budget=100_000_000,
            monthly_capital_injection=0,
            applicants_per_month=1,
            origination_months=1,
            loan_term_months=2,
            annual_interest_rate=0.12,
            max_hazard=1_000.0,
            default_collection_lag_months=4,
        )

        observation, _ = env.reset(seed=42)
        self.assertEqual(
            observation.shape,
            (CustomerState.FEATURE_COUNT + env.PORTFOLIO_FEATURE_COUNT,),
        )
        self.assertEqual(env.PORTFOLIO_FEATURE_COUNT, 30)
        self.assertTrue(env.observation_space.contains(observation))
        initial_cash = env.portfolio.available_budget

        # Grant one loan. Payment processing occurs only on the following
        # transition step, not as part of the applicant's action.
        observation_1, reward_1, terminated, _, info_1 = env.step(1)
        self.assertFalse(terminated)
        self.assertTrue(env.observation_space.contains(observation_1))
        self.assertEqual(env.phase, "transition")
        self.assertEqual(env.portfolio.month, 1)
        self.assertTrue(np.isfinite(reward_1))

        # Month 2: the installment is forced to be fully missed. Since this is
        # the first miss, no recognized-default penalty has been applied yet.
        observation_2, reward_2, terminated, _, info_2 = env.step(0)
        self.assertFalse(terminated)
        self.assertTrue(env.observation_space.contains(observation_2))
        self.assertEqual(env.phase, "settlement")
        self.assertEqual(env.portfolio.month, 2)
        self.assertEqual(info_2["default_penalty"], 0.0)
        self.assertAlmostEqual(info_2["principal_collected"], 0.0)
        self.assertAlmostEqual(info_2["interest_collected"], 0.0)
        self.assertEqual(info_2["missed_installment_count"], 1.0)
        self.assertGreater(info_2["delayed_payment_amount_scheduled"], 0.0)
        self.assertGreater(info_2["pending_delayed_receivables"], 0.0)
        self.assertAlmostEqual(info_2["net_default_loss"], 0.0)
        self.assertTrue(np.isfinite(reward_2))

        # Month 3: remaining installment is paid on time, but the overdue
        # installment remains outstanding and is not fractionally collected.
        _, _, terminated, _, info_3 = env.step(0)
        self.assertFalse(terminated)
        self.assertEqual(env.portfolio.month, 3)
        self.assertTrue(info_3["action_ignored"])
        self.assertEqual(info_3["delayed_payment_amount_scheduled"], 0.0)
        self.assertAlmostEqual(info_3["delayed_payment_amount_collected"], 0.0)
        self.assertGreater(info_3["pending_delayed_receivables"], 0.0)

        # Months 4 and 5 continue settlement; month 6 recognizes >90 DPD.
        for month in (4, 5):
            _, _, terminated, _, info = env.step(0)
            self.assertFalse(terminated)
            self.assertEqual(env.portfolio.month, month)
        _, reward_6, terminated, _, info_6 = env.step(0)
        self.assertTrue(terminated)
        self.assertEqual(env.portfolio.month, 6)
        self.assertEqual(info_6["expected_default_count"], 1.0)
        self.assertEqual(info_6["recognized_default_count"], 1.0)
        self.assertEqual(info_6["default_penalty"], 10.0)
        self.assertAlmostEqual(
            reward_6,
            info_6["financial_pnl"] / env.financial_reward_scale - 10.0,
        )
        self.assertEqual(info_6["missed_installment_count"], 0.0)
        self.assertGreater(info_6["delayed_payment_amount_collected"], 0.0)
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

    def test_repeated_applicant_uses_persistent_customer_state(self):
        env = make_env(
            initial_budget=500_000_000,
            applicants_per_month=2,
            origination_months=1,
            loan_term_months=2,
            max_hazard=0.0,
        )
        env.reset(seed=42)
        customer_id = env.current_row_index
        original_debt = env.current_customer.debt

        # Simulate the same empirical customer obtaining a loan, then appearing
        # again in the monthly cohort. Loading the source row must not reset debt.
        env.portfolio.grant_loan(
            customer=env.current_customer,
            approved_amount=100_000_000,
            customer_id=customer_id,
        )
        env.current_cohort_indices = np.array([customer_id, customer_id], dtype=int)
        env.current_applicant_position = 0
        env._load_current_applicant()

        self.assertAlmostEqual(
            env.current_customer.debt,
            original_debt + 100_000_000,
        )
        self.assertEqual(
            env.current_customer.loan_count,
            max(0, int(env.raw_df.iloc[customer_id]["LoanCounts"])) + 1,
        )
        env.close()

    def test_episode_waits_for_twelve_month_delayed_receipt(self):
        env = make_env(
            initial_budget=100_000_000,
            applicants_per_month=1,
            origination_months=1,
            loan_term_months=1,
            annual_interest_rate=0.0,
            max_hazard=1_000.0,
            default_collection_lag_months=12,
        )
        env.reset(seed=42)
        env.current_customer.risk = 100.0

        _, _, terminated, _, info = env.step(1)

        self.assertFalse(terminated)
        self.assertEqual(env.phase, "transition")
        self.assertEqual(env.portfolio.month, 1)

        # First transition triggers the missed installment, then begins settlement.
        _, _, terminated, _, info = env.step(0)
        self.assertFalse(terminated)
        self.assertEqual(env.phase, "settlement")
        self.assertEqual(env.portfolio.month, 2)
        self.assertEqual(env.portfolio.active_contract_count, 0)
        self.assertGreater(info["pending_delayed_receivables"], 0.0)

        for expected_month in range(3, 15):
            _, _, terminated, _, info = env.step(0)
            self.assertEqual(env.portfolio.month, expected_month)
            if expected_month < 14:
                self.assertFalse(terminated)

        self.assertTrue(terminated)
        self.assertTrue(env.portfolio.is_fully_settled)
        self.assertAlmostEqual(env.portfolio.pending_delayed_receivables, 0.0)
        self.assertAlmostEqual(env.portfolio.available_budget, 100_000_000)
        env.close()

    def test_first_time_applicants_are_unique_across_episode(self):
        env = make_env(
            applicants_per_month=4,
            origination_months=3,
            loan_term_months=2,
            reapplication_share=0.0,
            reapplication_cooldown_months=1,
            max_hazard=0.0,
        )
        env.reset(seed=71)
        episode_customer_indices = set()

        for month_index in range(3):
            cohort = [int(idx) for idx in env.current_cohort_indices]
            self.assertEqual(len(cohort), len(set(cohort)))
            self.assertTrue(
                episode_customer_indices.isdisjoint(cohort),
                "A first-time applicant must not be sampled again when reapplications are disabled.",
            )
            episode_customer_indices.update(cohort)

            for _ in cohort:
                _, _, terminated, _, _ = env.step(0)
                self.assertFalse(terminated)

            if month_index < 2:
                self.assertEqual(env.phase, "transition")
                _, _, terminated, _, _ = env.step(0)
                self.assertFalse(terminated)
                self.assertEqual(env.phase, "allocation")

        self.assertEqual(env.new_applicants_seen, 12)
        self.assertEqual(env.reapplications_seen, 0)
        env.close()

    def test_new_to_credit_status_clears_after_first_simulated_loan(self):
        env = make_env(
            initial_budget=300_000_000,
            applicants_per_month=1,
            origination_months=2,
            loan_term_months=4,
            max_hazard=0.0,
        )
        env.reset(seed=75)
        customer_id = env.current_row_index

        # Make the selected source row explicitly new-to-credit for this test.
        baseline = env.portfolio.customer_baselines[customer_id].copy()
        baseline.has_no_history = True
        env.portfolio.customer_baselines[customer_id] = baseline.copy()
        env.portfolio.customer_states[customer_id] = baseline.copy()
        env.current_customer = baseline.copy()

        self.assertEqual(
            env.reward_engine._history_reward(baseline, coverage=1.0),
            env.reward_engine.history_reward,
        )

        env.portfolio.grant_loan(
            customer=baseline,
            approved_amount=100_000_000,
            customer_id=customer_id,
        )
        updated = env.portfolio.get_customer_state(customer_id)
        self.assertFalse(updated.has_no_history)
        self.assertEqual(
            env.reward_engine._history_reward(updated, coverage=1.0),
            0.0,
        )
        env.close()

    def test_reapplication_respects_configured_three_month_cooldown(self):
        env = make_env(
            applicants_per_month=2,
            origination_months=4,
            loan_term_months=2,
            reapplication_share=1.0,
            reapplication_cooldown_months=3,
            max_hazard=0.0,
        )
        env.reset(seed=74)
        month_one = set(int(idx) for idx in env.current_cohort_indices)

        def advance_to_next_allocation_month():
            cohort_size = len(env.current_cohort_indices)
            for _ in range(cohort_size):
                env.step(0)
            self.assertEqual(env.phase, "transition")
            env.step(0)
            self.assertEqual(env.phase, "allocation")

        advance_to_next_allocation_month()
        month_two = set(int(idx) for idx in env.current_cohort_indices)
        self.assertTrue(month_one.isdisjoint(month_two))

        advance_to_next_allocation_month()
        month_three = set(int(idx) for idx in env.current_cohort_indices)
        self.assertTrue(month_one.isdisjoint(month_three))

        # Month 4 is three checkpoints after the original applications.
        advance_to_next_allocation_month()
        month_four = set(int(idx) for idx in env.current_cohort_indices)
        self.assertEqual(month_four, month_one)
        self.assertTrue(env.current_is_reapplication)
        env.close()

    def test_eligible_customer_can_reapply_while_loan_is_active(self):
        env = make_env(
            initial_budget=500_000_000,
            applicants_per_month=2,
            origination_months=2,
            loan_term_months=4,
            reapplication_share=1.0,
            reapplication_cooldown_months=1,
            max_hazard=0.0,
        )
        env.reset(seed=72)
        first_cohort = set(int(idx) for idx in env.current_cohort_indices)

        # Two customers receive loans in month 1.
        env.step(1)
        env.step(1)
        self.assertEqual(env.portfolio.active_contract_count, 2)
        self.assertEqual(env.phase, "transition")

        # At month 2, the cooldown has elapsed and no overdue installment
        # exists, so the same two customers may reapply despite active loans.
        env.step(0)
        second_cohort = set(int(idx) for idx in env.current_cohort_indices)
        self.assertEqual(second_cohort, first_cohort)
        self.assertEqual(env.portfolio.active_contract_count, 2)
        self.assertTrue(env.current_is_reapplication)
        self.assertEqual(env.reapplications_seen, 1)

        # The policy observes both that the applicant is returning and their
        # previous application count (fixed-cohort features start after the
        # customer feature vector).
        observation = env._get_observation()
        portfolio_offset = CustomerState.FEATURE_COUNT
        self.assertAlmostEqual(observation[portfolio_offset + 3], 1.0)
        self.assertGreater(observation[portfolio_offset + 4], 0.0)
        env.close()

    def test_unpaid_missed_installment_blocks_reapplication(self):
        env = make_env(
            initial_budget=500_000_000,
            applicants_per_month=2,
            origination_months=2,
            loan_term_months=4,
            annual_interest_rate=0.04,
            reapplication_share=1.0,
            reapplication_cooldown_months=1,
            max_hazard=1_000.0,
        )
        env.reset(seed=73)
        first_cohort = set(int(idx) for idx in env.current_cohort_indices)

        # A very high hazard deterministically forces a missed first payment
        # at the next monthly checkpoint. The resulting overdue blocks return.
        env.step(1)
        env.step(1)
        env.step(0)  # monthly transition and repayment/default-event simulation

        second_cohort = set(int(idx) for idx in env.current_cohort_indices)
        self.assertTrue(first_cohort.isdisjoint(second_cohort))
        self.assertEqual(env.reapplications_seen, 0)
        self.assertGreaterEqual(env.new_applicants_seen, 3)
        self.assertTrue(
            any(
                env.portfolio.has_unresolved_overdue(customer_id)
                for customer_id in first_cohort
            )
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
