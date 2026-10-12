"""Unit tests for legacy and amount-sensitive RewardEngine variants."""

import unittest
from types import SimpleNamespace

from reward_engine import RewardEngine


def make_customer(score=0.0, total_facilities=0.0, has_no_history=True):
    return SimpleNamespace(
        total_customer_score=score,
        total_facilities=total_facilities,
        has_no_history=has_no_history,
    )


class TestRewardEngine(unittest.TestCase):

    def test_legacy_mode_preserves_amount_independent_access_rewards(self):
        engine = RewardEngine()
        customer = make_customer(score=0, total_facilities=0, has_no_history=True)
        self.assertAlmostEqual(
            engine._social_reward(customer, 100_000_000), 2.2
        )
        self.assertAlmostEqual(
            engine._social_reward(customer, 500_000_000), 3.0
        )

    def test_experimental_mode_scales_both_access_rewards_with_coverage(self):
        engine = RewardEngine(scale_access_rewards_by_coverage=True)
        customer = make_customer(score=0, total_facilities=0, has_no_history=True)
        reward_100m = engine._social_reward(customer, 100_000_000)
        reward_200m = engine._social_reward(customer, 200_000_000)
        reward_500m = engine._social_reward(customer, 500_000_000)

        self.assertAlmostEqual(reward_100m, 0.6)
        self.assertAlmostEqual(reward_200m, 1.2)
        self.assertAlmostEqual(reward_500m, 3.0)
        self.assertLess(reward_100m, reward_200m)
        self.assertLess(reward_200m, reward_500m)

    def test_zero_amount_receives_zero_social_reward(self):
        for experimental in (False, True):
            engine = RewardEngine(
                scale_access_rewards_by_coverage=experimental
            )
            self.assertEqual(engine._social_reward(make_customer(), 0), 0.0)

    def test_facility_reward_uses_credit_eligibility_limit(self):
        engine = RewardEngine()
        low_limit = make_customer(total_facilities=0)
        high_limit = make_customer(total_facilities=500_000_000)
        self.assertAlmostEqual(engine._facility_reward(low_limit), 1.0)
        self.assertAlmostEqual(
            engine._facility_reward(high_limit), 1.0 / 1.5
        )
        self.assertGreater(
            engine._facility_reward(low_limit),
            engine._facility_reward(high_limit),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
