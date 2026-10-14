"""Unit tests for baseline policy evaluation metrics."""

import unittest

import pandas as pd

from evaluate_baseline_policies import METRICS, summarize


class TestBaselinePolicyMetrics(unittest.TestCase):
    @staticmethod
    def episode(policy, rate, defaults, loans):
        record = {metric: 0.0 for metric in METRICS}
        record.update({
            "policy": policy,
            "realized_default_rate": rate,
            "recognized_default_count": defaults,
            "loans_granted": loans,
        })
        return record

    def test_pooled_rate_is_based_on_totals_not_mean_of_episode_rates(self):
        episodes = pd.DataFrame([
            self.episode("test_policy", rate=1.0, defaults=1, loans=1),
            self.episode("test_policy", rate=0.0, defaults=0, loans=9),
        ])

        row = summarize(episodes).iloc[0]

        self.assertAlmostEqual(row["mean_episode_realized_default_rate"], 0.5)
        self.assertAlmostEqual(row["pooled_realized_default_rate"], 0.1)
        # Existing summary column is retained for backwards compatibility.
        self.assertAlmostEqual(row["mean_realized_default_rate"], 0.5)

    def test_pooled_rate_is_zero_when_no_loans_are_granted(self):
        episodes = pd.DataFrame([
            self.episode("no_loan", rate=0.0, defaults=0, loans=0),
        ])

        row = summarize(episodes).iloc[0]

        self.assertEqual(row["mean_episode_realized_default_rate"], 0.0)
        self.assertEqual(row["pooled_realized_default_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
