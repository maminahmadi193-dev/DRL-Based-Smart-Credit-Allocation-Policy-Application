"""Unit tests for env.exact_reference_mdp."""
import unittest
import numpy as np

from exact_reference_mdp import FiniteCreditReferenceMDP


class TestFiniteCreditReferenceMDP(unittest.TestCase):
    def setUp(self):
        self.mdp = FiniteCreditReferenceMDP(max_hazard=0.20)

    def test_reachable_state_set_is_finite_and_explicit(self):
        self.assertEqual(len(self.mdp.states), 7)
        self.assertEqual(self.mdp.states[0], self.mdp.start_state)
        self.assertEqual(set(self.mdp.states[1:]), {(1, cash) for cash in range(6)})

    def test_transition_probabilities_sum_to_one(self):
        for state in self.mdp.states:
            for action in range(self.mdp.action_count):
                outcomes = self.mdp.transition_outcomes(state, action)
                self.assertAlmostEqual(sum(outcome[0] for outcome in outcomes), 1.0, places=12)
                self.assertTrue(all(outcome[0] >= 0.0 for outcome in outcomes))

    def test_exact_dp_satisfies_bellman_optimality(self):
        q_star, values, policy = self.mdp.exact_q_star()
        for state in self.mdp.states:
            self.assertAlmostEqual(values[state], np.max(q_star[state]), places=10)
            self.assertEqual(policy[state], int(np.argmax(q_star[state])))
            for action in range(self.mdp.action_count):
                expected = sum(
                    p * (r + self.mdp.GAMMA * values.get(next_state, 0.0))
                    for p, next_state, r in self.mdp.transition_outcomes(state, action)
                )
                self.assertAlmostEqual(q_star[state][action], expected, places=10)

    def test_unaffordable_action_is_rejected_and_penalized(self):
        outcomes = self.mdp.transition_outcomes((1, 1), 5)
        self.assertEqual(len(outcomes), 1)
        self.assertAlmostEqual(outcomes[0][0], 1.0)
        self.assertEqual(outcomes[0][1], self.mdp.terminal_state)
        self.assertAlmostEqual(outcomes[0][2], -self.mdp.invalid_action_penalty)

    def test_zero_action_has_zero_reward_and_preserves_cash(self):
        self.assertEqual(
            self.mdp.transition_outcomes((0, 5), 0),
            [(1.0, (1, 5), 0.0)],
        )

    def test_paid_and_missed_outcomes_differ_in_next_period_liquidity(self):
        outcomes = self.mdp.transition_outcomes((0, 5), 2)
        self.assertEqual(len(outcomes), 2)
        paid = next(outcome for outcome in outcomes if outcome[1] == (1, 5))
        missed = next(outcome for outcome in outcomes if outcome[1] == (1, 3))
        self.assertGreater(paid[0], 0.0)
        self.assertGreater(missed[0], 0.0)
        self.assertAlmostEqual(paid[2] - missed[2], self.mdp.default_cost)

    def test_optimal_policy_not_worse_than_no_loan_policy(self):
        _, optimal_values, _ = self.mdp.exact_q_star()
        zero_policy = {state: 0 for state in self.mdp.states}
        zero_values = self.mdp.evaluate_policy_exact(zero_policy)
        self.assertGreaterEqual(
            optimal_values[self.mdp.start_state] + 1e-10,
            zero_values[self.mdp.start_state],
        )

    def test_q_learning_outputs_finite_values(self):
        q = self.mdp.train_tabular_q_learning(episodes=2_000, seed=7)
        self.assertTrue(all(np.all(np.isfinite(values)) for values in q.values()))


if __name__ == "__main__":
    unittest.main()
