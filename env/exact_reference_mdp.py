"""Exact finite-MDP reference benchmark for the credit-allocation project.

This is a REDUCED benchmark, not an exact clone of the full portfolio simulator.
It has two fixed unique applicants, a 500M starting budget, the same six action
amounts, the production social-reward and risk-update formulas, and stochastic
missed-payment outcomes that affect liquidity for the next applicant.

To keep the MDP exactly enumerable, each toy loan is fully amortized after one
period. Interest is counted in episode reward whether the installment is paid
on time or collected later; a missed principal is unavailable until settlement
and therefore cannot fund the next applicant. The missed-default penalty is
placed on the same transition. With gamma=1 these timing abstractions preserve
undiscounted episode return, but do not reproduce the full 18-month lifecycle,
90+ DPD timing, reapplications, or calibrated production hazard. The hazard
ceiling is deliberately elevated for an algorithmic stress test.

The reachable decision state is exactly (applicant_index, cash_in_100M_units):
the initial state is (0, 5), followed by (1, cash) for cash=0..5. Bellman
backward induction calculates Q* exactly for this reduced MDP. Tabular
Q-learning and SB3 DQN are then compared against that known solution.
"""
from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path
import sys
from typing import Dict, List, Sequence, Tuple

import numpy as np

_ENV_DIR = Path(__file__).resolve().parent
if str(_ENV_DIR) not in sys.path:
    sys.path.insert(0, str(_ENV_DIR))

from customer_simulator import CustomerSimulator
from customer_state import CustomerState
from reward_engine import RewardEngine

State = Tuple[int, int]
Outcome = Tuple[float, State, float]


class FiniteCreditReferenceMDP:
    UNIT = 100_000_000.0
    ACTION_UNITS = (0, 1, 2, 3, 4, 5)
    N_APPLICANTS = 2
    INITIAL_CASH_UNITS = 5
    GAMMA = 1.0

    def __init__(
        self,
        *,
        max_hazard: float = 0.20,
        annual_interest_rate: float = 0.04,
        risk_weight: float = 1.0,
        default_cost: float = 10.0,
        invalid_action_penalty: float = 0.25,
    ) -> None:
        if not 0.0 < max_hazard < 1.0:
            raise ValueError("max_hazard must be strictly between 0 and 1")
        if annual_interest_rate < 0 or risk_weight < 0 or default_cost < 0:
            raise ValueError("Rates and penalties must be non-negative")
        if invalid_action_penalty < 0:
            raise ValueError("invalid_action_penalty must be non-negative")

        self.max_hazard = float(max_hazard)
        self.monthly_interest_rate = float(annual_interest_rate) / 12.0
        self.risk_weight = float(risk_weight)
        self.default_cost = float(default_cost)
        self.invalid_action_penalty = float(invalid_action_penalty)
        self.financial_reward_scale = self.UNIT

        # Fixed unique applicants with distinct need, facilities, income and risk.
        self.customers: Sequence[CustomerState] = (
            CustomerState(
                monthly_income=900_000_000.0, income_score=70.0, job_score=70.0,
                asset_score=60.0, total_customer_score=20.0,
                total_facilities=100_000_000.0, debt=500_000_000.0,
                origin_amount=800_000_000.0, loan_count=10,
                debt_ratio=0.625, debt_per_loan=50_000_000.0,
                has_no_history=False, employed=True, score_min=450.0,
                score_max=550.0, risk_min=20.0, risk_max=30.0, risk=25.0,
            ),
            CustomerState(
                monthly_income=900_000_000.0, income_score=80.0, job_score=85.0,
                asset_score=75.0, total_customer_score=65.0,
                total_facilities=400_000_000.0, debt=600_000_000.0,
                origin_amount=900_000_000.0, loan_count=12,
                debt_ratio=2.0 / 3.0, debt_per_loan=50_000_000.0,
                has_no_history=False, employed=True, score_min=600.0,
                score_max=700.0, risk_min=50.0, risk_max=60.0, risk=55.0,
            ),
        )
        self.customer_simulator = CustomerSimulator(
            hazard_slope=0.08, hazard_center=60.0, max_hazard=self.max_hazard,
            hazard_intercept=0.0, horizon_months=1,
        )
        self.reward_engine = RewardEngine(
            history_reward=1.0, facility_weight=1.0, need_weight=1.0,
            risk_weight=self.risk_weight, default_cost=self.default_cost,
        )
        # Only states reachable from the genuine initial condition are included.
        self.start_state: State = (0, self.INITIAL_CASH_UNITS)
        self.terminal_state: State = (self.N_APPLICANTS, 0)
        self.states: List[State] = [self.start_state] + [
            (1, cash) for cash in range(self.INITIAL_CASH_UNITS + 1)
        ]

    @property
    def action_count(self) -> int:
        return len(self.ACTION_UNITS)

    def observation(self, state: State) -> np.ndarray:
        t, cash = state
        if t >= self.N_APPLICANTS:
            return np.array([1.0, 0.0], dtype=np.float32)
        return np.array(
            [t / self.N_APPLICANTS, cash / self.INITIAL_CASH_UNITS],
            dtype=np.float32,
        )

    @lru_cache(maxsize=None)
    def transition_outcomes(self, state: State, action: int) -> List[Outcome]:
        """Exact probability kernel: list of (probability, next_state, reward)."""
        t, cash = state
        action = int(action)
        if t >= self.N_APPLICANTS:
            raise ValueError("Cannot transition from terminal state")
        if not 0 <= action < self.action_count:
            raise ValueError(f"Action index must be 0..{self.action_count - 1}")

        units = self.ACTION_UNITS[action]
        next_t = t + 1
        no_loan_next = self.terminal_state if next_t == self.N_APPLICANTS else (next_t, cash)

        if units > cash:
            return [(1.0, no_loan_next, -self.invalid_action_penalty)]
        if units == 0:
            return [(1.0, no_loan_next, 0.0)]

        amount = units * self.UNIT
        customer = self.customers[t].copy()
        social = self.reward_engine._social_reward(customer, amount)

        updated = self.customer_simulator._update_customer_state(customer, amount)
        updated.risk = self.customer_simulator._update_risk(
            customer=updated, previous_customer=customer,
            approved_unsecured_amount=amount,
        )
        risk_delta = max(0.0, updated.risk - customer.risk)
        risk_penalty = self.reward_engine.risk_increase_penalty(risk_delta)
        interest_reward = (
            amount * self.monthly_interest_rate / self.financial_reward_scale
        )
        base_reward = social - risk_penalty + interest_reward

        hazard = self.customer_simulator._estimate_hazard(updated)
        miss_probability = self.customer_simulator._calculate_monthly_event_probability(hazard)

        # A paid loan returns principal before next month's decision. A missed
        # installment is ultimately collected, but only after allocation ends.
        paid_next = self.terminal_state if next_t == self.N_APPLICANTS else (next_t, cash)
        missed_cash = cash - units
        missed_next = (
            self.terminal_state if next_t == self.N_APPLICANTS else (next_t, missed_cash)
        )
        return [
            (1.0 - miss_probability, paid_next, base_reward),
            (miss_probability, missed_next, base_reward - self.default_cost),
        ]

    def exact_q_star(
        self,
    ) -> Tuple[Dict[State, np.ndarray], Dict[State, float], Dict[State, int]]:
        """Calculate exact Q* for every reachable decision state by backward induction."""
        q_star: Dict[State, np.ndarray] = {}
        values: Dict[State, float] = {self.terminal_state: 0.0}
        policy: Dict[State, int] = {}
        ordered = [(1, cash) for cash in range(self.INITIAL_CASH_UNITS + 1)]
        ordered.append(self.start_state)
        for state in ordered:
            q_values = np.zeros(self.action_count, dtype=np.float64)
            for action in range(self.action_count):
                q_values[action] = sum(
                    p * (r + self.GAMMA * values.get(next_state, 0.0))
                    for p, next_state, r in self.transition_outcomes(state, action)
                )
            q_star[state] = q_values
            policy[state] = int(np.argmax(q_values))
            values[state] = float(q_values[policy[state]])
        return q_star, values, policy

    def evaluate_policy_exact(self, policy: Dict[State, int]) -> Dict[State, float]:
        """Exact expected policy values under the known transition kernel."""
        values: Dict[State, float] = {self.terminal_state: 0.0}
        for state in sorted(policy, key=lambda item: item[0], reverse=True):
            action = int(policy[state])
            values[state] = float(sum(
                p * (r + self.GAMMA * values.get(next_state, 0.0))
                for p, next_state, r in self.transition_outcomes(state, action)
            ))
        return values

    def sample_transition(
        self, state: State, action: int, rng: np.random.Generator
    ) -> Tuple[State, float]:
        outcomes = self.transition_outcomes(state, action)
        probs = np.asarray([x[0] for x in outcomes], dtype=np.float64)
        probs /= probs.sum()
        _, next_state, reward = outcomes[int(rng.choice(len(outcomes), p=probs))]
        return next_state, float(reward)

    def train_tabular_q_learning(
        self,
        episodes: int = 500_000,
        seed: int = 42,
        epsilon_floor: float = 0.50,
        alpha_exponent: float = 0.60,
    ) -> Dict[State, np.ndarray]:
        """Tabular Q-learning with exploring starts and Robbins-Monro stepsizes."""
        if episodes < 1:
            raise ValueError("episodes must be positive")
        if not 0.5 < alpha_exponent <= 1.0:
            raise ValueError("alpha_exponent must be in (0.5, 1]")
        if not 0.0 < epsilon_floor < 1.0:
            raise ValueError("epsilon_floor must be in (0, 1)")

        rng = np.random.default_rng(seed)
        q = {s: np.zeros(self.action_count, dtype=np.float64) for s in self.states}
        visits = {s: np.zeros(self.action_count, dtype=np.int64) for s in self.states}
        for _ in range(episodes):
            # Exploring starts sample each state repeatedly; epsilon explores actions.
            state = self.states[int(rng.integers(len(self.states)))]
            while state[0] < self.N_APPLICANTS:
                if rng.random() < epsilon_floor:
                    action = int(rng.integers(self.action_count))
                else:
                    best = np.flatnonzero(np.isclose(q[state], np.max(q[state]), atol=1e-12))
                    action = int(rng.choice(best))
                next_state, reward = self.sample_transition(state, action, rng)
                visits[state][action] += 1
                alpha = 1.0 / float(visits[state][action]) ** alpha_exponent
                target = reward if next_state[0] >= self.N_APPLICANTS else (
                    reward + self.GAMMA * float(np.max(q[next_state]))
                )
                q[state][action] += alpha * (target - q[state][action])
                state = next_state
        return q


def train_dqn_reference(mdp: FiniteCreditReferenceMDP, timesteps: int, seed: int = 42):
    """Train SB3 DQN on the exact same reduced kernel, using exploring starts."""
    try:
        import gymnasium as gym
        from gymnasium import spaces
        from stable_baselines3 import DQN
    except ImportError as exc:
        raise RuntimeError("DQN comparison requires gymnasium and stable-baselines3") from exc

    class TinyCreditEnv(gym.Env):
        metadata = {"render_modes": []}

        def __init__(self):
            super().__init__()
            self.action_space = spaces.Discrete(mdp.action_count)
            self.observation_space = spaces.Box(
                low=np.array([0.0, 0.0], dtype=np.float32),
                high=np.array([1.0, 1.0], dtype=np.float32),
                dtype=np.float32,
            )
            self.rng = np.random.default_rng(seed)
            self.state = mdp.start_state

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            if seed is not None:
                self.rng = np.random.default_rng(seed)
            self.state = mdp.states[int(self.rng.integers(len(mdp.states)))]
            return mdp.observation(self.state), {"state": self.state}

        def step(self, action):
            self.state, reward = mdp.sample_transition(self.state, int(action), self.rng)
            terminated = self.state[0] >= mdp.N_APPLICANTS
            return mdp.observation(self.state), reward, terminated, False, {"state": self.state}

    env = TinyCreditEnv()
    model = DQN(
        "MlpPolicy", env, learning_rate=1e-3, buffer_size=10_000,
        learning_starts=100, batch_size=32, gamma=mdp.GAMMA, train_freq=1,
        target_update_interval=100, exploration_fraction=0.30,
        exploration_final_eps=0.05, policy_kwargs={"net_arch": [32, 32]},
        seed=seed, verbose=0,
    )
    model.learn(total_timesteps=timesteps, progress_bar=False)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q-learning-episodes", type=int, default=500_000)
    parser.add_argument("--dqn-timesteps", type=int, default=50_000)
    parser.add_argument("--skip-dqn", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="models/results/exact_reference_mdp")
    parser.add_argument("--max-hazard", type=float, default=0.20,
                        help="Elevated stress-test hazard ceiling; not production calibration.")
    args = parser.parse_args()

    mdp = FiniteCreditReferenceMDP(max_hazard=args.max_hazard)
    q_star, v_star, pi_star = mdp.exact_q_star()
    start_value = v_star[mdp.start_state]
    print("=" * 76)
    print("EXACT FINITE CREDIT-ALLOCATION REFERENCE MDP")
    print("Reduced benchmark, NOT Q* of the full portfolio environment.")
    print(f"Reachable decision states: {len(mdp.states)}")
    print(f"Actions (units of 100M): {list(mdp.ACTION_UNITS)}")
    print(f"Starting budget: {mdp.INITIAL_CASH_UNITS * mdp.UNIT:,.0f}")
    print(f"Stress max hazard: {mdp.max_hazard:.3f} (not production calibrated)")
    print(f"Exact optimal start value V*: {start_value:.6f}")
    print("Exact optimal policy (amount in million):")
    for state in mdp.states:
        amount_m = mdp.ACTION_UNITS[pi_star[state]] * 100
        print(f"  state={state}: {amount_m:.0f}M; Q*={np.round(q_star[state], 4).tolist()}")

    q_tab = mdp.train_tabular_q_learning(episodes=args.q_learning_episodes, seed=args.seed)
    pi_tab = {s: int(np.argmax(q_tab[s])) for s in mdp.states}
    v_tab = mdp.evaluate_policy_exact(pi_tab)
    errors_tab = np.concatenate([np.abs(q_tab[s] - q_star[s]) for s in mdp.states])
    gap_tab = start_value - v_tab[mdp.start_state]
    print("\nTABULAR Q-LEARNING VS EXACT Q*")
    print(f"  Episodes: {args.q_learning_episodes:,}")
    print(f"  Exact value of learned policy: {v_tab[mdp.start_state]:.6f}")
    print(f"  Optimality gap: {gap_tab:.6f}")
    print(f"  Max absolute Q error: {errors_tab.max():.6f}")
    print(f"  Q RMSE: {np.sqrt(np.mean(errors_tab ** 2)):.6f}")

    dqn_metrics = None
    model = None
    if not args.skip_dqn:
        model = train_dqn_reference(mdp, args.dqn_timesteps, args.seed)
        pi_dqn, errors = {}, []
        import torch
        for state in mdp.states:
            obs = mdp.observation(state)
            action, _ = model.predict(obs, deterministic=True)
            pi_dqn[state] = int(action)
            with torch.no_grad():
                q_pred = model.q_net(torch.as_tensor(obs, device=model.device).reshape(1, -1))
            errors.extend(np.abs(q_pred.cpu().numpy()[0] - q_star[state]).tolist())
        v_dqn = mdp.evaluate_policy_exact(pi_dqn)
        gap_dqn = start_value - v_dqn[mdp.start_state]
        dqn_metrics = {
            "timesteps": args.dqn_timesteps,
            "start_policy_value": float(v_dqn[mdp.start_state]),
            "optimality_gap": float(gap_dqn),
            "max_abs_q_error": float(np.max(errors)),
            "q_rmse": float(np.sqrt(np.mean(np.square(errors)))),
            "policy_action_indices": {str(s): int(a) for s, a in pi_dqn.items()},
        }
        print("\nDQN VS EXACT Q* (empirical benchmark; not a general guarantee)")
        print(f"  Timesteps: {args.dqn_timesteps:,}")
        print(f"  Exact value of DQN policy: {v_dqn[mdp.start_state]:.6f}")
        print(f"  Optimality gap: {gap_dqn:.6f}")
        print(f"  Max absolute Q error: {np.max(errors):.6f}")
        print(f"  Q RMSE: {np.sqrt(np.mean(np.square(errors))):.6f}")

    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    import pandas as pd
    rows = []
    for state in mdp.states:
        for action, units in enumerate(mdp.ACTION_UNITS):
            rows.append({
                "applicant_index": state[0], "cash_units": state[1],
                "action_index": action, "action_amount": units * mdp.UNIT,
                "q_star": float(q_star[state][action]),
                "q_tabular": float(q_tab[state][action]),
                "q_tabular_abs_error": float(abs(q_tab[state][action] - q_star[state][action])),
                "q_star_optimal_action_index": int(pi_star[state]),
                "q_tabular_greedy_action_index": int(pi_tab[state]),
            })
    pd.DataFrame(rows).to_csv(out_dir / "exact_vs_tabular_q.csv", index=False)
    summary = {
        "benchmark": "two_applicant_one_period_credit_allocation",
        "is_full_portfolio_model": False,
        "reachable_decision_states": len(mdp.states),
        "initial_budget": mdp.INITIAL_CASH_UNITS * mdp.UNIT,
        "max_hazard_stress": mdp.max_hazard,
        "start_state_v_star": float(start_value),
        "tabular_q_learning": {
            "episodes": args.q_learning_episodes,
            "start_policy_value": float(v_tab[mdp.start_state]),
            "optimality_gap": float(gap_tab),
            "max_abs_q_error": float(errors_tab.max()),
            "q_rmse": float(np.sqrt(np.mean(errors_tab ** 2))),
        },
        "dqn": dqn_metrics,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if model is not None:
        model.save(str(out_dir / "dqn_reference_model"))
    print(f"\nResults saved: {out_dir}")


if __name__ == "__main__":
    main()
