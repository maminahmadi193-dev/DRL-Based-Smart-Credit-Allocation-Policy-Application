from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
import torch

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from .loan_env import LoanEnv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


PATH = r"C:\\Users\\m.ahmadi\\Desktop\\FinalProject\\notebooks\\df_cleaned.csv"


# ============================================================
# Q-Value Convergence Callback
# ============================================================

class QValueCallback(BaseCallback):

    def __init__(self, eval_env, eval_freq=1, n_states=100, verbose=0):
        super().__init__(verbose)

        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_states = n_states

        self.timesteps = []
        self.q_history = []

        self.fixed_states = None

    def _on_training_start(self):

        states = []

        obs, _ = self.eval_env.reset()

        for _ in range(self.n_states):

            states.append(obs.copy())

            action = self.eval_env.action_space.sample()

            obs, _, terminated, truncated, _ = self.eval_env.step(action)

            if terminated or truncated:
                obs, _ = self.eval_env.reset()

        self.fixed_states = np.array(states)

    def _on_step(self):
        if self.num_timesteps % self.eval_freq == 0:
            obs_tensor, _ = self.model.policy.obs_to_tensor(
                self.fixed_states
            )

            self.model.policy.set_training_mode(False)

            with torch.no_grad():
                q_values = self.model.q_net(obs_tensor)

            q_values = q_values.cpu().numpy()

            # max_a Q(s,a)
            max_q_values = np.max(q_values, axis=1)

            # میانگین max Q روی stateهای ثابت
            mean_max_q = np.mean(max_q_values)

            self.timesteps.append(self.num_timesteps)
            self.q_history.append(mean_max_q)

        return True


def main():

    # ============================================================
    # Load Dataset
    # ============================================================

    df_raw = pd.read_csv(PATH)

    train_idx, test_idx = train_test_split(
        np.arange(len(df_raw)),
        test_size=0.2,
        random_state=42,
        shuffle=True
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

    train_raw = df_raw.iloc[train_idx].reset_index(drop=True)
    test_raw = df_raw.iloc[test_idx].reset_index(drop=True)

    train_features = train_raw.copy()

    train_features["Risk"] = (
        train_features["Risk_Min"] +
        train_features["Risk_Max"]
    ) / 2

    scaler = StandardScaler()

    scaler.fit(
        train_features[FEATURE_COLUMNS]
    )

    # ============================================================
    # Environment
    # ============================================================

    train_env = LoanEnv(train_raw, scaler)
    test_env = LoanEnv(test_raw, scaler)

    # ============================================================
    # Model
    # ============================================================

    model = DQN(
        policy="MlpPolicy",
        env=train_env,

        learning_rate=1e-3,
        buffer_size=10000,
        learning_starts=1000,
        batch_size=64,

        gamma=1.0,

        train_freq=1,
        target_update_interval=500,

        exploration_fraction=0.2,
        exploration_final_eps=0.05,

        verbose=1,
    )

    # ============================================================
    # Q-Value Convergence Callback
    # ============================================================

    callback = QValueCallback(
        eval_env=train_env,
        eval_freq=1,
        n_states=100
    )

    # ============================================================
    # Train
    # ============================================================

    model.learn(
        total_timesteps=100000,
        callback=callback
    )

    # ============================================================
    # Save Model
    # ============================================================

    model.save("models/dqn_loan3")
    np.save("models/q_timesteps.npy", callback.timesteps)
    np.save("models/q_history.npy", callback.q_history)

    # ============================================================
    # Plot Q-Value Convergence
    # ============================================================

    plt.figure(figsize=(10, 5))

    plt.plot(
        callback.timesteps,
        callback.q_history
    )

    plt.xlabel("Episode")
    plt.ylabel("Mean max Q-value")
    plt.title("DQN Q-value Convergence")

    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()