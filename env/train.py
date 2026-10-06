from stable_baselines3 import DQN

from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    CallbackList,
)

import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from loan_env import LoanEnv

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ============================================================================
# Configuration
# ============================================================================

PATH = r"C:\\Users\\m.ahmadi\\Desktop\\FinalProject\\notebooks\\df_cleaned.csv"

TOTAL_TIMESTEPS = 100000
CHECKPOINT_FREQ = 2000

CHECKPOINT_DIR = "models/checkpoints"
FINAL_MODEL_PATH = "models/dqn_loan"

Q_TIMESTEPS_PATH = "models/q_timesteps.npy"
Q_HISTORY_PATH = "models/q_history.npy"


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


# ============================================================================
# Q-Value Monitoring Callback
# ============================================================================
#
# IMPORTANT:
# This callback is ONLY for monitoring/visualization.
# It is NOT used to decide convergence.
#
# Convergence analysis is performed separately in test.py
# using the validation set.
#
# ============================================================================

class QValueCallback(BaseCallback):

    def __init__(
        self,
        eval_env,
        eval_freq=2000,
        n_states=100,
        verbose=0,
    ):
        super().__init__(verbose)

        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_states = n_states

        self.timesteps = []
        self.q_history = []

        self.fixed_states = None

    # ------------------------------------------------------------------------
    # Create fixed states before training starts
    # ------------------------------------------------------------------------

    def _on_training_start(self):

        states = []

        obs, _ = self.eval_env.reset()

        for _ in range(self.n_states):

            states.append(obs.copy())

            # Environment has one-step episodes,
            # so this simply moves to another random customer.
            action = self.eval_env.action_space.sample()

            obs, _, terminated, truncated, _ = (
                self.eval_env.step(action)
            )

            if terminated or truncated:

                obs, _ = self.eval_env.reset()

        self.fixed_states = np.asarray(
            states,
            dtype=np.float32
        )

    # ------------------------------------------------------------------------
    # Evaluate Q-values periodically
    # ------------------------------------------------------------------------

    def _on_step(self):

        if self.num_timesteps % self.eval_freq != 0:
            return True

        # Convert fixed states to tensors
        obs_tensor, _ = self.model.policy.obs_to_tensor(
            self.fixed_states
        )

        # Evaluation mode
        self.model.policy.set_training_mode(False)

        with torch.no_grad():

            q_values = self.model.q_net(
                obs_tensor
            )

        q_values = q_values.cpu().numpy()

        # max_a Q(s,a)
        max_q_values = np.max(
            q_values,
            axis=1
        )

        # Mean max Q over fixed states
        mean_max_q = np.mean(
            max_q_values
        )

        self.timesteps.append(
            self.num_timesteps
        )

        self.q_history.append(
            mean_max_q
        )

        # Restore training mode
        self.model.policy.set_training_mode(True)

        return True


# ============================================================================
# Main
# ============================================================================

def main():

    # =========================================================================
    # Create required directories
    # =========================================================================

    os.makedirs(
        "models",
        exist_ok=True
    )

    os.makedirs(
        CHECKPOINT_DIR,
        exist_ok=True
    )

    # =========================================================================
    # Load Dataset
    # =========================================================================

    print("=" * 70)
    print("Loading dataset...")
    print("=" * 70)

    df_raw = pd.read_csv(PATH)

    print(
        f"Total samples: {len(df_raw):,}"
    )

    # =========================================================================
    # Train / Validation / Test Split
    #
    # 70% Train
    # 10% Validation
    # 20% Test
    #
    # The validation and test sets are NOT used during training.
    #
    # The exact same split logic is used in test.py.
    # =========================================================================

    all_indices = np.arange(
        len(df_raw)
    )

    # -------------------------------------------------------------------------
    # 70% Train / 30% Temporary
    # -------------------------------------------------------------------------

    train_idx, temp_idx = train_test_split(
        all_indices,
        test_size=0.30,
        random_state=42,
        shuffle=True
    )

    # -------------------------------------------------------------------------
    # 10% Validation / 20% Test
    # -------------------------------------------------------------------------

    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=2 / 3,
        random_state=42,
        shuffle=True
    )

    # -------------------------------------------------------------------------
    # Create DataFrames
    # -------------------------------------------------------------------------

    train_raw = (
        df_raw
        .iloc[train_idx]
        .reset_index(drop=True)
    )

    val_raw = (
        df_raw
        .iloc[val_idx]
        .reset_index(drop=True)
    )

    test_raw = (
        df_raw
        .iloc[test_idx]
        .reset_index(drop=True)
    )

    print()
    print(
        f"Train      : {len(train_raw):,} "
        f"({len(train_raw) / len(df_raw):.1%})"
    )

    print(
        f"Validation : {len(val_raw):,} "
        f"({len(val_raw) / len(df_raw):.1%})"
    )

    print(
        f"Test       : {len(test_raw):,} "
        f"({len(test_raw) / len(df_raw):.1%})"
    )

    # =========================================================================
    # StandardScaler
    #
    # IMPORTANT:
    # Fit ONLY on training data.
    #
    # Validation and test are never used for fitting the scaler.
    # =========================================================================

    train_features = train_raw.copy()

    train_features["Risk"] = (
        train_features["Risk_Min"]
        + train_features["Risk_Max"]
    ) / 2

    scaler = StandardScaler()

    scaler.fit(
        train_features[FEATURE_COLUMNS]
    )

    print()
    print("StandardScaler fitted on training data only.")

    # =========================================================================
    # Training Environment
    #
    # Only train_raw is passed to DQN.
    #
    # Validation and test data remain untouched during training.
    # =========================================================================

    train_env = LoanEnv(
        train_raw,
        scaler
    )

    # =========================================================================
    # Checkpoint Callback
    #
    # Saves:
    #
    # models/checkpoints/
    #   dqn_loan_2000_steps.zip
    #   dqn_loan_4000_steps.zip
    #   ...
    #
    # =========================================================================

    checkpoint_callback = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path=CHECKPOINT_DIR,
        name_prefix="dqn_loan"
    )

    # =========================================================================
    # Q-Value Monitoring Callback
    #
    # This is ONLY for plotting Q-value behavior during training.
    # It does NOT determine convergence.
    # =========================================================================

    q_callback = QValueCallback(
        eval_env=train_env,
        eval_freq=CHECKPOINT_FREQ,
        n_states=100,
        verbose=0
    )

    # =========================================================================
    # Combine Callbacks
    # =========================================================================

    callback = CallbackList([
        checkpoint_callback,
        q_callback
    ])

    # =========================================================================
    # Create DQN Model
    # =========================================================================

    print()
    print("=" * 70)
    print("Creating DQN model...")
    print("=" * 70)

    model = DQN(
        policy="MlpPolicy",
        env=train_env,

        learning_rate=3e-4,

        buffer_size=10000,

        learning_starts=1000,

        batch_size=64,

        # Since each episode terminates after one action,
        # future-state discounting has no practical effect.
        gamma=1.0,

        train_freq=1,

        target_update_interval=500,

        exploration_fraction=0.2,

        exploration_final_eps=0.05,

        verbose=1,
    )

    # =========================================================================
    # Train
    # =========================================================================

    print()
    print("=" * 70)
    print("Starting training...")
    print("=" * 70)

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=callback
    )

    # =========================================================================
    # Save Final Model
    # =========================================================================

    model.save(
        FINAL_MODEL_PATH
    )

    print()
    print("=" * 70)
    print("Training finished.")
    print("=" * 70)

    print(
        f"Final model saved to:"
        f"\n{FINAL_MODEL_PATH}.zip"
    )

    # =========================================================================
    # Save Q-Value History
    # =========================================================================

    np.save(
        Q_TIMESTEPS_PATH,
        np.asarray(
            q_callback.timesteps
        )
    )

    np.save(
        Q_HISTORY_PATH,
        np.asarray(
            q_callback.q_history
        )
    )

    print(
        f"\nQ-value history saved to:"
        f"\n{Q_TIMESTEPS_PATH}"
        f"\n{Q_HISTORY_PATH}"
    )

    # =========================================================================
    # Plot Q-Value During Training
    # =========================================================================

    if len(q_callback.timesteps) > 0:

        plt.figure(
            figsize=(10, 5)
        )

        plt.plot(
            q_callback.timesteps,
            q_callback.q_history
        )

        plt.xlabel(
            "Training timestep"
        )

        plt.ylabel(
            "Mean max Q-value"
        )

        plt.title(
            "DQN Q-value During Training"
        )

        plt.grid(
            True
        )

        plt.tight_layout()

        plt.show()

    else:

        print(
            "\nNo Q-value data was collected."
        )


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    main()