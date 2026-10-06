import os

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from stable_baselines3 import DQN
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from loan_env import LoanEnv


# ============================================================
# Configuration
# ============================================================

PATH = r"C:\\Users\\m.ahmadi\\Desktop\\FinalProject\\notebooks\\df_cleaned.csv"

CHECKPOINT_DIR = "models/checkpoints"
RESULTS_DIR = "models/results"

N_VALIDATION_CUSTOMERS = 300
N_TEST_CUSTOMERS = 1000

# ------------------------------------------------------------
# Thresholds
# ------------------------------------------------------------

# Regret <= this value is considered near-optimal
REGRET_TOL = 0.01

# If the reward difference between two changed actions
# is <= this value, we consider them practically equivalent.
ACTION_FLIP_REWARD_TOL = 0.01

# If simulator's best and second-best actions differ by
# <= this amount, the decision itself is considered a near-tie.
SIMULATOR_GAP_TOL = 0.01

# ------------------------------------------------------------
# Test seed
# ------------------------------------------------------------

VALIDATION_SEED = 42
TEST_SEED = 123


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


# ============================================================
# Data split
# ============================================================

def load_data():

    df_raw = pd.read_csv(PATH)

    all_indices = np.arange(len(df_raw))

    # --------------------------------------------------------
    # 70% Train / 30% Temporary
    # --------------------------------------------------------

    train_idx, temp_idx = train_test_split(
        all_indices,
        test_size=0.30,
        random_state=42,
        shuffle=True
    )

    # --------------------------------------------------------
    # 10% Validation / 20% Test
    # --------------------------------------------------------

    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=2 / 3,
        random_state=42,
        shuffle=True
    )

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

    # --------------------------------------------------------
    # Fit scaler ONLY on train
    # --------------------------------------------------------

    train_features = train_raw.copy()

    train_features["Risk"] = (
        train_features["Risk_Min"]
        + train_features["Risk_Max"]
    ) / 2

    scaler = StandardScaler()

    scaler.fit(
        train_features[FEATURE_COLUMNS]
    )

    print("=" * 60)
    print("DATA")
    print("=" * 60)

    print(
        f"Train      : {len(train_raw):,}"
    )

    print(
        f"Validation : {len(val_raw):,}"
    )

    print(
        f"Test       : {len(test_raw):,}"
    )

    return (
        train_raw,
        val_raw,
        test_raw,
        scaler
    )


# ============================================================
# Build fixed validation states
# ============================================================

def build_fixed_validation_states(
    env,
    raw_df,
    scaler,
    n_customers,
    seed=42
):

    rng = np.random.default_rng(seed)

    n_customers = min(
        n_customers,
        len(raw_df)
    )

    indices = rng.choice(
        len(raw_df),
        size=n_customers,
        replace=False
    )

    states = []

    for idx in indices:

        row = raw_df.iloc[idx]

        customer = env._row_to_customer_state(
            row
        )

        observation = customer.to_array()

        observation = scaler.transform(
            observation.reshape(1, -1)
        )[0]

        states.append(
            observation.astype(np.float32)
        )

    return (
        indices,
        np.asarray(
            states,
            dtype=np.float32
        )
    )


# ============================================================
# Get checkpoints
# ============================================================

def get_checkpoints():

    files = []

    for filename in os.listdir(
        CHECKPOINT_DIR
    ):

        if not filename.endswith(".zip"):
            continue

        if not filename.startswith(
            "dqn_loan_"
        ):
            continue

        files.append(filename)

    if not files:

        raise FileNotFoundError(
            f"No checkpoints found in "
            f"{CHECKPOINT_DIR}"
        )

    def extract_step(filename):

        name = filename.replace(
            ".zip",
            ""
        )

        # dqn_loan_2000_steps
        return int(
            name.split("_")[-2]
        )

    files.sort(
        key=extract_step
    )

    return files, extract_step


# ============================================================
# Evaluate DQN Q-values
# ============================================================

def evaluate_q_values(
    model,
    states
):

    obs_tensor, _ = (
        model.policy.obs_to_tensor(
            states
        )
    )

    model.policy.set_training_mode(False)

    with torch.no_grad():

        q_values = model.q_net(
            obs_tensor
        )

    model.policy.set_training_mode(True)

    q_values = (
        q_values
        .cpu()
        .numpy()
    )

    # --------------------------------------------------------
    # Best and second-best Q actions
    # --------------------------------------------------------

    sorted_indices = np.argsort(
        q_values,
        axis=1
    )

    best_actions = (
        sorted_indices[:, -1]
    )

    second_actions = (
        sorted_indices[:, -2]
    )

    rows = np.arange(
        len(states)
    )

    best_q = q_values[
        rows,
        best_actions
    ]

    second_q = q_values[
        rows,
        second_actions
    ]

    q_margin = (
        best_q - second_q
    )

    return {
        "q_values": q_values,
        "best_actions": best_actions,
        "second_actions": second_actions,
        "q_margin": q_margin,
        "mean_max_q": float(
            np.mean(best_q)
        ),
        "std_max_q": float(
            np.std(best_q)
        ),
        "mean_q_margin": float(
            np.mean(q_margin)
        ),
        "median_q_margin": float(
            np.median(q_margin)
        ),
        "p10_q_margin": float(
            np.percentile(
                q_margin,
                10
            )
        )
    }


# ============================================================
# Evaluate simulator reward for ALL actions
# ============================================================

def build_simulator_reward_matrix(
    env,
    raw_df,
    indices
):

    n_customers = len(indices)

    n_actions = (
        env.action_space.n
    )

    reward_matrix = np.zeros(
        (
            n_customers,
            n_actions
        ),
        dtype=float
    )

    print()
    print("=" * 60)
    print("BUILDING SIMULATOR REWARD MATRIX")
    print("=" * 60)

    for i, idx in enumerate(indices):

        row = raw_df.iloc[idx]

        for action in range(
            n_actions
        ):

            customer = (
                env._row_to_customer_state(
                    row
                )
            )

            env.current_customer = (
                customer
            )

            # This is only needed to make
            # the environment internally consistent.
            env._get_scaled_observation()

            _, reward, _, _, _ = (
                env.step(action)
            )

            reward_matrix[
                i,
                action
            ] = float(reward)

        if (i + 1) % 50 == 0:
            print(
                f"Processed "
                f"{i + 1:,}/{n_customers:,}"
            )

    return reward_matrix


# ============================================================
# Analyze simulator action landscape
# ============================================================

def analyze_simulator_rewards(
    reward_matrix
):

    sorted_rewards = np.sort(
        reward_matrix,
        axis=1
    )

    best_rewards = (
        sorted_rewards[:, -1]
    )

    second_best_rewards = (
        sorted_rewards[:, -2]
    )

    best_actions = (
        np.argmax(
            reward_matrix,
            axis=1
        )
    )

    best_second_gap = (
        best_rewards
        - second_best_rewards
    )

    return {
        "best_rewards": best_rewards,
        "second_best_rewards": second_best_rewards,
        "best_actions": best_actions,
        "best_second_gap": best_second_gap
    }


# ============================================================
# Evaluate one checkpoint
# ============================================================

def evaluate_checkpoint(
    model,
    states,
    reward_matrix,
    simulator_info,
    previous_actions=None
):

    q_result = evaluate_q_values(
        model=model,
        states=states
    )

    current_actions = (
        q_result["best_actions"]
    )

    rows = np.arange(
        len(current_actions)
    )

    # --------------------------------------------------------
    # Reward achieved by DQN action
    # --------------------------------------------------------

    chosen_rewards = (
        reward_matrix[
            rows,
            current_actions
        ]
    )

    optimal_rewards = (
        simulator_info[
            "best_rewards"
        ]
    )

    # --------------------------------------------------------
    # Regret
    # --------------------------------------------------------

    regrets = (
        optimal_rewards
        - chosen_rewards
    )

    # Numerical noise can sometimes create tiny negatives.
    regrets = np.maximum(
        regrets,
        0.0
    )

    # --------------------------------------------------------
    # Exact optimality
    # --------------------------------------------------------

    optimal_actions = (
        simulator_info[
            "best_actions"
        ]
    )

    exact_optimal = (
        current_actions
        == optimal_actions
    )

    # --------------------------------------------------------
    # Near optimality
    # --------------------------------------------------------

    near_optimal = (
        regrets
        <= REGRET_TOL
    )

    # --------------------------------------------------------
    # Policy change from previous checkpoint
    # --------------------------------------------------------

    if previous_actions is None:

        policy_change = np.nan

        changed_mask = np.zeros(
            len(current_actions),
            dtype=bool
        )

        changed_reward_gap = np.array(
            [],
            dtype=float
        )

    else:

        changed_mask = (
            previous_actions
            != current_actions
        )

        policy_change = float(
            np.mean(changed_mask)
        )

        changed_indices = np.where(
            changed_mask
        )[0]

        changed_reward_gap = (
            np.abs(
                reward_matrix[
                    changed_indices,
                    current_actions[
                        changed_indices
                    ]
                ]
                -
                reward_matrix[
                    changed_indices,
                    previous_actions[
                        changed_indices
                    ]
                ]
            )
        )

    # --------------------------------------------------------
    # Are policy flips happening in near-ties?
    # --------------------------------------------------------

    if len(changed_reward_gap) == 0:

        flip_near_tie_rate = np.nan

        mean_changed_reward_gap = np.nan

        median_changed_reward_gap = np.nan

    else:

        flip_near_tie = (
            changed_reward_gap
            <= ACTION_FLIP_REWARD_TOL
        )

        flip_near_tie_rate = float(
            np.mean(
                flip_near_tie
            )
        )

        mean_changed_reward_gap = float(
            np.mean(
                changed_reward_gap
            )
        )

        median_changed_reward_gap = float(
            np.median(
                changed_reward_gap
            )
        )

    # --------------------------------------------------------
    # Simulator's inherent ambiguity
    # --------------------------------------------------------

    simulator_best_second_gap = (
        simulator_info[
            "best_second_gap"
        ]
    )

    simulator_near_tie = (
        simulator_best_second_gap
        <= SIMULATOR_GAP_TOL
    )

    return {
        "mean_max_q":
            q_result["mean_max_q"],

        "std_max_q":
            q_result["std_max_q"],

        "mean_q_margin":
            q_result["mean_q_margin"],

        "median_q_margin":
            q_result["median_q_margin"],

        "p10_q_margin":
            q_result["p10_q_margin"],

        "mean_regret":
            float(np.mean(regrets)),

        "median_regret":
            float(np.median(regrets)),

        "max_regret":
            float(np.max(regrets)),

        "exact_optimal_rate":
            float(np.mean(exact_optimal)),

        "near_optimal_rate":
            float(np.mean(near_optimal)),

        "policy_change":
            policy_change,

        "n_changed_actions":
            int(np.sum(changed_mask)),

        "mean_changed_reward_gap":
            mean_changed_reward_gap,

        "median_changed_reward_gap":
            median_changed_reward_gap,

        "flip_near_tie_rate":
            flip_near_tie_rate,

        "simulator_near_tie_rate":
            float(np.mean(simulator_near_tie)),

        "mean_simulator_best_second_gap":
            float(
                np.mean(
                    simulator_best_second_gap
                )
            ),

        "median_simulator_best_second_gap":
            float(
                np.median(
                    simulator_best_second_gap
                )
            ),

        "actions":
            current_actions.copy(),

        "regrets":
            regrets.copy(),

        "q_margin":
            q_result["q_margin"].copy()
    }


# ============================================================
# Main validation stability analysis
# ============================================================

def run_analysis():

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    (
        _,
        val_raw,
        _,
        scaler
    ) = load_data()

    val_env = LoanEnv(
        val_raw,
        scaler
    )

    # --------------------------------------------------------
    # Fixed validation states
    # --------------------------------------------------------

    (
        fixed_indices,
        fixed_states
    ) = build_fixed_validation_states(
        env=val_env,
        raw_df=val_raw,
        scaler=scaler,
        n_customers=N_VALIDATION_CUSTOMERS,
        seed=VALIDATION_SEED
    )

    print()
    print(
        f"Fixed validation customers: "
        f"{len(fixed_indices):,}"
    )

    # --------------------------------------------------------
    # Simulator reward matrix
    #
    # This is independent of the checkpoint,
    # so we calculate it only once.
    # --------------------------------------------------------

    reward_matrix = (
        build_simulator_reward_matrix(
            env=val_env,
            raw_df=val_raw,
            indices=fixed_indices
        )
    )

    simulator_info = (
        analyze_simulator_rewards(
            reward_matrix
        )
    )

    # Save simulator matrix
    reward_path = os.path.join(
        RESULTS_DIR,
        "validation_simulator_rewards.csv"
    )

    reward_df = pd.DataFrame(
        reward_matrix,
        columns=[
            f"action_{a}"
            for a in range(
                reward_matrix.shape[1]
            )
        ]
    )

    reward_df.to_csv(
        reward_path,
        index=False
    )

    # --------------------------------------------------------
    # Checkpoints
    # --------------------------------------------------------

    checkpoint_files, extract_step = (
        get_checkpoints()
    )

    results = []

    previous_actions = None

    # --------------------------------------------------------
    # Evaluate checkpoints
    # --------------------------------------------------------

    for filename in checkpoint_files:

        step = extract_step(
            filename
        )

        path = os.path.join(
            CHECKPOINT_DIR,
            filename
        )

        print()
        print("=" * 60)
        print(
            f"CHECKPOINT: {step:,}"
        )
        print("=" * 60)

        model = DQN.load(
            path
        )

        checkpoint_result = (
            evaluate_checkpoint(
                model=model,
                states=fixed_states,
                reward_matrix=reward_matrix,
                simulator_info=simulator_info,
                previous_actions=previous_actions
            )
        )

        results.append({

            "timestep":
                step,

            "mean_max_q":
                checkpoint_result[
                    "mean_max_q"
                ],

            "std_max_q":
                checkpoint_result[
                    "std_max_q"
                ],

            "mean_q_margin":
                checkpoint_result[
                    "mean_q_margin"
                ],

            "median_q_margin":
                checkpoint_result[
                    "median_q_margin"
                ],

            "p10_q_margin":
                checkpoint_result[
                    "p10_q_margin"
                ],

            "mean_regret":
                checkpoint_result[
                    "mean_regret"
                ],

            "median_regret":
                checkpoint_result[
                    "median_regret"
                ],

            "max_regret":
                checkpoint_result[
                    "max_regret"
                ],

            "exact_optimal_rate":
                checkpoint_result[
                    "exact_optimal_rate"
                ],

            "near_optimal_rate":
                checkpoint_result[
                    "near_optimal_rate"
                ],

            "policy_change":
                checkpoint_result[
                    "policy_change"
                ],

            "n_changed_actions":
                checkpoint_result[
                    "n_changed_actions"
                ],

            "mean_changed_reward_gap":
                checkpoint_result[
                    "mean_changed_reward_gap"
                ],

            "median_changed_reward_gap":
                checkpoint_result[
                    "median_changed_reward_gap"
                ],

            "flip_near_tie_rate":
                checkpoint_result[
                    "flip_near_tie_rate"
                ],

            "simulator_near_tie_rate":
                checkpoint_result[
                    "simulator_near_tie_rate"
                ],

            "mean_simulator_best_second_gap":
                checkpoint_result[
                    "mean_simulator_best_second_gap"
                ],

            "median_simulator_best_second_gap":
                checkpoint_result[
                    "median_simulator_best_second_gap"
                ],
        })

        print(
            f"Mean Q               : "
            f"{checkpoint_result['mean_max_q']:.6f}"
        )

        print(
            f"Mean Q margin        : "
            f"{checkpoint_result['mean_q_margin']:.6f}"
        )

        print(
            f"Mean regret          : "
            f"{checkpoint_result['mean_regret']:.6f}"
        )

        print(
            f"Exact optimality     : "
            f"{checkpoint_result['exact_optimal_rate']:.2%}"
        )

        print(
            f"Near optimality      : "
            f"{checkpoint_result['near_optimal_rate']:.2%}"
        )

        if not np.isnan(
            checkpoint_result[
                "policy_change"
            ]
        ):

            print(
                f"Policy change        : "
                f"{checkpoint_result['policy_change']:.2%}"
            )

            print(
                f"Changed actions      : "
                f"{checkpoint_result['n_changed_actions']}"
            )

            print(
                f"Changed reward gap   : "
                f"{checkpoint_result['mean_changed_reward_gap']:.6f}"
            )

            print(
                f"Flip near-tie rate   : "
                f"{checkpoint_result['flip_near_tie_rate']:.2%}"
            )

        print(
            f"Simulator tie rate   : "
            f"{checkpoint_result['simulator_near_tie_rate']:.2%}"
        )

        previous_actions = (
            checkpoint_result[
                "actions"
            ]
        )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    results_path = os.path.join(
        RESULTS_DIR,
        "policy_stability_analysis.csv"
    )

    results_df.to_csv(
        results_path,
        index=False
    )

    print()
    print("=" * 60)
    print("RESULT SAVED")
    print("=" * 60)

    print(
        results_path
    )

    # ========================================================
    # Plot 1: Policy change
    # ========================================================

    plt.figure(
        figsize=(10, 5)
    )

    plt.plot(
        results_df["timestep"],
        results_df["policy_change"],
        marker="o"
    )

    plt.xlabel(
        "Training timestep"
    )

    plt.ylabel(
        "Policy change rate"
    )

    plt.title(
        "Policy Change Across Checkpoints"
    )

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            RESULTS_DIR,
            "policy_change.png"
        ),
        dpi=150
    )

    plt.show()

    # ========================================================
    # Plot 2: Q margin
    # ========================================================

    plt.figure(
        figsize=(10, 5)
    )

    plt.plot(
        results_df["timestep"],
        results_df["mean_q_margin"],
        marker="o"
    )

    plt.xlabel(
        "Training timestep"
    )

    plt.ylabel(
        "Mean Q margin"
    )

    plt.title(
        "Q-value Margin Between Best and Second-best Action"
    )

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            RESULTS_DIR,
            "q_margin.png"
        ),
        dpi=150
    )

    plt.show()

    # ========================================================
    # Plot 3: Simulator ambiguity
    # ========================================================

    plt.figure(
        figsize=(10, 5)
    )

    plt.plot(
        results_df["timestep"],
        results_df[
            "mean_simulator_best_second_gap"
        ],
        marker="o"
    )

    plt.xlabel(
        "Training timestep"
    )

    plt.ylabel(
        "Simulator reward gap"
    )

    plt.title(
        "Gap Between Best and Second-best Simulator Actions"
    )

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            RESULTS_DIR,
            "simulator_action_gap.png"
        ),
        dpi=150
    )

    plt.show()

    # ========================================================
    # Plot 4: Regret
    # ========================================================

    plt.figure(
        figsize=(10, 5)
    )

    plt.plot(
        results_df["timestep"],
        results_df["mean_regret"],
        marker="o"
    )

    plt.xlabel(
        "Training timestep"
    )

    plt.ylabel(
        "Mean regret"
    )

    plt.title(
        "Mean Regret Across Checkpoints"
    )

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            RESULTS_DIR,
            "mean_regret.png"
        ),
        dpi=150
    )

    plt.show()


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    run_analysis()