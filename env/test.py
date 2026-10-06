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

EVAL_FREQ = 2000
N_VALIDATION_CUSTOMERS = 300

Q_TOL = 0.01
POLICY_TOL = 0.01

CONVERGENCE_WINDOW = 5

# Number of test customers used for final evaluation.
# Set to None to use the entire test set.
N_TEST_CUSTOMERS = 1000

N_TEST_ROLLOUTS = 5


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
        train_features["Risk_Min"] +
        train_features["Risk_Max"]
    ) / 2

    scaler = StandardScaler()

    scaler.fit(
        train_features[FEATURE_COLUMNS]
    )

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
# Build fixed states
# ============================================================

def build_fixed_states(
    env,
    raw_df,
    scaler,
    n_customers,
    seed=42
):
    """
    Select a fixed set of customers from validation data.

    The exact same customers are used for every checkpoint.
    """

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

        customer = env._row_to_customer_state(row)

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
# Evaluate Q-values
# ============================================================

def evaluate_q_values(
    model,
    states
):
    """
    Evaluate Q-values on fixed validation states.
    """

    obs_tensor, _ = model.policy.obs_to_tensor(
        states
    )

    model.policy.set_training_mode(False)

    with torch.no_grad():

        q_values = model.q_net(
            obs_tensor
        )

    q_values = q_values.cpu().numpy()

    max_q_values = np.max(
        q_values,
        axis=1
    )

    greedy_actions = np.argmax(
        q_values,
        axis=1
    )

    return {
        "mean_max_q": float(
            np.mean(max_q_values)
        ),
        "std_max_q": float(
            np.std(max_q_values)
        ),
        "greedy_actions": greedy_actions,
        "q_values": q_values,
    }


# ============================================================
# Calculate policy change
# ============================================================

def calculate_policy_change(
    previous_actions,
    current_actions
):
    """
    Fraction of validation states whose
    greedy action changed.
    """

    if previous_actions is None:
        return np.nan

    return float(
        np.mean(
            previous_actions != current_actions
        )
    )


# ============================================================
# Calculate Q-value change
# ============================================================

def calculate_q_change(
    previous_q,
    current_q
):
    """
    Relative change in mean max-Q.
    """

    if previous_q is None:
        return np.nan

    return float(
        abs(current_q - previous_q)
        / max(abs(previous_q), 1e-8)
    )


# ============================================================
# Convergence detection
# ============================================================

def detect_convergence(
    q_changes,
    policy_changes,
    q_tol,
    policy_tol,
    window
):

    if len(q_changes) < window:
        return False

    recent_q = np.asarray(
        q_changes[-window:],
        dtype=float
    )

    recent_policy = np.asarray(
        policy_changes[-window:],
        dtype=float
    )

    if np.any(np.isnan(recent_q)):
        return False

    if np.any(np.isnan(recent_policy)):
        return False

    q_stable = np.all(
        recent_q <= q_tol
    )

    policy_stable = np.all(
        recent_policy <= policy_tol
    )

    return (
        q_stable
        and policy_stable
    )


# ============================================================
# Find checkpoints
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
# Validation convergence analysis
# ============================================================

def run_convergence_analysis(
    val_env,
    val_raw,
    scaler
):

    _, fixed_states = build_fixed_states(
        env=val_env,
        raw_df=val_raw,
        scaler=scaler,
        n_customers=N_VALIDATION_CUSTOMERS,
        seed=42
    )

    checkpoint_files, extract_step = \
        get_checkpoints()

    results = []

    previous_actions = None
    previous_mean_q = None

    q_changes = []
    policy_changes = []

    converged_at = None

    # --------------------------------------------------------
    # Evaluate every checkpoint on SAME validation states
    # --------------------------------------------------------

    for filename in checkpoint_files:

        step = extract_step(filename)

        path = os.path.join(
            CHECKPOINT_DIR,
            filename
        )

        print(
            f"\nValidation checkpoint: "
            f"{step:,}"
        )

        model = DQN.load(path)

        q_result = evaluate_q_values(
            model=model,
            states=fixed_states
        )

        current_mean_q = \
            q_result["mean_max_q"]

        current_actions = \
            q_result["greedy_actions"]

        q_change = calculate_q_change(
            previous_q=previous_mean_q,
            current_q=current_mean_q
        )

        policy_change = calculate_policy_change(
            previous_actions=previous_actions,
            current_actions=current_actions
        )

        q_changes.append(q_change)
        policy_changes.append(policy_change)

        results.append({
            "timestep": step,
            "mean_max_q": current_mean_q,
            "std_max_q": q_result["std_max_q"],
            "q_change": q_change,
            "policy_change": policy_change,
        })

        if not np.isnan(q_change):

            print(
                f"Q change      : "
                f"{q_change:.4%}"
            )

        if not np.isnan(policy_change):

            print(
                f"Policy change : "
                f"{policy_change:.4%}"
            )

        # ----------------------------------------------------
        # Detect first convergence point
        # ----------------------------------------------------

        if converged_at is None:

            stable = detect_convergence(
                q_changes=q_changes,
                policy_changes=policy_changes,
                q_tol=Q_TOL,
                policy_tol=POLICY_TOL,
                window=CONVERGENCE_WINDOW
            )

            if stable:
                converged_at = step

        previous_actions = \
            current_actions.copy()

        previous_mean_q = current_mean_q

    results_df = pd.DataFrame(results)

    return (
        results_df,
        converged_at
    )


# ============================================================
# Final test evaluation
# ============================================================

def evaluate_on_test(
    model,
    test_env,
    test_raw,
    n_customers=None,
    n_rollouts=5,
    seed=42
):
    """
    Final evaluation on the untouched test set.

    This function is called only once, after the model
    has been selected using validation.
    """

    rng = np.random.default_rng(seed)

    if (
        n_customers is None
        or n_customers >= len(test_raw)
    ):

        indices = np.arange(
            len(test_raw)
        )

    else:

        indices = rng.choice(
            len(test_raw),
            size=n_customers,
            replace=False
        )

    rewards = []
    selected_actions = []
    approved_amounts = []
    default_events = []

    for idx in indices:

        row = test_raw.iloc[idx]

        for _ in range(n_rollouts):

            customer = \
                test_env._row_to_customer_state(row)

            test_env.current_customer = \
                customer

            observation = \
                test_env._get_scaled_observation()

            action, _ = model.predict(
                observation,
                deterministic=True
            )

            action = int(action)

            _, reward, _, _, info = \
                test_env.step(action)

            rewards.append(
                float(reward)
            )

            selected_actions.append(
                action
            )

            approved_amounts.append(
                info["approved_amount"]
            )

            default_events.append(
                info["default_event"]
            )

    return {
        "mean_reward": float(
            np.mean(rewards)
        ),

        "std_reward": float(
            np.std(rewards)
        ),

        "default_rate": float(
            np.mean(default_events)
        ),

        "mean_approved_amount": float(
            np.mean(approved_amounts)
        ),

        "action_distribution": (
            pd.Series(selected_actions)
            .value_counts()
            .sort_index()
        ),
    }


# ============================================================
# Main
# ============================================================

def main():

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    (
        train_raw,
        val_raw,
        test_raw,
        scaler
    ) = load_data()

    val_env = LoanEnv(
        val_raw,
        scaler
    )

    test_env = LoanEnv(
        test_raw,
        scaler
    )

    # ========================================================
    # 1. Validation / convergence
    # ========================================================

    print("\n" + "=" * 60)
    print("VALIDATION / CONVERGENCE ANALYSIS")
    print("=" * 60)

    convergence_results, converged_at = \
        run_convergence_analysis(
            val_env=val_env,
            val_raw=val_raw,
            scaler=scaler
        )

    convergence_path = os.path.join(
        RESULTS_DIR,
        "convergence_results.csv"
    )

    convergence_results.to_csv(
        convergence_path,
        index=False
    )

    # --------------------------------------------------------
    # Select model
    # --------------------------------------------------------

    checkpoint_files, extract_step = \
        get_checkpoints()

    if converged_at is not None:

        selected_step = converged_at

    else:

        # No convergence under current criteria.
        # Use final checkpoint determined by training budget.
        selected_step = extract_step(
            checkpoint_files[-1]
        )

    selected_filename = next(
        filename
        for filename in checkpoint_files
        if extract_step(filename)
        == selected_step
    )

    selected_model_path = os.path.join(
        CHECKPOINT_DIR,
        selected_filename
    )

    print("\n" + "=" * 60)
    print("MODEL SELECTION")
    print("=" * 60)

    if converged_at is not None:

        print(
            f"Convergence detected at "
            f"{selected_step:,} steps."
        )

    else:

        print(
            "No convergence detected."
        )

        print(
            f"Using final checkpoint: "
            f"{selected_step:,}"
        )

    print(
        f"Selected model: "
        f"{selected_filename}"
    )

    # ========================================================
    # 2. FINAL TEST
    # ========================================================

    print("\n" + "=" * 60)
    print("FINAL TEST EVALUATION")
    print("=" * 60)

    final_model = DQN.load(
        selected_model_path
    )

    test_results = evaluate_on_test(
        model=final_model,
        test_env=test_env,
        test_raw=test_raw,
        n_customers=N_TEST_CUSTOMERS,
        n_rollouts=N_TEST_ROLLOUTS,
        seed=123
    )

    print(
        f"Mean reward           : "
        f"{test_results['mean_reward']:.6f}"
    )

    print(
        f"Reward std            : "
        f"{test_results['std_reward']:.6f}"
    )

    print(
        f"Default rate          : "
        f"{test_results['default_rate']:.4%}"
    )

    print(
        f"Mean approved amount  : "
        f"{test_results['mean_approved_amount']:,.2f}"
    )

    print("\nAction distribution:")

    print(
        test_results["action_distribution"]
    )

    # ========================================================
    # Save final test result
    # ========================================================

    test_summary = pd.DataFrame([{
        "selected_checkpoint": selected_step,
        "mean_reward": test_results["mean_reward"],
        "reward_std": test_results["std_reward"],
        "default_rate": test_results["default_rate"],
        "mean_approved_amount":
            test_results["mean_approved_amount"],
    }])

    test_path = os.path.join(
        RESULTS_DIR,
        "final_test_results.csv"
    )

    test_summary.to_csv(
        test_path,
        index=False
    )

    # ========================================================
    # Plot convergence
    # ========================================================

    plt.figure(figsize=(10, 5))

    plt.plot(
        convergence_results["timestep"],
        convergence_results["mean_max_q"]
    )

    plt.xlabel("Training timestep")
    plt.ylabel("Mean max Q-value")
    plt.title(
        "DQN Q-value Stability on Validation Set"
    )

    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # --------------------------------------------------------
    # Policy change
    # --------------------------------------------------------

    plt.figure(figsize=(10, 5))

    plt.plot(
        convergence_results["timestep"],
        convergence_results["policy_change"]
    )

    plt.axhline(
        POLICY_TOL,
        linestyle="--",
        label="Policy tolerance"
    )

    plt.xlabel("Training timestep")
    plt.ylabel("Policy change rate")
    plt.title(
        "DQN Policy Stability on Validation Set"
    )

    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()