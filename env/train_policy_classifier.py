import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from customer_state import CustomerState
from customer_simulator import CustomerSimulator
from reward_engine import RewardEngine


# ============================================================================
# Configuration
# ============================================================================

PATH = r"C:\Users\m.ahmadi\Desktop\FinalProject\notebooks\df_cleaned.csv"

RANDOM_STATE = 42

EPOCHS = 100
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
PATIENCE = 10

HIDDEN_DIM = 128

MODEL_PATH = "models/policy_classifier.pt"
SCALER_PATH = "models/policy_classifier_scaler.pkl"

TIE_TOLERANCE = 0.01


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


ACTIONS = np.array(
    [
        0,
        100_000_000,
        200_000_000,
        300_000_000,
        400_000_000,
        500_000_000,
    ],
    dtype=np.float32,
)


# ============================================================================
# Reproducibility
# ============================================================================

random.seed(RANDOM_STATE)

np.random.seed(RANDOM_STATE)

torch.manual_seed(RANDOM_STATE)

if torch.cuda.is_available():

    torch.cuda.manual_seed_all(RANDOM_STATE)


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================================
# Convert DataFrame Row -> CustomerState
# ============================================================================

def row_to_customer_state(row):

    return CustomerState(

        monthly_income=row["MountlyIncome"],

        income_score=row["IncomeScore"],

        job_score=row["JobScore"],

        asset_score=row["AssetScore"],

        total_customer_score=row["TotalCustomerScore"],

        total_facilities=row["TotalFacilities"],

        debt=row["DebtAmount"],

        origin_amount=row["OriginAmount"],

        loan_count=int(row["LoanCounts"]),

        debt_ratio=row["DebtRatio"],

        debt_per_loan=row["DebtPerLoan"],

        has_no_history=bool(row["Has_No_History"]),

        employed=bool(row["Employed"]),

        score_min=row["Score_Min"],

        score_max=row["Score_Max"],

        risk_min=row["Risk_Min"],

        risk_max=row["Risk_Max"],

        risk=(
            row["Risk_Min"]
            + row["Risk_Max"]
        ) / 2,

    )


# ============================================================================
# DataFrame -> Model Input
# ============================================================================

def dataframe_to_features(
    df,
    scaler,
):

    features = df.copy()

    features["Risk"] = (
        features["Risk_Min"]
        + features["Risk_Max"]
    ) / 2

    raw_features = features[
        FEATURE_COLUMNS
    ].values.astype(np.float32)

    scaled_features = scaler.transform(
        raw_features
    ).astype(np.float32)

    return scaled_features


# ============================================================================
# Generate Oracle Rewards
# ============================================================================

def generate_reward_targets(
    df,
    simulator,
    reward_engine,
):

    rewards = np.zeros(
        (len(df), len(ACTIONS)),
        dtype=np.float32
    )

    for i, (_, row) in enumerate(df.iterrows()):

        customer = row_to_customer_state(row)

        for action_index, amount in enumerate(ACTIONS):

            previous_customer = customer.copy()

            outcome = simulator.simulate(

                customer=customer.copy(),

                approved_unsecured_amount=float(
                    amount
                ),

            )

            reward = reward_engine.calculate_expected(

                previous_customer=previous_customer,

                outcome=outcome,

                approved_unsecured_amount=float(
                    amount
                ),

            )

            rewards[
                i,
                action_index
            ] = reward

    return rewards


# ============================================================================
# Convert Rewards -> Oracle Actions
# ============================================================================

def rewards_to_oracle_actions(
    rewards
):

    return np.argmax(
        rewards,
        axis=1
    ).astype(np.int64)


# ============================================================================
# Policy Classifier
# ============================================================================

class PolicyClassifier(nn.Module):

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
    ):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                input_dim,
                hidden_dim
            ),

            nn.ReLU(),

            nn.Linear(
                hidden_dim,
                hidden_dim
            ),

            nn.ReLU(),

            nn.Linear(
                hidden_dim,
                output_dim
            ),

        )

    def forward(self, x):

        return self.network(x)


# ============================================================================
# Evaluation
# ============================================================================

def evaluate(
    model,
    X,
    oracle_rewards,
    oracle_actions,
):

    model.eval()

    X_tensor = torch.from_numpy(
        X
    ).to(device)

    with torch.no_grad():

        logits = model(
            X_tensor
        )

        predicted_actions = torch.argmax(
            logits,
            dim=1
        ).cpu().numpy()

    # ------------------------------------------------------------------------
    # Reward obtained by predicted policy
    # ------------------------------------------------------------------------

    predicted_policy_rewards = (

        oracle_rewards[
            np.arange(
                len(oracle_rewards)
            ),
            predicted_actions
        ]

    )

    # ------------------------------------------------------------------------
    # Oracle reward
    # ------------------------------------------------------------------------

    best_rewards = np.max(
        oracle_rewards,
        axis=1
    )

    # ------------------------------------------------------------------------
    # Regret
    # ------------------------------------------------------------------------

    regret = (
        best_rewards
        - predicted_policy_rewards
    )

    # ------------------------------------------------------------------------
    # Exact optimality
    # ------------------------------------------------------------------------

    exact_optimality = np.mean(
        predicted_actions
        == oracle_actions
    )

    # ------------------------------------------------------------------------
    # Near optimality
    # ------------------------------------------------------------------------

    near_optimality = np.mean(
        regret <= TIE_TOLERANCE
    )

    return {

        "mean_regret": float(
            np.mean(regret)
        ),

        "median_regret": float(
            np.median(regret)
        ),

        "exact_optimality": float(
            exact_optimality
        ),

        "near_optimality": float(
            near_optimality
        ),

        "predicted_actions":
            predicted_actions,

        "regret":
            regret,

    }


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

    # =========================================================================
    # Load Dataset
    # =========================================================================

    print("=" * 70)

    print("Loading dataset...")

    print("=" * 70)

    df_raw = pd.read_csv(
        PATH
    )

    print(
        f"Total samples: {len(df_raw):,}"
    )

    # =========================================================================
    # Train / Validation / Test Split
    #
    # Same split as train.py
    #
    # 70% Train
    # 10% Validation
    # 20% Test
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
    # Fit ONLY on training data
    # =========================================================================

    train_features = train_raw.copy()

    train_features["Risk"] = (

        train_features["Risk_Min"]

        + train_features["Risk_Max"]

    ) / 2

    scaler = StandardScaler()

    scaler.fit(
        train_features[
            FEATURE_COLUMNS
        ]
    )

    print()

    print(
        "StandardScaler fitted on training data only."
    )

    # =========================================================================
    # Prepare Inputs
    # =========================================================================

    X_train = dataframe_to_features(
        train_raw,
        scaler
    )

    X_val = dataframe_to_features(
        val_raw,
        scaler
    )

    print()

    print(
        f"Training input shape   : {X_train.shape}"
    )

    print(
        f"Validation input shape : {X_val.shape}"
    )

    # =========================================================================
    # Create Simulator and Reward Engine
    # =========================================================================

    simulator = CustomerSimulator()

    reward_engine = RewardEngine()

    # =========================================================================
    # Generate Oracle Rewards
    # =========================================================================

    print()

    print("=" * 70)

    print("Generating training oracle rewards...")

    print("=" * 70)

    train_rewards = generate_reward_targets(

        train_raw,

        simulator,

        reward_engine

    )

    print()

    print("=" * 70)

    print("Generating validation oracle rewards...")

    print("=" * 70)

    val_rewards = generate_reward_targets(

        val_raw,

        simulator,

        reward_engine

    )

    # =========================================================================
    # Oracle Actions
    # =========================================================================

    train_oracle_actions = rewards_to_oracle_actions(
        train_rewards
    )

    val_oracle_actions = rewards_to_oracle_actions(
        val_rewards
    )

    print()

    print(
        "Training oracle actions generated."
    )

    print(
        "Validation oracle actions generated."
    )

    # =========================================================================
    # Print Training Oracle Distribution
    # =========================================================================

    print()

    print("=" * 70)

    print("TRAINING ORACLE ACTION DISTRIBUTION")

    print("=" * 70)

    for action_index, amount in enumerate(ACTIONS):

        count = np.sum(
            train_oracle_actions
            == action_index
        )

        percentage = (
            count
            / len(train_oracle_actions)
        )

        if amount == 0:

            label = "0"

        else:

            label = f"{amount / 100_000_000:.0f}00M"

        print(
            f"{label:>6} : "
            f"{count:6d} "
            f"({percentage:6.2%})"
        )

    # =========================================================================
    # Tensor Dataset
    # =========================================================================

    X_train_tensor = torch.from_numpy(
        X_train
    )

    y_train_tensor = torch.from_numpy(
        train_oracle_actions
    )

    train_dataset = TensorDataset(

        X_train_tensor,

        y_train_tensor

    )

    train_loader = DataLoader(

        train_dataset,

        batch_size=BATCH_SIZE,

        shuffle=True

    )

    # =========================================================================
    # Create Model
    # =========================================================================

    print()

    print("=" * 70)

    print("Creating Policy Classifier...")

    print("=" * 70)

    model = PolicyClassifier(

        input_dim=X_train.shape[1],

        hidden_dim=HIDDEN_DIM,

        output_dim=len(ACTIONS),

    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(

        model.parameters(),

        lr=LEARNING_RATE

    )

    # =========================================================================
    # Training
    # =========================================================================

    print()

    print("=" * 70)

    print("Starting training...")

    print("=" * 70)

    best_val_regret = float("inf")

    best_state_dict = None

    epochs_without_improvement = 0

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        model.train()

        total_loss = 0.0

        total_samples = 0

        for batch_x, batch_y in train_loader:

            batch_x = batch_x.to(
                device
            )

            batch_y = batch_y.to(
                device
            )

            optimizer.zero_grad()

            logits = model(
                batch_x
            )

            loss = criterion(
                logits,
                batch_y
            )

            loss.backward()

            optimizer.step()

            batch_size = batch_x.shape[0]

            total_loss += (
                loss.item()
                * batch_size
            )

            total_samples += batch_size

        train_loss = (
            total_loss
            / total_samples
        )

        # ---------------------------------------------------------------------
        # Validation
        # ---------------------------------------------------------------------

        validation_metrics = evaluate(

            model,

            X_val,

            val_rewards,

            val_oracle_actions

        )

        print(

            f"Epoch {epoch:03d} | "

            f"Train CE: "
            f"{train_loss:.6f} | "

            f"Regret: "
            f"{validation_metrics['mean_regret']:.6f} | "

            f"Exact: "
            f"{validation_metrics['exact_optimality']:.2%} | "

            f"Near: "
            f"{validation_metrics['near_optimality']:.2%}"

        )

        # ---------------------------------------------------------------------
        # Save Best Model
        # ---------------------------------------------------------------------

        if (
            validation_metrics["mean_regret"]
            < best_val_regret
        ):

            best_val_regret = (
                validation_metrics[
                    "mean_regret"
                ]
            )

            best_state_dict = {

                key:
                    value.detach()
                    .cpu()
                    .clone()

                for key, value
                in model.state_dict().items()

            }

            epochs_without_improvement = 0

        else:

            epochs_without_improvement += 1

        # ---------------------------------------------------------------------
        # Early Stopping
        # ---------------------------------------------------------------------

        if (
            epochs_without_improvement
            >= PATIENCE
        ):

            print()

            print(
                "Early stopping."
            )

            break

    # =========================================================================
    # Restore Best Model
    # =========================================================================

    if best_state_dict is not None:

        model.load_state_dict(
            best_state_dict
        )

    # =========================================================================
    # Save Model
    # =========================================================================

    torch.save(

        {

            "model_state_dict":
                model.state_dict(),

            "input_dim":
                X_train.shape[1],

            "hidden_dim":
                HIDDEN_DIM,

            "output_dim":
                len(ACTIONS),

            "actions":
                ACTIONS,

        },

        MODEL_PATH

    )

    # =========================================================================
    # Final Validation
    # =========================================================================

    final_metrics = evaluate(

        model,

        X_val,

        val_rewards,

        val_oracle_actions

    )

    print()

    print("=" * 70)

    print("FINAL VALIDATION")

    print("=" * 70)

    print(
        f"Mean regret        : "
        f"{final_metrics['mean_regret']:.6f}"
    )

    print(
        f"Median regret      : "
        f"{final_metrics['median_regret']:.6f}"
    )

    print(
        f"Exact optimality   : "
        f"{final_metrics['exact_optimality']:.2%}"
    )

    print(
        f"Near optimality    : "
        f"{final_metrics['near_optimality']:.2%}"
    )

    print()

    print(
        f"Model saved to:"
        f"\n{MODEL_PATH}"
    )


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":

    main()