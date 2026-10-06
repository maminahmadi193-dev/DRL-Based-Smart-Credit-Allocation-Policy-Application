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

PATH = r"C:\\Users\\m.ahmadi\\Desktop\\FinalProject\\notebooks\\df_cleaned.csv"

RANDOM_STATE = 42

EPOCHS = 100
BATCH_SIZE = 256
LEARNING_RATE = 4e-3
PATIENCE = 10

HIDDEN_DIM = 128

MODEL_PATH = "models/q_regression.pt"
SCALER_PATH = "models/q_regression_scaler.pkl"

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
# Convert DataFrame -> Model Input
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
# Generate Reward Targets
# ============================================================================

def generate_reward_targets(
    df,
    simulator,
    reward_engine,
):

    targets = np.zeros(
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

            targets[
                i,
                action_index
            ] = reward

    return targets


# ============================================================================
# Q Regression Network
# ============================================================================

class QRegressionNetwork(nn.Module):

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
    true_rewards,
):

    model.eval()

    X_tensor = torch.from_numpy(
        X
    ).to(device)

    with torch.no_grad():

        predicted_rewards = (
            model(X_tensor)
            .cpu()
            .numpy()
        )

    # ------------------------------------------------------------------------
    # Predicted policy
    # ------------------------------------------------------------------------

    predicted_actions = np.argmax(
        predicted_rewards,
        axis=1
    )

    # ------------------------------------------------------------------------
    # Oracle policy
    # ------------------------------------------------------------------------

    oracle_actions = np.argmax(
        true_rewards,
        axis=1
    )

    # ------------------------------------------------------------------------
    # Oracle reward
    # ------------------------------------------------------------------------

    oracle_rewards = np.max(
        true_rewards,
        axis=1
    )

    # ------------------------------------------------------------------------
    # Reward obtained by predicted policy
    # ------------------------------------------------------------------------

    predicted_policy_rewards = (

        true_rewards[
            np.arange(
                len(true_rewards)
            ),
            predicted_actions
        ]

    )

    # ------------------------------------------------------------------------
    # Regret
    # ------------------------------------------------------------------------

    regret = (
        oracle_rewards
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

    # ------------------------------------------------------------------------
    # Q regression error
    # ------------------------------------------------------------------------

    mse = np.mean(
        (
            predicted_rewards
            - true_rewards
        ) ** 2
    )

    return {

        "mse": float(mse),

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

        "predicted_rewards":
            predicted_rewards,

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
    # 70% Train
    # 10% Validation
    # 20% Test
    #
    # Exact same split logic as train.py
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
    # Prepare Model Inputs
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
    #
    # Same simulator and reward logic currently used by LoanEnv.
    # =========================================================================

    simulator = CustomerSimulator()

    reward_engine = RewardEngine()

    # =========================================================================
    # Generate Training Targets
    #
    # For each customer:
    #
    # [R(s,0), R(s,100M), ..., R(s,500M)]
    #
    # =========================================================================

    print()

    print("=" * 70)

    print("Generating training reward targets...")

    print("=" * 70)

    Y_train = generate_reward_targets(

        train_raw,

        simulator,

        reward_engine

    )

    # =========================================================================
    # Generate Validation Targets
    # =========================================================================

    print()

    print("=" * 70)

    print("Generating validation reward targets...")

    print("=" * 70)

    Y_val = generate_reward_targets(

        val_raw,

        simulator,

        reward_engine

    )

    print()

    print(
        f"Training target shape   : {Y_train.shape}"
    )

    print(
        f"Validation target shape : {Y_val.shape}"
    )

    # =========================================================================
    # Save Scaler
    # =========================================================================

    import joblib

    joblib.dump(
        scaler,
        SCALER_PATH
    )

    # =========================================================================
    # Tensor Dataset
    # =========================================================================

    X_train_tensor = torch.from_numpy(
        X_train
    )

    Y_train_tensor = torch.from_numpy(
        Y_train
    )

    train_dataset = TensorDataset(

        X_train_tensor,

        Y_train_tensor

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

    print("Creating Q Regression model...")

    print("=" * 70)

    model = QRegressionNetwork(

        input_dim=X_train.shape[1],

        hidden_dim=HIDDEN_DIM,

        output_dim=len(ACTIONS),

    ).to(device)

    criterion = nn.MSELoss()

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

            predictions = model(
                batch_x
            )

            loss = criterion(
                predictions,
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

            Y_val

        )

        print(

            f"Epoch {epoch:03d} | "

            f"Train MSE: "
            f"{train_loss:.6f} | "

            f"Val MSE: "
            f"{validation_metrics['mse']:.6f} | "

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
    # Save Final Model
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

        Y_val

    )

    print()

    print("=" * 70)

    print("FINAL VALIDATION")

    print("=" * 70)

    print(
        f"Mean MSE           : "
        f"{final_metrics['mse']:.6f}"
    )

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

    print(
        f"\nScaler saved to:"
        f"\n{SCALER_PATH}"
    )


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":

    main()