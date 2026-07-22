from stable_baselines3 import DQN
import pandas as pd
import numpy as np
from .loan_env import LoanEnv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


PATH = r"C:\\Users\\m.ahmadi\\Desktop\\FinalProject\\notebooks\\df_cleaned.csv"

def main():

    # ============================================
    # Load Dataset
    # ============================================

    df_raw = pd.read_csv(PATH)

    train_idx, test_idx = train_test_split(
        np.arange(len(df_raw)),
        test_size = 0.2 ,
        random_state = 42 ,
        shuffle = True
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
    scaler.fit(train_features[FEATURE_COLUMNS])


    # ============================================
    # Environment
    # ============================================

    train_env = LoanEnv(train_raw, scaler)
    test_env = LoanEnv(test_raw, scaler)

    # ============================================
    # Model
    # ============================================

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

    # ============================================
    # Train
    # ============================================

    model.learn(
        total_timesteps=100000
    )

    # ============================================
    # Save
    # ============================================

    model.save("models/dqn_loan2")


if __name__ == "__main__":
    main()