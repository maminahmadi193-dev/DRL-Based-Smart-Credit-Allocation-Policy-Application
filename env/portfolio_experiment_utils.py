"""Shared, leak-free data preparation for portfolio DQN experiments."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


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

REQUIRED_RAW_COLUMNS = [
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
]


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_dataset_path(data_path=None) -> Path:
    root = project_root()

    if data_path:
        path = Path(data_path).expanduser()
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Dataset not found: {path}")
        return path.resolve()

    candidates = [
        root / "notebooks" / "df_cleaned.csv",
        root / "df_cleaned.csv",
        Path.cwd() / "notebooks" / "df_cleaned.csv",
        Path.cwd() / "df_cleaned.csv",
    ]
    for path in candidates:
        if path.is_file():
            return path.resolve()

    raise FileNotFoundError(
        "Could not find notebooks/df_cleaned.csv. Pass --data PATH explicitly."
    )


def load_split_and_scaler(data_path=None, split_seed=42):
    """Load real cleaned data; split 70/10/20 and fit scaler on train only."""

    source_path = resolve_dataset_path(data_path)
    df = pd.read_csv(source_path)

    missing = sorted(set(REQUIRED_RAW_COLUMNS) - set(df.columns))
    if missing:
        raise ValueError(
            f"Dataset {source_path} is missing required columns: {missing}"
        )

    for column in REQUIRED_RAW_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["Risk"] = (df["Risk_Min"] + df["Risk_Max"]) / 2.0
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=REQUIRED_RAW_COLUMNS + ["Risk"]).reset_index(drop=True)

    if df.empty:
        raise ValueError(f"No complete usable rows found in {source_path}")

    all_indices = np.arange(len(df))
    train_idx, temporary_idx = train_test_split(
        all_indices,
        test_size=0.30,
        random_state=split_seed,
        shuffle=True,
    )
    validation_idx, test_idx = train_test_split(
        temporary_idx,
        test_size=2 / 3,
        random_state=split_seed,
        shuffle=True,
    )

    train_raw = df.iloc[train_idx].reset_index(drop=True)
    validation_raw = df.iloc[validation_idx].reset_index(drop=True)
    test_raw = df.iloc[test_idx].reset_index(drop=True)

    # Fit preprocessing on training rows only to prevent data leakage.
    scaler = StandardScaler()
    scaler.fit(train_raw[FEATURE_COLUMNS])

    split_info = {
        "dataset_path": str(source_path),
        "dataset_rows_used": int(len(df)),
        "split_seed": int(split_seed),
        "train_rows": int(len(train_raw)),
        "validation_rows": int(len(validation_raw)),
        "test_rows": int(len(test_raw)),
        "feature_columns": list(FEATURE_COLUMNS),
    }

    return train_raw, validation_raw, test_raw, scaler, split_info
