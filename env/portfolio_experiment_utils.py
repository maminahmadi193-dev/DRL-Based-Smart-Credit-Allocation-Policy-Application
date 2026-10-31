"""Shared, leak-free data preparation for portfolio DQN experiments."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from customer_simulator import CustomerSimulator


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
    "Has_No_History",
    "Employed",
    "Score_Min",
    "Score_Max",
    "Risk_Min",
    "Risk_Max",
]


def normalize_customer_financial_features(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize financial source fields and rebuild derived debt features.

    This matches PortfolioLoanEnv._row_to_customer_state so scaler fitting,
    risk calibration, and runtime observations use identical values.
    """
    required = ["DebtAmount", "OriginAmount", "LoanCounts"]
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(
            f"Customer financial normalization is missing columns: {missing}"
        )

    normalized = frame.copy()
    debt = np.maximum(
        normalized["DebtAmount"].to_numpy(dtype=float),
        0.0,
    )
    origin_amount = np.maximum(
        normalized["OriginAmount"].to_numpy(dtype=float),
        0.0,
    )
    loan_counts = np.maximum(
        np.trunc(normalized["LoanCounts"].to_numpy(dtype=float)),
        0.0,
    ).astype(np.int64)

    normalized["DebtAmount"] = debt
    normalized["OriginAmount"] = origin_amount
    normalized["LoanCounts"] = loan_counts

    if "TotalFacilities" in normalized.columns:
        normalized["TotalFacilities"] = np.maximum(
            normalized["TotalFacilities"].to_numpy(dtype=float),
            0.0,
        )

    normalized["DebtRatio"] = np.divide(
        debt,
        origin_amount,
        out=np.zeros_like(debt, dtype=float),
        where=origin_amount > 0.0,
    )
    normalized["DebtPerLoan"] = np.divide(
        debt,
        loan_counts,
        out=np.zeros_like(debt, dtype=float),
        where=loan_counts > 0,
    )
    return normalized


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
    missing_income_rows = int(df["MountlyIncome"].isna().sum())
    # The source represents applicants without a recorded monthly salary as
    # blank; business semantics for this project treat those as zero income.
    df["MountlyIncome"] = df["MountlyIncome"].fillna(0.0)
    non_income_required = [
        column for column in REQUIRED_RAW_COLUMNS
        if column != "MountlyIncome"
    ]
    df = df.dropna(subset=non_income_required + ["Risk"]).reset_index(drop=True)

    if df.empty:
        raise ValueError(f"No complete usable rows found in {source_path}")

    # Rebuild debt-derived columns before splitting and fitting the scaler.
    # Runtime customer states use these same normalized financial values.
    df = normalize_customer_financial_features(df)

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
        "missing_income_imputed_to_zero_rows": missing_income_rows,
        "split_seed": int(split_seed),
        "train_rows": int(len(train_raw)),
        "validation_rows": int(len(validation_raw)),
        "test_rows": int(len(test_raw)),
        "feature_columns": list(FEATURE_COLUMNS),
    }

    return train_raw, validation_raw, test_raw, scaler, split_info



def reference_risk_scores_after_loan(
    raw_df: pd.DataFrame,
    reference_loan_amount: float = 100_000_000.0,
) -> np.ndarray:
    """
    Estimate each row's risk score after a hypothetical reference loan.

    This mirrors CustomerSimulator's current state and risk updates in a
    vectorized way, using no observed default labels. It is used only to define
    the exposure against which the population default-rate calibration is made.
    """
    if not np.isfinite(reference_loan_amount) or reference_loan_amount <= 0.0:
        raise ValueError("reference_loan_amount must be finite and positive.")

    required = [
        "MountlyIncome", "DebtAmount", "OriginAmount", "LoanCounts",
        "Risk_Min", "Risk_Max",
    ]
    missing = sorted(set(required) - set(raw_df.columns))
    if missing:
        raise ValueError(f"Reference-risk calculation is missing columns: {missing}")

    amount = float(reference_loan_amount)
    risk_simulator = CustomerSimulator()
    normalized = normalize_customer_financial_features(raw_df)
    income = normalized["MountlyIncome"].to_numpy(dtype=float)
    debt = normalized["DebtAmount"].to_numpy(dtype=float)
    origin_amount = normalized["OriginAmount"].to_numpy(dtype=float)
    loan_counts = normalized["LoanCounts"].to_numpy(dtype=np.int64)
    previous_debt_ratio = normalized["DebtRatio"].to_numpy(dtype=float)
    previous_debt_per_loan = normalized["DebtPerLoan"].to_numpy(dtype=float)
    initial_risk = (
        raw_df["Risk_Min"].to_numpy(dtype=float)
        + raw_df["Risk_Max"].to_numpy(dtype=float)
    ) / 2.0

    updated_debt = debt + amount
    updated_origin_amount = origin_amount + amount
    updated_loan_counts = loan_counts + 1
    updated_debt_ratio = np.divide(
        updated_debt,
        updated_origin_amount,
        out=np.zeros_like(updated_debt, dtype=float),
        where=updated_origin_amount > 0.0,
    )
    updated_debt_per_loan = np.divide(
        updated_debt,
        updated_loan_counts,
        out=np.zeros_like(updated_debt, dtype=float),
        where=updated_loan_counts > 0,
    )

    # Mirror CustomerSimulator._update_risk exactly: only positive
    # debt-ratio and relative debt-per-loan changes increase risk.
    debt_ratio_change = np.maximum(
        0.0,
        updated_debt_ratio - previous_debt_ratio,
    )
    debt_per_loan_change = np.zeros_like(updated_debt_per_loan)
    has_previous_average = previous_debt_per_loan > 0.0
    debt_per_loan_change[has_previous_average] = np.maximum(
        0.0,
        (
            updated_debt_per_loan[has_previous_average]
            - previous_debt_per_loan[has_previous_average]
        ) / previous_debt_per_loan[has_previous_average],
    )
    debt_per_loan_change[~has_previous_average] = np.clip(
        updated_debt_per_loan[~has_previous_average] / 500_000_000.0,
        0.0,
        1.0,
    )
    capacity_pressure = np.ones_like(income)
    has_income = income > 0.0
    capacity_pressure[has_income] = amount / (income[has_income] + amount)

    delta_risk = np.maximum(
        0.0,
        risk_simulator.alpha * debt_ratio_change
        + risk_simulator.beta * debt_per_loan_change
        + risk_simulator.gamma * capacity_pressure,
    )
    updated_risk = initial_risk + delta_risk
    return np.clip(updated_risk, 0.0, 100.0)


def calibrate_hazard_intercept_for_population(
    raw_df: pd.DataFrame,
    target_annual_default_rate: float = 0.013,
    reference_loan_amount: float = 100_000_000.0,
    calibration_horizon_months: int = 12,
    max_hazard: float = 0.005,
    hazard_slope: float = 0.08,
    hazard_center: float = 60.0,
) -> dict:
    """
    Calibrate the hazard logistic intercept against a population and a
    transparent reference exposure. No applicant-level default labels are used.
    """
    reference_risks = reference_risk_scores_after_loan(
        raw_df,
        reference_loan_amount=reference_loan_amount,
    )
    intercept = CustomerSimulator.calibrate_hazard_intercept(
        reference_risks=reference_risks,
        target_term_default_rate=target_annual_default_rate,
        max_hazard=max_hazard,
        hazard_slope=hazard_slope,
        hazard_center=hazard_center,
        horizon_months=calibration_horizon_months,
    )
    z = np.clip(
        hazard_slope * (np.clip(reference_risks, 0.0, 100.0) - hazard_center)
        + intercept,
        -60.0,
        60.0,
    )
    monthly_hazards = max_hazard / (1.0 + np.exp(-z))
    annual_pds = -np.expm1(-monthly_hazards * int(calibration_horizon_months))
    return {
        "method": "logistic_intercept_mean_annual_pd_calibration",
        "target_annual_default_rate": float(target_annual_default_rate),
        "calibrated_reference_annual_default_rate": float(np.mean(annual_pds)),
        "hazard_intercept": float(intercept),
        "reference_loan_amount": float(reference_loan_amount),
        "calibration_horizon_months": int(calibration_horizon_months),
        "reference_population_rows": int(len(reference_risks)),
        "risk_assumption": "post-loan risk under the reference loan amount",
        "max_hazard_per_month": float(max_hazard),
        "hazard_slope": float(hazard_slope),
        "hazard_center": float(hazard_center),
    }
