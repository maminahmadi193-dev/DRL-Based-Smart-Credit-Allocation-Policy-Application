"""Decompose the social reward used by PortfolioLoanEnv.

Run from the repository root:
    python env/analyze_social_reward_components.py --require-real-data

This is a diagnostic only: it does not train DQN or modify RewardEngine.
It calls RewardEngine's actual component methods and checks that their sum
matches RewardEngine._social_reward for every sampled row/action.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from customer_state import CustomerState
from reward_engine import RewardEngine


ACTION_AMOUNTS = np.array(
    [0, 100_000_000, 200_000_000, 300_000_000, 400_000_000, 500_000_000],
    dtype=np.int64,
)

REQUIRED_COLUMNS = [
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

COMPONENTS = ("history", "facility", "need", "social")


def find_default_dataset(project_root: Path) -> Path | None:
    env_path = os.environ.get("PORTFOLIO_BASELINE_DATA")
    candidates = []
    if env_path:
        candidates.append(Path(env_path).expanduser())
    candidates.extend([
        project_root / "df_cleaned.csv",
        project_root / "notebooks" / "df_cleaned.csv",
        project_root.parent / "notebooks" / "df_cleaned.csv",
        Path.cwd() / "df_cleaned.csv",
        Path.cwd() / "notebooks" / "df_cleaned.csv",
    ])

    seen = set()
    for path in candidates:
        try:
            resolved = path.resolve()
        except OSError:
            resolved = path
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.is_file():
            return resolved
    return None


def load_data(data_path: str | None, require_real_data: bool, project_root: Path):
    source = Path(data_path).expanduser() if data_path else None
    if source is None:
        source = find_default_dataset(project_root)
    elif not source.is_absolute():
        source = (Path.cwd() / source).resolve()

    if source is None or not source.is_file():
        if require_real_data:
            raise FileNotFoundError(
                "Could not find df_cleaned.csv. Pass --data PATH or set "
                "PORTFOLIO_BASELINE_DATA."
            )
        raise FileNotFoundError(
            "No dataset was found. This diagnostic intentionally requires the "
            "real dataset unless an explicit dataset path is supplied."
        )

    df = pd.read_csv(source)
    missing = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    for col in REQUIRED_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=REQUIRED_COLUMNS).reset_index(drop=True)
    if df.empty:
        raise ValueError(f"No usable rows remain after cleaning {source}")
    return df, source.resolve()


def customer_from_row(row) -> CustomerState:
    return CustomerState(
        monthly_income=float(row["MountlyIncome"]),
        income_score=float(row["IncomeScore"]),
        job_score=float(row["JobScore"]),
        asset_score=float(row["AssetScore"]),
        total_customer_score=float(row["TotalCustomerScore"]),
        total_facilities=float(row["TotalFacilities"]),
        debt=float(row["DebtAmount"]),
        origin_amount=float(row["OriginAmount"]),
        loan_count=int(row["LoanCounts"]),
        debt_ratio=float(row["DebtRatio"]),
        debt_per_loan=float(row["DebtPerLoan"]),
        has_no_history=bool(float(row["Has_No_History"]) >= 0.5),
        employed=bool(float(row["Employed"]) >= 0.5),
        score_min=float(row["Score_Min"]),
        score_max=float(row["Score_Max"]),
        risk_min=float(row["Risk_Min"]),
        risk_max=float(row["Risk_Max"]),
        risk=(float(row["Risk_Min"]) + float(row["Risk_Max"])) / 2.0,
    )


def calculate_components(df: pd.DataFrame, engine: RewardEngine):
    n = len(df)
    shape = (n, len(ACTION_AMOUNTS))
    history = np.zeros(shape, dtype=np.float64)
    facility = np.zeros(shape, dtype=np.float64)
    need = np.zeros(shape, dtype=np.float64)
    social = np.zeros(shape, dtype=np.float64)

    print("\nCalculating actual RewardEngine components...")
    rows = df.to_dict(orient="records")
    for i, row in enumerate(rows):
        customer = customer_from_row(row)

        # The environment explicitly returns zero social reward for action 0.
        for j, amount in enumerate(ACTION_AMOUNTS):
            if amount == 0:
                continue

            h = float(engine._history_reward(customer))
            f = float(engine._facility_reward(customer))
            nd = float(engine._need_reward(customer, float(amount)))
            total = float(engine._social_reward(customer, float(amount)))

            if not np.isclose(h + f + nd, total, rtol=1e-10, atol=1e-10):
                raise AssertionError(
                    f"Reward decomposition mismatch at row={i}, amount={amount}: "
                    f"components={h + f + nd}, social={total}"
                )

            history[i, j] = h
            facility[i, j] = f
            need[i, j] = nd
            social[i, j] = total

        if (i + 1) % 20_000 == 0 or i + 1 == n:
            print(f"  Processed {i + 1:,}/{n:,} rows")

    arrays = {
        "history": history,
        "facility": facility,
        "need": need,
        "social": social,
    }
    return arrays


def stats_for_slice(arrays: dict[str, np.ndarray], mask: np.ndarray, action_index: int,
                    engine: RewardEngine) -> dict:
    result = {"customers": int(mask.sum()), "action_amount": int(ACTION_AMOUNTS[action_index])}
    for component in COMPONENTS:
        values = arrays[component][mask, action_index]
        result[f"mean_{component}_reward"] = float(np.mean(values)) if len(values) else np.nan
        result[f"median_{component}_reward"] = float(np.median(values)) if len(values) else np.nan
        result[f"std_{component}_reward"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0

    social_mean = result["mean_social_reward"]
    for component in ("history", "facility", "need"):
        result[f"{component}_share_of_mean_social"] = (
            result[f"mean_{component}_reward"] / social_mean
            if social_mean > 0 else np.nan
        )

    need_values = arrays["need"][mask, action_index]
    if len(need_values) and ACTION_AMOUNTS[action_index] > 0 and engine.need_weight > 0:
        result["pct_need_reward_saturated"] = float(
            np.mean(need_values >= engine.need_weight - 1e-10)
        )
    else:
        result["pct_need_reward_saturated"] = np.nan

    # For nonzero actions, show how much need/social reward is gained above a 100M loan.
    if ACTION_AMOUNTS[action_index] > 0 and len(need_values):
        need_delta = (
            arrays["need"][mask, action_index]
            - arrays["need"][mask, 1]
        )
        social_delta = (
            arrays["social"][mask, action_index]
            - arrays["social"][mask, 1]
        )
        result["mean_need_increment_vs_100M"] = float(np.mean(need_delta))
        result["pct_customers_need_reward_increases_vs_100M"] = float(
            np.mean(need_delta > 1e-10)
        )
        result["mean_social_increment_vs_100M"] = float(np.mean(social_delta))
    else:
        result["mean_need_increment_vs_100M"] = 0.0 if ACTION_AMOUNTS[action_index] == 100_000_000 else np.nan
        result["pct_customers_need_reward_increases_vs_100M"] = 0.0 if ACTION_AMOUNTS[action_index] == 100_000_000 else np.nan
        result["mean_social_increment_vs_100M"] = 0.0 if ACTION_AMOUNTS[action_index] == 100_000_000 else np.nan
    return result


def build_action_summary(arrays: dict[str, np.ndarray], engine: RewardEngine) -> pd.DataFrame:
    all_rows = []
    all_customers = np.ones(arrays["social"].shape[0], dtype=bool)
    for j in range(len(ACTION_AMOUNTS)):
        all_rows.append(stats_for_slice(arrays, all_customers, j, engine))
    return pd.DataFrame(all_rows)


def build_group_summary(
    groups: np.ndarray,
    group_field: str,
    arrays: dict[str, np.ndarray],
    engine: RewardEngine,
) -> pd.DataFrame:
    results = []
    for group in pd.unique(groups):
        mask = groups == group
        for j in range(len(ACTION_AMOUNTS)):
            row = {group_field: str(group)}
            row.update(stats_for_slice(arrays, mask, j, engine))
            results.append(row)
    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(
        description="Decompose RewardEngine's social reward by loan amount and customer profile."
    )
    parser.add_argument("--data", default=None, help="Path to df_cleaned.csv; autodetected if omitted.")
    parser.add_argument("--require-real-data", action="store_true",
                        help="Fail explicitly if the real dataset is not found.")
    parser.add_argument("--output-dir", default=None,
                        help="Output folder; defaults to models/results.")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    df, source_path = load_data(args.data, args.require_real_data, project_root)
    engine = RewardEngine()

    output_dir = Path(args.output_dir) if args.output_dir else project_root / "models" / "results"
    if not output_dir.is_absolute():
        output_dir = (Path.cwd() / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    arrays = calculate_components(df, engine)
    action_summary = build_action_summary(arrays, engine)

    clipped_score = df["TotalCustomerScore"].clip(0, 100)
    score_bands = pd.cut(
        clipped_score,
        bins=[-0.001, 20, 40, 60, 80, 100],
        labels=["0-20", "20-40", "40-60", "60-80", "80-100"],
        include_lowest=True,
    ).astype(str).to_numpy()

    has_history = np.where(
        df["Has_No_History"].to_numpy(dtype=float) >= 0.5,
        "no_history",
        "has_history",
    )

    facilities_rank = df["TotalFacilities"].rank(method="first", pct=True).to_numpy()
    facility_bands = np.select(
        [
            facilities_rank <= 0.25,
            facilities_rank <= 0.50,
            facilities_rank <= 0.75,
        ],
        ["Q1_lowest_facilities", "Q2", "Q3"],
        default="Q4_highest_facilities",
    )

    by_score = build_group_summary(score_bands, "score_band", arrays, engine)
    by_history = build_group_summary(has_history, "history_group", arrays, engine)
    by_facilities = build_group_summary(facility_bands, "facility_quartile", arrays, engine)

    outputs = {
        "social_reward_components_by_action.csv": action_summary,
        "social_reward_components_by_score_band.csv": by_score,
        "social_reward_components_by_history.csv": by_history,
        "social_reward_components_by_facility_quartile.csv": by_facilities,
    }
    for filename, frame in outputs.items():
        frame.to_csv(output_dir / filename, index=False)

    config = {
        "dataset": str(source_path),
        "dataset_rows_used": int(len(df)),
        "action_amounts": ACTION_AMOUNTS.tolist(),
        "reward_engine_parameters": {
            "history_reward": engine.history_reward,
            "facility_weight": engine.facility_weight,
            "need_weight": engine.need_weight,
            "risk_weight": engine.risk_weight,
            "default_cost": engine.default_cost,
            "facility_decay": engine.facility_decay,
            "need_decay": engine.need_decay,
        },
        "note": (
            "Only social reward components are decomposed. In PortfolioLoanEnv, "
            "the current reward path calls _social_reward and separately adds "
            "portfolio financial PnL; RewardEngine.calculate/calculate_expected "
            "are not used by that path."
        ),
        "outputs": list(outputs.keys()),
    }
    with (output_dir / "social_reward_components_config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    display_columns = [
        "action_amount", "mean_history_reward", "mean_facility_reward",
        "mean_need_reward", "mean_social_reward", "history_share_of_mean_social",
        "facility_share_of_mean_social", "need_share_of_mean_social",
        "pct_need_reward_saturated", "mean_need_increment_vs_100M",
        "pct_customers_need_reward_increases_vs_100M",
        "mean_social_increment_vs_100M",
    ]

    print("\n" + "=" * 100)
    print("SOCIAL REWARD COMPONENTS BY LOAN AMOUNT")
    print("=" * 100)
    print(action_summary[display_columns].to_string(
        index=False, float_format=lambda x: f"{x:,.4f}"
    ))

    print("\nSocial reward by score band:")
    print(by_score[[
        "score_band", "action_amount", "customers",
        "mean_history_reward", "mean_facility_reward",
        "mean_need_reward", "mean_social_reward",
        "pct_need_reward_saturated",
    ]].to_string(index=False, float_format=lambda x: f"{x:,.4f}"))

    print("\nSocial reward by credit-history group:")
    print(by_history[[
        "history_group", "action_amount", "customers",
        "mean_history_reward", "mean_facility_reward",
        "mean_need_reward", "mean_social_reward",
    ]].to_string(index=False, float_format=lambda x: f"{x:,.4f}"))

    print("\nSaved diagnostic outputs to:")
    print(output_dir.resolve())
    for filename in outputs:
        print(f"  {filename}")
    print("  social_reward_components_config.json")
    print("\nNo training was run and no reward formula was changed.")


if __name__ == "__main__":
    main()
