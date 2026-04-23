from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    return df


def load_event_study_data(path: str) -> pd.DataFrame:
    """
    Expected minimum columns in Excel:
    - Company
    - Date
    - Stock_Price
    - Market_Price

    Optional columns:
    - Event_Day
    - Event_Date
    - Stock_Return
    - Market_Return
    - Alpha
    - Beta
    - Expected_Return
    - AR
    - CAR
    """
    df = pd.read_excel(path)
    df = _standardize_columns(df)

    # Flexible column mapping
    rename_map = {}
    if "company_name" in df.columns and "company" not in df.columns:
        rename_map["company_name"] = "company"
    if "stockprice" in df.columns and "stock_price" not in df.columns:
        rename_map["stockprice"] = "stock_price"
    if "marketprice" in df.columns and "market_price" not in df.columns:
        rename_map["marketprice"] = "market_price"
    if "eventday" in df.columns and "event_day" not in df.columns:
        rename_map["eventday"] = "event_day"
    if "eventdate" in df.columns and "event_date" not in df.columns:
        rename_map["eventdate"] = "event_date"

    if rename_map:
        df = df.rename(columns=rename_map)

    required = {"company", "date", "stock_price", "market_price"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values(["company", "date"]).reset_index(drop=True)

    # Force numeric
    for col in ["stock_price", "market_price", "event_day"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def fill_missing_prices(df: pd.DataFrame) -> pd.DataFrame:
    """
    If price values are missing, fill using interpolation within each company.
    This should be used carefully; keep the original data too.
    """
    df = df.copy()
    for col in ["stock_price", "market_price"]:
        if col in df.columns:
            df[col] = df.groupby("company")[col].transform(
                lambda s: s.interpolate(method="linear", limit_direction="both")
            )
            df[col] = df.groupby("company")[col].transform(
                lambda s: s.fillna(method="ffill").fillna(method="bfill")
            )
    return df


def calculate_returns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["company", "date"]).reset_index(drop=True)

    df["stock_return"] = df.groupby("company")["stock_price"].pct_change()
    df["market_return"] = df.groupby("company")["market_price"].pct_change()

    # 🔥 CLEAN RETURNS HERE
    import numpy as np
    df["stock_return"] = df["stock_return"].replace([np.inf, -np.inf], np.nan)
    df["market_return"] = df["market_return"].replace([np.inf, -np.inf], np.nan)

    return df


def assign_event_day(
    df: pd.DataFrame,
    event_dates: dict[str, str | pd.Timestamp],
) -> pd.DataFrame:
    """
    event_dates example:
    {
        "Infosys": "2025-04-21",
        "TCS": "2025-04-11"
    }

    The exact event day row gets 0, before becomes negative, after becomes positive,
    based on trading-day index around the first available event date in the sheet.
    """
    df = df.copy()
    df["event_day"] = np.nan

    for company, event_date in event_dates.items():
        event_date = pd.to_datetime(event_date)
        mask = df["company"].eq(company)
        company_df = df.loc[mask].sort_values("date").copy()

        # Find the first trading date on or after the event date
        available_dates = company_df["date"].dropna().sort_values().tolist()
        if not available_dates:
            continue

        if event_date in available_dates:
            event_idx = available_dates.index(event_date)
        else:
            future_dates = [d for d in available_dates if d >= event_date]
            if future_dates:
                event_date = future_dates[0]
                event_idx = available_dates.index(event_date)
            else:
                # fallback: use last available date
                event_date = available_dates[-1]
                event_idx = len(available_dates) - 1

        company_df = company_df.reset_index(drop=True)
        company_df["event_day"] = np.arange(len(company_df)) - event_idx
        df.loc[company_df.index, "event_day"] = company_df["event_day"].values

        # assign back by date/company properly
        df.loc[mask, "event_day"] = df.loc[mask].sort_values("date").assign(
            event_day=np.arange(len(company_df)) - event_idx
        )["event_day"].values

    return df


def estimate_market_model(
    df: pd.DataFrame,
    estimation_start: int = -120,
    estimation_end: int = -20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fits alpha and beta per company using rows in estimation window.
    Returns:
      - results_df with alpha/beta per company
      - df with expected_return, ar, car for event window rows
    """
    df = df.copy()
    if "event_day" not in df.columns:
        raise ValueError("event_day column not found. Assign event day first.")

    results = []
    df["alpha"] = np.nan
    df["beta"] = np.nan
    df["expected_return"] = np.nan
    df["ar"] = np.nan
    df["car"] = np.nan

    for company, g in df.groupby("company", sort=False):
        g = g.sort_values("date").copy()
        est = g[(g["event_day"] >= estimation_start) & (g["event_day"] <= estimation_end)].copy()

        est = est.dropna(subset=["stock_return", "market_return"])
        if len(est) < 5:
            results.append({
                "company": company,
                "alpha": np.nan,
                "beta": np.nan,
                "n_estimation_rows": len(est),
            })
            continue

        X = est[["market_return"]].values
        y = est["stock_return"].values

        model = LinearRegression()
        model.fit(X, y)

        beta = float(model.coef_[0])
        alpha = float(model.intercept_)

        results.append({
            "company": company,
            "alpha": alpha,
            "beta": beta,
            "n_estimation_rows": len(est),
        })

        company_mask = df["company"].eq(company)
        df.loc[company_mask, "alpha"] = alpha
        df.loc[company_mask, "beta"] = beta
        df.loc[company_mask, "expected_return"] = alpha + beta * df.loc[company_mask, "market_return"]
        df.loc[company_mask, "ar"] = df.loc[company_mask, "stock_return"] - df.loc[company_mask, "expected_return"]

        # CAR only for event window
        event_mask = company_mask & df["event_day"].between(-8, 8, inclusive="both")
        event_ar = df.loc[event_mask].sort_values("date")["ar"].copy()
        df.loc[event_mask, "car"] = event_ar.cumsum().values

    results_df = pd.DataFrame(results)
    return results_df, df


def run_t_tests(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-sample t-test on AR for event window [-8, +8]
    WITHOUT using scipy (manual calculation)
    """
    import numpy as np

    rows = []

    for company, g in df.groupby("company", sort=False):
        ev = g[g["event_day"].between(-8, 8)]["ar"].dropna()

        n = len(ev)

        if n < 2:
            rows.append({
                "company": company,
                "n": n,
                "t_stat": np.nan,
                "p_value": np.nan,
                "mean_ar": np.nan,
            })
            continue

        mean = ev.mean()
        std = ev.std(ddof=1)

        if std == 0:
            t_stat = 0
        else:
            t_stat = mean / (std / np.sqrt(n))

        # Approximate p-value (normal approx)
        from math import erf, sqrt
        p_value = 2 * (1 - 0.5 * (1 + erf(abs(t_stat) / sqrt(2))))

        rows.append({
            "company": company,
            "n": n,
            "t_stat": float(t_stat),
            "p_value": float(p_value),
            "mean_ar": float(mean),
        })

    return pd.DataFrame(rows)


def prepare_final_output(df: pd.DataFrame) -> pd.DataFrame:
    """
    Final tidy dataset for VS Code / Streamlit.
    """
    keep_cols = [
        c for c in [
            "company", "date", "event_day",
            "stock_price", "market_price",
            "stock_return", "market_return",
            "alpha", "beta", "expected_return", "ar", "car"
        ]
        if c in df.columns
    ]

    out = df[keep_cols].copy()
    out = out.sort_values(["company", "date"]).reset_index(drop=True)
    return out