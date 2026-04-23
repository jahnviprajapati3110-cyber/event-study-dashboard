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

    # 🔥 GLOBAL CLEANING (VERY IMPORTANT)
    df = df.replace([np.inf, -np.inf], np.nan)

    results = []
    df["alpha"] = np.nan
    df["beta"] = np.nan
    df["expected_return"] = np.nan
    df["ar"] = np.nan
    df["car"] = np.nan

    for company, g in df.groupby("company", sort=False):
        g = g.sort_values("date").copy()

        est = g[
            (g["event_day"] >= estimation_start) &
            (g["event_day"] <= estimation_end)
        ].copy()

        # 🔥 CLEAN ESTIMATION DATA
        est = est.replace([np.inf, -np.inf], np.nan)
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

        df.loc[company_mask, "expected_return"] = (
            alpha + beta * df.loc[company_mask, "market_return"]
        )

        df.loc[company_mask, "ar"] = (
            df.loc[company_mask, "stock_return"] -
            df.loc[company_mask, "expected_return"]
        )

        # 🔥 CLEAN AR BEFORE CAR
        df.loc[company_mask, "ar"] = df.loc[company_mask, "ar"].replace(
            [np.inf, -np.inf], np.nan
        )

        # CAR only for event window
        event_mask = company_mask & df["event_day"].between(-8, 8, inclusive="both")
        event_ar = df.loc[event_mask].sort_values("date")["ar"].dropna()

        df.loc[event_mask, "car"] = event_ar.cumsum().values

    results_df = pd.DataFrame(results)
    return results_df, df