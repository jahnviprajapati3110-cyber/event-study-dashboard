from __future__ import annotations

import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from event_study import (
    load_event_study_data,
    fill_missing_prices,
    calculate_returns,
    estimate_market_model,
    run_t_tests,
    prepare_final_output,
)

st.set_page_config(
    page_title="Event Study Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📈 Earnings Event Study Dashboard")
st.caption("A professional event-study dashboard for NIFTY 50 companies using the market model.")

DATA_PATH = "data/Event_Study_FY2025.xlsx"

@st.cache_data
def load_all_data(path: str):
    df = load_event_study_data(path)
    df = fill_missing_prices(df)
    df = calculate_returns(df)
    return df

try:
    df = load_all_data(DATA_PATH)
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

st.sidebar.header("Controls")
companies = sorted(df["company"].dropna().unique().tolist())
selected_company = st.sidebar.selectbox("Select Company", companies)

event_start = st.sidebar.number_input("Event window start", value=-8, step=1)
event_end = st.sidebar.number_input("Event window end", value=8, step=1)
est_start = st.sidebar.number_input("Estimation window start", value=-120, step=1)
est_end = st.sidebar.number_input("Estimation window end", value=-20, step=1)

company_dates = df[df["company"].eq(selected_company)][["date"]].drop_duplicates().sort_values("date")
min_date = company_dates["date"].min()
max_date = company_dates["date"].max()

st.sidebar.markdown("---")
st.sidebar.write("Date range in file:")
st.sidebar.write(f"**{min_date.date()}** to **{max_date.date()}**")

# Compute models
coeff_df, modeled_df = estimate_market_model(
    df,
    estimation_start=est_start,
    estimation_end=est_end,
)
t_df = run_t_tests(modeled_df)
final_df = prepare_final_output(modeled_df)

company_df = final_df[final_df["company"].eq(selected_company)].copy()
event_df = company_df[company_df["event_day"].between(event_start, event_end, inclusive="both")].copy()
est_df = company_df[company_df["event_day"].between(est_start, est_end, inclusive="both")].copy()

coeff_row = coeff_df[coeff_df["company"].eq(selected_company)]
test_row = t_df[t_df["company"].eq(selected_company)]

alpha = coeff_row["alpha"].iloc[0] if not coeff_row.empty else None
beta = coeff_row["beta"].iloc[0] if not coeff_row.empty else None
p_value = test_row["p_value"].iloc[0] if not test_row.empty else None
mean_ar = test_row["mean_ar"].iloc[0] if not test_row.empty else None
car_end = event_df["car"].iloc[-1] if not event_df.empty else None

col1, col2, col3, col4 = st.columns(4)
col1.metric("Alpha", f"{alpha:.6f}" if pd.notna(alpha) else "NA")
col2.metric("Beta", f"{beta:.6f}" if pd.notna(beta) else "NA")
col3.metric("CAR at end of window", f"{car_end:.6f}" if pd.notna(car_end) else "NA")
col4.metric("AR t-test p-value", f"{p_value:.6f}" if pd.notna(p_value) else "NA")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Overview", "Event Study Charts", "Statistics", "Data Table", "Export"]
)

with tab1:
    st.subheader(f"{selected_company} — Event Summary")
    st.write(
        "This dashboard compares actual stock returns with market-expected returns around the earnings event. "
        "AR shows the abnormal movement, and CAR shows the cumulative impact across the event window."
    )

    if pd.notna(car_end):
        if car_end > 0:
            st.success("Positive cumulative reaction in the event window.")
        elif car_end < 0:
            st.error("Negative cumulative reaction in the event window.")
        else:
            st.info("Neutral cumulative reaction in the event window.")

    st.write("**Interpretation guide:**")
    st.write("- Positive AR/CAR: market reaction is favorable")
    st.write("- Negative AR/CAR: market reaction is weak or negative")
    st.write("- Significant p-value: abnormal return is statistically meaningful")

with tab2:
    st.subheader("Return Reaction Chart")

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        row_heights=[0.65, 0.35])

    fig.add_trace(
        go.Scatter(
            x=event_df["date"], y=event_df["stock_return"],
            mode="lines+markers", name="Actual Stock Return"
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=event_df["date"], y=event_df["expected_return"],
            mode="lines+markers", name="Expected Return"
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=event_df["date"], y=event_df["market_return"],
            mode="lines+markers", name="Market Return"
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Bar(
            x=event_df["event_day"], y=event_df["ar"],
            name="Abnormal Return (AR)"
        ),
        row=2, col=1
    )

    fig.update_layout(
        height=700,
        legend_orientation="h",
        template="plotly_dark",
        margin=dict(l=20, r=20, t=40, b=20),
    )
    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_xaxes(title_text="Event Day", row=2, col=1)
    fig.update_yaxes(title_text="Return", row=1, col=1)
    fig.update_yaxes(title_text="AR", row=2, col=1)

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("CAR Curve")
    car_fig = go.Figure()
    car_fig.add_trace(
        go.Scatter(
            x=event_df["event_day"],
            y=event_df["car"],
            mode="lines+markers",
            name="CAR"
        )
    )
    car_fig.update_layout(
        template="plotly_dark",
        height=450,
        xaxis_title="Event Day",
        yaxis_title="CAR",
        margin=dict(l=20, r=20, t=40, b=20),
    )
    st.plotly_chart(car_fig, use_container_width=True)

with tab3:
    st.subheader("Statistical Test Results")
    st.dataframe(t_df[t_df["company"].eq(selected_company)], use_container_width=True)

    if pd.notna(p_value):
        if p_value < 0.05:
            st.success("Abnormal return is statistically significant at 5% level.")
        else:
            st.warning("Abnormal return is not statistically significant at 5% level.")

    st.subheader("Model Coefficients")
    st.dataframe(coeff_df[coeff_df["company"].eq(selected_company)], use_container_width=True)

with tab4:
    st.subheader("Prepared Dataset for This Company")
    st.dataframe(event_df, use_container_width=True)

    st.subheader("Event Window Only")
    st.dataframe(
        event_df[[
            "date", "event_day", "stock_return", "market_return",
            "expected_return", "ar", "car"
        ]],
        use_container_width=True
    )

with tab5:
    st.subheader("Download Final Combined Data")
    export_df = prepare_final_output(modeled_df)

    csv = export_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download cleaned analysis dataset as CSV",
        data=csv,
        file_name="event_study_fy2025_cleaned.csv",
        mime="text/csv",
    )

    st.subheader("Download Company-wise Test Summary")
    csv2 = t_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download t-test results as CSV",
        data=csv2,
        file_name="event_study_fy2025_ttests.csv",
        mime="text/csv",
    )

st.markdown("---")
st.caption("Built for event-study analysis: alpha, beta, expected return, AR, CAR, and significance testing.")