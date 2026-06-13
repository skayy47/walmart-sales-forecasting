"""
Walmart Sales Forecasting — Interactive Streamlit Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Walmart Sales Forecast",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Remove top padding */
.block-container { padding-top: 1.5rem; }

/* KPI cards */
.kpi-card {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
}
.kpi-value {
    font-size: 2rem;
    font-weight: 700;
    color: #38bdf8;
    margin: 0;
}
.kpi-label {
    font-size: 0.8rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 0.2rem;
}
.kpi-delta-good { color: #4ade80; font-size: 0.85rem; }
.kpi-delta-bad  { color: #f87171; font-size: 0.85rem; }

/* Section header */
.section-title {
    font-size: 1.15rem;
    font-weight: 600;
    color: #e2e8f0;
    border-left: 3px solid #38bdf8;
    padding-left: 0.75rem;
    margin-bottom: 1rem;
}

/* Sidebar nav label */
section[data-testid="stSidebar"] { background: #0f172a; }
</style>
""", unsafe_allow_html=True)

# ── Data loading ────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent / "outputs" / "powerbi_exports"
IMG_DIR  = Path(__file__).parent / "outputs" / "dashboard_images"

@st.cache_data
def load_data():
    clean    = pd.read_csv(DATA_DIR / "powerbi_clean_walmart.csv",      parse_dates=["Date"])
    xgb      = pd.read_csv(DATA_DIR / "powerbi_xgb_predictions.csv",    parse_dates=["Date"])
    prophet  = pd.read_csv(DATA_DIR / "powerbi_prophet_predictions.csv", parse_dates=["Date"])
    forecast = pd.read_csv(DATA_DIR / "powerbi_future_forecast.csv",     parse_dates=["Date"])

    # Standardise prophet columns
    prophet = prophet.rename(columns={
        "Actual_Weekly_Sales":    "Actual",
        "Predicted_Weekly_Sales": "Predicted",
    })

    return clean, xgb, prophet, forecast

try:
    clean_df, xgb_df, prophet_df, forecast_df = load_data()
    data_ok = True
except Exception as e:
    st.error(f"Could not load data: {e}")
    data_ok = False

# ── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛒 Walmart Forecast")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["📊 Overview", "🤖 Model Comparison", "🔮 Forecast", "📈 Feature Analysis", "🖼️ Gallery"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.78rem; color:#64748b; line-height:1.6'>
    <b>Models</b><br>
    Prophet (Optuna) · XGBoost · SARIMAX<br><br>
    <b>Best</b><br>
    Prophet → MAPE <b style='color:#4ade80'>2.17%</b><br><br>
    <b>Data</b><br>
    Walmart Kaggle · 2010–2012<br>
    45 stores · 421K records
    </div>
    """, unsafe_allow_html=True)


# ── Helper: Plotly theme ─────────────────────────────────────────────────────────
LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(15,23,42,0.6)",
    font=dict(color="#cbd5e1", size=13),
    xaxis=dict(gridcolor="#1e293b", linecolor="#334155"),
    yaxis=dict(gridcolor="#1e293b", linecolor="#334155"),
    legend=dict(bgcolor="rgba(15,23,42,0.8)", bordercolor="#334155", borderwidth=1),
    margin=dict(l=0, r=0, t=40, b=0),
    hovermode="x unified",
)

def fmt_millions(val: float) -> str:
    return f"${val/1e6:.1f}M"


# ══════════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════════
if page == "📊 Overview":
    st.title("📊 Walmart Weekly Sales — Overview")

    if not data_ok:
        st.stop()

    # ── KPI row ──
    total_sales  = clean_df["Weekly_Sales"].sum()
    avg_weekly   = clean_df["Weekly_Sales"].mean()
    peak_week    = clean_df.loc[clean_df["Weekly_Sales"].idxmax()]
    low_week     = clean_df.loc[clean_df["Weekly_Sales"].idxmin()]
    n_weeks      = len(clean_df)

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(f"""
        <div class="kpi-card">
          <p class="kpi-value">{fmt_millions(total_sales)}</p>
          <p class="kpi-label">Total Sales (2y)</p>
        </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="kpi-card">
          <p class="kpi-value">{fmt_millions(avg_weekly)}</p>
          <p class="kpi-label">Avg Weekly Sales</p>
        </div>""", unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="kpi-card">
          <p class="kpi-value">{fmt_millions(peak_week['Weekly_Sales'])}</p>
          <p class="kpi-label">Peak Week · {peak_week['Date'].strftime('%b %Y')}</p>
        </div>""", unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
        <div class="kpi-card">
          <p class="kpi-value">{n_weeks}</p>
          <p class="kpi-label">Weeks of Data</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Historical sales chart ──
    st.markdown('<p class="section-title">Historical Weekly Sales</p>', unsafe_allow_html=True)

    df_plot = clean_df.copy()
    df_plot["roll_12"] = df_plot["Weekly_Sales"].rolling(12).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_plot["Date"], y=df_plot["Weekly_Sales"],
        name="Weekly Sales", line=dict(color="#38bdf8", width=1.5),
        fill="tozeroy", fillcolor="rgba(56,189,248,0.06)",
    ))
    fig.add_trace(go.Scatter(
        x=df_plot["Date"], y=df_plot["roll_12"],
        name="12-wk Avg", line=dict(color="#fb923c", width=2, dash="dot"),
    ))
    fig.update_layout(**LAYOUT, title="Walmart Total Weekly Sales — All Stores Combined")
    fig.update_yaxes(tickprefix="$", tickformat=",.0f")
    st.plotly_chart(fig, use_container_width=True)

    # ── YoY monthly heatmap ──
    st.markdown('<p class="section-title">Monthly Sales Heatmap (Year-over-Year)</p>', unsafe_allow_html=True)

    df_heat = clean_df.copy()
    df_heat["Year"]  = df_heat["Date"].dt.year
    df_heat["Month"] = df_heat["Date"].dt.strftime("%b")
    pivot = df_heat.groupby(["Year", "Month"])["Weekly_Sales"].sum().reset_index()
    month_order = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    pivot["Month"] = pd.Categorical(pivot["Month"], categories=month_order, ordered=True)
    pivot = pivot.sort_values(["Year","Month"])
    heat_matrix = pivot.pivot(index="Year", columns="Month", values="Weekly_Sales")

    fig2 = px.imshow(
        heat_matrix,
        color_continuous_scale="Blues",
        labels=dict(color="Sales ($)"),
        text_auto=".3s",
        aspect="auto",
    )
    fig2.update_layout(**LAYOUT, title="Monthly Sales by Year")
    st.plotly_chart(fig2, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════════
# PAGE 2 — MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Model Comparison":
    st.title("🤖 Model Comparison")

    if not data_ok:
        st.stop()

    # ── Metrics table ──
    st.markdown('<p class="section-title">Benchmark Results</p>', unsafe_allow_html=True)

    metrics = pd.DataFrame([
        {"Model": "Prophet (Optuna)",   "MAE": "$1,024,014", "RMSE": "$1,352,138", "MAPE": "2.17%", "Rank": "🥇"},
        {"Model": "XGBoost (Optuna+FE)","MAE": "$1,378,265", "RMSE": "$1,884,462", "MAPE": "—",     "Rank": "🥈"},
        {"Model": "XGBoost (baseline)", "MAE": "$1,221,143", "RMSE": "$1,625,766", "MAPE": "2.63%", "Rank": "🥈"},
        {"Model": "Prophet+Regressors", "MAE": "$2,300,821", "RMSE": "$2,748,733", "MAPE": "4.95%", "Rank": "🥉"},
        {"Model": "SARIMAX",            "MAE": "$3,298,372", "RMSE": "$3,629,445", "MAPE": "7.09%", "Rank": "4"},
    ])
    st.dataframe(metrics, use_container_width=True, hide_index=True)

    # ── Actual vs XGBoost vs Prophet ──
    st.markdown('<p class="section-title">Actual vs XGBoost vs Prophet (Test Period)</p>', unsafe_allow_html=True)

    merge = xgb_df[["Date","Actual","Predicted"]].rename(columns={"Predicted":"XGB"})
    merge = merge.merge(
        prophet_df[["Date","Predicted"]].rename(columns={"Predicted":"Prophet"}),
        on="Date", how="inner"
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=merge["Date"], y=merge["Actual"],  name="Actual",           line=dict(color="#f1f5f9", width=2.5)))
    fig.add_trace(go.Scatter(x=merge["Date"], y=merge["XGB"],     name="XGBoost (Optuna)", line=dict(color="#38bdf8", width=2)))
    fig.add_trace(go.Scatter(x=merge["Date"], y=merge["Prophet"], name="Prophet (Optuna)", line=dict(color="#fb923c", width=2, dash="dot")))
    fig.update_layout(**LAYOUT, title="Actual vs Predicted — Test Window (May–Oct 2012)")
    fig.update_yaxes(tickprefix="$", tickformat=",.0f")
    st.plotly_chart(fig, use_container_width=True)

    # ── Residual analysis ──
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<p class="section-title">XGBoost Residuals Over Time</p>', unsafe_allow_html=True)
        xgb_df["Residual"] = xgb_df["Actual"] - xgb_df["Predicted"]
        fig3 = go.Figure()
        fig3.add_hline(y=0, line_color="#64748b", line_dash="dot")
        fig3.add_trace(go.Scatter(
            x=xgb_df["Date"], y=xgb_df["Residual"],
            mode="lines+markers", line=dict(color="#a78bfa", width=1.5),
            marker=dict(size=5), name="Residual",
        ))
        fig3.update_layout(**LAYOUT, title="XGBoost Residuals")
        fig3.update_yaxes(tickprefix="$", tickformat=",.0f")
        st.plotly_chart(fig3, use_container_width=True)

    with col2:
        st.markdown('<p class="section-title">Residual Distribution</p>', unsafe_allow_html=True)
        fig4 = go.Figure()
        fig4.add_trace(go.Histogram(
            x=xgb_df["Residual"], nbinsx=20,
            marker_color="#38bdf8", opacity=0.8, name="XGBoost",
        ))
        fig4.add_trace(go.Histogram(
            x=prophet_df["Actual"] - prophet_df["Predicted"], nbinsx=20,
            marker_color="#fb923c", opacity=0.6, name="Prophet",
        ))
        fig4.update_layout(**LAYOUT, title="Residual Distribution", barmode="overlay")
        fig4.update_xaxes(tickprefix="$", tickformat=",.0f")
        st.plotly_chart(fig4, use_container_width=True)

    # ── Error % scatter ──
    st.markdown('<p class="section-title">Actual vs Predicted Scatter (XGBoost)</p>', unsafe_allow_html=True)
    perfect_line = [xgb_df["Actual"].min(), xgb_df["Actual"].max()]
    fig5 = go.Figure()
    fig5.add_trace(go.Scatter(
        x=perfect_line, y=perfect_line,
        mode="lines", line=dict(color="#64748b", dash="dash"), name="Perfect",
    ))
    fig5.add_trace(go.Scatter(
        x=xgb_df["Actual"], y=xgb_df["Predicted"],
        mode="markers", marker=dict(color="#38bdf8", size=9, opacity=0.8),
        name="XGBoost",
    ))
    fig5.update_layout(**LAYOUT, title="Actual vs Predicted Scatter")
    fig5.update_xaxes(tickprefix="$", tickformat=",.0f", title="Actual")
    fig5.update_yaxes(tickprefix="$", tickformat=",.0f", title="Predicted")
    st.plotly_chart(fig5, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════════
# PAGE 3 — FORECAST
# ══════════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Forecast":
    st.title("🔮 Future Forecast")

    if not data_ok:
        st.stop()

    st.markdown('<p class="section-title">Prophet Forecast with Confidence Bands</p>', unsafe_allow_html=True)

    # Only show future dates (beyond training data)
    train_end = clean_df["Date"].max()
    future_only = forecast_df[forecast_df["Date"] > train_end].copy()
    history_tail = clean_df.tail(52).copy()

    fig = go.Figure()

    # History
    fig.add_trace(go.Scatter(
        x=history_tail["Date"], y=history_tail["Weekly_Sales"],
        name="Historical (last 52 wks)", line=dict(color="#94a3b8", width=1.5),
    ))

    # Confidence band
    if "yhat_upper" in future_only.columns and "yhat_lower" in future_only.columns:
        fig.add_trace(go.Scatter(
            x=pd.concat([future_only["Date"], future_only["Date"][::-1]]),
            y=pd.concat([future_only["yhat_upper"], future_only["yhat_lower"][::-1]]),
            fill="toself", fillcolor="rgba(251,146,60,0.12)",
            line=dict(color="rgba(0,0,0,0)"), name="95% CI",
        ))

    # Forecast line
    fig.add_trace(go.Scatter(
        x=future_only["Date"], y=future_only["yhat"],
        name="Prophet Forecast", line=dict(color="#fb923c", width=2.5),
    ))

    fig.add_vline(x=str(train_end.date()), line_color="#4ade80", line_dash="dash",
                  annotation_text="Training end", annotation_position="top right")

    fig.update_layout(**LAYOUT, title="Walmart Weekly Sales — Future Forecast")
    fig.update_yaxes(tickprefix="$", tickformat=",.0f")
    st.plotly_chart(fig, use_container_width=True)

    # ── Forecast table ──
    st.markdown('<p class="section-title">Forecast Table</p>', unsafe_allow_html=True)
    if not future_only.empty:
        tbl = future_only[["Date","yhat","yhat_lower","yhat_upper"]].copy()
        tbl.columns = ["Date","Forecast","Lower Bound","Upper Bound"]
        for col in ["Forecast","Lower Bound","Upper Bound"]:
            tbl[col] = tbl[col].map(lambda x: f"${x:,.0f}")
        st.dataframe(tbl.reset_index(drop=True), use_container_width=True, hide_index=True)
    else:
        # Show full forecast when no future-only slice available
        tbl = forecast_df[["Date","yhat","yhat_lower","yhat_upper"]].tail(20).copy()
        tbl.columns = ["Date","Forecast","Lower Bound","Upper Bound"]
        for col in ["Forecast","Lower Bound","Upper Bound"]:
            tbl[col] = tbl[col].map(lambda x: f"${x:,.0f}")
        st.dataframe(tbl.reset_index(drop=True), use_container_width=True, hide_index=True)

    # ── Seasonal components ──
    st.markdown('<p class="section-title">Seasonal Components</p>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        if "yearly" in forecast_df.columns:
            fig6 = go.Figure()
            df_comp = forecast_df.sort_values("Date")
            fig6.add_trace(go.Scatter(
                x=df_comp["Date"], y=df_comp["yearly"],
                line=dict(color="#a78bfa", width=2), name="Yearly",
            ))
            fig6.update_layout(**LAYOUT, title="Yearly Seasonality Component")
            fig6.update_yaxes(tickformat=",.0f")
            st.plotly_chart(fig6, use_container_width=True)

    with col2:
        if "weekly" in forecast_df.columns:
            fig7 = go.Figure()
            fig7.add_trace(go.Scatter(
                x=df_comp["Date"], y=df_comp["weekly"],
                line=dict(color="#34d399", width=2), name="Weekly",
            ))
            fig7.update_layout(**LAYOUT, title="Weekly Seasonality Component")
            fig7.update_yaxes(tickformat=",.0f")
            st.plotly_chart(fig7, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════════
# PAGE 4 — FEATURE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════════
elif page == "📈 Feature Analysis":
    st.title("📈 Feature Analysis")

    if not data_ok:
        st.stop()

    # ── Rolling stats ──
    st.markdown('<p class="section-title">Rolling Statistics (12-Week Window)</p>', unsafe_allow_html=True)

    df_roll = clean_df.copy()
    df_roll["roll_mean"] = df_roll["Weekly_Sales"].rolling(12).mean()
    df_roll["roll_std"]  = df_roll["Weekly_Sales"].rolling(12).std()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_roll["Date"], y=df_roll["Weekly_Sales"],
        name="Weekly Sales", line=dict(color="#38bdf8", width=1.5), opacity=0.5,
    ))
    fig.add_trace(go.Scatter(
        x=df_roll["Date"], y=df_roll["roll_mean"],
        name="12-wk Mean", line=dict(color="#f1f5f9", width=2.5),
    ))
    fig.update_layout(**LAYOUT, title="Sales + 12-Week Rolling Mean")
    fig.update_yaxes(tickprefix="$", tickformat=",.0f")
    st.plotly_chart(fig, use_container_width=True)

    # ── Feature importance (static PNG) ──
    st.markdown('<p class="section-title">XGBoost Feature Importance (Top 20)</p>', unsafe_allow_html=True)

    # Rebuild as interactive Plotly from known values (from notebook output)
    features = {
        "weekofyear": 0.371, "month": 0.261, "roll_std_4": 0.091, "lag_1": 0.085,
        "lag_4": 0.048, "lag_12": 0.028, "lag_2": 0.018, "cpi_unemp": 0.014,
        "Temperature": 0.013, "roll_mean_4": 0.013, "roll_mean_12": 0.013,
        "Fuel_Price": 0.009, "quarter": 0.008, "fuel_temp": 0.007,
        "CPI": 0.006, "roll_std_12": 0.004, "Unemployment": 0.004,
        "year": 0.002, "IsHoliday_flag": 0.002, "is_month_end": 0.001,
    }
    fi_df = pd.DataFrame(list(features.items()), columns=["Feature","Importance"])
    fi_df = fi_df.sort_values("Importance", ascending=True)

    fig2 = go.Figure(go.Bar(
        x=fi_df["Importance"], y=fi_df["Feature"],
        orientation="h",
        marker=dict(
            color=fi_df["Importance"],
            colorscale="Blues",
            showscale=False,
        ),
        text=fi_df["Importance"].map(lambda x: f"{x:.3f}"),
        textposition="outside",
    ))
    fig2.update_layout(**LAYOUT, title="Top 20 Feature Importances — XGBoost (Optuna + FE)",
                       height=550, margin=dict(l=100, r=60, t=40, b=0))
    st.plotly_chart(fig2, use_container_width=True)

    # ── Correlation insight ──
    st.markdown('<p class="section-title">Key Insights</p>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("📅 **Calendar dominates**\n\n`weekofyear` + `month` account for **63%** of feature importance. Sales are strongly seasonal.")
    with col2:
        st.info("⏱️ **Short-lag wins**\n\n`lag_1` (1-week) outperforms longer lags — the most recent week is the best single predictor.")
    with col3:
        st.info("📉 **Volatility matters**\n\n`roll_std_4` ranks 3rd — recent sales volatility is a stronger signal than the mean.")


# ══════════════════════════════════════════════════════════════════════════════════
# PAGE 5 — GALLERY
# ══════════════════════════════════════════════════════════════════════════════════
elif page == "🖼️ Gallery":
    st.title("🖼️ Dashboard Gallery")
    st.markdown("All charts are auto-generated by `notebooks/02_ml_forecasting.ipynb`.")

    charts = [
        ("A_actual_vs_xgb.png",        "Actual vs XGBoost (Optuna + FE)"),
        ("B_actual_vs_prophet.png",     "Actual vs Prophet (Optuna)"),
        ("C_model_comparison.png",      "Model Comparison — All Models"),
        ("D_residuals_over_time.png",   "Residuals Over Time"),
        ("E_residual_distribution.png", "Residual Distribution"),
        ("F_feature_importance.png",    "Top 20 Feature Importances"),
        ("G_rolling_stats.png",         "Rolling Statistics"),
        ("H_seasonal_decomposition.png","Seasonal Decomposition"),
        ("I_future_forecast.png",       "Future Forecast"),
    ]

    for i in range(0, len(charts), 2):
        cols = st.columns(2)
        for j, col in enumerate(cols):
            if i + j < len(charts):
                fname, title = charts[i + j]
                img_path = IMG_DIR / fname
                if img_path.exists():
                    with col:
                        st.markdown(f'<p class="section-title">{title}</p>', unsafe_allow_html=True)
                        st.image(str(img_path), use_container_width=True)
                else:
                    with col:
                        st.warning(f"Image not found: {fname}")
