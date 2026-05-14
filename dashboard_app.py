"""
dashboard_app.py
Sri Lanka Economic Indicators Dashboard
────────────────────────────────────────
Author : Adithya Wijerathne
Data   : World Bank Open Data (LKA indicators)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.linear_model import LinearRegression

# ──────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sri Lanka Economic Dashboard",
    page_icon="🇱🇰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────
# LOAD DATA
# ──────────────────────────────────────────────────────────────
@st.cache_data                           # cache so the file isn't re-read on every interaction
def load_data():
    df = pd.read_csv("data_clean/final_economic_data.csv")
    df["year"] = df["year"].astype(int)
    df = df.sort_values("year").reset_index(drop=True)
    return df

df = load_data()

# ──────────────────────────────────────────────────────────────
# HISTORICAL CRISIS EVENTS
# These are real events we annotate on the charts so the viewer
# understands *why* the data moves the way it does.
# ──────────────────────────────────────────────────────────────
EVENTS = [
    {"year": 2001, "label": "9/11 + civil war peak",   "color": "#E74C3C"},
    {"year": 2004, "label": "Indian Ocean Tsunami",     "color": "#E67E22"},
    {"year": 2008, "label": "Global financial crisis",  "color": "#E74C3C"},
    {"year": 2009, "label": "Civil war ends",           "color": "#27AE60"},
    {"year": 2019, "label": "Easter Sunday attacks",    "color": "#E67E22"},
    {"year": 2020, "label": "COVID-19 pandemic",        "color": "#8E44AD"},
    {"year": 2022, "label": "Economic crisis – worst in 73 yrs", "color": "#C0392B"},
]

# ──────────────────────────────────────────────────────────────
# HELPER: ADD EVENT ANNOTATIONS TO A PLOTLY FIGURE
# ──────────────────────────────────────────────────────────────
def add_events(fig, df_filtered, show_events):
    """
    Draws a vertical dashed line for each historical event that falls
    within the currently selected year range.
    """
    if not show_events:
        return fig
    y_min = df_filtered.select_dtypes(include="number").min().min()
    y_max = df_filtered.select_dtypes(include="number").max().max()
    for ev in EVENTS:
        if ev["year"] in df_filtered["year"].values:
            fig.add_vline(
                x=ev["year"],
                line_dash="dot",
                line_color=ev["color"],
                line_width=1.5,
                annotation_text=ev["label"],
                annotation_position="top left",
                annotation_font_size=10,
                annotation_font_color=ev["color"],
            )
    return fig

# ──────────────────────────────────────────────────────────────
# HELPER: COMPUTE FORECAST (simple linear regression)
# ──────────────────────────────────────────────────────────────
def compute_forecast(df, col, horizon=3):
    """
    Fits a straight-line (LinearRegression) to the last 10 years of
    data and projects 'horizon' years into the future.
    Returns a small DataFrame with the forecast years and values.
    """
    recent = df.tail(10).copy()
    X = recent[["year"]]
    y = recent[col]
    model = LinearRegression().fit(X, y)
    future_years = np.arange(df["year"].max() + 1,
                             df["year"].max() + 1 + horizon).reshape(-1, 1)
    preds = model.predict(future_years)
    return pd.DataFrame({"year": future_years.flatten(), col: preds})

# ──────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://flagcdn.com/w80/lk.png", width=60)
    st.title("Filters")

    year_range = st.slider(
        "Year range",
        min_value=int(df["year"].min()),
        max_value=int(df["year"].max()),
        value=(2000, int(df["year"].max())),
    )

    show_events    = st.toggle("Show historical events",  value=True)
    show_forecast  = st.toggle("Show trend forecast",     value=True)
    show_crisis    = st.toggle("Show crisis deep-dive",   value=True)

    st.divider()
    st.caption("Data source: World Bank Open Data  \nIndicators: NY.GDP, FP.CPI, SL.UEM, PA.NUS")

# Filter the master DataFrame to the selected range
df_f = df[(df["year"] >= year_range[0]) & (df["year"] <= year_range[1])].copy()

# ──────────────────────────────────────────────────────────────
# TITLE
# ──────────────────────────────────────────────────────────────
st.title("🇱🇰 Sri Lanka Economic Indicators Dashboard")
st.caption(
    "Tracks GDP growth, inflation, unemployment, and exchange rate from "
    "World Bank data (1991–2024). Includes crisis annotation and trend forecasting."
)
st.divider()

# ──────────────────────────────────────────────────────────────
# SECTION 1 — KPI METRIC CARDS WITH YEAR-OVER-YEAR DELTA
# ──────────────────────────────────────────────────────────────
"""
## Section 1 — Key Performance Indicators

Each card shows the most recent value AND how it changed vs the previous
year (year-over-year delta). The delta colour tells you immediately
whether the change is good or bad.

Arrow meanings:
  ▲ green  = good (GDP up, unemployment down, inflation down)
  ▼ red    = bad
  ● grey   = neutral / stable
"""
st.subheader("Latest indicators (with year-over-year change)")

latest = df_f.iloc[-1]
prev   = df_f.iloc[-2] if len(df_f) > 1 else latest

def delta_display(current, previous, higher_is_better=True):
    """
    Returns (delta_string, colour_string) for a metric card.
    higher_is_better = True  → GDP (more growth = good)
    higher_is_better = False → inflation, unemployment, exchange rate
    """
    delta = current - previous
    if higher_is_better:
        colour = "normal" if delta >= 0 else "inverse"
    else:
        colour = "inverse" if delta >= 0 else "normal"
    return round(delta, 2), colour

c1, c2, c3, c4 = st.columns(4)

gdp_d, gdp_c    = delta_display(latest["gdp_growth"],    prev["gdp_growth"],    True)
inf_d, inf_c    = delta_display(latest["inflation"],      prev["inflation"],     False)
unem_d, unem_c  = delta_display(latest["unemployment"],  prev["unemployment"],  False)
fx_d,  fx_c    = delta_display(latest["exchange_rate"],  prev["exchange_rate"], False)

c1.metric("GDP Growth",     f"{latest['gdp_growth']:.1f}%",    f"{gdp_d:+.1f}%",  delta_color=gdp_c)
c2.metric("Inflation",      f"{latest['inflation']:.1f}%",     f"{inf_d:+.1f}%",  delta_color=inf_c)
c3.metric("Unemployment",   f"{latest['unemployment']:.1f}%",  f"{unem_d:+.1f}%", delta_color=unem_c)
c4.metric("Exchange Rate",  f"LKR {latest['exchange_rate']:.0f}", f"{fx_d:+.0f}", delta_color=fx_c)

st.divider()

# ──────────────────────────────────────────────────────────────
# SECTION 2 — ANNOTATED TREND CHARTS WITH FORECAST
# ──────────────────────────────────────────────────────────────
st.subheader("Trend charts — annotated with historical events")

CHART_CFG = [
    ("gdp_growth",    "GDP Growth (%)",            "% annual",   True),
    ("inflation",     "Inflation (%)",             "% annual",   False),
    ("unemployment",  "Unemployment Rate (%)",     "% of labour",False),
    ("exchange_rate", "Exchange Rate (LKR / USD)", "LKR",        False),
]

for col, title, ylab, higher_good in CHART_CFG:
    fig = go.Figure()

    # ── Actual data line ──
    fig.add_trace(go.Scatter(
        x=df_f["year"], y=df_f[col],
        mode="lines+markers",
        name="Actual",
        line=dict(color="#2471A3", width=2),
        marker=dict(size=5),
        hovertemplate="%{x}: %{y:.2f}<extra></extra>",
    ))

    # ── Forecast dashed line ──
    if show_forecast:
        fc = compute_forecast(df_f, col)
        fig.add_trace(go.Scatter(
            x=fc["year"], y=fc[col],
            mode="lines+markers",
            name="Trend forecast",
            line=dict(color="#E67E22", width=2, dash="dash"),
            marker=dict(size=5, symbol="diamond"),
            hovertemplate="Forecast %{x}: %{y:.2f}<extra></extra>",
        ))

    # ── Event annotations ──
    add_events(fig, df_f, show_events)

    fig.update_layout(
        title=title,
        xaxis_title="Year",
        yaxis_title=ylab,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=380,
        margin=dict(l=40, r=40, t=60, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ──────────────────────────────────────────────────────────────
# SECTION 3 — CORRELATION HEATMAP + ANALYST INTERPRETATION
# ──────────────────────────────────────────────────────────────
"""
## Section 3 — Correlation analysis

A correlation value ranges from -1 to +1:
  +1 = the two indicators move in exactly the same direction
  -1 = they move in exactly opposite directions
   0 = no linear relationship

We compute this for every pair of our 4 indicators using Pearson correlation.
"""
st.subheader("Correlation between indicators")

corr_cols = ["gdp_growth", "inflation", "unemployment", "exchange_rate"]
corr_labels = ["GDP Growth", "Inflation", "Unemployment", "Exchange Rate"]
corr_matrix = df_f[corr_cols].corr().round(2)

fig_corr = px.imshow(
    corr_matrix,
    x=corr_labels,
    y=corr_labels,
    color_continuous_scale="RdBu_r",
    zmin=-1, zmax=1,
    text_auto=True,
    title="Pearson Correlation Matrix",
)
fig_corr.update_layout(height=380, margin=dict(l=40, r=40, t=60, b=40))
st.plotly_chart(fig_corr, use_container_width=True)

# ── Analyst interpretation — this is the actual analyst work ──
st.markdown("#### 🔍 Analyst interpretation")

# Dynamically compute the values for honest interpretation
gdp_inf_r  = corr_matrix.loc["gdp_growth",   "inflation"]
gdp_unem_r = corr_matrix.loc["gdp_growth",   "unemployment"]
inf_fx_r   = corr_matrix.loc["inflation",    "exchange_rate"]

interp1 = "positively" if gdp_inf_r > 0 else "negatively"
interp2 = "positively" if gdp_unem_r > 0 else "negatively"
interp3 = "strong positive" if inf_fx_r > 0.5 else ("moderate positive" if inf_fx_r > 0 else "negative")

st.info(f"""
**Finding 1 — GDP vs Inflation (r = {gdp_inf_r}):**  
GDP growth and inflation are {interp1} correlated across this dataset. This is consistent with
demand-pull inflation: strong economic growth increases consumer spending, which pushes prices up.

**Finding 2 — GDP vs Unemployment (r = {gdp_unem_r}):**  
GDP growth and unemployment move {interp2} — reflecting Okun's Law: when the economy grows,
firms hire more workers, reducing unemployment.

**Finding 3 — Inflation vs Exchange Rate (r = {inf_fx_r}):**  
There is a {interp3} relationship between inflation and the exchange rate. This is critical for Sri Lanka:
as the LKR depreciates against the USD, imports become more expensive, which directly feeds into
consumer price inflation. The 2022 crisis (LKR crashed from ~200 to ~370 per USD) illustrates this perfectly.
""")

st.divider()

# ──────────────────────────────────────────────────────────────
# SECTION 4 — 2022 CRISIS DEEP-DIVE
# ──────────────────────────────────────────────────────────────
if show_crisis:
    st.subheader("2022 economic crisis — deep-dive analysis")
    st.caption("Zoomed into 2017–2024 to show the lead-up, crisis peak, and recovery trajectory.")

    crisis_df = df[(df["year"] >= 2017) & (df["year"] <= 2024)].copy()

    fig_crisis = go.Figure()

    # Normalise each indicator to its 2019 value (= 100) so we can
    # compare 4 very different scales on the same chart
    base_year = 2019
    base_row  = crisis_df[crisis_df["year"] == base_year].iloc[0]

    for col, label, colour in [
        ("gdp_growth",   "GDP Growth",   "#2471A3"),
        ("inflation",    "Inflation",    "#E74C3C"),
        ("unemployment", "Unemployment", "#27AE60"),
        ("exchange_rate","Exchange Rate","#E67E22"),
    ]:
        if base_row[col] != 0:
            normalised = (crisis_df[col] / base_row[col]) * 100
        else:
            normalised = crisis_df[col]
        fig_crisis.add_trace(go.Scatter(
            x=crisis_df["year"], y=normalised.round(1),
            mode="lines+markers",
            name=label,
            line=dict(color=colour, width=2),
        ))

    # Mark the crisis peak
    fig_crisis.add_vline(x=2022, line_dash="dash", line_color="#C0392B", line_width=2,
                         annotation_text="Crisis peak", annotation_font_color="#C0392B")
    fig_crisis.add_hline(y=100, line_dash="dot", line_color="gray",
                         annotation_text="2019 baseline (=100)")

    fig_crisis.update_layout(
        title="All indicators indexed to 2019 baseline (2019 = 100)",
        yaxis_title="Index (2019 = 100)",
        xaxis_title="Year",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=420,
        margin=dict(l=40, r=40, t=60, b=40),
    )
    st.plotly_chart(fig_crisis, use_container_width=True)

    # Written recovery analysis
    st.markdown("#### Recovery analysis")
    latest_gdp   = df[df["year"] == 2024]["gdp_growth"].values[0]
    latest_inf   = df[df["year"] == 2024]["inflation"].values[0]
    latest_fx    = df[df["year"] == 2024]["exchange_rate"].values[0]
    crisis_inf   = df[df["year"] == 2022]["inflation"].values[0]
    crisis_fx    = df[df["year"] == 2022]["exchange_rate"].values[0]

    st.warning(f"""
    **What happened in 2022:**  
    Sri Lanka's economy contracted by **7.8%** — the worst recession since independence. Inflation surged to
    **{crisis_inf:.1f}%** (food inflation exceeded 90% at its peak). The LKR collapsed from ~LKR 200 to
    **LKR {crisis_fx:.0f}** per USD, making fuel, medicine, and essential imports unaffordable.
    The government declared a sovereign debt default in April 2022.

    **Recovery indicators (as of 2024):**  
    ✅ GDP growth has returned to **+{latest_gdp:.1f}%** — positive, but not back to pre-crisis levels.  
    ✅ Inflation has fallen sharply to **{latest_inf:.1f}%** — from the 46% peak.  
    ⚠️ Exchange rate remains elevated at **LKR {latest_fx:.0f}** — the rupee has not recovered to pre-crisis levels.  
    ⚠️ Full economic recovery is projected to take until **2026–2027** under IMF programme conditions.
    """)

    st.divider()

# ──────────────────────────────────────────────────────────────
# SECTION 5 — YEAR-OVER-YEAR CHANGE TABLE
# ──────────────────────────────────────────────────────────────
st.subheader("Year-over-year change (Δ vs previous year)")
st.caption("Positive delta for GDP = good. Positive delta for inflation/unemployment/FX = bad.")

df_delta = df_f[["year"] + corr_cols].copy()
for col in corr_cols:
    df_delta[f"Δ {col}"] = df_delta[col].diff().round(2)

display_cols = ["year", "gdp_growth", "Δ gdp_growth",
                "inflation", "Δ inflation",
                "unemployment", "Δ unemployment",
                "exchange_rate", "Δ exchange_rate"]
st.dataframe(
    df_delta[display_cols].sort_values("year", ascending=False),
    use_container_width=True,
    hide_index=True,
)

st.divider()

# ──────────────────────────────────────────────────────────────
# SECTION 6 — RAW DATA + DOWNLOAD
# ──────────────────────────────────────────────────────────────
with st.expander("Raw data table"):
    st.dataframe(df_f, use_container_width=True, hide_index=True)
    csv = df_f.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download filtered data as CSV",
        data=csv,
        file_name="sl_economic_data_filtered.csv",
        mime="text/csv",
    )

st.caption("Dashboard by Adithya Wijerathne | Data: World Bank Open Data | Built with Streamlit + Plotly")
