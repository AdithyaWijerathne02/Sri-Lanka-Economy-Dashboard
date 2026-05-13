import streamlit as st
import pandas as pd
import plotly.express as px

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Sri Lanka Economy Dashboard",
    layout="wide"
)

# ---------- LOAD DATA ----------
df = pd.read_csv("data_clean/final_economic_data.csv")

# Ensure correct format
df["year"] = df["year"].astype(str)

# ---------- TITLE ----------
st.title("🇱🇰 Sri Lanka Economic Indicators Dashboard")

st.markdown("""
This dashboard shows key macroeconomic indicators of Sri Lanka:
GDP growth, inflation, unemployment, and exchange rate trends.
""")

st.divider()

# ---------- KPI SECTION ----------
col1, col2, col3, col4 = st.columns(4)

col1.metric("Latest GDP Growth", f"{df['gdp_growth'].iloc[-1]:.2f}%")
col2.metric("Latest Inflation", f"{df['inflation'].iloc[-1]:.2f}%")
col3.metric("Latest Unemployment", f"{df['unemployment'].iloc[-1]:.2f}%")
col4.metric("Latest Exchange Rate", f"{df['exchange_rate'].iloc[-1]:.2f}")

st.divider()

# ---------- SIDEBAR FILTER ----------
year_range = st.sidebar.slider(
    "Select Year Range",
    int(df["year"].min()),
    int(df["year"].max()),
    (int(df["year"].min()), int(df["year"].max()))
)

df_filtered = df[
    (df["year"].astype(int) >= year_range[0]) &
    (df["year"].astype(int) <= year_range[1])
]

# ---------- CHARTS ----------

st.subheader("GDP Growth Trend")
fig1 = px.line(df_filtered, x="year", y="gdp_growth")
st.plotly_chart(fig1, use_container_width=True)

st.subheader("Inflation Trend")
fig2 = px.line(df_filtered, x="year", y="inflation")
st.plotly_chart(fig2, use_container_width=True)

st.subheader("Unemployment Trend")
fig3 = px.line(df_filtered, x="year", y="unemployment")
st.plotly_chart(fig3, use_container_width=True)

st.subheader("Exchange Rate Trend (LKR/USD)")
fig4 = px.line(df_filtered, x="year", y="exchange_rate")
st.plotly_chart(fig4, use_container_width=True)

# ---------- DATA VIEW ----------
st.subheader("Raw Data View")
st.dataframe(df_filtered)
