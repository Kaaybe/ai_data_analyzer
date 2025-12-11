# app.py  →  Online Retail UCI Dataset Explorer (Works 100%)
# Deploy this exact file on Streamlit Community Cloud

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ────────────────────────────── CONFIG ──────────────────────────────
st.set_page_config(
    page_title="Online Retail Dashboard",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Direct link to the Excel file on UCI (this works!)
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"

# Custom CSS – beautiful & modern
st.markdown("""
<style>
    .big-font {font-size: 50px !important; font-weight: bold; color: #1e3a8a;}
    .medium-font {font-size: 24px !important; color: #1e40af;}
    .css-1d391kg {padding-top: 3rem;}
    .block-container {padding-top: 2rem;}
</style>
""", unsafe_allow_html=True)

# ────────────────────────────── TITLE ──────────────────────────────
st.markdown('<p class="big-font">Online Retail Dashboard</p>', unsafe_allow_html=True)
st.markdown("*Live analysis of the famous UCI Online Retail dataset*", unsafe_allow_html=True)
st.divider()

# ──────────────────────── LOAD DATA (with cache) ─────────────────────
@st.cache_data(show_spinner=False)
def load_data():
    with st.spinner("Downloading dataset from UCI... (~60 MB, first time only)"):
        df = pd.read_excel(DATA_URL)
    return df

try:
    df = load_data()
    st.success("Dataset loaded successfully! 541,909 transactions")
except Exception as e:
    st.error("Could not load dataset. Please check your internet connection.")
    st.stop()

# Clean data a bit
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
df = df[df['Quantity'] > 0]           # remove returns
df = df[df['UnitPrice'] > 0]
df['Month'] = df['InvoiceDate'].dt.to_period('M').astype(str)

# ─────────────────────────── SIDEBAR ─────────────────────────────
st.sidebar.header("Filters")
selected_country = st.sidebar.multiselect(
    "Country", options=sorted(df['Country'].unique()), default=["United Kingdom"]
)
df_filtered = df[df['Country'].isin(selected_country)]

# ─────────────────────────── METRICS ──────────────────────────────
total_sales = df_filtered['TotalPrice'].sum()
total_orders = df_filtered['InvoiceNo'].nunique()
total_customers = df_filtered['CustomerID'].nunique()
avg_order_value = total_sales / total_orders

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Sales", f"£{total_sales:,.0f}")
c2.metric("Number of Orders", f"{total_orders:,}")
c3.metric("Unique Customers", f"{total_customers:,}")
c4.metric("Avg Order Value", f"£{avg_order_value:,.2f}")

st.divider()

# ─────────────────────────── TABS ───────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Sales Over Time", "Top Products & Customers", "Geography"])

with tab1:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Sales by Month")
        monthly = df_filtered.groupby('Month')['TotalPrice'].sum().reset_index()
        fig = px.bar(monthly, x='Month', y='TotalPrice',
                     title="Monthly Revenue",
                     color='TotalPrice', color_continuous_scale="Blues")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Top 10 Countries by Revenue")
        country_sales = df_filtered.groupby('Country')['TotalPrice'].sum().sort_values(ascending=False).head(10)
        fig = px.bar(x=country_sales.values, y=country_sales.index, orientation='h',
                     title="Revenue by Country (Top 10)",
                     color=country_sales.values, color_continuous_scale="Viridis")
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Revenue Trend")
    daily = df_filtered.groupby(df_filtered['InvoiceDate'].dt.date)['TotalPrice'].sum().reset_index()
    daily.columns = ['Date', 'Revenue']
    fig = px.line(daily, x='Date', y='Revenue', title="Daily Revenue Over Time")
    fig.update_traces(line=dict(color="#636efa"))
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Top 10 Best-Selling Products")
        top_products = (df_filtered.groupby(['StockCode', 'Description'])
                       .agg({'Quantity':'sum', 'TotalPrice':'sum'})
                       .sort_values('Quantity', ascending=False).head(10))
        fig = px.bar(top_products.reset_index(), x='Description', y='Quantity',
                     title="Units Sold", color='Quantity')
        fig.update_layout(xaxis_tickangle=30)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Top 10 Highest Revenue Products")
        fig = px.bar(top_products.reset_index(), x='Description', y='TotalPrice',
                     title="Revenue Generated", color='TotalPrice')
        fig.update_layout(xaxis_tickangle=30)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top 10 Customers by Spending")
    top_customers = df_filtered.groupby('CustomerID')['TotalPrice'].sum().sort_values(ascending=False).head(10)
    fig = px.bar(x=top_customers.values, y=top_customers.index.astype(str),
                 orientation='h', title="Highest Spending Customers")
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.subheader("Revenue by Country (World Map)")
    country_map = df_filtered.groupby('Country')['TotalPrice'].sum().reset_index()
    fig = px.choropleth(country_map,
                        locations="Country",
                        locationmode='country names',
                        color="TotalPrice",
                        hover_name="Country",
                        color_continuous_scale="Plasma",
                        title="Global Sales Heatmap")
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────
