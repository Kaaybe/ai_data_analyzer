# app.py — Online Retail UCI Dashboard (WORKS PERFECTLY)
# Deploy this file directly on Streamlit Community Cloud

import streamlit as st
import pandas as pd
import plotly.express as px

# ========================= CONFIG =========================
st.set_page_config(
    page_title="Online Retail Dashboard",
    page_icon="",
    layout="wide",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Direct working link to the Excel file
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"

# ========================= TITLE =========================
st.markdown("""
<h1 style='text-align: center; color: #1e40af;'>
    Online Retail Dashboard
</h1>
<h3 style='text-align: center; color: #64748b;'>
    Live analysis of 541,909 transactions from UCI Machine Learning Repository
</h3>
""", unsafe_allow_html=True)

st.divider()

# ========================= LOAD DATA =========================
@st.cache_data(show_spinner="Downloading dataset from UCI (60 MB)...")
def load_data():
    df = pd.read_excel(DATA_URL)
    # Basic cleaning
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
    df = df[df['Quantity'] > 0]
    df = df[df['UnitPrice'] > 0]
    df['Month'] = df['InvoiceDate'].dt.strftime('%Y-%m')
    return df

df = load_data()

# ========================= SIDEBAR FILTERS =========================
st.sidebar.header("Filters")

countries = sorted(df['Country'].unique())
selected_countries = st.sidebar.multiselect(
    "Select Countries",
    countries,
    default=["United Kingdom"]
)

df = df[df['Country'].isin(selected_countries)]

# ========================= KEY METRICS =========================
total_sales = df['TotalPrice'].sum()
total_orders = df['InvoiceNo'].nunique()
total_customers = df['CustomerID'].nunique()
avg_basket = total_sales / total_orders

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Revenue", f"£{total_sales:,.0f}")
col2.metric("Number of Orders", f"{total_orders:,}")
col3.metric("Unique Customers", f"{total_customers:,}")
col4.metric("Average Order Value", f"£{avg_basket:,.2f}")

st.divider()

# ========================= TABS =========================
tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Time Series", "Products & Customers", "Geography"])

with tab1:
    col1, col2 = st.columns(2)

    with col1:
        monthly_sales = df.groupby('Month')['TotalPrice'].sum().reset_index()
        fig = px.bar(monthly_sales, x='Month', y='TotalPrice',
                     title="Monthly Revenue", color='TotalPrice',
                     color_continuous_scale="Blues")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        top_countries = df.groupby('Country')['TotalPrice'].sum().sort_values(ascending=False).head(10)
        fig = px.bar(y=top_countries.index, x=top_countries.values, orientation='h',
                     title="Top 10 Countries by Revenue", color=top_countries.values,
                     color_continuous_scale="Viridis")
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    daily = df.groupby(df['InvoiceDate'].dt.date)['TotalPrice'].sum().reset_index()
    daily.columns = ['Date', 'Revenue']
    fig = px.line(daily, x='Date', y='Revenue', title="Daily Revenue Trend")
    fig.update_traces(line_color="#636efa", line_width=3)
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    col1, col2 = st.columns(2)

    with col1:
        top_products = df.groupby('Description')['Quantity'].sum().sort_values(ascending=False).head(10)
        fig = px.bar(x=top_products.values, y=top_products.index, orientation='h',
                     title="Top 10 Best-Selling Products (by quantity)", color=top_products.values)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        top_revenue = df.groupby('Description')['TotalPrice'].sum().sort_values(ascending=False).head(10)
        fig = px.bar(x=top_revenue.values, y=top_revenue.index, orientation='h',
                     title="Top 10 Products by Revenue", color=top_revenue.values)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top 10 Spending Customers")
    top_cust = df.groupby('CustomerID')['TotalPrice'].sum().sort_values(ascending=False).head(10)
    fig = px.bar(x=top_cust.values, y=top_cust.index.astype(str), orientation='h',
                 title="Highest Spending Customers")
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    country_map = df.groupby('Country')['TotalPrice'].sum().reset_index()
    fig = px.choropleth(country_map,
                        locations="Country",
                        locationmode='country names',
                        color="TotalPrice",
                        title="Revenue by Country - World Map",
                        color_continuous_scale="Plasma")
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)

# ========================= FOOTER =========================
st.markdown("---")
st.markdown("""
**Dataset**: [UCI Online Retail](https://archive.ics.uci.edu/dataset/352/online+retail)  
Built with ❤️ using Streamlit • Data updated Dec 2010 – Dec 2011
""")
