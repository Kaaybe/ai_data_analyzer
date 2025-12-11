import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Page Configuration
st.set_page_config(
    page_title="Online Retail Analytics",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(120deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 1rem 0;
        text-align: center;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        box-shadow: 0 8px 16px rgba(102, 126, 234, 0.2);
        text-align: center;
    }
    
    .metric-card h3 {
        font-size: 0.9rem;
        font-weight: 500;
        margin: 0 0 0.5rem 0;
        opacity: 0.9;
    }
    
    .metric-card h2 {
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
    }
    
    .info-box {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        padding: 1.2rem;
        border-radius: 10px;
        border-left: 5px solid #0ea5e9;
        margin: 1rem 0;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        border-radius: 10px;
        padding: 0 24px;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def load_online_retail_data():
    """Load Online Retail dataset from UCI repository"""
    try:
        # Direct link to the Excel file
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"
        
        with st.spinner("📥 Loading Online Retail dataset from UCI repository..."):
            df = pd.read_excel(url)
            
        # Basic cleaning
        df = df.dropna(subset=['CustomerID'])
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
        df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
        
        # Remove cancelled orders (negative quantities)
        df = df[df['Quantity'] > 0]
        df = df[df['UnitPrice'] > 0]
        
        return df
    except Exception as e:
        st.error(f"❌ Error loading dataset: {str(e)}")
        st.info("💡 Please check your internet connection and try again.")
        return None

def show_overview(df):
    """Display overview statistics"""
    st.markdown("## 📊 Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Total Transactions</h3>
            <h2>{len(df):,}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Unique Customers</h3>
            <h2>{df['CustomerID'].nunique():,}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Total Revenue</h3>
            <h2>£{df['TotalPrice'].sum():,.0f}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Unique Products</h3>
            <h2>{df['StockCode'].nunique():,}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Dataset info
    st.markdown("""
    <div class="info-box">
        <h3>🛒 About the Online Retail Dataset</h3>
        <p><strong>Source:</strong> UCI Machine Learning Repository</p>
        <p><strong>Description:</strong> Transactional data from a UK-based online retail company (2010-2011)</p>
        <p><strong>Contains:</strong> All transactions between 01/12/2010 and 09/12/2011</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📅 Date Range")
        st.write(f"**Start:** {df['InvoiceDate'].min().strftime('%Y-%m-%d')}")
        st.write(f"**End:** {df['InvoiceDate'].max().strftime('%Y-%m-%d')}")
    
    with col2:
        st.markdown("### 🌍 Countries Served")
        st.write(f"**Total Countries:** {df['Country'].nunique()}")
        top_countries = df['Country'].value_counts().head(5)
        st.write("**Top 5:**")
        for country, count in top_countries.items():
            st.write(f"- {country}: {count:,} transactions")
    
    st.markdown("### 📋 Sample Data")
    st.dataframe(df.head(10), use_container_width=True)

def show_sales_analysis(df):
    """Display sales analysis"""
    st.markdown("## 💰 Sales Analysis")
    
    # Time-based analysis
    df['Year'] = df['InvoiceDate'].dt.year
    df['Month'] = df['InvoiceDate'].dt.month
    df['YearMonth'] = df['InvoiceDate'].dt.to_period('M').astype(str)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📈 Revenue Over Time")
        monthly_revenue = df.groupby('YearMonth')['TotalPrice'].sum().reset_index()
        fig = px.line(monthly_revenue, x='YearMonth', y='TotalPrice',
                     title='Monthly Revenue Trend',
                     labels={'TotalPrice': 'Revenue (£)', 'YearMonth': 'Month'})
        fig.update_traces(line_color='#667eea', line_width=3)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 📦 Orders Over Time")
        monthly_orders = df.groupby('YearMonth')['InvoiceNo'].nunique().reset_index()
        fig = px.bar(monthly_orders, x='YearMonth', y='InvoiceNo',
                    title='Monthly Order Count',
                    labels={'InvoiceNo': 'Number of Orders', 'YearMonth': 'Month'},
                    color='InvoiceNo',
                    color_continuous_scale='viridis')
        st.plotly_chart(fig, use_container_width=True)
    
    # Top products
    st.markdown("### 🏆 Top Products by Revenue")
    top_products = df.groupby('Description')['TotalPrice'].sum().sort_values(ascending=False).head(10).reset_index()
    fig = px.bar(top_products, x='TotalPrice', y='Description',
                orientation='h',
                title='Top 10 Products by Revenue',
                labels={'TotalPrice': 'Revenue (£)', 'Description': 'Product'},
                color='TotalPrice',
                color_continuous_scale='blues')
    st.plotly_chart(fig, use_container_width=True)
    
    # Revenue by country
    st.markdown("### 🌍 Revenue by Country")
    country_revenue = df.groupby('Country')['TotalPrice'].sum().sort_values(ascending=False).head(10).reset_index()
    fig = px.bar(country_revenue, x='Country', y='TotalPrice',
                title='Top 10 Countries by Revenue',
                labels={'TotalPrice': 'Revenue (£)'},
                color='TotalPrice',
                color_continuous_scale='viridis')
    st.plotly_chart(fig, use_container_width=True)

def show_customer_analysis(df):
    """Display customer analysis"""
    st.markdown("## 👥 Customer Analysis")
    
    # Customer metrics
    customer_stats = df.groupby('CustomerID').agg({
        'InvoiceNo': 'nunique',
        'TotalPrice': 'sum',
        'Quantity': 'sum'
    }).reset_index()
    customer_stats.columns = ['CustomerID', 'Orders', 'Revenue', 'Items']
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Avg Orders/Customer</h3>
            <h2>{customer_stats['Orders'].mean():.1f}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Avg Revenue/Customer</h3>
            <h2>£{customer_stats['Revenue'].mean():.0f}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Avg Items/Customer</h3>
            <h2>{customer_stats['Items'].mean():.0f}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 💎 Top Customers by Revenue")
        top_customers = customer_stats.nlargest(10, 'Revenue')
        fig = px.bar(top_customers, x='CustomerID', y='Revenue',
                    title='Top 10 Customers',
                    labels={'Revenue': 'Total Revenue (£)'},
                    color='Revenue',
                    color_continuous_scale='purples')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 📊 Customer Segmentation")
        fig = px.histogram(customer_stats, x='Revenue', nbins=50,
                          title='Customer Revenue Distribution',
                          labels={'Revenue': 'Revenue (£)', 'count': 'Number of Customers'},
                          color_discrete_sequence=['#764ba2'])
        st.plotly_chart(fig, use_container_width=True)
    
    # RFM Analysis Preview
    st.markdown("### 🎯 Customer Behavior Metrics")
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.scatter(customer_stats, x='Orders', y='Revenue',
                        title='Orders vs Revenue',
                        labels={'Orders': 'Number of Orders', 'Revenue': 'Total Revenue (£)'},
                        color='Items',
                        color_continuous_scale='viridis',
                        hover_data=['CustomerID'])
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.box(customer_stats, y='Revenue',
                    title='Revenue Distribution',
                    labels={'Revenue': 'Revenue (£)'},
                    color_discrete_sequence=['#667eea'])
        st.plotly_chart(fig, use_container_width=True)

def show_product_analysis(df):
    """Display product analysis"""
    st.markdown("## 📦 Product Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🔥 Most Popular Products")
        top_products_qty = df.groupby('Description')['Quantity'].sum().sort_values(ascending=False).head(10).reset_index()
        fig = px.bar(top_products_qty, x='Quantity', y='Description',
                    orientation='h',
                    title='Top 10 Products by Quantity Sold',
                    labels={'Quantity': 'Total Quantity Sold'},
                    color='Quantity',
                    color_continuous_scale='reds')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 💰 Highest Value Products")
        avg_price = df.groupby('Description')['UnitPrice'].mean().sort_values(ascending=False).head(10).reset_index()
        fig = px.bar(avg_price, x='UnitPrice', y='Description',
                    orientation='h',
                    title='Top 10 Products by Average Price',
                    labels={'UnitPrice': 'Average Price (£)'},
                    color='UnitPrice',
                    color_continuous_scale='greens')
        st.plotly_chart(fig, use_container_width=True)
    
    # Price distribution
    st.markdown("### 💵 Price Distribution")
    price_data = df[df['UnitPrice'] < df['UnitPrice'].quantile(0.95)]  # Remove outliers for better visualization
    fig = px.histogram(price_data, x='UnitPrice', nbins=50,
                      title='Product Price Distribution (95th percentile)',
                      labels={'UnitPrice': 'Unit Price (£)', 'count': 'Frequency'},
                      color_discrete_sequence=['#667eea'])
    st.plotly_chart(fig, use_container_width=True)

def main():
    # Header
    st.markdown('<h1 class="main-header">🛒 Online Retail Analytics Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #64748b; font-size: 1.1rem; margin-bottom: 2rem;">Comprehensive Analysis of E-Commerce Transactions</p>', unsafe_allow_html=True)
    
    # Load data
    df = load_online_retail_data()
    
    if df is None:
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 🎛️ Dashboard Controls")
        st.markdown("---")
        
        # Date filter
        st.markdown("### 📅 Date Range Filter")
        min_date = df['InvoiceDate'].min().date()
        max_date = df['InvoiceDate'].max().date()
        
        date_range = st.date_input(
            "Select date range:",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
        
        if len(date_range) == 2:
            mask = (df['InvoiceDate'].dt.date >= date_range[0]) & (df['InvoiceDate'].dt.date <= date_range[1])
            df = df[mask]
        
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.info("This dashboard analyzes the Online Retail dataset from UCI, containing transactions from a UK-based company.")
        
        st.markdown("### 📊 Data Quality")
        st.write(f"**Records:** {len(df):,}")
        st.write(f"**Missing Values:** {df.isnull().sum().sum()}")
    
    # Main tabs
    tabs = st.tabs(["📊 Overview", "💰 Sales", "👥 Customers", "📦 Products"])
    
    with tabs[0]:
        show_overview(df)
    
    with tabs[1]:
        show_sales_analysis(df)
    
    with tabs[2]:
        show_customer_analysis(df)
    
    with tabs[3]:
        show_product_analysis(df)

if __name__ == "__main__":
    main()
