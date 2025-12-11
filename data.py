import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
# The following ML/PCA imports are kept but the functions are removed/simplified 
# as they are not suitable for the Online Retail dataset without major changes.
# Keeping the imports ensures the app works if you later re-add generic ML logic.
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import warnings
import io
warnings.filterwarnings('ignore')

# Configuration
APP_CONFIG = {
    "title": "🛒 Online Retail Transaction Data Analyzer",
    "version": "1.0",
    "description": "Exploratory Data Analysis of the UCI Online Retail Dataset"
}

# --- CORE FUNCTION: DATA LOADING ---
@st.cache_data
def load_online_retail_dataset():
    """Load the Online Retail dataset directly from the UCI Excel link."""
    st.info("🔄 Attempting to load Online Retail data from UCI. This may take a moment...")
    
    # Direct link to the Excel file from the UCI dataset page (ID 352)
    # NOTE: pd.read_excel requires 'openpyxl' to be installed (must be in requirements.txt)
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"
    
    try:
        df = pd.read_excel(url)
        # Drop rows with missing CustomerID (common practice for this dataset)
        df.dropna(subset=['CustomerID'], inplace=True)
        # Ensure CustomerID is an integer
        df['CustomerID'] = df['CustomerID'].astype(int)
        
        st.success(f"✅ Data loaded successfully! Total {len(df)} transactions.")
        return df
    except Exception as e:
        st.error(f"FATAL ERROR LOADING DATA. Check your requirements.txt for 'openpyxl'. Details: {e}")
        return pd.DataFrame() # Return empty DataFrame on failure


# Custom CSS for better styling (kept as provided)
def load_custom_css():
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
    
    .subtitle {
        text-align: center;
        color: #64748b;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        box-shadow: 0 8px 16px rgba(102, 126, 234, 0.2);
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
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
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    .success-box {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        padding: 1.2rem;
        border-radius: 10px;
        border-left: 5px solid #22c55e;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 55px;
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        border-radius: 10px;
        padding: 0 28px;
        font-weight: 600;
        font-size: 0.95rem;
        border: 2px solid transparent;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: linear-gradient(135deg, #e2e8f0 0%, #cbd5e1 100%);
        border-color: #667eea;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: 2px solid #5a67d8;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.6rem 2rem;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(102, 126, 234, 0.4);
    }
    
    .data-source-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        border: 2px solid #e2e8f0;
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    
    .data-source-card:hover {
        border-color: #667eea;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.1);
    }
    
    .section-header {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1e293b;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #667eea;
    }
    
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
    }
    
    .sidebar-section {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    </style>
    """, unsafe_allow_html=True)

# --- SIMPLIFIED ANALYSIS FUNCTIONS FOR ONLINE RETAIL ---

def display_dataset_info():
    """Display information about the Online Retail dataset"""
    st.markdown("""
    <div class="info-box">
        <h3>🛒 About the Online Retail Dataset</h3>
        <p><strong>Source:</strong> UCI Machine Learning Repository (ID 352)</p>
        <p><strong>Description:</strong> A transactional dataset containing all the transactions occurring between 01/12/2010 and 09/12/2011 for a UK-based non-store online retail.</p>
        <p><strong>Key Columns:</strong> InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country</p>
    </div>
    """, unsafe_allow_html=True)

def perform_eda(df):
    """Perform basic exploratory data analysis for Online Retail data"""
    
    st.markdown('<h2 class="section-header">📊 Exploratory Data Analysis</h2>', unsafe_allow_html=True)
    
    df_clean = df.copy()
    
    # Calculate Total Sales
    df_clean['TotalSales'] = df_clean['Quantity'] * df_clean['UnitPrice']
    
    # Create tabs for different analyses
    tab1, tab2, tab3 = st.tabs([
        "📈 Overview",
        "🗺️ Geographic Analysis",
        "💰 Sales Metrics"
    ])
    
    with tab1:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Total Transactions</h3>
                <h2>{len(df_clean):,}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Unique Products</h3>
                <h2>{df_clean['StockCode'].nunique()}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Unique Customers</h3>
                <h2>{df_clean['CustomerID'].nunique()}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Countries</h3>
                <h2>{df_clean['Country'].nunique()}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("### 📋 Data Sample")
        st.dataframe(df_clean.head(10), use_container_width=True, height=300)
        
        st.markdown("### 📊 Statistical Summary (Sales)")
        st.dataframe(df_clean[['Quantity', 'UnitPrice', 'TotalSales']].describe(), use_container_width=True)
        
    with tab2:
        st.markdown("### 🗺️ Top 10 Countries by Transaction Count")
        country_counts = df_clean['Country'].value_counts().head(10).reset_index()
        country_counts.columns = ['Country', 'Count']
        
        fig = px.bar(country_counts, x='Country', y='Count',
                     title='Transaction Volume by Country',
                     color='Count', color_continuous_scale=px.colors.sequential.Plasma)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.markdown("### 📈 Sales Trend Over Time (Monthly)")
        # Prepare data for time series
        sales_trend = df_clean.set_index('InvoiceDate').resample('M')['TotalSales'].sum().reset_index()
        
        fig = px.line(sales_trend, x='InvoiceDate', y='TotalSales',
                      title='Monthly Total Sales',
                      markers=True, line_shape='spline')
        fig.update_traces(line=dict(color='#667eea', width=3))
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### 📦 Top 10 Bestselling Products (by Quantity)")
        top_products = df_clean.groupby('Description')['Quantity'].sum().sort_values(ascending=False).head(10).reset_index()
        
        fig = px.bar(top_products, x='Quantity', y='Description', 
                     orientation='h', 
                     color='Quantity',
                     color_continuous_scale=px.colors.sequential.Viridis,
                     title='Top 10 Products by Total Quantity Sold')
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)


# --- MAIN APPLICATION LOGIC ---

def main():
    load_custom_css()
    
    st.markdown(f'<h1 class="main-header">{APP_CONFIG["title"]}</h1>', unsafe_allow_html=True)
    st.markdown(f'<p class="subtitle">{APP_CONFIG["description"]}</p>', unsafe_allow_html=True)
    
    display_dataset_info()
    
    df = load_online_retail_dataset()
    
    if not df.empty:
        perform_eda(df)
        
        # Display note about ML section removal
        st.markdown("""
        <div class="info-box" style="margin-top: 3rem;">
        <strong>NOTE:</strong> Advanced sections (ML Analysis and PCA) have been removed or commented out 
        because they were based on the Breast Cancer dataset's structure and are not directly compatible 
        with the Online Retail transaction data (which is suited for clustering or association rule mining).
        </div>
        """, unsafe_allow_html=True)
        
        # Placeholders for future development:
        # perform_ml_analysis(df) 
        # perform_pca_analysis(df)

if __name__ == "__main__":
    main()
