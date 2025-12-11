import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Page Configuration
st.set_page_config(
    page_title="Car Evaluation Analytics",
    page_icon="🚗",
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
    
    .success-box {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        padding: 1.2rem;
        border-radius: 10px;
        border-left: 5px solid #22c55e;
        margin: 1rem 0;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background: linear-gradient(135deg, #e2e8f0 0%, #cbd5e1 100%);
        border-radius: 10px;
        padding: 0 24px;
        font-weight: 600;
        color: #1e293b;
        border: 2px solid #cbd5e1;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        border: 2px solid #5a67d8;
    }

    /* FIXED TEXT VISIBILITY — SAFE VERSION */
    p, span, div, label {
        color: #1e293b !important;
    }

</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_car_evaluation_data():
    """Load Car Evaluation dataset from UCI repository"""
    try:
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data"
        columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
        
        with st.spinner("🚗 Loading Car Evaluation dataset from UCI repository..."):
            df = pd.read_csv(url, names=columns, header=None)
        return df
    except Exception as e:
        st.error(f"❌ Error loading dataset: {str(e)}")
        st.info("💡 Please check your internet connection and try again.")
        return None


def get_feature_descriptions():
    return {
        'buying': 'Buying price (vhigh, high, med, low)',
        'maint': 'Maintenance price (vhigh, high, med, low)',
        'doors': 'Number of doors (2, 3, 4, 5more)',
        'persons': 'Capacity in terms of persons (2, 4, more)',
        'lug_boot': 'Size of luggage boot (small, med, big)',
        'safety': 'Estimated safety (low, med, high)',
        'class': 'Car acceptability (unacc, acc, good, vgood)'
    }


def show_overview(df):
    st.markdown("## 📊 Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Total Cars</h3>
            <h2>{len(df):,}</h2>
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Features</h3>
            <h2>{len(df.columns)-1}</h2>
        </div>""", unsafe_allow_html=True)

    with col3:
        acceptable = df[df['class'].isin(['acc', 'good', 'vgood'])].shape[0]
        st.markdown(f"""
        <div class="metric-card">
            <h3>Acceptable</h3>
            <h2>{acceptable}</h2>
        </div>""", unsafe_allow_html=True)

    with col4:
        unacceptable = df[df['class'] == 'unacc'].shape[0]
        st.markdown(f"""
        <div class="metric-card">
            <h3>Unacceptable</h3>
            <h2>{unacceptable}</h2>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("""
    <div class="info-box">
        <h3>🚗 About the Car Evaluation Dataset</h3>
        <p><strong
