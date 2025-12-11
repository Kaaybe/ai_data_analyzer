import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from ucimlrepo import fetch_ucirepo # Import necessary for clean data loading
import warnings
warnings.filterwarnings('ignore')

# Configuration
APP_CONFIG = {
    "title": "🚗 Car Evaluation Classification Analyzer",
    "version": "1.0",
    "description": "AI-Powered Analysis of Car Evaluation Decisions (Categorical Data)"
}

# --- CUSTOM CSS (Kept as provided) ---
def load_custom_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    * { font-family: 'Inter', sans-serif; }
    
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

# --- CORE FUNCTION: DATA LOADING (Updated for Car Evaluation ID 19) ---
@st.cache_data
def load_car_evaluation_dataset():
    """Load the Car Evaluation dataset using ucimlrepo"""
    try:
        # Car Evaluation dataset ID is 19
        dataset = fetch_ucirepo(id=19) 
        X = dataset.data.features
        y = dataset.data.targets
        
        # Combine features and target into a single DataFrame
        df = pd.concat([X, y], axis=1)
        
        # The dataset is clean, but columns are named generally by ucimlrepo
        df.columns = ['Buying', 'Maint', 'Doors', 'Persons', 'Lug_Boot', 'Safety', 'Evaluation']
        
        st.success(f"✅ Data loaded successfully! Total {len(df)} car evaluations.")
        return df
    except Exception as e:
        st.error(f"Error loading dataset using ucimlrepo: {e}. Ensure 'ucimlrepo' is in requirements.txt")
        return pd.DataFrame()

def display_dataset_info():
    """Display information about the Car Evaluation dataset"""
    st.markdown("""
    <div class="info-box">
        <h3>🚗 About the Car Evaluation Dataset</h3>
        <p><strong>Source:</strong> UCI Machine Learning Repository (ID 19)</p>
        <p><strong>Instances:</strong> 1728 evaluations</p>
        <p><strong>Features:</strong> 6 categorical attributes (Buying Price, Maintenance Cost, Doors, Persons, Luggage Boot, Safety)</p>
        <p><strong>Target:</strong> Car Evaluation (unacc, acc, good, vgood)</p>
        <p><strong>Purpose:</strong> Predict the acceptance level of a car based on its attributes.</p>
    </div>
    """, unsafe_allow_html=True)

# --- EDA (Revised for Categorical Data) ---

def perform_eda(df):
    """Perform exploratory data analysis for Car Evaluation data"""
    
    st.markdown('<h2 class="section-header">📊 Exploratory Data Analysis</h2>', unsafe_allow_html=True)
    
    # Create tabs for different analyses
    tab1, tab2, tab3 = st.tabs([
        "📈 Overview",
        "🎯 Evaluation Distribution",
        "🔗 Feature Impact"
    ])
    
    with tab1:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Total Samples</h3>
                <h2>{len(df):,}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Features</h3>
                <h2>{len(df.columns)-1}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            unique_evaluations = df['Evaluation'].nunique()
            st.markdown(f"""
            <div class="metric-card">
                <h3>Unique Classes</h3>
                <h2>{unique_evaluations}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Most Common</h3>
                <h2>{df['Evaluation'].mode()[0]}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("### 📋 Data Sample")
        st.dataframe(df.head(10), use_container_width=True, height=300)
        
    with tab2:
        st.markdown("### 🥧 Evaluation Distribution")
        
        # Define specific colors for evaluation classes
        color_map = {'unacc': '#ef4444', 'acc': '#3b82f6', 'good': '#f59e0b', 'vgood': '#22c55e'}
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            fig = px.pie(df, names='Evaluation',
                         title='Overall Car Evaluation Distribution',
                         color='Evaluation',
                         color_discrete_map=color_map,
                         hole=0.4)
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 📊 Count by Evaluation")
            evaluation_counts = df['Evaluation'].value_counts().reset_index()
            evaluation_counts.columns = ['Evaluation', 'Count']
            
            fig = px.bar(evaluation_counts, x='Evaluation', y='Count',
                         color='Evaluation',
                         color_discrete_map=color_map,
                         text='Count',
                         title='Count of Each Evaluation Class')
            fig.update_traces(textposition='outside')
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.markdown("### 📈 Evaluation Split by Feature")
        feature_cols = df.columns[:-1].tolist()
        
        selected_feature = st.selectbox("Select Feature to Compare:", feature_cols, index=5) # Default to Safety
        
        if selected_feature:
            fig = px.histogram(df, x=selected_feature, color='Evaluation',
                               barmode='group',
                               color_discrete_map=color_map,
                               title=f'Evaluation Breakdown by {selected_feature}')
            st.plotly_chart(fig, use_container_width=True)

# --- ML ANALYSIS (Revised for Categorical Classification) ---

def perform_ml_analysis(df):
    """Perform machine learning analysis using Random Forest after encoding."""
    
    st.markdown('<h2 class="section-header">🤖 Machine Learning Classification</h2>', unsafe_allow_html=True)
    
    # 1. Feature Engineering: One-Hot Encoding for all features
    X = df.drop(columns=['Evaluation'])
    y = df['Evaluation']
    
    # Use pandas get_dummies for One-Hot Encoding (OHE)
    X_encoded = pd.get_dummies(X, drop_first=True)
    
    # 2. Sidebar for ML configuration
    with st.expander("⚙️ Model Configuration (Random Forest)", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            test_size = st.slider("Test set size:", 0.1, 0.5, 0.25, 0.05)
        with col2:
            n_estimators = st.slider("Number of trees:", 50, 300, 150, 50)
        with col3:
            random_state = st.number_input("Random seed:", 1, 100, 42)
    
    # 3. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # 4. Train model
    with st.spinner("🔄 Training Random Forest classifier..."):
        rf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
        rf.fit(X_train, y_train)
        
        train_score = rf.score(X_train, y_train)
        test_score = rf.score(X_test, y_test)
    
    # 5. Display metrics
    col1, col2, col3, col4 = st.columns(4)
