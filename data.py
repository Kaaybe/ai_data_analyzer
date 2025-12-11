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
import warnings
import io

warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
APP_CONFIG = {
    "title": "🔬 Breast Cancer Wisconsin Diagnostic Analyzer",
    "version": "3.0",
    "description": "AI-Powered Analysis of Breast Cancer Diagnostic Features"
}

st.set_page_config(
    page_title=APP_CONFIG["title"],
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 1. CUSTOM CSS ---
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

# --- 2. DATA LOADING FUNCTIONS ---

@st.cache_data(show_spinner="Downloading default dataset...")
def load_dataset_from_url():
    """Load the Breast Cancer Wisconsin dataset directly from UCI"""
    try:
        # Direct download URL for the dataset
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
        
        # Column names based on dataset documentation
        column_names = ['ID', 'Diagnosis'] + [
            f'{feature}_{stat}' 
            for feature in ['radius', 'texture', 'perimeter', 'area', 'smoothness', 
                          'compactness', 'concavity', 'concave_points', 'symmetry', 'fractal_dimension']
            for stat in ['mean', 'se', 'worst']
        ]
        
        df = pd.read_csv(url, header=None, names=column_names)
        
        # Drop the 'ID' column as it's not a feature
        df = df.drop(columns=['ID'])
        
        return df
    except Exception as e:
        st.error(f"Error loading dataset from URL. Please check your network connection: {e}")
        return None

# Simplified loading to rely on the robust URL loader (ucimlrepo requires an extra dependency/install step)
def load_default_dataset():
    """Wrapper for the main data loading function."""
    return load_dataset_from_url()

@st.cache_data(show_spinner="Loading custom dataset...")
def load_dataset_from_custom_url(url):
    """Load dataset from a custom URL provided by user"""
    try:
        df = pd.read_csv(url)
        return df
    except Exception as e:
        st.error(f"Unable to load dataset from URL (tried as CSV). Error: {e}")
        return None

# --- 3. UI/ANALYSIS FUNCTIONS ---

def display_dataset_info():
    """Display information about the dataset"""
    st.markdown("""
    <div class="info-box">
        <h3>🔬 About the Breast Cancer Wisconsin Dataset</h3>
        <p><strong>Source:</strong> UCI Machine Learning Repository</p>
        <p><strong>Instances:</strong> 569 samples</p>
        <p><strong>Features:</strong> 30 numeric features (mean, SE, and worst values for 10 measurements)</p>
        <p><strong>Target:</strong> Diagnosis (B = Benign, M = Malignant)</p>
        <p><strong>Purpose:</strong> Diagnostic prediction based on fine needle aspirate (FNA) images of breast masses</p>
    </div>
    """, unsafe_allow_html=True)

def perform_eda(df):
    """Perform comprehensive exploratory data analysis"""
    
    st.markdown('<h2 class="section-header">📊 Exploratory Data Analysis</h2>', unsafe_allow_html=True)
    
    # Create tabs for different analyses
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Overview", 
        "📉 Distributions", 
        "🔗 Correlations", 
        "📦 Box Plots",
        "🎯 Feature Importance"
    ])
    
    with tab1:
        # Ensure the Diagnosis column exists before calculating metrics
        has_diagnosis = 'Diagnosis' in df.columns
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Total Samples</h3>
                <h2>{len(df)}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Features</h3>
                <h2>{len(df.columns) - 1 if has_diagnosis else len(df.columns)}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            if has_diagnosis:
                benign_count = (df['Diagnosis'] == 'B').sum()
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Benign (B)</h3>
                    <h2>{benign_count}</h2>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Missing Values</h3>
                    <h2>{df.isnull().sum().sum()}</h2>
                </div>
                """, unsafe_allow_html=True)
        
        with col4:
            if has_diagnosis:
                malignant_count = (df['Diagnosis'] == 'M').sum()
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Malignant (M)</h3>
                    <h2>{malignant_count}</h2>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Numeric Cols</h3>
                    <h2>{len(df.select_dtypes(include=[np.number]).columns)}</h2>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("### 📋 Data Sample")
        st.dataframe(df.head(15), use_container_width=True, height=400)
        
        st.markdown("### 📊 Statistical Summary")
        st.dataframe(df.describe().T, use_container_width=True) # Transpose for better view
        
        if has_diagnosis:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("### 🥧 Diagnosis Distribution")
                fig = px.pie(df, names='Diagnosis', 
                             title='Distribution of Diagnosis',
                             color='Diagnosis',
                             color_discrete_map={'B':'#22c55e', 'M':'#ef4444'},
                             hole=0.4)
                fig.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("### 📊 Count by Diagnosis")
                diagnosis_counts = df['Diagnosis'].value_counts().reset_index()
                diagnosis_counts.columns = ['Diagnosis', 'Count']
                fig = px.bar(diagnosis_counts, x='Diagnosis', y='Count',
                             color='Diagnosis',
                             color_discrete_map={'B':'#22c55e', 'M':'#ef4444'},
                             text='Count')
                fig.update_traces(textposition='outside')
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown("### 📊 Feature Distributions")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if numeric_cols:
            col1, col2 = st.columns([2, 1])
            with col1:
                selected_features = st.multiselect(
                    "Select features to visualize:",
                    numeric_cols,
                    default=numeric_cols[:3] if len(numeric_cols) >= 3 else numeric_cols
                )
            with col2:
                chart_type = st.radio("Chart type:", ["Histogram", "Violin Plot", "Both"])
            
            if selected_features:
                for feature in selected_features:
                    st.markdown(f"#### {feature}")
                    if chart_type == "Both":
                        fig = make_subplots(
                            rows=1, cols=2,
                            subplot_titles=(f'{feature} - Distribution', f'{feature} - Box Plot')
                        )
                        
                        fig.add_trace(
                            go.Histogram(x=df[feature], name=feature, nbinsx=30, 
                                         marker_color='#667eea'),
                            row=1, col=1
                        )
                        
                        fig.add_trace(
                            go.Box(y=df[feature], name=feature, marker_color='#764ba2'),
                            row=1, col=2
                        )
                        
                        fig.update_layout(height=350, showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)
                    elif chart_type == "Histogram":
                        fig = px.histogram(df, x=feature, nbins=30, 
                                             title=f'{feature} Distribution',
                                             color_discrete_sequence=['#667eea'])
                        st.plotly_chart(fig, use_container_width=True)
                    else:  # Violin Plot
                        fig = px.violin(df, y=feature, box=True, 
                                         title=f'{feature} Distribution',
                                         color_discrete_sequence=['#764ba2'])
                        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown("### 🔗 Correlation Analysis")
        numeric_df = df.select_dtypes(include=[np.number])
        
        if not numeric_df.empty:
            col1, col2 = st.columns([3, 1])
            
            with col2:
                st.markdown("#### Settings")
                show_values = st.checkbox("Show correlation values", value=False)
                color_scale = st.selectbox("Color scheme:", 
                                            ["RdBu_r", "Viridis", "Plasma", "Turbo"])
            
            with col1:
                corr_matrix = numeric_df.corr()
                
                fig = px.imshow(corr_matrix,
                                 text_auto='.2f' if show_values else False,
                                 aspect='auto',
                                 color_continuous_scale=color_scale,
                                 title='Feature Correlation Heatmap')
                fig.update_layout(height=700)
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("### 🔍 Highly Correlated Features")
            threshold = st.slider("Correlation threshold:", 0.5, 1.0, 0.8, 0.05)
            
            high_corr = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if abs(corr_matrix.iloc[i, j]) > threshold:
                        high_corr.append({
                            'Feature 1': corr_matrix.columns[i],
                            'Feature 2': corr_matrix.columns[j],
                            'Correlation': round(corr_matrix.iloc[i, j], 3)
                        })
            
            if high_corr:
                high_corr_df = pd.DataFrame(high_corr).sort_values('Correlation', 
                                                                    key=abs, 
                                                                    ascending=False)
                st.dataframe(high_corr_df, use_container_width=True)
            else:
                st.info(f"No correlations above {threshold}")
    
    with tab4:
        st.markdown("### 📦 Feature Comparison")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if numeric_cols:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                selected_feature = st.selectbox(
                    "Select feature for comparison:",
                    numeric_cols
                )
            
            with col2:
                plot_type = st.radio("Plot type:", ["Box", "Violin", "Strip"])
            
            if has_diagnosis and selected_feature:
                if plot_type == "Box":
                    fig = px.box(df, x='Diagnosis', y=selected_feature,
                                 color='Diagnosis',
                                 title=f'{selected_feature} by Diagnosis',
                                 color_discrete_map={'B':'#22c55e', 'M':'#ef4444'},
                                 points="all")
                elif plot_type == "Violin":
                    fig = px.violin(df, x='Diagnosis', y=selected_feature,
                                     color='Diagnosis',
                                     title=f'{selected_feature} by Diagnosis',
                                     color_discrete_map={'B':'#22c55e', 'M':'#ef4444'},
                                     box=True)
                else:  # Strip
                    fig = px.strip(df, x='Diagnosis', y=selected_feature,
                                     color='Diagnosis',
                                     title=f'{selected_feature} by Diagnosis',
                                     color_discrete_map={'B':'#22c55e', 'M':'#ef4444'})
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Statistical test
                benign = df[df['Diagnosis'] == 'B'][selected_feature].dropna()
                malignant = df[df['Diagnosis'] == 'M'][selected_feature].dropna()
                
                if len(benign) > 0 and len(malignant) > 0:
                    t_stat, p_value = stats.ttest_ind(benign, malignant)
                    
                    st.markdown(f"""
                    <div class="{'success-box' if p_value < 0.05 else 'info-box'}">
                    <strong>📊 Statistical Test (T-Test):</strong><br><br>
                    <strong>T-statistic:</strong> {t_stat:.4f}<br>
                    <strong>P-value:</strong> {p_value:.4e}<br>
                    <strong>Result:</strong> {'<span style="color: #22c55e; font-weight: 600;">Significant difference detected! (p < 0.05)</span>' if p_value < 0.05 else '<span style="color: #f59e0b; font-weight: 600;">No significant difference (p ≥ 0.05)</span>'}
                    </div>
                    """, unsafe_allow_html=True)
            elif selected_feature:
                fig = px.histogram(df, x=selected_feature, nbins=30,
                                 title=f'{selected_feature} Distribution')
                st.plotly_chart(fig, use_container_width=True)
    
    with tab5:
        st.markdown("### 🎯 Feature Importance Analysis")
        
        if has_diagnosis:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if numeric_cols and len(numeric_cols) > 1:
                col1, col2 = st.columns([3, 1])
                
                with col2:
                    top_n = st.slider("Number of features:", 5, 20, 15)
                    show_all = st.checkbox("Show all features")
                
                # Use .copy() to avoid SettingWithCopyWarning
                X = df[numeric_cols].fillna(df[numeric_cols].mean()).copy()
                y = df['Diagnosis'].map({'B': 0, 'M': 1})
                
                with st.spinner("Computing feature importance..."):
                    rf = RandomForestClassifier(n_estimators=100, random_state=42)
                    rf.fit(X, y)
                
                importance_df = pd.DataFrame({
                    'Feature': numeric_cols,
                    'Importance': rf.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                if not show_all:
                    importance_df = importance_df.head(top_n)
                
                with col1:
                    fig = px.bar(importance_df, 
                                 x='Importance', 
                                 y='Feature',
                                 orientation='h',
                                 title=f'Top {len(importance_df)} Most Important Features',
                                 color='Importance',
                                 color_continuous_scale='viridis')
                    fig.update_layout(height=max(400, len(importance_df) * 25))
                    st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("### 📋 Feature Importance Table")
                st.dataframe(importance_df.reset_index(drop=True), use_container_width=True)
            else:
                st.warning("Not enough numeric features to calculate importance.")
        else:
            st.warning("Cannot perform Feature Importance Analysis: 'Diagnosis' column is missing.")

def perform_ml_analysis(df):
    """Perform machine learning analysis"""
    
    st.markdown('<h2 class="section-header">🤖 Machine Learning Analysis</h2>', unsafe_allow_html=True)
    
    if 'Diagnosis' not in df.columns:
        st.warning("⚠️ Diagnosis column not found. ML analysis requires a target variable.")
        return
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 2:
        st.warning("⚠️ Not enough numeric features for ML analysis.")
        return
    
    # Sidebar for ML configuration
    with st.expander("⚙️ Model Configuration", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            test_size = st.slider("Test set size:", 0.1, 0.5, 0.2, 0.05)
        with col2:
            n_estimators = st.slider("Number of trees:", 50, 300, 100, 50)
        with col3:
            random_state = st.number_input("Random seed:", 1, 100, 42)
    
    # Prepare data
    X = df[numeric_cols].fillna(df[numeric_cols].mean())
    y = df['Diagnosis'].map({'B': 0, 'M': 1})
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    with st.spinner("🔄 Training Random Forest model..."):
        rf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
        rf.fit(X_train_scaled, y_train)
        
        train_score = rf.score(X_train_scaled, y_train)
        test_score = rf.score(X_test_scaled, y_test)
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Training Accuracy</h3>
            <h2>{train_score*100:.2f}%</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Testing Accuracy</h3>
            <h2>{test_score*100:.2f}%</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Training Samples</h3>
            <h2>{len(X_train)}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Testing Samples</h3>
            <h2>{len(X_test)}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # Predictions
    y_pred = rf.predict(X_test_scaled)
    y_pred_proba = rf.predict_proba(X_test_scaled)[:, 1]
    
    # Create two columns for visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        
        fig = px.imshow(cm, 
                         text_auto=True,
                         labels=dict(x="Predicted", y="Actual"),
                         x=['Benign (0)', 'Malignant (1)'],
                         y=['Benign (0)', 'Malignant (1)'],
                         color_continuous_scale='Blues')
        fig.update_layout(title='Confusion Matrix', height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 📈 ROC Curve")
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr, mode='lines', 
            name=f'ROC (AUC = {roc_auc:.3f})',
            line=dict(color='#667eea', width=3)
        ))
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode='lines', 
            name='Random',
            line=dict(dash='dash', color='gray', width=2)
        ))
        fig.update_layout(
            title='Receiver Operating Characteristic',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Classification Report
    st.markdown("### 📋 Detailed Classification Report")
    report = classification_report(y_test, y_pred, 
                                   target_names=['Benign', 'Malignant'], 
                                   output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    
    # Style the dataframe
    st.dataframe(
        report_df.style.background_gradient(cmap='RdYlGn', subset=['precision', 'recall', 'f1-score']),
        use_container_width=True
    )

def perform_pca_analysis(df):
    """Perform PCA dimensionality reduction"""
    
    st.markdown('<h2 class="section-header">🔬 Principal Component Analysis</h2>', unsafe_allow_html=True)
    
    has_diagnosis = 'Diagnosis' in df.columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 2:
        st.warning("⚠️ Need at least 2 numeric features for PCA.")
        return
    
    X = df[numeric_cols].fillna(df[numeric_cols].mean())
    
    # PCA Configuration
    with st.expander("⚙️ PCA Configuration", expanded=True):
        n_components = st.slider("Number of components:", 2, min(10, len(numeric_cols)), 
                                 min(5, len(numeric_cols)))
        
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Run PCA
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(X_scaled)
    
    # 1. Explained Variance Plot
    st.markdown("### 📉 Explained Variance Ratio (Scree Plot)")
    
    explained_variance = pca.explained_variance_ratio_
    variance_df = pd.DataFrame({
        'Component': [f'PC{i+1}' for i in range(len(explained_variance))],
        'Explained Variance': explained_variance,
        'Cumulative Variance': np.cumsum(explained_variance)
    })
    
    # Plotly Scree Plot
    fig_scree = go.Figure()
    fig_scree.add_trace(go.Bar(
        x=variance_df['Component'],
        y=variance_df['Explained Variance'],
        name='Individual',
        marker_color='#667eea'
    ))
    fig_scree.add_trace(go.Scatter(
        x=variance_df['Component'],
        y=variance_df['Cumulative Variance'],
        name='Cumulative',
        mode='lines+markers',
        line=dict(color='#ef4444', width=2),
        yaxis='y2' # Use a secondary axis
    ))
    
    fig_scree.update_layout(
        title='Scree Plot: Explained Variance by Principal Component',
        xaxis_title='Principal Component',
        yaxis_title='Explained Variance Ratio',
        yaxis2=dict(
            title='Cumulative Explained Variance',
            overlaying='y',
            side='right'
        ),
        height=450
    )
    st.plotly_chart(fig_scree, use_container_width=True)
    
    st.markdown(f"""
    <div class="success-box">
    <strong>Cumulative Variance:</strong> The first **{n_components}** principal components explain **{np.cumsum(explained_variance)[-1]*100:.2f}%** of the total variance.
    </div>
    """, unsafe_allow_html=True)
    
    # 2. PCA Scatter Plot (Only if n_components >= 2)
    if n_components >= 2 and has_diagnosis:
        st.markdown("### 🗺️ 2D PCA Visualization")
        
        pca_2d_df = pd.DataFrame(data=principal_components[:, 0:2], columns=['PC1', 'PC2'])
        pca_2d_df['Diagnosis'] = df['Diagnosis'].values
        
        fig_scatter = px.scatter(
            pca_2d_df,
            x='PC1',
            y='PC2',
            color='Diagnosis',
            color_discrete_map={'B':'#22c55e', 'M':'#ef4444'},
            title='PCA 2D Projection (PC1 vs PC2)',
            hover_data=['Diagnosis']
        )
        fig_scatter.update_layout(height=600)
        st.plotly_chart(fig_scatter, use_container_width=True)
        st.info("This plot shows how well the data separates using the two most important components.")

# --- 4. MAIN FUNCTION ---

def main():
    load_custom_css()
    
    st.markdown(f'<h1 class="main-header">{APP_CONFIG["title"]}</h1>', unsafe_allow_html=True)
    st.markdown(f'<p class="subtitle">{APP_CONFIG["description"]}</p>', unsafe_allow_html=True)

    df = None
    
    # --- Sidebar for Data Source ---
    with st.sidebar:
        st.header("Data Source")
        
        data_source = st.radio(
            "Select Data Source:",
            ("Default Breast Cancer Data (UCI)", "Upload CSV File", "Custom URL"),
            index=0
        )
        
        if data_source == "Default Breast Cancer Data (UCI)":
            df = load_default_dataset()
