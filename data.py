import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import warnings
warnings.filterwarnings('ignore')

# Configuration
APP_CONFIG = {
    "title": "🔬 Breast Cancer Wisconsin Diagnostic Analyzer",
    "version": "2.0",
    "description": "AI-Powered Analysis of Breast Cancer Diagnostic Features"
}

# Custom CSS for better styling
def load_custom_css():
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(120deg, #a1c4fd 0%, #c2e9fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: #f0f2f6;
        border-radius: 8px 8px 0 0;
        padding: 0 24px;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background-color: #667eea;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

def load_default_dataset():
    """Load the Breast Cancer Wisconsin dataset"""
    try:
        from ucimlrepo import fetch_ucirepo
        dataset = fetch_ucirepo(id=17)
        X = dataset.data.features
        y = dataset.data.targets
        df = pd.concat([X, y], axis=1)
        return df
    except:
        st.error("Unable to load default dataset. Please upload your own CSV file.")
        return None

def get_feature_info():
    """Return information about features in the dataset"""
    return {
        "radius": "Mean of distances from center to points on the perimeter",
        "texture": "Standard deviation of gray-scale values",
        "perimeter": "Perimeter of the cell nucleus",
        "area": "Area of the cell nucleus",
        "smoothness": "Local variation in radius lengths",
        "compactness": "Perimeter² / area - 1.0",
        "concavity": "Severity of concave portions of the contour",
        "concave_points": "Number of concave portions of the contour",
        "symmetry": "Symmetry of the cell nucleus",
        "fractal_dimension": "Coastline approximation - 1"
    }

def perform_eda(df):
    """Perform comprehensive exploratory data analysis"""
    
    st.markdown("## 📊 Exploratory Data Analysis")
    
    # Create tabs for different analyses
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Overview", 
        "📉 Distributions", 
        "🔗 Correlations", 
        "📦 Box Plots",
        "🎯 Feature Importance"
    ])
    
    with tab1:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h3>Total Samples</h3>
                <h2>{}</h2>
            </div>
            """.format(len(df)), unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <h3>Features</h3>
                <h2>{}</h2>
            </div>
            """.format(len(df.columns)-1), unsafe_allow_html=True)
        
        with col3:
            if 'Diagnosis' in df.columns:
                benign_count = (df['Diagnosis'] == 'B').sum()
                st.markdown("""
                <div class="metric-card">
                    <h3>Benign (B)</h3>
                    <h2>{}</h2>
                </div>
                """.format(benign_count), unsafe_allow_html=True)
        
        with col4:
            if 'Diagnosis' in df.columns:
                malignant_count = (df['Diagnosis'] == 'M').sum()
                st.markdown("""
                <div class="metric-card">
                    <h3>Malignant (M)</h3>
                    <h2>{}</h2>
                </div>
                """.format(malignant_count), unsafe_allow_html=True)
        
        st.markdown("### 📋 Data Sample")
        st.dataframe(df.head(10), use_container_width=True)
        
        st.markdown("### 📊 Statistical Summary")
        st.dataframe(df.describe(), use_container_width=True)
        
        if 'Diagnosis' in df.columns:
            st.markdown("### 🥧 Diagnosis Distribution")
            fig = px.pie(df, names='Diagnosis', 
                        title='Distribution of Diagnosis (B=Benign, M=Malignant)',
                        color='Diagnosis',
                        color_discrete_map={'B':'#00cc96', 'M':'#ef553b'})
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown("### 📊 Feature Distributions")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if numeric_cols:
            selected_features = st.multiselect(
                "Select features to visualize:",
                numeric_cols[:10],
                default=numeric_cols[:3]
            )
            
            if selected_features:
                for feature in selected_features:
                    fig = make_subplots(
                        rows=1, cols=2,
                        subplot_titles=(f'{feature} - Distribution', f'{feature} - Box Plot')
                    )
                    
                    fig.add_trace(
                        go.Histogram(x=df[feature], name=feature, nbinsx=30),
                        row=1, col=1
                    )
                    
                    fig.add_trace(
                        go.Box(y=df[feature], name=feature),
                        row=1, col=2
                    )
                    
                    fig.update_layout(height=300, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown("### 🔗 Correlation Matrix")
        numeric_df = df.select_dtypes(include=[np.number])
        
        if not numeric_df.empty:
            corr_matrix = numeric_df.corr()
            
            fig = px.imshow(corr_matrix,
                          text_auto='.2f',
                          aspect='auto',
                          color_continuous_scale='RdBu_r',
                          title='Feature Correlation Heatmap')
            fig.update_layout(height=600)
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
                st.dataframe(pd.DataFrame(high_corr), use_container_width=True)
            else:
                st.info(f"No correlations above {threshold}")
    
    with tab4:
        st.markdown("### 📦 Feature Comparison by Diagnosis")
        
        if 'Diagnosis' in df.columns:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if numeric_cols:
                selected_feature = st.selectbox(
                    "Select feature for box plot comparison:",
                    numeric_cols[:10]
                )
                
                fig = px.box(df, x='Diagnosis', y=selected_feature,
                           color='Diagnosis',
                           title=f'{selected_feature} Distribution by Diagnosis',
                           color_discrete_map={'B':'#00cc96', 'M':'#ef553b'})
                st.plotly_chart(fig, use_container_width=True)
                
                # Statistical test
                if selected_feature:
                    benign = df[df['Diagnosis'] == 'B'][selected_feature].dropna()
                    malignant = df[df['Diagnosis'] == 'M'][selected_feature].dropna()
                    
                    t_stat, p_value = stats.ttest_ind(benign, malignant)
                    
                    st.markdown(f"""
                    <div class="info-box">
                    <strong>T-Test Results:</strong><br>
                    T-statistic: {t_stat:.4f}<br>
                    P-value: {p_value:.4e}<br>
                    {'<strong>Significant difference detected!</strong>' if p_value < 0.05 else 'No significant difference'}
                    </div>
                    """, unsafe_allow_html=True)
    
    with tab5:
        st.markdown("### 🎯 Feature Importance Analysis")
        
        if 'Diagnosis' in df.columns:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if numeric_cols and len(numeric_cols) > 1:
                X = df[numeric_cols].fillna(df[numeric_cols].mean())
                y = df['Diagnosis'].map({'B': 0, 'M': 1})
                
                rf = RandomForestClassifier(n_estimators=100, random_state=42)
                rf.fit(X, y)
                
                importance_df = pd.DataFrame({
                    'Feature': numeric_cols,
                    'Importance': rf.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                fig = px.bar(importance_df.head(15), 
                           x='Importance', 
                           y='Feature',
                           orientation='h',
                           title='Top 15 Most Important Features',
                           color='Importance',
                           color_continuous_scale='viridis')
                st.plotly_chart(fig, use_container_width=True)

def perform_ml_analysis(df):
    """Perform machine learning analysis"""
    
    st.markdown("## 🤖 Machine Learning Analysis")
    
    if 'Diagnosis' not in df.columns:
        st.warning("Diagnosis column not found. ML analysis requires a target variable.")
        return
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 2:
        st.warning("Not enough numeric features for ML analysis.")
        return
    
    # Prepare data
    X = df[numeric_cols].fillna(df[numeric_cols].mean())
    y = df['Diagnosis'].map({'B': 0, 'M': 1})
    
    # Train-test split
    test_size = st.slider("Test set size:", 0.1, 0.5, 0.2, 0.05)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    with st.spinner("Training Random Forest model..."):
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train_scaled, y_train)
        
        train_score = rf.score(X_train_scaled, y_train)
        test_score = rf.score(X_test_scaled, y_test)
    
    col1, col2 = st.columns(2)
    
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
    
    # Predictions
    y_pred = rf.predict(X_test_scaled)
    y_pred_proba = rf.predict_proba(X_test_scaled)[:, 1]
    
    # Confusion Matrix
    st.markdown("### 📊 Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    
    fig = px.imshow(cm, 
                    text_auto=True,
                    labels=dict(x="Predicted", y="Actual"),
                    x=['Benign', 'Malignant'],
                    y=['Benign', 'Malignant'],
                    color_continuous_scale='Blues')
    fig.update_layout(title='Confusion Matrix')
    st.plotly_chart(fig, use_container_width=True)
    
    # ROC Curve
    st.markdown("### 📈 ROC Curve")
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC curve (AUC = {roc_auc:.2f})'))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', line=dict(dash='dash')))
    fig.update_layout(
        title='Receiver Operating Characteristic (ROC) Curve',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Classification Report
    st.markdown("### 📋 Classification Report")
    report = classification_report(y_test, y_pred, target_names=['Benign', 'Malignant'], output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df, use_container_width=True)

def perform_pca_analysis(df):
    """Perform PCA dimensionality reduction"""
    
    st.markdown("## 🔬 Principal Component Analysis (PCA)")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 2:
        st.warning("Need at least 2 numeric features for PCA.")
        return
    
    X = df[numeric_cols].fillna(df[numeric_cols].mean())
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # PCA
    n_components = min(10, len(numeric_cols))
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    
    # Explained variance
    st.markdown("### 📊 Explained Variance Ratio")
    var_df = pd.DataFrame({
        'PC': [f'PC{i+1}' for i in range(n_components)],
        'Variance': pca.explained_variance_ratio_,
        'Cumulative': np.cumsum(pca.explained_variance_ratio_)
    })
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Bar(x=var_df['PC'], y=var_df['Variance'], name='Individual'),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(x=var_df['PC'], y=var_df['Cumulative'], 
                  mode='lines+markers', name='Cumulative'),
        secondary_y=True
    )
    
    fig.update_layout(title='PCA Explained Variance', height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # 2D visualization
    if 'Diagnosis' in df.columns and n_components >= 2:
        st.markdown("### 🎨 2D PCA Visualization")
        
        pca_df = pd.DataFrame({
            'PC1': X_pca[:, 0],
            'PC2': X_pca[:, 1],
            'Diagnosis': df['Diagnosis'].values
        })
        
        fig = px.scatter(pca_df, x='PC1', y='PC2', color='Diagnosis',
                        title='First Two Principal Components',
                        color_discrete_map={'B':'#00cc96', 'M':'#ef553b'})
        st.plotly_chart(fig, use_container_width=True)

def main():
    st.set_page_config(
        page_title=APP_CONFIG["title"],
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    load_custom_css()
    
    # Header
    st.markdown(f'<h1 class="main-header">{APP_CONFIG["title"]}</h1>', unsafe_allow_html=True)
    st.markdown(f"**{APP_CONFIG['description']}** | Version {APP_CONFIG['version']}")
    
    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        
        data_source = st.radio(
            "Select Data Source:",
            ["Use Default Dataset", "Upload CSV File"]
        )
        
        st.markdown("---")
        st.markdown("### 📖 About")
        st.info("""
        This app analyzes the **Breast Cancer Wisconsin (Diagnostic)** dataset.
        
        **Features:** 30 numeric features computed from cell nucleus images
        
        **Target:** Diagnosis (B=Benign, M=Malignant)
        """)
        
        st.markdown("---")
        st.markdown("### 🔗 Resources")
        st.markdown("[UCI ML Repository](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)")
    
    # Load data
    df = None
    
    if data_source == "Use Default Dataset":
        with st.spinner("Loading default dataset..."):
            df = load_default_dataset()
            if df is not None:
                st.success("✅ Default dataset loaded successfully!")
    else:
        uploaded_file = st.file_uploader("Upload a CSV file", type=['csv'])
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success("✅ File uploaded successfully!")
            except Exception as e:
                st.error(f"Error reading file: {e}")
    
    # Analysis
    if df is not None:
        st.markdown("---")
        
        analysis_type = st.selectbox(
            "Select Analysis Type:",
            ["Exploratory Data Analysis", "Machine Learning Analysis", "PCA Analysis"]
        )
        
        if analysis_type == "Exploratory Data Analysis":
            perform_eda(df)
        elif analysis_type == "Machine Learning Analysis":
            perform_ml_analysis(df)
        elif analysis_type == "PCA Analysis":
            perform_pca_analysis(df)
    else:
        st.info("👆 Please select a data source from the sidebar to begin analysis.")
        
        # Show feature information
        st.markdown("## 📚 Dataset Features")
        feature_info = get_feature_info()
        
        for feature, description in feature_info.items():
            st.markdown(f"**{feature}:** {description}")

if __name__ == "__main__":
    main()
