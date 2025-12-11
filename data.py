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
def load_car_evaluation_data():
    """Load Car Evaluation dataset from UCI repository"""
    try:
        # Direct link to the car evaluation dataset
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data"
        
        # Column names based on dataset documentation
        columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
        
        with st.spinner("🚗 Loading Car Evaluation dataset from UCI repository..."):
            df = pd.read_csv(url, names=columns, header=None)
            
        return df
    except Exception as e:
        st.error(f"❌ Error loading dataset: {str(e)}")
        st.info("💡 Please check your internet connection and try again.")
        return None

def get_feature_descriptions():
    """Return descriptions of all features"""
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
    """Display overview statistics"""
    st.markdown("## 📊 Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Total Cars</h3>
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
        acceptable = df[df['class'].isin(['acc', 'good', 'vgood'])].shape[0]
        st.markdown(f"""
        <div class="metric-card">
            <h3>Acceptable</h3>
            <h2>{acceptable}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        unacceptable = df[df['class'] == 'unacc'].shape[0]
        st.markdown(f"""
        <div class="metric-card">
            <h3>Unacceptable</h3>
            <h2>{unacceptable}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Dataset info
    st.markdown("""
    <div class="info-box">
        <h3>🚗 About the Car Evaluation Dataset</h3>
        <p><strong>Source:</strong> UCI Machine Learning Repository</p>
        <p><strong>Created by:</strong> Marko Bohanec</p>
        <p><strong>Purpose:</strong> Derived from a simple hierarchical decision model for car evaluation</p>
        <p><strong>Instances:</strong> 1728 (complete set of all possible combinations)</p>
        <p><strong>Task:</strong> Classification of cars into acceptability classes</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature descriptions
    st.markdown("### 📋 Feature Descriptions")
    descriptions = get_feature_descriptions()
    
    desc_df = pd.DataFrame([
        {'Feature': k, 'Description': v} 
        for k, v in descriptions.items()
    ])
    st.dataframe(desc_df, use_container_width=True, hide_index=True)
    
    # Sample data
    st.markdown("### 🔍 Sample Data")
    st.dataframe(df.head(20), use_container_width=True, height=400)
    
    # Class distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 Acceptability Distribution")
        class_counts = df['class'].value_counts().reset_index()
        class_counts.columns = ['Acceptability', 'Count']
        
        color_map = {
            'unacc': '#ef4444',
            'acc': '#f59e0b',
            'good': '#3b82f6',
            'vgood': '#22c55e'
        }
        
        fig = px.pie(class_counts, values='Count', names='Acceptability',
                    title='Car Acceptability Classes',
                    color='Acceptability',
                    color_discrete_map=color_map,
                    hole=0.4)
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 📊 Class Counts")
        fig = px.bar(class_counts, x='Acceptability', y='Count',
                    color='Acceptability',
                    color_discrete_map=color_map,
                    text='Count',
                    title='Distribution of Acceptability Classes')
        fig.update_traces(textposition='outside')
        st.plotly_chart(fig, use_container_width=True)

def show_feature_analysis(df):
    """Display feature analysis"""
    st.markdown("## 🔍 Feature Analysis")
    
    # Analyze each feature
    features = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']
    
    for i in range(0, len(features), 2):
        col1, col2 = st.columns(2)
        
        with col1:
            if i < len(features):
                feature = features[i]
                st.markdown(f"### {feature.replace('_', ' ').title()}")
                
                # Value counts
                value_counts = df[feature].value_counts().reset_index()
                value_counts.columns = [feature, 'Count']
                
                fig = px.bar(value_counts, x=feature, y='Count',
                            color='Count',
                            color_continuous_scale='viridis',
                            text='Count',
                            title=f'{feature.replace("_", " ").title()} Distribution')
                fig.update_traces(textposition='outside')
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if i + 1 < len(features):
                feature = features[i + 1]
                st.markdown(f"### {feature.replace('_', ' ').title()}")
                
                value_counts = df[feature].value_counts().reset_index()
                value_counts.columns = [feature, 'Count']
                
                fig = px.bar(value_counts, x=feature, y='Count',
                            color='Count',
                            color_continuous_scale='plasma',
                            text='Count',
                            title=f'{feature.replace("_", " ").title()} Distribution')
                fig.update_traces(textposition='outside')
                st.plotly_chart(fig, use_container_width=True)

def show_relationship_analysis(df):
    """Display relationship analysis between features and class"""
    st.markdown("## 🔗 Feature Relationships")
    
    features = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']
    
    # Feature selection
    selected_feature = st.selectbox(
        "Select feature to analyze:",
        features,
        format_func=lambda x: x.replace('_', ' ').title()
    )
    
    st.markdown(f"### {selected_feature.replace('_', ' ').title()} vs Acceptability")
    
    # Cross-tabulation
    crosstab = pd.crosstab(df[selected_feature], df['class'])
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Stacked bar chart
        fig = go.Figure()
        
        colors = {
            'unacc': '#ef4444',
            'acc': '#f59e0b',
            'good': '#3b82f6',
            'vgood': '#22c55e'
        }
        
        for col in crosstab.columns:
            fig.add_trace(go.Bar(
                name=col,
                x=crosstab.index,
                y=crosstab[col],
                marker_color=colors.get(col, '#667eea')
            ))
        
        fig.update_layout(
            barmode='stack',
            title=f'{selected_feature.replace("_", " ").title()} Distribution by Acceptability',
            xaxis_title=selected_feature.replace('_', ' ').title(),
            yaxis_title='Count',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### Cross-tabulation")
        st.dataframe(crosstab, use_container_width=True)
        
        st.markdown("#### Percentage Distribution")
        crosstab_pct = crosstab.div(crosstab.sum(axis=1), axis=0) * 100
        st.dataframe(crosstab_pct.round(1), use_container_width=True)
    
    # Heatmap
    st.markdown("### 🔥 Heatmap: Feature vs Acceptability")
    
    fig = px.imshow(crosstab.T,
                    labels=dict(x=selected_feature.replace('_', ' ').title(), 
                               y="Acceptability", 
                               color="Count"),
                    x=crosstab.index,
                    y=crosstab.columns,
                    color_continuous_scale='viridis',
                    text_auto=True)
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

def show_comparative_analysis(df):
    """Display comparative analysis"""
    st.markdown("## 📊 Comparative Analysis")
    
    features = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']
    
    col1, col2 = st.columns(2)
    
    with col1:
        feature1 = st.selectbox(
            "Select first feature:",
            features,
            format_func=lambda x: x.replace('_', ' ').title()
        )
    
    with col2:
        feature2 = st.selectbox(
            "Select second feature:",
            [f for f in features if f != feature1],
            format_func=lambda x: x.replace('_', ' ').title()
        )
    
    st.markdown(f"### {feature1.replace('_', ' ').title()} vs {feature2.replace('_', ' ').title()}")
    
    # Create contingency table
    contingency = pd.crosstab(df[feature1], df[feature2])
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig = px.imshow(contingency,
                       labels=dict(x=feature2.replace('_', ' ').title(),
                                  y=feature1.replace('_', ' ').title(),
                                  color="Count"),
                       x=contingency.columns,
                       y=contingency.index,
                       color_continuous_scale='blues',
                       text_auto=True,
                       title=f'{feature1.replace("_", " ").title()} vs {feature2.replace("_", " ").title()}')
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### Contingency Table")
        st.dataframe(contingency, use_container_width=True, height=400)
    
    # Analysis by acceptability
    st.markdown("### 🎯 Analysis by Acceptability Class")
    
    selected_class = st.selectbox(
        "Select acceptability class:",
        df['class'].unique(),
        format_func=lambda x: x.upper()
    )
    
    filtered_df = df[df['class'] == selected_class]
    
    col1, col2, col3 = st.columns(3)
    
    for idx, feature in enumerate([feature1, feature2, 'safety']):
        with [col1, col2, col3][idx]:
            value_counts = filtered_df[feature].value_counts().reset_index()
            value_counts.columns = [feature, 'Count']
            
            fig = px.pie(value_counts, 
                        values='Count', 
                        names=feature,
                        title=f'{feature.replace("_", " ").title()} ({selected_class.upper()})',
                        hole=0.4)
            st.plotly_chart(fig, use_container_width=True)

def show_statistics(df):
    """Display statistical summary"""
    st.markdown("## 📈 Statistical Summary")
    
    # Overall statistics
    st.markdown("### 📊 Dataset Statistics")
    
    stats_data = []
    for col in df.columns:
        unique_values = df[col].nunique()
        most_common = df[col].mode()[0]
        most_common_count = (df[col] == most_common).sum()
        
        stats_data.append({
            'Feature': col.replace('_', ' ').title(),
            'Unique Values': unique_values,
            'Most Common': most_common,
            'Most Common Count': most_common_count,
            'Percentage': f"{(most_common_count/len(df)*100):.1f}%"
        })
    
    stats_df = pd.DataFrame(stats_data)
    st.dataframe(stats_df, use_container_width=True, hide_index=True)
    
    # Acceptability breakdown
    st.markdown("### 🎯 Acceptability Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Class Distribution")
        class_stats = df['class'].value_counts().reset_index()
        class_stats.columns = ['Class', 'Count']
        class_stats['Percentage'] = (class_stats['Count'] / len(df) * 100).round(2)
        st.dataframe(class_stats, use_container_width=True, hide_index=True)
    
    with col2:
        st.markdown("#### Key Insights")
        
        total = len(df)
        unacc = (df['class'] == 'unacc').sum()
        acc = (df['class'] == 'acc').sum()
        good = (df['class'] == 'good').sum()
        vgood = (df['class'] == 'vgood').sum()
        
        st.markdown(f"""
        <div class="success-box">
        <strong>📊 Quick Stats:</strong><br><br>
        • Unacceptable: {unacc} ({unacc/total*100:.1f}%)<br>
        • Acceptable: {acc} ({acc/total*100:.1f}%)<br>
        • Good: {good} ({good/total*100:.1f}%)<br>
        • Very Good: {vgood} ({vgood/total*100:.1f}%)<br><br>
        <strong>Acceptable Cars (acc+good+vgood):</strong> {acc+good+vgood} ({(acc+good+vgood)/total*100:.1f}%)
        </div>
        """, unsafe_allow_html=True)
    
    # Feature importance for each class
    st.markdown("### 🔍 Feature Analysis by Class")
    
    selected_class = st.selectbox(
        "Select class for detailed analysis:",
        df['class'].unique(),
        format_func=lambda x: x.upper(),
        key='stats_class'
    )
    
    class_df = df[df['class'] == selected_class]
    
    feature_cols = [col for col in df.columns if col != 'class']
    
    cols = st.columns(3)
    
    for idx, feature in enumerate(feature_cols):
        with cols[idx % 3]:
            st.markdown(f"**{feature.replace('_', ' ').title()}**")
            value_counts = class_df[feature].value_counts().head(3)
            for val, count in value_counts.items():
                st.write(f"• {val}: {count} ({count/len(class_df)*100:.1f}%)")

def main():
    # Header
    st.markdown('<h1 class="main-header">🚗 Car Evaluation Analysis Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #64748b; font-size: 1.1rem; margin-bottom: 2rem;">Comprehensive Analysis of Car Acceptability Dataset</p>', unsafe_allow_html=True)
    
    # Load data
    df = load_car_evaluation_data()
    
    if df is None:
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 🎛️ Dashboard Controls")
        st.markdown("---")
        
        # Filters
        st.markdown("### 🔍 Filters")
        
        # Class filter
        classes = ['All'] + df['class'].unique().tolist()
        selected_class = st.selectbox("Filter by Acceptability:", classes)
        
        if selected_class != 'All':
            df = df[df['class'] == selected_class]
        
        # Safety filter
        safety_levels = ['All'] + df['safety'].unique().tolist()
        selected_safety = st.selectbox("Filter by Safety:", safety_levels)
        
        if selected_safety != 'All':
            df = df[df['safety'] == selected_safety]
        
        st.markdown("---")
        
        # Info
        st.markdown("### ℹ️ About")
        st.info("This dashboard analyzes the Car Evaluation dataset from UCI, containing 1,728 instances of car acceptability classifications.")
        
        st.markdown("### 📊 Current Data")
        st.write(f"**Records:** {len(df):,}")
        st.write(f"**Features:** {len(df.columns)-1}")
        
        if len(df) < 1728:
            st.warning(f"⚠️ Filtered view ({len(df)} records)")
    
    # Main tabs
    tabs = st.tabs(["📊 Overview", "🔍 Features", "🔗 Relationships", "📊 Comparison", "📈 Statistics"])
    
    with tabs[0]:
        show_overview(df)
    
    with tabs[1]:
        show_feature_analysis(df)
    
    with tabs[2]:
        show_relationship_analysis(df)
    
    with tabs[3]:
        show_comparative_analysis(df)
    
    with tabs[4]:
        show_statistics(df)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #64748b; padding: 1rem;'>
        <p>🚗 Car Evaluation Dataset | UCI Machine Learning Repository</p>
        <p>Dashboard created with Streamlit & Plotly</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
