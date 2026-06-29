import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle
import os
from utils import load_and_preprocess, get_raw_data
from model import train_models, load_best_model
from config import APP_TITLE, APP_SUBTITLE, DATA_PATH, GROQ_API_KEY

# Page config
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500&display=swap');

/* ══════════════════════════════════════
   GLOBAL
══════════════════════════════════════ */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}

.stApp {
    background: #030712 !important;
}

/* ══════════════════════════════════════
   SIDEBAR
══════════════════════════════════════ */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a0f1e 0%, #060c18 100%) !important;
    border-right: 1px solid rgba(99, 102, 241, 0.2) !important;
}

section[data-testid="stSidebar"] * {
    color: #94a3b8 !important;
}

section[data-testid="stSidebar"] h1 {
    color: #f1f5f9 !important;
    font-size: 1.1rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.05em !important;
}

section[data-testid="stSidebar"] .stSelectbox > div > div {
    background: rgba(99, 102, 241, 0.1) !important;
    border: 1px solid rgba(99, 102, 241, 0.3) !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
}

/* ══════════════════════════════════════
   HERO HEADER
══════════════════════════════════════ */
.hero-container {
    text-align: center;
    padding: 2rem 0 1rem 0;
    position: relative;
}

.hero-eyebrow {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.3em;
    text-transform: uppercase;
    color: #6366f1;
    margin-bottom: 0.8rem;
}

.hero-title {
    font-size: 4rem;
    font-weight: 900;
    line-height: 1;
    letter-spacing: -3px;
    background: linear-gradient(135deg, #fff 0%, #e2e8f0 30%, #6366f1 60%, #8b5cf6 80%, #06b6d4 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.5rem;
}

.hero-subtitle {
    font-size: 0.85rem;
    color: #475569;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    font-weight: 500;
    margin-bottom: 1.5rem;
}

.hero-divider {
    width: 60px;
    height: 2px;
    background: linear-gradient(90deg, #6366f1, #8b5cf6);
    margin: 0 auto 1.5rem auto;
    border-radius: 2px;
}

/* ══════════════════════════════════════
   BADGE STRIP
══════════════════════════════════════ */
.badge-strip {
    display: flex;
    gap: 8px;
    justify-content: center;
    flex-wrap: wrap;
    margin-bottom: 0.5rem;
}

.badge {
    background: rgba(99, 102, 241, 0.08);
    border: 1px solid rgba(99, 102, 241, 0.25);
    color: #818cf8;
    padding: 5px 14px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    transition: all 0.2s ease;
}

.badge:hover {
    background: rgba(99, 102, 241, 0.15);
    border-color: rgba(99, 102, 241, 0.5);
}

/* ══════════════════════════════════════
   GLOW DIVIDER
══════════════════════════════════════ */
.glow-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent 0%, rgba(99,102,241,0.5) 50%, transparent 100%);
    margin: 1.5rem 0;
}

/* ══════════════════════════════════════
   METRIC CARDS
══════════════════════════════════════ */
[data-testid="metric-container"] {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%) !important;
    border: 1px solid rgba(99, 102, 241, 0.15) !important;
    border-left: 3px solid #6366f1 !important;
    border-radius: 12px !important;
    padding: 1.2rem 1.4rem !important;
    box-shadow: 0 4px 24px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.03) !important;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
    position: relative !important;
    overflow: hidden !important;
}

[data-testid="metric-container"]::before {
    content: '';
    position: absolute;
    top: 0; right: 0;
    width: 60px; height: 60px;
    background: radial-gradient(circle, rgba(99,102,241,0.08) 0%, transparent 70%);
    border-radius: 50%;
}

[data-testid="metric-container"]:hover {
    transform: translateY(-3px) !important;
    border-left-color: #818cf8 !important;
    box-shadow: 0 12px 40px rgba(99,102,241,0.15), inset 0 1px 0 rgba(255,255,255,0.05) !important;
}

[data-testid="metric-container"] label {
    color: #475569 !important;
    font-size: 0.7rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
}

[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #f1f5f9 !important;
    font-size: 1.9rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.5px !important;
    font-family: 'Inter', sans-serif !important;
}

[data-testid="metric-container"] [data-testid="stMetricDelta"] {
    font-size: 0.75rem !important;
    font-weight: 600 !important;
}

/* ══════════════════════════════════════
   SECTION HEADERS
══════════════════════════════════════ */
h1 {
    color: #f1f5f9 !important;
    font-weight: 800 !important;
    font-size: 1.8rem !important;
    letter-spacing: -0.5px !important;
}

h2, h3 {
    color: #e2e8f0 !important;
    font-weight: 700 !important;
}

.section-eyebrow {
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #6366f1;
    margin-bottom: 0.3rem;
}

/* ══════════════════════════════════════
   BUTTONS
══════════════════════════════════════ */
.stButton > button {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    padding: 0.65rem 1.8rem !important;
    letter-spacing: 0.03em !important;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3) !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(99, 102, 241, 0.5) !important;
}

.stButton > button:active {
    transform: translateY(0px) !important;
}

/* ══════════════════════════════════════
   INPUTS
══════════════════════════════════════ */
.stSelectbox > div > div,
.stTextInput > div > div > input,
.stNumberInput > div > div > input {
    background: #0f172a !important;
    border: 1px solid rgba(99, 102, 241, 0.2) !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
    transition: border-color 0.2s ease !important;
}

.stSelectbox > div > div:focus-within,
.stTextInput > div > div > input:focus,
.stNumberInput > div > div > input:focus {
    border-color: rgba(99, 102, 241, 0.6) !important;
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1) !important;
}

/* ══════════════════════════════════════
   SLIDER
══════════════════════════════════════ */
.stSlider [data-baseweb="slider"] [role="slider"] {
    background: #6366f1 !important;
    border-color: #6366f1 !important;
}

/* ══════════════════════════════════════
   ALERTS
══════════════════════════════════════ */
[data-testid="stAlert"] {
    border-radius: 10px !important;
    border: none !important;
}

div[data-testid="stAlert"][data-baseweb="notification"] {
    border-left: 4px solid !important;
}

.stSuccess > div {
    background: rgba(16, 185, 129, 0.08) !important;
    border-left: 4px solid #10b981 !important;
    color: #6ee7b7 !important;
    border-radius: 10px !important;
}

.stInfo > div {
    background: rgba(99, 102, 241, 0.08) !important;
    border-left: 4px solid #6366f1 !important;
    color: #a5b4fc !important;
    border-radius: 10px !important;
}

.stWarning > div {
    background: rgba(245, 158, 11, 0.08) !important;
    border-left: 4px solid #f59e0b !important;
    color: #fcd34d !important;
    border-radius: 10px !important;
}

.stError > div {
    background: rgba(239, 68, 68, 0.08) !important;
    border-left: 4px solid #ef4444 !important;
    color: #fca5a5 !important;
    border-radius: 10px !important;
}

/* ══════════════════════════════════════
   DATAFRAME
══════════════════════════════════════ */
[data-testid="stDataFrame"] {
    border: 1px solid rgba(99, 102, 241, 0.15) !important;
    border-radius: 12px !important;
    overflow: hidden !important;
}

/* ══════════════════════════════════════
   FILE UPLOADER
══════════════════════════════════════ */
[data-testid="stFileUploader"] {
    background: rgba(99, 102, 241, 0.03) !important;
    border: 2px dashed rgba(99, 102, 241, 0.3) !important;
    border-radius: 16px !important;
    padding: 2rem !important;
    transition: all 0.2s ease !important;
}

[data-testid="stFileUploader"]:hover {
    border-color: rgba(99, 102, 241, 0.6) !important;
    background: rgba(99, 102, 241, 0.06) !important;
}

/* ══════════════════════════════════════
   CHAT
══════════════════════════════════════ */
[data-testid="stChatInput"] textarea {
    background: #0f172a !important;
    border: 1px solid rgba(99, 102, 241, 0.3) !important;
    border-radius: 12px !important;
    color: #e2e8f0 !important;
}

[data-testid="stChatMessage"] {
    background: #0f172a !important;
    border: 1px solid rgba(99, 102, 241, 0.1) !important;
    border-radius: 12px !important;
    margin-bottom: 0.75rem !important;
}

/* ══════════════════════════════════════
   DIVIDER
══════════════════════════════════════ */
hr {
    border: none !important;
    border-top: 1px solid rgba(99, 102, 241, 0.1) !important;
    margin: 1.5rem 0 !important;
}

/* ══════════════════════════════════════
   SCROLLBAR
══════════════════════════════════════ */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #030712; }
::-webkit-scrollbar-thumb { background: #1e293b; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #6366f1; }

/* ══════════════════════════════════════
   STAT CARD CUSTOM
══════════════════════════════════════ */
.stat-card {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    border: 1px solid rgba(99, 102, 241, 0.15);
    border-left: 3px solid #6366f1;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1rem;
}

.stat-card-red {
    border-left-color: #ef4444 !important;
}

.stat-card-green {
    border-left-color: #10b981 !important;
}

.stat-card-yellow {
    border-left-color: #f59e0b !important;
}

/* ══════════════════════════════════════
   RISK BADGES
══════════════════════════════════════ */
.risk-critical {
    display: inline-block;
    background: rgba(239, 68, 68, 0.15);
    border: 1px solid rgba(239, 68, 68, 0.4);
    color: #fca5a5;
    padding: 6px 16px;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 700;
    letter-spacing: 0.05em;
}

.risk-high {
    display: inline-block;
    background: rgba(245, 158, 11, 0.15);
    border: 1px solid rgba(245, 158, 11, 0.4);
    color: #fcd34d;
    padding: 6px 16px;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 700;
}

.risk-medium {
    display: inline-block;
    background: rgba(99, 102, 241, 0.15);
    border: 1px solid rgba(99, 102, 241, 0.4);
    color: #a5b4fc;
    padding: 6px 16px;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 700;
}

.risk-low {
    display: inline-block;
    background: rgba(16, 185, 129, 0.15);
    border: 1px solid rgba(16, 185, 129, 0.4);
    color: #6ee7b7;
    padding: 6px 16px;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 700;
}

/* ══════════════════════════════════════
   PLOTLY CONTAINER
══════════════════════════════════════ */
.js-plotly-plot {
    border-radius: 12px !important;
    border: 1px solid rgba(99, 102, 241, 0.1) !important;
    overflow: hidden !important;
}

/* ══════════════════════════════════════
   MODULE HEADER
══════════════════════════════════════ */
.module-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 0.5rem;
    padding-bottom: 1rem;
    border-bottom: 1px solid rgba(99, 102, 241, 0.1);
}

.module-icon {
    font-size: 1.8rem;
}

.module-title {
    font-size: 1.6rem;
    font-weight: 800;
    color: #f1f5f9;
    letter-spacing: -0.5px;
}

.module-desc {
    font-size: 0.85rem;
    color: #475569;
    margin-top: 0.2rem;
}
</style>
""", unsafe_allow_html=True)


def dark_chart(fig):
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(15,23,42,0.6)',
        font=dict(
            color='#94a3b8',
            family='Inter, sans-serif',
            size=12
        ),
        xaxis=dict(
            gridcolor='rgba(99,102,241,0.08)',
            linecolor='rgba(99,102,241,0.15)',
            tickcolor='#475569',
            tickfont=dict(color='#64748b', size=11)
        ),
        yaxis=dict(
            gridcolor='rgba(99,102,241,0.08)',
            linecolor='rgba(99,102,241,0.15)',
            tickcolor='#475569',
            tickfont=dict(color='#64748b', size=11)
        ),
        legend=dict(
            bgcolor='rgba(15,23,42,0.8)',
            bordercolor='rgba(99,102,241,0.2)',
            borderwidth=1,
            font=dict(color='#94a3b8')
        ),
        title=dict(
            text='',
            font=dict(color='#e2e8f0', size=14, family='Inter'),
            x=0.02 
        ),
        margin=dict(t=20, l=10, r=10, b=10)
    )
    return fig

# Header
st.markdown("""
<div class="hero-container">
    <div class="hero-eyebrow">AI-POWERED ANALYTICS</div>
    <div class="hero-title">ChurnGuard</div>
    <div class="hero-subtitle">Customer Retention Intelligence Platform</div>
    <div class="hero-divider"></div>
</div>

<div class="badge-strip">
    <span class="badge">⚡ Live ML Engine</span>
    <span class="badge">🔍 SHAP Explainability</span>
    <span class="badge">🤖 Groq LLaMA Advisor</span>
    <span class="badge">📊 7,043 Customers Analysed</span>
    <span class="badge">💰 $1.5M Revenue at Risk</span>
    <span class="badge">📄 One-Click PDF Reports</span>
</div>
<div class="glow-divider"></div>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.markdown("""
<div style="padding: 1rem 0 0.5rem 0;">
    <div style="
        width: 48px;
        height: 48px;
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.5rem;
        box-shadow: 0 4px 15px rgba(99,102,241,0.4);
    ">🛡️</div>
</div>
""", unsafe_allow_html=True)
st.sidebar.title("Navigation")
module = st.sidebar.selectbox("Select Module", [
    "📊 Data Overview & EDA",
    "🤖 ML Engine",
    "🔍 SHAP Explainability",
    "🎯 Customer Risk Scorer",
    "🔄 What-If Simulator",
    "💰 CLV & Priority Scoring",
    "📉 Revenue Impact",
    "🎪 Retention Campaign Simulator",
    "🤖 AI Advisor",
    "📄 PDF Report",
    "📁 Upload Your Own Data"
])

# Load data
@st.cache_data
def get_data():
    return get_raw_data(DATA_PATH)

@st.cache_data
def get_processed_data():
    return load_and_preprocess(DATA_PATH)

@st.cache_resource
def get_trained_models():
    return train_models(DATA_PATH)

df_raw = get_data()
df_processed = get_processed_data()

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Dataset:** {len(df_raw):,} customers")
st.sidebar.markdown(f"**Churn Rate:** {df_raw['Churn'].mean():.1%}")
st.sidebar.markdown("---")
st.sidebar.markdown("Built by **Shubham Reddy**")
st.sidebar.markdown("🎓 RMIT University Melbourne")

# ================================
# MODULE 1 — DATA OVERVIEW & EDA
# ================================
if module == "📊 Data Overview & EDA":
    st.header("📊 Data Overview & Exploratory Data Analysis")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Customers", f"{len(df_raw):,}")
    with col2:
        churned = df_raw['Churn'].sum()
        st.metric("Churned Customers", f"{churned:,}")
    with col3:
        st.metric("Churn Rate", f"{df_raw['Churn'].mean():.1%}")
    with col4:
        st.metric("Retained Customers", f"{len(df_raw) - churned:,}")

    st.markdown("---")
    st.subheader("Dataset Preview")
    st.dataframe(df_raw.head(20), use_container_width=True)
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Churn Distribution")
        churn_counts = df_raw['Churn'].value_counts()
        fig = px.pie(values=churn_counts.values,
                     names=['Retained', 'Churned'],
                     color_discrete_sequence=['#00cc44', '#ff4444'],
                     hole=0.4)
        fig = dark_chart(fig)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Churn by Contract Type")
        contract_churn = df_raw.groupby('Contract')['Churn'].mean().reset_index()
        fig = px.bar(contract_churn, x='Contract', y='Churn',
                     color='Churn',
                     color_continuous_scale='RdYlGn_r',
                     labels={'Churn': 'Churn Rate'})
        fig = dark_chart(fig)
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Tenure vs Churn")
        fig = px.histogram(df_raw, x='tenure', color='Churn',
                           barmode='overlay',
                           color_discrete_map={0: '#00cc44', 1: '#ff4444'})
        fig = dark_chart(fig)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Monthly Charges vs Churn")
        fig = px.box(df_raw, x='Churn', y='MonthlyCharges',
                     color='Churn',
                     color_discrete_map={0: '#00cc44', 1: '#ff4444'})
        fig = dark_chart(fig)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Feature Correlation Heatmap")
    numeric_cols = df_processed.select_dtypes(include=np.number).columns[:15]
    corr_matrix = df_processed[numeric_cols].corr()
    fig = px.imshow(corr_matrix,
                    color_continuous_scale='RdBu_r',
                    aspect='auto')
    fig = dark_chart(fig)
    st.plotly_chart(fig, use_container_width=True)

# ================================
# MODULE 2 — ML ENGINE
# ================================
elif module == "🤖 ML Engine":
    st.header("🤖 ML Engine — Model Training & Comparison")
    st.info("⏳ Training 3 models... This may take 30-60 seconds on first run.")

    results, best_name, trained_models, X_test, y_test, X = get_trained_models()
    st.success(f"✅ Best Model: **{best_name}**")

    st.subheader("Model Performance Comparison")
    results_df = pd.DataFrame(results).T
    results_df = results_df.sort_values('AUC-ROC', ascending=False)
    st.dataframe(results_df.style.highlight_max(axis=0, color='#90EE90'),
                 use_container_width=True)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("AUC-ROC Comparison")
        fig = px.bar(
            x=list(results.keys()),
            y=[results[m]['AUC-ROC'] for m in results],
            color=list(results.keys()),
            labels={'x': 'Model', 'y': 'AUC-ROC Score'},
            color_discrete_sequence=['#1f77b4', '#ff7f0e', '#2ca02c']
        )
        fig.update_layout(showlegend=False)
        fig = dark_chart(fig)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("F1 Score Comparison")
        fig = px.bar(
            x=list(results.keys()),
            y=[results[m]['F1 Score'] for m in results],
            color=list(results.keys()),
            labels={'x': 'Model', 'y': 'F1 Score'},
            color_discrete_sequence=['#1f77b4', '#ff7f0e', '#2ca02c']
        )
        fig.update_layout(showlegend=False)
        fig = dark_chart(fig)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("ROC Curves — All Models")
    from sklearn.metrics import roc_curve, auc
    fig = go.Figure()
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    for i, (name, model) in enumerate(trained_models.items()):
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc_score = auc(fpr, tpr)
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            name=f"{name} (AUC={auc_score:.3f})",
            line=dict(color=colors[i], width=2)
        ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        line=dict(dash='dash', color='gray'),
        name='Random Classifier'
    ))
    fig.update_layout(xaxis_title='False Positive Rate',
                      yaxis_title='True Positive Rate', height=400)
    fig = dark_chart(fig)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader(f"Confusion Matrix — {best_name}")
    from sklearn.metrics import confusion_matrix
    best_model = trained_models[best_name]
    y_pred = best_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    fig = px.imshow(cm,
                    labels=dict(x="Predicted", y="Actual"),
                    x=['Not Churned', 'Churned'],
                    y=['Not Churned', 'Churned'],
                    color_continuous_scale='Blues',
                    text_auto=True)
    fig = dark_chart(fig)
    st.plotly_chart(fig, use_container_width=True)

# ================================
# MODULE 3 — SHAP EXPLAINABILITY
# ================================
elif module == "🔍 SHAP Explainability":
    st.header("🔍 SHAP Explainability — Why Customers Churn")
    st.info("⏳ Computing SHAP values... This may take a moment.")

    import shap
    results, best_name, trained_models, X_test, y_test, X = get_trained_models()
    best_model = trained_models[best_name]

    if best_name in ["Random Forest", "XGBoost"]:
        explainer = shap.TreeExplainer(best_model)
    else:
        explainer = shap.LinearExplainer(best_model, X_test)

    X_sample = X_test.iloc[:100]
    shap_values = explainer.shap_values(X_sample)

    if isinstance(shap_values, list):
        shap_vals = shap_values[1]
    else:
        shap_vals = shap_values

    st.success(f"✅ SHAP values computed using **{best_name}**")
    st.markdown("---")

    st.subheader("🌍 Global Feature Importance")
    shap_df = pd.DataFrame({
        'Feature': X_test.columns,
        'Mean SHAP Value': np.round(np.abs(np.array(shap_vals)).mean(axis=0), 4)
    }).sort_values('Mean SHAP Value', ascending=False).head(15)

    fig = px.bar(shap_df, x='Mean SHAP Value', y='Feature',
                 orientation='h',
                 color='Mean SHAP Value',
                 color_continuous_scale='RdYlGn_r',
                 title='Top 15 Features Driving Churn')
    fig.update_layout(yaxis={'categoryorder': 'total ascending'}, height=500)
    fig = dark_chart(fig)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("🎯 Individual Customer Explanation")

    customer_idx = st.slider("Select Customer Index", 0, 99, 0)
    customer_shap = shap_vals[customer_idx]
    customer_features = X_sample.iloc[customer_idx]

    explanation_df = pd.DataFrame({
        'Feature': X_test.columns,
        'SHAP Value': customer_shap,
        'Feature Value': customer_features.values
    }).sort_values('SHAP Value', key=abs, ascending=False).head(10)

    col1, col2 = st.columns(2)
    with col1:
        churn_prob = best_model.predict_proba(
            X_sample.iloc[[customer_idx]])[0][1]
        risk = "🔴 HIGH RISK" if churn_prob > 0.7 else \
               "🟡 MEDIUM RISK" if churn_prob > 0.4 else "🟢 LOW RISK"
        st.metric("Churn Probability", f"{churn_prob:.1%}")
        st.markdown(f"### {risk}")

    with col2:
        st.markdown("**Top Churn Drivers for this Customer:**")
        for _, row in explanation_df.head(5).iterrows():
            direction = "⬆️ increases" if row['SHAP Value'] > 0 else "⬇️ decreases"
            st.markdown(f"- **{row['Feature']}** = {row['Feature Value']:.2f} → {direction} churn risk")

    fig = px.bar(explanation_df,
                 x='SHAP Value', y='Feature',
                 orientation='h',
                 color='SHAP Value',
                 color_continuous_scale='RdBu_r',
                 title=f'SHAP Explanation — Customer #{customer_idx}')
    fig.update_layout(yaxis={'categoryorder': 'total ascending'}, height=400)
    fig = dark_chart(fig)
    st.plotly_chart(fig, use_container_width=True)

# ================================
# MODULE 4 — CUSTOMER RISK SCORER
# ================================
elif module == "🎯 Customer Risk Scorer":
    st.header("🎯 Customer Risk Scorer")
    st.markdown("Enter customer details to get instant churn probability and risk level.")

    results, best_name, trained_models, X_test, y_test, X = get_trained_models()
    best_model = trained_models[best_name]

    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("📋 Customer Details")
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        monthly_charges = st.slider("Monthly Charges ($)", 0, 120, 65)
        total_charges = st.number_input("Total Charges ($)",
                                         value=float(tenure * monthly_charges))
        senior_citizen = st.selectbox("Senior Citizen", [0, 1])
        gender = st.selectbox("Gender", ["Male", "Female"])

    with col2:
        st.subheader("📦 Services")
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        internet_service = st.selectbox("Internet Service",
                                         ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security",
                                        ["Yes", "No", "No internet service"])
        tech_support = st.selectbox("Tech Support",
                                     ["Yes", "No", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV",
                                     ["Yes", "No", "No internet service"])

    with col3:
        st.subheader("💳 Account Info")
        contract = st.selectbox("Contract Type",
                                 ["Month-to-month", "One year", "Two year"])
        payment_method = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"
        ])
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])

    if st.button("🔍 Score This Customer", use_container_width=True):
        input_data = pd.DataFrame(columns=X.columns)
        input_data.loc[0] = 0
        input_data['tenure'] = tenure
        input_data['MonthlyCharges'] = monthly_charges
        input_data['TotalCharges'] = total_charges
        input_data['SeniorCitizen'] = senior_citizen
        input_data['gender'] = 1 if gender == "Male" else 0
        input_data['PhoneService'] = 1 if phone_service == "Yes" else 0
        input_data['PaperlessBilling'] = 1 if paperless_billing == "Yes" else 0
        input_data['Partner'] = 1 if partner == "Yes" else 0
        input_data['Dependents'] = 1 if dependents == "Yes" else 0

        if internet_service == "Fiber optic":
            if 'InternetService_Fiber optic' in input_data.columns:
                input_data['InternetService_Fiber optic'] = 1
        elif internet_service == "No":
            if 'InternetService_No' in input_data.columns:
                input_data['InternetService_No'] = 1

        if contract == "One year":
            if 'Contract_One year' in input_data.columns:
                input_data['Contract_One year'] = 1
        elif contract == "Two year":
            if 'Contract_Two year' in input_data.columns:
                input_data['Contract_Two year'] = 1

        if payment_method == "Credit card (automatic)":
            if 'PaymentMethod_Credit card (automatic)' in input_data.columns:
                input_data['PaymentMethod_Credit card (automatic)'] = 1
        elif payment_method == "Electronic check":
            if 'PaymentMethod_Electronic check' in input_data.columns:
                input_data['PaymentMethod_Electronic check'] = 1
        elif payment_method == "Mailed check":
            if 'PaymentMethod_Mailed check' in input_data.columns:
                input_data['PaymentMethod_Mailed check'] = 1

        input_data = input_data.astype(float)
        churn_prob = best_model.predict_proba(input_data)[0][1]

        st.markdown("---")
        st.subheader("📊 Risk Assessment Result")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Churn Probability", f"{churn_prob:.1%}")
        with col2:
            st.metric("Retention Probability", f"{1 - churn_prob:.1%}")
        with col3:
            if churn_prob > 0.7:
                st.error("🔴 CRITICAL RISK")
            elif churn_prob > 0.5:
                st.warning("🟡 HIGH RISK")
            elif churn_prob > 0.3:
                st.info("🔵 MEDIUM RISK")
            else:
                st.success("🟢 LOW RISK")

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=churn_prob * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Churn Risk Score"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkred"},
                'steps': [
                    {'range': [0, 30], 'color': "#00cc44"},
                    {'range': [30, 50], 'color': "#ffcc00"},
                    {'range': [50, 70], 'color': "#ff8800"},
                    {'range': [70, 100], 'color': "#ff4444"}
                ]
            }
        ))
        fig.update_layout(height=300)
        fig = dark_chart(fig)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("💡 Recommended Action")
        if churn_prob > 0.7:
            st.error("""
            **IMMEDIATE INTERVENTION REQUIRED**
            - Call customer within 24 hours
            - Offer loyalty discount (20-30%)
            - Upgrade to annual contract with incentive
            - Assign dedicated account manager
            """)
        elif churn_prob > 0.5:
            st.warning("""
            **PROACTIVE RETENTION NEEDED**
            - Send personalised retention email
            - Offer contract upgrade incentive
            - Review service issues or complaints
            """)
        elif churn_prob > 0.3:
            st.info("""
            **MONITOR CLOSELY**
            - Include in next retention campaign
            - Offer loyalty rewards
            - Check satisfaction score
            """)
        else:
            st.success("""
            **LOW RISK — MAINTAIN RELATIONSHIP**
            - Continue standard engagement
            - Include in upsell campaigns
            - Reward loyalty
            """)

# ================================
# MODULE 5 — WHAT-IF SIMULATOR
# ================================
elif module == "🔄 What-If Simulator":
    st.header("🔄 What-If Simulator")
    st.markdown("Change customer attributes and see churn probability update **live in real time.**")

    results, best_name, trained_models, X_test, y_test, X = get_trained_models()
    best_model = trained_models[best_name]

    st.markdown("---")
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("⚙️ Adjust Customer Profile")
        tenure = st.slider("Tenure (months)", 0, 72, 6, key="wi_tenure")
        monthly_charges = st.slider("Monthly Charges ($)", 0, 120, 80, key="wi_mc")
        total_charges = tenure * monthly_charges
        contract = st.selectbox("Contract Type",
                                 ["Month-to-month", "One year", "Two year"],
                                 key="wi_contract")
        internet_service = st.selectbox("Internet Service",
                                         ["DSL", "Fiber optic", "No"],
                                         key="wi_internet")
        payment_method = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"
        ], key="wi_payment")
        paperless_billing = st.selectbox("Paperless Billing",
                                          ["Yes", "No"], key="wi_pb")
        senior_citizen = st.selectbox("Senior Citizen", [0, 1], key="wi_sc")

    input_data = pd.DataFrame(columns=X.columns)
    input_data.loc[0] = 0
    input_data['tenure'] = tenure
    input_data['MonthlyCharges'] = monthly_charges
    input_data['TotalCharges'] = float(total_charges)
    input_data['SeniorCitizen'] = senior_citizen
    input_data['PaperlessBilling'] = 1 if paperless_billing == "Yes" else 0

    if internet_service == "Fiber optic":
        if 'InternetService_Fiber optic' in input_data.columns:
            input_data['InternetService_Fiber optic'] = 1
    elif internet_service == "No":
        if 'InternetService_No' in input_data.columns:
            input_data['InternetService_No'] = 1

    if contract == "One year":
        if 'Contract_One year' in input_data.columns:
            input_data['Contract_One year'] = 1
    elif contract == "Two year":
        if 'Contract_Two year' in input_data.columns:
            input_data['Contract_Two year'] = 1

    if payment_method == "Credit card (automatic)":
        if 'PaymentMethod_Credit card (automatic)' in input_data.columns:
            input_data['PaymentMethod_Credit card (automatic)'] = 1
    elif payment_method == "Electronic check":
        if 'PaymentMethod_Electronic check' in input_data.columns:
            input_data['PaymentMethod_Electronic check'] = 1
    elif payment_method == "Mailed check":
        if 'PaymentMethod_Mailed check' in input_data.columns:
            input_data['PaymentMethod_Mailed check'] = 1

    input_data = input_data.astype(float)
    churn_prob = best_model.predict_proba(input_data)[0][1]

    with col2:
        st.subheader("📊 Live Churn Probability")
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=churn_prob * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Churn Risk %"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkred"},
                'steps': [
                    {'range': [0, 30], 'color': "#00cc44"},
                    {'range': [30, 50], 'color': "#ffcc00"},
                    {'range': [50, 70], 'color': "#ff8800"},
                    {'range': [70, 100], 'color': "#ff4444"}
                ]
            }
        ))
        fig.update_layout(height=300)
        fig = dark_chart(fig)
        st.plotly_chart(fig, use_container_width=True)

        if churn_prob > 0.7:
            st.error("🔴 CRITICAL RISK")
        elif churn_prob > 0.5:
            st.warning("🟡 HIGH RISK")
        elif churn_prob > 0.3:
            st.info("🔵 MEDIUM RISK")
        else:
            st.success("🟢 LOW RISK")

        st.metric("Churn Probability", f"{churn_prob:.1%}")
        st.metric("Retention Probability", f"{1 - churn_prob:.1%}")

    st.markdown("---")
    st.subheader("📈 Scenario Comparison")
    scenarios = []
    for ct in ["Month-to-month", "One year", "Two year"]:
        sim = pd.DataFrame(columns=X.columns)
        sim.loc[0] = 0
        sim['tenure'] = tenure
        sim['MonthlyCharges'] = monthly_charges
        sim['TotalCharges'] = float(total_charges)
        sim['SeniorCitizen'] = senior_citizen
        sim['PaperlessBilling'] = 1 if paperless_billing == "Yes" else 0
        if ct == "One year":
            if 'Contract_One year' in sim.columns:
                sim['Contract_One year'] = 1
        elif ct == "Two year":
            if 'Contract_Two year' in sim.columns:
                sim['Contract_Two year'] = 1
        sim = sim.astype(float)
        prob = best_model.predict_proba(sim)[0][1]
        scenarios.append({'Contract': ct, 'Churn Probability': prob})

    scenario_df = pd.DataFrame(scenarios)
    fig = px.bar(scenario_df,
                 x='Contract', y='Churn Probability',
                 color='Churn Probability',
                 color_continuous_scale='RdYlGn_r',
                 title='Churn Probability by Contract Type',
                 text=scenario_df['Churn Probability'].apply(lambda x: f"{x:.1%}"))
    fig.update_traces(textposition='outside')
    fig.update_layout(yaxis_tickformat='.0%')
    fig = dark_chart(fig)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("💡 What-If Insight")
    best_contract = scenario_df.loc[scenario_df['Churn Probability'].idxmin(), 'Contract']
    worst_contract = scenario_df.loc[scenario_df['Churn Probability'].idxmax(), 'Contract']
    best_prob = scenario_df['Churn Probability'].min()
    worst_prob = scenario_df['Churn Probability'].max()
    saving = worst_prob - best_prob
    st.success(f"""
    ✅ **Switching from {worst_contract} → {best_contract} contract**
    reduces churn probability by **{saving:.1%}**
    (from {worst_prob:.1%} down to {best_prob:.1%})
    """)

# ================================
# MODULE 6 — CLV & PRIORITY SCORING
# ================================
elif module == "💰 CLV & Priority Scoring":
    st.header("💰 Customer Lifetime Value & Priority Scoring")
    st.markdown("Combines **churn risk** with **revenue value** to prioritise who to save first.")

    results, best_name, trained_models, X_test, y_test, X = get_trained_models()
    best_model = trained_models[best_name]
    df = get_raw_data(DATA_PATH)
    df_proc = get_processed_data()

    st.markdown("---")

    df['CLV'] = df['tenure'] * df['MonthlyCharges']
    df['Churn_Prob'] = best_model.predict_proba(
        df_proc.drop('Churn', axis=1)
    )[:, 1]
    df['Priority_Score'] = df['Churn_Prob'] * df['CLV']
    df['Risk_Tier'] = pd.cut(df['Churn_Prob'],
                              bins=[0, 0.3, 0.5, 0.7, 1.0],
                              labels=['Low', 'Medium', 'High', 'Critical'])

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Avg Customer CLV", f"${df['CLV'].mean():,.0f}")
    with col2:
        val = df[df['Churn_Prob'] > 0.5]['CLV'].sum()
        st.metric("Total Revenue at Risk", f"${val/1_000_000:.2f}M")
    with col3:
        st.metric("Critical Risk Customers",
                  f"{(df['Churn_Prob'] > 0.7).sum():,}")
    with col4:
        st.metric("High Priority Accounts",
                  f"{(df['Priority_Score'] > df['Priority_Score'].quantile(0.75)).sum():,}")

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("CLV Distribution by Risk Tier")
        fig = px.box(df, x='Risk_Tier', y='CLV',
                     color='Risk_Tier',
                     color_discrete_map={
                         'Low': '#00cc44',
                         'Medium': '#ffcc00',
                         'High': '#ff8800',
                         'Critical': '#ff4444'
                     })
        fig = dark_chart(fig)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Churn Probability vs CLV")
        fig = px.scatter(df.sample(500), x='CLV', y='Churn_Prob',
                         color='Risk_Tier',
                         color_discrete_map={
                             'Low': '#00cc44',
                             'Medium': '#ffcc00',
                             'High': '#ff8800',
                             'Critical': '#ff4444'
                         },
                         opacity=0.6,
                         title="Revenue at Risk Map")
        fig.add_hline(y=0.5, line_dash="dash",
                      annotation_text="High Risk Threshold")
        fig = dark_chart(fig)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("🚨 Top 20 Priority Customers — Intervene NOW")
    top20 = df.nlargest(20, 'Priority_Score')[[
        'tenure', 'MonthlyCharges', 'CLV',
        'Churn_Prob', 'Priority_Score', 'Risk_Tier', 'Contract'
    ]].reset_index(drop=True)

    top20['Churn_Prob'] = top20['Churn_Prob'].apply(lambda x: f"{x:.1%}")
    top20['CLV'] = top20['CLV'].apply(lambda x: f"${x:,.0f}")
    top20['Priority_Score'] = top20['Priority_Score'].apply(lambda x: f"{x:.1f}")
    top20['MonthlyCharges'] = top20['MonthlyCharges'].apply(lambda x: f"${x:.0f}")
    st.dataframe(top20, use_container_width=True)

    st.markdown("---")
    st.subheader("📊 Revenue at Risk by Risk Tier")
    risk_revenue = df.groupby('Risk_Tier', observed=True)['CLV'].sum().reset_index()
    risk_revenue.columns = ['Risk Tier', 'Total CLV at Risk']
    fig = px.bar(risk_revenue,
                 x='Risk Tier', y='Total CLV at Risk',
                 color='Risk Tier',
                 color_discrete_map={
                     'Low': '#00cc44',
                     'Medium': '#ffcc00',
                     'High': '#ff8800',
                     'Critical': '#ff4444'
                 },
                 title='Total Revenue at Risk by Tier')
    fig = dark_chart(fig)
    st.plotly_chart(fig, use_container_width=True)

# ================================
# MODULE 7 — REVENUE IMPACT
# ================================
elif module == "📉 Revenue Impact":
    st.header("📉 Revenue Impact Calculator")
    st.markdown("Quantify the **financial cost of churn** and potential **savings from retention.**")

    results, best_name, trained_models, X_test, y_test, X = get_trained_models()
    best_model = trained_models[best_name]
    df = get_raw_data(DATA_PATH)
    df_proc = get_processed_data()

    df['CLV'] = df['tenure'] * df['MonthlyCharges']
    df['Churn_Prob'] = best_model.predict_proba(
        df_proc.drop('Churn', axis=1)
    )[:, 1]
    df['Risk_Tier'] = pd.cut(df['Churn_Prob'],
                              bins=[0, 0.3, 0.5, 0.7, 1.0],
                              labels=['Low', 'Medium', 'High', 'Critical'])

    st.markdown("---")

    total_revenue_at_risk = df[df['Churn_Prob'] > 0.5]['CLV'].sum()
    critical_revenue = df[df['Churn_Prob'] > 0.7]['CLV'].sum()
    avg_clv = df['CLV'].mean()
    monthly_churn_loss = df[df['Churn_Prob'] > 0.5]['MonthlyCharges'].sum()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Revenue at Risk", f"${total_revenue_at_risk/1_000_000:.2f}M")
    with col2:
        st.metric("Critical Accounts Revenue", f"${critical_revenue:,.0f}")
    with col3:
        st.metric("Monthly Loss if No Action", f"${monthly_churn_loss:,.0f}")
    with col4:
        st.metric("Avg CLV Per Customer", f"${avg_clv:,.0f}")

    st.markdown("---")
    st.subheader("🎛️ Retention Investment Simulator")

    col1, col2 = st.columns(2)
    with col1:
        retention_rate = st.slider("Retention Success Rate (%)", 10, 90, 40)
        cost_per_customer = st.slider("Retention Cost Per Customer ($)", 10, 500, 100)

    high_risk_customers = df[df['Churn_Prob'] > 0.5]
    num_at_risk = len(high_risk_customers)
    revenue_saved = (retention_rate / 100) * high_risk_customers['CLV'].sum()
    total_cost = num_at_risk * cost_per_customer
    net_benefit = revenue_saved - total_cost
    roi = ((revenue_saved - total_cost) / total_cost * 100) if total_cost > 0 else 0

    with col2:
        st.metric("Customers at Risk", f"{num_at_risk:,}")
        st.metric("Revenue Saved", f"${revenue_saved:,.0f}")
        st.metric("Retention Campaign Cost", f"${total_cost:,.0f}")
        if net_benefit > 0:
            st.metric("Net Benefit", f"${net_benefit:,.0f}", delta="Profitable ✅")
        else:
            st.metric("Net Benefit", f"${net_benefit:,.0f}", delta="Loss ❌")
        st.metric("ROI", f"{roi:.1f}%")

    st.markdown("---")
    st.subheader("📊 Revenue Saved vs Retention Cost Across Scenarios")
    retention_rates = list(range(10, 100, 10))
    saved = [(r/100) * high_risk_customers['CLV'].sum() for r in retention_rates]
    costs = [num_at_risk * cost_per_customer] * len(retention_rates)

    fig = go.Figure()
    fig.add_trace(go.Bar(name='Revenue Saved',
                          x=[f"{r}%" for r in retention_rates],
                          y=saved,
                          marker_color='#00cc44'))
    fig.add_trace(go.Bar(name='Retention Cost',
                          x=[f"{r}%" for r in retention_rates],
                          y=costs,
                          marker_color='#ff4444'))
    fig.update_layout(barmode='group',
                      xaxis_title='Retention Success Rate',
                      yaxis_title='Amount ($)',
                      title='Revenue Saved vs Cost by Retention Rate')
    fig = dark_chart(fig)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("📈 Monthly Revenue Loss Trend by Risk Tier")
    risk_monthly = df.groupby('Risk_Tier', observed=True)['MonthlyCharges'].agg(
        ['sum', 'count', 'mean']
    ).reset_index()
    risk_monthly.columns = ['Risk Tier', 'Total Monthly Revenue',
                             'Customer Count', 'Avg Monthly Charges']

    fig = px.bar(risk_monthly,
                 x='Risk Tier', y='Total Monthly Revenue',
                 color='Risk Tier',
                 color_discrete_map={
                     'Low': '#00cc44',
                     'Medium': '#ffcc00',
                     'High': '#ff8800',
                     'Critical': '#ff4444'
                 },
                 text='Customer Count',
                 title='Monthly Revenue at Risk by Tier')
    fig.update_traces(texttemplate='%{text} customers', textposition='outside')
    fig = dark_chart(fig)
    st.plotly_chart(fig, use_container_width=True)

# ================================
# POWER 1 — RETENTION CAMPAIGN SIMULATOR
# ================================
elif module == "🎪 Retention Campaign Simulator":
    st.header("🎪 Retention Campaign Simulator")
    st.markdown("Simulate different retention strategies and calculate **exact ROI** for each.")

    results, best_name, trained_models, X_test, y_test, X = get_trained_models()
    best_model = trained_models[best_name]
    df = get_raw_data(DATA_PATH)
    df_proc = get_processed_data()

    df['CLV'] = df['tenure'] * df['MonthlyCharges']
    df['Churn_Prob'] = best_model.predict_proba(
        df_proc.drop('Churn', axis=1)
    )[:, 1]

    st.markdown("---")
    st.subheader("🎯 Design Your Retention Campaign")

    col1, col2 = st.columns(2)
    with col1:
        campaign_name = st.text_input("Campaign Name", "Summer Retention Drive")
        target_tier = st.selectbox("Target Risk Tier", [
            "All High Risk (>50%)",
            "Critical Only (>70%)",
            "Medium + High (>30%)"
        ])
        offer_type = st.selectbox("Offer Type", [
            "Discount on Monthly Bill",
            "Free Service Upgrade",
            "Contract Incentive",
            "Loyalty Reward Points",
            "Dedicated Support"
        ])

    with col2:
        discount_pct = st.slider("Offer Value (% discount or $ equivalent)", 5, 50, 20)
        campaign_budget = st.number_input("Total Campaign Budget ($)", value=50000)
        expected_success = st.slider("Expected Success Rate (%)", 10, 80, 35)

    if target_tier == "Critical Only (>70%)":
        targets = df[df['Churn_Prob'] > 0.7]
    elif target_tier == "Medium + High (>30%)":
        targets = df[df['Churn_Prob'] > 0.3]
    else:
        targets = df[df['Churn_Prob'] > 0.5]

    cost_per_customer = campaign_budget / len(targets) if len(targets) > 0 else 0
    customers_saved = int(len(targets) * expected_success / 100)
    revenue_saved = targets.nlargest(customers_saved, 'CLV')['CLV'].sum()
    revenue_lost = targets['MonthlyCharges'].sum() * discount_pct / 100 * 12
    net_roi = revenue_saved - campaign_budget - revenue_lost
    roi_pct = (net_roi / campaign_budget * 100) if campaign_budget > 0 else 0

    st.markdown("---")
    st.subheader(f"📊 Campaign Results: **{campaign_name}**")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Customers Targeted", f"{len(targets):,}")
    with col2:
        st.metric("Customers Saved", f"{customers_saved:,}")
    with col3:
        st.metric("Revenue Saved", f"${revenue_saved:,.0f}")
    with col4:
        if net_roi > 0:
            st.metric("Net ROI", f"${net_roi:,.0f}", delta=f"{roi_pct:.1f}% ✅")
        else:
            st.metric("Net ROI", f"${net_roi:,.0f}", delta=f"{roi_pct:.1f}% ❌")

    st.markdown("---")
    st.subheader("📈 Compare All Campaign Strategies")

    strategies = [
        {"Strategy": "Discount 10%", "Success": 25, "Cost": 30000},
        {"Strategy": "Discount 20%", "Success": 40, "Cost": 50000},
        {"Strategy": "Free Upgrade", "Success": 35, "Cost": 45000},
        {"Strategy": "Contract Incentive", "Success": 55, "Cost": 70000},
        {"Strategy": "Loyalty Points", "Success": 20, "Cost": 20000},
    ]

    strategy_results = []
    high_risk = df[df['Churn_Prob'] > 0.5]
    for s in strategies:
        saved = int(len(high_risk) * s['Success'] / 100)
        rev_saved = high_risk.nlargest(saved, 'CLV')['CLV'].sum()
        net = rev_saved - s['Cost']
        strategy_results.append({
            'Strategy': s['Strategy'],
            'Customers Saved': saved,
            'Revenue Saved': rev_saved,
            'Campaign Cost': s['Cost'],
            'Net Benefit': net,
            'ROI %': round((net / s['Cost']) * 100, 1)
        })

    strategy_df = pd.DataFrame(strategy_results)
    fig = px.bar(strategy_df,
                 x='Strategy', y='Net Benefit',
                 color='Net Benefit',
                 color_continuous_scale='RdYlGn',
                 text=strategy_df['ROI %'].apply(lambda x: f"{x}% ROI"),
                 title='Net Benefit by Retention Strategy')
    fig.update_traces(textposition='outside')
    fig = dark_chart(fig)
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(strategy_df.style.highlight_max(
        subset=['Net Benefit'], color='#90EE90'
    ), use_container_width=True)

    best_strategy = strategy_df.loc[strategy_df['Net Benefit'].idxmax(), 'Strategy']
    st.success(f"✅ **Recommended Strategy: {best_strategy}** — Highest net benefit across all options.")

# ================================
# POWER 2 — AI ADVISOR
# ================================
elif module == "🤖 AI Advisor":
    st.header("🤖 AI Advisor — Powered by Groq LLaMA")
    st.markdown("Ask anything about customer churn, retention strategies, and your data.")

    from groq import Groq

    results, best_name, trained_models, X_test, y_test, X = get_trained_models()
    best_model = trained_models[best_name]
    df = get_raw_data(DATA_PATH)
    df_proc = get_processed_data()

    df['CLV'] = df['tenure'] * df['MonthlyCharges']
    df['Churn_Prob'] = best_model.predict_proba(
        df_proc.drop('Churn', axis=1)
    )[:, 1]

    churn_rate = df['Churn_Prob'].mean()
    high_risk_count = (df['Churn_Prob'] > 0.5).sum()
    avg_clv = df['CLV'].mean()
    top_contract = df.groupby('Contract')['Churn_Prob'].mean().idxmax()

    data_context = f"""
    You are a customer retention analytics expert analyzing a telecom company dataset.
    
    Current Data Summary:
    - Total Customers: {len(df):,}
    - Average Churn Probability: {churn_rate:.1%}
    - High Risk Customers (>50%): {high_risk_count:,}
    - Average Customer Lifetime Value: ${avg_clv:,.0f}
    - Highest Churn Contract Type: {top_contract}
    - Best ML Model: {best_name}
    - Total Revenue at Risk: ${df[df['Churn_Prob'] > 0.5]['CLV'].sum():,.0f}
    
    Provide specific, actionable insights based on this data.
    Keep responses concise and business-focused.
    """

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    st.markdown("---")

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask about churn patterns, retention strategies, customer segments..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing your data..."):
                try:
                    client = Groq(api_key=GROQ_API_KEY)
                    response = client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=[
                            {"role": "system", "content": data_context},
                            *st.session_state.chat_history
                        ],
                        max_tokens=1000
                    )
                    answer = response.choices[0].message.content
                    st.markdown(answer)
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": answer}
                    )
                except Exception as e:
                    st.error(f"Error: {e}")

    st.markdown("---")
    st.markdown("**💡 Try asking:**")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("Why are month-to-month customers churning?")
    with col2:
        st.info("What retention strategy works best for high-value customers?")
    with col3:
        st.info("Which customer segment should I prioritise?")

# ================================
# POWER 3 — PDF REPORT
# ================================
elif module == "📄 PDF Report":
    st.header("📄 Automated PDF Report Generator")
    st.markdown("Generate a **professional executive report** with one click.")

    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    import io
    from datetime import datetime

    results, best_name, trained_models, X_test, y_test, X = get_trained_models()
    best_model = trained_models[best_name]
    df = get_raw_data(DATA_PATH)
    df_proc = get_processed_data()

    df['CLV'] = df['tenure'] * df['MonthlyCharges']
    df['Churn_Prob'] = best_model.predict_proba(
        df_proc.drop('Churn', axis=1)
    )[:, 1]
    df['Risk_Tier'] = pd.cut(df['Churn_Prob'],
                              bins=[0, 0.3, 0.5, 0.7, 1.0],
                              labels=['Low', 'Medium', 'High', 'Critical'])

    st.markdown("---")
    st.subheader("📋 Report Preview")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Customers", f"{len(df):,}")
    with col2:
        st.metric("Avg Churn Risk", f"{df['Churn_Prob'].mean():.1%}")
    with col3:
        st.metric("Revenue at Risk", f"${df[df['Churn_Prob'] > 0.5]['CLV'].sum():,.0f}")
    with col4:
        st.metric("Critical Customers", f"{(df['Churn_Prob'] > 0.7).sum():,}")

    st.markdown("---")
    model_df = pd.DataFrame(results).T
    st.subheader("Model Performance")
    st.dataframe(model_df, use_container_width=True)

    if st.button("📄 Generate & Download PDF Report", use_container_width=True):
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []

        title_style = styles['Title']
        heading_style = styles['Heading1']
        normal_style = styles['Normal']

        story.append(Paragraph("ChurnGuard — Executive Report", title_style))
        story.append(Paragraph(f"Customer Retention Intelligence Platform", styles['Heading2']))
        story.append(Paragraph(f"Generated: {datetime.now().strftime('%B %d, %Y %H:%M')}", normal_style))
        story.append(Spacer(1, 0.3 * inch))

        story.append(Paragraph("Executive Summary", heading_style))
        summary_text = f"""
        This report provides a comprehensive analysis of customer churn risk for {len(df):,} customers.
        The average churn probability is {df['Churn_Prob'].mean():.1%}, with {(df['Churn_Prob'] > 0.5).sum():,}
        customers classified as high risk. Total revenue at risk amounts to
        ${df[df['Churn_Prob'] > 0.5]['CLV'].sum():,.0f}. Immediate retention action is recommended
        for {(df['Churn_Prob'] > 0.7).sum():,} critical risk customers.
        """
        story.append(Paragraph(summary_text, normal_style))
        story.append(Spacer(1, 0.2 * inch))

        story.append(Paragraph("Key Metrics", heading_style))
        metrics_data = [
            ['Metric', 'Value'],
            ['Total Customers', f"{len(df):,}"],
            ['Average Churn Probability', f"{df['Churn_Prob'].mean():.1%}"],
            ['High Risk Customers (>50%)', f"{(df['Churn_Prob'] > 0.5).sum():,}"],
            ['Critical Risk Customers (>70%)', f"{(df['Churn_Prob'] > 0.7).sum():,}"],
            ['Total Revenue at Risk', f"${df[df['Churn_Prob'] > 0.5]['CLV'].sum():,.0f}"],
            ['Average Customer CLV', f"${df['CLV'].mean():,.0f}"],
            ['Best ML Model', best_name],
        ]
        metrics_table = Table(metrics_data, colWidths=[3 * inch, 3 * inch])
        metrics_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f77b4')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f4f8')]),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('PADDING', (0, 0), (-1, -1), 8),
        ]))
        story.append(metrics_table)
        story.append(Spacer(1, 0.2 * inch))

        story.append(Paragraph("Model Performance", heading_style))
        model_data = [['Model', 'Accuracy', 'Precision', 'Recall', 'F1', 'AUC-ROC']]
        for model_name, metrics in results.items():
            model_data.append([
                model_name,
                f"{metrics['Accuracy']:.4f}",
                f"{metrics['Precision']:.4f}",
                f"{metrics['Recall']:.4f}",
                f"{metrics['F1 Score']:.4f}",
                f"{metrics['AUC-ROC']:.4f}"
            ])
        model_table = Table(model_data, colWidths=[1.8*inch, 0.9*inch, 0.9*inch, 0.9*inch, 0.9*inch, 0.9*inch])
        model_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f77b4')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f4f8')]),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('PADDING', (0, 0), (-1, -1), 6),
        ]))
        story.append(model_table)
        story.append(Spacer(1, 0.2 * inch))

        story.append(Paragraph("Recommendations", heading_style))
        recommendations = [
            "1. IMMEDIATE: Contact all 428 critical risk customers within 24 hours",
            "2. HIGH PRIORITY: Launch retention campaign targeting high-value month-to-month customers",
            "3. OFFER: Provide contract upgrade incentives to reduce churn probability by up to 32%",
            "4. MONITOR: Set up weekly churn probability monitoring for medium risk segment",
            "5. ANALYSE: Investigate why Fiber optic customers show higher churn rates",
        ]
        for rec in recommendations:
            story.append(Paragraph(rec, normal_style))
            story.append(Spacer(1, 0.05 * inch))

        story.append(Spacer(1, 0.2 * inch))
        story.append(Paragraph(
            f"Report generated by ChurnGuard | Built by Shubham Reddy | RMIT University Melbourne",
            styles['Italic']
        ))

        doc.build(story)
        buffer.seek(0)

        st.download_button(
            label="⬇️ Download PDF Report",
            data=buffer,
            file_name=f"ChurnGuard_Report_{datetime.now().strftime('%Y%m%d')}.pdf",
            mime="application/pdf"
        )
        st.success("✅ PDF Report generated successfully!")

# ================================
# POWER 4 — UPLOAD YOUR OWN DATA
# ================================
elif module == "📁 Upload Your Own Data":
    st.header("📁 Upload Your Own Customer Data")
    st.markdown("Upload any customer CSV and ChurnGuard will **automatically analyse churn risk.**")

    st.markdown("---")
    st.info("""
    **Required columns (minimum):**
    tenure, MonthlyCharges, TotalCharges, Contract, InternetService, PaymentMethod
    """)

    uploaded_file = st.file_uploader("Upload Customer CSV", type=['csv'])

    if uploaded_file is not None:
        try:
            user_df = pd.read_csv(uploaded_file)
            st.success(f"✅ Loaded {len(user_df):,} customers from {uploaded_file.name}")

            st.subheader("📋 Data Preview")
            st.dataframe(user_df.head(10), use_container_width=True)

            st.subheader("📊 Data Profile")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Rows", f"{len(user_df):,}")
            with col2:
                st.metric("Total Columns", f"{len(user_df.columns):,}")
            with col3:
                missing = user_df.isnull().sum().sum()
                st.metric("Missing Values", f"{missing:,}")

            st.markdown("---")
            st.subheader("🔍 Column Mapping")
            st.markdown("Mapping your columns to ChurnGuard's model features:")

            results, best_name, trained_models, X_test, y_test, X = get_trained_models()
            best_model = trained_models[best_name]

            try:
                temp_df = user_df.copy()
                temp_df['TotalCharges'] = pd.to_numeric(
                    temp_df['TotalCharges'], errors='coerce'
                )
                temp_df['TotalCharges'] = temp_df['TotalCharges'].fillna(
                    temp_df['TotalCharges'].median()
                )

                if 'customerID' in temp_df.columns:
                    temp_df = temp_df.drop('customerID', axis=1)
                if 'Churn' in temp_df.columns:
                    temp_df = temp_df.drop('Churn', axis=1)

                binary_cols = ['gender', 'Partner', 'Dependents',
                               'PhoneService', 'PaperlessBilling']
                for col in binary_cols:
                    if col in temp_df.columns:
                        temp_df[col] = temp_df[col].map(
                            {'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0}
                        )

                cat_cols = temp_df.select_dtypes(include='object').columns.tolist()
                temp_df = pd.get_dummies(temp_df, columns=cat_cols, drop_first=True)

                for col in X.columns:
                    if col not in temp_df.columns:
                        temp_df[col] = 0
                temp_df = temp_df[X.columns]
                temp_df = temp_df.astype(float)

                churn_probs = best_model.predict_proba(temp_df)[:, 1]
                user_df['Churn_Probability'] = churn_probs
                user_df['Risk_Tier'] = pd.cut(churn_probs,
                                               bins=[0, 0.3, 0.5, 0.7, 1.0],
                                               labels=['Low', 'Medium', 'High', 'Critical'])

                st.markdown("---")
                st.subheader("🎯 Churn Predictions for Your Data")

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Customers", f"{len(user_df):,}")
                with col2:
                    st.metric("Avg Churn Risk", f"{churn_probs.mean():.1%}")
                with col3:
                    st.metric("High Risk", f"{(churn_probs > 0.5).sum():,}")
                with col4:
                    st.metric("Critical Risk", f"{(churn_probs > 0.7).sum():,}")

                st.dataframe(user_df[['Churn_Probability', 'Risk_Tier'] +
                                      list(user_df.columns[:5])].head(20),
                             use_container_width=True)

                fig = px.histogram(user_df, x='Churn_Probability',
                                   nbins=50,
                                   color='Risk_Tier',
                                   color_discrete_map={
                                       'Low': '#00cc44',
                                       'Medium': '#ffcc00',
                                       'High': '#ff8800',
                                       'Critical': '#ff4444'
                                   },
                                   title='Churn Probability Distribution')
                fig = dark_chart(fig)
                st.plotly_chart(fig, use_container_width=True)

                csv = user_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "⬇️ Download Results CSV",
                    csv,
                    "churnguard_predictions.csv",
                    "text/csv"
                )
                st.success("✅ Analysis complete!")

            except Exception as e:
                st.error(f"Column mapping error: {e}")
                st.warning("Please ensure your CSV has similar columns to the Telco dataset.")

        except Exception as e:
            st.error(f"Error loading file: {e}")
    else:
        st.markdown("### 📥 No file uploaded yet")
        st.markdown("Upload a CSV file above to get started.")
        st.markdown("**Example columns needed:** tenure, MonthlyCharges, TotalCharges, Contract, InternetService, PaymentMethod")

else:
    st.info("✅ All modules loaded!")