import os
import sys

# Force root directory into sys.path before imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.data.generator import generate_synthetic_customer_data
from app.models.churn_engine import HybridChurnEngine

st.set_page_config(
    page_title="OmniMetrics | AI Business Decision Engine",
    page_icon="📊",
    layout="wide",
)


@st.cache_resource
def load_and_train_engine():
    df_raw = generate_synthetic_customer_data(n_samples=600)
    engine = HybridChurnEngine()
    metrics = engine.train(df_raw)
    return engine, df_raw, metrics


st.title("📊 OmniMetrics: AI Predictive Decision Analytics Engine")
st.caption(
    "Hybrid AI Architecture: PyTorch/Hugging Face Sentiment NLP + XGBoost Tabular Machine Learning"
)

with st.spinner("Loading NLP Transformers & Training XGBoost Model..."):
    engine, df_data, metrics = load_and_train_engine()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Model Forecast Accuracy", f"{metrics['accuracy']*100:.1f}%")
col2.metric("ROC-AUC Score", f"{metrics['roc_auc']:.3f}")
col3.metric("F1-Score", f"{metrics['f1_score']:.3f}")
col4.metric("Total Monitored Accounts", f"{len(df_data)}")

st.markdown("---")

tab1, tab2 = st.tabs(["🚀 Live Prediction Simulator", "📈 Analytics & Insights"])

with tab1:
    st.subheader("Interactive Customer Churn Risk Simulator")

    c1, c2 = st.columns(2)

    with c1:
        tenure = st.slider("Tenure (Months)", 1, 60, 12)
        charges = st.slider("Monthly Charges ($)", 20.0, 250.0, 89.99)
        logins = st.slider("Monthly Login Frequency", 0, 30, 8)
        tickets = st.slider("Support Tickets Raised", 0, 10, 3)

    with c2:
        ticket_text = st.text_area(
            "Latest Customer Support Ticket / Feedback:",
            value="Support response took 3 days and the app keeps throwing 500 errors. Considering canceling.",
            height=140,
        )

    if st.button("Analyze Customer Risk", type="primary"):
        cust_input = {
            "tenure_months": tenure,
            "monthly_charges": charges,
            "login_frequency": logins,
            "support_tickets_count": tickets,
        }

        churn_prob, sentiment_score = engine.predict_single(cust_input, ticket_text)

        r1, r2 = st.columns(2)

        with r1:
            fig_gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=churn_prob * 100,
                    domain={"x": [0, 1], "y": [0, 1]},
                    title={"text": "Churn Probability (%)"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "red" if churn_prob > 0.5 else "green"},
                        "steps": [
                            {"range": [0, 40], "color": "lightgreen"},
                            {"range": [40, 70], "color": "yellow"},
                            {"range": [70, 100], "color": "salmon"},
                        ],
                    },
                )
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

        with r2:
            st.markdown("### AI Diagnostic Breakdown")
            st.write(
                f"**NLP Sentiment Risk Score:** `{sentiment_score:.2f}` (0 = Positive, 1 = Negative)"
            )

            if churn_prob > 0.6:
                st.error("⚠️ **HIGH RISK OF CHURN DETECTED**: Retention outreach recommended.")
            elif churn_prob > 0.35:
                st.warning("🟡 **MEDIUM RISK**: Monitor ticket resolutions.")
            else:
                st.success("✅ **LOW RISK**: High engagement and positive feedback.")

with tab2:
    st.subheader("Feature Importance Analysis (XGBoost)")

    importance = engine.model.feature_importances_
    df_imp = pd.DataFrame({
        "Feature": engine.feature_names,
        "Importance": importance,
    }).sort_values("Importance", ascending=True)

    fig_bar = px.bar(
        df_imp,
        x="Importance",
        y="Feature",
        orientation="h",
        title="XGBoost Feature Importances",
    )
    st.plotly_chart(fig_bar, use_container_width=True)