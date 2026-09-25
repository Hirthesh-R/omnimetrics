# 📊 OmniMetrics: AI Business Decision & Predictive Analytics Engine

OmniMetrics is a hybrid predictive analytics engine that combines PyTorch-based NLP sentiment signals from unstructured customer support tickets with tabular account metrics to forecast customer churn using XGBoost.

---

## 🏗️ Architecture & Features

* **Two-Tier Hybrid ML Pipeline**: Integrates transformer sentiment scoring with structured time-series tabular modeling.
* **NLP Sentiment Engine**: Utilizes Hugging Face DistilBERT (`distilbert-base-uncased-finetuned-sst-2-english`) via PyTorch to generate continuous risk indicators from support ticket text.
* **Predictive Churn Engine**: Feature engineering with Scikit-Learn scaling and high-accuracy churn forecasting powered by XGBoost.
* **Interactive Dashboard**: Full-featured Streamlit UI featuring real-time risk gauges, diagnostic breakdowns, and XGBoost feature importance visualizations.
* **Automated CI/CD**: Continuous Integration pipeline configured via GitHub Actions executing automated `pytest` suites.

---

## 🛠️ Tech Stack

* **Language**: Python 3.12
* **NLP & Deep Learning**: PyTorch, Hugging Face Transformers
* **Machine Learning**: XGBoost, Scikit-Learn, Pandas, NumPy
* **Visualization & UI**: Streamlit, Plotly
* **DevOps & Testing**: Docker, Pytest, GitHub Actions

---

## 🚀 Quickstart Guide

### 1. Clone Repository & Install Dependencies

```bash
git clone [https://github.com/Hirthesh-R/omnimetrics.git](https://github.com/Hirthesh-R/omnimetrics.git)
cd omnimetrics
pip install -r requirements.txt