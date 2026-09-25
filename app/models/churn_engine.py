import os
import sys

# Add project root directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from app.models.sentiment_engine import TicketSentimentEngine


class HybridChurnEngine:

  def __init__(self):
    self.sentiment_engine = TicketSentimentEngine()
    self.scaler = StandardScaler()
    self.model = XGBClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss",
    )
    self.feature_names = []

  def prepare_features(
      self, df: pd.DataFrame, fit_scaler: bool = False
  ) -> pd.DataFrame:
    df_transformed = df.copy()

    if "sentiment_risk_score" not in df_transformed.columns:
      df_transformed["sentiment_risk_score"] = (
          self.sentiment_engine.batch_analyze_tickets(
              df_transformed["latest_ticket_text"].tolist()
          )
      )

    df_transformed["ticket_per_tenure"] = (
        df_transformed["support_tickets_count"]
        / (df_transformed["tenure_months"] + 1)
    )
    df_transformed["risk_interaction"] = (
        df_transformed["sentiment_risk_score"]
        * df_transformed["support_tickets_count"]
    )

    features = [
        "tenure_months",
        "monthly_charges",
        "login_frequency",
        "support_tickets_count",
        "sentiment_risk_score",
        "ticket_per_tenure",
        "risk_interaction",
    ]

    self.feature_names = features
    X = df_transformed[features]

    if fit_scaler:
      X_scaled = self.scaler.fit_transform(X)
    else:
      X_scaled = self.scaler.transform(X)

    return pd.DataFrame(X_scaled, columns=features)

  def train(self, df: pd.DataFrame) -> dict:
    X = self.prepare_features(df, fit_scaler=True)
    y = df["churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    self.model.fit(X_train, y_train)

    y_pred = self.model.predict(X_test)
    y_proba = self.model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }
    return metrics

  def predict_single(
      self, customer_data: dict, ticket_text: str
  ) -> tuple[float, float]:
    sentiment_score = self.sentiment_engine.get_sentiment_score(ticket_text)

    row = {
        "tenure_months": customer_data["tenure_months"],
        "monthly_charges": customer_data["monthly_charges"],
        "login_frequency": customer_data["login_frequency"],
        "support_tickets_count": customer_data["support_tickets_count"],
        "latest_ticket_text": ticket_text,
        "sentiment_risk_score": sentiment_score,
    }

    df_single = pd.DataFrame([row])
    X_single = self.prepare_features(df_single, fit_scaler=False)

    churn_prob = float(self.model.predict_proba(X_single)[0][1])
    return churn_prob, sentiment_score