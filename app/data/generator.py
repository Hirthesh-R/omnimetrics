import numpy as np
import pandas as pd


def generate_synthetic_customer_data(n_samples: int = 500) -> pd.DataFrame:
  np.random.seed(42)

  customer_ids = [f"CUST-{1000 + i}" for i in range(n_samples)]
  tenure_months = np.random.randint(1, 48, size=n_samples)
  monthly_charges = np.round(np.random.uniform(29.99, 199.99, size=n_samples), 2)
  login_frequency = np.random.randint(1, 30, size=n_samples)
  support_tickets_count = np.random.poisson(lam=2, size=n_samples)

  positive_tickets = [
      "Loving the platform, very easy to set up!",
      "Great customer service and fast API response times.",
      "The new analytics feature saved our team hours of manual labor.",
      "Super happy with the recent platform stability updates.",
  ]
  negative_tickets = [
      "Terrible downtime today, our pipeline broke unexpectedly.",
      "The pricing is too high for the buggy service provided.",
      "Support response took 3 days. I am considering canceling.",
      "Constant billing errors and confusing interface.",
  ]

  tickets = []
  churn_labels = []

  for i in range(n_samples):
    risk_score = (
        (support_tickets_count[i] * 0.25)
        - (tenure_months[i] * 0.02)
        - (login_frequency[i] * 0.03)
        + (monthly_charges[i] * 0.005)
    )

    prob_churn = 1 / (1 + np.exp(-risk_score))
    churn = 1 if prob_churn > 0.55 else 0
    churn_labels.append(churn)

    if churn == 1 and np.random.rand() > 0.2:
      tickets.append(np.random.choice(negative_tickets))
    else:
      tickets.append(np.random.choice(positive_tickets))

  return pd.DataFrame({
      "customer_id": customer_ids,
      "tenure_months": tenure_months,
      "monthly_charges": monthly_charges,
      "login_frequency": login_frequency,
      "support_tickets_count": support_tickets_count,
      "latest_ticket_text": tickets,
      "churn": churn_labels,
  })


if __name__ == "__main__":
  df = generate_synthetic_customer_data()
  print(f"Generated {len(df)} customer records.")
  print(df.head())