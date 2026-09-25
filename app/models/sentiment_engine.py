import torch
from transformers import pipeline


class TicketSentimentEngine:

  def __init__(self):
    device = 0 if torch.cuda.is_available() else -1
    self.classifier = pipeline(
        "sentiment-analysis",
        model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
        device=device,
    )

  def get_sentiment_score(self, text: str) -> float:
    if not text or not text.strip():
      return 0.5

    result = self.classifier(text[:512])[0]
    label = result["label"]
    confidence = result["score"]

    if label == "NEGATIVE":
      return float(confidence)
    else:
      return float(1.0 - confidence)

  def batch_analyze_tickets(self, tickets: list[str]) -> list[float]:
    return [self.get_sentiment_score(ticket) for ticket in tickets]