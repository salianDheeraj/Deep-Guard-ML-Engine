from pydantic import BaseModel
from typing import List

class PredictionResponse(BaseModel):
    prediction: str           # "real" or "fake"
    overall_confidence: float  # Confidence score between 0 and 1

