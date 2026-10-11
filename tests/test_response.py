import json

import pytest
from fastapi.testclient import TestClient

from sentiment_api.api.models.sentiment import PredictResponse
from sentiment_api.main import app

client = TestClient(app)
LABELS = {"negative", "neutral", "positive"}
SAMPLES = [
    "I love this",
    "I hate this",
    "I am neutral",
]


@pytest.mark.parametrize("text", SAMPLES)
def test_predict_returns_valid_json(text: str) -> None:
    response = client.post("/predict", json={"text": text})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    body = json.loads(response.text)
    parsed = PredictResponse.model_validate(body)
    assert parsed.prediction in LABELS
