from fastapi.testclient import TestClient
from sentiment_api.main import app

client = TestClient(app)


def test_predict_positive() -> None:
    response = client.post("/predict", json={"text": "Positive sentence"})
    assert response.status_code == 200
    assert response.json() == {"prediction": "positive"}
