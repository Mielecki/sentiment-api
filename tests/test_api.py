from fastapi.testclient import TestClient
from sentiment_api.main import app

client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_positive() -> None:
    response = client.post("/predict", json={"text": "I love this"})
    assert response.status_code == 200
    assert response.json() == {"prediction": "positive"}
