from fastapi import FastAPI
from sentiment_api.api.models.sentiment import PredictRequest, PredictResponse
from sentiment_api.inference import predict as predict_sentiment

app = FastAPI()


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "Welcome to the Sentiment API!"}


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
def predict(request: PredictRequest) -> PredictResponse:
    return PredictResponse(prediction=predict_sentiment(request.text))
