from fastapi import FastAPI
from sentiment_api.api.models.sentiment import PredictRequest, PredictResponse

app = FastAPI()


@app.post("/predict")
def predict(request: PredictRequest) -> PredictResponse:
    return PredictResponse(prediction="positive")
