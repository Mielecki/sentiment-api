import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from sentiment_api.api.models.sentiment import PredictRequest
from sentiment_api.main import app

client = TestClient(app)


def test_non_empty_text_is_accepted() -> None:
    request = PredictRequest(text="string")

    assert request.text == "string"


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"text": ""}, "String should have at least 1 character"),
        ({}, "Field required"),
        ({"text": None}, "Input should be a valid string"),
        ({"text": 123}, "Input should be a valid string"),
    ],
)
def test_invalid_text_is_rejected(payload: dict[str, object], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        PredictRequest.model_validate(payload)
