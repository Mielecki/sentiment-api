from pathlib import Path

import pytest
from joblib import load

from sentiment_api.inference import predict

CLASSIFIER_PATH = Path(__file__).resolve().parents[1] / "models" / "classifier.joblib"
LABELS = {"negative", "neutral", "positive"}
SAMPLES = [
    "I love this",
    "I hate this",
    "I am neutral",
]


def test_classifier_loads_without_error() -> None:
    classifier = load(CLASSIFIER_PATH)

    assert callable(getattr(classifier, "predict"))


@pytest.mark.parametrize("text", SAMPLES)
def test_inference_returns_label(text: str) -> None:
    assert predict(text) in LABELS
