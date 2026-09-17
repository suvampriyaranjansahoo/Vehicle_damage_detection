import numpy as np
import pytest

from app.inference import _confidence_assessment, preprocess
from PIL import Image


def test_preprocess_contract():
    batch = preprocess(Image.new("RGB", (31, 47), "white"))
    assert batch.shape == (1, 224, 224, 3)
    assert batch.dtype == np.float32
    assert float(batch.min()) >= 0
    assert float(batch.max()) <= 1


def test_low_confidence_detection():
    result = _confidence_assessment(np.array([0.34, 0.33, 0.20, 0.05, 0.03, 0.03, 0.02]))
    assert result["review_recommended"] is True
    assert result["status"] == "review_recommended"


def test_confident_detection():
    result = _confidence_assessment(np.array([0.90, 0.03, 0.02, 0.02, 0.01, 0.01, 0.01]))
    assert result["review_recommended"] is False
    assert result["status"] == "confident"
