"""
This module contains tests for the ML client. Run with 'python -m pytest test_app.py' 
or to see with coverage run with 'python -m pytest --cov=app test_app.py'
"""

from unittest.mock import patch, MagicMock
from io import BytesIO
import base64
import pytest
from PIL import Image
from app import app, detect_objects

app.config["TESTING"] = True


@pytest.fixture(name="test_client")
def fixture_test_client():
    """Mock client fixture"""
    with app.test_client() as client:
        yield client


@patch("app.model")
def test_detect_objects(mock_model):
    """Test object detection using mocked YOLOv5 model predictions."""
    # mock YOLOv5's output
    mock_results = MagicMock()
    mock_results.pandas.return_value.xyxy = [
        MagicMock(
            to_dict=MagicMock(
                return_value=[
                    {"name": "person", "confidence": 0.98},
                    {"name": "cat", "confidence": 0.85},
                ]
            )
        )
    ]
    mock_model.return_value = mock_results
    image = Image.new("RGB", (224, 224), color="white")  # create a blank image

    # run detection
    detected_objects = detect_objects(image)

    # check if returned detected objects match expected format
    assert isinstance(detected_objects, list)
    assert len(detected_objects) == 2  # we mocked 2 predictions
    for obj in detected_objects:
        assert "label" in obj
        assert "confidence" in obj


def test_detect_route_no_file(test_client):
    """Test /api/detect route when no file is provided."""
    response = test_client.post("/api/detect")
    data = response.get_data(as_text=True)

    assert response.status_code == 400
    assert "No image file provided." in data


def test_detect_route_with_file(test_client):
    """Test /api/detect route with an image file."""
    # create a blank image and save to buffer
    image = Image.new("RGB", (224, 224), color="white")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)

    with patch("app.detect_objects") as mock_detect_objects:
        mock_detect_objects.return_value = [
            {"label": "person", "confidence": 0.98},
            {"label": "cat", "confidence": 0.85},
        ]

        response = test_client.post(
            "/api/detect",
            content_type="multipart/form-data",
            data={"file": (buffer, "test.png")},
        )
        data = response.get_json()

    # check response status and content
    assert response.status_code == 200
    assert "detected_objects" in data
    assert len(data["detected_objects"]) == 2
    assert data["detected_objects"][0]["label"] == "person"
    assert data["detected_objects"][1]["label"] == "cat"


def test_encode_image():
    """Test image encoding to base64."""
    # create a sample blank image
    image = Image.new("RGB", (100, 100), color="white")
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    encoded_image = base64.b64encode(buffered.getvalue()).decode("utf-8")

    # check encoding result is a string and not empty
    assert isinstance(encoded_image, str)
    assert len(encoded_image) > 0
