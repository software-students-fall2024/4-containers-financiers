"""
This module contains tests for the ML client. Run with 'python -m pytest test_app.py' 
or to see with coverage run with 'python -m pytest --cov=app test_app.py'
"""


from unittest.mock import patch, MagicMock
from io import BytesIO
import base64
import pytest
from PIL import Image
import cv2
import requests
from app import app, detect_objects

app.config["TESTING"] = True


@pytest.fixture(name="test_client")
def fixture_test_client():
    """Mock client fixture"""
    with app.test_client() as client:
        yield client


@patch("app.model")
def test_detect_objects(mock_model):
    """Test the object detection functionality by mocking YOLOv5's predictions."""
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
    """Test the encoding of an image to base64.
    Verifies that the encoded image is a valid non-empty string."""
    # create a sample blank image
    image = Image.new("RGB", (100, 100), color="white")
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    encoded_image = base64.b64encode(buffered.getvalue()).decode("utf-8")

    # check encoding result is a string and not empty
    assert isinstance(encoded_image, str)
    assert len(encoded_image) > 0

# send POST request to /api/detect for image screenshots from webcam

# capture an image from a webcam feed
def capture_image_from_webcam():
    """Capture an image from the webcam and return it as bytes.
    Initializes the webcam, captures a frame, and converts it to JPEG format.
    Returns the image bytes if successful, or None if the capture fails."""
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    # Read a frame from the webcam
    ret, frame = cap.read()
    # Release the webcam
    cap.release()
    cv2.destroyAllWindows()

    if not ret:
        print("Failed to capture image")
        return None

    # Convert to JPEG format for the API
    _, buffer = cv2.imencode('.jpg', frame)
    image_bytes = buffer.tobytes()
    return image_bytes

def send_image_to_detect(image_bytes):
    """Send the captured image to the /api/detect route for object detection.
    Sends the image as a form-data payload and prints the detection result.
    Handles errors if the request fails."""
    # Create a form-data payload
    files = {'file': ('webcam-image.jpg', image_bytes, 'image/jpeg')}
    url = 'http://localhost:3001/api/detect'

    try:
        # Send the image to the ML client for detection
        response = requests.post(url, files=files)
        response.raise_for_status()  # Raise an error if the request was unsuccessful
        result = response.json()
        print("Detection Result:", result)
        display_detection_result(result)
    except requests.RequestException as e:
        print("Error:", e)

def display_detection_result(result):
    """Display the detection results in the console.
    Iterates over the detected objects and prints the label and confidence score."""
    print(f"Timestamp: {result['timestamp']}")
    print("Detected Objects:")
    for obj in result['detected_objects']:
        print(f" - {obj['label']}: {obj['confidence']:.2f}")

# Capture and process an image when script runs
image_bytes_original = capture_image_from_webcam()
if image_bytes_original:
    send_image_to_detect(image_bytes_original)
