"""
This is the main module for the web app front end.
"""

import atexit
from datetime import datetime, timezone
import os
import cv2
from bson.binary import Binary
import requests
from dotenv import load_dotenv
from flask import Flask, Response, jsonify, render_template
from pymongo import MongoClient

# Load environment variables
load_dotenv()

app = Flask(__name__)
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
client = MongoClient(MONGO_URI)
db = client["object_detection"]
collection = db["detected_objects"]

# Initialize camera
camera = cv2.VideoCapture(0)  # pylint: disable=no-member
atexit.register(lambda: camera.release())  # pylint: disable=W0108


@app.route("/")
def index():
    """Index returns index home page HTML."""
    return render_template("index.html")


@app.route("/capture_and_process", methods=["POST"])
def capture_and_process():
    """Capture a frame, save it to MongoDB, and trigger processing."""
    # Capture a frame
    success, frame = camera.read()
    if not success:
        return jsonify({"error": "Failed to capture frame"}), 500

    # Encode the frame to JPEG format
    _, buffer = cv2.imencode(".jpg", frame)  # pylint: disable=no-member
    image_binary = Binary(buffer)

    # Save to MongoDB
    document = {
        "image": image_binary,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "pending",  # Waiting for processing
        "detections": [],
    }
    result = collection.insert_one(document)

    # Trigger processing in the ML app
    ml_response = requests.post("http://localhost:5001/process_pending", timeout=90)
    if ml_response.status_code != 200:
        return jsonify({"error": "Processing failed"}), 500

    return (
        jsonify(
            {
                "message": "Frame captured and processed",
                "id": str(result.inserted_id),
                "detections": ml_response.json().get("detections", []),
            }
        ),
        200,
    )


@app.route("/latest_detection", methods=["GET"])
def latest_detection():
    """Fetch the latest detection results from MongoDB."""
    detection = collection.find_one(
        {"status": "processed"},
        sort=[("timestamp", -1)],
    )
    if not detection:
        return jsonify({"message": "No processed detections available"}), 404

    detections = detection.get("detections", [])
    return jsonify({"timestamp": detection["timestamp"], "labels": detections})


@app.route("/video_feed")
def video_feed():
    """Stream the video feed."""
    return Response(
        generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


def generate_frames():
    """Generate video frames for streaming."""
    while True:
        success, frame = camera.read()
        if not success:
            break
        _, buffer = cv2.imencode(".jpg", frame)  # pylint: disable=no-member
        frame_bytes = buffer.tobytes()  # Convert buffer to bytes
        yield b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
