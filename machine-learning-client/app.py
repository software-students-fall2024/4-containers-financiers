"""
This module contains the code for the ML client that communicates with the frontend
"""

from datetime import datetime, timezone
from io import BytesIO
import os
import torch
from flask import Flask, jsonify
from PIL import Image
from pymongo.mongo_client import MongoClient

# MongoDB connection
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
client = MongoClient(MONGO_URI)
db = client["object_detection"]
collection = db["detected_objects"]

# Initialize Flask app
app = Flask(__name__)

# Load YOLOv5 model
model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True).to("cpu")


def detect_objects(image):
    """Perform object detection on the provided image."""
    results = model(image)
    detections = results.pandas().xyxy[0].to_dict(orient="records")
    return [
        {"label": det["name"], "confidence": float(det["confidence"])}
        for det in detections
    ]


@app.route("/")
def index():
    """Health check endpoint for the ML app."""
    return jsonify({"status": "running"}), 200


@app.route("/process_pending", methods=["POST"])
def process_pending():
    """
    Process a pending image from MongoDB.
    This endpoint is triggered when the frontend captures a frame.
    """
    document = collection.find_one({"status": "pending"})
    if not document:
        return jsonify({"message": "No pending frames to process"}), 404

    image_data = BytesIO(document["image"])
    image = Image.open(image_data)

    detections = detect_objects(image)

    collection.update_one(
        {"_id": document["_id"]},
        {
            "$set": {
                "status": "processed",
                "detections": detections,
                "processed_at": datetime.now(timezone.utc),
            }
        },
    )

    return jsonify({"message": "Frame processed", "detections": detections}), 200


if __name__ == "__main__":
    # Start automatic processing
    app.run(host="0.0.0.0", port=5001)
