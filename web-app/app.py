import os
import cv2
from flask import Flask, render_template, jsonify, Response
from pymongo import MongoClient
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# MongoDB setup
mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017")
logging.debug(f"Connecting to MongoDB at {mongo_uri}")
try:
    client = MongoClient(mongo_uri)
    db = client["object_detection"]
    collection = db["detected_objects"]
    logging.debug("MongoDB connection established.")
except Exception as e:
    logging.error(f"Failed to connect to MongoDB: {e}")

# Initialize video capture
logging.debug("Initializing video capture.")
camera = cv2.VideoCapture(0)  # Use 0 for default webcam
if not camera.isOpened():
    logging.error("Failed to access the camera. Check permissions or device.")

def generate_frames():
    """
    Generate video frames from the camera for streaming.
    """
    while True:
        success, frame = camera.read()
        if not success:
            logging.error("Failed to capture video frame.")
            break
        try:
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        except Exception as e:
            logging.error(f"Error encoding video frame: {e}")
            break

@app.route("/video_feed")
def video_feed():
    logging.debug("Video feed requested.")
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/")
def index():
    logging.debug("Index page requested.")
    return render_template("index.html")

@app.route("/data")
def get_data():
    logging.debug("Data endpoint requested.")
    try:
        data = list(collection.find({}, {"_id": 0}))
        logging.debug(f"Data fetched from MongoDB: {data}")
        return jsonify(data)
    except Exception as e:
        logging.error(f"Error fetching data from MongoDB: {e}")
        return jsonify({"error": "Failed to fetch data from the database"}), 500

if __name__ == "__main__":
    try:
        logging.debug("Starting Flask application.")
        app.run(host="0.0.0.0", port=5000, debug=True)
    finally:
        logging.debug("Releasing camera resources.")
        camera.release()