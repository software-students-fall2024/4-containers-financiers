import cv2

camera = cv2.VideoCapture(0)  # Use 0 for default camera

if not camera.isOpened():
    print("Error: Camera not accessible. Check permissions or device.")
else:
    print("Camera is accessible!")
    success, frame = camera.read()
    if success:
        print("Frame captured successfully.")
    else:
        print("Error capturing frame.")
camera.release()
