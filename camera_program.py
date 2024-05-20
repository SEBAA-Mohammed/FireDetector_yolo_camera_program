import cv2
import cvzone
import math
from time import time
import pandas as pd
from ultralytics import YOLO
from supabase import create_client

# Camera configuration
cameraId = 1  # Use 0 for the default camera
direction_id = 1
# Supabase configuration
url = "https://tbbphdosaauasqbbmcsz.supabase.co"
api_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InRiYnBoZG9zYWF1YXNxYmJtY3N6Iiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTcxNTg4MjE4NywiZXhwIjoyMDMxNDU4MTg3fQ.m1AEIQV6NllQkLf9FPLbZmCU87I_07NrHJR66S_YnZU"
supabase = create_client(url, api_key)

# Initialize YOLO model
model = YOLO("__best.pt")
classNames = ['fire', 'smoke']



# cap = cv2.VideoCapture(0)

cap = cv2.VideoCapture("fire.mp4")



# Initialize time variable for controlling insertion frequency
last_insert_time = time()

while cap.isOpened():
    # Read frame from the camera
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection on the frame
    results = model(frame)

    # Process detection results
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(frame, (x1, y1, w, h))
            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            print(conf)

            # Class Name
            cls = int(box.cls[0])
            cvzone.putTextRect(frame, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)

            className = classNames[cls]

            # Check if it's time to insert data
            if time() - last_insert_time > 3 and className and conf > 0.3 and (
                    className == 'fire' or className == 'smoke'):
                # Insert detection information into the database
                detectionInfo = {
                    "type": className,
                    "position": "left",
                    "camera_id": cameraId,
                    "direction_id": direction_id
                }
                response = supabase.table("detections").insert(detectionInfo).execute()
                print("Data inserted successfully:", response)
                # Update last insert time
                last_insert_time = time()

    # Display the frame with detected objects
    cv2.imshow("Frame", frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()