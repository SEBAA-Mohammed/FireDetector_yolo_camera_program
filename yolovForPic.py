import cv2
import cvzone
import math
import time
import pandas as pd
from ultralytics import YOLO
from supabase import create_client, Client

#camera coordination
cameraId = 1;

#supabase configuration
url = "https://tbbphdosaauasqbbmcsz.supabase.co"
api_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InRiYnBoZG9zYWF1YXNxYmJtY3N6Iiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTcxNTg4MjE4NywiZXhwIjoyMDMxNDU4MTg3fQ.m1AEIQV6NllQkLf9FPLbZmCU87I_07NrHJR66S_YnZU"
# Initialize the Supabase client
supabase = create_client(url, api_key)

detectionInfo = {
    "type": "",
    "position": "top-right",
    "camera_id": cameraId,
}

# Load the image
img = cv2.imread("testpic.jpg")

# Initialize YOLO model
model = YOLO("__best.pt")
classNames = ['fire', 'smoke']

# Perform object detection on the image
results = model(img)

# Create a DataFrame to store results
df = pd.DataFrame(columns=['Class', 'Confidence', 'X1', 'Y1', 'X2', 'Y2'])

fire_detected = False

# Process detection results
for r in results:
    boxes = r.boxes
    for box in boxes:
        # Bounding Box
        x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(img, (x1, y1, w, h))
        # Confidence
        conf = math.ceil((box.conf[0] * 100)) / 100
        print(conf)

        # Class Name
        cls = int(box.cls[0])
        cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)

        className = classNames[cls]

        if className and conf > 0.5:
            detectionInfo["type"] = className
            print(detectionInfo)
            fire_detected = True


if fire_detected:
    response = supabase.table("detections").insert(detectionInfo).execute()
    print("Data inserted successfully:", response)

# Display the image with detected objects
cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
