from ultralytics import YOLO
import cv2
import cvzone
import math
import time
import pandas as ps

# cap = cv2.VideoCapture(0)



cap = cv2.VideoCapture("fire.mp4")


cap.set(3, 1280)
cap.set(4, 720)

model = YOLO("__best.pt")
classNames = ['fire', 'smoke']

while True:
    success, img = cap.read()
    results = model(img, stream=True)
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
            # Class Name
            cls = int(box.cls[0])

            if classNames[cls] == 'fire' and conf > 0.50:
                print("Fire detected")
            elif classNames[cls] == 'smoke' and conf > 0.50:
                print("Smoke detected")

            cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
