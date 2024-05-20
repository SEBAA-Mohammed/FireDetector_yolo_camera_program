from ultralytics import  YOLO

model = YOLO("yolov8s.yaml")

results = model.train(data="data.yaml", epochs=50)