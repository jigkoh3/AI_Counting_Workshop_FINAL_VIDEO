import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)  # or replace with RTSP URL

while True:
    ret, frame = cap.read()
    if not ret:
        break
    results = model(frame)[0]
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
    cv2.imshow("Live Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
