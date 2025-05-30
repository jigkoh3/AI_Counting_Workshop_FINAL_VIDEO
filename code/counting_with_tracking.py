from ultralytics import YOLO
import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort

model = YOLO("yolov8n.pt")
tracker = DeepSort()

cap = cv2.VideoCapture("dataset/sample_video.mp4")
count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    results = model(frame)[0]
    detections = [(d.boxes.xyxy[0].tolist(), d.conf[0], 'sack') for d in results if d.boxes]
    tracks = tracker.update_tracks(detections, frame=frame)
    for track in tracks:
        if not track.is_confirmed():
            continue
        bbox = track.to_ltrb()
        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255,0,0), 2)
    cv2.imshow("Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
