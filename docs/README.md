# AI Sack Counting System

This project demonstrates how to train an object detection model using YOLOv8
and track objects crossing a virtual line for counting purposes.

## Folders
- code/: scripts for training, streaming, and tracking
- dataset/: sample images and YOLOv8 labels
- labs/: workshop labs
- slides/: slide deck

## Quick Start
```
pip install ultralytics opencv-python deep_sort_realtime
python code/train_yolo.py --data dataset/data.yaml
python code/counting_with_tracking.py --source dataset/sample_video.mp4
```
