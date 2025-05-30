from ultralytics import YOLO
import argparse

def train(data_yaml, model="yolov8n.pt", epochs=50, imgsz=640):
    model = YOLO(model)  # load pretrained model (nano version)
    model.train(data=data_yaml, epochs=epochs, imgsz=imgsz)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8 for sack counting")
    parser.add_argument("--data", type=str, required=True, help="Path to data.yaml")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="YOLOv8 model to use")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")

    args = parser.parse_args()

    train(args.data, args.model, args.epochs, args.imgsz)
