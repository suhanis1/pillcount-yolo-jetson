"""Usage: python3 scripts/train.py [data.yaml]   (best run on Google Colab GPU)"""
import sys
from ultralytics import YOLO
data = sys.argv[1] if len(sys.argv) > 1 else "data/pills/data.yaml"
YOLO("yolov8n.pt").train(data=data, epochs=50, imgsz=640, project="runs", name="pillcount")
