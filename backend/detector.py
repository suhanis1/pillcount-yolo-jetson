from ultralytics import YOLO
from backend import config as C
from backend.decisions import Detection


class PillDetector:
    def __init__(self, weights=C.WEIGHTS_PATH, conf=C.CONF_IGNORE, imgsz=C.IMG_SIZE):
        self.model = YOLO(weights)
        self.conf, self.imgsz = conf, imgsz

    def detect(self, frame):
        r = self.model.predict(frame, conf=self.conf, imgsz=self.imgsz, verbose=False)[0]
        return [Detection(r.names[int(b.cls)], float(b.conf), tuple(map(float, b.xyxy[0])))
                for b in r.boxes]
