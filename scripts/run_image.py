import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
from backend import config as C
from backend.detector import PillDetector
from backend.pipeline import process_frame

p = argparse.ArgumentParser()
p.add_argument("image")
p.add_argument("--weights", default=C.WEIGHTS_PATH)
p.add_argument("--target", type=int, default=C.DEFAULT_TARGET)
p.add_argument("--bottle", type=int, default=C.DEFAULT_BOTTLE)
p.add_argument("--expected", default=C.DEFAULT_EXPECTED)
p.add_argument("--out", default="outputs/result.jpg")
a = p.parse_args()

frame = cv2.imread(a.image)
if frame is None:
    sys.exit(f"Could not read image: {a.image}")
img, r = process_frame(frame, PillDetector(a.weights), a.target, a.bottle, a.expected)
os.makedirs(os.path.dirname(a.out), exist_ok=True)
cv2.imwrite(a.out, img)
print(r["banner"], "| total:", r["total"], "| counts:", r["counts"],
      "| full bottles:", r["full_bottles"], "| leftover:", r["leftover"])
print("Saved:", a.out)
