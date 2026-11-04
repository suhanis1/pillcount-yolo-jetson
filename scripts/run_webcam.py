import argparse, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
from backend import config as C
from backend.decisions import Stabilizer
from backend.detector import PillDetector
from backend.pipeline import process_frame

p = argparse.ArgumentParser()
p.add_argument("--camera", type=int, default=0)
p.add_argument("--weights", default=C.WEIGHTS_PATH)
p.add_argument("--target", type=int, default=C.DEFAULT_TARGET)
p.add_argument("--bottle", type=int, default=C.DEFAULT_BOTTLE)
p.add_argument("--expected", default=C.DEFAULT_EXPECTED)
p.add_argument("--max-frames", type=int, default=300)
a = p.parse_args()

det, stab = PillDetector(a.weights), Stabilizer()
cap = cv2.VideoCapture(a.camera)
if not cap.isOpened():
    sys.exit("Cannot open webcam. Check --device /dev/video0 in the docker run command.")
os.makedirs("outputs", exist_ok=True)
t0 = time.time()
for i in range(a.max_frames):
    ok, frame = cap.read()
    if not ok:
        break
    img, r = process_frame(frame, det, a.target, a.bottle, a.expected)
    stable = stab.update((r["banner"], r["total"]), time.time())
    print(f"frame {i} | {r['banner']} | total {r['total']} | stable {stable}")
    if i % 30 == 0:
        cv2.imwrite("outputs/webcam_latest.jpg", img)
print(f"FPS: {i + 1} frames in {time.time() - t0:.1f}s = {(i + 1) / (time.time() - t0):.1f}")
cap.release()
