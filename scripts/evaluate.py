"""Usage: python3 scripts/evaluate.py labels.csv   (csv columns: image,true_count)"""
import csv, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
from backend.detector import PillDetector
from backend.decisions import split_by_confidence

det = PillDetector()
rows = list(csv.DictReader(open(sys.argv[1])))
exact, errs = 0, []
for r in rows:
    sure, _ = split_by_confidence(det.detect(cv2.imread(r["image"])))
    err = abs(len(sure) - int(r["true_count"]))
    exact += err == 0
    errs.append(err)
    print(r["image"], "true", r["true_count"], "pred", len(sure))
print(f"Exact-count accuracy: {exact / len(rows):.0%} | Mean absolute error: {sum(errs) / len(errs):.2f}")
