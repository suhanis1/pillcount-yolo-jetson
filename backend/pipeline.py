import cv2
from backend import config as C
from backend.decisions import analyze

GREEN, GREY, RED, YELLOW = (0, 200, 0), (150, 150, 150), (0, 0, 255), (0, 215, 255)
BANNER_COLOR = {"ok": GREEN, "short": YELLOW, "over": YELLOW,
                "wrong": RED, "spread": YELLOW, "none": GREY}


def _box(img, d, color, label):
    x1, y1, x2, y2 = map(int, d.box)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    cv2.putText(img, label, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


def process_frame(frame, detector, target=C.DEFAULT_TARGET, bottle=C.DEFAULT_BOTTLE,
                  expected=C.DEFAULT_EXPECTED, conf_sure=C.CONF_SURE):
    """Main backend entry point (the frontend will call this).
    Returns (annotated_image, result_dict)."""
    result = analyze(detector.detect(frame), target, bottle, expected, conf_sure)
    out = frame.copy()
    wrong_ids = {id(d) for d in result["wrong"]}
    for d in result["unsure"]:
        _box(out, d, GREY, f"unsure {d.conf:.2f}")
    for d in result["sure"]:
        _box(out, d, RED if id(d) in wrong_ids else GREEN, f"{d.cls_name} {d.conf:.2f}")
    text = result["banner"] + (f" | total {result['total']}" if result["total"] is not None else "")
    cv2.rectangle(out, (0, 0), (out.shape[1], 30), BANNER_COLOR[result["status"]], -1)
    cv2.putText(out, text, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    return out, result
