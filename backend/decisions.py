from dataclasses import dataclass
from statistics import median
from backend import config as C


@dataclass
class Detection:
    cls_name: str
    conf: float
    box: tuple  # (x1, y1, x2, y2)


def split_by_confidence(dets, low=C.CONF_IGNORE, high=C.CONF_SURE):
    sure = [d for d in dets if d.conf >= high]
    unsure = [d for d in dets if low <= d.conf < high]
    return sure, unsure


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area = lambda r: (r[2] - r[0]) * (r[3] - r[1])
    union = area(a) + area(b) - inter
    return inter / union if union > 0 else 0.0


def needs_spread(dets, iou_thr=C.IOU_OVERLAP, size_factor=C.SIZE_FACTOR):
    for i in range(len(dets)):
        for j in range(i + 1, len(dets)):
            if iou(dets[i].box, dets[j].box) > iou_thr:
                return True
    if len(dets) >= 3:
        areas = [(d.box[2] - d.box[0]) * (d.box[3] - d.box[1]) for d in dets]
        m = median(areas)
        if m > 0 and max(areas) > size_factor * m:
            return True
    return False


def rule1_target(total, target):
    if total == target:
        return "ok", "Count correct"
    if total < target:
        return "short", f"Short by {target - total}"
    return "over", f"Over by {total - target}"


def rule2_wrong_pill(dets, expected_type):
    return [d for d in dets if d.cls_name != expected_type]


def rule3_batching(total, bottle_size):
    return divmod(total, bottle_size)  # (full bottles, leftover)


def analyze(all_dets, target, bottle_size, expected_type, conf_sure=C.CONF_SURE):
    sure, unsure = split_by_confidence(all_dets, high=conf_sure)
    base = {"sure": sure, "unsure": unsure, "wrong": [], "counts": {},
            "total": None, "full_bottles": None, "leftover": None}
    if not sure:
        return {**base, "status": "none", "banner": "No pills detected"}
    if needs_spread(sure):
        return {**base, "status": "spread", "banner": "Spread pills out"}
    total = len(sure)
    counts = {}
    for d in sure:
        counts[d.cls_name] = counts.get(d.cls_name, 0) + 1
    status, banner = rule1_target(total, target)
    wrong = rule2_wrong_pill(sure, expected_type)
    full, left = rule3_batching(total, bottle_size)
    if wrong:
        status, banner = "wrong", "Wrong pill detected"
    return {**base, "status": status, "banner": banner, "total": total,
            "counts": counts, "wrong": wrong, "full_bottles": full, "leftover": left}


class Stabilizer:
    """Accept a banner/count only after it stays unchanged for `hold` seconds."""
    def __init__(self, hold=C.HOLD_SECONDS):
        self.hold, self.value, self.since, self.stable = hold, None, None, None

    def update(self, value, now):
        if value != self.value:
            self.value, self.since = value, now
        elif now - self.since >= self.hold:
            self.stable = value
        return self.stable
