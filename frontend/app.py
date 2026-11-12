import os, sys
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from backend import config as C
from backend.decisions import Detection
from backend.pipeline import process_frame

st.set_page_config(page_title="PillCount", layout="wide")
WEIGHTS = os.path.join(ROOT, C.WEIGHTS_PATH)
DEMO_CLASSES = ["tablet", "capsule"]


class FakeDetector:
    """Returns pre-made detections (used by Demo mode only)."""
    def __init__(self, dets):
        self.dets = dets

    def detect(self, frame):
        return self.dets


@st.cache_resource
def load_detector(path):
    from backend.detector import PillDetector
    return PillDetector(path)


def make_demo(n, add_wrong, add_overlap, seed):
    rng = np.random.default_rng(seed)
    frame = np.full((480, 640, 3), 90, dtype=np.uint8)
    r = 16
    centers, dets = [], []

    def free_spot():
        for _ in range(5000):
            x, y = int(rng.integers(30, 610)), int(rng.integers(50, 450))
            if all((x - cx) ** 2 + (y - cy) ** 2 >= (3 * r) ** 2 for cx, cy in centers):
                return x, y
        return None

    def add(cls, conf, spot):
        x, y = spot
        centers.append((x, y))
        dets.append(Detection(cls, conf, (x - r, y - r, x + r, y + r)))
        color = (220, 220, 220) if cls == "tablet" else (60, 160, 230)
        cv2.circle(frame, (x, y), r, color, -1)

    for _ in range(n):
        spot = free_spot()
        if spot:
            add("tablet", float(rng.uniform(0.75, 0.98)), spot)
    if add_wrong:
        spot = free_spot()
        if spot:
            add("capsule", 0.92, spot)
    if add_overlap and dets:
        x, y = centers[0]
        add("tablet", 0.90, (x + 8, y + 8))
    return frame, dets


def show_banner(status, text):
    if status == "ok":
        st.success(text)
    elif status == "wrong":
        st.error(text)
    elif status in ("short", "over", "spread"):
        st.warning(text)
    else:
        st.info(text)


st.title("PillCount: Smart Pill Counter and Fill Checker")

st.sidebar.header("Settings")
mode = st.sidebar.radio("Input", ["Upload photo", "Webcam snapshot", "Demo (simulated pills)"])
target = st.sidebar.number_input("Target pill count", 1, 1000, C.DEFAULT_TARGET)
bottle = st.sidebar.number_input("Bottle size", 1, 1000, C.DEFAULT_BOTTLE)
conf = st.sidebar.slider("Confidence threshold", 0.40, 0.95, C.CONF_SURE, 0.05)

frame, detector, classes = None, None, DEMO_CLASSES

if mode == "Demo (simulated pills)":
    st.caption("Demo mode uses simulated detections to demonstrate the counting rules. "
               "It does not use the trained model.")
    n = st.sidebar.slider("Pills on tray", 0, 50, 30)
    add_wrong = st.sidebar.checkbox("Mix in a wrong pill")
    add_overlap = st.sidebar.checkbox("Make two pills overlap")
    seed = st.sidebar.number_input("Random seed", 0, 999, 1)
    frame, dets = make_demo(int(n), add_wrong, add_overlap, int(seed))
    detector = FakeDetector(dets)
else:
    if not os.path.exists(WEIGHTS):
        st.error("Trained model not found (models/best.pt). Use Demo mode for now, "
                 "or add the trained weights to the models folder.")
        st.stop()
    detector = load_detector(WEIGHTS)
    classes = list(detector.model.names.values())
    if mode == "Upload photo":
        up = st.file_uploader("Upload a photo of pills", type=["jpg", "jpeg", "png"])
        if up is not None:
            frame = cv2.imdecode(np.frombuffer(up.getvalue(), np.uint8), cv2.IMREAD_COLOR)
    else:
        shot = st.camera_input("Take a photo of the tray")
        if shot is not None:
            frame = cv2.imdecode(np.frombuffer(shot.getvalue(), np.uint8), cv2.IMREAD_COLOR)

expected = st.sidebar.selectbox("Expected pill type", classes, index=classes.index(C.DEFAULT_EXPECTED) if C.DEFAULT_EXPECTED in classes else 0)

if frame is None:
    st.info("Provide an image to begin.")
    st.stop()

annotated, r = process_frame(frame, detector, int(target), int(bottle), expected, conf)

left, right = st.columns([3, 2])
with left:
    st.image(annotated, channels="BGR", caption="Detections")
with right:
    show_banner(r["status"], r["banner"])
    total_text = "-" if r["total"] is None else str(r["total"])
    st.markdown("Total count")
    st.markdown(f"<h1 style='font-size:80px;margin:0'>{total_text}</h1>", unsafe_allow_html=True)
    if r["counts"]:
        st.table(pd.DataFrame({"Pill type": list(r["counts"].keys()),
                               "Count": list(r["counts"].values())}))
    if r["full_bottles"] is not None:
        st.write(f"Full bottles of {int(bottle)}: **{r['full_bottles']}**, leftover pills: **{r['leftover']}**")
    if r["unsure"]:
        st.caption(f"{len(r['unsure'])} unsure detection(s) shown in grey and not counted.")

if "history" not in st.session_state:
    st.session_state.history = []

st.subheader("Count history")
if st.button("Add this count to history"):
    st.session_state.history.append({
        "time": datetime.now().strftime("%H:%M:%S"),
        "input": mode,
        "total": r["total"],
        "target": int(target),
        "status": r["banner"],
        "full_bottles": r["full_bottles"],
        "leftover": r["leftover"],
    })
if st.session_state.history:
    hist = pd.DataFrame(st.session_state.history)
    st.dataframe(hist)
    st.download_button("Download history (CSV)", hist.to_csv(index=False),
                       "pillcount_history.csv", "text/csv")
