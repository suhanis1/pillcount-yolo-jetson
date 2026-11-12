# PillCount: Pill Counter and Fill Checker (YOLOv8n on NVIDIA Jetson)

DATA 690 course project. A camera detects every pill on a tray, counts them, and checks the count against a target amount.

## Docker image
`nvcr.io/nvidia/l4t-ml:r36.2.0-py3` (NVIDIA JetPack 6 / L4T ML image with PyTorch 2.1.0, CUDA and OpenCV)

## Repository structure
- `backend/`: core logic
  - `config.py`: thresholds and defaults
  - `detector.py`: YOLO wrapper
  - `decisions.py`: Rules 1-3, confidence handling, overlap check, count stabilizer
  - `pipeline.py`: `process_frame()`, the entry point the frontend will call
- `scripts/`: `download_dataset.py`, `train.py`, `run_image.py`, `run_webcam.py`, `evaluate.py`
- `models/`: trained weights (`best.pt`), `sample_images/`: test photos, `outputs/`: result images
- `requirements.txt`: Python packages

## Setup on the Jetson
```bash
git clone https://github.com/suhanis1/pillcount-yolo-jetson.git
cd pillcount-yolo-jetson
sudo docker run -it --rm --runtime nvidia --network host \
  --device /dev/video0 \
  -v $(pwd):/workspace/pillcount-yolo-jetson \
  -w /workspace/pillcount-yolo-jetson \
  nvcr.io/nvidia/l4t-ml:r36.2.0-py3 bash
```
Inside the container (installed without dependencies so the GPU build of PyTorch is kept):
```bash
pip install --no-deps ultralytics ultralytics-thop
pip install pyyaml tqdm scipy requests pandas psutil py-cpuinfo
```

## Run
```bash
python3 scripts/run_image.py sample_images/test1.jpg --target 30 --bottle 30 --expected tablet
python3 scripts/run_webcam.py --target 30 --bottle 30 --expected tablet
python3 scripts/evaluate.py labels.csv
```

## Decision rules
1. Target count: total equal to target shows "Count correct"; lower shows "Short by X"; higher shows "Over by X".
2. Wrong pill: a detection of an unexpected type shows "Wrong pill detected".
3. Bottle batching: reports the number of full bottles that can be filled and the leftover pills.

Uncertain detections:
- Confidence below 40% is ignored; 40-60% is drawn grey as "unsure" and not counted.
- Overlapping boxes or one oversized box shows "Spread pills out" instead of a count.
- A count is accepted only after staying unchanged for about 1 second; no detections shows "No pills detected".

## Training
`scripts/download_dataset.py` fetches the Roboflow dataset and `scripts/train.py` fine-tunes YOLOv8n. Trained weights go in `models/best.pt`.

## Tested so far (inside the Docker container on the Jetson)
- PyTorch 2.1.0 with CUDA available: True
- Ultralytics and OpenCV imports work
- Decision logic (Rules 1-3, wrong-pill alert) returns the expected results on synthetic detections
- USB webcam capture works (640x480); sample frame in `outputs/webcam_test.jpg`

## Status
Backend code is complete and tested on the Jetson. The Streamlit frontend is complete (see below). Model training on the pill dataset is in progress.

## Notes
- `run_image.py` and `run_webcam.py` require trained weights at `models/best.pt` and, for images, a photo in `sample_images/`.
- `evaluate.py` expects a CSV with the columns `image,true_count`.
- `requirements.txt` lists all packages for a normal Python environment. Inside the Jetson Docker image, PyTorch and OpenCV are already installed, so only the `pip install` lines above are needed.

## Frontend (Streamlit)
Run on the MacBook (or on the Jetson inside the Docker container):

    pip install streamlit
    streamlit run frontend/app.py

Open http://localhost:8501. Features: input choice (uploaded photo, webcam snapshot, or demo mode), target count, bottle size, expected pill type, confidence slider, annotated image, per-type count table, total count, status banner (correct / short / over / wrong pill / spread pills out), and a count history that can be downloaded as CSV. Screenshots are in `outputs/frontend_screenshots/`.

Demo mode uses simulated detections so the decision rules can be shown before the trained model (`models/best.pt`) is added. Upload and webcam modes use the trained model once it is present.

## Example results (Upload photo mode, confidence threshold 0.60, expected type `normal`)
- Photo 1: 10 round pills on a plain cloth. The app counted 6; 2 more were shown as "unsure" and 2 were not detected.
- Photo 2: 10 round pills plus 1 oval pill. The app counted 10 ("Count correct", 1 full bottle of 10, 0 leftover); the oval pill was drawn in grey as "unsure" and not counted.

Results vary with angle and lighting between photos, which is why the planned evaluation uses 20 piles with known counts and fine-tuning with my own labeled photos.
