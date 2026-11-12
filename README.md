# PillCount: Pill Counter and Fill Checker

DATA 690 computer vision project. PillCount uses a camera to find the pills on a tray, counts them, and checks the count against a target amount. It uses a YOLOv8n model that I fine-tuned on a pill dataset, a Python backend for the counting rules, and a Streamlit page for the user interface. The project is built for the NVIDIA Jetson.

## Docker image

I use `nvcr.io/nvidia/l4t-ml:r36.2.0-py3` on the Jetson. It comes with PyTorch 2.1.0, CUDA and OpenCV already installed.

## What is in this repo

- `backend/` has the main logic.
  - `config.py` holds the thresholds and default settings.
  - `detector.py` wraps the YOLO model.
  - `decisions.py` has the three rules, the confidence handling, the overlap check and the count stabilizer.
  - `pipeline.py` has `process_frame()`, which the frontend calls.
- `frontend/app.py` is the Streamlit page.
- `scripts/` has `download_dataset.py`, `train.py`, `run_image.py`, `run_webcam.py` and `evaluate.py`.
- `models/best.pt` is the trained model.
- `outputs/` has the webcam test frame and the frontend screenshots.
- `requirements.txt` lists the Python packages.

## How the counting works

For every image the backend runs the model, drops weak detections, counts the pills of each type and then applies three rules:

1. **Target count:** if the total matches the target it shows "Count correct", otherwise "Short by X" or "Over by X".
2. **Wrong pill:** if a detection is not the expected pill type, it shows "Wrong pill detected" and the box turns red.
3. **Bottle batching:** it reports how many full bottles can be filled from the pills on the tray and how many are left over.

To handle uncertain detections:
- Detections below 40% confidence are ignored.
- Detections between 40% and 60% are drawn in grey as "unsure" and are not counted.
- If boxes overlap a lot, or one box is much bigger than the rest, it shows "Spread pills out" instead of a count.
- If nothing is found it says "No pills detected" instead of 0.
- A count is only accepted after it stays the same for about 1 second (used in the webcam script).

## Model

I fine-tuned YOLOv8n for 40 epochs on the Roboflow Universe "PILL DETECTION" dataset, which has 2,350 images and 5 classes (damaged, empty, faulty, normal, pill). I trained it on a Google Colab T4 GPU and saved the weights as `models/best.pt`.

Validation results on 352 images:

| Class | Precision | Recall | mAP50 |
|---|---|---|---|
| all | 0.897 | 0.909 | 0.918 |
| pill | 0.869 | 0.913 | 0.915 |
| normal | 0.887 | 0.954 | 0.936 |
| faulty | 0.905 | 0.911 | 0.928 |
| damaged | 0.862 | 0.788 | 0.833 |
| empty | 0.964 | 0.978 | 0.979 |

The overall mAP50-95 is 0.781, and inference took about 3.6 ms per image on the T4.

## Running the frontend

On my MacBook:

    python3 -m venv .venv
    source .venv/bin/activate
    pip install streamlit opencv-python numpy pandas ultralytics
    streamlit run frontend/app.py

The page opens at http://localhost:8501. In the sidebar I can choose the input (uploaded photo, webcam snapshot or demo mode), the target count, the bottle size, the expected pill type and the confidence threshold. The page shows the image with boxes, a table with the count per pill type, the total in large text, a status banner and a count history that can be downloaded as a CSV file. Demo mode uses simulated detections, so the rules can be shown without a photo. Screenshots are in `outputs/frontend_screenshots/`.

## Running the backend on the Jetson

    git clone https://github.com/suhanis1/pillcount-yolo-jetson.git
    cd pillcount-yolo-jetson
    sudo docker run -it --rm --runtime nvidia --network host \
      --device /dev/video0 \
      -v $(pwd):/workspace/pillcount-yolo-jetson \
      -w /workspace/pillcount-yolo-jetson \
      nvcr.io/nvidia/l4t-ml:r36.2.0-py3 bash

Inside the container I install the packages without dependencies so the GPU build of PyTorch is not replaced:

    pip install --no-deps ultralytics ultralytics-thop
    pip install pyyaml tqdm scipy requests pandas psutil py-cpuinfo

Then I can run the scripts:

    python3 scripts/run_image.py sample_images/test1.jpg --target 30 --bottle 30 --expected normal
    python3 scripts/run_webcam.py --target 30 --bottle 30 --expected normal
    python3 scripts/evaluate.py labels.csv

`evaluate.py` expects a CSV file with the columns `image,true_count`.

## What I have tested so far

- On the Jetson, inside the Docker container: PyTorch sees the GPU, the Ultralytics and OpenCV imports work, the decision rules give the right messages on test detections, and the USB webcam captures a 640x480 frame (`outputs/webcam_test.jpg`).
- On the MacBook: the Streamlit page works with uploaded photos and with demo mode.

Two example photos (confidence threshold 0.60, expected type `normal`):
- Photo 1 had 10 round pills. The app counted 6, showed 2 as unsure and missed 2.
- Photo 2 had 10 round pills and 1 oval pill. The app counted 10 ("Count correct", 1 full bottle of 10, 0 leftover) and showed the oval pill in grey as unsure.

The results change with the angle and the lighting, so the count is not always exact yet.

## Limitations and next steps

- The dataset classes are quality-inspection labels (normal, faulty, damaged, empty, pill), not tablet and capsule types. The counter currently counts every detected class, so the wrong-pill rule only works for these classes.
- Next I want to take my own photos with the Jetson webcam, label them in Roboflow and fine-tune the model again.
- I also still need to run the frontend on the Jetson and test 20 piles with known counts (exact-count rate and mean absolute error).
