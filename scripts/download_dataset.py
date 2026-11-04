"""Usage: ROBOFLOW_API_KEY=xxx python3 scripts/download_dataset.py <workspace> <project> <version>"""
import os, sys
from roboflow import Roboflow
rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])
rf.workspace(sys.argv[1]).project(sys.argv[2]).version(int(sys.argv[3])).download("yolov8", location="data/pills")
