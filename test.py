# script.py

import subprocess
import os
import glob

# Check NVIDIA GPU
try:
    subprocess.run(["nvidia-smi"], check=True)
except Exception as e:
    print("Could not run nvidia-smi:", e)

# Import YOLO and ultralytics
try:
    import ultralytics
    from ultralytics import YOLO
    ultralytics.checks()
except ImportError:
    print("Ultralytics not installed. Please run: pip install ultralytics")

# Import Roboflow and download dataset
try:
    from roboflow import Roboflow

    rf = Roboflow(api_key="bhm43Q95N2cTZnIdRNqV")
    project = rf.workspace(
        "ayush-wattal-ritu-bhamrah-ritanjali-jena-sai-kumar-sannidhi"
    ).project("building-image")
    version = project.version(18)
    dataset = version.download("yolov11")

except ImportError:
    print("Roboflow not installed. Please run: pip install roboflow")

# Train YOLO model
try:
    # Build the YOLO train command
    train_command = [
        "yolo",
        "task=detect",
        "mode=train",
        f"data={dataset.location}/data.yaml",
        "model=yolo11n.pt",
        "epochs=100",
        "imgsz=640",
    ]

    subprocess.run(train_command, check=True)

except Exception as e:
    print("Error running YOLO training:", e)

# Optional: work with Google Drive (only needed if using Colab)
try:
    from google.colab import drive
    drive.mount('/content/drive')
except ImportError:
    print("Google Colab not detected. Skipping drive mount.")

# Optional: list all images in training folder
train_images = glob.glob(os.path.join(dataset.location, "train", "*.*"))
print(f"Found {len(train_images)} training images.")
