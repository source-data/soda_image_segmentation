import shutil
import os
from ultralytics import YOLOv10

# Load the model
model = YOLOv10("/app/runs/soda_figure_segmentation/train/weights/best.pt")

# Validate the model
metrics = model.val(
    data="data/imageCLEF_data.yml",
    imgsz=512,
    batch=32,
    conf=0.8,
    iou=0.3,
    # device=device_str,
    save_json=True,
    max_det=20,
    half=True,
    dnn=True,
    plots=True,
    rect=True,
    split="val"
    )

# Define the source and destination directories
source_dir = "runs/detect"
destination_dir = "runs/imageCLEF"

# Check if the source directory exists
if os.path.exists(source_dir):
    # Rename the directory
    shutil.move(source_dir, destination_dir)
    print(f"Renamed '{source_dir}' to '{destination_dir}'")
else:
    print(f"The directory '{source_dir}' does not exist")

# Print metrics
print(metrics)
