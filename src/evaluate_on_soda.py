import argparse
import os
import shutil
from ultralytics import YOLO, YOLOv10
import torch

def evaluate_model(model_path, data_yaml, img_size, batch_size, conf_threshold, iou_threshold, device, save_json, max_det, half, dnn, plots, rect, split, model_version):
    """
    Evaluate a trained YOLO model on a given dataset.

    Args:
        model_path (str): Path to the trained YOLO model file.
        data_yaml (str): Path to the data YAML file.
        img_size (int): Image size for validation.
        batch_size (int): Batch size for validation.
        conf_threshold (float): Confidence threshold for predictions.
        iou_threshold (float): IoU threshold for evaluation.
        device (str): Device to use for evaluation.
        save_json (bool): Whether to save results in JSON format.
        max_det (int): Maximum number of detections per image.
        half (bool): Whether to use half precision.
        dnn (bool): Whether to use OpenCV DNN for inference.
        plots (bool): Whether to plot validation results.
        rect (bool): Whether to use rectangular training.
        split (str): Dataset split to use for validation.
        model_version (int): YOLO model version to use (e.g., 8, 10).
    """
    if model_version < 10:
        model = YOLO(model_path)  # load the model for YOLOv8 and below
    else:
        model = YOLOv10(model_path)  # load the model for YOLOv10

    # Validate the device
    if device.lower() != "cpu" and torch.cuda.device_count() <= int(device):
        raise ValueError(f"Invalid CUDA device index: {device}")

    # Set the device
    device_str = f"cuda:{device}" if device.lower() != "cpu" else "cpu"
    if device_str != "cpu":
        torch.cuda.set_device(int(device))

    # Evaluate the model
    results = model.val(
        data=data_yaml,
        imgsz=img_size,
        batch=batch_size,
        conf=conf_threshold,
        iou=iou_threshold,
        device=device_str,
        save_json=save_json,
        max_det=max_det,
        half=half,
        dnn=dnn,
        plots=plots,
        rect=rect,
        split=split
    )
    
    return results

def rename_output_folder():
    """
    Rename the default output folder 'runs/detect' to 'runs/soda'.
    """
    src = 'runs/detect'
    dst = 'runs/soda'
    if os.path.exists(src):
        shutil.move(src, dst)
        print(f"Renamed '{src}' to '{dst}'")
    else:
        print(f"Source folder '{src}' does not exist")

def parse_arguments():
    """
    Parse command-line arguments for model evaluation.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Evaluate a trained YOLO model on the SODA dataset.")
    parser.add_argument('--model', type=str, required=True, help="Path to the trained YOLO model file.")
    parser.add_argument('--data', type=str, required=True, help="Path to the data YAML file.")
    parser.add_argument('--imgsz', type=int, default=420, help="Image size for validation.")
    parser.add_argument('--batch', type=int, default=16, help="Batch size for validation.")
    parser.add_argument('--conf', type=float, default=0.3, help="Confidence threshold for predictions.")
    parser.add_argument('--iou', type=float, default=0.9, help="IoU threshold for evaluation.")
    parser.add_argument('--device', type=str, default="0", help="Device to use for evaluation.")
    parser.add_argument('--save_json', action='store_true', help="Save results in JSON format.")
    parser.add_argument('--max_det', type=int, default=20, help="Maximum number of detections per image.")
    parser.add_argument('--half', action='store_true', help="Use half precision.")
    parser.add_argument('--dnn', action='store_true', help="Use OpenCV DNN for inference.")
    parser.add_argument('--plots', action='store_true', help="Plot validation results.")
    parser.add_argument('--rect', action='store_true', help="Use rectangular training.")
    parser.add_argument('--split', type=str, default="test", help="Dataset split to use for validation.")
    parser.add_argument('--model_version', type=int, default=10, help="YOLO model version to use (e.g., 8, 10).")

    return parser.parse_args()

def main():
    """
    Main function to evaluate the YOLO model on the SODA dataset.
    """
    args = parse_arguments()
    results = evaluate_model(
        model_path=args.model,
        data_yaml=args.data,
        img_size=args.imgsz,
        batch_size=args.batch,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        device=args.device,
        save_json=args.save_json,
        max_det=args.max_det,
        half=args.half,
        dnn=args.dnn,
        plots=args.plots,
        rect=args.rect,
        split=args.split,
        model_version=args.model_version
    )
    
    print("Evaluation results:", results)
    
    # Rename the output folder
    rename_output_folder()

if __name__ == "__main__":
    main()
