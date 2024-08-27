# src/train_object_detection.py

import argparse
from ultralytics import YOLO, YOLOv10

def train_model(data_path, epochs, img_size, device, batch_size, workers, project, model_version):
    """
    Train a YOLO model with the specified parameters.

    Args:
        data_path (str): Path to the data YAML file.
        epochs (int): Number of training epochs.
        img_size (int): Image size for training.
        device (list): List of GPU devices to use.
        batch_size (int): Batch size for training.
        workers (int): Number of data loading workers.
        project (str): Directory to save training results.
        model_version (int): YOLO model version to use (e.g., 8, 10).
    """
    if model_version < 10:
        model = YOLO(f"yolov{model_version}x.pt")  # load a pretrained model (recommended for training)
    else:
        model = YOLOv10.from_pretrained('jameslahm/yolov10x')

    # Train the model
    results = model.train(
        data=data_path,
        epochs=epochs,
        imgsz=img_size,
        device=device,
        batch=batch_size,
        workers=workers,
        project=project,
    )
    return results

def parse_arguments():
    """
    Parse command-line arguments for training the YOLO model.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Train a YOLO object detection model.")
    parser.add_argument('--data', type=str, default="data/source_data.yml", help="Path to the data YAML file.")
    parser.add_argument('--epochs', type=int, default=30, help="Number of training epochs.")
    parser.add_argument('--imgsz', type=int, default=720, help="Image size for training.")
    parser.add_argument('--device', type=int, nargs='+', default=[0, 1, 2, 3], help="List of GPU devices to use.")
    parser.add_argument('--batch', type=int, default=-1, help="Batch size for training.")
    parser.add_argument('--workers', type=int, default=16, help="Number of data loading workers.")
    parser.add_argument('--project', type=str, default="runs/train", help="Directory to save training results.")
    parser.add_argument('--model_version', type=int, default=10, help="YOLO model version to use (e.g., 8, 10).")

    return parser.parse_args()

def main():
    """
    Main function to train the YOLO model based on command-line arguments.
    """
    args = parse_arguments()
    train_model(
        data_path=args.data,
        epochs=args.epochs,
        img_size=args.imgsz,
        device=args.device,
        batch_size=args.batch,
        workers=args.workers,
        project=args.project,
        model_version=args.model_version
    )

if __name__ == "__main__":
    main()
