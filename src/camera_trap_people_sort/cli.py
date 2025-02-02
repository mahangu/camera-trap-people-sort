"""
Command-line interface for sorting camera trap images by number of people detected.
Uses multiple AI models for improved accuracy and uncertainty handling.
"""

import argparse
import shutil
from datetime import datetime
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image
from PytorchWildlife.models.detection.ultralytics_based.megadetectorv6 import MegaDetectorV6
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from ultralytics import YOLO

MODEL_DESCRIPTIONS = {
    "yolo": "YOLOv8x - Fast and accurate general object detection",
    "pose": "YOLOv8x-pose - Specialized in detecting people through pose estimation",
    "megadetector": "MegaDetector V6 - Optimized for camera trap imagery",
    "frcnn": "Faster R-CNN - Highly accurate but slower general object detection",
}


def parse_args(args=None) -> argparse.Namespace:
    """Parse command line arguments for model selection and configuration."""
    parser = argparse.ArgumentParser(
        description="Process images to count people using multiple AI models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Available Models:\n"
        + "\n".join(f"  {k}: {v}" for k, v in MODEL_DESCRIPTIONS.items()),
    )

    parser.add_argument(
        "--input-path",
        type=str,
        default=".",
        help="Path to directory containing images (default: current directory)",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="output",
        help="Path for output directory (default: ./output)",
    )
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Recursively search for images in subdirectories",
    )

    parser.add_argument("--no-yolo", action="store_true", help="Disable YOLOv8x model")
    parser.add_argument("--no-pose", action="store_true", help="Disable YOLOv8x-pose model")
    parser.add_argument("--no-megadetector", action="store_true", help="Disable MegaDetector V6")
    parser.add_argument("--no-frcnn", action="store_true", help="Disable Faster R-CNN")
    parser.add_argument(
        "--yolo-only", action="store_true", help="Use only YOLOv8x (fastest option)"
    )

    parser.add_argument(
        "--confidence",
        type=float,
        default=0.3,
        help="Confidence threshold for detections (default: 0.3)",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.2,
        help="Minimum confidence to consider a detection (default: 0.2)",
    )

    parser.add_argument(
        "--list-models", action="store_true", help="List available models and their descriptions"
    )

    parsed_args = parser.parse_args(args)

    if parsed_args.list_models:
        print("\nAvailable Models:")
        for model, desc in MODEL_DESCRIPTIONS.items():
            print(f"{model}:")
            print(f"  {desc}")
        exit(0)

    if parsed_args.yolo_only:
        parsed_args.use_yolo = True
        parsed_args.use_pose = parsed_args.use_megadetector = parsed_args.use_frcnn = False
    else:
        parsed_args.use_yolo = not parsed_args.no_yolo
        parsed_args.use_pose = not parsed_args.no_pose
        parsed_args.use_megadetector = not parsed_args.no_megadetector
        parsed_args.use_frcnn = not parsed_args.no_frcnn

    if not any(
        [
            parsed_args.use_yolo,
            parsed_args.use_pose,
            parsed_args.use_megadetector,
            parsed_args.use_frcnn,
        ]
    ):
        print("Warning: All models were disabled. Using YOLOv8x as fallback.")
        parsed_args.use_yolo = True

    return parsed_args


def load_models(args: argparse.Namespace) -> dict:
    """Load selected AI models based on command line arguments."""
    models = {}

    if args.use_yolo:
        print("Loading YOLO model...")
        models["yolo"] = YOLO("yolov8x.pt")

    if args.use_pose:
        print("Loading YOLOv8 Pose model...")
        models["pose"] = YOLO("yolov8x-pose.pt")

    if args.use_megadetector:
        print("Loading MegaDetector V6 (YOLOv9-Extra)...")
        models["megadetector"] = MegaDetectorV6(
            pretrained=True, version="MDV6-yolov9-e", device="cpu"
        )

    if args.use_frcnn:
        print("Loading Faster R-CNN...")
        models["frcnn"] = fasterrcnn_resnet50_fpn_v2(pretrained=True)
        models["frcnn"].eval()

    return models


def process_image(
    image_path: Path, models: dict, confidence_threshold: float, min_confidence: float
) -> tuple:
    """Process a single image using selected models to detect people."""
    all_scores = []
    confident_counts = []

    if "yolo" in models:
        yolo_scores = process_image_yolo(models["yolo"], image_path, min_confidence)
        yolo_confident = len([s for s in yolo_scores if s >= confidence_threshold])
        if yolo_confident > 0:
            confident_counts.append(yolo_confident)
        all_scores.extend(("YOLO", score) for score in yolo_scores)

    if "pose" in models:
        pose_scores = process_image_pose(models["pose"], image_path, min_confidence)
        pose_confident = len([s for s in pose_scores if s >= confidence_threshold])
        if pose_confident > 0:
            confident_counts.append(pose_confident)
        all_scores.extend(("YOLOv8-Pose", score) for score in pose_scores)

    if "megadetector" in models:
        mega_scores = process_image_megadetector(models["megadetector"], image_path, min_confidence)
        mega_confident = len([s for s in mega_scores if s >= confidence_threshold])
        if mega_confident > 0:
            confident_counts.append(mega_confident)
        all_scores.extend(("MegaDetector", score) for score in mega_scores)

    if "frcnn" in models:
        frcnn_scores = process_image_frcnn(models["frcnn"], image_path, min_confidence)
        frcnn_confident = len([s for s in frcnn_scores if s >= confidence_threshold])
        if frcnn_confident > 0:
            confident_counts.append(frcnn_confident)
        all_scores.extend(("Faster R-CNN", score) for score in frcnn_scores)

    if not confident_counts:
        # If no model is confident, but they detect something
        if all_scores:
            final_count = max(
                len([s for s in scores if s[1] >= min_confidence]) for model, scores in all_scores
            )
            is_uncertain = True
        else:
            final_count = 0
            is_uncertain = False
    else:
        if len(confident_counts) > 1:
            final_count = sorted(confident_counts)[len(confident_counts) // 2]
        else:
            final_count = confident_counts[0]

        if len(confident_counts) > 1:
            # Count how many models are close to the final count
            models_in_agreement = sum(
                1 for count in confident_counts if abs(count - final_count) <= 1
            )
            is_uncertain = models_in_agreement < len(confident_counts) / 2
        else:
            is_uncertain = False

    return final_count, is_uncertain, all_scores


def process_image_yolo(model: YOLO, image_path: Path, min_confidence: float) -> list:
    """Process image with YOLO to detect humans."""
    results = model(image_path, conf=min_confidence)
    boxes = [box for box in results[0].boxes if box.cls == 0]  # class 0 is person
    return [box.conf.item() for box in boxes]


def process_image_pose(model: YOLO, image_path: Path, min_confidence: float) -> list:
    """Process image with YOLOv8-Pose to detect humans."""
    results = model(image_path, conf=min_confidence)
    return [box.conf.item() for box in results[0].boxes]


def process_image_megadetector(
    model: MegaDetectorV6, image_path: Path, min_confidence: float
) -> list:
    """Process image with MegaDetector to detect humans."""
    with torch.no_grad():
        results = model.single_image_detection(str(image_path), det_conf_thres=min_confidence)
    detections = results["detections"]
    return [
        conf for conf, class_id in zip(detections.confidence, detections.class_id) if class_id == 1
    ]  # class 1 is person in MegaDetector


def process_image_frcnn(model: torch.nn.Module, image_path: Path, min_confidence: float) -> list:
    """Process image with Faster R-CNN to detect humans."""
    image = Image.open(image_path).convert("RGB")
    transform = T.Compose([T.ToTensor()])
    img_tensor = transform(image)

    with torch.no_grad():
        prediction = model([img_tensor])

    return [
        score.item()
        for label, score in zip(prediction[0]["labels"], prediction[0]["scores"])
        if label == 1 and score >= min_confidence
    ]  # class 1 is person in COCO


def ensure_output_dir(
    count: int,
    timestamp: str,
    uncertain: bool = False,
    models: dict = None,
    base_path: str = "output",
) -> Path:
    """Create and return output directory for the given parameters."""
    model_names = "_".join(sorted(models.keys())) if models else "unknown"

    if uncertain:
        output_dir = Path(base_path) / f"{timestamp}_{model_names}/uncertain"
    else:
        output_dir = Path(base_path) / f"{timestamp}_{model_names}/{count}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def main():
    """Main entry point for the CLI application."""
    args = parse_args()
    models = load_models(args)

    input_path = Path(args.input_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_names = "_".join(sorted(models.keys()))

    print("\nActive Models:")
    for model_name in models.keys():
        print(f"- {model_name}: {MODEL_DESCRIPTIONS[model_name]}")

    print("\nConfidence Settings:")
    print(f"- Detection threshold: {args.confidence}")
    print(f"- Minimum confidence: {args.min_confidence}")

    print("\nPaths:")
    print(f"- Input: {input_path.absolute()}")
    print(f"- Output: {Path(args.output_path).absolute()}")
    print(f"- Recursive: {'Yes' if args.recursive else 'No'}")

    print("\nProcessing images...")
    print("-" * 50)

    # Use recursive glob if -r flag is set, otherwise just search current directory
    glob_pattern = "**/*.[jJ][pP][eE]?[gG]" if args.recursive else "*.[jJ][pP][eE]?[gG]"
    for image_path in input_path.glob(glob_pattern):
        try:
            print(f"\nProcessing {image_path.relative_to(input_path)}...")
            count, is_uncertain, scores = process_image(
                image_path, models, args.confidence, args.min_confidence
            )

            if scores:
                print("Detection scores:")
                for model, score in scores:
                    print(f"  {model}: {score:.2f}")

            if is_uncertain:
                print(f"Found {count} people in {image_path.name} (uncertain detection)")
                dest_dir = ensure_output_dir(
                    count, timestamp, uncertain=True, models=models, base_path=args.output_path
                )
                shutil.copy2(image_path, dest_dir / image_path.name)
                print(f"Copied to {dest_dir}/")
            elif count > 0:
                print(f"Found {count} people in {image_path.name}")
                dest_dir = ensure_output_dir(
                    count, timestamp, uncertain=False, models=models, base_path=args.output_path
                )
                shutil.copy2(image_path, dest_dir / image_path.name)
                print(f"Copied to {dest_dir}/")
            else:
                print("No people detected")

        except Exception as e:
            print(f"Error processing {image_path}: {e}")

    print("\nProcessing complete!")
    print(f"Results are in {Path(args.output_path) / f'{timestamp}_{model_names}'}")


if __name__ == "__main__":
    main()
