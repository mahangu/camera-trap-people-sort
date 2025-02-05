"""
Command-line interface for sorting camera trap images by number of people detected.
Uses multiple AI models for improved accuracy and uncertainty handling.
"""

import argparse
import logging
import shutil
import sys
import warnings
from datetime import datetime
from multiprocessing import Pool, cpu_count
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image
from PytorchWildlife.models.detection.ultralytics_based.megadetectorv6 import MegaDetectorV6
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from tqdm.auto import tqdm
from ultralytics import YOLO

# Filter out specific warnings
warnings.filterwarnings(
    "ignore", category=UserWarning, message=".*parameter .pretrained. is deprecated.*"
)
warnings.filterwarnings(
    "ignore", category=UserWarning, message=".*Arguments other than a weight enum.*"
)

MODEL_DESCRIPTIONS = {
    "yolo": "YOLOv8x - Fast and accurate general object detection",
    "pose": "YOLOv8x-pose - Specialized in detecting people through pose estimation",
    "megadetector": "MegaDetector V6 - Optimized for camera trap imagery",
    "frcnn": "Faster R-CNN - Highly accurate but slower general object detection",
}

# Global variables to store models in each process
PROCESS_MODELS = {}


def init_worker(model_config):
    """Initialize worker process with models."""
    global PROCESS_MODELS

    # Suppress logging during model loading
    logging.getLogger("ultralytics").setLevel(logging.WARNING)
    sys.stdout = open("/dev/null", "w")  # Suppress stdout during model loading

    # Determine device
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    try:
        if model_config.get("use_yolo"):
            PROCESS_MODELS["yolo"] = YOLO("yolov8x.pt")
            if device == "mps":
                PROCESS_MODELS["yolo"].to(device)

        if model_config.get("use_pose"):
            PROCESS_MODELS["pose"] = YOLO("yolov8x-pose.pt")
            if device == "mps":
                PROCESS_MODELS["pose"].to(device)

        if model_config.get("use_megadetector"):
            PROCESS_MODELS["megadetector"] = MegaDetectorV6(
                pretrained=True, version="MDV6-yolov9-e", device=device
            )

        if model_config.get("use_frcnn"):
            PROCESS_MODELS["frcnn"] = fasterrcnn_resnet50_fpn_v2(pretrained=True)
            if device == "mps":
                PROCESS_MODELS["frcnn"].to(device)
            PROCESS_MODELS["frcnn"].eval()

    finally:
        # Restore stdout
        sys.stdout = sys.__stdout__


def process_single_image(args):
    """Process a single image using pre-loaded models."""
    image_path, confidence_threshold, min_confidence, timestamp, base_path = args
    try:
        count, is_uncertain, scores = process_image(
            image_path, PROCESS_MODELS, confidence_threshold, min_confidence
        )

        # Handle file copying within the process
        if is_uncertain:
            dest_dir = ensure_output_dir(
                count, timestamp, uncertain=True, models=PROCESS_MODELS, base_path=base_path
            )
        elif count > 0:
            dest_dir = ensure_output_dir(
                count, timestamp, uncertain=False, models=PROCESS_MODELS, base_path=base_path
            )
        else:
            dest_dir = ensure_output_dir(
                "no_people",
                timestamp,
                uncertain=False,
                models=PROCESS_MODELS,
                base_path=base_path,
            )

        shutil.copy2(image_path, dest_dir / image_path.name)
        return True, image_path, count, is_uncertain, None

    except Exception as e:
        return False, image_path, None, None, str(e)


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
    # Suppress ultralytics logging
    logging.getLogger("ultralytics").setLevel(logging.WARNING)

    args = parse_args()

    # Create a lightweight config dict instead of loading models
    model_config = {
        "use_yolo": args.use_yolo,
        "use_pose": args.use_pose,
        "use_megadetector": args.use_megadetector,
        "use_frcnn": args.use_frcnn,
    }

    # Get list of selected models for display
    selected_models = [k for k, v in model_config.items() if v]
    model_names = "_".join(m.replace("use_", "") for m in selected_models)

    input_path = Path(args.input_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("\nActive Models:")
    for model_name in [m.replace("use_", "") for m in selected_models]:
        print(f"- {model_name}: {MODEL_DESCRIPTIONS[model_name]}")

    print("\nConfidence Settings:")
    print(f"- Detection threshold: {args.confidence}")
    print(f"- Minimum confidence: {args.min_confidence}")

    print("\nPaths:")
    print(f"- Input: {input_path.absolute()}")
    print(f"- Output: {Path(args.output_path).absolute()}")
    print(f"- Recursive: {'Yes' if args.recursive else 'No'}")

    # Convert input path to absolute Path object and resolve any special characters
    input_dir = Path(args.input_path).resolve()
    if not input_dir.exists():
        print(f"Error: Input directory '{input_dir}' does not exist")
        return
    if not input_dir.is_dir():
        print(f"Error: '{input_dir}' is not a directory")
        return

    image_files = []
    extensions = ["*.jpg", "*.jpeg", "*.JPG", "*.JPEG"]

    print(f"Searching for images in: {input_dir}")

    if args.recursive:
        for ext in extensions:
            try:
                found = list(input_dir.rglob(ext))
                if found:
                    print(f"\nFound {len(found)} files with pattern {ext}")
                image_files.extend(found)
            except Exception as e:
                print(f"Error searching for {ext}: {e}")
    else:
        for ext in extensions:
            try:
                found = list(input_dir.glob(ext))
                if found:
                    print(f"\nFound {len(found)} files with pattern {ext}")
                image_files.extend(found)
            except Exception as e:
                print(f"Error searching for {ext}: {e}")

    if not image_files:
        print(f"\nNo JPEG images found in {input_dir}")
        print("Note: Supported extensions are: .jpg, .jpeg, .JPG, .JPEG")
        print("\nTrying alternative search method...")
        try:
            all_files = list(input_dir.rglob("*") if args.recursive else input_dir.glob("*"))
            jpeg_files = [
                f for f in all_files if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg"]
            ]
            if jpeg_files:
                print(f"\nFound {len(jpeg_files)} JPEG files using alternative method")
                image_files = jpeg_files
            else:
                print("Still no JPEG files found.")
                return
        except Exception as e:
            print(f"Error in alternative search: {e}")
            return

    total_images = len(image_files)
    print(f"\nFound {total_images} total images to process")

    # Determine optimal number of processes (leave 1 core free for system)
    num_processes = max(1, cpu_count() - 1)
    print(f"\nProcessing images using {num_processes} parallel processes...")

    # Prepare arguments for parallel processing (without model_config as it's handled in init)
    process_args = [
        (image_path, args.confidence, args.min_confidence, timestamp, args.output_path)
        for image_path in image_files
    ]

    # Initialize the process pool with models
    with Pool(num_processes, initializer=init_worker, initargs=(model_config,)) as pool:
        # Use imap_unordered for potentially better performance
        results = list(
            tqdm(
                pool.imap_unordered(process_single_image, process_args),
                total=total_images,
                desc="Processing images",
                unit="img",
                ncols=80,
                position=0,
                leave=True,
                dynamic_ncols=False,  # Fixed width
                ascii=True,  # Use ASCII characters for better compatibility
                mininterval=0.5,  # Update interval in seconds
                bar_format=(
                    "{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
                ),
            )
        )

    # Report results
    successful = 0
    failed = 0
    uncertain_count = 0
    people_counts = {}

    print("\nProcessing Summary:")
    for success, image_path, count, is_uncertain, error in results:
        if success:
            successful += 1
            if is_uncertain:
                uncertain_count += 1
            elif count == "no_people":
                people_counts["no_people"] = people_counts.get("no_people", 0) + 1
            else:
                people_counts[count] = people_counts.get(count, 0) + 1
        else:
            failed += 1
            print(f"Error processing {image_path}: {error}")

    print(f"\nSuccessfully processed: {successful} images")
    print(f"Uncertain detections: {uncertain_count}")
    print("\nPeople count distribution:")
    if "no_people" in people_counts:
        print(f"- No people: {people_counts['no_people']}")
    for count in sorted(k for k in people_counts.keys() if k != "no_people"):
        print(f"- {count} people: {people_counts[count]}")

    if failed > 0:
        print(f"\nFailed to process: {failed} images")

    print(f"\nResults are in {Path(args.output_path) / f'{timestamp}_{model_names}'}")


if __name__ == "__main__":
    main()
