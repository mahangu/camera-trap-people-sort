from pathlib import Path
import shutil
from datetime import datetime
import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO
from PytorchWildlife.models.detection.ultralytics_based.megadetectorv6 import MegaDetectorV6
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
import torchvision.transforms as T
import argparse

# Model descriptions for help text
MODEL_DESCRIPTIONS = {
    'yolo': 'YOLOv8x - Fast and accurate general object detection',
    'pose': 'YOLOv8x-pose - Specialized in detecting people through pose estimation',
    'megadetector': 'MegaDetector V6 - Optimized for camera trap imagery',
    'frcnn': 'Faster R-CNN - Highly accurate but slower general object detection'
}

def parse_args():
    parser = argparse.ArgumentParser(
        description='Process images to count people using multiple AI models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='Available Models:\n' + '\n'.join(f'  {k}: {v}' for k, v in MODEL_DESCRIPTIONS.items())
    )
    
    # Model selection arguments - now they disable models instead of enabling
    parser.add_argument('--no-yolo', action='store_true', help='Disable YOLOv8x model')
    parser.add_argument('--no-pose', action='store_true', help='Disable YOLOv8x-pose model')
    parser.add_argument('--no-megadetector', action='store_true', help='Disable MegaDetector V6')
    parser.add_argument('--no-frcnn', action='store_true', help='Disable Faster R-CNN')
    parser.add_argument('--yolo-only', action='store_true', help='Use only YOLOv8x (fastest option)')
    
    # Configuration arguments
    parser.add_argument('--confidence', type=float, default=0.3,
                      help='Confidence threshold for detections (default: 0.3)')
    parser.add_argument('--min-confidence', type=float, default=0.2,
                      help='Minimum confidence to consider a detection (default: 0.2)')
    
    # List models option
    parser.add_argument('--list-models', action='store_true',
                      help='List available models and their descriptions')
    
    args = parser.parse_args()
    
    # If --list-models is specified, print model descriptions and exit
    if args.list_models:
        print("\nAvailable Models:")
        for model, desc in MODEL_DESCRIPTIONS.items():
            print(f"{model}:")
            print(f"  {desc}")
        exit(0)
    
    # Convert the disable flags to use flags
    if args.yolo_only:
        args.use_yolo = True
        args.use_pose = False
        args.use_megadetector = False
        args.use_frcnn = False
    else:
        args.use_yolo = not args.no_yolo
        args.use_pose = not args.no_pose
        args.use_megadetector = not args.no_megadetector
        args.use_frcnn = not args.no_frcnn
    
    # If all models were disabled, warn and use YOLO as fallback
    if not any([args.use_yolo, args.use_pose, args.use_megadetector, args.use_frcnn]):
        print("Warning: All models were disabled. Using YOLOv8x as fallback.")
        args.use_yolo = True
    
    return args

# Load models based on CLI arguments
def load_models(args):
    models = {}
    
    if args.use_yolo:
        print("Loading YOLO model...")
        models['yolo'] = YOLO('yolov8x.pt')
    
    if args.use_pose:
        print("Loading YOLOv8 Pose model...")
        models['pose'] = YOLO('yolov8x-pose.pt')
    
    if args.use_megadetector:
        print("Loading MegaDetector V6 (YOLOv9-Extra)...")
        models['megadetector'] = MegaDetectorV6(pretrained=True, version='MDV6-yolov9-e', device='cpu')
    
    if args.use_frcnn:
        print("Loading Faster R-CNN...")
        models['frcnn'] = fasterrcnn_resnet50_fpn_v2(pretrained=True)
        models['frcnn'].eval()
    
    return models

def process_image(image_path, models, confidence_threshold, min_confidence):
    """Process a single image using selected models."""
    all_scores = []
    confident_counts = []
    
    # Process with each selected model
    if 'yolo' in models:
        yolo_scores = process_image_yolo(models['yolo'], image_path, min_confidence)
        yolo_confident = len([s for s in yolo_scores if s >= confidence_threshold])
        if yolo_confident > 0:
            confident_counts.append(yolo_confident)
        for score in yolo_scores:
            all_scores.append(('YOLO', score))
    
    if 'pose' in models:
        pose_scores = process_image_pose(models['pose'], image_path, min_confidence)
        pose_confident = len([s for s in pose_scores if s >= confidence_threshold])
        if pose_confident > 0:
            confident_counts.append(pose_confident)
        for score in pose_scores:
            all_scores.append(('YOLOv8-Pose', score))
    
    if 'megadetector' in models:
        mega_scores = process_image_megadetector(models['megadetector'], image_path, min_confidence)
        mega_confident = len([s for s in mega_scores if s >= confidence_threshold])
        if mega_confident > 0:
            confident_counts.append(mega_confident)
        for score in mega_scores:
            all_scores.append(('MegaDetector', score))
    
    if 'frcnn' in models:
        frcnn_scores = process_image_frcnn(models['frcnn'], image_path, min_confidence)
        frcnn_confident = len([s for s in frcnn_scores if s >= confidence_threshold])
        if frcnn_confident > 0:
            confident_counts.append(frcnn_confident)
        for score in frcnn_scores:
            all_scores.append(('Faster R-CNN', score))
    
    # Determine final count and uncertainty
    if not confident_counts:
        # If no model is confident, but they detect something
        if all_scores:
            final_count = max(len([s for s in scores if s[1] >= min_confidence]) 
                            for model, scores in all_scores)
            is_uncertain = True
        else:
            final_count = 0
            is_uncertain = False
    else:
        # If at least one model is confident
        # Use median count when we have multiple confident detections
        if len(confident_counts) > 1:
            final_count = sorted(confident_counts)[len(confident_counts)//2]
        else:
            final_count = confident_counts[0]

        # Only mark as uncertain if majority of models disagree significantly
        if len(confident_counts) > 1:
            # Count how many models are close to the final count
            models_in_agreement = sum(1 for count in confident_counts if abs(count - final_count) <= 1)
            # Mark as uncertain if less than half of the models agree
            is_uncertain = models_in_agreement < len(confident_counts) / 2
        else:
            is_uncertain = False
    
    return final_count, is_uncertain, all_scores

def process_image_yolo(model, image_path, min_confidence):
    """Process image with YOLO to detect humans."""
    results = model(image_path, conf=min_confidence)
    boxes = [box for box in results[0].boxes if box.cls == 0]  # class 0 is person
    scores = [box.conf.item() for box in boxes]
    return scores

def process_image_pose(model, image_path, min_confidence):
    """Process image with YOLOv8-Pose to detect humans."""
    results = model(image_path, conf=min_confidence)
    scores = [box.conf.item() for box in results[0].boxes]
    return scores

def process_image_megadetector(model, image_path, min_confidence):
    """Process image with MegaDetector to detect humans."""
    with torch.no_grad():
        results = model.single_image_detection(str(image_path), det_conf_thres=min_confidence)
    detections = results['detections']
    person_scores = [conf for conf, class_id in zip(detections.confidence, detections.class_id) 
                    if class_id == 1]
    return person_scores

def process_image_frcnn(model, image_path, min_confidence):
    """Process image with Faster R-CNN to detect humans."""
    image = Image.open(image_path).convert('RGB')
    transform = T.Compose([T.ToTensor()])
    img_tensor = transform(image)
    
    with torch.no_grad():
        prediction = model([img_tensor])
    
    scores = []
    for pred in prediction:
        for label, score in zip(pred['labels'], pred['scores']):
            if label == 1 and score >= min_confidence:  # 1 is person in COCO
                scores.append(score.item())
    return scores

def ensure_output_dir(count, timestamp, uncertain=False, models=None):
    """Create output directory for specific count if it doesn't exist."""
    # Create model name string with underscores
    model_names = '_'.join(sorted(models.keys())) if models else 'unknown'
    
    if uncertain:
        output_dir = Path(f'output/{timestamp}_{model_names}/uncertain')
    else:
        output_dir = Path(f'output/{timestamp}_{model_names}/{count}')
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def main():
    args = parse_args()
    models = load_models(args)
    
    input_dir = Path('input')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_names = '_'.join(sorted(models.keys()))  # Using underscore here too
    
    print(f"\nActive Models:")
    for model_name in models.keys():
        print(f"- {model_name}: {MODEL_DESCRIPTIONS[model_name]}")
    
    print(f"\nConfidence Settings:")
    print(f"- Detection threshold: {args.confidence}")
    print(f"- Minimum confidence: {args.min_confidence}")
    
    print("\nProcessing images...")
    print("-" * 50)
    
    # Process each image
    for image_path in input_dir.glob('*.jp*g'):
        try:
            print(f"\nProcessing {image_path.name}...")
            count, is_uncertain, scores = process_image(
                image_path, models, args.confidence, args.min_confidence)
            
            # Print confidence scores
            if scores:
                print("Detection scores:")
                for model, score in scores:
                    print(f"  {model}: {score:.2f}")
            
            if is_uncertain:
                print(f"Found {count} people in {image_path.name} (uncertain detection)")
                dest_dir = ensure_output_dir(count, timestamp, uncertain=True, models=models)
                shutil.copy2(image_path, dest_dir / image_path.name)
                print(f"Copied to {dest_dir}/")
            elif count > 0:
                print(f"Found {count} people in {image_path.name}")
                dest_dir = ensure_output_dir(count, timestamp, uncertain=False, models=models)
                shutil.copy2(image_path, dest_dir / image_path.name)
                print(f"Copied to {dest_dir}/")
            else:
                print("No people detected")
                
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
    
    print("\nProcessing complete!")
    print(f"Results are in the output/{timestamp}_{model_names}/ directory")

if __name__ == "__main__":
    main() 
