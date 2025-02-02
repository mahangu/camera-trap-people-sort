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

# Load models
print("Loading YOLO model...")
yolo_model = YOLO('yolov8x.pt')  # Using XLarge model for maximum accuracy

print("Loading YOLOv8 Pose model...")
pose_model = YOLO('yolov8x-pose.pt')  # Using XLarge pose model

print("Loading MegaDetector V6 (YOLOv9-Extra)...")
megadetector = MegaDetectorV6(pretrained=True, version='MDV6-yolov9-e', device='cpu')

print("Loading Faster R-CNN...")
frcnn_model = fasterrcnn_resnet50_fpn_v2(pretrained=True)
frcnn_model.eval()

# Configuration
CONFIDENCE_THRESHOLD = 0.8  # More reasonable threshold for both models
MIN_CONFIDENCE = 0.1  # Minimum confidence to even consider a detection

def ensure_output_dir(count, timestamp, uncertain=False):
    """Create output directory for specific count if it doesn't exist."""
    if uncertain:
        output_dir = Path(f'output/{timestamp}/uncertain')
    else:
        output_dir = Path(f'output/{timestamp}/{count}')
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def process_image_yolo(image_path):
    """Process image with YOLO to detect humans."""
    results = yolo_model(image_path, conf=MIN_CONFIDENCE)
    boxes = [box for box in results[0].boxes if box.cls == 0]  # class 0 is person
    scores = [box.conf.item() for box in boxes]
    return scores

def process_image_pose(image_path):
    """Process image with YOLOv8-Pose to detect humans."""
    results = pose_model(image_path, conf=MIN_CONFIDENCE)
    scores = [box.conf.item() for box in results[0].boxes]
    return scores

def process_image_megadetector(image_path):
    """Process image with MegaDetector to detect humans."""
    # Run detection
    with torch.no_grad():
        results = megadetector.single_image_detection(str(image_path), det_conf_thres=MIN_CONFIDENCE)
    
    # Extract person detections (class 1 is person in MegaDetector)
    detections = results['detections']
    person_scores = [conf for conf, class_id in zip(detections.confidence, detections.class_id) if class_id == 1]
    return person_scores

def process_image_frcnn(image_path):
    """Process image with Faster R-CNN to detect humans."""
    # Load and transform image
    image = Image.open(image_path).convert('RGB')
    transform = T.Compose([T.ToTensor()])
    img_tensor = transform(image)
    
    # Run detection
    with torch.no_grad():
        prediction = frcnn_model([img_tensor])
    
    # Extract person detections (class 1 is person in COCO)
    scores = []
    for pred in prediction:
        for label, score in zip(pred['labels'], pred['scores']):
            if label == 1 and score >= MIN_CONFIDENCE:  # 1 is person in COCO
                scores.append(score.item())
    return scores

def process_image(image_path):
    """Process a single image using all four models."""
    # Get detections from all models
    yolo_scores = process_image_yolo(image_path)
    mega_scores = process_image_megadetector(image_path)
    pose_scores = process_image_pose(image_path)
    frcnn_scores = process_image_frcnn(image_path)
    
    # Combine detections
    all_scores = []
    
    # Count confident detections from each model
    yolo_confident = len([s for s in yolo_scores if s >= CONFIDENCE_THRESHOLD])
    mega_confident = len([s for s in mega_scores if s >= CONFIDENCE_THRESHOLD])
    pose_confident = len([s for s in pose_scores if s >= CONFIDENCE_THRESHOLD])
    frcnn_confident = len([s for s in frcnn_scores if s >= CONFIDENCE_THRESHOLD])
    
    # Add scores to output
    for score in yolo_scores:
        all_scores.append(('YOLO', score))
    for score in mega_scores:
        all_scores.append(('MegaDetector', score))
    for score in pose_scores:
        all_scores.append(('YOLOv8-Pose', score))
    for score in frcnn_scores:
        all_scores.append(('Faster R-CNN', score))
    
    # Determine final count and uncertainty
    confident_counts = [count for count in [yolo_confident, mega_confident, pose_confident, frcnn_confident] if count > 0]
    
    if not confident_counts:
        # If no model is confident, but they detect something
        if yolo_scores or mega_scores or pose_scores or frcnn_scores:
            final_count = max(len(yolo_scores), len(mega_scores), len(pose_scores), len(frcnn_scores))
            is_uncertain = True
        else:
            final_count = 0
            is_uncertain = False
    else:
        # If at least one model is confident
        final_count = max(confident_counts)
        # Mark as uncertain if models disagree significantly
        max_diff = max((abs(a - b) for i, a in enumerate(confident_counts) for b in confident_counts[i + 1:]), default=0)
        is_uncertain = max_diff > 1
    
    return final_count, is_uncertain, all_scores

def main():
    input_dir = Path('input')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    print(f"Using models with confidence threshold: {CONFIDENCE_THRESHOLD}")
    print(f"Minimum detection confidence: {MIN_CONFIDENCE}")
    print("Processing images...")
    print("-" * 50)
    
    # Process each image
    for image_path in input_dir.glob('*.jp*g'):  # handles both .jpg and .jpeg
        try:
            print(f"\nProcessing {image_path.name}...")
            count, is_uncertain, scores = process_image(image_path)
            
            # Print confidence scores
            if scores:
                print("Detection scores:")
                for model, score in scores:
                    print(f"  {model}: {score:.2f}")
            
            if is_uncertain:
                print(f"Found {count} people in {image_path.name} (uncertain detection)")
                dest_dir = ensure_output_dir(count, timestamp, uncertain=True)
                shutil.copy2(image_path, dest_dir / image_path.name)
                print(f"Copied to output/{timestamp}/uncertain/")
            elif count > 0:
                print(f"Found {count} people in {image_path.name}")
                dest_dir = ensure_output_dir(count, timestamp)
                shutil.copy2(image_path, dest_dir / image_path.name)
                print(f"Copied to output/{timestamp}/{count}/")
            else:
                print("No people detected")
                
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
    
    print("\nProcessing complete!")
    print(f"Results are in the output/{timestamp}/ directory")

if __name__ == "__main__":
    main() 