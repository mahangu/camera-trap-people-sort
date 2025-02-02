# Camera Trap People Sort

A tool to sort camera trap images by the number of people detected in them, using multiple AI models for improved accuracy.

## Features

- Uses multiple state-of-the-art AI models:
  - YOLOv8x (fast general object detection)
  - YOLOv8x-pose (pose-based person detection)
  - MegaDetector V6 (specialized for camera traps)
  - Faster R-CNN (highly accurate object detection)
- Intelligent uncertainty handling
- Configurable confidence thresholds
- Easy to use command-line interface
- Optional recursive directory scanning
- Flexible input/output paths

## Installation

Install from GitHub (latest development version):
```bash
pipx install git+https://github.com/mahangu/camera-trap-people-sort.git
```

Note: If you don't have pipx installed, you can install it with:

**macOS (using Homebrew):**
```bash
brew install pipx
```

**Other platforms:**
```bash
python -m pip install --user pipx
python -m pipx ensurepath
```

## Usage

Basic usage (processes images in current directory):
```bash
camera-trap-people-sort
```

Process images recursively:
```bash
camera-trap-people-sort -r
# or
camera-trap-people-sort --recursive
```

Specify input and output paths:
```bash
camera-trap-people-sort --input-path ~/my-photos --output-path ~/sorted-photos

# With recursive scanning:
camera-trap-people-sort -r --input-path ~/camera-trap-archive --output-path ~/sorted
```

Fast mode (YOLO only):
```bash
camera-trap-people-sort --yolo-only
```

Skip specific models:
```bash
camera-trap-people-sort --no-frcnn  # Skip slower Faster R-CNN
camera-trap-people-sort --no-pose --no-frcnn  # Skip pose and Faster R-CNN
```

Adjust confidence thresholds:
```bash
camera-trap-people-sort --confidence 0.6 --min-confidence 0.2
```

List available models:
```bash
camera-trap-people-sort --list-models
```

## Output Structure

The script organizes images into directories by timestamp and models used:

```
output/
├── 20250202_171655_yolo_megadetector/  # Example with two models
│   ├── 1/  # Images with 1 person
│   ├── 2/  # Images with 2 people
│   └── uncertain/  # Images with uncertain detections
```

## License

This project is licensed under the GNU Affero General Public License v3 (AGPL-3.0). See the [LICENSE](LICENSE) file for details. 