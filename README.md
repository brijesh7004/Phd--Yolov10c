# YOLOv10 Implementation

This repository contains a custom implementation of the YOLOv10 object detection model without relying on the ultralytics library. It supports training, validation, and inference with various model variants.

## Features

- Custom YOLOv10 architecture implementation
- Support for different model variants (N, S, M, L, X)
- Training with data augmentation
- Validation and evaluation
- Model checkpointing and best model saving
- Non-maximum suppression for inference
- Object detection in images
- Support for loading weights from various formats

## Requirements

Install the required packages:

```bash
pip install -r requirements.txt
```

## Project Structure

```
.
├── data/
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   └── labels/
│       ├── train/
│       └── val/
├── models/
│   ├── __init__.py
│   └── yolo_v10.py
├── yolov10.py
├── train.py
├── requirements.txt
└── README.md
```

## Data Format

The data should be organized in the YOLO format:

```
data/
├── images/
│   ├── train/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── val/
│       ├── image1.jpg
│       ├── image2.jpg
│       └── ...
└── labels/
    ├── train/
    │   ├── image1.txt
    │   ├── image2.txt
    │   └── ...
    └── val/
        ├── image1.txt
        ├── image2.txt
        └── ...
```

Each label file should contain one annotation per line in the format:
```
class_id x_center y_center width height
```
where all values are normalized to be between 0 and 1.

## Available Models

YOLOv10 comes in several variants:
- YOLOv10-n (nano): Smallest, fastest model
- YOLOv10-s (small): Balance of speed and accuracy
- YOLOv10-m (medium): More accurate, medium speed
- YOLOv10-l (large): Higher accuracy, slower
- YOLOv10-x (xlarge): Most accurate, slowest

## Usage - Unified Script

Use the `yolov10.py` script to train, validate, or detect with a single command. The mode (`train`, `val`, or `detect`) must be the first argument.

### Training

```bash
python yolov10.py train --data-dir data --variant n --epochs 100 --batch-size 16 --lr 0.001
```

### Validation

```bash
python yolov10.py val --data-dir data --variant n --weights output/best_model.pt
```

### Detection

```bash
python yolov10.py detect --image path/to/image.jpg --variant n --weights output/best_model.pt
```

## Training Script

You can also use the `train.py` script directly for more control over the training process:

```bash
python train.py --data-dir data --variant n --epochs 100
```

## Common Parameters

- `--variant`: Model size (n, s, m, l, x)
- `--weights`: Path to pre-trained weights
- `--img-size`: Input image size (default: 640)
- `--num-classes`: Number of classes (default: 80)
- `--data-dir`: Data directory
- `--output-dir`: Output directory

## Training Parameters

- `--batch-size`: Batch size (default: 16)
- `--epochs`: Number of epochs (default: 100)
- `--lr`: Learning rate (default: 0.001)
- `--weight-decay`: Weight decay (default: 0.0005)
- `--num-workers`: Number of dataloader workers (default: 4)

## Detection Parameters

- `--conf-thres`: Confidence threshold (default: 0.25)
- `--iou-thres`: IoU threshold (default: 0.45)
- `--output`: Path to save the output visualization
- `--class-names`: Path to a file containing class names

## Weight Format

The implementation supports loading weights from different formats:
- Custom YOLOv10 weights
- Checkpoint weights (with model_state_dict)
- Weights from ultralytics models (converted automatically)

## Troubleshooting

1. **Bounding Box Errors**: The implementation now automatically fixes invalid bounding boxes by clamping coordinates to the valid range [0, 1].

2. **Command Syntax**: When using the unified script (`yolov10.py`), ensure the mode argument (`train`, `val`, or `detect`) is the first argument.

3. **Weight Loading Issues**: If weights fail to load with strict matching, the implementation will automatically try non-strict loading.

4. **CUDA Out of Memory**: Reduce the batch size or image size if you encounter memory issues when using a GPU.

5. **Image Loading Errors**: If an image cannot be loaded, the system will print a warning and continue with other images.

## Model Variants

- N (Nano): Smallest model, fastest inference
- S (Small): Good balance of speed and accuracy
- M (Medium): Better accuracy than S
- L (Large): High accuracy
- X (XLarge): Highest accuracy, slowest inference

## License

This project is licensed under the MIT License - see the LICENSE file for details. 