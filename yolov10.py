import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from models.yolo_v10 import YOLOv10, create_model
from train import YOLODataset, create_transforms, collate_fn

def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45):
    """Non-maximum suppression"""
    # Convert predictions to numpy for easier manipulation
    prediction = prediction.cpu().numpy()
    
    # Get boxes with confidence above threshold
    mask = prediction[..., 4] > conf_thres
    prediction = prediction[mask]
    
    if len(prediction) == 0:
        return []
    
    # Convert to [x1, y1, x2, y2, conf, class] format
    boxes = np.zeros((len(prediction), 6))
    boxes[:, 0] = prediction[:, 0] - prediction[:, 2] / 2  # x1
    boxes[:, 1] = prediction[:, 1] - prediction[:, 3] / 2  # y1
    boxes[:, 2] = prediction[:, 0] + prediction[:, 2] / 2  # x2
    boxes[:, 3] = prediction[:, 1] + prediction[:, 3] / 2  # y2
    boxes[:, 4] = prediction[:, 4]  # confidence
    boxes[:, 5] = prediction[:, 5]  # class
    
    # Perform NMS
    indices = cv2.dnn.NMSBoxes(
        boxes[:, :4].tolist(),
        boxes[:, 4].tolist(),
        conf_thres,
        iou_thres
    )
    
    if len(indices) > 0:
        return boxes[indices.flatten()]
    return []

def compute_iou(box1, box2):
    """Compute IoU between two boxes"""
    # Box format: [x1, y1, x2, y2]
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # Calculate intersection area
    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
    
    # Calculate union area
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - intersection_area
    
    # Calculate IoU
    iou = intersection_area / union_area if union_area > 0 else 0
    
    return iou

def load_model_weights(model, weights_path):
    """Load weights from a .pt file, handling various formats"""
    print(f"Loading weights from {weights_path}")
    try:
        # First attempt - direct load for our own format
        state_dict = torch.load(weights_path, map_location='cpu')
        
        # Check if it's a checkpoint format
        if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        
        # Check if it's an ultralytics format (all keys start with 'model.')
        if isinstance(state_dict, dict) and all(k.startswith('model.') for k in state_dict.keys()):
            # Remove 'model.' prefix
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('model.'):
                    new_key = k.replace('model.', '')
                    new_state_dict[new_key] = v
            state_dict = new_state_dict
            
        # Try to load state dict, first with strict=True
        try:
            model.load_state_dict(state_dict)
            print("Successfully loaded weights with strict matching")
        except Exception as e:
            print(f"Warning: Could not load with strict matching: {e}")
            print("Trying non-strict loading...")
            # Try with strict=False as fallback
            model.load_state_dict(state_dict, strict=False)
            print("Successfully loaded weights with non-strict matching")
            
    except Exception as e:
        print(f"Warning: Failed to load weights: {e}")
        print("Using randomly initialized weights")
    
    return model

def detect_objects(model, image_path, device, img_size=640, conf_thres=0.25, iou_thres=0.45, class_names=None):
    """Detect objects in an image"""
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    original_height, original_width = image.shape[:2]
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Apply transformations
    transform = create_transforms(img_size, is_training=False)
    transformed = transform(image=image_rgb, bboxes=[], class_labels=[])
    image_tensor = transformed['image'].unsqueeze(0).to(device)
    
    # Inference
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
    
    # Apply NMS
    detections = non_max_suppression(outputs[0][0], conf_thres, iou_thres)
    
    # Convert detections to original image scale
    results = []
    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        
        # Convert normalized coordinates to pixel coordinates
        x1 = int(x1 * original_width)
        y1 = int(y1 * original_height)
        x2 = int(x2 * original_width)
        y2 = int(y2 * original_height)
        
        class_id = int(cls)
        class_name = class_names[class_id] if class_names and class_id < len(class_names) else f"Class {class_id}"
        
        results.append({
            'bbox': [x1, y1, x2, y2],
            'confidence': conf,
            'class_id': class_id,
            'class_name': class_name
        })
    
    return results, image_rgb

def visualize_detections(image, detections, output_path=None):
    """Visualize detections on the image"""
    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    
    # Define colors for different classes
    colors = np.random.randint(0, 255, size=(80, 3), dtype=np.uint8)
    
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        conf = det['confidence']
        class_id = det['class_id']
        class_name = det['class_name']
        
        # Create rectangle
        rect = Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor=colors[class_id % 80].tolist(), facecolor='none'
        )
        plt.gca().add_patch(rect)
        
        # Add label
        plt.text(
            x1, y1 - 10, f"{class_name} {conf:.2f}",
            color=colors[class_id % 80].tolist(),
            fontsize=12, bbox=dict(facecolor='white', alpha=0.7)
        )
    
    plt.axis('off')
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        print(f"Saved visualization to {output_path}")
    else:
        plt.show()

def train(args):
    """Train YOLOv10 model"""
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = create_model(args.variant, args.num_classes)
    model = model.to(device)
    
    # Load pre-trained weights if specified
    if args.weights:
        if os.path.exists(args.weights):
            model = load_model_weights(model, args.weights)
        else:
            print(f"Warning: Weights file {args.weights} not found. Training from scratch.")
    
    # Create data transforms
    train_transform = create_transforms(args.img_size, is_training=True)
    val_transform = create_transforms(args.img_size, is_training=False)
    
    try:
        # Create datasets
        train_dataset = YOLODataset(
            data_dir=args.data_dir,
            split='train',
            img_size=args.img_size,
            transform=train_transform
        )
        
        val_dataset = YOLODataset(
            data_dir=args.data_dir,
            split='val',
            img_size=args.img_size,
            transform=val_transform
        )
        
        # Create dataloaders
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=collate_fn
        )
        
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=collate_fn
        )
    except Exception as e:
        print(f"Error creating datasets or dataloaders: {e}")
        print("Make sure your data directory has the correct structure:")
        print("data_dir/")
        print("  ├── images/")
        print("  │   ├── train/")
        print("  │   └── val/")
        print("  └── labels/")
        print("      ├── train/")
        print("      └── val/")
        return
    
    # Create optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=10, verbose=True
    )
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1} started")

        # Train epoch
        model.train()
        total_loss = 0.0
        
        for i, (imgs, targets) in enumerate(train_dataloader):
            print(f"Batch {i} started using total images {len(train_dataloader)}")

            imgs = imgs.to(device)
            
            # Move targets to device if they are tensors
            targets = [t.to(device) if isinstance(t, torch.Tensor) else t for t in targets]
            
            # Forward pass
            outputs = model(imgs)
            
            # Simple loss for demonstration
            loss = nn.MSELoss()(outputs[0][0].reshape(-1), torch.zeros(outputs[0][0].reshape(-1).shape).to(device))
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # if i % 10 == 0:
            print(f"Epoch {epoch+1}, Batch {i}/{len(train_dataloader)}, Loss: {loss.item():.4f}")
        
        avg_train_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1} completed. Average train loss: {avg_train_loss:.4f}")
        
        # Validate
        model.eval()
        total_val_loss = 0.0
        
        with torch.no_grad():
            for imgs, targets in val_dataloader:
                imgs = imgs.to(device)
                
                # Forward pass
                outputs = model(imgs)
                
                # Calculate loss
                loss = nn.MSELoss()(outputs[0][0].reshape(-1), torch.zeros(outputs[0][0].reshape(-1).shape).to(device))
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_dataloader)
        print(f"Validation loss: {avg_val_loss:.4f}")
        
        # Update learning rate
        scheduler.step(avg_val_loss)
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss
        }
        
        torch.save(checkpoint, os.path.join(args.output_dir, f'checkpoint_epoch_{epoch}.pt'))
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(checkpoint, os.path.join(args.output_dir, 'best_model.pt'))
            print(f"Saved best model with validation loss: {avg_val_loss:.4f}")
    
    print("Training complete!")

def validate(args):
    """Validate YOLOv10 model"""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = create_model(args.variant, args.num_classes)
    model = model.to(device)
    
    # Load weights
    if args.weights and os.path.exists(args.weights):
        model = load_model_weights(model, args.weights)
    else:
        print(f"Error: Weights file {args.weights} not found.")
        return
    
    # Create dataset
    val_transform = create_transforms(args.img_size, is_training=False)
    val_dataset = YOLODataset(
        data_dir=args.data_dir,
        split='val',
        img_size=args.img_size,
        transform=val_transform
    )
    
    # Create dataloader
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    # Validation
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for imgs, targets in val_dataloader:
            imgs = imgs.to(device)
            
            # Forward pass
            outputs = model(imgs)
            
            # Calculate loss
            loss = nn.MSELoss()(outputs[0][0].reshape(-1), torch.zeros(outputs[0][0].reshape(-1).shape).to(device))
            total_loss += loss.item()
    
    avg_loss = total_loss / len(val_dataloader)
    print(f"Validation loss: {avg_loss:.4f}")

def detect(args):
    """Detect objects in an image"""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = create_model(args.variant, args.num_classes)
    model = model.to(device)
    
    # Load weights
    if args.weights and os.path.exists(args.weights):
        model = load_model_weights(model, args.weights)
    else:
        print(f"Error: Weights file {args.weights} not found.")
        return
    
    # Load class names if provided
    class_names = None
    if args.class_names and os.path.exists(args.class_names):
        with open(args.class_names, 'r') as f:
            class_names = [line.strip() for line in f.readlines()]
    
    # Detect objects
    try:
        detections, image = detect_objects(
            model, args.image, device, args.img_size, args.conf_thres, args.iou_thres, class_names
        )
        
        # Print detections
        print(f"Found {len(detections)} objects:")
        for i, det in enumerate(detections):
            print(f"{i+1}. {det['class_name']} (Confidence: {det['confidence']:.2f})")
        
        # Visualize detections
        visualize_detections(image, detections, args.output)
    except Exception as e:
        print(f"Error during detection: {e}")
        print("Please make sure the image exists and is a valid image file.")

def parse_arguments():
    parser = argparse.ArgumentParser(description='YOLOv10 - Train, Validate, Detect')
    
    # Create parent parser with common arguments
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument('--variant', type=str, default='n', choices=['n', 's', 'm', 'l', 'x'], 
                              help='Model variant (default: %(default)s)')
    parent_parser.add_argument('--weights', type=str, default=None, 
                              help='Path to model weights (default: %(default)s)')
    parent_parser.add_argument('--img-size', type=int, default=640, 
                              help='Input image size (default: %(default)s)')
    parent_parser.add_argument('--num-classes', type=int, default=80, 
                              help='Number of classes (default: %(default)s)')
    parent_parser.add_argument('--data-dir', type=str, default='data', 
                              help='Data directory (default: %(default)s)')
    parent_parser.add_argument('--output-dir', type=str, default='output', 
                              help='Output directory (default: %(default)s)')
    
    # Create subparsers for each mode
    subparsers = parser.add_subparsers(dest='mode', required=True,
                                       help='Mode: train, val, detect')
    
    # Create train parser
    train_parser = subparsers.add_parser('train', parents=[parent_parser],
                                        help='Train YOLOv10 model')
    train_parser.add_argument('--batch-size', type=int, default=16, 
                             help='Batch size (default: %(default)s)')
    train_parser.add_argument('--epochs', type=int, default=100, 
                             help='Number of epochs (default: %(default)s)')
    train_parser.add_argument('--lr', type=float, default=0.001, 
                             help='Learning rate (default: %(default)s)')
    train_parser.add_argument('--weight-decay', type=float, default=0.0005, 
                             help='Weight decay (default: %(default)s)')
    train_parser.add_argument('--num-workers', type=int, default=4, 
                             help='Number of dataloader workers (default: %(default)s)')
    
    # Create val parser
    val_parser = subparsers.add_parser('val', parents=[parent_parser],
                                      help='Validate YOLOv10 model')
    val_parser.add_argument('--batch-size', type=int, default=16, 
                           help='Batch size (default: %(default)s)')
    val_parser.add_argument('--num-workers', type=int, default=4, 
                           help='Number of dataloader workers (default: %(default)s)')
    
    # Create detect parser
    detect_parser = subparsers.add_parser('detect', parents=[parent_parser],
                                         help='Detect objects in an image')
    detect_parser.add_argument('--image', type=str, required=True, 
                              help='Path to input image')
    detect_parser.add_argument('--conf-thres', type=float, default=0.25, 
                              help='Confidence threshold (default: %(default)s)')
    detect_parser.add_argument('--iou-thres', type=float, default=0.45, 
                              help='IoU threshold (default: %(default)s)')
    detect_parser.add_argument('--output', type=str, default=None, 
                              help='Path to output visualization')
    detect_parser.add_argument('--class-names', type=str, default=None, 
                              help='Path to class names file')
    
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    if args.mode == 'train':
        train(args)
    elif args.mode == 'val':
        validate(args)
    elif args.mode == 'detect':
        detect(args)

if __name__ == '__main__':
    main() 