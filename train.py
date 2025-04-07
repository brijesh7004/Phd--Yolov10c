import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import json
from models.yolo_v10 import YOLOv10
import argparse
from pathlib import Path

class YOLODataset(Dataset):
    def __init__(self, data_dir, split='train', img_size=640, transform=None):
        self.data_dir = Path(data_dir)
        self.img_dir = self.data_dir / 'images' / split
        self.label_dir = self.data_dir / 'labels' / split
        self.img_size = img_size
        self.transform = transform
        
        # Get all image files
        self.img_files = list(self.img_dir.glob('*.jpg')) + list(self.img_dir.glob('*.png'))
        self.img_files.sort()
        
        print(f"Found {len(self.img_files)} images in {self.img_dir}")
        
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.img_files[idx]
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Warning: Could not read image {img_path}, using empty image")
            img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Load labels (YOLO format: class_id, x_center, y_center, width, height)
        label_path = self.label_dir / f"{img_path.stem}.txt"
        bboxes = []
        class_labels = []
        
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    try:
                        data = line.strip().split()
                        if len(data) == 5:
                            class_id = int(data[0])
                            x_center = float(data[1])
                            y_center = float(data[2])
                            width = float(data[3])
                            height = float(data[4])
                            
                            # Strict validation and clamping to ensure values are within [0, 1]
                            if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 < width <= 1 and 0 < height <= 1):
                                x_center = max(0, min(x_center, 1.0))
                                y_center = max(0, min(y_center, 1.0))
                                width = max(0.001, min(width, 1.0))
                                height = max(0.001, min(height, 1.0))
                                
                                # Ensure box doesn't go out of bounds
                                if x_center - width/2 < 0:
                                    width = 2 * x_center
                                if x_center + width/2 > 1:
                                    width = 2 * (1 - x_center)
                                if y_center - height/2 < 0:
                                    height = 2 * y_center
                                if y_center + height/2 > 1:
                                    height = 2 * (1 - y_center)
                                
                                print(f"Warning: Fixed bbox in {label_path}: {line.strip()} to [{x_center}, {y_center}, {width}, {height}]")
                            
                            # Store as [x_center, y_center, width, height]
                            bboxes.append([x_center, y_center, width, height])
                            class_labels.append(class_id)
                    except Exception as e:
                        print(f"Error parsing line in {label_path}: {line.strip()} - {e}")
                        continue
        
        # Apply transformations
        if self.transform and len(bboxes) > 0:
            try:
                transformed = self.transform(image=img, bboxes=bboxes, class_labels=class_labels)
                img = transformed['image']
                transformed_bboxes = transformed['bboxes']
                transformed_class_labels = transformed['class_labels']
                
                # Create target tensor with class_id appended
                if len(transformed_bboxes) > 0:
                    targets = torch.tensor([list(box) + [cls] for box, cls in zip(transformed_bboxes, transformed_class_labels)])
                else:
                    targets = torch.zeros((0, 5))
                    
            except Exception as e:
                print(f"Error applying transform: {e}, using empty targets")
                img = self.transform(image=img)['image']
                targets = torch.zeros((0, 5))
        else:
            if self.transform:
                img = self.transform(image=img, bboxes=[], class_labels=[])['image']
            else:
                img = torch.from_numpy(img.transpose(2, 0, 1)) / 255.0
            
            # Create a target tensor with class_id appended
            if len(bboxes) > 0:
                targets = torch.tensor([box + [cls] for box, cls in zip(bboxes, class_labels)])
            else:
                targets = torch.zeros((0, 5))
                
        return img, targets

def collate_fn(batch):
    """Custom collate function to handle variable-sized targets"""
    imgs, targets = zip(*batch)
    # Stack images
    imgs = torch.stack(imgs)
    # Return targets as list since they can have different shapes
    return imgs, targets

def create_transforms(img_size, is_training=True):
    """Create transformations for data augmentation"""
    if is_training:
        transform = A.Compose([
            A.Resize(height=img_size, width=img_size),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
    else:
        transform = A.Compose([
            A.Resize(height=img_size, width=img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
    
    return transform

def compute_loss(outputs, targets, device):
    """Simple YOLOv10 loss function"""
    # For simplicity, use MSE for detection (position) and BCE for classification
    mse_loss = nn.MSELoss()
    bce_loss = nn.BCEWithLogitsLoss()
    
    # Extract outputs
    pred_boxes = outputs[0][0][..., :4]  # [batch, grid, grid, anchors, 4]
    pred_conf = outputs[0][0][..., 4]    # [batch, grid, grid, anchors, 1]
    pred_cls = outputs[0][0][..., 5:]    # [batch, grid, grid, anchors, num_classes]
    
    # Placeholder for loss calculation
    # This is a simplified version and should be replaced with a proper YOLOv10 loss
    loss_box = torch.tensor(0.0).to(device)
    loss_conf = torch.tensor(0.0).to(device)
    loss_cls = torch.tensor(0.0).to(device)
    
    # Loss calculation example
    if len(targets) > 0:
        # Compute box regression loss
        loss_box = mse_loss(pred_boxes.flatten(), torch.zeros_like(pred_boxes.flatten()))
        
        # Compute confidence loss
        loss_conf = bce_loss(pred_conf.flatten(), torch.zeros_like(pred_conf.flatten()))
        
        # Compute classification loss
        loss_cls = bce_loss(pred_cls.flatten(), torch.zeros_like(pred_cls.flatten()))
    
    # Combine losses
    loss = loss_box + loss_conf + loss_cls
    
    return loss, {
        'box_loss': loss_box.item(),
        'conf_loss': loss_conf.item(),
        'cls_loss': loss_cls.item(),
        'total_loss': loss.item()
    }

def train_epoch(model, dataloader, optimizer, device, epoch):
    model.train()
    total_loss = 0.0
    
    for i, (imgs, targets) in enumerate(dataloader):
        imgs = imgs.to(device)
        
        # Move targets to device if they are tensors
        targets = [t.to(device) if isinstance(t, torch.Tensor) else t for t in targets]
        
        # Forward pass
        outputs = model(imgs)
        
        # Calculate loss
        loss, loss_components = compute_loss(outputs, targets, device)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if i % 10 == 0:
            print(f"Epoch {epoch}, Batch {i}/{len(dataloader)}, Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch} completed. Average loss: {avg_loss:.4f}")
    
    return avg_loss

def validate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for imgs, targets in dataloader:
            imgs = imgs.to(device)
            
            # Move targets to device if they are tensors
            targets = [t.to(device) if isinstance(t, torch.Tensor) else t for t in targets]
            
            # Forward pass
            outputs = model(imgs)
            
            # Calculate loss
            loss, _ = compute_loss(outputs, targets, device)
            total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    print(f"Validation loss: {avg_loss:.4f}")
    
    return avg_loss

def main():
    parser = argparse.ArgumentParser(description='YOLOv10 Training')
    parser.add_argument('--data-dir', type=str, required=True, help='Data directory containing images and labels folders')
    parser.add_argument('--output-dir', type=str, default='output', help='Output directory')
    parser.add_argument('--variant', type=str, default='n', choices=['n', 's', 'm', 'l', 'x'], help='Model variant')
    parser.add_argument('--img-size', type=int, default=640, help='Input image size')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.0005, help='Weight decay')
    parser.add_argument('--num-classes', type=int, default=80, help='Number of classes')
    parser.add_argument('--weights', type=str, default=None, help='Path to pre-trained weights')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = YOLOv10(variant=args.variant, num_classes=args.num_classes)
    model = model.to(device)
    
    # Load pre-trained weights if specified
    if args.weights:
        if os.path.exists(args.weights):
            state_dict = torch.load(args.weights, map_location=device)
            
            if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
                # Checkpoint format
                model.load_state_dict(state_dict['model_state_dict'])
            else:
                # Try direct load
                try:
                    model.load_state_dict(state_dict)
                except:
                    # Try non-strict loading as a fallback
                    model.load_state_dict(state_dict, strict=False)
                    print("Loaded weights with non-strict matching")
                    
            print(f"Loaded weights from {args.weights}")
        else:
            print(f"Warning: Weights file {args.weights} not found. Training from scratch.")
    
    # Create data transforms
    train_transform = create_transforms(args.img_size, is_training=True)
    val_transform = create_transforms(args.img_size, is_training=False)
    
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
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    # Create optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=10, verbose=True
    )
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        # Train
        train_loss = train_epoch(model, train_dataloader, optimizer, device, epoch)
        
        # Validate
        val_loss = validate(model, val_dataloader, device)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss
        }
        
        torch.save(checkpoint, os.path.join(args.output_dir, f'checkpoint_epoch_{epoch}.pt'))
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(checkpoint, os.path.join(args.output_dir, 'best_model.pt'))
            print(f"Saved best model with validation loss: {val_loss:.4f}")
    
    print("Training complete!")

if __name__ == '__main__':
    main() 