"""
Custom Training Script for HMAY-TSF Model
Trains the model without using YOLO's training loop
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
import numpy as np
from datetime import datetime
import argparse
import json
import pandas as pd
import csv
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import cv2
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR, ReduceLROnPlateau
import torch.amp as amp
import math
from pathlib import Path
from tqdm import tqdm

# Import our custom model
from hmay_tsf_model import UltraOptimizedHMAY_TSF

class CustomDataset:
    """Custom dataset for training without YOLO"""
    
    def __init__(self, data_yaml_path, img_size=640, is_training=True):
        self.img_size = img_size
        self.is_training = is_training
        
        # Load dataset configuration
        with open(data_yaml_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Get paths
        self.data_path = Path(self.config['path'])
        self.train_path = self.data_path / self.config['train']
        self.val_path = self.data_path / self.config['val']
        self.class_names = self.config['names']
        self.num_classes = self.config['nc']
        
        # Get image and label paths
        if is_training:
            self.img_dir = self.data_path / 'images' / 'train'
            self.label_dir = self.data_path / 'labels' / 'train'
        else:
            self.img_dir = self.data_path / 'images' / 'val'
            self.label_dir = self.data_path / 'labels' / 'val'
        
        # Get all image files
        self.img_files = list(self.img_dir.glob('*.jpg')) + list(self.img_dir.glob('*.png'))
        print(f"Found {len(self.img_files)} images in {self.img_dir}")
    
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        label_path = self.label_dir / (img_path.stem + '.txt')
        
        # Load image
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize image
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)  # HWC to CHW
        
        # Load labels
        labels = []
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        labels.append([class_id, x_center, y_center, width, height])
        
        labels = torch.tensor(labels, dtype=torch.float32) if labels else torch.zeros((0, 5))
        
        return img, labels

def collate_fn(batch):
    """Custom collate function for batching"""
    imgs, labels = zip(*batch)
    imgs = torch.stack(imgs)
    return imgs, labels

class CustomLoss(nn.Module):
    """Custom loss function for object detection"""
    
    def __init__(self, num_classes=4):
        super().__init__()
        self.num_classes = num_classes
        self.box_loss_weight = 0.05
        self.cls_loss_weight = 0.5
        self.obj_loss_weight = 1.0
        
    def forward(self, predictions, targets):
        """
        predictions: [batch, 3, H, W, 5+num_classes]
        targets: list of [num_objects, 5] tensors
        """
        total_loss = 0
        box_loss = 0
        cls_loss = 0
        obj_loss = 0
        
        batch_size = predictions.size(0)
        
        for i in range(batch_size):
            pred = predictions[i]  # [3, H, W, 5+num_classes]
            target = targets[i]    # [num_objects, 5]
            
            if len(target) == 0:
                # No objects in this image
                continue
            
            # Reshape predictions
            B, C, H, W, D = pred.shape
            pred = pred.view(-1, D)  # [3*H*W, 5+num_classes]
            
            # Extract components
            pred_xy = pred[:, :2]    # [3*H*W, 2]
            pred_wh = pred[:, 2:4]   # [3*H*W, 2]
            pred_obj = pred[:, 4]    # [3*H*W]
            pred_cls = pred[:, 5:]   # [3*H*W, num_classes]
            
            # Create target tensors
            target_xy = target[:, 1:3]  # [num_objects, 2]
            target_wh = target[:, 3:5]  # [num_objects, 2]
            target_cls = target[:, 0].long()  # [num_objects]
            
            # Calculate IoU for positive samples
            iou_scores = self.calculate_iou(pred_xy, pred_wh, target_xy, target_wh)
            positive_mask = iou_scores > 0.5
            
            if positive_mask.sum() > 0:
                # Box loss (MSE for positive samples)
                box_loss += F.mse_loss(pred_xy[positive_mask], target_xy.repeat(positive_mask.sum() // len(target_xy), 1))
                
                # Classification loss
                target_cls_expanded = target_cls.repeat(positive_mask.sum() // len(target_cls))
                cls_loss += F.cross_entropy(pred_cls[positive_mask], target_cls_expanded)
                
                # Objectness loss
                obj_loss += F.binary_cross_entropy_with_logits(pred_obj[positive_mask], torch.ones_like(pred_obj[positive_mask]))
            
            # Negative samples
            negative_mask = iou_scores < 0.3
            if negative_mask.sum() > 0:
                obj_loss += F.binary_cross_entropy_with_logits(pred_obj[negative_mask], torch.zeros_like(pred_obj[negative_mask]))
        
        # Average losses
        if batch_size > 0:
            box_loss /= batch_size
            cls_loss /= batch_size
            obj_loss /= batch_size
        
        total_loss = self.box_loss_weight * box_loss + self.cls_loss_weight * cls_loss + self.obj_loss_weight * obj_loss
        
        return total_loss, box_loss, cls_loss, obj_loss
    
    def calculate_iou(self, pred_xy, pred_wh, target_xy, target_wh):
        """Calculate IoU between predictions and targets"""
        # Convert to corner format
        pred_x1 = pred_xy[:, 0] - pred_wh[:, 0] / 2
        pred_y1 = pred_xy[:, 1] - pred_wh[:, 1] / 2
        pred_x2 = pred_xy[:, 0] + pred_wh[:, 0] / 2
        pred_y2 = pred_xy[:, 1] + pred_wh[:, 1] / 2
        
        target_x1 = target_xy[:, 0] - target_wh[:, 0] / 2
        target_y1 = target_xy[:, 1] - target_wh[:, 1] / 2
        target_x2 = target_xy[:, 0] + target_wh[:, 0] / 2
        target_y2 = target_xy[:, 1] + target_wh[:, 1] / 2
        
        # Calculate intersection
        x1 = torch.max(pred_x1.unsqueeze(1), target_x1.unsqueeze(0))
        y1 = torch.max(pred_y1.unsqueeze(1), target_y1.unsqueeze(0))
        x2 = torch.min(pred_x2.unsqueeze(1), target_x2.unsqueeze(0))
        y2 = torch.min(pred_y2.unsqueeze(1), target_y2.unsqueeze(0))
        
        intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
        
        # Calculate union
        pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
        target_area = (target_x2 - target_x1) * (target_y2 - target_y1)
        
        pred_area = pred_area.unsqueeze(1)
        target_area = target_area.unsqueeze(0)
        
        union = pred_area + target_area - intersection
        
        # Calculate IoU
        iou = intersection / (union + 1e-6)
        
        return iou.max(dim=1)[0]  # Return max IoU for each prediction

class CustomTrainer:
    """Custom trainer for HMAY-TSF model without YOLO"""
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.model.to(device)
        
        # Loss function
        self.criterion = CustomLoss(num_classes=4)
        
        # Optimizer
        self.optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0005)
        
        # Scheduler
        self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=5, T_mult=2)
        
        # Mixed precision
        self.scaler = amp.GradScaler()
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch_idx, (images, targets) in enumerate(progress_bar):
            images = images.to(self.device)
            
            # Forward pass
            with amp.autocast():
                predictions = self.model(images)
                loss, box_loss, cls_loss, obj_loss = self.criterion(predictions, targets)
            
            # Backward pass
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Box': f'{box_loss.item():.4f}',
                'Cls': f'{cls_loss.item():.4f}',
                'Obj': f'{obj_loss.item():.4f}'
            })
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        self.train_losses.append(avg_loss)
        
        return avg_loss
    
    def validate_epoch(self, val_loader):
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc="Validation")
            
            for batch_idx, (images, targets) in enumerate(progress_bar):
                images = images.to(self.device)
                
                # Forward pass
                with amp.autocast():
                    predictions = self.model(images)
                    loss, box_loss, cls_loss, obj_loss = self.criterion(predictions, targets)
                
                total_loss += loss.item()
                num_batches += 1
                
                # Update progress bar
                progress_bar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Box': f'{box_loss.item():.4f}',
                    'Cls': f'{cls_loss.item():.4f}',
                    'Obj': f'{obj_loss.item():.4f}'
                })
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        self.val_losses.append(avg_loss)
        
        return avg_loss
    
    def train(self, train_loader, val_loader, epochs=20, save_dir='./runs/custom_train'):
        """Main training loop"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Starting custom training for {epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Save directory: {save_dir}")
        
        best_model_path = save_dir / 'best_model.pth'
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            print("="*50)
            
            # Training
            train_loss = self.train_epoch(train_loader)
            
            # Validation
            val_loss = self.validate_epoch(val_loader)
            
            # Learning rate scheduling
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Print metrics
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Learning Rate: {current_lr:.6f}")
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'best_val_loss': self.best_val_loss,
                }, best_model_path)
                print(f"✅ New best model saved! Val Loss: {val_loss:.4f}")
            
            # Save checkpoint every 5 epochs
            if (epoch + 1) % 5 == 0:
                checkpoint_path = save_dir / f'checkpoint_epoch_{epoch+1}.pth'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'best_val_loss': self.best_val_loss,
                }, checkpoint_path)
                print(f"💾 Checkpoint saved: {checkpoint_path}")
        
        # Save final model
        final_model_path = save_dir / 'final_model.pth'
        torch.save({
            'epoch': epochs,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'best_val_loss': self.best_val_loss,
        }, final_model_path)
        
        print(f"\n✅ Training completed!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"Final model saved: {final_model_path}")
        
        return self.model

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Custom HMAY-TSF Training without YOLO')
    parser.add_argument('--data', type=str, default='./Aerial-Vehicles-1/data.yaml', help='Dataset YAML path')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--img-size', type=int, default=640, help='Image size')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--save-dir', type=str, default='./runs/custom_train', help='Save directory')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = CustomDataset(args.data, args.img_size, is_training=True)
    val_dataset = CustomDataset(args.data, args.img_size, is_training=False)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Create model
    print("Creating model...")
    model = UltraOptimizedHMAY_TSF(
        model_size='n',
        num_classes=4,
        pretrained=True,
        use_yolov11=False
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=args.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    # Create trainer
    trainer = CustomTrainer(model, device=args.device)
    
    # Resume trainer state if specified
    if args.resume:
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        trainer.best_val_loss = checkpoint['best_val_loss']
    
    # Train
    trained_model = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        save_dir=args.save_dir
    )
    
    print("🎯 Custom training completed successfully!")

if __name__ == "__main__":
    main() 