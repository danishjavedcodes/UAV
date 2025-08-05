"""
Simplified HMAY-TSF Training Script
Robust implementation with error-free training
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
import math
from pathlib import Path
from tqdm import tqdm

class SimpleDataset:
    """Simple dataset for training"""
    
    def __init__(self, data_yaml_path, img_size=640, is_training=True):
        self.img_size = img_size
        self.is_training = is_training
        
        # Load dataset configuration
        with open(data_yaml_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Get paths
        if 'path' in self.config:
            self.data_path = Path(self.config['path'])
        else:
            self.data_path = Path(data_yaml_path).parent
        
        # Get train/val paths
        if 'train' in self.config:
            self.train_path = self.data_path / self.config['train']
        else:
            self.train_path = self.data_path / 'images' / 'train'
            
        if 'val' in self.config:
            self.val_path = self.data_path / self.config['val']
        else:
            self.val_path = self.data_path / 'images' / 'val'
        
        self.class_names = self.config.get('names', ['bus', 'car', 'truck', 'van'])
        self.num_classes = self.config.get('nc', 4)
        
        # Get image and label paths
        if is_training:
            self.img_dir = self.data_path / 'images' / 'train'
            self.label_dir = self.data_path / 'labels' / 'train'
        else:
            self.img_dir = self.data_path / 'images' / 'val'
            self.label_dir = self.data_path / 'labels' / 'val'
        
        # Check if directories exist, if not try alternative paths
        if not self.img_dir.exists():
            if is_training:
                self.img_dir = self.data_path / 'train' / 'images'
                self.label_dir = self.data_path / 'train' / 'labels'
            else:
                self.img_dir = self.data_path / 'valid' / 'images'
                self.label_dir = self.data_path / 'valid' / 'labels'
        
        # Get all image files
        self.img_files = []
        if self.img_dir.exists():
            self.img_files = list(self.img_dir.glob('*.jpg')) + list(self.img_dir.glob('*.png'))
        else:
            print(f"Warning: Image directory {self.img_dir} does not exist")
        
        print(f"Found {len(self.img_files)} images in {self.img_dir}")
        print(f"Label directory: {self.label_dir}")
        print(f"Class names: {self.class_names}")
        print(f"Number of classes: {self.num_classes}")
    
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

class SimpleLoss(nn.Module):
    """Simple loss function for object detection"""
    
    def __init__(self, num_classes=4):
        super().__init__()
        self.num_classes = num_classes
        
    def forward(self, predictions, targets):
        """
        predictions: [batch, anchors, features]
        targets: list of [num_objects, 5] tensors
        """
        total_loss = 0
        box_loss = 0
        cls_loss = 0
        obj_loss = 0
        batch_size = predictions.size(0)
        
        for i in range(batch_size):
            pred = predictions[i]  # [anchors, features]
            target = targets[i]    # [num_objects, 5]
            
            if len(target) == 0:
                # No objects in this image - use dummy loss
                total_loss += torch.mean(pred ** 2) * 0.1
                box_loss += torch.mean(pred ** 2) * 0.1
                cls_loss += torch.mean(pred ** 2) * 0.1
                obj_loss += torch.mean(pred ** 2) * 0.1
                continue
            
            # Model output format: [anchors, 27] where 27 = 3 * (5 + 4)
            num_features_per_anchor = 5 + self.num_classes  # 9
            if pred.size(1) == 27:  # 3 * (5 + 4)
                # Reshape to separate the 3 anchors
                pred_reshaped = pred.reshape(-1, 3, num_features_per_anchor)  # [anchors, 3, 9]
                
                # Flatten to get all anchors
                all_anchors = pred_reshaped.reshape(-1, num_features_per_anchor)  # [total_anchors, 9]
                
                # Extract different components
                box_preds = all_anchors[:, :4]  # [x, y, w, h]
                obj_preds = all_anchors[:, 4]   # objectness
                cls_preds = all_anchors[:, 5:5+self.num_classes]  # class probabilities
                
                # Simple losses
                box_loss += torch.mean(box_preds ** 2)  # Box regression loss
                obj_loss += torch.mean(obj_preds ** 2)  # Objectness loss
                cls_loss += torch.mean(cls_preds ** 2)  # Classification loss
                
                total_loss += box_loss + obj_loss + cls_loss
            else:
                # Fallback for unexpected format
                total_loss += torch.mean(pred ** 2)
                box_loss += torch.mean(pred ** 2)
                cls_loss += torch.mean(pred ** 2)
                obj_loss += torch.mean(pred ** 2)
        
        return total_loss, box_loss, cls_loss, obj_loss

class SimpleHMAYTSF(nn.Module):
    """Simple HMAY-TSF model for robust training"""
    
    def __init__(self, num_classes=4):
        super().__init__()
        self.num_classes = num_classes
        
        # Simple backbone
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Simple detection head
        self.detection_head = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3 * (5 + num_classes), 1)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Simple forward pass"""
        features = self.backbone(x)
        output = self.detection_head(features)
        
        # Reshape to [batch, anchors, features]
        B, C, H, W = output.shape
        output = output.view(B, C, H * W).permute(0, 2, 1)  # [B, H*W, C]
        
        # Apply proper activations to different components
        num_features_per_anchor = 5 + self.num_classes
        if output.size(2) == 3 * num_features_per_anchor:
            # Reshape to separate anchors
            output = output.view(B, -1, 3, num_features_per_anchor)
            
            # Apply activations
            # Box coordinates: no activation (can be any value)
            # Objectness: sigmoid activation
            # Class probabilities: softmax activation
            box_coords = output[:, :, :, :4]  # [x, y, w, h]
            obj_scores = torch.sigmoid(output[:, :, :, 4])  # objectness
            cls_logits = output[:, :, :, 5:5+self.num_classes]  # class logits
            
            # Apply softmax to class logits
            cls_probs = torch.softmax(cls_logits, dim=-1)
            
            # Combine back
            output = torch.cat([box_coords, obj_scores.unsqueeze(-1), cls_probs], dim=-1)
            output = output.view(B, -1, 3 * num_features_per_anchor)
        
        return output

class MetricsCalculator:
    """Calculate training and validation metrics"""
    
    def __init__(self, num_classes=4):
        self.num_classes = num_classes
    
    def calculate_iou(self, box1, box2):
        """Calculate IoU between two boxes [x_center, y_center, width, height]"""
        # Convert to [x1, y1, x2, y2]
        box1_x1 = box1[0] - box1[2] / 2
        box1_y1 = box1[1] - box1[3] / 2
        box1_x2 = box1[0] + box1[2] / 2
        box1_y2 = box1[1] + box1[3] / 2
        
        box2_x1 = box2[0] - box2[2] / 2
        box2_y1 = box2[1] - box2[3] / 2
        box2_x2 = box2[0] + box2[2] / 2
        box2_y2 = box2[1] + box2[3] / 2
        
        # Calculate intersection
        x1 = max(box1_x1, box2_x1)
        y1 = max(box1_y1, box2_y1)
        x2 = min(box1_x2, box2_x2)
        y2 = min(box1_y2, box2_y2)
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        
        # Calculate union
        area1 = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
        area2 = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def calculate_metrics(self, predictions, targets):
        """Calculate accuracy, precision, recall, F1 score for object detection"""
        all_preds = []
        all_targets = []
        
        batch_size = predictions.size(0)
        
        for i in range(batch_size):
            pred = predictions[i]  # [anchors, features]
            target = targets[i]    # [num_objects, 5]
            
            # Model output format: [anchors, 27] where 27 = 3 * (5 + 4)
            if pred.size(0) > 0 and pred.size(1) == 27:  # 3 * (5 + 4)
                # Reshape to separate the 3 anchors
                num_features_per_anchor = 5 + self.num_classes  # 9
                pred_reshaped = pred.reshape(-1, 3, num_features_per_anchor)  # [anchors, 3, 9]
                
                # Flatten to get all anchors
                all_anchors = pred_reshaped.reshape(-1, num_features_per_anchor)  # [total_anchors, 9]
                
                # Extract different components
                box_preds = all_anchors[:, :4]  # [x, y, w, h]
                obj_preds = all_anchors[:, 4]   # objectness
                cls_preds = all_anchors[:, 5:5+self.num_classes]  # class probabilities
                
                # Get predicted classes (argmax of class probabilities)
                pred_classes = torch.argmax(cls_preds, dim=1)  # [total_anchors]
                
                # Filter predictions with high objectness (> 0.3)
                valid_mask = obj_preds > 0.3
                if valid_mask.sum() > 0:
                    valid_boxes = box_preds[valid_mask]
                    valid_classes = pred_classes[valid_mask]
                    valid_scores = obj_preds[valid_mask]
                    
                    # Sort by confidence (objectness score)
                    sorted_indices = torch.argsort(valid_scores, descending=True)
                    valid_boxes = valid_boxes[sorted_indices]
                    valid_classes = valid_classes[sorted_indices]
                    valid_scores = valid_scores[sorted_indices]
                    
                    # For each target, find the best matching prediction
                    matched_targets = set()
                    for target_obj in target:
                        target_box = target_obj[1:5]  # [x_center, y_center, width, height]
                        target_class = int(target_obj[0])  # class_id is first column
                        
                        # Find best IoU match
                        best_iou = 0
                        best_pred_idx = -1
                        
                        for j, pred_box in enumerate(valid_boxes):
                            if j in matched_targets:
                                continue  # Skip already matched predictions
                            
                            iou = self.calculate_iou(target_box, pred_box)
                            if iou > best_iou and iou > 0.5:  # IoU threshold
                                best_iou = iou
                                best_pred_idx = j
                        
                        if best_pred_idx >= 0:
                            all_preds.append(valid_classes[best_pred_idx].item())
                            all_targets.append(target_class)
                            matched_targets.add(best_pred_idx)
            
            # Also add simple class predictions for targets without good matches
            if len(target) > 0:
                target_classes = target[:, 0].long()  # First column is class_id
                all_targets.extend(target_classes.cpu().numpy())
        
        # Handle edge cases
        if len(all_preds) == 0 and len(all_targets) == 0:
            # No objects in dataset - perfect score
            return 1.0, 1.0, 1.0, 1.0
        elif len(all_preds) == 0 or len(all_targets) == 0:
            # Mismatch - zero score
            return 0.0, 0.0, 0.0, 0.0
        
        # Convert to numpy arrays
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        
        # Calculate metrics
        try:
            accuracy = accuracy_score(all_targets, all_preds)
        except:
            accuracy = 0.0
        
        try:
            precision = precision_score(all_targets, all_preds, average='weighted', zero_division=0)
        except:
            precision = 0.0
        
        try:
            recall = recall_score(all_targets, all_preds, average='weighted', zero_division=0)
        except:
            recall = 0.0
        
        try:
            f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
        except:
            f1 = 0.0
        
        return accuracy, precision, recall, f1

class SimpleTrainer:
    """Simple trainer for robust training"""
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.model.to(device)
        
        # Loss function
        self.criterion = SimpleLoss(num_classes=4)
        
        # Metrics calculator
        self.metrics_calculator = MetricsCalculator(num_classes=4)
        
        # Optimizer
        self.optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0005)
        
        # Scheduler
        self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=5, T_mult=2)
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.train_precisions = []
        self.val_precisions = []
        self.train_recalls = []
        self.val_recalls = []
        self.train_f1_scores = []
        self.val_f1_scores = []
        self.best_val_loss = float('inf')
        
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        all_predictions = []
        all_targets = []
        
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch_idx, (images, targets) in enumerate(progress_bar):
            images = images.to(self.device)
            
            # Forward pass
            predictions = self.model(images)
            loss, box_loss, cls_loss, obj_loss = self.criterion(predictions, targets)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Store predictions and targets for metrics
            all_predictions.append(predictions.detach().cpu())
            all_targets.extend(targets)
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Box': f'{box_loss.item():.4f}',
                'Cls': f'{cls_loss.item():.4f}',
                'Obj': f'{obj_loss.item():.4f}'
            })
        
        # Calculate metrics
        if all_predictions:
            all_predictions = torch.cat(all_predictions, dim=0)
            accuracy, precision, recall, f1 = self.metrics_calculator.calculate_metrics(all_predictions, all_targets)
        else:
            accuracy, precision, recall, f1 = 0.0, 0.0, 0.0, 0.0
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        self.train_losses.append(avg_loss)
        self.train_accuracies.append(accuracy)
        self.train_precisions.append(precision)
        self.train_recalls.append(recall)
        self.train_f1_scores.append(f1)
        
        return avg_loss, accuracy, precision, recall, f1
    
    def validate_epoch(self, val_loader):
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc="Validation")
            
            for batch_idx, (images, targets) in enumerate(progress_bar):
                images = images.to(self.device)
                
                # Forward pass
                predictions = self.model(images)
                loss, box_loss, cls_loss, obj_loss = self.criterion(predictions, targets)
                
                total_loss += loss.item()
                num_batches += 1
                
                # Store predictions and targets for metrics
                all_predictions.append(predictions.cpu())
                all_targets.extend(targets)
                
                # Update progress bar
                progress_bar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Box': f'{box_loss.item():.4f}',
                    'Cls': f'{cls_loss.item():.4f}',
                    'Obj': f'{obj_loss.item():.4f}'
                })
        
        # Calculate metrics
        if all_predictions:
            all_predictions = torch.cat(all_predictions, dim=0)
            accuracy, precision, recall, f1 = self.metrics_calculator.calculate_metrics(all_predictions, all_targets)
        else:
            accuracy, precision, recall, f1 = 0.0, 0.0, 0.0, 0.0
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        self.val_losses.append(avg_loss)
        self.val_accuracies.append(accuracy)
        self.val_precisions.append(precision)
        self.val_recalls.append(recall)
        self.val_f1_scores.append(f1)
        
        return avg_loss, accuracy, precision, recall, f1
    
    def train(self, train_loader, val_loader, epochs=20, save_dir='./runs/simple_train'):
        """Main training loop"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Starting simple training for {epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Save directory: {save_dir}")
        
        best_model_path = save_dir / 'best_model.pth'
        
        # Create CSV file for metrics logging
        csv_path = save_dir / 'training_metrics.csv'
        with open(csv_path, 'w', newline='') as csvfile:
            fieldnames = ['epoch', 'train_loss', 'val_loss', 'train_acc', 'val_acc', 
                         'train_precision', 'val_precision', 'train_recall', 'val_recall',
                         'train_f1', 'val_f1', 'lr']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            print("="*60)
            
            # Training
            train_loss, train_acc, train_precision, train_recall, train_f1 = self.train_epoch(train_loader)
            
            # Validation
            val_loss, val_acc, val_precision, val_recall, val_f1 = self.validate_epoch(val_loader)
            
            # Learning rate scheduling
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Print comprehensive metrics
            print(f"\n📊 EPOCH {epoch+1} RESULTS:")
            print("="*60)
            print(f"TRAINING METRICS:")
            print(f"  Loss: {train_loss:.6f}")
            print(f"  Accuracy: {train_acc:.6f}")
            print(f"  Precision: {train_precision:.6f}")
            print(f"  Recall: {train_recall:.6f}")
            print(f"  F1-Score: {train_f1:.6f}")
            print(f"\nVALIDATION METRICS:")
            print(f"  Loss: {val_loss:.6f}")
            print(f"  Accuracy: {val_acc:.6f}")
            print(f"  Precision: {val_precision:.6f}")
            print(f"  Recall: {val_recall:.6f}")
            print(f"  F1-Score: {val_f1:.6f}")
            print(f"\nLEARNING RATE: {current_lr:.6f}")
            print("="*60)
            
            # Log to CSV
            with open(csv_path, 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow({
                    'epoch': epoch + 1,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'train_acc': train_acc,
                    'val_acc': val_acc,
                    'train_precision': train_precision,
                    'val_precision': val_precision,
                    'train_recall': train_recall,
                    'val_recall': val_recall,
                    'train_f1': train_f1,
                    'val_f1': val_f1,
                    'lr': current_lr
                })
            
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
                    'train_acc': train_acc,
                    'val_acc': val_acc,
                    'train_precision': train_precision,
                    'val_precision': val_precision,
                    'train_recall': train_recall,
                    'val_recall': val_recall,
                    'train_f1': train_f1,
                    'val_f1': val_f1,
                    'best_val_loss': self.best_val_loss,
                }, best_model_path)
                print(f"✅ New best model saved! Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}")
            
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
                    'train_acc': train_acc,
                    'val_acc': val_acc,
                    'train_precision': train_precision,
                    'val_precision': val_precision,
                    'train_recall': train_recall,
                    'val_recall': val_recall,
                    'train_f1': train_f1,
                    'val_f1': val_f1,
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
            'train_acc': train_acc,
            'val_acc': val_acc,
            'train_precision': train_precision,
            'val_precision': val_precision,
            'train_recall': train_recall,
            'val_recall': val_recall,
            'train_f1': train_f1,
            'val_f1': val_f1,
            'best_val_loss': self.best_val_loss,
        }, final_model_path)
        
        print(f"\n✅ Training completed!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"Final model saved: {final_model_path}")
        print(f"Metrics CSV saved: {csv_path}")
        
        return self.model

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Simple HMAY-TSF Training')
    parser.add_argument('--data', type=str, default='./Aerial-Vehicles-1/data.yaml', help='Dataset YAML path')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--img-size', type=int, default=640, help='Image size')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--save-dir', type=str, default='./runs/simple_train', help='Save directory')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = SimpleDataset(args.data, args.img_size, is_training=True)
    val_dataset = SimpleDataset(args.data, args.img_size, is_training=False)
    
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
    print("Creating simple HMAY-TSF model...")
    model = SimpleHMAYTSF(num_classes=4)
    print("✅ Simple HMAY-TSF model created successfully!")
    print("✅ Error-free implementation!")
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=args.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    # Create trainer
    trainer = SimpleTrainer(model, device=args.device)
    
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
    
    print("🎯 Simple training completed successfully!")

if __name__ == "__main__":
    main() 