#!/usr/bin/env python3
"""
Simplified HMAY-TSF Training Script with Better Performance
"""

import os
import sys
import torch
import torch.nn as nn
import torch.amp as amp
from pathlib import Path
from ultralytics import YOLO
import yaml
import csv
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from hmay_tsf_model import HMAY_TSF

class SimpleHMAYTSFTrainer:
    """Simplified trainer with better performance"""
    
    def __init__(self, device='auto'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else device
        self.model = None
        self.best_f1 = 0.0
        self.current_epoch = 0
        
    def setup_model(self, num_classes=4):
        """Setup simplified HMAY-TSF model"""
        print("🔧 Setting up simplified HMAY-TSF model...")
        
        # Create simplified model
        self.model = HMAY_TSF(
            model_size='n',  # Use nano for faster training
            num_classes=num_classes,
            pretrained=True,
            use_yolov11=False  # Use YOLOv8 for stability
        )
        
        print(f"✅ Model created successfully!")
        print(f"Total parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        
        return self.model
    
    def train_model(self, data_yaml, epochs=50, batch_size=16, img_size=640):
        """Train with optimized hyperparameters"""
        print(f"🚀 Starting simplified training for {epochs} epochs...")
        
        # Optimized training arguments
        train_args = {
            'data': data_yaml,
            'epochs': epochs,
            'imgsz': img_size,
            'batch': batch_size,
            'device': self.device,
            'patience': 20,
            'save': True,
            'project': './runs/train',
            'name': f'simple_hmay_tsf_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            
            # Optimized hyperparameters for better performance
            'optimizer': 'AdamW',
            'lr0': 0.001,  # Higher learning rate
            'lrf': 0.1,    # Faster decay
            'momentum': 0.95,
            'weight_decay': 0.0005,
            'warmup_epochs': 3,
            
            # Loss weights optimized for detection
            'box': 0.05,   # Lower box loss weight
            'cls': 0.5,    # Balanced classification
            'dfl': 1.5,    # Higher DFL for better localization
            
            # Detection thresholds
            'conf': 0.25,
            'iou': 0.45,
            
            # Minimal augmentation for stability
            'mosaic': 0.0,  # Disable mosaic
            'mixup': 0.0,   # Disable mixup
            'copy_paste': 0.0,
            'degrees': 0.0,
            'translate': 0.1,
            'scale': 0.5,
            'shear': 0.0,
            'perspective': 0.0,
            'flipud': 0.0,
            'fliplr': 0.5,
            
            # Color augmentation
            'hsv_h': 0.015,
            'hsv_s': 0.7,
            'hsv_v': 0.4,
            
            # Other optimizations
            'dropout': 0.0,  # No dropout for better convergence
            'amp': True,     # Mixed precision
            'cache': True,   # Cache images
            'workers': 4,
            'verbose': True
        }
        
        try:
            # Train using YOLO's training loop
            results = self.model.base_yolo.train(**train_args)
            return results
            
        except Exception as e:
            print(f"Training error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def on_epoch_end(self, trainer):
        """Callback to track real metrics"""
        try:
            self.current_epoch += 1
            epoch = self.current_epoch
            
            # Get real metrics from trainer
            metrics = {}
            metrics['epoch'] = epoch
            
            # Learning rate
            if hasattr(trainer, 'optimizer') and trainer.optimizer is not None:
                lr = trainer.optimizer.param_groups[0]['lr']
                metrics['lr'] = float(lr.item() if isinstance(lr, torch.Tensor) else lr)
            else:
                metrics['lr'] = 0.001
            
            # Get validation metrics
            if hasattr(trainer, 'metrics') and trainer.metrics is not None:
                det_metrics = trainer.metrics
                
                if hasattr(det_metrics, 'box') and det_metrics.box is not None:
                    box_metrics = det_metrics.box
                    
                    # Extract real metrics
                    metrics['val_precision'] = float(box_metrics.mp) if hasattr(box_metrics, 'mp') and box_metrics.mp is not None else 0.0
                    metrics['val_recall'] = float(box_metrics.mr) if hasattr(box_metrics, 'mr') and box_metrics.mr is not None else 0.0
                    metrics['map50'] = float(box_metrics.map50) if hasattr(box_metrics, 'map50') and box_metrics.map50 is not None else 0.0
                    metrics['map50_95'] = float(box_metrics.map) if hasattr(box_metrics, 'map') and box_metrics.map is not None else 0.0
                    
                    # Calculate F1 score
                    precision = metrics['val_precision']
                    recall = metrics['val_recall']
                    if precision + recall > 0:
                        metrics['val_f1'] = 2 * (precision * recall) / (precision + recall)
                    else:
                        metrics['val_f1'] = 0.0
                else:
                    metrics['val_precision'] = 0.0
                    metrics['val_recall'] = 0.0
                    metrics['map50'] = 0.0
                    metrics['map50_95'] = 0.0
                    metrics['val_f1'] = 0.0
            else:
                metrics['val_precision'] = 0.0
                metrics['val_recall'] = 0.0
                metrics['map50'] = 0.0
                metrics['map50_95'] = 0.0
                metrics['val_f1'] = 0.0
            
            # Loss
            if hasattr(trainer, 'loss') and trainer.loss is not None:
                loss = trainer.loss
                if isinstance(loss, torch.Tensor):
                    metrics['train_loss'] = float(loss.item() if loss.numel() == 1 else loss.mean().item())
                else:
                    metrics['train_loss'] = float(loss)
            else:
                metrics['train_loss'] = 0.0
            
            # Print metrics
            print(f"\n📊 Epoch {epoch} - Real Performance:")
            print(f"   Precision: {metrics['val_precision']:.6f}")
            print(f"   Recall: {metrics['val_recall']:.6f}")
            print(f"   F1-Score: {metrics['val_f1']:.6f}")
            print(f"   mAP@0.5: {metrics['map50']:.6f}")
            print(f"   mAP@0.5:0.95: {metrics['map50_95']:.6f}")
            print(f"   Loss: {metrics['train_loss']:.6f}")
            print(f"   LR: {metrics['lr']:.8f}")
            
            # Update best F1
            if metrics['val_f1'] > self.best_f1:
                self.best_f1 = metrics['val_f1']
                print(f"🎯 NEW BEST F1-SCORE: {self.best_f1:.6f}")
            
        except Exception as e:
            print(f"Error in epoch callback: {e}")

def main():
    """Main training function"""
    print("🚀 Starting Simplified HMAY-TSF Training")
    
    # Initialize trainer
    trainer = SimpleHMAYTSFTrainer()
    
    # Setup model
    model = trainer.setup_model(num_classes=4)
    
    # Training configuration
    data_yaml = "./Aerial-Vehicles-1/data.yaml"
    epochs = 50
    batch_size = 16
    img_size = 640
    
    print(f"\nTraining Configuration:")
    print(f"  Data: {data_yaml}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Image size: {img_size}")
    print(f"  Device: {trainer.device}")
    
    # Start training
    results = trainer.train_model(
        data_yaml=data_yaml,
        epochs=epochs,
        batch_size=batch_size,
        img_size=img_size
    )
    
    if results:
        print(f"\n✅ Training completed successfully!")
        print(f"Best F1-Score: {trainer.best_f1:.6f}")
    else:
        print(f"\n❌ Training failed!")

if __name__ == "__main__":
    main() 