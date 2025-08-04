#!/usr/bin/env python3
"""
Simple YOLO Training with Optimized Hyperparameters for Better Performance
"""

import os
import sys
import torch
from pathlib import Path
from ultralytics import YOLO
from datetime import datetime

def train_simple_yolo():
    """Train YOLO with optimized hyperparameters"""
    print("🚀 Starting Simple YOLO Training with Optimized Hyperparameters")
    
    # Load YOLO model
    model = YOLO('yolov8n.pt')
    
    # Training configuration with optimized hyperparameters
    train_args = {
        'data': './Aerial-Vehicles-1/data.yaml',
        'epochs': 100,
        'imgsz': 640,
        'batch': 16,
        'device': 'auto',
        'patience': 50,
        'save': True,
        'project': './runs/train',
        'name': f'simple_yolo_optimized_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
        
        # Optimized hyperparameters for better performance
        'optimizer': 'AdamW',
        'lr0': 0.001,  # Higher learning rate
        'lrf': 0.1,    # Faster decay
        'momentum': 0.95,
        'weight_decay': 0.0005,
        'warmup_epochs': 5,
        
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
    
    print(f"\nTraining Configuration:")
    print(f"  Data: {train_args['data']}")
    print(f"  Epochs: {train_args['epochs']}")
    print(f"  Batch size: {train_args['batch']}")
    print(f"  Image size: {train_args['imgsz']}")
    print(f"  Learning rate: {train_args['lr0']}")
    print(f"  Optimizer: {train_args['optimizer']}")
    
    try:
        # Start training
        results = model.train(**train_args)
        
        print(f"\n✅ Training completed successfully!")
        print(f"Results saved to: {results.save_dir}")
        
        # Print final metrics
        if hasattr(results, 'results_dict'):
            metrics = results.results_dict
            print(f"\n📊 Final Performance:")
            print(f"  mAP@0.5: {metrics.get('metrics/mAP50(B)', 'N/A')}")
            print(f"  mAP@0.5:0.95: {metrics.get('metrics/mAP50-95(B)', 'N/A')}")
            print(f"  Precision: {metrics.get('metrics/precision(B)', 'N/A')}")
            print(f"  Recall: {metrics.get('metrics/recall(B)', 'N/A')}")
        
        return results
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main function"""
    print("🎯 Simple YOLO Training for UAV Detection")
    print("=" * 50)
    
    # Check if dataset exists
    data_yaml = './Aerial-Vehicles-1/data.yaml'
    if not os.path.exists(data_yaml):
        print(f"❌ Dataset not found: {data_yaml}")
        print("Please ensure the dataset is downloaded and configured correctly.")
        return
    
    # Start training
    results = train_simple_yolo()
    
    if results:
        print(f"\n🎉 Training completed successfully!")
        print(f"Model saved to: {results.save_dir}")
    else:
        print(f"\n❌ Training failed!")

if __name__ == "__main__":
    main() 