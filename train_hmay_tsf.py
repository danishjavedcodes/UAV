"""
Training Script for HMAY-TSF Model
Simplified implementation focusing on practical results
"""

import os
import torch
import torch.nn as nn
from ultralytics import YOLO
import yaml
import wandb
from pathlib import Path
import numpy as np
from datetime import datetime
import argparse
import json

from hmay_tsf_model import HMAY_TSF, prepare_visdrone_dataset
from data_preparation import prepare_visdrone_dataset as prep_dataset, get_dataloader

class HMAYTSFTrainer:
    """Simplified trainer for HMAY-TSF model"""
    
    def __init__(self, model_size='s', device='auto', project_name='HMAY-TSF'):
        self.device = device if device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_size = model_size
        self.project_name = project_name
        
        print(f"Using device: {self.device}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        # Initialize model
        self.model = None
        self.best_map = 0.0
        
    def setup_model(self, num_classes=11, pretrained=True):
        """Setup the HMAY-TSF model"""
        print("Setting up HMAY-TSF model...")
        
        # For simplicity, start with enhanced YOLOv8
        model_name = f'yolov8{self.model_size}.pt' if pretrained else f'yolov8{self.model_size}.yaml'
        self.model = YOLO(model_name)
        
        print(f"Model {model_name} loaded successfully!")
        return self.model
    
    def train_model(self, data_yaml, epochs=100, img_size=640, batch_size=16, 
                   save_dir='./runs/train', patience=50, resume=False):
        """Train the model with enhanced configuration"""
        
        print(f"Starting training with:")
        print(f"  Data: {data_yaml}")
        print(f"  Epochs: {epochs}")
        print(f"  Image size: {img_size}")
        print(f"  Batch size: {batch_size}")
        print(f"  Device: {self.device}")
        
        # Enhanced training arguments for UAV object detection
        train_args = {
            'data': data_yaml,
            'epochs': epochs,
            'imgsz': img_size,
            'batch': batch_size,
            'device': self.device,
            'workers': 8,
            'patience': patience,
            'save': True,
            'save_period': 10,
            'cache': False,  # Set to True if you have enough RAM
            'project': save_dir,
            'name': f'hmay_tsf_{self.model_size}_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            'exist_ok': True,
            'resume': resume,
            
            # Optimization settings for small objects
            'optimizer': 'AdamW',
            'lr0': 0.001,  # Lower learning rate for better convergence
            'lrf': 0.01,
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 3,
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,
            
            # Augmentation settings
            'hsv_h': 0.015,
            'hsv_s': 0.7,
            'hsv_v': 0.4,
            'degrees': 0.373,
            'translate': 0.245,
            'scale': 0.898,
            'shear': 0.602,
            'perspective': 0.0,
            'flipud': 0.00856,
            'fliplr': 0.5,
            'mosaic': 1.0,
            'mixup': 0.243,
            'copy_paste': 0.362,
            
            # Loss settings for small objects
            'box': 7.5,
            'cls': 0.5,
            'dfl': 1.5,
            
            # Additional settings
            'verbose': True,
            'plots': True,
            'val': True,
        }
        
        print("Starting training...")
        try:
            # Start training
            results = self.model.train(**train_args)
            
            print("Training completed successfully!")
            print(f"Best mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
            print(f"Best mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")
            
            return results
            
        except Exception as e:
            print(f"Training failed with error: {e}")
            return None
    
    def evaluate_model(self, data_yaml, weights_path=None):
        """Evaluate the trained model"""
        print("Evaluating model...")
        
        if weights_path:
            self.model = YOLO(weights_path)
        
        try:
            # Validation
            val_results = self.model.val(data=data_yaml, device=self.device)
            
            print("Evaluation Results:")
            print(f"mAP50: {val_results.box.map50:.4f}")
            print(f"mAP50-95: {val_results.box.map:.4f}")
            print(f"Precision: {val_results.box.mp:.4f}")
            print(f"Recall: {val_results.box.mr:.4f}")
            
            return val_results
            
        except Exception as e:
            print(f"Evaluation failed with error: {e}")
            return None
    
    def predict_and_visualize(self, source, save_dir='./runs/predict', conf=0.25):
        """Run prediction and save results"""
        print(f"Running prediction on {source}...")
        
        try:
            results = self.model.predict(
                source=source,
                save=True,
                save_txt=True,
                save_conf=True,
                conf=conf,
                project=save_dir,
                name='hmay_tsf_predictions',
                exist_ok=True,
                device=self.device
            )
            
            print(f"Predictions saved to {save_dir}")
            return results
            
        except Exception as e:
            print(f"Prediction failed with error: {e}")
            return None

def main():
    parser = argparse.ArgumentParser(description='Train HMAY-TSF Model')
    parser.add_argument('--data', type=str, default='./dataset', help='Dataset path')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--img-size', type=int, default=640, help='Image size')
    parser.add_argument('--model-size', type=str, default='s', choices=['n', 's', 'm', 'l', 'x'], help='Model size')
    parser.add_argument('--device', type=str, default='auto', help='Device to use')
    parser.add_argument('--resume', action='store_true', help='Resume training')
    parser.add_argument('--evaluate-only', action='store_true', help='Only evaluate')
    parser.add_argument('--weights', type=str, help='Path to weights for evaluation')
    parser.add_argument('--predict', type=str, help='Source for prediction')
    
    args = parser.parse_args()
    
    print("="*50)
    print("HMAY-TSF: Hybrid Multi-Scale Adaptive YOLO with Temporal-Spatial Fusion")
    print("Simplified Implementation for UAV Traffic Object Detection")
    print("="*50)
    
    # Initialize trainer
    trainer = HMAYTSFTrainer(model_size=args.model_size, device=args.device)
    
    # Prepare dataset
    print("Preparing dataset...")
    data_yaml = prep_dataset(args.data)
    
    if not data_yaml:
        print("Failed to prepare dataset!")
        return
    
    # Setup model
    trainer.setup_model()
    
    if args.predict:
        # Prediction mode
        if args.weights:
            trainer.model = YOLO(args.weights)
        trainer.predict_and_visualize(args.predict)
        
    elif args.evaluate_only:
        # Evaluation mode
        trainer.evaluate_model(data_yaml, args.weights)
        
    else:
        # Training mode
        print("Starting training phase...")
        results = trainer.train_model(
            data_yaml=data_yaml,
            epochs=args.epochs,
            img_size=args.img_size,
            batch_size=args.batch_size,
            resume=args.resume
        )
        
        if results:
            print("Training completed successfully!")
            
            # Run evaluation on the best model
            print("Running final evaluation...")
            best_weights = results.save_dir / 'weights' / 'best.pt'
            if best_weights.exists():
                trainer.evaluate_model(data_yaml, str(best_weights))
            
            # Test prediction on validation set
            val_img_dir = Path(args.data) / 'images' / 'val'
            if val_img_dir.exists():
                print("Running test predictions...")
                trainer.predict_and_visualize(str(val_img_dir))
        
        else:
            print("Training failed!")

if __name__ == "__main__":
    main() 