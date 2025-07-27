#!/usr/bin/env python3
"""
Quick Start Script for HMAY-TSF Training
Demonstrates how to train the model with class balancing
"""

import os
import sys
import yaml
import torch
import argparse
from pathlib import Path

def check_requirements():
    """Check if all required packages are installed"""
    required_packages = [
        'torch', 'torchvision', 'ultralytics', 'opencv-python',
        'albumentations', 'matplotlib', 'seaborn', 'tensorboard',
        'tqdm', 'scikit-learn', 'Pillow'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("Missing required packages:")
        for package in missing_packages:
            print(f"  - {package}")
        print("\nInstall them using:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True

def check_dataset():
    """Check if dataset is properly structured"""
    dataset_path = Path('./dataset')
    
    if not dataset_path.exists():
        print("Error: Dataset directory './dataset' not found!")
        print("Please ensure your dataset is in the correct structure:")
        print("dataset/")
        print("├── dataset.yaml")
        print("├── images/")
        print("│   ├── train/")
        print("│   ├── val/")
        print("│   └── test/")
        print("└── labels/")
        print("    ├── train/")
        print("    ├── val/")
        print("    └── test/")
        return False
    
    # Check dataset.yaml
    yaml_path = dataset_path / 'dataset.yaml'
    if not yaml_path.exists():
        print("Error: dataset.yaml not found!")
        return False
    
    # Check directories
    required_dirs = [
        'images/train', 'images/val', 'images/test',
        'labels/train', 'labels/val', 'labels/test'
    ]
    
    for dir_path in required_dirs:
        full_path = dataset_path / dir_path
        if not full_path.exists():
            print(f"Error: Directory '{dir_path}' not found!")
            return False
    
    print("✓ Dataset structure is correct")
    return True

def analyze_class_distribution():
    """Analyze and display class distribution"""
    print("\n" + "="*60)
    print("CLASS DISTRIBUTION ANALYSIS")
    print("="*60)
    
    try:
        from data_preparation import ClassBalancedAugmentation
        
        analyzer = ClassBalancedAugmentation(
            dataset_path='./dataset',
            target_samples_per_class=5000
        )
        
        class_counts = analyzer.analyze_class_distribution()
        class_weights = analyzer.calculate_class_weights()
        
        print("\nClass Distribution:")
        print("-" * 40)
        total_samples = sum(class_counts.values())
        
        for class_id in sorted(class_counts.keys()):
            count = class_counts[class_id]
            weight = class_weights[class_id]
            percentage = (count / total_samples) * 100
            print(f"Class {class_id:2d}: {count:6,} samples ({percentage:5.2f}%) - Weight: {weight:.3f}")
        
        # Calculate imbalance metrics
        max_count = max(class_counts.values())
        min_count = min(class_counts.values())
        imbalance_ratio = max_count / min_count
        
        print(f"\nImbalance Analysis:")
        print(f"  Most common class: {max_count:,} samples")
        print(f"  Least common class: {min_count:,} samples")
        print(f"  Imbalance ratio: {imbalance_ratio:.2f}:1")
        
        if imbalance_ratio > 10:
            print("  ⚠️  Severe class imbalance detected!")
            print("  ✓ Class balancing strategies will be applied")
        elif imbalance_ratio > 5:
            print("  ⚠️  Moderate class imbalance detected")
            print("  ✓ Class balancing strategies will be applied")
        else:
            print("  ✓ Class distribution is relatively balanced")
        
        return True
        
    except Exception as e:
        print(f"Error analyzing class distribution: {e}")
        return False

def create_balanced_dataset():
    """Create balanced dataset if needed"""
    print("\n" + "="*60)
    print("DATASET BALANCING")
    print("="*60)
    
    try:
        from balance_dataset import main as balance_main
        
        print("Creating balanced dataset...")
        print("This will help address class imbalance issues.")
        
        # You can run the balancing script here
        # For now, we'll just inform the user
        print("To create a balanced dataset, run:")
        print("python balance_dataset.py --target_samples 5000")
        
        return True
        
    except Exception as e:
        print(f"Error in dataset balancing: {e}")
        return False

def setup_training_config():
    """Setup training configuration"""
    print("\n" + "="*60)
    print("TRAINING CONFIGURATION")
    print("="*60)
    
    config_path = Path('./config.yaml')
    
    if not config_path.exists():
        print("Creating default configuration file...")
        
        default_config = {
            'model_size': 's',
            'num_classes': 11,
            'pretrained': True,
            'img_size': 640,
            'batch_size': 16,
            'epochs': 100,
            'lr': 0.001,
            'weight_decay': 0.0005,
            'num_workers': 4,
            'patience': 10,
            'focal_alpha': 1.0,
            'focal_gamma': 2.0,
            'target_samples_per_class': 5000,
            'use_super_resolution': False,
            'copy_paste_augmentation': True,
            'active_learning': False,
            'gradient_clip': 10.0,
            'warmup_epochs': 5,
            'cosine_annealing': True,
            'label_smoothing': 0.1,
            'box_loss_weight': 7.5,
            'cls_loss_weight': 0.5,
            'dfl_loss_weight': 1.5,
            'val_frequency': 1,
            'save_frequency': 5,
            'output_dir': './runs/hmay_tsf_training',
            'save_best_only': True,
            'save_last': True,
            'tensorboard': True,
            'log_frequency': 10,
            'plot_class_distribution': True,
            'resume_training': False,
            'checkpoint_path': None,
            'device': 'auto',
            'mixed_precision': True,
            'compile_model': False,
            'data_yaml': 'dataset/dataset.yaml',
            'train_split': 0.8,
            'val_split': 0.2,
            'class_weights': {'auto_calculate': True},
            'tsf_enabled': True,
            'tsf_sequence_length': 8,
            'tsf_attention_heads': 16,
            'sr_enabled': False,
            'sr_scale_factor': 2,
            'sr_blocks': 12,
            'adaptive_anchors': True,
            'num_anchors': 12,
            'anchor_optimization': True
        }
        
        with open(config_path, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False, indent=2)
        
        print("✓ Default configuration created: config.yaml")
    else:
        print("✓ Configuration file exists: config.yaml")
    
    return True

def check_hardware():
    """Check hardware capabilities"""
    print("\n" + "="*60)
    print("HARDWARE CHECK")
    print("="*60)
    
    # Check CUDA availability
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        print(f"✓ CUDA available")
        print(f"  GPU: {gpu_name}")
        print(f"  Memory: {gpu_memory:.1f} GB")
        print(f"  Device count: {gpu_count}")
        
        if gpu_memory < 8:
            print("  ⚠️  Low GPU memory detected. Consider reducing batch size.")
            print("     Recommended: batch_size = 8")
        else:
            print("  ✓ Sufficient GPU memory for training")
    else:
        print("⚠️  CUDA not available. Training will use CPU (slow)")
        print("   Consider using a GPU for faster training")
    
    # Check system memory
    import psutil
    ram_gb = psutil.virtual_memory().total / 1024**3
    print(f"  System RAM: {ram_gb:.1f} GB")
    
    if ram_gb < 16:
        print("  ⚠️  Low system memory. Consider reducing num_workers.")
        print("     Recommended: num_workers = 2")
    
    return True

def start_training():
    """Start the training process"""
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)
    
    print("To start training, run one of the following commands:")
    print()
    print("1. Basic training with default settings:")
    print("   python train_hmay_tsf.py")
    print()
    print("2. Training with custom configuration:")
    print("   python train_hmay_tsf.py --config config.yaml")
    print()
    print("3. Training with custom dataset path:")
    print("   python train_hmay_tsf.py --data_yaml dataset/dataset.yaml")
    print()
    print("4. Resume training from checkpoint:")
    print("   python train_hmay_tsf.py --resume")
    print()
    print("5. Training with custom output directory:")
    print("   python train_hmay_tsf.py --output_dir ./my_training_run")
    print()
    
    # Ask user if they want to start training now
    response = input("Do you want to start training now? (y/n): ").lower().strip()
    
    if response in ['y', 'yes']:
        print("\nStarting training...")
        os.system("python train_hmay_tsf.py")
    else:
        print("\nTraining not started. You can run the commands above when ready.")

def main():
    """Main quick start function"""
    print("="*80)
    print("HMAY-TSF QUICK START")
    print("Enhanced Object Detection with Class Balancing")
    print("="*80)
    
    # Check requirements
    print("\n1. Checking requirements...")
    if not check_requirements():
        return
    
    # Check dataset
    print("\n2. Checking dataset structure...")
    if not check_dataset():
        return
    
    # Analyze class distribution
    print("\n3. Analyzing class distribution...")
    if not analyze_class_distribution():
        return
    
    # Setup configuration
    print("\n4. Setting up training configuration...")
    if not setup_training_config():
        return
    
    # Check hardware
    print("\n5. Checking hardware...")
    if not check_hardware():
        return
    
    # Optional: Create balanced dataset
    print("\n6. Dataset balancing options...")
    create_balanced_dataset()
    
    # Start training
    print("\n7. Ready to train!")
    start_training()
    
    print("\n" + "="*80)
    print("QUICK START COMPLETE")
    print("="*80)
    print("\nFor more information, see:")
    print("- README.md")
    print("- methodology.txt")
    print("- config.yaml")
    print("\nHappy training! 🚀")

if __name__ == "__main__":
    main() 