#!/usr/bin/env python3
"""
Quick Start Script for HMAY-TSF Implementation
Automates setup, training, and evaluation
"""

import os
import sys
import subprocess
import time
from pathlib import Path
import argparse

def print_banner():
    """Print welcome banner"""
    print("="*60)
    print("HMAY-TSF: Hybrid Multi-Scale Adaptive YOLO")
    print("UAV Traffic Object Detection - Quick Start")
    print("="*60)

def check_requirements():
    """Check if basic requirements are met"""
    print("🔍 Checking requirements...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        return False
    
    # Check if required files exist
    required_files = [
        'requirements.txt',
        'train_hmay_tsf.py',
        'hmay_tsf_model.py',
        'data_preparation.py'
    ]
    
    missing_files = [f for f in required_files if not Path(f).exists()]
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    
    # Check dataset
    dataset_path = Path('./dataset')
    if not dataset_path.exists():
        print("⚠️  Dataset directory not found")
        return False
    
    print("✅ Basic requirements check passed")
    return True

def install_dependencies():
    """Install required dependencies"""
    print("📦 Installing dependencies...")
    
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'], 
                      check=True, capture_output=True)
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def setup_dataset():
    """Setup dataset configuration"""
    print("📁 Setting up dataset...")
    
    try:
        from data_preparation import prepare_visdrone_dataset
        yaml_path = prepare_visdrone_dataset('./dataset')
        if yaml_path:
            print(f"✅ Dataset configured: {yaml_path}")
            return yaml_path
        else:
            print("❌ Failed to setup dataset")
            return None
    except Exception as e:
        print(f"❌ Dataset setup error: {e}")
        return None

def quick_training_demo(epochs=5, batch_size=8):
    """Run a quick training demo"""
    print(f"🚀 Starting quick training demo ({epochs} epochs)...")
    
    cmd = [
        sys.executable, 'train_hmay_tsf.py',
        '--epochs', str(epochs),
        '--batch-size', str(batch_size),
        '--model-size', 'n',  # Use nano model for speed
        '--data', './dataset'
    ]
    
    try:
        print("Command:", ' '.join(cmd))
        result = subprocess.run(cmd, check=True, capture_output=False)
        print("✅ Quick training demo completed!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed: {e}")
        return False

def run_evaluation_demo():
    """Run evaluation on a pre-trained model or recent training"""
    print("📊 Running evaluation demo...")
    
    # Look for recent training results
    runs_dir = Path('./runs/train')
    if runs_dir.exists():
        recent_runs = sorted(runs_dir.glob('hmay_tsf_*'), 
                           key=lambda x: x.stat().st_mtime, reverse=True)
        
        if recent_runs:
            best_weights = recent_runs[0] / 'weights' / 'best.pt'
            if best_weights.exists():
                cmd = [
                    sys.executable, 'evaluate_model.py',
                    '--model', str(best_weights),
                    '--data', './dataset/dataset.yaml'
                ]
                
                try:
                    subprocess.run(cmd, check=True)
                    print("✅ Evaluation demo completed!")
                    return True
                except subprocess.CalledProcessError as e:
                    print(f"❌ Evaluation failed: {e}")
    
    print("⚠️  No trained models found for evaluation")
    return False

def run_prediction_demo():
    """Run prediction demo on test images"""
    print("🔮 Running prediction demo...")
    
    test_images = Path('./dataset/images/test')
    if not test_images.exists():
        print("⚠️  Test images not found")
        return False
    
    # Look for trained model
    runs_dir = Path('./runs/train')
    if runs_dir.exists():
        recent_runs = sorted(runs_dir.glob('hmay_tsf_*'), 
                           key=lambda x: x.stat().st_mtime, reverse=True)
        
        if recent_runs:
            best_weights = recent_runs[0] / 'weights' / 'best.pt'
            if best_weights.exists():
                cmd = [
                    sys.executable, 'train_hmay_tsf.py',
                    '--predict', str(test_images),
                    '--weights', str(best_weights)
                ]
                
                try:
                    subprocess.run(cmd, check=True)
                    print("✅ Prediction demo completed!")
                    print(f"📁 Results saved in ./runs/predict/")
                    return True
                except subprocess.CalledProcessError as e:
                    print(f"❌ Prediction failed: {e}")
    
    print("⚠️  No trained models found for prediction")
    return False

def show_dataset_info():
    """Show dataset information"""
    print("📈 Dataset Information:")
    
    dataset_path = Path('./dataset')
    if not dataset_path.exists():
        print("❌ Dataset not found")
        return
    
    # Count files in each split
    splits = ['train', 'val', 'test']
    for split in splits:
        img_dir = dataset_path / 'images' / split
        label_dir = dataset_path / 'labels' / split
        
        if img_dir.exists() and label_dir.exists():
            img_count = len(list(img_dir.glob('*.jpg')))
            label_count = len(list(label_dir.glob('*.txt')))
            print(f"  {split.upper()}: {img_count} images, {label_count} labels")
        else:
            print(f"  {split.upper()}: Not found")

def main():
    parser = argparse.ArgumentParser(description='HMAY-TSF Quick Start')
    parser.add_argument('--setup-only', action='store_true', help='Only setup, no training')
    parser.add_argument('--demo-epochs', type=int, default=5, help='Epochs for demo training')
    parser.add_argument('--demo-batch-size', type=int, default=8, help='Batch size for demo')
    parser.add_argument('--skip-training', action='store_true', help='Skip training demo')
    parser.add_argument('--full-training', action='store_true', help='Run full training (100 epochs)')
    
    args = parser.parse_args()
    
    print_banner()
    
    # Step 1: Check requirements
    if not check_requirements():
        print("❌ Requirements check failed. Please fix issues and try again.")
        return
    
    # Step 2: Install dependencies
    if not install_dependencies():
        print("❌ Failed to install dependencies")
        return
    
    # Step 3: Setup dataset
    yaml_path = setup_dataset()
    if not yaml_path:
        print("❌ Failed to setup dataset")
        return
    
    # Show dataset info
    show_dataset_info()
    
    if args.setup_only:
        print("✅ Setup completed! You can now run training manually.")
        return
    
    # Step 4: Training
    if not args.skip_training:
        if args.full_training:
            print("🚀 Starting full training (this may take several hours)...")
            cmd = [sys.executable, 'train_hmay_tsf.py', '--epochs', '100']
            subprocess.run(cmd)
        else:
            # Quick demo training
            quick_training_demo(args.demo_epochs, args.demo_batch_size)
    
    # Step 5: Evaluation
    run_evaluation_demo()
    
    # Step 6: Prediction demo
    run_prediction_demo()
    
    print("\n" + "="*60)
    print("🎉 Quick start completed!")
    print("\nNext steps:")
    print("1. Review results in ./runs/ directory")
    print("2. Run full training: python train_hmay_tsf.py --epochs 100")
    print("3. Evaluate model: python evaluate_model.py --model <weights_path>")
    print("4. Check README.md for advanced usage")
    print("="*60)

if __name__ == "__main__":
    main() 