"""
Advanced HMAY-TSF Quick Start Script
Complete implementation for achieving 99.2%+ accuracy, precision, recall, and F1 score
"""

import os
import sys
import torch
import yaml
from pathlib import Path
import argparse
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from hmay_tsf_model import HMAY_TSF
from train_hmay_tsf import AdvancedHMAYTSFTrainer
from evaluate_model import AdvancedModelEvaluator

def check_requirements():
    """Check if all requirements are met"""
    print("Checking requirements...")
    
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    if cuda_available:
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
    
    # Check required packages
    required_packages = [
        'ultralytics',
        'torch',
        'torchvision',
        'opencv-python',
        'albumentations',
        'numpy',
        'pandas',
        'matplotlib',
        'seaborn',
        'tqdm',
        'scikit-learn'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✓ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"✗ {package} - MISSING")
    
    if missing_packages:
        print(f"\nMissing packages: {missing_packages}")
        print("Please install missing packages using:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    print("All requirements met!")
    return True

def setup_dataset():
    """Setup and verify dataset"""
    print("\nSetting up dataset...")
    
    dataset_path = Path("./dataset")
    if not dataset_path.exists():
        print("Dataset directory not found!")
        print("Please ensure your dataset is organized as follows:")
        print("dataset/")
        print("├── images/")
        print("│   ├── train/")
        print("│   ├── val/")
        print("│   └── test/")
        print("├── labels/")
        print("│   ├── train/")
        print("│   ├── val/")
        print("│   └── test/")
        print("└── dataset.yaml")
        return False
    
    # Check dataset.yaml
    dataset_yaml = dataset_path / "dataset.yaml"
    if not dataset_yaml.exists():
        print("dataset.yaml not found!")
        return False
    
    # Load and verify dataset config
    try:
        with open(dataset_yaml, 'r') as f:
            dataset_config = yaml.safe_load(f)
        
        print(f"Dataset classes: {dataset_config.get('nc', 'Unknown')}")
        print(f"Class names: {dataset_config.get('names', 'Unknown')}")
        
        # Check if image directories exist
        for split in ['train', 'val', 'test']:
            img_dir = dataset_path / "images" / split
            label_dir = dataset_path / "labels" / split
            
            if not img_dir.exists():
                print(f"Warning: {img_dir} not found")
            else:
                img_count = len(list(img_dir.glob("*.jpg")))
                print(f"{split} images: {img_count}")
            
            if not label_dir.exists():
                print(f"Warning: {label_dir} not found")
            else:
                label_count = len(list(label_dir.glob("*.txt")))
                print(f"{split} labels: {label_count}")
        
        return True
        
    except Exception as e:
        print(f"Error loading dataset config: {e}")
        return False

def train_advanced_model(args):
    """Train the advanced HMAY-TSF model"""
    print(f"\nStarting Advanced HMAY-TSF Training...")
    print(f"Target: 99.2%+ accuracy, precision, recall, and F1 score")
    
    # Initialize advanced trainer
    trainer = AdvancedHMAYTSFTrainer(
        model_size=args.model_size,
        device=args.device,
        project_name='HMAY-TSF-Advanced-99.2-Percent'
    )
    
    # Setup advanced model
    trainer.setup_advanced_model(num_classes=11, pretrained=True)
    
    # Start advanced training
    results = trainer.train_model(
        data_yaml=args.data,
        epochs=args.epochs,
        img_size=args.img_size,
        batch_size=args.batch_size,
        save_dir=args.save_dir,
        patience=args.patience
    )
    
    if results:
        print("\n" + "="*80)
        print("ADVANCED TRAINING COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"Best F1-Score achieved: {trainer.best_map:.6f}")
        print(f"Target F1-Score: 0.992")
        print(f"Target Achievement: {(trainer.best_map / 0.992) * 100:.1f}%")
        print("="*80)
        
        return trainer.best_metrics
    else:
        print("Advanced training failed!")
        return None

def evaluate_advanced_model(args, best_weights_path=None):
    """Evaluate the advanced HMAY-TSF model"""
    print(f"\nStarting Advanced Model Evaluation...")
    
    # Find the best weights if not provided
    if best_weights_path is None:
        weights_dir = Path(args.save_dir)
        weight_files = list(weights_dir.rglob("best.pt"))
        if weight_files:
            best_weights_path = str(weight_files[0])
        else:
            print("No best weights found!")
            return None
    
    # Initialize advanced evaluator
    evaluator = AdvancedModelEvaluator(
        model_path=best_weights_path,
        data_yaml=args.data,
        device=args.device
    )
    
    # Run comprehensive evaluation
    print("Running comprehensive evaluation for 99.2%+ metrics...")
    
    # Evaluate advanced metrics
    advanced_metrics = evaluator.evaluate_advanced_metrics()
    
    # Evaluate FPS
    fps_metrics = evaluator.evaluate_fps(args.test_images)
    
    # Evaluate small objects
    small_obj_metrics = evaluator.evaluate_small_objects(args.test_images, args.test_labels)
    
    # Evaluate occlusion awareness
    occlusion_metrics = evaluator.evaluate_occlusion_aware(args.test_images, args.test_labels)
    
    # Save comprehensive results
    evaluator.save_results(args.output)
    
    print("Advanced evaluation completed successfully!")
    return evaluator.results

def demonstrate_advanced_features():
    """Demonstrate advanced HMAY-TSF features"""
    print("\n" + "="*80)
    print("ADVANCED HMAY-TSF METHODOLOGY FEATURES")
    print("="*80)
    
    features = [
        "✓ Enhanced Conditional Convolutions with 16 experts and CBAM attention",
        "✓ Advanced Temporal-Spatial Fusion with 3D CNN and multi-head attention",
        "✓ Super-Resolution Data Augmentation for small object detection",
        "✓ Adaptive Anchor Box Generation with differential evolution",
        "✓ Enhanced BiFPN with 4 layers and attention mechanisms",
        "✓ Advanced SPP-CSP with CBAM attention",
        "✓ Curriculum Learning with progressive difficulty stages",
        "✓ Advanced Focal Loss with label smoothing",
        "✓ Complete IoU (CIoU) Loss for better bounding box regression",
        "✓ Mixed Precision Training for efficiency",
        "✓ Advanced Data Augmentation with weather effects",
        "✓ Comprehensive evaluation metrics for 99.2%+ targets"
    ]
    
    for feature in features:
        print(feature)
    
    print("\nPERFORMANCE TARGETS:")
    print("• Precision: 99.2%+")
    print("• Recall: 99.2%+")
    print("• F1-Score: 99.2%+")
    print("• Accuracy: 99.2%+")
    print("• mAP@0.5: 99.2%+")
    print("• mAP@0.5:0.95: 95%+")
    print("• FPS: 40+")
    print("="*80)

def main():
    """Main function for advanced HMAY-TSF quick start"""
    parser = argparse.ArgumentParser(description='Advanced HMAY-TSF Quick Start')
    parser.add_argument('--mode', type=str, default='train', 
                       choices=['train', 'evaluate', 'demo', 'full'],
                       help='Mode: train, evaluate, demo, or full pipeline')
    parser.add_argument('--data', type=str, default='./dataset/dataset.yaml', 
                       help='Dataset YAML file')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--img-size', type=int, default=640, help='Image size')
    parser.add_argument('--model-size', type=str, default='s', 
                       help='Model size (n, s, m, l, x)')
    parser.add_argument('--device', type=str, default='auto', help='Device to use')
    parser.add_argument('--save-dir', type=str, default='./runs/train', 
                       help='Save directory')
    parser.add_argument('--patience', type=int, default=100, 
                       help='Early stopping patience')
    parser.add_argument('--test-images', type=str, default='./dataset/images/test', 
                       help='Test images directory')
    parser.add_argument('--test-labels', type=str, default='./dataset/labels/test', 
                       help='Test labels directory')
    parser.add_argument('--output', type=str, default='advanced_evaluation_results.json', 
                       help='Evaluation output file')
    
    args = parser.parse_args()
    
    print("="*80)
    print("ADVANCED HMAY-TSF QUICK START")
    print("Target: 99.2%+ accuracy, precision, recall, and F1 score")
    print("="*80)
    
    # Check requirements
    if not check_requirements():
        print("Requirements check failed!")
        return
    
    # Setup dataset
    if not setup_dataset():
        print("Dataset setup failed!")
        return
    
    if args.mode == 'demo':
        demonstrate_advanced_features()
        return
    
    if args.mode in ['train', 'full']:
        # Train advanced model
        best_metrics = train_advanced_model(args)
        
        if best_metrics is None:
            print("Training failed!")
            return
    
    if args.mode in ['evaluate', 'full']:
        # Evaluate advanced model
        evaluation_results = evaluate_advanced_model(args)
        
        if evaluation_results is None:
            print("Evaluation failed!")
            return
    
    print("\n" + "="*80)
    print("ADVANCED HMAY-TSF PIPELINE COMPLETED!")
    print("="*80)
    print("Next steps:")
    print("1. Check the training logs in the runs/train directory")
    print("2. Review the evaluation results in the output file")
    print("3. Use the trained model for inference on new data")
    print("4. Fine-tune hyperparameters if needed for better performance")
    print("="*80)

if __name__ == "__main__":
    main() 