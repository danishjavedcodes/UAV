"""
Quick Start Script for Enhanced HMAY-TSF
Achieve 99-99.8% accuracy, precision, recall, and F1 score
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

from train_hmay_tsf import EnhancedHMAYTSFTrainer
from evaluate_model import EnhancedModelEvaluator
from hmay_tsf_model import HMAY_TSF

def check_requirements():
    """Check if all requirements are met"""
    print("Checking requirements...")
    
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    if cuda_available:
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # Check dataset
    dataset_path = Path("./dataset")
    if not dataset_path.exists():
        print("❌ Dataset not found! Please ensure the dataset is in ./dataset/")
        return False
    
    # Check dataset structure
    required_dirs = ["images/train", "images/val", "images/test", "labels/train", "labels/val", "labels/test"]
    for dir_path in required_dirs:
        if not (dataset_path / dir_path).exists():
            print(f"❌ Required directory not found: {dir_path}")
            return False
    
    print("✅ All requirements met!")
    return True

def setup_environment():
    """Setup the environment for enhanced training"""
    print("Setting up enhanced environment...")
    
    # Create necessary directories
    directories = [
        "./runs/train",
        "./runs/predict", 
        "./logs",
        "./checkpoints",
        "./results"
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {dir_path}")
    
    # Load configuration
    config_path = Path("./config.yaml")
    if not config_path.exists():
        print("❌ config.yaml not found!")
        return None

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("✅ Environment setup complete!")
    return config

def run_enhanced_training(config, args):
    """Run enhanced training to achieve 99%+ metrics"""
    print("\n" + "="*80)
    print("STARTING ENHANCED HMAY-TSF TRAINING")
    print("Target: 99-99.8% accuracy, precision, recall, and F1 score")
    print("="*80)
    
    # Initialize enhanced trainer
    trainer = EnhancedHMAYTSFTrainer(
        model_size=args.model_size,
        device=args.device,
        project_name='HMAY-TSF-Enhanced-99-Percent'
    )
    
    # Setup enhanced model
    print("Setting up enhanced model...")
    trainer.setup_enhanced_model(
        num_classes=config['model']['num_classes'],
        pretrained=config['model']['pretrained']
    )
    
    # Start enhanced training
    print("Starting enhanced training...")
    results = trainer.train_model(
        data_yaml=args.data,
        epochs=config['training']['epochs'],
        img_size=config['training']['img_size'],
        batch_size=config['training']['batch_size'],
        save_dir=args.save_dir,
        patience=config['training']['patience']
    )
    
    if results:
        print("\n✅ Enhanced training completed successfully!")
        print(f"Best F1-Score achieved: {trainer.best_map:.6f}")
        
        # Save best model path
        best_model_path = Path(args.save_dir) / f"enhanced_hmay_tsf_{args.model_size}_{datetime.now().strftime('%Y%m%d_%H%M%S')}" / "weights" / "best.pt"
        if best_model_path.exists():
            print(f"Best model saved to: {best_model_path}")
            return str(best_model_path)
        else:
            print("⚠️ Best model not found, using last model")
            last_model_path = Path(args.save_dir) / f"enhanced_hmay_tsf_{args.model_size}_{datetime.now().strftime('%Y%m%d_%H%M%S')}" / "weights" / "last.pt"
            return str(last_model_path) if last_model_path.exists() else None
    else:
        print("❌ Enhanced training failed!")
        return None

def run_enhanced_evaluation(model_path, config, args):
    """Run enhanced evaluation to verify 99%+ metrics"""
    print("\n" + "="*80)
    print("RUNNING ENHANCED EVALUATION")
    print("Verifying 99-99.8% accuracy, precision, recall, and F1 score")
    print("="*80)
    
    if not model_path or not Path(model_path).exists():
        print("❌ Model not found for evaluation!")
        return None
    
    # Initialize enhanced evaluator
    evaluator = EnhancedModelEvaluator(
        model_path=model_path,
        data_yaml=args.data,
        device=args.device
    )
    
    # Run comprehensive evaluation
    print("Running comprehensive evaluation...")
    
    # Evaluate enhanced metrics
    enhanced_metrics = evaluator.evaluate_enhanced_metrics()
    
    # Evaluate FPS
    fps_metrics = evaluator.evaluate_fps(args.test_images)
    
    # Evaluate small objects
    small_obj_metrics = evaluator.evaluate_small_objects(args.test_images, args.test_labels)
    
    # Evaluate occlusion awareness
    occlusion_metrics = evaluator.evaluate_occlusion_aware(args.test_images, args.test_labels)
    
    # Save comprehensive results
    output_path = f"./results/enhanced_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    evaluator.save_results(output_path)
    
    print("✅ Enhanced evaluation completed!")
    return enhanced_metrics

def print_performance_summary(metrics):
    """Print performance summary"""
    print("\n" + "="*80)
    print("PERFORMANCE SUMMARY")
    print("="*80)
    
    if metrics:
        print(f"Precision: {metrics.get('precision', 0):.6f} (Target: 0.99)")
        print(f"Recall: {metrics.get('recall', 0):.6f} (Target: 0.99)")
        print(f"F1-Score: {metrics.get('f1_score', 0):.6f} (Target: 0.99)")
        print(f"Accuracy: {metrics.get('accuracy', 0):.6f} (Target: 0.99)")
        print(f"mAP@0.5: {metrics.get('mAP50', 0):.6f} (Target: 0.99)")
        print(f"mAP@0.5:0.95: {metrics.get('mAP50-95', 0):.6f} (Target: 0.95)")
        
        # Check if targets are met
        targets_met = 0
        total_targets = 6
        
        if metrics.get('precision', 0) >= 0.99:
            targets_met += 1
            print("✅ Precision target met")
        else:
            print("❌ Precision target not met")
            
        if metrics.get('recall', 0) >= 0.99:
            targets_met += 1
            print("✅ Recall target met")
        else:
            print("❌ Recall target not met")
            
        if metrics.get('f1_score', 0) >= 0.99:
            targets_met += 1
            print("✅ F1-Score target met")
        else:
            print("❌ F1-Score target not met")
            
        if metrics.get('accuracy', 0) >= 0.99:
            targets_met += 1
            print("✅ Accuracy target met")
        else:
            print("❌ Accuracy target not met")
            
        if metrics.get('mAP50', 0) >= 0.99:
            targets_met += 1
            print("✅ mAP@0.5 target met")
        else:
            print("❌ mAP@0.5 target not met")
            
        if metrics.get('mAP50-95', 0) >= 0.95:
            targets_met += 1
            print("✅ mAP@0.5:0.95 target met")
        else:
            print("❌ mAP@0.5:0.95 target not met")
        
        print(f"\nOverall: {targets_met}/{total_targets} targets met")
        
        if targets_met == total_targets:
            print("🎉 ALL TARGETS ACHIEVED! 99%+ performance confirmed!")
        elif targets_met >= total_targets * 0.8:
            print("👍 Most targets achieved! Performance is excellent!")
        else:
            print("⚠️ Some targets not met. Consider additional training or hyperparameter tuning.")
    
    print("="*80)

def main():
    """Main function for quick start"""
    parser = argparse.ArgumentParser(description='Enhanced HMAY-TSF Quick Start')
    parser.add_argument('--mode', type=str, default='full', choices=['train', 'evaluate', 'full'], 
                       help='Mode: train, evaluate, or full (train + evaluate)')
    parser.add_argument('--data', type=str, default='./dataset/dataset.yaml', help='Dataset YAML file')
    parser.add_argument('--model-size', type=str, default='s', choices=['n', 's', 'm', 'l', 'x'], 
                       help='Model size')
    parser.add_argument('--device', type=str, default='auto', help='Device to use')
    parser.add_argument('--save-dir', type=str, default='./runs/train', help='Save directory')
    parser.add_argument('--test-images', type=str, default='./dataset/images/test', help='Test images directory')
    parser.add_argument('--test-labels', type=str, default='./dataset/labels/test', help='Test labels directory')
    parser.add_argument('--model-path', type=str, help='Path to model for evaluation (required for evaluate mode)')
    
    args = parser.parse_args()
    
    print("🚀 Enhanced HMAY-TSF Quick Start")
    print("Target: 99-99.8% accuracy, precision, recall, and F1 score")
    print("="*80)
    
    # Check requirements
    if not check_requirements():
        print("❌ Requirements not met. Please fix the issues above.")
        return
    
    # Setup environment
    config = setup_environment()
    if not config:
        print("❌ Environment setup failed!")
        return
    
    # Run based on mode
    model_path = None
    metrics = None
    
    if args.mode in ['train', 'full']:
        print(f"\n🎯 Mode: Training (Target: 99%+ metrics)")
        model_path = run_enhanced_training(config, args)
    
    if args.mode in ['evaluate', 'full']:
        print(f"\n🔍 Mode: Evaluation (Verifying 99%+ metrics)")
        
        # Use provided model path or the one from training
        eval_model_path = args.model_path if args.model_path else model_path
        
        if not eval_model_path:
            print("❌ No model path provided for evaluation!")
        return
    
        metrics = run_enhanced_evaluation(eval_model_path, config, args)
    
    # Print performance summary
    print_performance_summary(metrics)
    
    print("\n🎉 Enhanced HMAY-TSF Quick Start completed!")
    print("Check the results directory for detailed evaluation reports.")

if __name__ == "__main__":
    main() 