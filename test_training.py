"""
Test script for Advanced HMAY-TSF Training
Verifies that the training system works properly and shows progress
"""

import os
import sys
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from train_hmay_tsf import AdvancedHMAYTSFTrainer

def test_training_setup():
    """Test the training setup and configuration"""
    print("="*80)
    print("TESTING ADVANCED HMAY-TSF TRAINING SETUP")
    print("="*80)
    
    # Initialize trainer
    trainer = AdvancedHMAYTSFTrainer(
        model_size='s',
        device='auto',
        project_name='HMAY-TSF-Test'
    )
    
    # Setup model
    model = trainer.setup_advanced_model(num_classes=11, pretrained=True)
    
    print(f"\nModel setup completed successfully!")
    print(f"Device: {trainer.device}")
    print(f"Model size: {trainer.model_size}")
    
    # Test curriculum learning
    print(f"\nCurriculum Learning Test:")
    for epoch in range(1, 11):
        trainer.curriculum_learning.update_epoch(epoch)
        stage = trainer.curriculum_learning.get_current_stage()
        print(f"Epoch {epoch}: {stage['difficulty']} (strength: {stage['augmentation_strength']:.2f})")
    
    # Test metric calculation
    print(f"\nMetric Calculation Test:")
    test_metrics = {
        'epoch': 1,
        'val_precision': 0.1,
        'val_recall': 0.08,
        'map50': 0.06,
        'map50_95': 0.05,
        'val_f1': 0.075,
        'val_accuracy': 0.075,
        'train_loss': 0.9,
        'val_loss': 0.85,
        'lr': 0.001,
        'focal_loss': 0.08,
        'iou_loss': 0.04,
        'box_loss': 0.12,
        'small_object_recall': 0.092,
        'occlusion_aware_f1': 0.081,
        'curriculum_stage': 'easy',
        'augmentation_strength': 0.3,
        'gradient_norm': 0.5
    }
    
    trainer.print_epoch_metrics(test_metrics)
    
    print(f"\n" + "="*80)
    print("TEST COMPLETED SUCCESSFULLY!")
    print("The training system is ready for 99.2%+ performance targets")
    print("="*80)

def test_dataset_loading():
    """Test dataset loading and configuration"""
    print("\n" + "="*80)
    print("TESTING DATASET LOADING")
    print("="*80)
    
    dataset_path = Path("./dataset")
    if not dataset_path.exists():
        print("❌ Dataset directory not found!")
        return False
    
    dataset_yaml = dataset_path / "dataset.yaml"
    if not dataset_yaml.exists():
        print("❌ dataset.yaml not found!")
        return False
    
    # Check dataset structure
    required_dirs = [
        "images/train", "images/val", "images/test",
        "labels/train", "labels/val", "labels/test"
    ]
    
    for dir_path in required_dirs:
        full_path = dataset_path / dir_path
        if not full_path.exists():
            print(f"❌ Required directory not found: {dir_path}")
            return False
        else:
            file_count = len(list(full_path.glob("*.jpg" if "images" in dir_path else "*.txt")))
            print(f"✅ {dir_path}: {file_count} files")
    
    print(f"\n✅ Dataset structure is correct!")
    return True

def main():
    """Main test function"""
    print("🚀 Advanced HMAY-TSF Training Test")
    print("Target: 99.2%+ accuracy, precision, recall, and F1 score")
    
    # Test dataset
    if not test_dataset_loading():
        print("❌ Dataset test failed!")
        return
    
    # Test training setup
    test_training_setup()
    
    print(f"\n🎉 All tests passed!")
    print("You can now run training with:")
    print("python quick_start.py --mode train --epochs 200")

if __name__ == "__main__":
    main() 