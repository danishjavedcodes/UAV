#!/usr/bin/env python3
"""
Test script to verify HMAY-TSF model initialization
"""

import torch
import sys
import os

def test_model_initialization():
    """Test if the HMAY-TSF model can be initialized"""
    print("Testing HMAY-TSF model initialization...")
    
    try:
        # Import the model
        from hmay_tsf_model_simple import HMAY_TSF
        
        print("✓ Model imported successfully")
        
        # Test model initialization
        model = HMAY_TSF(
            model_size='s',
            num_classes=11,
            pretrained=False  # Don't download pretrained weights for testing
        )
        
        print("✓ Model initialized successfully")
        
        # Test forward pass
        dummy_input = torch.randn(1, 3, 640, 640)
        
        with torch.no_grad():
            output = model(dummy_input, is_training=False)
            print("✓ Forward pass successful")
            print(f"  Output type: {type(output)}")
            if isinstance(output, list):
                print(f"  Number of outputs: {len(output)}")
                for i, out in enumerate(output):
                    print(f"  Output {i} shape: {out.shape}")
        
        # Print model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"✓ Model statistics:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        
        print("\n🎉 Model test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_training_components():
    """Test training components"""
    print("\nTesting training components...")
    
    try:
        from train_hmay_tsf import HMAYTSFTrainer, FocalLoss, BalancedObjectDetectionLoss
        
        print("✓ Training components imported successfully")
        
        # Test loss functions
        focal_loss = FocalLoss(alpha=1, gamma=2)
        print("✓ FocalLoss initialized")
        
        balanced_loss = BalancedObjectDetectionLoss(num_classes=11)
        print("✓ BalancedObjectDetectionLoss initialized")
        
        # Test with dummy data
        dummy_pred = torch.randn(10, 11)
        dummy_target = torch.randint(0, 11, (10,))
        
        focal_output = focal_loss(dummy_pred, dummy_target)
        print(f"✓ FocalLoss forward pass: {focal_output.item():.4f}")
        
        print("\n🎉 Training components test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Training components test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("="*60)
    print("HMAY-TSF MODEL TEST")
    print("="*60)
    
    # Test model initialization
    model_ok = test_model_initialization()
    
    # Test training components
    training_ok = test_training_components()
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    if model_ok and training_ok:
        print("🎉 All tests passed! The model is ready for training.")
        print("\nYou can now run:")
        print("  python train_hmay_tsf.py")
        print("  python quick_start.py")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        print("\nCommon issues:")
        print("  - Missing dependencies (run: pip install -r requirements.txt)")
        print("  - CUDA/GPU issues (try running on CPU)")
        print("  - Import errors (check file paths)")

if __name__ == "__main__":
    main() 