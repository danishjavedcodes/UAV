#!/usr/bin/env python3
"""
Test script for simplified HMAY-TSF model
"""

import torch
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from hmay_tsf_model import HMAY_TSF

def test_model():
    """Test the simplified model"""
    print("🧪 Testing simplified HMAY-TSF model...")
    
    # Create model
    model = HMAY_TSF(
        model_size='n',
        num_classes=4,
        pretrained=True,
        use_yolov11=False
    )
    
    print(f"✅ Model created successfully!")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Test forward pass
    try:
        # Create dummy input
        batch_size = 2
        channels = 3
        height = 640
        width = 640
        
        dummy_input = torch.randn(batch_size, channels, height, width)
        print(f"Input shape: {dummy_input.shape}")
        
        # Forward pass
        with torch.no_grad():
            output = model(dummy_input)
            print(f"Output shape: {output.shape}")
            print(f"✅ Forward pass successful!")
            
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test model summary
    try:
        print("\n📊 Model Summary:")
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Frozen parameters: {frozen_params:,}")
        print(f"  Freeze ratio: {frozen_params/total_params*100:.1f}%")
        
    except Exception as e:
        print(f"❌ Model summary failed: {e}")
        return False
    
    print("✅ Model test completed successfully!")
    return True

if __name__ == "__main__":
    test_model() 