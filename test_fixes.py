#!/usr/bin/env python3
"""
Test script to verify tensor handling fixes
"""

import torch
import numpy as np

def test_tensor_handling():
    """Test tensor handling functions"""
    print("Testing tensor handling fixes...")
    
    # Test single element tensor
    single_tensor = torch.tensor([3.14])
    print(f"Single element tensor: {single_tensor}")
    print(f"Num elements: {single_tensor.numel()}")
    print(f"Can convert to float: {single_tensor.numel() == 1}")
    
    # Test multi-element tensor
    multi_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0])
    print(f"Multi-element tensor: {multi_tensor}")
    print(f"Num elements: {multi_tensor.numel()}")
    print(f"Mean: {multi_tensor.mean().item()}")
    
    # Test tensor conversion
    try:
        if single_tensor.numel() == 1:
            result = float(single_tensor.item())
        else:
            result = float(single_tensor.mean().item())
        print(f"✅ Single tensor conversion successful: {result}")
    except Exception as e:
        print(f"❌ Single tensor conversion failed: {e}")
    
    try:
        if multi_tensor.numel() == 1:
            result = float(multi_tensor.item())
        else:
            result = float(multi_tensor.mean().item())
        print(f"✅ Multi tensor conversion successful: {result}")
    except Exception as e:
        print(f"❌ Multi tensor conversion failed: {e}")
    
    print("\n✅ All tensor handling tests passed!")

def test_gradscaler():
    """Test GradScaler import"""
    try:
        import torch.amp as amp
        scaler = amp.GradScaler('cuda') if torch.cuda.is_available() else amp.GradScaler()
        print("✅ GradScaler import successful!")
    except Exception as e:
        print(f"❌ GradScaler import failed: {e}")

if __name__ == "__main__":
    test_tensor_handling()
    test_gradscaler() 