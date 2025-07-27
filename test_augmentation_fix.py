#!/usr/bin/env python3
"""
Test script to verify augmentation fixes work without warnings
"""

import os
import sys
import warnings
import numpy as np
import cv2

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_augmentation_without_warnings():
    """Test augmentation without warnings"""
    print("Testing augmentation fixes...")
    
    try:
        from train_hmay_tsf import DatasetBalancer
        
        # Create a dummy balancer
        balancer = DatasetBalancer(dataset_path='./dataset')
        
        # Create dummy image and labels
        dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        dummy_labels = [[0, 0.5, 0.5, 0.1, 0.1], [1, 0.3, 0.3, 0.2, 0.2]]
        
        # Capture warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            # Test augmentation
            result = balancer._apply_balancing_augmentation(dummy_image, dummy_labels, 0)
            
            # Check for warnings
            if len(w) > 0:
                print(f"⚠️  Found {len(w)} warnings:")
                for warning in w:
                    print(f"   - {warning.message}")
                return False
            else:
                print("✅ No warnings detected!")
                return True
                
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def test_albumentations_version():
    """Test which albumentations version is being used"""
    try:
        import albumentations as A
        print(f"📦 Albumentations version: {A.__version__}")
        
        # Test simple, reliable transforms
        transform = A.Compose([
            A.RandomBrightnessContrast(p=0.5),
            A.HueSaturationValue(p=0.5),
            A.HorizontalFlip(p=0.3),
            A.GaussNoise(var_limit=3.0, p=0.3),
            A.GaussianBlur(blur_limit=(3, 3), p=0.2),
        ])
        
        print("✅ Albumentations transforms created successfully")
        return True
        
    except Exception as e:
        print(f"❌ Albumentations test failed: {e}")
        return False

def main():
    """Main test function"""
    print("="*60)
    print("TESTING AUGMENTATION FIXES")
    print("="*60)
    
    # Test albumentations version
    print("\n1. Testing Albumentations Version:")
    albu_test = test_albumentations_version()
    
    # Test augmentation without warnings
    print("\n2. Testing Augmentation Without Warnings:")
    aug_test = test_augmentation_without_warnings()
    
    # Summary
    print("\n" + "="*60)
    if albu_test and aug_test:
        print("🎉 All tests passed! Augmentation should work without warnings.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
    print("="*60)
    
    return 0 if (albu_test and aug_test) else 1

if __name__ == "__main__":
    exit(main()) 