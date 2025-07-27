#!/usr/bin/env python3
"""
Debug script to check dataset structure
"""

import os
import glob
from pathlib import Path

def check_dataset_structure():
    """Check the dataset structure and identify issues"""
    print("="*60)
    print("DATASET STRUCTURE DEBUG")
    print("="*60)
    
    # Check current directory
    current_dir = os.getcwd()
    print(f"Current directory: {current_dir}")
    
    # Check if dataset directory exists
    dataset_paths = [
        './dataset',
        '../dataset',
        'dataset',
        '/kaggle/working/UAV/dataset',
        '/kaggle/working/dataset'
    ]
    
    for path in dataset_paths:
        if os.path.exists(path):
            print(f"✓ Found dataset at: {path}")
            dataset_path = path
            break
    else:
        print("❌ No dataset directory found!")
        return False
    
    # Check dataset structure
    print(f"\nChecking structure of: {dataset_path}")
    
    # Check for images
    img_paths = [
        os.path.join(dataset_path, 'images'),
        os.path.join(dataset_path, 'train'),
        os.path.join(dataset_path, 'images/train'),
    ]
    
    for img_path in img_paths:
        if os.path.exists(img_path):
            print(f"✓ Found images at: {img_path}")
            img_files = glob.glob(os.path.join(img_path, '*.jpg'))
            print(f"  - {len(img_files)} image files")
            break
    else:
        print("❌ No image directory found!")
    
    # Check for labels
    label_paths = [
        os.path.join(dataset_path, 'labels'),
        os.path.join(dataset_path, 'labels/train'),
        os.path.join(dataset_path, 'train/labels'),
    ]
    
    for label_path in label_paths:
        if os.path.exists(label_path):
            print(f"✓ Found labels at: {label_path}")
            label_files = glob.glob(os.path.join(label_path, '*.txt'))
            print(f"  - {len(label_files)} label files")
            
            # Check first few label files
            if label_files:
                print("\nChecking first 3 label files:")
                for i, label_file in enumerate(label_files[:3]):
                    try:
                        with open(label_file, 'r') as f:
                            lines = f.readlines()
                            print(f"  {label_file}: {len(lines)} lines")
                            if lines:
                                print(f"    First line: {lines[0].strip()}")
                    except Exception as e:
                        print(f"  Error reading {label_file}: {e}")
            break
    else:
        print("❌ No label directory found!")
    
    # Check dataset.yaml
    yaml_paths = [
        os.path.join(dataset_path, 'dataset.yaml'),
        'dataset/dataset.yaml',
        'dataset.yaml'
    ]
    
    for yaml_path in yaml_paths:
        if os.path.exists(yaml_path):
            print(f"✓ Found dataset.yaml at: {yaml_path}")
            try:
                import yaml
                with open(yaml_path, 'r') as f:
                    config = yaml.safe_load(f)
                print(f"  - Classes: {config.get('nc', 'N/A')}")
                print(f"  - Path: {config.get('path', 'N/A')}")
                print(f"  - Train: {config.get('train', 'N/A')}")
                print(f"  - Val: {config.get('val', 'N/A')}")
            except Exception as e:
                print(f"  Error reading yaml: {e}")
            break
    else:
        print("❌ No dataset.yaml found!")
    
    return True

def test_class_analysis():
    """Test the class analysis functionality"""
    print("\n" + "="*60)
    print("TESTING CLASS ANALYSIS")
    print("="*60)
    
    try:
        from data_preparation import ClassBalancedAugmentation
        
        # Try different paths
        test_paths = [
            './dataset',
            '../dataset',
            'dataset',
            '/kaggle/working/UAV/dataset',
            '/kaggle/working/dataset'
        ]
        
        for path in test_paths:
            if os.path.exists(path):
                print(f"Testing with path: {path}")
                try:
                    analyzer = ClassBalancedAugmentation(dataset_path=path)
                    print("✓ Class analysis successful!")
                    return True
                except Exception as e:
                    print(f"❌ Error with {path}: {e}")
                    continue
        
        print("❌ All paths failed!")
        return False
        
    except Exception as e:
        print(f"❌ Error importing or running class analysis: {e}")
        return False

if __name__ == "__main__":
    print("Starting dataset debug...")
    
    # Check structure
    structure_ok = check_dataset_structure()
    
    # Test class analysis
    if structure_ok:
        analysis_ok = test_class_analysis()
    else:
        analysis_ok = False
    
    print("\n" + "="*60)
    print("DEBUG SUMMARY")
    print("="*60)
    
    if structure_ok and analysis_ok:
        print("🎉 Dataset structure and analysis working correctly!")
    else:
        print("❌ Issues found with dataset structure or analysis")
        print("\nCommon solutions:")
        print("1. Check if dataset path is correct")
        print("2. Ensure label files are in .txt format")
        print("3. Verify dataset.yaml configuration")
        print("4. Check file permissions") 