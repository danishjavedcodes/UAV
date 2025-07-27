#!/usr/bin/env python3
"""
Test script for dataset balancing functionality
"""

import os
import sys
from pathlib import Path
import tempfile
import shutil
import yaml

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from train_hmay_tsf import DatasetBalancer, run_dataset_balancing_only

def create_test_dataset():
    """Create a small test dataset with imbalanced classes"""
    print("Creating test dataset with imbalanced classes...")
    
    # Create temporary test dataset
    test_dataset_path = Path("./test_dataset")
    test_dataset_path.mkdir(exist_ok=True)
    
    # Create directory structure
    (test_dataset_path / "images" / "train").mkdir(parents=True, exist_ok=True)
    (test_dataset_path / "images" / "val").mkdir(parents=True, exist_ok=True)
    (test_dataset_path / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (test_dataset_path / "labels" / "val").mkdir(parents=True, exist_ok=True)
    
    # Create dummy images and labels with imbalanced distribution
    # Class 0: 100 instances (most common)
    # Class 1: 50 instances
    # Class 2: 25 instances
    # Class 3: 10 instances (least common)
    
    class_distribution = {
        0: 100,  # ignored regions - most common
        1: 50,   # pedestrian
        2: 25,   # people
        3: 10,   # bicycle - least common
        4: 30,   # car
        5: 20,   # van
        6: 15,   # truck
        7: 12,   # tricycle
        8: 8,    # awning-tricycle
        9: 18,   # bus
        10: 35   # motor
    }
    
    # Create dummy images and labels
    image_counter = 0
    for class_id, count in class_distribution.items():
        for i in range(count):
            # Create dummy image file
            img_name = f"test_img_{image_counter:04d}.jpg"
            img_path = test_dataset_path / "images" / "train" / img_name
            img_path.touch()  # Create empty file
            
            # Create dummy label file
            label_name = f"test_img_{image_counter:04d}.txt"
            label_path = test_dataset_path / "labels" / "train" / label_name
            
            # Create YOLO format label: class_id x_center y_center width height
            with open(label_path, 'w') as f:
                f.write(f"{class_id} 0.5 0.5 0.1 0.1\n")
            
            image_counter += 1
    
    # Create some validation data
    for i in range(10):
        img_name = f"val_img_{i:04d}.jpg"
        img_path = test_dataset_path / "images" / "val" / img_name
        img_path.touch()
        
        label_name = f"val_img_{i:04d}.txt"
        label_path = test_dataset_path / "labels" / "val" / label_name
        with open(label_path, 'w') as f:
            f.write(f"{i % 11} 0.5 0.5 0.1 0.1\n")
    
    # Create dataset YAML
    yaml_config = {
        'path': str(test_dataset_path.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/val',  # Use val as test for simplicity
        'nc': 11,
        'names': {
            0: 'ignored regions', 1: 'pedestrian', 2: 'people', 3: 'bicycle',
            4: 'car', 5: 'van', 6: 'truck', 7: 'tricycle', 8: 'awning-tricycle',
            9: 'bus', 10: 'motor'
        }
    }
    
    yaml_path = test_dataset_path / 'dataset.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_config, f, default_flow_style=False)
    
    print(f"Test dataset created at: {test_dataset_path}")
    print(f"Class distribution: {class_distribution}")
    print(f"Total images: {image_counter}")
    
    return str(test_dataset_path)

def test_dataset_balancing():
    """Test the dataset balancing functionality"""
    print("="*80)
    print("TESTING DATASET BALANCING FUNCTIONALITY")
    print("="*80)
    
    # Create test dataset
    test_dataset_path = create_test_dataset()
    
    try:
        # Test 1: Analyze original distribution
        print("\n" + "="*50)
        print("TEST 1: Analyzing original distribution")
        print("="*50)
        
        balancer = DatasetBalancer(dataset_path=test_dataset_path)
        original_analysis = balancer.analyze_class_distribution('train')
        
        if original_analysis:
            print("✅ Original distribution analysis successful")
            print(f"   Total images: {original_analysis['total_images']}")
            print(f"   Total annotations: {original_analysis['total_annotations']}")
        else:
            print("❌ Original distribution analysis failed")
            return False
        
        # Test 2: Create balanced dataset
        print("\n" + "="*50)
        print("TEST 2: Creating balanced dataset")
        print("="*50)
        
        target_samples = 50  # Target 50 samples per class
        balanced_yaml = balancer.create_balanced_dataset(target_samples_per_class=target_samples)
        
        if balanced_yaml:
            print("✅ Balanced dataset creation successful")
            print(f"   Balanced dataset: {balanced_yaml}")
        else:
            print("❌ Balanced dataset creation failed")
            return False
        
        # Test 3: Verify balance
        print("\n" + "="*50)
        print("TEST 3: Verifying balance")
        print("="*50)
        
        # Analyze balanced dataset
        balanced_analysis = balancer.analyze_class_distribution('train')
        
        if balanced_analysis:
            class_counts = list(balanced_analysis['class_counts'].values())
            if class_counts:
                max_count = max(class_counts)
                min_count = min(class_counts)
                balance_ratio = max_count / min_count if min_count > 0 else float('inf')
                
                print(f"   Most common class: {max_count}")
                print(f"   Least common class: {min_count}")
                print(f"   Balance ratio: {balance_ratio:.2f}:1")
                
                if balance_ratio <= 2.0:
                    print("✅ Good balance achieved!")
                else:
                    print("⚠️  Balance could be improved")
        
        # Test 4: Plot distribution
        print("\n" + "="*50)
        print("TEST 4: Creating distribution plots")
        print("="*50)
        
        plot_path = Path(test_dataset_path) / 'balanced_dataset' / 'test_distribution_plot.png'
        balancer.plot_class_distribution(save_path=str(plot_path))
        
        if plot_path.exists():
            print("✅ Distribution plot created successfully")
        else:
            print("❌ Distribution plot creation failed")
        
        # Test 5: Standalone function
        print("\n" + "="*50)
        print("TEST 5: Testing standalone balancing function")
        print("="*50)
        
        result = run_dataset_balancing_only(test_dataset_path, target_samples_per_class=60)
        
        if result:
            print("✅ Standalone balancing function successful")
        else:
            print("❌ Standalone balancing function failed")
        
        print("\n" + "="*80)
        print("ALL TESTS COMPLETED SUCCESSFULLY!")
        print("="*80)
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Cleanup test dataset
        print(f"\nCleaning up test dataset: {test_dataset_path}")
        if Path(test_dataset_path).exists():
            shutil.rmtree(test_dataset_path)

def main():
    """Main test function"""
    print("Starting dataset balancing tests...")
    
    success = test_dataset_balancing()
    
    if success:
        print("\n🎉 All tests passed! Dataset balancing functionality is working correctly.")
        return 0
    else:
        print("\n❌ Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    exit(main()) 