#!/usr/bin/env python3
"""
Test script to verify dataset balancing functionality
"""

import os
import tempfile
import shutil
from pathlib import Path
from balance_dataset import DatasetBalancer

def create_test_dataset():
    """Create a small test dataset with known imbalance"""
    print("🧪 Creating test dataset...")
    
    # Create temporary directory
    test_dir = Path("test_dataset")
    test_dir.mkdir(exist_ok=True)
    
    # Create directory structure
    (test_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
    (test_dir / "images" / "val").mkdir(parents=True, exist_ok=True)
    (test_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (test_dir / "labels" / "val").mkdir(parents=True, exist_ok=True)
    
    # Create test images (empty files)
    for i in range(20):
        (test_dir / "images" / "train" / f"image_{i:03d}.jpg").touch()
        (test_dir / "images" / "val" / f"image_{i:03d}.jpg").touch()
    
    # Create imbalanced labels
    # Class 0: 50 instances (dominant)
    # Class 1: 20 instances
    # Class 2: 10 instances (minority)
    
    # Train labels
    train_labels = [
        # File 0: 10 instances of class 0
        "\n".join([f"0 0.5 0.5 0.1 0.1"] * 10),
        # File 1: 10 instances of class 0
        "\n".join([f"0 0.5 0.5 0.1 0.1"] * 10),
        # File 2: 10 instances of class 0
        "\n".join([f"0 0.5 0.5 0.1 0.1"] * 10),
        # File 3: 10 instances of class 0
        "\n".join([f"0 0.5 0.5 0.1 0.1"] * 10),
        # File 4: 10 instances of class 0
        "\n".join([f"0 0.5 0.5 0.1 0.1"] * 10),
        # File 5: 10 instances of class 1
        "\n".join([f"1 0.5 0.5 0.1 0.1"] * 10),
        # File 6: 10 instances of class 1
        "\n".join([f"1 0.5 0.5 0.1 0.1"] * 10),
        # File 7: 10 instances of class 2
        "\n".join([f"2 0.5 0.5 0.1 0.1"] * 10),
    ]
    
    # Val labels
    val_labels = [
        # File 0: 5 instances of class 0
        "\n".join([f"0 0.5 0.5 0.1 0.1"] * 5),
        # File 1: 5 instances of class 0
        "\n".join([f"0 0.5 0.5 0.1 0.1"] * 5),
        # File 2: 5 instances of class 1
        "\n".join([f"1 0.5 0.5 0.1 0.1"] * 5),
        # File 3: 5 instances of class 1
        "\n".join([f"1 0.5 0.5 0.1 0.1"] * 5),
    ]
    
    # Write train labels
    for i, content in enumerate(train_labels):
        with open(test_dir / "labels" / "train" / f"image_{i:03d}.txt", 'w') as f:
            f.write(content)
    
    # Write val labels
    for i, content in enumerate(val_labels):
        with open(test_dir / "labels" / "val" / f"image_{i:03d}.txt", 'w') as f:
            f.write(content)
    
    # Create dataset.yaml
    yaml_content = """names:
  0: class_0
  1: class_1
  2: class_2
nc: 3
path: ./test_dataset
train: images/train
val: images/val
"""
    
    with open(test_dir / "dataset.yaml", 'w') as f:
        f.write(yaml_content)
    
    print(f"✅ Test dataset created at: {test_dir}")
    print("   Expected distribution:")
    print("   - Class 0: 50 instances (dominant)")
    print("   - Class 1: 20 instances")
    print("   - Class 2: 10 instances (minority)")
    
    return test_dir

def test_balancing():
    """Test the balancing functionality"""
    print("\n🧪 Testing dataset balancing...")
    
    # Create test dataset
    test_dataset = create_test_dataset()
    
    # Create balancer
    balancer = DatasetBalancer(
        dataset_path=str(test_dataset),
        output_path="test_dataset_balanced"
    )
    
    # Run balancing
    success = balancer.balance_dataset(
        target_samples_per_class=10,  # Target 10 instances per class
        seed=42
    )
    
    if success:
        print("\n✅ Balancing test completed successfully!")
        
        # Verify results
        verify_balance_results()
    else:
        print("\n❌ Balancing test failed!")
    
    # Cleanup
    print("\n🧹 Cleaning up test files...")
    if test_dataset.exists():
        shutil.rmtree(test_dataset)
    if Path("test_dataset_balanced").exists():
        shutil.rmtree("test_dataset_balanced")

def verify_balance_results():
    """Verify the balance results"""
    print("\n🔍 Verifying balance results...")
    
    balanced_dir = Path("test_dataset_balanced")
    
    if not balanced_dir.exists():
        print("❌ Balanced dataset not found!")
        return
    
    # Count instances in balanced dataset
    class_counts = {0: 0, 1: 0, 2: 0}
    
    for split in ['train', 'val']:
        labels_dir = balanced_dir / "labels" / split
        if not labels_dir.exists():
            continue
            
        for label_file in labels_dir.glob('*.txt'):
            try:
                with open(label_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            parts = line.split()
                            if len(parts) >= 5:
                                class_id = int(parts[0])
                                class_counts[class_id] += 1
            except Exception as e:
                print(f"⚠️  Error reading {label_file}: {e}")
    
    print(f"\n📊 Final class distribution:")
    for class_id, count in sorted(class_counts.items()):
        print(f"   Class {class_id}: {count} instances")
    
    # Check if balanced
    counts = list(class_counts.values())
    if len(set(counts)) <= 1 or max(counts) - min(counts) <= 2:
        print("✅ Dataset is well balanced!")
    else:
        print("❌ Dataset is not properly balanced!")
        print(f"   Min: {min(counts)}, Max: {max(counts)}, Difference: {max(counts) - min(counts)}")

if __name__ == "__main__":
    test_balancing() 