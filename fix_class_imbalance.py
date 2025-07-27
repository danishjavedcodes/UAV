"""
Fix Class Imbalance Script for HMAY-TSF Model
Creates balanced dataset and retrains model to address class bias
"""

import os
import sys
import argparse
from pathlib import Path
import shutil
import numpy as np
from collections import Counter, defaultdict
import yaml

def analyze_class_distribution(dataset_path='./dataset'):
    """Analyze current class distribution"""
    print("="*60)
    print("ANALYZING CLASS DISTRIBUTION")
    print("="*60)
    
    train_labels_path = Path(dataset_path) / 'labels' / 'train'
    
    if not train_labels_path.exists():
        print(f"Training labels not found: {train_labels_path}")
        return None
    
    # Collect all class labels
    all_labels = []
    for label_file in train_labels_path.glob('*.txt'):
        with open(label_file, 'r') as f:
            for line in f.readlines():
                if line.strip():
                    class_id = int(line.split()[0])
                    all_labels.append(class_id)
    
    # Count classes
    class_counts = Counter(all_labels)
    total_samples = sum(class_counts.values())
    
    print("Current Class Distribution:")
    for class_id in sorted(class_counts.keys()):
        count = class_counts[class_id]
        percentage = (count / total_samples) * 100
        print(f"  Class {class_id}: {count:,} ({percentage:.2f}%)")
    
    # Calculate imbalance ratio
    max_count = max(class_counts.values())
    min_count = min(class_counts.values())
    imbalance_ratio = max_count / min_count
    
    print(f"\nImbalance Analysis:")
    print(f"  Most common class: {max_count:,} samples")
    print(f"  Rarest class: {min_count:,} samples")
    print(f"  Imbalance ratio: {imbalance_ratio:.2f}:1")
    
    if imbalance_ratio > 10:
        print(f"  ⚠️  SEVERE IMBALANCE DETECTED")
    elif imbalance_ratio > 5:
        print(f"  ⚠️  MODERATE IMBALANCE DETECTED")
    else:
        print(f"  ✅  Good balance")
    
    return class_counts

def create_balanced_dataset(dataset_path='./dataset', output_path='./dataset_balanced', target_samples=None):
    """Create a balanced dataset by oversampling rare classes and undersampling common classes"""
    print("\n" + "="*60)
    print("CREATING BALANCED DATASET")
    print("="*60)
    
    # Analyze current distribution
    class_counts = analyze_class_distribution(dataset_path)
    if class_counts is None:
        return None
    
    # Set target samples per class
    if target_samples is None:
        # Use median as target to balance the dataset
        counts = list(class_counts.values())
        target_samples = int(np.median(counts))
    
    print(f"Target samples per class: {target_samples:,}")
    
    # Create output directory structure
    output_path = Path(output_path)
    (output_path / 'images' / 'train').mkdir(parents=True, exist_ok=True)
    (output_path / 'images' / 'val').mkdir(parents=True, exist_ok=True)
    (output_path / 'labels' / 'train').mkdir(parents=True, exist_ok=True)
    (output_path / 'labels' / 'val').mkdir(parents=True, exist_ok=True)
    
    # Collect all training files
    train_images_path = Path(dataset_path) / 'images' / 'train'
    train_labels_path = Path(dataset_path) / 'labels' / 'train'
    
    # Group files by class
    class_files = defaultdict(list)
    
    print("Collecting files by class...")
    for label_file in train_labels_path.glob('*.txt'):
        image_file = train_images_path / f"{label_file.stem}.jpg"
        if image_file.exists():
            with open(label_file, 'r') as f:
                for line in f.readlines():
                    if line.strip():
                        class_id = int(line.split()[0])
                        class_files[class_id].append((image_file, label_file))
                        break  # Only count each image once
    
    # Create balanced dataset
    balanced_files = []
    
    print("Balancing classes...")
    for class_id, files in class_files.items():
        current_count = len(files)
        target_count = target_samples
        
        if current_count < target_count:
            # Oversample rare classes
            oversample_factor = target_count // current_count + 1
            selected_files = files * oversample_factor
            selected_files = selected_files[:target_count]
            print(f"  Class {class_id}: {current_count:,} -> {len(selected_files):,} (oversampled)")
        else:
            # Undersample common classes
            selected_files = files[:target_count]
            print(f"  Class {class_id}: {current_count:,} -> {len(selected_files):,} (undersampled)")
        
        balanced_files.extend(selected_files)
    
    # Shuffle and split into train/val
    np.random.shuffle(balanced_files)
    split_idx = int(len(balanced_files) * 0.8)
    train_files = balanced_files[:split_idx]
    val_files = balanced_files[split_idx:]
    
    print(f"\nCopying files...")
    print(f"  Train: {len(train_files):,} images")
    print(f"  Val: {len(val_files):,} images")
    
    # Copy train files
    for i, (image_file, label_file) in enumerate(train_files):
        new_image_path = output_path / 'images' / 'train' / f"{i:06d}.jpg"
        new_label_path = output_path / 'labels' / 'train' / f"{i:06d}.txt"
        
        shutil.copy2(image_file, new_image_path)
        shutil.copy2(label_file, new_label_path)
    
    # Copy val files
    for i, (image_file, label_file) in enumerate(val_files):
        new_image_path = output_path / 'images' / 'val' / f"{i:06d}.jpg"
        new_label_path = output_path / 'labels' / 'val' / f"{i:06d}.txt"
        
        shutil.copy2(image_file, new_image_path)
        shutil.copy2(label_file, new_label_path)
    
    print(f"\nBalanced dataset created at: {output_path}")
    
    # Verify balance
    verify_balance(output_path)
    
    return str(output_path)

def verify_balance(dataset_path):
    """Verify the balanced dataset distribution"""
    print("\n" + "="*60)
    print("VERIFYING BALANCED DATASET")
    print("="*60)
    
    train_labels_path = Path(dataset_path) / 'labels' / 'train'
    
    # Collect all class labels
    all_labels = []
    for label_file in train_labels_path.glob('*.txt'):
        with open(label_file, 'r') as f:
            for line in f.readlines():
                if line.strip():
                    class_id = int(line.split()[0])
                    all_labels.append(class_id)
    
    # Count classes
    class_counts = Counter(all_labels)
    total_samples = sum(class_counts.values())
    
    print("Balanced Dataset Distribution:")
    for class_id in sorted(class_counts.keys()):
        count = class_counts[class_id]
        percentage = (count / total_samples) * 100
        print(f"  Class {class_id}: {count:,} ({percentage:.2f}%)")
    
    # Calculate new imbalance ratio
    max_count = max(class_counts.values())
    min_count = min(class_counts.values())
    imbalance_ratio = max_count / min_count
    
    print(f"\nBalance Analysis:")
    print(f"  Most common class: {max_count:,} samples")
    print(f"  Rarest class: {min_count:,} samples")
    print(f"  New imbalance ratio: {imbalance_ratio:.2f}:1")
    
    if imbalance_ratio < 2:
        print(f"  ✅  EXCELLENT BALANCE ACHIEVED")
    elif imbalance_ratio < 3:
        print(f"  ✅  GOOD BALANCE ACHIEVED")
    else:
        print(f"  ⚠️  Still some imbalance, but much improved")

def create_balanced_yaml(dataset_path, output_path='./dataset_balanced.yaml'):
    """Create YAML configuration for balanced dataset"""
    print(f"\nCreating balanced dataset YAML: {output_path}")
    
    dataset_config = {
        'path': str(Path(dataset_path).absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': 11,  # Number of classes
        'names': {
            0: 'ignored regions',
            1: 'pedestrian',
            2: 'people',
            3: 'bicycle',
            4: 'car',
            5: 'van',
            6: 'truck',
            7: 'tricycle',
            8: 'awning-tricycle',
            9: 'bus',
            10: 'motor'
        },
        'balanced': True,
        'description': 'Balanced dataset created to address class imbalance'
    }
    
    with open(output_path, 'w') as f:
        yaml.dump(dataset_config, f, default_flow_style=False)
    
    print(f"YAML configuration created: {output_path}")
    return output_path

def main():
    parser = argparse.ArgumentParser(description='Fix class imbalance in HMAY-TSF dataset')
    parser.add_argument('--dataset', type=str, default='./dataset',
                       help='Path to original dataset')
    parser.add_argument('--output', type=str, default='./dataset_balanced',
                       help='Path to output balanced dataset')
    parser.add_argument('--target_samples', type=int, default=None,
                       help='Target samples per class (default: median)')
    parser.add_argument('--analyze_only', action='store_true',
                       help='Only analyze current distribution')
    parser.add_argument('--create_yaml', action='store_true',
                       help='Create YAML configuration file')
    
    args = parser.parse_args()
    
    print("HMAY-TSF CLASS IMBALANCE FIXER")
    print("="*60)
    
    # Analyze current distribution
    class_counts = analyze_class_distribution(args.dataset)
    
    if args.analyze_only:
        return
    
    if class_counts is None:
        print("Error: Could not analyze dataset distribution")
        return
    
    # Create balanced dataset
    balanced_path = create_balanced_dataset(
        dataset_path=args.dataset,
        output_path=args.output,
        target_samples=args.target_samples
    )
    
    if balanced_path is None:
        print("Error: Could not create balanced dataset")
        return
    
    # Create YAML configuration
    if args.create_yaml:
        yaml_path = create_balanced_yaml(balanced_path)
        print(f"\nNext steps:")
        print(f"1. Retrain your model with the balanced dataset:")
        print(f"   python train_hmay_tsf.py --data {yaml_path}")
        print(f"2. Test the new model with:")
        print(f"   python test_inference.py --model ./runs/train/best.pt")
        print(f"3. Use the inference script:")
        print(f"   python infer.py --source ./dataset/images/test --output ./results")
    
    print(f"\nClass imbalance fix completed!")
    print(f"Balanced dataset: {balanced_path}")

if __name__ == "__main__":
    main() 