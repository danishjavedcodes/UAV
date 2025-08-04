"""
Dataset Downsampling Script for HMAY-TSF
Reduces overrepresented classes to balance the dataset
"""

import os
import shutil
import random
import argparse
from pathlib import Path
import yaml
from collections import defaultdict
import numpy as np

class DatasetDownsampler:
    """Downsample overrepresented classes to balance the dataset"""
    
    def __init__(self, dataset_path='./dataset', target_samples_per_class=5000):
        self.dataset_path = Path(dataset_path)
        self.target_samples_per_class = target_samples_per_class
        self.class_counts = None
        self.class_files = None
        
        # Class names for reference
        self.class_names = {
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
        }
        
        self.analyze_class_distribution()
    
    def analyze_class_distribution(self):
        """Analyze current class distribution in the dataset"""
        print("Analyzing class distribution...")
        
        self.class_counts = defaultdict(int)
        self.class_files = defaultdict(list)
        
        # Analyze train, val, and test sets
        for split in ['train', 'val', 'test']:
            labels_dir = self.dataset_path / 'labels' / split
            images_dir = self.dataset_path / 'images' / split
            
            if not labels_dir.exists():
                print(f"Warning: {labels_dir} does not exist, skipping...")
                continue
            
            print(f"Processing {split} split...")
            
            for label_file in labels_dir.glob('*.txt'):
                image_file = images_dir / f"{label_file.stem}.jpg"
                
                if not image_file.exists():
                    print(f"Warning: Image {image_file} not found for label {label_file}")
                    continue
                
                # Read labels and count classes
                classes_in_file = set()
                try:
                    with open(label_file, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                class_id = int(parts[0])
                                classes_in_file.add(class_id)
                                self.class_counts[class_id] += 1
                                self.class_files[class_id].append((str(label_file), str(image_file), split))
                except Exception as e:
                    print(f"Error reading {label_file}: {e}")
        
        print("\nCurrent Class Distribution:")
        total_samples = sum(self.class_counts.values())
        for class_id in sorted(self.class_counts.keys()):
            count = self.class_counts[class_id]
            percentage = (count / total_samples) * 100
            class_name = self.class_names.get(class_id, f'class_{class_id}')
            print(f"  Class {class_id} ({class_name}): {count:,} ({percentage:.2f}%)")
        
        # Calculate imbalance ratio
        max_count = max(self.class_counts.values())
        min_count = min(self.class_counts.values())
        imbalance_ratio = max_count / min_count
        print(f"\nImbalance Ratio: {imbalance_ratio:.2f}")
        
        return self.class_counts
    
    def calculate_downsample_targets(self):
        """Calculate target samples for each class after downsampling"""
        print(f"\nCalculating downsampling targets (target: {self.target_samples_per_class} per class)...")
        
        downsample_targets = {}
        for class_id, count in self.class_counts.items():
            if count > self.target_samples_per_class:
                # Need to downsample
                keep_ratio = self.target_samples_per_class / count
                target_count = self.target_samples_per_class
                downsample_targets[class_id] = {
                    'current': count,
                    'target': target_count,
                    'keep_ratio': keep_ratio,
                    'remove_count': count - target_count
                }
                print(f"  Class {class_id} ({self.class_names.get(class_id, f'class_{class_id}')}):")
                print(f"    Current: {count:,} → Target: {target_count:,}")
                print(f"    Keep ratio: {keep_ratio:.3f} ({keep_ratio*100:.1f}%)")
                print(f"    Remove: {count - target_count:,} samples")
            else:
                # Keep all samples
                downsample_targets[class_id] = {
                    'current': count,
                    'target': count,
                    'keep_ratio': 1.0,
                    'remove_count': 0
                }
                print(f"  Class {class_id} ({self.class_names.get(class_id, f'class_{class_id}')}):")
                print(f"    Current: {count:,} → Target: {count:,} (no downsampling needed)")
        
        return downsample_targets
    
    def downsample_dataset(self, output_path='./dataset_downsampled'):
        """Create downsampled dataset"""
        output_path = Path(output_path)
        
        # Calculate targets
        targets = self.calculate_downsample_targets()
        
        # Create output directory structure
        for split in ['train', 'val', 'test']:
            (output_path / 'images' / split).mkdir(parents=True, exist_ok=True)
            (output_path / 'labels' / split).mkdir(parents=True, exist_ok=True)
        
        print(f"\nCreating downsampled dataset at: {output_path}")
        
        # Process each class
        final_class_counts = defaultdict(int)
        total_copied = 0
        
        for class_id in sorted(self.class_counts.keys()):
            class_name = self.class_names.get(class_id, f'class_{class_id}')
            target_info = targets[class_id]
            
            print(f"\nProcessing Class {class_id} ({class_name}):")
            
            if target_info['remove_count'] == 0:
                # No downsampling needed - copy all files
                print(f"  No downsampling needed, copying all {target_info['current']} files...")
                for label_file, image_file, split in self.class_files[class_id]:
                    self._copy_file_pair(label_file, image_file, output_path, split)
                    final_class_counts[class_id] += 1
                    total_copied += 1
            else:
                # Downsampling needed
                print(f"  Downsampling from {target_info['current']} to {target_info['target']} files...")
                
                # Randomly select files to keep
                files_to_keep = random.sample(
                    self.class_files[class_id], 
                    target_info['target']
                )
                
                # Copy selected files
                for label_file, image_file, split in files_to_keep:
                    self._copy_file_pair(label_file, image_file, output_path, split)
                    final_class_counts[class_id] += 1
                    total_copied += 1
        
        # Create dataset YAML
        self._create_dataset_yaml(output_path)
        
        # Print final statistics
        print(f"\n{'='*60}")
        print("DOWNSAMPLING COMPLETED")
        print(f"{'='*60}")
        print(f"Total files copied: {total_copied:,}")
        print(f"Output directory: {output_path}")
        
        print(f"\nFinal Class Distribution:")
        total_final = sum(final_class_counts.values())
        for class_id in sorted(final_class_counts.keys()):
            count = final_class_counts[class_id]
            percentage = (count / total_final) * 100
            class_name = self.class_names.get(class_id, f'class_{class_id}')
            print(f"  Class {class_id} ({class_name}): {count:,} ({percentage:.2f}%)")
        
        # Calculate new imbalance ratio
        max_final = max(final_class_counts.values())
        min_final = min(final_class_counts.values())
        new_imbalance_ratio = max_final / min_final
        print(f"\nNew Imbalance Ratio: {new_imbalance_ratio:.2f}")
        
        # Calculate improvement
        old_imbalance = max(self.class_counts.values()) / min(self.class_counts.values())
        improvement = (old_imbalance - new_imbalance_ratio) / old_imbalance * 100
        print(f"Imbalance Reduction: {improvement:.1f}%")
        
        return output_path
    
    def _copy_file_pair(self, label_file, image_file, output_path, split):
        """Copy a label-image file pair to the output directory"""
        label_path = Path(label_file)
        image_path = Path(image_file)
        
        # Copy label file
        dest_label = output_path / 'labels' / split / label_path.name
        shutil.copy2(label_path, dest_label)
        
        # Copy image file
        dest_image = output_path / 'images' / split / image_path.name
        shutil.copy2(image_path, dest_image)
    
    def _create_dataset_yaml(self, output_path):
        """Create dataset YAML configuration file"""
        dataset_config = {
            'path': str(output_path.absolute()),
            'train': 'images/train',
            'val': 'images/val', 
            'test': 'images/test',
            'nc': 11,
            'names': self.class_names,
            'downsampled': True,
            'target_samples_per_class': self.target_samples_per_class
        }
        
        yaml_path = output_path / 'dataset.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(dataset_config, f, default_flow_style=False)
        
        print(f"Dataset YAML created: {yaml_path}")
        return str(yaml_path)
    
    def create_balanced_splits(self, output_path='./dataset_downsampled', train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
        """Create balanced train/val/test splits after downsampling"""
        print(f"\nCreating balanced splits (train: {train_ratio}, val: {val_ratio}, test: {test_ratio})...")
        
        # Create temporary directory for balanced splits
        temp_path = Path(output_path) / 'temp_balanced'
        temp_path.mkdir(parents=True, exist_ok=True)
        
        # Group files by class
        class_files = defaultdict(list)
        for class_id, files in self.class_files.items():
            class_files[class_id].extend(files)
        
        # Create balanced splits
        for class_id, files in class_files.items():
            class_name = self.class_names.get(class_id, f'class_{class_id}')
            print(f"\nProcessing Class {class_id} ({class_name}): {len(files)} files")
            
            # Shuffle files
            random.shuffle(files)
            
            # Calculate split sizes
            total_files = len(files)
            train_size = int(total_files * train_ratio)
            val_size = int(total_files * val_ratio)
            test_size = total_files - train_size - val_size
            
            # Split files
            train_files = files[:train_size]
            val_files = files[train_size:train_size + val_size]
            test_files = files[train_size + val_size:]
            
            print(f"  Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")
            
            # Copy files to appropriate splits
            for split_name, split_files in [('train', train_files), ('val', val_files), ('test', test_files)]:
                for label_file, image_file, _ in split_files:
                    self._copy_file_pair(label_file, image_file, temp_path, split_name)
        
        # Create final dataset YAML
        self._create_dataset_yaml(temp_path)
        
        print(f"\nBalanced dataset created at: {temp_path}")
        return temp_path

def main():
    parser = argparse.ArgumentParser(description='Downsample overrepresented classes in dataset')
    parser.add_argument('--dataset_path', type=str, default='./dataset', 
                       help='Path to original dataset')
    parser.add_argument('--output_path', type=str, default='./dataset_downsampled',
                       help='Path to output downsampled dataset')
    parser.add_argument('--target_samples', type=int, default=5000,
                       help='Target samples per class')
    parser.add_argument('--create_balanced_splits', action='store_true',
                       help='Create balanced train/val/test splits')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                       help='Training set ratio')
    parser.add_argument('--val_ratio', type=float, default=0.2,
                       help='Validation set ratio')
    parser.add_argument('--test_ratio', type=float, default=0.1,
                       help='Test set ratio')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    print("="*80)
    print("HMAY-TSF DATASET DOWNSAMPLING")
    print("="*80)
    
    if not os.path.exists(args.dataset_path):
        print(f"Error: Dataset path {args.dataset_path} does not exist!")
        return
    
    # Initialize downsampler
    print(f"Initializing dataset downsampler...")
    downsampler = DatasetDownsampler(
        dataset_path=args.dataset_path,
        target_samples_per_class=args.target_samples
    )
    
    if args.create_balanced_splits:
        # Create balanced splits
        output_path = downsampler.create_balanced_splits(
            output_path=args.output_path,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio
        )
    else:
        # Simple downsampling
        output_path = downsampler.downsample_dataset(output_path=args.output_path)
    
    print(f"\n{'='*80}")
    print("NEXT STEPS")
    print(f"{'='*80}")
    print(f"1. Use the downsampled dataset for training:")
    print(f"   python train_hmay_tsf.py --data {output_path}/dataset.yaml")
    print(f"2. The model should now be less biased towards bicycle class")
    print(f"3. Monitor class-wise performance during training")
    print(f"4. If needed, adjust target_samples_per_class for further balancing")
    
    print(f"\n{'='*80}")

if __name__ == "__main__":
    main() 