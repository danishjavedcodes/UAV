"""
Dataset Balancing Script for HMAY-TSF Model
Balances the dataset so that all classes have equal number of samples
"""

import os
import shutil
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import json
from collections import defaultdict, Counter
import yaml
from tqdm import tqdm
import cv2

class DatasetBalancer:
    """Advanced dataset balancer for YOLO format datasets"""
    
    def __init__(self, dataset_path='./dataset', output_path='./dataset_balanced'):
        self.dataset_path = Path(dataset_path)
        self.output_path = Path(output_path)
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
        
        # Create output directories
        self.output_path.mkdir(parents=True, exist_ok=True)
        (self.output_path / 'images' / 'train').mkdir(parents=True, exist_ok=True)
        (self.output_path / 'images' / 'val').mkdir(parents=True, exist_ok=True)
        (self.output_path / 'images' / 'test').mkdir(parents=True, exist_ok=True)
        (self.output_path / 'labels' / 'train').mkdir(parents=True, exist_ok=True)
        (self.output_path / 'labels' / 'val').mkdir(parents=True, exist_ok=True)
        (self.output_path / 'labels' / 'test').mkdir(parents=True, exist_ok=True)
        
        # Statistics
        self.original_stats = {}
        self.balanced_stats = {}
        self.class_distribution = defaultdict(list)
        
    def analyze_dataset(self):
        """Analyze current dataset distribution"""
        print("🔍 Analyzing dataset distribution...")
        
        splits = ['train', 'val', 'test']
        
        for split in splits:
            print(f"\n📊 Analyzing {split} split...")
            
            labels_dir = self.dataset_path / 'labels' / split
            images_dir = self.dataset_path / 'images' / split
            
            if not labels_dir.exists():
                print(f"⚠️  Warning: {labels_dir} does not exist, skipping...")
                continue
                
            # Count classes in each file
            class_counts = defaultdict(int)
            file_class_mapping = defaultdict(list)
            total_files = 0
            
            label_files = list(labels_dir.glob('*.txt'))
            
            for label_file in tqdm(label_files, desc=f"Processing {split} labels"):
                if not label_file.exists():
                    continue
                    
                image_file = images_dir / f"{label_file.stem}.jpg"
                if not image_file.exists():
                    continue
                    
                total_files += 1
                file_classes = set()
                
                try:
                    with open(label_file, 'r') as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                parts = line.split()
                                if len(parts) >= 5:
                                    class_id = int(parts[0])
                                    class_counts[class_id] += 1
                                    file_classes.add(class_id)
                    
                    # Store which classes are in this file
                    for class_id in file_classes:
                        file_class_mapping[class_id].append(str(label_file))
                        
                except Exception as e:
                    print(f"⚠️  Error reading {label_file}: {e}")
                    continue
            
            self.original_stats[split] = {
                'total_files': total_files,
                'class_counts': dict(class_counts),
                'file_class_mapping': dict(file_class_mapping)
            }
            
            print(f"   Total files: {total_files}")
            print(f"   Class distribution:")
            for class_id, count in sorted(class_counts.items()):
                class_name = self.class_names.get(class_id, f"class_{class_id}")
                print(f"     {class_id} ({class_name}): {count} instances")
    
    def plot_original_distribution(self):
        """Plot original class distribution"""
        print("\n📈 Plotting original class distribution...")
        
        # Combine all splits
        total_class_counts = defaultdict(int)
        for split, stats in self.original_stats.items():
            for class_id, count in stats['class_counts'].items():
                total_class_counts[class_id] += count
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Bar plot
        classes = sorted(total_class_counts.keys())
        counts = [total_class_counts[c] for c in classes]
        class_labels = [f"{c}\n({self.class_names.get(c, f'class_{c}')})" for c in classes]
        
        bars = ax1.bar(class_labels, counts, color='skyblue', alpha=0.7)
        ax1.set_title('Original Class Distribution', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Class ID (Class Name)', fontsize=12)
        ax1.set_ylabel('Number of Instances', fontsize=12)
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + max(counts)*0.01,
                    f'{count:,}', ha='center', va='bottom', fontsize=10)
        
        # Pie chart
        total_instances = sum(counts)
        percentages = [count/total_instances*100 for count in counts]
        
        wedges, texts, autotexts = ax2.pie(counts, labels=class_labels, autopct='%1.1f%%',
                                          startangle=90, colors=plt.cm.Set3(np.linspace(0, 1, len(classes))))
        ax2.set_title('Class Distribution (Percentage)', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_path / 'original_class_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Save statistics
        stats_data = {
            'total_instances': total_instances,
            'class_distribution': dict(total_class_counts),
            'class_names': self.class_names,
            'percentages': {class_id: percentages[i] for i, class_id in enumerate(classes)}
        }
        
        with open(self.output_path / 'original_statistics.json', 'w') as f:
            json.dump(stats_data, f, indent=2)
        
        print(f"📊 Original distribution saved to: {self.output_path / 'original_class_distribution.png'}")
        print(f"📄 Statistics saved to: {self.output_path / 'original_statistics.json'}")
        
        return total_class_counts
    
    def calculate_balanced_samples(self, target_samples_per_class=None):
        """Calculate how many samples to use for each class to achieve balance"""
        print("\n⚖️  Calculating balanced sample distribution...")
        
        # Find the class with minimum samples
        all_class_counts = defaultdict(int)
        for split, stats in self.original_stats.items():
            for class_id, count in stats['class_counts'].items():
                all_class_counts[class_id] += count
        
        if not all_class_counts:
            print("❌ No class data found!")
            return None
        
        min_samples = min(all_class_counts.values())
        
        if target_samples_per_class is None:
            target_samples_per_class = min_samples
        
        print(f"🎯 Target samples per class: {target_samples_per_class:,}")
        print(f"📉 Minimum samples in any class: {min_samples:,}")
        
        # Calculate how many samples to take from each class
        balanced_distribution = {}
        for class_id, count in all_class_counts.items():
            if count >= target_samples_per_class:
                balanced_distribution[class_id] = target_samples_per_class
            else:
                # For classes with fewer samples, use all available
                balanced_distribution[class_id] = count
                print(f"⚠️  Class {class_id} ({self.class_names.get(class_id, f'class_{class_id}')}) "
                      f"has only {count} samples (less than target {target_samples_per_class})")
        
        return balanced_distribution
    
    def select_balanced_samples(self, balanced_distribution):
        """Select balanced samples from each class"""
        print("\n🎯 Selecting balanced samples...")
        
        selected_files = defaultdict(list)
        
        for split, stats in self.original_stats.items():
            print(f"\n📁 Processing {split} split...")
            
            file_class_mapping = stats['file_class_mapping']
            
            for class_id, target_count in balanced_distribution.items():
                if class_id not in file_class_mapping:
                    continue
                
                available_files = file_class_mapping[class_id]
                print(f"   Class {class_id} ({self.class_names.get(class_id, f'class_{class_id}')}): "
                      f"{len(available_files)} files available, need {target_count} samples")
                
                # Calculate how many files we need to select
                # We'll select files that contain this class
                files_with_class = [f for f in available_files if f.endswith('.txt')]
                
                if len(files_with_class) >= target_count:
                    # Randomly select files
                    selected = random.sample(files_with_class, target_count)
                else:
                    # Use all available files
                    selected = files_with_class
                    print(f"     ⚠️  Only {len(files_with_class)} files available, using all")
                
                selected_files[class_id].extend(selected)
        
        return selected_files
    
    def create_balanced_dataset(self, selected_files, balanced_distribution):
        """Create the balanced dataset"""
        print("\n🔄 Creating balanced dataset...")
        
        # Track which files we've already copied to avoid duplicates
        copied_files = set()
        
        # Create train/val/test split for balanced dataset
        splits = ['train', 'val', 'test']
        split_ratios = [0.7, 0.2, 0.1]  # 70% train, 20% val, 10% test
        
        balanced_splits = {split: [] for split in splits}
        
        for class_id, files in selected_files.items():
            print(f"\n📦 Processing class {class_id} ({self.class_names.get(class_id, f'class_{class_id}')})...")
            
            # Shuffle files for random distribution
            random.shuffle(files)
            
            # Split files according to ratios
            total_files = len(files)
            train_end = int(total_files * split_ratios[0])
            val_end = train_end + int(total_files * split_ratios[1])
            
            balanced_splits['train'].extend(files[:train_end])
            balanced_splits['val'].extend(files[train_end:val_end])
            balanced_splits['test'].extend(files[val_end:])
            
            print(f"   Train: {len(files[:train_end])} files")
            print(f"   Val: {len(files[train_end:val_end])} files")
            print(f"   Test: {len(files[val_end:])} files")
        
        # Copy files to balanced dataset
        for split, files in balanced_splits.items():
            print(f"\n📁 Copying {split} files...")
            
            split_files = list(set(files))  # Remove duplicates
            random.shuffle(split_files)
            
            for file_path in tqdm(split_files, desc=f"Copying {split} files"):
                if file_path in copied_files:
                    continue
                
                try:
                    # Get the original split from the file path
                    original_split = None
                    for orig_split in ['train', 'val', 'test']:
                        if f"/{orig_split}/" in file_path:
                            original_split = orig_split
                            break
                    
                    if original_split is None:
                        continue
                    
                    # Source paths
                    label_src = Path(file_path)
                    image_src = self.dataset_path / 'images' / original_split / f"{label_src.stem}.jpg"
                    
                    # Destination paths
                    label_dst = self.output_path / 'labels' / split / label_src.name
                    image_dst = self.output_path / 'images' / split / f"{label_src.stem}.jpg"
                    
                    # Copy files
                    if label_src.exists() and image_src.exists():
                        shutil.copy2(label_src, label_dst)
                        shutil.copy2(image_src, image_dst)
                        copied_files.add(file_path)
                    
                except Exception as e:
                    print(f"⚠️  Error copying {file_path}: {e}")
                    continue
        
        print(f"\n✅ Balanced dataset created at: {self.output_path}")
        print(f"📊 Total files copied: {len(copied_files)}")
    
    def create_balanced_dataset_yaml(self):
        """Create dataset.yaml for balanced dataset"""
        yaml_content = {
            'names': self.class_names,
            'nc': len(self.class_names),
            'path': str(self.output_path.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test'
        }
        
        yaml_path = self.output_path / 'dataset.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_content, f, default_flow_style=False, sort_keys=False)
        
        print(f"📄 Dataset YAML created: {yaml_path}")
    
    def analyze_balanced_dataset(self):
        """Analyze the balanced dataset"""
        print("\n🔍 Analyzing balanced dataset...")
        
        splits = ['train', 'val', 'test']
        
        for split in splits:
            print(f"\n📊 Analyzing balanced {split} split...")
            
            labels_dir = self.output_path / 'labels' / split
            if not labels_dir.exists():
                continue
                
            class_counts = defaultdict(int)
            total_files = 0
            
            label_files = list(labels_dir.glob('*.txt'))
            
            for label_file in tqdm(label_files, desc=f"Processing balanced {split}"):
                if not label_file.exists():
                    continue
                    
                total_files += 1
                
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
                    continue
            
            self.balanced_stats[split] = {
                'total_files': total_files,
                'class_counts': dict(class_counts)
            }
            
            print(f"   Total files: {total_files}")
            print(f"   Class distribution:")
            for class_id, count in sorted(class_counts.items()):
                class_name = self.class_names.get(class_id, f"class_{class_id}")
                print(f"     {class_id} ({class_name}): {count} instances")
    
    def plot_balanced_distribution(self):
        """Plot balanced class distribution"""
        print("\n📈 Plotting balanced class distribution...")
        
        # Combine all splits
        total_class_counts = defaultdict(int)
        for split, stats in self.balanced_stats.items():
            for class_id, count in stats['class_counts'].items():
                total_class_counts[class_id] += count
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Bar plot
        classes = sorted(total_class_counts.keys())
        counts = [total_class_counts[c] for c in classes]
        class_labels = [f"{c}\n({self.class_names.get(c, f'class_{c}')})" for c in classes]
        
        bars = ax1.bar(class_labels, counts, color='lightgreen', alpha=0.7)
        ax1.set_title('Balanced Class Distribution', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Class ID (Class Name)', fontsize=12)
        ax1.set_ylabel('Number of Instances', fontsize=12)
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + max(counts)*0.01,
                    f'{count:,}', ha='center', va='bottom', fontsize=10)
        
        # Pie chart
        total_instances = sum(counts)
        percentages = [count/total_instances*100 for count in counts]
        
        wedges, texts, autotexts = ax2.pie(counts, labels=class_labels, autopct='%1.1f%%',
                                          startangle=90, colors=plt.cm.Set3(np.linspace(0, 1, len(classes))))
        ax2.set_title('Balanced Class Distribution (Percentage)', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_path / 'balanced_class_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Save statistics
        stats_data = {
            'total_instances': total_instances,
            'class_distribution': dict(total_class_counts),
            'class_names': self.class_names,
            'percentages': {class_id: percentages[i] for i, class_id in enumerate(classes)}
        }
        
        with open(self.output_path / 'balanced_statistics.json', 'w') as f:
            json.dump(stats_data, f, indent=2)
        
        print(f"📊 Balanced distribution saved to: {self.output_path / 'balanced_class_distribution.png'}")
        print(f"📄 Statistics saved to: {self.output_path / 'balanced_statistics.json'}")
    
    def generate_balance_report(self):
        """Generate comprehensive balance report"""
        print("\n📋 Generating balance report...")
        
        report = {
            'dataset_balancing_report': {
                'original_dataset_path': str(self.dataset_path),
                'balanced_dataset_path': str(self.output_path),
                'timestamp': pd.Timestamp.now().isoformat(),
                'class_names': self.class_names,
                'original_statistics': self.original_stats,
                'balanced_statistics': self.balanced_stats
            }
        }
        
        # Calculate improvement metrics
        original_totals = defaultdict(int)
        balanced_totals = defaultdict(int)
        
        for split, stats in self.original_stats.items():
            for class_id, count in stats['class_counts'].items():
                original_totals[class_id] += count
        
        for split, stats in self.balanced_stats.items():
            for class_id, count in stats['class_counts'].items():
                balanced_totals[class_id] += count
        
        # Calculate balance metrics
        original_counts = list(original_totals.values())
        balanced_counts = list(balanced_totals.values())
        
        if original_counts and balanced_counts:
            original_std = np.std(original_counts)
            balanced_std = np.std(balanced_counts)
            original_cv = original_std / np.mean(original_counts) if np.mean(original_counts) > 0 else 0
            balanced_cv = balanced_std / np.mean(balanced_counts) if np.mean(balanced_counts) > 0 else 0
            
            report['balance_metrics'] = {
                'original_standard_deviation': float(original_std),
                'balanced_standard_deviation': float(balanced_std),
                'original_coefficient_of_variation': float(original_cv),
                'balanced_coefficient_of_variation': float(balanced_cv),
                'balance_improvement': float(original_cv - balanced_cv) if original_cv > 0 else 0
            }
        
        # Save report
        report_path = self.output_path / 'balance_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📄 Balance report saved to: {report_path}")
        
        # Print summary
        print("\n" + "="*80)
        print("🎯 DATASET BALANCING SUMMARY")
        print("="*80)
        
        if 'balance_metrics' in report:
            metrics = report['balance_metrics']
            print(f"📊 Original Coefficient of Variation: {metrics['original_coefficient_of_variation']:.4f}")
            print(f"📊 Balanced Coefficient of Variation: {metrics['balanced_coefficient_of_variation']:.4f}")
            print(f"📈 Balance Improvement: {metrics['balance_improvement']:.4f}")
        
        print(f"📁 Original dataset: {self.dataset_path}")
        print(f"📁 Balanced dataset: {self.output_path}")
        print(f"📄 Dataset YAML: {self.output_path / 'dataset.yaml'}")
        print("="*80)
    
    def balance_dataset(self, target_samples_per_class=None, seed=42):
        """Main method to balance the dataset"""
        print("🚀 Starting dataset balancing process...")
        print(f"📁 Original dataset: {self.dataset_path}")
        print(f"📁 Output dataset: {self.output_path}")
        
        # Set random seed for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        
        # Step 1: Analyze original dataset
        self.analyze_dataset()
        
        # Step 2: Plot original distribution
        original_distribution = self.plot_original_distribution()
        
        # Step 3: Calculate balanced distribution
        balanced_distribution = self.calculate_balanced_samples(target_samples_per_class)
        
        if balanced_distribution is None:
            print("❌ Failed to calculate balanced distribution!")
            return False
        
        # Step 4: Select balanced samples
        selected_files = self.select_balanced_samples(balanced_distribution)
        
        # Step 5: Create balanced dataset
        self.create_balanced_dataset(selected_files, balanced_distribution)
        
        # Step 6: Create dataset YAML
        self.create_balanced_dataset_yaml()
        
        # Step 7: Analyze balanced dataset
        self.analyze_balanced_dataset()
        
        # Step 8: Plot balanced distribution
        self.plot_balanced_distribution()
        
        # Step 9: Generate report
        self.generate_balance_report()
        
        print("\n✅ Dataset balancing completed successfully!")
        return True

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Balance YOLO dataset for equal class distribution')
    parser.add_argument('--dataset-path', type=str, default='./dataset', 
                       help='Path to original dataset')
    parser.add_argument('--output-path', type=str, default='./dataset_balanced',
                       help='Path to output balanced dataset')
    parser.add_argument('--target-samples', type=int, default=None,
                       help='Target number of samples per class (default: minimum class count)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Create balancer
    balancer = DatasetBalancer(
        dataset_path=args.dataset_path,
        output_path=args.output_path
    )
    
    # Balance dataset
    success = balancer.balance_dataset(
        target_samples_per_class=args.target_samples,
        seed=args.seed
    )
    
    if success:
        print("\n🎉 Dataset balancing completed successfully!")
        print(f"📁 Use the balanced dataset at: {args.output_path}")
        print(f"📄 Update your training script to use: {args.output_path}/dataset.yaml")
    else:
        print("\n❌ Dataset balancing failed!")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 