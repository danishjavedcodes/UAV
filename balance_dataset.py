"""
Class Balancing Script for HMAY-TSF
Creates a balanced dataset to address severe class imbalance
"""

import os
import sys
from pathlib import Path
import argparse
from data_preparation import ClassBalancedAugmentation, create_balanced_dataset_yaml

def main():
    parser = argparse.ArgumentParser(description='Balance dataset classes')
    parser.add_argument('--dataset_path', type=str, default='./dataset', 
                       help='Path to original dataset')
    parser.add_argument('--output_path', type=str, default='./dataset_balanced',
                       help='Path to output balanced dataset')
    parser.add_argument('--target_samples', type=int, default=5000,
                       help='Target samples per class')
    parser.add_argument('--skip_balancing', action='store_true',
                       help='Skip balancing and only create YAML')
    
    args = parser.parse_args()
    
    print("="*80)
    print("HMAY-TSF CLASS BALANCING SYSTEM")
    print("="*80)
    
    # Check if dataset exists
    if not os.path.exists(args.dataset_path):
        print(f"Error: Dataset path {args.dataset_path} does not exist!")
        return
    
    # Initialize class balancing system
    print(f"Initializing class balancing system...")
    balancer = ClassBalancedAugmentation(
        dataset_path=args.dataset_path,
        target_samples_per_class=args.target_samples
    )
    
    if not args.skip_balancing:
        # Create balanced dataset
        print(f"\nCreating balanced dataset...")
        print(f"Original dataset: {args.dataset_path}")
        print(f"Balanced dataset: {args.output_path}")
        print(f"Target samples per class: {args.target_samples}")
        
        balancer.create_balanced_dataset(output_path=args.output_path)
        
        print(f"\nBalanced dataset created successfully!")
        print(f"Location: {args.output_path}")
    
    # Create balanced dataset YAML
    print(f"\nCreating balanced dataset YAML...")
    yaml_path = create_balanced_dataset_yaml(args.output_path)
    
    print(f"\nBalanced dataset YAML created: {yaml_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("CLASS BALANCING SUMMARY")
    print("="*80)
    
    print("Original Class Distribution:")
    for class_id in sorted(balancer.class_counts.keys()):
        count = balancer.class_counts[class_id]
        percentage = (count / sum(balancer.class_counts.values())) * 100
        print(f"  Class {class_id}: {count:,} ({percentage:.2f}%)")
    
    print(f"\nClass Weights Applied:")
    for class_id in sorted(balancer.class_counts.keys()):
        weight = balancer.class_weights[class_id]
        print(f"  Class {class_id}: {weight:.3f}")
    
    print(f"\nImbalance Ratio: {max(balancer.class_counts.values()) / min(balancer.class_counts.values()):.2f}")
    print(f"Average Class Weight: {sum(balancer.class_weights.values()) / len(balancer.class_weights):.3f}")
    
    print(f"\nNext Steps:")
    print(f"1. Use the balanced dataset for training:")
    print(f"   python train_hmay_tsf.py --data {yaml_path}")
    print(f"2. The training script will automatically use class weights and balanced sampling")
    print(f"3. Monitor class balancing metrics during training")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    main() 