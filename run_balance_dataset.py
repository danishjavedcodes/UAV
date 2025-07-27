#!/usr/bin/env python3
"""
Simple script to run dataset balancing
"""

from balance_dataset import DatasetBalancer
import os

def main():
    """Run dataset balancing"""
    print("🚀 Starting dataset balancing process...")
    
    # Check if original dataset exists
    if not os.path.exists('./dataset'):
        print("❌ Error: Original dataset not found at ./dataset")
        print("Please ensure your dataset is in the ./dataset directory")
        return 1
    
    # Create balancer
    balancer = DatasetBalancer(
        dataset_path='./dataset',
        output_path='./dataset_balanced'
    )
    
    # Balance dataset
    success = balancer.balance_dataset(
        target_samples_per_class=None,  # Use minimum class count
        seed=42  # For reproducibility
    )
    
    if success:
        print("\n🎉 Dataset balancing completed successfully!")
        print("📁 Balanced dataset created at: ./dataset_balanced")
        print("📄 Dataset YAML: ./dataset_balanced/dataset.yaml")
        print("\n💡 Next steps:")
        print("1. The training script has been updated to use the balanced dataset by default")
        print("2. Run training with: python train_hmay_tsf.py")
        print("3. Or specify custom dataset: python train_hmay_tsf.py --data ./dataset_balanced/dataset.yaml")
    else:
        print("\n❌ Dataset balancing failed!")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 