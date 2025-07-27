# Class Balancing System for HMAY-TSF

## Overview

This document describes the comprehensive class balancing system implemented to address severe class imbalance in the UAV traffic detection dataset. The system combines multiple techniques to ensure fair representation of all classes during training.

## Problem Statement

The original dataset has severe class imbalance:
- **Class 3 (bicycle)**: 144,867 instances (42.2%)
- **Class 4 (car)**: 24,956 instances (7.3%)
- **Class 5 (van)**: 12,875 instances (3.8%)
- **Class 6 (truck)**: 4,812 instances (1.4%)
- **Class 7 (tricycle)**: 3,246 instances (0.9%)
- **Class 8 (awning-tricycle)**: 5,926 instances (1.7%)
- **Class 9 (bus)**: 29,647 instances (8.6%)

This imbalance ratio of **44.6:1** (144,867:3,246) causes the model to be heavily biased towards the majority class.

## Solution Components

### 1. Class Balancing System (`ClassBalancingSystem`)

**Location**: `train_hmay_tsf.py`

**Features**:
- **Automatic class distribution analysis**
- **Inverse frequency class weights calculation**
- **Weighted random sampling**
- **Real-time class balancing metrics**

**Usage**:
```python
# Automatically initialized in trainer
class_balancing = ClassBalancingSystem(dataset_path='./dataset', num_classes=11)
```

### 2. Advanced Focal Loss with Class Weights

**Location**: `train_hmay_tsf.py` - `AdvancedFocalLoss` class

**Features**:
- **Class-weighted cross-entropy loss**
- **Balanced focal loss for severe imbalance**
- **Label smoothing for better generalization**
- **Dynamic weight adjustment**

**Configuration**:
```python
focal_loss = AdvancedFocalLoss(
    alpha=1, 
    gamma=2, 
    label_smoothing=0.1, 
    class_weights=class_weights,
    use_balanced_focal=True
)
```

### 3. Dataset Balancing Augmentation

**Location**: `data_preparation.py` - `ClassBalancedAugmentation` class

**Features**:
- **Oversampling of rare classes**
- **Undersampling of common classes**
- **Class-specific augmentation strategies**
- **Synthetic data generation**

**Techniques**:
- **Vehicle-specific augmentations** for classes 4, 5, 6, 9 (car, van, truck, bus)
- **General augmentations** for other classes
- **Brightness/contrast adjustments**
- **Noise and blur effects**
- **Geometric transformations**

### 4. Balanced Dataset Creation

**Location**: `balance_dataset.py`

**Features**:
- **Complete dataset rebalancing**
- **Target samples per class configuration**
- **Automatic YAML generation**
- **Progress monitoring**

## Usage Instructions

### Step 1: Analyze Current Class Distribution

```bash
python balance_dataset.py --dataset_path ./dataset --skip_balancing
```

This will analyze the current class distribution without creating a balanced dataset.

### Step 2: Create Balanced Dataset

```bash
python balance_dataset.py --dataset_path ./dataset --output_path ./dataset_balanced --target_samples 5000
```

**Parameters**:
- `--dataset_path`: Path to original dataset
- `--output_path`: Path for balanced dataset
- `--target_samples`: Target samples per class (default: 5000)
- `--skip_balancing`: Skip balancing, only create YAML

### Step 3: Train with Balanced Dataset

```bash
python train_hmay_tsf.py --data ./dataset_balanced/dataset_balanced.yaml
```

The training script automatically:
- Uses class weights in focal loss
- Applies balanced sampling
- Monitors class balancing metrics
- Logs imbalance ratios

## Configuration

### Class Balancing Settings in `config.yaml`

```yaml
methodology:
  use_class_balancing: true  # Enable class balancing

advanced:
  focal_loss:
    use_balanced_focal: true
    class_weights: true
```

### Class Weights Calculation

The system calculates class weights using inverse frequency with smoothing:

```python
weight = sqrt(max_count / class_count)
weight = min(weight, 10.0)  # Cap maximum weight
```

**Example weights**:
- Class 3 (bicycle): 1.0 (majority class)
- Class 7 (tricycle): 6.7 (rare class)
- Class 6 (truck): 5.5 (rare class)

## Monitoring and Metrics

### Training Metrics

The system tracks class balancing metrics during training:

- **Class Imbalance Ratio**: Ratio of most common to least common class
- **Average Class Weight**: Mean of all class weights
- **Per-class performance metrics**

### CSV Logging

Class balancing metrics are logged to `advanced_training_metrics.csv`:

```csv
epoch,class_imbalance_ratio,avg_class_weight,...
1,44.6,3.2,...
2,44.6,3.2,...
...
```

### Console Output

During training, you'll see:

```
CLASS BALANCING METRICS:
  Class Imbalance Ratio: 44.60
  Average Class Weight: 3.245
```

## Methodology Compliance

This class balancing system fully complies with the HMAY-TSF methodology:

### ✅ Super-Resolution Data Augmentation (SRDA)
- Enhanced resolution for small objects
- Improves detection of rare classes

### ✅ Copy-Paste Augmentation Scheme
- Increases representation of small objects
- Mitigates size imbalance issues

### ✅ Active Learning Framework (ALF)
- Uncertainty-based sampling
- Reduces annotation costs

### ✅ Advanced Loss Functions
- Focal loss with class balancing
- IoU loss variants

### ✅ Real-Time Optimization
- Efficient class weight calculation
- Minimal computational overhead

## Expected Results

### Before Class Balancing
- **mAP**: ~30-40% (biased towards majority class)
- **Class-wise F1**: Poor performance on rare classes
- **Imbalance Ratio**: 44.6:1

### After Class Balancing
- **mAP**: 40-50% (balanced across all classes)
- **Class-wise F1**: Improved performance on rare classes
- **Imbalance Ratio**: Reduced to ~5:1
- **Fair representation**: All classes contribute equally to training

## Troubleshooting

### Common Issues

1. **Memory Issues**: Reduce `target_samples` parameter
2. **Slow Balancing**: Use `--skip_balancing` for analysis only
3. **Poor Performance**: Check class weight calculations

### Performance Tips

1. **Use SSD storage** for faster dataset creation
2. **Increase RAM** for larger target sample sizes
3. **Monitor GPU memory** during training with balanced sampling

## Advanced Usage

### Custom Class Weights

Modify class weights manually in `train_hmay_tsf.py`:

```python
custom_weights = torch.tensor([1.0, 2.0, 3.0, 1.0, 2.0, 4.0, 5.0, 6.0, 3.0, 2.0, 1.0])
focal_loss = AdvancedFocalLoss(class_weights=custom_weights)
```

### Class-Specific Augmentation

Add custom augmentation strategies in `data_preparation.py`:

```python
def apply_class_specific_augmentation(self, image, labels, class_id, augmentation_id):
    if class_id == 7:  # tricycle
        # Special augmentation for tricycles
        transform = A.Compose([...])
```

## Conclusion

The class balancing system provides a comprehensive solution to address severe class imbalance in UAV traffic detection. By combining multiple techniques, it ensures fair representation of all classes while maintaining the advanced methodology of HMAY-TSF.

For questions or issues, refer to the main methodology document or contact the development team. 