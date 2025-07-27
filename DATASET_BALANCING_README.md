# Dataset Balancing for HMAY-TSF Model

This document explains the comprehensive dataset balancing functionality implemented in the HMAY-TSF training script to ensure equal class distribution across all 11 classes in the VisDrone dataset.

## Overview

The dataset balancing system addresses class imbalance issues by:
1. **Analyzing** the current class distribution in your dataset
2. **Balancing** classes through intelligent sampling and augmentation
3. **Verifying** the balance quality
4. **Visualizing** the before/after distribution

## Features

### 🔍 **Class Distribution Analysis**
- Detailed analysis of current class distribution
- Identification of imbalanced classes
- Calculation of balance ratios
- Comprehensive reporting

### ⚖️ **Intelligent Balancing**
- **Oversampling**: For underrepresented classes
- **Undersampling**: For overrepresented classes  
- **Augmentation**: Advanced data augmentation for underrepresented classes
- **Target-based**: Configurable target samples per class

### 📊 **Visualization & Reporting**
- Before/after distribution plots
- Balance quality assessment
- Detailed metrics reporting
- CSV logging of all metrics

### 🎯 **Integration**
- Seamless integration with training pipeline
- Standalone balancing option
- Configurable parameters
- Automatic verification

## Quick Start

### 1. Basic Usage (with training)

```bash
# Run training with automatic dataset balancing
python train_hmay_tsf.py --epochs 10 --batch-size 8

# Run with custom target samples per class
python train_hmay_tsf.py --epochs 10 --target-samples 1000

# Disable dataset balancing
python train_hmay_tsf.py --disable-balancing --epochs 10
```

### 2. Standalone Dataset Balancing

```bash
# Balance dataset only (without training)
python train_hmay_tsf.py --balance-only

# Balance with custom target samples
python train_hmay_tsf.py --balance-only --target-samples 800

# Balance with custom dataset path
python train_hmay_tsf.py --balance-only --dataset-path ./my_dataset --target-samples 600
```

### 3. Test the Functionality

```bash
# Run comprehensive tests
python test_dataset_balancing.py
```

## Detailed Usage

### Command Line Arguments

| Argument | Description | Default | Example |
|----------|-------------|---------|---------|
| `--enable-balancing` | Enable dataset balancing | `True` | `--enable-balancing` |
| `--disable-balancing` | Disable dataset balancing | `False` | `--disable-balancing` |
| `--target-samples` | Target samples per class | `Auto` | `--target-samples 1000` |
| `--balance-only` | Run balancing without training | `False` | `--balance-only` |
| `--dataset-path` | Custom dataset path | `./dataset` | `--dataset-path ./my_data` |

### Python API Usage

```python
from train_hmay_tsf import DatasetBalancer, AdvancedHMAYTSFTrainer

# Initialize dataset balancer
balancer = DatasetBalancer(
    dataset_path='./dataset',
    target_samples_per_class=1000
)

# Analyze current distribution
analysis = balancer.analyze_class_distribution('train')

# Create balanced dataset
balanced_yaml = balancer.create_balanced_dataset()

# Initialize trainer with balancing
trainer = AdvancedHMAYTSFTrainer(
    model_size='s',
    enable_dataset_balancing=True,
    target_samples_per_class=1000
)

# Train with balanced dataset
trainer.train_model('./dataset/dataset.yaml', epochs=10)
```

## How It Works

### 1. Analysis Phase
```
Original Dataset Analysis:
├── Count instances per class
├── Calculate distribution percentages
├── Identify imbalance ratios
└── Generate detailed report
```

### 2. Balancing Phase
```
For each class:
├── If class_count < target:
│   ├── Use all available samples
│   ├── Apply augmentation to reach target
│   └── Generate augmented samples
├── If class_count > target:
│   ├── Randomly sample target_count samples
│   └── Ensure diversity in selection
└── If class_count == target:
    └── Use as-is
```

### 3. Augmentation Strategy
The system applies intelligent augmentation for underrepresented classes:

- **Geometric Augmentations**: Rotation, scaling, translation
- **Color Augmentations**: Brightness, contrast, hue, saturation
- **Noise Augmentations**: Gaussian noise, blur effects
- **Advanced Augmentations**: Elastic transform, grid distortion
- **Weather Effects**: Rain, fog, sun flare (when available)

### 4. Verification Phase
```
Balance Verification:
├── Count instances in balanced dataset
├── Calculate new balance ratios
├── Assess balance quality
└── Generate verification report
```

## Output Structure

After balancing, your dataset will have this structure:

```
dataset/
├── balanced_dataset/           # New balanced dataset
│   ├── images/
│   │   ├── train/             # Balanced training images
│   │   ├── val/               # Original validation images
│   │   └── test/              # Original test images
│   ├── labels/
│   │   ├── train/             # Balanced training labels
│   │   ├── val/               # Original validation labels
│   │   └── test/              # Original test labels
│   └── dataset.yaml           # Balanced dataset config
├── class_distribution_comparison.png  # Before/after plot
└── balance_report.json        # Detailed balance report
```

## Balance Quality Assessment

The system automatically assesses balance quality:

| Balance Ratio | Quality | Description |
|---------------|---------|-------------|
| ≤ 1.5:1 | ✅ Excellent | Perfect balance achieved |
| ≤ 2.0:1 | ✅ Good | Good balance, suitable for training |
| ≤ 3.0:1 | ⚠️ Moderate | Acceptable but could be improved |
| > 3.0:1 | ❌ Poor | Needs further balancing |

## Configuration Options

### Target Samples Strategy

1. **Automatic (Default)**: Uses median class count as target
2. **Manual**: Specify exact target samples per class
3. **Minimum**: Ensures at least 500 samples per class

### Augmentation Intensity

The system automatically adjusts augmentation intensity based on:
- How far a class is from the target
- Available samples for that class
- Overall dataset characteristics

## Performance Impact

### Storage Requirements
- **Original dataset**: ~X GB
- **Balanced dataset**: ~2-3X GB (depending on augmentation)
- **Temporary files**: ~X GB during processing

### Processing Time
- **Analysis**: 1-5 minutes (depending on dataset size)
- **Balancing**: 10-60 minutes (depending on imbalance severity)
- **Verification**: 1-3 minutes

### Memory Usage
- **Peak memory**: ~2-4 GB during augmentation
- **Recommended RAM**: 8+ GB for large datasets

## Troubleshooting

### Common Issues

1. **"No samples for class X"**
   - **Cause**: Class has zero instances in dataset
   - **Solution**: Check dataset labels or use data collection

2. **"Balance ratio still high"**
   - **Cause**: Severe imbalance or insufficient augmentation
   - **Solution**: Increase target samples or add more source data

3. **"Memory error during balancing"**
   - **Cause**: Large dataset or insufficient RAM
   - **Solution**: Reduce batch size or use smaller target samples

4. **"Augmentation failed"**
   - **Cause**: Corrupted images or invalid labels
   - **Solution**: Check dataset integrity and label format

### Debug Mode

Enable verbose logging for debugging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# This will show detailed progress and any issues
balancer = DatasetBalancer(dataset_path='./dataset')
```

## Best Practices

### 1. Dataset Preparation
- Ensure all images have corresponding labels
- Verify label format is YOLO-compatible
- Check for corrupted files before balancing

### 2. Target Selection
- Start with automatic target calculation
- Adjust based on your specific needs
- Consider computational resources

### 3. Quality Control
- Always verify balance after creation
- Check augmented image quality
- Monitor training performance with balanced dataset

### 4. Resource Management
- Monitor disk space during balancing
- Ensure sufficient RAM for augmentation
- Consider using SSD for faster I/O

## Integration with Training

The balanced dataset integrates seamlessly with the training pipeline:

```python
# Training automatically uses balanced dataset
trainer = AdvancedHMAYTSFTrainer(enable_dataset_balancing=True)
results = trainer.train_model('./dataset/dataset.yaml')

# Training metrics include balance information
print(f"Balance ratio: {trainer.best_metrics.get('balance_ratio', 'N/A')}")
```

## Monitoring and Logging

The system provides comprehensive logging:

- **CSV logs**: Detailed metrics for each epoch
- **JSON reports**: Balance analysis and training summary
- **Visualization**: Distribution plots and comparison charts
- **Console output**: Real-time progress and status updates

## Advanced Features

### Custom Augmentation Pipelines
You can customize augmentation strategies:

```python
# Modify augmentation in DatasetBalancer class
def _apply_balancing_augmentation(self, image, labels, target_class_id):
    # Add your custom augmentation logic here
    pass
```

### Multi-Class Balancing
The system handles all 11 VisDrone classes:
- ignored regions (0)
- pedestrian (1)
- people (2)
- bicycle (3)
- car (4)
- van (5)
- truck (6)
- tricycle (7)
- awning-tricycle (8)
- bus (9)
- motor (10)

### Progressive Balancing
For very large datasets, you can implement progressive balancing:

```python
# Balance in stages for large datasets
for stage in range(3):
    target = base_target * (stage + 1)
    balancer.create_balanced_dataset(target_samples_per_class=target)
```

## Support and Maintenance

### Version Compatibility
- **Python**: 3.7+
- **PyTorch**: 1.8+
- **Albumentations**: 1.0+
- **OpenCV**: 4.5+

### Updates and Improvements
The dataset balancing system is actively maintained with:
- Performance optimizations
- New augmentation techniques
- Better balance algorithms
- Enhanced visualization

### Contributing
To contribute improvements:
1. Test with `test_dataset_balancing.py`
2. Ensure balance quality metrics improve
3. Update documentation
4. Submit pull request

---

For questions or issues, please refer to the main project documentation or create an issue in the repository. 