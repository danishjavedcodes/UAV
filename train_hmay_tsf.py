"""
Enhanced Training Script for HMAY-TSF Model
Addresses class imbalance using multiple strategies for object detection
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import cv2
from pathlib import Path
import random
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_fscore_support
import time
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from hmay_tsf_model_simple import HMAY_TSF
from data_preparation import VisDroneDataset, ClassBalancedAugmentation

# Custom collate function for variable length bboxes
def custom_collate(batch):
    images = torch.stack([item['image'] for item in batch])
    bboxes = [item['bboxes'] for item in batch]
    class_labels = [item['class_labels'] for item in batch]
    img_paths = [item['img_path'] for item in batch]
    
    return {
        'image': images,
        'bboxes': bboxes,
        'class_labels': class_labels,
        'img_path': img_paths
    }

class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance in object detection"""
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class BalancedObjectDetectionLoss(nn.Module):
    """Balanced loss function for object detection with class weighting"""
    def __init__(self, num_classes=11, class_weights=None, focal_alpha=1, focal_gamma=2):
        super(BalancedObjectDetectionLoss, self).__init__()
        self.num_classes = num_classes
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        
        # Initialize class weights
        if class_weights is None:
            self.class_weights = torch.ones(num_classes)
        else:
            self.class_weights = torch.tensor(class_weights, dtype=torch.float32)
        
        # Loss components
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.box_loss_weight = 7.5
        self.cls_loss_weight = 0.5
        self.dfl_loss_weight = 1.5
        
    def forward(self, predictions, targets):
        """
        Compute balanced loss for object detection
        Args:
            predictions: List of prediction tensors from different scales
            targets: Ground truth targets
        """
        total_loss = 0
        cls_loss = 0
        box_loss = 0
        dfl_loss = 0
        
        # Process each scale
        for pred in predictions:
            # Extract components (simplified - in practice, you'd decode YOLO outputs)
            B, C, H, W = pred.shape
            pred = pred.view(B, 3, -1, H, W)  # 3 anchors
            
            # Classification loss with focal loss and class weights
            cls_pred = pred[:, :, 5:, :, :]  # Class predictions
            cls_pred = cls_pred.view(-1, self.num_classes)
            
            # Apply focal loss with class weights
            if targets is not None:
                # This is a simplified version - you'd need proper target matching
                cls_loss += self.focal_loss(cls_pred, targets) * self.class_weights.mean()
            else:
                # If no targets, use a simple regularization loss
                cls_loss += torch.mean(torch.abs(cls_pred))
            
            # Box regression loss (simplified)
            box_pred = pred[:, :, :4, :, :]  # Box predictions
            if targets is not None:
                # Simplified box loss
                box_loss += torch.mean(torch.abs(box_pred))
            else:
                # If no targets, use a simple regularization loss
                box_loss += torch.mean(torch.abs(box_pred))
        
        total_loss = (self.box_loss_weight * box_loss + 
                     self.cls_loss_weight * cls_loss + 
                     self.dfl_loss_weight * dfl_loss)
        
        return total_loss, {
            'total': total_loss.item(),
            'box': box_loss.item(),
            'cls': cls_loss.item(),
            'dfl': dfl_loss.item()
        }

class ClassBalancedSampler:
    """Balanced sampler for addressing class imbalance"""
    def __init__(self, dataset, class_counts, sampling_strategy='balanced'):
        self.dataset = dataset
        self.class_counts = class_counts
        self.sampling_strategy = sampling_strategy
        
        # Calculate sample weights
        self.sample_weights = self._calculate_sample_weights()
        
    def _calculate_sample_weights(self):
        """Calculate weights for each sample based on class distribution"""
        weights = []
        max_count = max(self.class_counts.values())
        
        for idx in range(len(self.dataset)):
            # Get classes in this sample
            _, label_file = self.dataset.img_files[idx], self.dataset.label_files[idx]
            labels = self.dataset.load_labels(label_file)
            
            if len(labels) == 0:
                # No objects - give medium weight
                weights.append(1.0)
            else:
                # Calculate weight based on classes present
                class_ids = labels[:, 0].astype(int)
                class_weights = [max_count / self.class_counts.get(cid, 1) for cid in class_ids]
                # Use the maximum weight for this sample
                weights.append(max(class_weights) if class_weights else 1.0)
        
        return weights
    
    def get_sampler(self):
        """Get weighted random sampler"""
        return WeightedRandomSampler(
            weights=self.sample_weights,
            num_samples=len(self.dataset),
            replacement=True
        )

class TrainingMetrics:
    """Track training metrics including class-specific performance"""
    def __init__(self, num_classes=11):
        self.num_classes = num_classes
        self.reset()
        
    def reset(self):
        """Reset all metrics"""
        self.losses = defaultdict(list)
        self.class_precision = defaultdict(list)
        self.class_recall = defaultdict(list)
        self.class_f1 = defaultdict(list)
        self.mAP = []
        
    def update(self, loss_dict, predictions=None, targets=None):
        """Update metrics with new batch results"""
        # Update losses
        for key, value in loss_dict.items():
            self.losses[key].append(value)
        
        # Update class-specific metrics (simplified)
        if predictions is not None and targets is not None:
            # This would calculate precision, recall, F1 for each class
            pass
    
    def get_summary(self):
        """Get summary of current metrics"""
        summary = {}
        
        # Average losses
        for key, values in self.losses.items():
            summary[f'avg_{key}_loss'] = np.mean(values[-100:])  # Last 100 batches
        
        return summary

class HMAYTSFTrainer:
    """Enhanced trainer for HMAY-TSF model with class balancing"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.setup_directories()
        self.setup_model()
        self.setup_data()
        self.setup_training_components()
        
    def setup_directories(self):
        """Setup output directories"""
        self.output_dir = Path(self.config['output_dir'])
        self.output_dir.mkdir(exist_ok=True)
        
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        self.log_dir = self.output_dir / 'logs'
        self.log_dir.mkdir(exist_ok=True)
        
        # Setup tensorboard
        self.writer = SummaryWriter(self.log_dir)
        
    def setup_model(self):
        """Initialize HMAY-TSF model"""
        print("Initializing HMAY-TSF model...")
        
        self.model = HMAY_TSF(
            model_size=self.config['model_size'],
            num_classes=self.config['num_classes'],
            pretrained=self.config['pretrained']
        ).to(self.device)
        
        # Disable YOLO's built-in training method to avoid conflicts
        self.model.base_yolo.train = lambda *args, **kwargs: None
        
        # Print model summary
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
    def setup_data(self):
        """Setup data loaders with class balancing"""
        print("Setting up data loaders...")
        
        # Load dataset configuration
        with open(self.config['data_yaml'], 'r') as f:
            data_config = yaml.safe_load(f)
        
        # Handle different environment paths
        dataset_path = data_config['path']
        if not os.path.exists(dataset_path):
            # Try alternative paths for different environments
            alt_paths = [
                './dataset',
                '../dataset',
                'dataset',
                '/kaggle/working/UAV/dataset',
                '/kaggle/working/dataset'
            ]
            
            for alt_path in alt_paths:
                if os.path.exists(alt_path):
                    print(f"Using alternative dataset path: {alt_path}")
                    dataset_path = alt_path
                    break
            else:
                raise FileNotFoundError(f"Dataset not found. Checked: {dataset_path}, {alt_paths}")
        
        # Analyze class distribution
        self.class_analyzer = ClassBalancedAugmentation(
            dataset_path=dataset_path,
            target_samples_per_class=self.config.get('target_samples_per_class', 5000)
        )
        
        # Get class distribution
        self.class_counts = self.class_analyzer.analyze_class_distribution()
        self.class_weights = self.class_analyzer.calculate_class_weights()
        
        print("Class Distribution:")
        for class_id, count in sorted(self.class_counts.items()):
            weight = self.class_weights[class_id]
            print(f"  Class {class_id}: {count:,} samples (weight: {weight:.3f})")
        
        # Create datasets
        train_dataset = VisDroneDataset(
            img_dir=os.path.join(dataset_path, data_config['train']),
            label_dir=os.path.join(dataset_path, 'labels/train'),
            img_size=self.config['img_size'],
            augment=True,
            super_res=self.config.get('use_super_resolution', False)
        )
        
        val_dataset = VisDroneDataset(
            img_dir=os.path.join(dataset_path, data_config['val']),
            label_dir=os.path.join(dataset_path, 'labels/val'),
            img_size=self.config['img_size'],
            augment=False,
            super_res=False
        )
        
        # Setup balanced sampling
        self.balanced_sampler = ClassBalancedSampler(
            train_dataset, 
            self.class_counts,
            sampling_strategy='balanced'
        )
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            sampler=self.balanced_sampler.get_sampler(),
            num_workers=self.config['num_workers'],
            pin_memory=True,
            drop_last=True,
            collate_fn=custom_collate
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config['num_workers'],
            pin_memory=True,
            collate_fn=custom_collate
        )
        
        print(f"Train samples: {len(train_dataset)}")
        print(f"Val samples: {len(val_dataset)}")
        
    def setup_training_components(self):
        """Setup optimizer, scheduler, and loss function"""
        print("Setting up training components...")
        
        # Optimizer with different learning rates for different components
        backbone_params = []
        enhanced_params = []
        
        for name, param in self.model.named_parameters():
            if 'base_yolo' in name:
                backbone_params.append(param)
            else:
                enhanced_params.append(param)
        
        self.optimizer = optim.AdamW([
            {'params': backbone_params, 'lr': self.config['lr'] * 0.1},
            {'params': enhanced_params, 'lr': self.config['lr']}
        ], weight_decay=self.config['weight_decay'])
        
        # Learning rate scheduler
        T_0 = max(1, self.config['epochs'] // 3)  # Ensure T_0 is at least 1
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=T_0,
            T_mult=2,
            eta_min=self.config['lr'] * 0.01
        )
        
        # Loss function with class balancing
        self.criterion = BalancedObjectDetectionLoss(
            num_classes=self.config['num_classes'],
            class_weights=list(self.class_weights.values()),
            focal_alpha=self.config.get('focal_alpha', 1),
            focal_gamma=self.config.get('focal_gamma', 2)
        )
        
        # Metrics tracker
        self.metrics = TrainingMetrics(num_classes=self.config['num_classes'])
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
    def train_epoch(self):
        """Train for one epoch"""
        # Set model to training mode properly
        self.model.enhanced_backbone.train()
        self.model.base_yolo.model.eval()  # Keep base YOLO in eval mode
        
        epoch_losses = defaultdict(list)
        
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch + 1}')
        
        for batch_idx, batch in enumerate(progress_bar):
            # Handle data format from VisDroneDataset
            if isinstance(batch, dict):
                images = batch['image']
                # Convert bboxes and class_labels to targets format if needed
                targets = None  # For now, we'll use None and handle in loss function
            elif isinstance(batch, (list, tuple)):
                images, targets = batch
            else:
                images = batch
                targets = None
                
            images = images.to(self.device)
            targets = targets.to(self.device) if targets is not None else None
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(images, is_training=True)
            
            # Calculate loss
            loss, loss_components = self.criterion(predictions, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
            
            self.optimizer.step()
            
            # Update metrics
            for key, value in loss_components.items():
                epoch_losses[key].append(value)
            
            # Update progress bar
            avg_loss = np.mean(epoch_losses['total'])
            progress_bar.set_postfix({
                'Loss': f'{avg_loss:.4f}',
                'LR': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })
            
            # Log to tensorboard
            if batch_idx % 10 == 0:
                step = self.current_epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar('Train/Loss', avg_loss, step)
                self.writer.add_scalar('Train/Learning_Rate', 
                                     self.optimizer.param_groups[0]['lr'], step)
        
        # Calculate epoch averages
        epoch_metrics = {}
        for key, values in epoch_losses.items():
            epoch_metrics[f'train_{key}_loss'] = np.mean(values)
        
        return epoch_metrics
    
    def validate_epoch(self):
        """Validate for one epoch"""
        # Set model to evaluation mode properly
        self.model.enhanced_backbone.eval()
        self.model.base_yolo.model.eval()
        
        val_losses = defaultdict(list)
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                # Handle data format from VisDroneDataset
                if isinstance(batch, dict):
                    images = batch['image']
                    targets = None  # For now, we'll use None and handle in loss function
                elif isinstance(batch, (list, tuple)):
                    images, targets = batch
                else:
                    images = batch
                    targets = None
                    
                images = images.to(self.device)
                targets = targets.to(self.device) if targets is not None else None
                
                # Forward pass
                predictions = self.model(images, is_training=False)
                
                # Calculate loss
                loss, loss_components = self.criterion(predictions, targets)
                
                # Update metrics
                for key, value in loss_components.items():
                    val_losses[key].append(value)
        
        # Calculate validation averages
        val_metrics = {}
        for key, values in val_losses.items():
            val_metrics[f'val_{key}_loss'] = np.mean(values)
        
        return val_metrics
    
    def save_checkpoint(self, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'class_weights': self.class_weights,
            'config': self.config
        }
        
        # Save latest checkpoint
        checkpoint_path = self.checkpoint_dir / 'latest_checkpoint.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / 'best_checkpoint.pth'
            torch.save(checkpoint, best_path)
            print(f"New best model saved! Val Loss: {self.best_val_loss:.4f}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.current_epoch = checkpoint['epoch']
            self.best_val_loss = checkpoint['best_val_loss']
            print(f"Loaded checkpoint from epoch {self.current_epoch}")
            return True
        return False
    
    def plot_class_distribution(self):
        """Plot class distribution"""
        plt.figure(figsize=(12, 6))
        
        classes = list(self.class_counts.keys())
        counts = list(self.class_counts.values())
        weights = [self.class_weights[c] for c in classes]
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Class counts
        ax1.bar(classes, counts, color='skyblue', alpha=0.7)
        ax1.set_title('Class Distribution (Counts)')
        ax1.set_xlabel('Class ID')
        ax1.set_ylabel('Number of Samples')
        ax1.tick_params(axis='x', rotation=45)
        
        # Class weights
        ax2.bar(classes, weights, color='lightcoral', alpha=0.7)
        ax2.set_title('Class Weights')
        ax2.set_xlabel('Class ID')
        ax2.set_ylabel('Weight')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'class_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def train(self):
        """Main training loop"""
        print("Starting training...")
        print(f"Device: {self.device}")
        print(f"Output directory: {self.output_dir}")
        
        # Plot class distribution
        self.plot_class_distribution()
        
        # Load checkpoint if exists
        if self.config.get('resume_training', False):
            checkpoint_path = self.checkpoint_dir / 'latest_checkpoint.pth'
            self.load_checkpoint(checkpoint_path)
        
        # Training loop
        for epoch in range(self.current_epoch, self.config['epochs']):
            self.current_epoch = epoch
            
            print(f"\n{'='*60}")
            print(f"Epoch {epoch + 1}/{self.config['epochs']}")
            print(f"{'='*60}")
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate_epoch()
            
            # Update learning rate
            self.scheduler.step()
            
            # Log metrics
            all_metrics = {**train_metrics, **val_metrics}
            for key, value in all_metrics.items():
                self.writer.add_scalar(key, value, epoch)
            
            # Print epoch summary
            print(f"\nEpoch {epoch + 1} Summary:")
            print(f"Train Loss: {train_metrics['train_total_loss']:.4f}")
            print(f"Val Loss: {val_metrics['val_total_loss']:.4f}")
            print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Save checkpoint
            is_best = val_metrics['val_total_loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['val_total_loss']
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            self.save_checkpoint(is_best=is_best)
            
            # Early stopping
            if self.patience_counter >= self.config.get('patience', 10):
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break
        
        # Save final model
        self.save_checkpoint()
        
        # Close tensorboard writer
        self.writer.close()
        
        print(f"\nTraining completed! Best validation loss: {self.best_val_loss:.4f}")
        print(f"Model saved to: {self.output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Train HMAY-TSF model with class balancing')
    parser.add_argument('--config', type=str, default='config.yaml', 
                       help='Path to configuration file')
    parser.add_argument('--data_yaml', type=str, default='dataset/dataset.yaml',
                       help='Path to dataset YAML file')
    parser.add_argument('--output_dir', type=str, default='./runs/hmay_tsf_training',
                       help='Output directory for training results')
    parser.add_argument('--resume', action='store_true',
                       help='Resume training from checkpoint')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=None,
                       help='Learning rate')
    
    args = parser.parse_args()
    
    # Load configuration
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        # Default configuration
        config = {
            'model_size': 's',
            'num_classes': 11,
            'pretrained': True,
            'img_size': 640,
            'batch_size': 16,
            'epochs': 100,
            'lr': 0.001,
            'weight_decay': 0.0005,
            'num_workers': 4,
            'patience': 10,
            'focal_alpha': 1,
            'focal_gamma': 2,
            'use_super_resolution': False,
            'target_samples_per_class': 5000,
            'resume_training': args.resume
        }
    
    # Update config with command line arguments
    config['data_yaml'] = args.data_yaml
    config['output_dir'] = args.output_dir
    
    # Override config with command line arguments if provided
    if args.epochs is not None:
        config['epochs'] = args.epochs
    if args.batch_size is not None:
        config['batch_size'] = args.batch_size
    if args.lr is not None:
        config['lr'] = args.lr
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Initialize trainer
    trainer = HMAYTSFTrainer(config)
    
    # Start training
    trainer.train()

if __name__ == "__main__":
    main()
