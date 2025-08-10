"""
Evaluation Script for HMAY-TSF Model
====================================

This script loads the trained model from train.py and evaluates it on the test dataset,
calculating all metrics and generating visualizations including:
1. Bar chart for accuracy, precision, recall, f1 score
2. Confusion matrix

USAGE:
======
python eval.py --model-path ./runs/ultra_enhanced_train/best_model.pth --data ./Aerial-Vehicles-1/data.yaml

ARGUMENTS:
==========
--model-path: Path to the trained model checkpoint (default: ./runs/ultra_enhanced_train/best_model.pth)
--data: Path to the dataset YAML file (default: ./Aerial-Vehicles-1/data.yaml)
--batch-size: Batch size for evaluation (default: 8)
--img-size: Image size for evaluation (default: 640)
--device: Device to use (default: cuda)
--save-dir: Directory to save evaluation results (default: ./runs/evaluation)

OUTPUTS:
========
The script will create the following files in the save directory:
1. performance_metrics.png - Bar chart showing accuracy, precision, recall, and F1-score
2. confusion_matrix.png - Confusion matrix heatmap
3. evaluation_results.json - Detailed results in JSON format
4. metrics.csv - Metrics in CSV format
5. confusion_matrix.csv - Confusion matrix in CSV format

EXAMPLE:
========
# Basic usage with default settings
python eval.py

# Custom model path and data location
python eval.py --model-path ./my_model.pth --data ./my_data.yaml

# Custom batch size and device
python eval.py --batch-size 16 --device cpu

# Custom save directory
python eval.py --save-dir ./my_evaluation_results
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the required classes from train.py
try:
    from train import ImprovedHMAYTSF, SimpleDataset, collate_fn, CBAM, SPP, ChannelAttention, SpatialAttention
except ImportError as e:
    print(f"Error importing from train.py: {e}")
    print("Please make sure train.py is in the same directory as eval.py")
    sys.exit(1)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import yaml
import numpy as np
from datetime import datetime
import argparse
import json
import pandas as pd
import csv
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class TestDataset:
    """Test dataset for evaluation"""
    
    def __init__(self, data_yaml_path, img_size=640):
        self.img_size = img_size
        
        # Load dataset configuration
        try:
            with open(data_yaml_path, 'r') as f:
                self.config = yaml.safe_load(f)
        except Exception as e:
            print(f"Error loading YAML file {data_yaml_path}: {e}")
            # Use default configuration
            self.config = {'names': ['bus', 'car', 'truck', 'van'], 'nc': 4}
        
        # Get paths
        if 'path' in self.config:
            self.data_path = Path(self.config['path'])
        else:
            self.data_path = Path(data_yaml_path).parent
        
        # Get test paths
        self.img_dir = self.data_path / 'test' / 'images'
        self.label_dir = self.data_path / 'test' / 'labels'
        
        self.class_names = self.config.get('names', ['bus', 'car', 'truck', 'van'])
        self.num_classes = self.config.get('nc', 4)
        
        # Get all image files
        self.img_files = []
        if self.img_dir.exists():
            self.img_files = list(self.img_dir.glob('*.jpg')) + list(self.img_dir.glob('*.png'))
        else:
            print(f"Warning: Image directory {self.img_dir} does not exist")
            # Try alternative paths
            alt_paths = [
                self.data_path / 'test' / 'images',
                self.data_path / 'images' / 'test',
                self.data_path / 'test'
            ]
            for alt_path in alt_paths:
                if alt_path.exists():
                    self.img_dir = alt_path
                    self.img_files = list(self.img_dir.glob('*.jpg')) + list(self.img_dir.glob('*.png'))
                    print(f"Found images in alternative path: {self.img_dir}")
                    break
        
        print(f"Found {len(self.img_files)} test images in {self.img_dir}")
        print(f"Label directory: {self.label_dir}")
        print(f"Class names: {self.class_names}")
        print(f"Number of classes: {self.num_classes}")
    
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        label_path = self.label_dir / (img_path.stem + '.txt')
        
        # Load image
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                raise ValueError(f"Could not load image: {img_path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a dummy image
            img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        
        # Load labels
        labels = []
        if label_path.exists():
            try:
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            x_center = float(parts[1])
                            y_center = float(parts[2])
                            width = float(parts[3])
                            height = float(parts[4])
                            labels.append([class_id, x_center, y_center, width, height])
            except Exception as e:
                print(f"Error loading labels for {label_path}: {e}")
        
        labels = torch.tensor(labels, dtype=torch.float32) if labels else torch.zeros((0, 5))
        
        # Resize image
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = img.astype(np.float32) / 255.0
        
        # Normalize with ImageNet stats
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = (img - mean) / std
        
        img = torch.from_numpy(img).permute(2, 0, 1)  # HWC to CHW
        
        return img, labels

class ModelEvaluator:
    """Model evaluator for comprehensive evaluation"""
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
        
        self.class_names = ['bus', 'car', 'truck', 'van']
        self.num_classes = 4
        
    def evaluate_model(self, test_loader):
        """Evaluate the model on test data"""
        all_predictions = []
        all_targets = []
        all_pred_boxes = []
        all_target_boxes = []
        
        print("Evaluating model on test data...")
        
        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(tqdm(test_loader, desc="Evaluating")):
                try:
                    images = images.to(self.device)
                    
                    # Forward pass
                    predictions = self.model(images)
                    
                    # Process predictions and targets
                    for i in range(images.size(0)):
                        pred = predictions[i]  # [anchors, features]
                        target = targets[i]    # [num_objects, 5]
                        
                        # Decode predictions
                        pred_boxes, pred_classes, pred_scores = self._decode_predictions(pred)
                        
                        # Store predictions and targets
                        if len(pred_classes) > 0:
                            all_predictions.extend(pred_classes.cpu().numpy())
                            all_pred_boxes.append(pred_boxes.cpu().numpy())
                        else:
                            all_pred_boxes.append(np.zeros((0, 4)))
                        
                        if len(target) > 0:
                            target_classes = target[:, 0].long().cpu().numpy()
                            all_targets.extend(target_classes)
                            all_target_boxes.append(target[:, 1:5].cpu().numpy())
                        else:
                            all_target_boxes.append(np.zeros((0, 4)))
                            
                except Exception as e:
                    print(f"Warning: Error in evaluation batch {batch_idx}: {e}")
                    continue
        
        return all_predictions, all_targets, all_pred_boxes, all_target_boxes
    
    def _decode_predictions(self, predictions):
        """Decode raw predictions to boxes, classes, and scores"""
        try:
            # Model output format per location: 3 anchors * (5 + C)
            num_features_per_anchor = 5 + self.num_classes
            if predictions.size(1) == 3 * num_features_per_anchor:
                # Split anchors
                anchors = predictions.reshape(-1, 3, num_features_per_anchor).reshape(-1, num_features_per_anchor)
                box_logits = anchors[:, :4]
                obj_logits = anchors[:, 4]
                cls_logits = anchors[:, 5:5 + self.num_classes]
                
                # Decode
                box_preds = torch.sigmoid(box_logits)
                obj_scores = torch.sigmoid(obj_logits)
                cls_scores = torch.softmax(cls_logits, dim=1)
                pred_classes = torch.argmax(cls_scores, dim=1)
                
                # Filter by objectness threshold
                valid_mask = obj_scores > 0.05
                if valid_mask.any():
                    valid_boxes = box_preds[valid_mask]
                    valid_classes = pred_classes[valid_mask]
                    valid_scores = obj_scores[valid_mask]
                    
                    # Keep top predictions
                    top_k = min(500, valid_scores.numel())
                    if valid_scores.numel() > top_k:
                        topk_scores, topk_idx = torch.topk(valid_scores, k=top_k, largest=True)
                        valid_boxes = valid_boxes[topk_idx]
                        valid_classes = valid_classes[topk_idx]
                        valid_scores = topk_scores
                    
                    return valid_boxes, valid_classes, valid_scores
                else:
                    return torch.zeros((0, 4)), torch.zeros((0,), dtype=torch.long), torch.zeros((0,))
            else:
                return torch.zeros((0, 4)), torch.zeros((0,), dtype=torch.long), torch.zeros((0,))
        except Exception as e:
            print(f"Warning: Error in prediction decoding: {e}")
            return torch.zeros((0, 4)), torch.zeros((0,), dtype=torch.long), torch.zeros((0,))
    
    def calculate_metrics(self, all_predictions, all_targets):
        """Calculate comprehensive metrics"""
        try:
            # Ensure same length
            min_len = min(len(all_predictions), len(all_targets))
            if min_len == 0:
                return 0.0, 0.0, 0.0, 0.0, np.zeros((self.num_classes, self.num_classes))
            
            all_preds = np.array(all_predictions[:min_len])
            all_targets = np.array(all_targets[:min_len])
            
            # Calculate metrics
            accuracy = accuracy_score(all_targets, all_preds)
            precision = precision_score(all_targets, all_preds, average='weighted', zero_division=0)
            recall = recall_score(all_targets, all_preds, average='weighted', zero_division=0)
            f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
            
            # Calculate confusion matrix
            cm = confusion_matrix(all_targets, all_preds, labels=range(self.num_classes))
            
            return accuracy, precision, recall, f1, cm
            
        except Exception as e:
            print(f"Warning: Error in metrics calculation: {e}")
            return 0.0, 0.0, 0.0, 0.0, np.zeros((self.num_classes, self.num_classes))
    
    def create_visualizations(self, accuracy, precision, recall, f1, confusion_mat, save_dir):
        """Create and save visualizations"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style for better plots
        try:
            plt.style.use('seaborn-v0_8')
        except:
            try:
                plt.style.use('seaborn')
            except:
                plt.style.use('default')
        
        # 1. Bar chart for metrics
        fig, ax = plt.subplots(figsize=(10, 6))
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        values = [accuracy, precision, recall, f1]
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
        
        bars = ax.bar(metrics, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_title('Model Performance Metrics', fontsize=16, fontweight='bold', pad=20)
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_ylim(0, 1.1)
        ax.grid(axis='y', alpha=0.3)
        
        # Customize ticks
        ax.tick_params(axis='both', which='major', labelsize=10)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'performance_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Confusion matrix
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Normalize confusion matrix
        cm_normalized = confusion_mat.astype('float') / confusion_mat.sum(axis=1)[:, np.newaxis]
        cm_normalized = np.nan_to_num(cm_normalized)  # Handle division by zero
        
        # Create heatmap
        try:
            sns.heatmap(cm_normalized, annot=True, fmt='.3f', cmap='Blues', 
                       xticklabels=self.class_names, yticklabels=self.class_names,
                       ax=ax, cbar_kws={'label': 'Normalized Count'})
        except Exception as e:
            print(f"Warning: Error creating seaborn heatmap: {e}")
            # Fallback to matplotlib
            im = ax.imshow(cm_normalized, cmap='Blues', aspect='auto')
            ax.set_xticks(range(len(self.class_names)))
            ax.set_yticks(range(len(self.class_names)))
            ax.set_xticklabels(self.class_names)
            ax.set_yticklabels(self.class_names)
            
            # Add text annotations
            for i in range(len(self.class_names)):
                for j in range(len(self.class_names)):
                    text = ax.text(j, i, f'{cm_normalized[i, j]:.3f}',
                                 ha="center", va="center", color="black" if cm_normalized[i, j] < 0.5 else "white")
            
            plt.colorbar(im, ax=ax, label='Normalized Count')
        
        ax.set_title('Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Visualizations saved to {save_dir}")
    
    def save_results(self, accuracy, precision, recall, f1, confusion_mat, save_dir):
        """Save evaluation results to files"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metrics to JSON
        results = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'confusion_matrix': confusion_mat.tolist(),
            'class_names': self.class_names,
            'evaluation_date': datetime.now().isoformat()
        }
        
        with open(save_dir / 'evaluation_results.json', 'w') as f:
            json.dump(results, f, indent=4)
        
        # Save metrics to CSV
        metrics_df = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
            'Value': [accuracy, precision, recall, f1]
        })
        metrics_df.to_csv(save_dir / 'metrics.csv', index=False)
        
        # Save confusion matrix to CSV
        cm_df = pd.DataFrame(confusion_mat, 
                           columns=self.class_names, 
                           index=self.class_names)
        cm_df.to_csv(save_dir / 'confusion_matrix.csv')
        
        print(f"✅ Results saved to {save_dir}")
        
        return results

def find_model_file(base_path):
    """Find the best model file to load"""
    base_path = Path(base_path)
    
    # Check if the exact path exists
    if base_path.exists():
        return str(base_path)
    
    # Check for common model file names in the directory
    possible_names = [
        'best_model.pth',
        'final_model.pth', 
        'model.pth',
        'checkpoint.pth'
    ]
    
    for name in possible_names:
        model_path = base_path / name
        if model_path.exists():
            return str(model_path)
    
    # Check in subdirectories
    if base_path.exists():
        for subdir in base_path.iterdir():
            if subdir.is_dir():
                for name in possible_names:
                    model_path = subdir / name
                    if model_path.exists():
                        return str(model_path)
    
    return None

def load_model(model_path, device='cuda'):
    """Load the trained model"""
    try:
        # Check if model path exists
        if not Path(model_path).exists():
            print(f"Model path {model_path} does not exist.")
            print("Searching for model files...")
            found_path = find_model_file(model_path)
            if found_path:
                print(f"Found model at: {found_path}")
                model_path = found_path
            else:
                print("No model files found. Please check the model path.")
                return None
        
        # Create model instance
        model = ImprovedHMAYTSF(num_classes=4)
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            # Assume the checkpoint is just the state dict
            model.load_state_dict(checkpoint)
        
        print(f"✅ Model loaded successfully from {model_path}")
        if 'epoch' in checkpoint:
            print(f"✅ Model was trained for {checkpoint['epoch']} epochs")
        if 'best_val_loss' in checkpoint:
            print(f"✅ Best validation loss: {checkpoint['best_val_loss']:.4f}")
        
        return model
        
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please make sure the model file exists and is compatible.")
        return None

def print_summary(results, save_dir):
    """Print a nice summary of the evaluation results"""
    print("\n" + "="*60)
    print("🎯 EVALUATION SUMMARY")
    print("="*60)
    print(f"📊 Model Performance:")
    print(f"   • Accuracy:  {results['accuracy']:.1%}")
    print(f"   • Precision: {results['precision']:.1%}")
    print(f"   • Recall:    {results['recall']:.1%}")
    print(f"   • F1-Score:  {results['f1_score']:.1%}")
    print()
    print(f"📁 Results saved to: {save_dir}")
    print(f"   • performance_metrics.png - Bar chart of metrics")
    print(f"   • confusion_matrix.png - Confusion matrix heatmap")
    print(f"   • evaluation_results.json - Detailed results")
    print(f"   • metrics.csv - Metrics in CSV format")
    print(f"   • confusion_matrix.csv - Confusion matrix in CSV format")
    print()
    print("🎨 Visualizations created:")
    print("   • Performance metrics bar chart")
    print("   • Confusion matrix heatmap")
    print("="*60)

def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description='Evaluate HMAY-TSF Model on Test Data')
    parser.add_argument('--model-path', type=str, default='./runs/ultra_enhanced_train/best_model.pth', 
                       help='Path to trained model checkpoint')
    parser.add_argument('--data', type=str, default='./Aerial-Vehicles-1/data.yaml', 
                       help='Dataset YAML path')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size for evaluation')
    parser.add_argument('--img-size', type=int, default=640, help='Image size')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--save-dir', type=str, default='./runs/evaluation', 
                       help='Directory to save evaluation results')
    
    args = parser.parse_args()
    
    # Check if CUDA is available
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, switching to CPU")
        args.device = 'cpu'
    
    print("🚀 Starting Model Evaluation...")
    print(f"Model path: {args.model_path}")
    print(f"Test data: {args.data}")
    print(f"Device: {args.device}")
    print(f"Save directory: {args.save_dir}")
    
    # Load model
    model = load_model(args.model_path, args.device)
    if model is None:
        print("❌ Failed to load model. Exiting.")
        return
    
    # Create test dataset
    print("\n📁 Creating test dataset...")
    try:
        test_dataset = TestDataset(args.data, args.img_size)
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=2,
            collate_fn=collate_fn,
            pin_memory=True
        )
        print(f"✅ Test dataset created with {len(test_dataset)} images")
    except Exception as e:
        print(f"❌ Error creating test dataset: {e}")
        return
    
    # Create evaluator
    evaluator = ModelEvaluator(model, args.device)
    
    # Evaluate model
    print("\n🔍 Evaluating model...")
    all_predictions, all_targets, all_pred_boxes, all_target_boxes = evaluator.evaluate_model(test_loader)
    
    # Calculate metrics
    print("\n📊 Calculating metrics...")
    accuracy, precision, recall, f1, confusion_mat = evaluator.calculate_metrics(all_predictions, all_targets)
    
    # Print results
    print("\n" + "="*50)
    print("📊 EVALUATION RESULTS")
    print("="*50)
    print(f"Accuracy:  {accuracy:.4f} ({accuracy:.1%})")
    print(f"Precision: {precision:.4f} ({precision:.1%})")
    print(f"Recall:    {recall:.4f} ({recall:.1%})")
    print(f"F1-Score:  {f1:.4f} ({f1:.1%})")
    print("="*50)
    
    # Create visualizations
    print("\n🎨 Creating visualizations...")
    evaluator.create_visualizations(accuracy, precision, recall, f1, confusion_mat, args.save_dir)
    
    # Save results
    print("\n💾 Saving results...")
    results = evaluator.save_results(accuracy, precision, recall, f1, confusion_mat, args.save_dir)
    
    # Print summary
    print_summary(results, args.save_dir)
    
    print(f"\n🎯 Evaluation completed successfully!")
    print(f"📊 Performance: {accuracy:.1%} accuracy, {f1:.1%} F1-score")

if __name__ == "__main__":
    main() 