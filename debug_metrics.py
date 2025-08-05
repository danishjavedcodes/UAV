"""
Debug script to test metrics calculation with real data
"""

import torch
import numpy as np
from train import SimpleHMAYTSF, MetricsCalculator, SimpleDataset
from torch.utils.data import DataLoader

def debug_metrics():
    """Debug metrics calculation with real data"""
    print("Debugging metrics calculation...")
    
    # Create model
    model = SimpleHMAYTSF(num_classes=4)
    model.eval()
    
    # Create dataset
    try:
        dataset = SimpleDataset('./Aerial-Vehicles-1/data.yaml', img_size=640, is_training=True)
        print(f"Dataset created with {len(dataset)} samples")
        
        # Get a few samples
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)
        
        for batch_idx, (images, targets) in enumerate(dataloader):
            if batch_idx >= 2:  # Only test first 2 batches
                break
                
            print(f"\nBatch {batch_idx + 1}:")
            print(f"Images shape: {images.shape}")
            print(f"Number of targets: {len(targets)}")
            
            for i, target in enumerate(targets):
                print(f"  Target {i}: {target.shape} - {target}")
            
            # Forward pass
            with torch.no_grad():
                predictions = model(images)
            
            print(f"Predictions shape: {predictions.shape}")
            
            # Test metrics calculation
            metrics_calc = MetricsCalculator(num_classes=4)
            accuracy, precision, recall, f1 = metrics_calc.calculate_metrics(predictions, targets)
            
            print(f"Metrics: Acc={accuracy:.4f}, Prec={precision:.4f}, Rec={recall:.4f}, F1={f1:.4f}")
            
            # Test individual sample
            pred = predictions[0]
            target = targets[0]
            
            print(f"\nSample prediction shape: {pred.shape}")
            print(f"Sample target shape: {target.shape}")
            print(f"Sample target: {target}")
            
            # Process prediction
            if pred.size(1) == 27:
                num_features_per_anchor = 5 + 4  # 9
                pred_reshaped = pred.reshape(-1, 3, num_features_per_anchor)
                all_anchors = pred_reshaped.reshape(-1, num_features_per_anchor)
                
                box_preds = all_anchors[:, :4]
                obj_preds = all_anchors[:, 4]
                cls_preds = all_anchors[:, 5:9]
                
                pred_classes = torch.argmax(cls_preds, dim=1)
                valid_mask = obj_preds > 0.3
                
                print(f"Valid predictions: {valid_mask.sum()}/{len(valid_mask)}")
                if valid_mask.sum() > 0:
                    valid_classes = pred_classes[valid_mask]
                    print(f"Valid class predictions: {valid_classes[:10]}")
                    print(f"Class distribution: {torch.bincount(valid_classes, minlength=4)}")
            
            break  # Only test first batch
            
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Testing with dummy data...")
        
        # Test with dummy data
        model = SimpleHMAYTSF(num_classes=4)
        model.eval()
        
        # Create dummy input and targets
        images = torch.randn(2, 3, 640, 640)
        targets = [
            torch.tensor([[0, 0.5, 0.5, 0.1, 0.1], [1, 0.3, 0.3, 0.2, 0.2]]),
            torch.tensor([[2, 0.7, 0.7, 0.15, 0.15]])
        ]
        
        with torch.no_grad():
            predictions = model(images)
        
        metrics_calc = MetricsCalculator(num_classes=4)
        accuracy, precision, recall, f1 = metrics_calc.calculate_metrics(predictions, targets)
        
        print(f"Dummy metrics: Acc={accuracy:.4f}, Prec={precision:.4f}, Rec={recall:.4f}, F1={f1:.4f}")
    
    print("\n✅ Metrics debugging completed!")

if __name__ == "__main__":
    debug_metrics() 