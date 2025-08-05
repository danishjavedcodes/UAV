"""
Test script to verify model output format and debug metrics calculation
"""

import torch
import torch.nn as nn
import numpy as np
from train import SimpleHMAYTSF, MetricsCalculator

def test_model_output():
    """Test the model output format"""
    print("Testing model output format...")
    
    # Create model
    model = SimpleHMAYTSF(num_classes=4)
    model.eval()
    
    # Create dummy input
    batch_size = 2
    img_size = 640
    x = torch.randn(batch_size, 3, img_size, img_size)
    
    # Forward pass
    with torch.no_grad():
        output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected features per anchor: {3 * (5 + 4)} = 27")
    print(f"Actual features per anchor: {output.shape[2]}")
    
    # Test metrics calculator
    metrics_calc = MetricsCalculator(num_classes=4)
    
    # Create dummy targets
    targets = [
        torch.tensor([[0, 0.5, 0.5, 0.1, 0.1], [1, 0.3, 0.3, 0.2, 0.2]]),  # 2 objects
        torch.tensor([[2, 0.7, 0.7, 0.15, 0.15]])  # 1 object
    ]
    
    # Calculate metrics
    accuracy, precision, recall, f1 = metrics_calc.calculate_metrics(output, targets)
    
    print(f"\nMetrics calculation test:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    # Test individual prediction processing
    print(f"\nTesting individual prediction processing...")
    pred = output[0]  # First batch
    target = targets[0]
    
    print(f"Prediction shape: {pred.shape}")
    print(f"Target shape: {target.shape}")
    
    # Process prediction
    num_features_per_anchor = 5 + 4
    if pred.size(1) >= 3 * num_features_per_anchor:
        pred_reshaped = pred.reshape(-1, 3, num_features_per_anchor)
        all_anchors = pred_reshaped.reshape(-1, num_features_per_anchor)
        
        print(f"Reshaped prediction: {pred_reshaped.shape}")
        print(f"All anchors shape: {all_anchors.shape}")
        
        # Extract components
        box_preds = all_anchors[:, :4]
        obj_preds = all_anchors[:, 4]
        cls_preds = all_anchors[:, 5:9]
        
        print(f"Box predictions shape: {box_preds.shape}")
        print(f"Objectness predictions shape: {obj_preds.shape}")
        print(f"Class predictions shape: {cls_preds.shape}")
        
        # Get predicted classes
        pred_classes = torch.argmax(cls_preds, dim=1)
        print(f"Predicted classes shape: {pred_classes.shape}")
        print(f"Sample predicted classes: {pred_classes[:10]}")
        
        # Filter by objectness
        valid_mask = obj_preds > 0.3
        print(f"Valid predictions: {valid_mask.sum()}/{len(valid_mask)}")
        
        if valid_mask.sum() > 0:
            valid_preds = pred_classes[valid_mask]
            print(f"Valid predicted classes: {valid_preds[:10]}")
    
    print("\n✅ Model output format test completed!")

if __name__ == "__main__":
    test_model_output() 