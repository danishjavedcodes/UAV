"""
Test script to understand model prediction behavior
"""

import torch
import torch.nn as nn
import numpy as np
from train import SimpleHMAYTSF

def test_model_predictions():
    """Test model prediction behavior"""
    print("Testing model prediction behavior...")
    
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
    
    print(f"Model output shape: {output.shape}")
    
    # Test individual prediction processing
    pred = output[0]  # First batch
    print(f"Single prediction shape: {pred.shape}")
    
    # Process prediction
    if pred.size(1) == 27:
        num_features_per_anchor = 5 + 4  # 9
        pred_reshaped = pred.reshape(-1, 3, num_features_per_anchor)
        all_anchors = pred_reshaped.reshape(-1, num_features_per_anchor)
        
        print(f"All anchors shape: {all_anchors.shape}")
        
        # Extract components
        box_preds = all_anchors[:, :4]
        obj_preds = all_anchors[:, 4]
        cls_preds = all_anchors[:, 5:9]
        
        print(f"Box predictions range: [{box_preds.min():.4f}, {box_preds.max():.4f}]")
        print(f"Objectness predictions range: [{obj_preds.min():.4f}, {obj_preds.max():.4f}]")
        print(f"Class predictions range: [{cls_preds.min():.4f}, {cls_preds.max():.4f}]")
        
        # Get predicted classes
        pred_classes = torch.argmax(cls_preds, dim=1)
        print(f"Predicted classes shape: {pred_classes.shape}")
        print(f"Class distribution: {torch.bincount(pred_classes, minlength=4)}")
        
        # Test different objectness thresholds
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
        for threshold in thresholds:
            valid_mask = obj_preds > threshold
            valid_count = valid_mask.sum()
            print(f"Threshold {threshold}: {valid_count}/{len(valid_mask)} valid predictions")
            
            if valid_count > 0:
                valid_classes = pred_classes[valid_mask]
                print(f"  Class distribution: {torch.bincount(valid_classes, minlength=4)}")
        
        # Test class prediction confidence
        class_probs = torch.softmax(cls_preds, dim=1)
        max_probs = torch.max(class_probs, dim=1)[0]
        print(f"Class confidence range: [{max_probs.min():.4f}, {max_probs.max():.4f}]")
        print(f"Mean class confidence: {max_probs.mean():.4f}")
        
        # Test objectness distribution
        print(f"Objectness mean: {obj_preds.mean():.4f}")
        print(f"Objectness std: {obj_preds.std():.4f}")
        
        # Show some sample predictions
        print("\nSample predictions:")
        for i in range(min(10, len(all_anchors))):
            print(f"  Anchor {i}: obj={obj_preds[i]:.4f}, class={pred_classes[i]}, conf={max_probs[i]:.4f}")
    
    print("\n✅ Model prediction test completed!")

if __name__ == "__main__":
    test_model_predictions() 