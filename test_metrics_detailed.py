"""
Detailed test to understand metrics calculation
"""

import torch
import numpy as np
from train import SimpleHMAYTSF, MetricsCalculator

def test_metrics_detailed():
    """Detailed test of metrics calculation"""
    print("Testing metrics calculation in detail...")
    
    # Create model
    model = SimpleHMAYTSF(num_classes=4)
    model.eval()
    
    # Create dummy input and targets
    images = torch.randn(2, 3, 640, 640)
    targets = [
        torch.tensor([[0, 0.5, 0.5, 0.1, 0.1], [1, 0.3, 0.3, 0.2, 0.2]]),
        torch.tensor([[2, 0.7, 0.7, 0.15, 0.15]])
    ]
    
    print(f"Targets:")
    for i, target in enumerate(targets):
        print(f"  Target {i}: {target}")
    
    # Forward pass
    with torch.no_grad():
        predictions = model(images)
    
    print(f"Predictions shape: {predictions.shape}")
    
    # Test metrics calculation step by step
    metrics_calc = MetricsCalculator(num_classes=4)
    
    all_preds = []
    all_targets = []
    
    for i in range(2):
        pred = predictions[i]
        target = targets[i]
        
        print(f"\nProcessing batch {i}:")
        print(f"  Prediction shape: {pred.shape}")
        print(f"  Target shape: {target.shape}")
        print(f"  Target: {target}")
        
        if pred.size(1) == 27:
            num_features_per_anchor = 5 + 4
            pred_reshaped = pred.reshape(-1, 3, num_features_per_anchor)
            all_anchors = pred_reshaped.reshape(-1, num_features_per_anchor)
            
            box_preds = all_anchors[:, :4]
            obj_preds = all_anchors[:, 4]
            cls_preds = all_anchors[:, 5:9]
            
            pred_classes = torch.argmax(cls_preds, dim=1)
            valid_mask = obj_preds > 0.1
            
            print(f"  Valid predictions: {valid_mask.sum()}/{len(valid_mask)}")
            
            if valid_mask.sum() > 0:
                valid_classes = pred_classes[valid_mask]
                print(f"  Valid class predictions: {valid_classes[:10]}")
                print(f"  Class distribution: {torch.bincount(valid_classes, minlength=4)}")
                
                all_preds.extend(valid_classes.cpu().numpy())
            
            if len(target) > 0:
                target_classes = target[:, 0].long()
                print(f"  Target classes: {target_classes}")
                all_targets.extend(target_classes.cpu().numpy())
    
    print(f"\nFinal results:")
    print(f"  All predictions: {len(all_preds)}")
    print(f"  All targets: {len(all_targets)}")
    
    if len(all_preds) > 0:
        print(f"  Prediction distribution: {np.bincount(all_preds, minlength=4)}")
    if len(all_targets) > 0:
        print(f"  Target distribution: {np.bincount(all_targets, minlength=4)}")
    
    # Calculate metrics manually
    if len(all_preds) > 0 and len(all_targets) > 0:
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        
        # Make sure we have the same number of predictions and targets
        min_len = min(len(all_preds), len(all_targets))
        if min_len > 0:
            all_preds = all_preds[:min_len]
            all_targets = all_targets[:min_len]
            
            print(f"  Using {min_len} matched predictions/targets")
            print(f"  Predictions: {all_preds}")
            print(f"  Targets: {all_targets}")
            
            accuracy = accuracy_score(all_targets, all_preds)
            precision = precision_score(all_targets, all_preds, average='weighted', zero_division=0)
            recall = recall_score(all_targets, all_preds, average='weighted', zero_division=0)
            f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
            
            print(f"  Manual metrics: Acc={accuracy:.4f}, Prec={precision:.4f}, Rec={recall:.4f}, F1={f1:.4f}")
    
    print("\n✅ Detailed metrics test completed!")

if __name__ == "__main__":
    test_metrics_detailed() 