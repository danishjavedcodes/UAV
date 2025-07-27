"""
Test Inference Script to Diagnose Class Imbalance Issues
"""

import os
import torch
from ultralytics import YOLO
import numpy as np
from collections import Counter
import argparse

def test_model_predictions(model_path, test_images_dir, num_samples=50):
    """Test model predictions on a sample of images"""
    
    print("="*60)
    print("MODEL PREDICTION DIAGNOSIS")
    print("="*60)
    
    # Load model
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)
    
    # Class names
    class_names = {
        0: 'ignored regions',
        1: 'pedestrian',
        2: 'people',
        3: 'bicycle',
        4: 'car',
        5: 'van',
        6: 'truck',
        7: 'tricycle',
        8: 'awning-tricycle',
        9: 'bus',
        10: 'motor'
    }
    
    # Get test images
    if not os.path.exists(test_images_dir):
        print(f"Test directory not found: {test_images_dir}")
        return
    
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png']:
        image_files.extend([f for f in os.listdir(test_images_dir) if f.lower().endswith(ext)])
    
    if not image_files:
        print(f"No images found in {test_images_dir}")
        return
    
    # Sample images
    if len(image_files) > num_samples:
        image_files = np.random.choice(image_files, num_samples, replace=False)
    
    print(f"Testing on {len(image_files)} images...")
    
    # Collect predictions
    all_predictions = []
    class_predictions = Counter()
    
    for i, image_file in enumerate(image_files):
        image_path = os.path.join(test_images_dir, image_file)
        
        # Predict
        results = model.predict(image_path, conf=0.25, iou=0.45, verbose=False)
        
        if results[0].boxes is not None:
            boxes = results[0].boxes
            for box in boxes:
                class_id = int(box.cls[0].cpu().numpy())
                confidence = float(box.conf[0].cpu().numpy())
                
                all_predictions.append({
                    'image': image_file,
                    'class_id': class_id,
                    'class_name': class_names.get(class_id, f'class_{class_id}'),
                    'confidence': confidence
                })
                
                class_predictions[class_id] += 1
    
    # Analyze results
    print(f"\nTotal detections: {len(all_predictions)}")
    
    if not all_predictions:
        print("No detections found!")
        return
    
    print("\nClass Distribution in Predictions:")
    total_detections = sum(class_predictions.values())
    
    for class_id in sorted(class_predictions.keys()):
        count = class_predictions[class_id]
        percentage = (count / total_detections) * 100
        class_name = class_names.get(class_id, f'class_{class_id}')
        print(f"  {class_name} (Class {class_id}): {count} ({percentage:.1f}%)")
    
    # Check for bias
    max_class = max(class_predictions, key=class_predictions.get)
    max_count = class_predictions[max_class]
    max_percentage = (max_count / total_detections) * 100
    
    print(f"\nBias Analysis:")
    print(f"  Most predicted class: {class_names.get(max_class, f'class_{max_class}')}")
    print(f"  Percentage: {max_percentage:.1f}%")
    
    if max_percentage > 80:
        print(f"  ⚠️  HIGH BIAS DETECTED: Model is heavily biased towards {class_names.get(max_class, f'class_{max_class}')}")
    elif max_percentage > 60:
        print(f"  ⚠️  MODERATE BIAS DETECTED: Model shows bias towards {class_names.get(max_class, f'class_{max_class}')}")
    else:
        print(f"  ✅  Good class distribution")
    
    # Show sample predictions
    print(f"\nSample Predictions (first 10):")
    for i, pred in enumerate(all_predictions[:10]):
        print(f"  {i+1}. {pred['image']}: {pred['class_name']} (conf: {pred['confidence']:.3f})")
    
    return class_predictions

def main():
    parser = argparse.ArgumentParser(description='Test model predictions for class bias')
    parser.add_argument('--model', type=str, default='./runs/train/best.pt',
                       help='Path to trained model')
    parser.add_argument('--test_dir', type=str, default='./dataset/images/test',
                       help='Directory with test images')
    parser.add_argument('--samples', type=int, default=50,
                       help='Number of images to test')
    
    args = parser.parse_args()
    
    # Test predictions
    predictions = test_model_predictions(args.model, args.test_dir, args.samples)
    
    if predictions:
        print(f"\nRecommendations:")
        print(f"1. If high bias detected, retrain with balanced dataset")
        print(f"2. Use the create_balanced_dataset() method in ClassBalancingSystem")
        print(f"3. Adjust class weights in the loss function")
        print(f"4. Consider data augmentation for rare classes")

if __name__ == "__main__":
    main() 