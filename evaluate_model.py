"""
Comprehensive Evaluation Script for HMAY-TSF
Includes standard metrics and occlusion-aware detection metrics
"""

import torch
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import time
import argparse

class ModelEvaluator:
    """Comprehensive model evaluator"""
    
    def __init__(self, model_path, data_yaml, device='auto'):
        self.device = device if device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = YOLO(model_path)
        self.data_yaml = data_yaml
        self.results = {}
        
    def evaluate_standard_metrics(self):
        """Evaluate standard object detection metrics"""
        print("Evaluating standard metrics...")
        
        # Run validation
        val_results = self.model.val(data=self.data_yaml, device=self.device, plots=True)
        
        metrics = {
            'mAP50': float(val_results.box.map50),
            'mAP50-95': float(val_results.box.map),
            'precision': float(val_results.box.mp),
            'recall': float(val_results.box.mr),
            'mAP_per_class': val_results.box.maps.tolist() if hasattr(val_results.box, 'maps') else []
        }
        
        self.results['standard_metrics'] = metrics
        return metrics
    
    def evaluate_fps(self, test_images_dir, num_samples=100):
        """Evaluate inference speed (FPS)"""
        print("Evaluating inference speed...")
        
        test_images = list(Path(test_images_dir).glob('*.jpg'))[:num_samples]
        
        if not test_images:
            print("No test images found!")
            return None
        
        # Warm up
        dummy_img = cv2.imread(str(test_images[0]))
        for _ in range(10):
            _ = self.model.predict(dummy_img, verbose=False)
        
        # Measure inference time
        start_time = time.time()
        
        for img_path in test_images:
            img = cv2.imread(str(img_path))
            _ = self.model.predict(img, verbose=False)
        
        total_time = time.time() - start_time
        fps = len(test_images) / total_time
        
        fps_metrics = {
            'fps': fps,
            'avg_inference_time': total_time / len(test_images),
            'total_time': total_time,
            'num_images': len(test_images)
        }
        
        self.results['fps_metrics'] = fps_metrics
        return fps_metrics
    
    def evaluate_small_objects(self, test_images_dir, labels_dir, size_threshold=0.05):
        """Evaluate performance on small objects"""
        print("Evaluating small object detection...")
        
        test_images = list(Path(test_images_dir).glob('*.jpg'))
        small_obj_results = []
        
        for img_path in test_images[:50]:  # Limit for faster evaluation
            # Load ground truth
            label_path = Path(labels_dir) / (img_path.stem + '.txt')
            if not label_path.exists():
                continue
                
            # Load image and get dimensions
            img = cv2.imread(str(img_path))
            h, w = img.shape[:2]
            
            # Load labels
            gt_boxes = []
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    if line.strip():
                        parts = line.strip().split()
                        if len(parts) == 5:
                            class_id, x_center, y_center, width, height = map(float, parts)
                            area = width * height
                            gt_boxes.append({
                                'class': int(class_id),
                                'area': area,
                                'is_small': area < size_threshold
                            })
            
            # Get predictions
            results = self.model.predict(img, verbose=False)
            pred_boxes = []
            
            if results and len(results[0].boxes) > 0:
                boxes = results[0].boxes
                for i in range(len(boxes)):
                    conf = float(boxes.conf[i])
                    cls = int(boxes.cls[i])
                    xyxy = boxes.xyxy[i].cpu().numpy()
                    
                    # Convert to normalized coordinates
                    x1, y1, x2, y2 = xyxy
                    width_norm = (x2 - x1) / w
                    height_norm = (y2 - y1) / h
                    area = width_norm * height_norm
                    
                    pred_boxes.append({
                        'class': cls,
                        'confidence': conf,
                        'area': area,
                        'is_small': area < size_threshold
                    })
            
            small_obj_results.append({
                'image': img_path.name,
                'gt_small_objects': len([box for box in gt_boxes if box['is_small']]),
                'pred_small_objects': len([box for box in pred_boxes if box['is_small']]),
                'gt_total_objects': len(gt_boxes),
                'pred_total_objects': len(pred_boxes)
            })
        
        # Calculate small object metrics
        total_gt_small = sum([r['gt_small_objects'] for r in small_obj_results])
        total_pred_small = sum([r['pred_small_objects'] for r in small_obj_results])
        total_gt = sum([r['gt_total_objects'] for r in small_obj_results])
        total_pred = sum([r['pred_total_objects'] for r in small_obj_results])
        
        small_obj_metrics = {
            'small_object_recall': total_pred_small / max(total_gt_small, 1),
            'small_object_ratio_gt': total_gt_small / max(total_gt, 1),
            'small_object_ratio_pred': total_pred_small / max(total_pred, 1),
            'total_small_objects_gt': total_gt_small,
            'total_small_objects_pred': total_pred_small
        }
        
        self.results['small_object_metrics'] = small_obj_metrics
        return small_obj_metrics
    
    def occlusion_aware_evaluation(self, test_images_dir, labels_dir):
        """
        Occlusion-Aware Detection Metric (OADM)
        Simplified version based on object size as proxy for occlusion
        """
        print("Evaluating occlusion-aware metrics...")
        
        test_images = list(Path(test_images_dir).glob('*.jpg'))
        occlusion_results = {
            'no_occlusion': [],      # Large objects (>0.1 area)
            'light_occlusion': [],   # Medium objects (0.05-0.1 area)
            'heavy_occlusion': []    # Small objects (<0.05 area)
        }
        
        for img_path in test_images[:100]:  # Limit for faster evaluation
            label_path = Path(labels_dir) / (img_path.stem + '.txt')
            if not label_path.exists():
                continue
            
            # Load image
            img = cv2.imread(str(img_path))
            
            # Load ground truth
            gt_objects = {'no_occlusion': 0, 'light_occlusion': 0, 'heavy_occlusion': 0}
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    if line.strip():
                        parts = line.strip().split()
                        if len(parts) == 5:
                            _, _, _, width, height = map(float, parts)
                            area = width * height
                            
                            if area > 0.1:
                                gt_objects['no_occlusion'] += 1
                            elif area > 0.05:
                                gt_objects['light_occlusion'] += 1
                            else:
                                gt_objects['heavy_occlusion'] += 1
            
            # Get predictions
            results = self.model.predict(img, verbose=False)
            pred_objects = {'no_occlusion': 0, 'light_occlusion': 0, 'heavy_occlusion': 0}
            
            if results and len(results[0].boxes) > 0:
                boxes = results[0].boxes
                h, w = img.shape[:2]
                
                for i in range(len(boxes)):
                    conf = float(boxes.conf[i])
                    if conf > 0.25:  # Confidence threshold
                        xyxy = boxes.xyxy[i].cpu().numpy()
                        x1, y1, x2, y2 = xyxy
                        
                        # Calculate normalized area
                        width_norm = (x2 - x1) / w
                        height_norm = (y2 - y1) / h
                        area = width_norm * height_norm
                        
                        if area > 0.1:
                            pred_objects['no_occlusion'] += 1
                        elif area > 0.05:
                            pred_objects['light_occlusion'] += 1
                        else:
                            pred_objects['heavy_occlusion'] += 1
            
            # Store results for each occlusion level
            for level in occlusion_results.keys():
                occlusion_results[level].append({
                    'gt': gt_objects[level],
                    'pred': pred_objects[level]
                })
        
        # Calculate OADM metrics
        oadm_metrics = {}
        for level, results in occlusion_results.items():
            total_gt = sum([r['gt'] for r in results])
            total_pred = sum([r['pred'] for r in results])
            
            precision = total_pred / max(total_pred, 1) if total_pred > 0 else 0
            recall = total_pred / max(total_gt, 1) if total_gt > 0 else 0
            f1 = 2 * precision * recall / max(precision + recall, 1e-6)
            
            oadm_metrics[level] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'total_gt': total_gt,
                'total_pred': total_pred
            }
        
        # Calculate weighted OADM score
        weights = {'no_occlusion': 1.0, 'light_occlusion': 1.5, 'heavy_occlusion': 2.0}
        weighted_f1 = sum([oadm_metrics[level]['f1'] * weights[level] for level in weights.keys()])
        weighted_f1 /= sum(weights.values())
        
        oadm_metrics['weighted_oadm'] = weighted_f1
        self.results['oadm_metrics'] = oadm_metrics
        
        return oadm_metrics
    
    def save_results(self, output_path='evaluation_results.json'):
        """Save evaluation results"""
        output_path = Path(output_path)
        
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"Results saved to {output_path}")
        
        # Print summary
        print("\n" + "="*50)
        print("EVALUATION SUMMARY")
        print("="*50)
        
        if 'standard_metrics' in self.results:
            std_metrics = self.results['standard_metrics']
            print(f"mAP50: {std_metrics['mAP50']:.4f}")
            print(f"mAP50-95: {std_metrics['mAP50-95']:.4f}")
            print(f"Precision: {std_metrics['precision']:.4f}")
            print(f"Recall: {std_metrics['recall']:.4f}")
        
        if 'fps_metrics' in self.results:
            fps_metrics = self.results['fps_metrics']
            print(f"FPS: {fps_metrics['fps']:.2f}")
            print(f"Avg Inference Time: {fps_metrics['avg_inference_time']*1000:.2f}ms")
        
        if 'small_object_metrics' in self.results:
            small_metrics = self.results['small_object_metrics']
            print(f"Small Object Recall: {small_metrics['small_object_recall']:.4f}")
        
        if 'oadm_metrics' in self.results:
            oadm_metrics = self.results['oadm_metrics']
            print(f"Weighted OADM Score: {oadm_metrics['weighted_oadm']:.4f}")
        
        print("="*50)

def main():
    parser = argparse.ArgumentParser(description='Evaluate HMAY-TSF Model')
    parser.add_argument('--model', type=str, required=True, help='Path to model weights')
    parser.add_argument('--data', type=str, default='./dataset/dataset.yaml', help='Dataset YAML path')
    parser.add_argument('--test-images', type=str, default='./dataset/images/test', help='Test images directory')
    parser.add_argument('--test-labels', type=str, default='./dataset/labels/test', help='Test labels directory')
    parser.add_argument('--output', type=str, default='evaluation_results.json', help='Output results file')
    parser.add_argument('--device', type=str, default='auto', help='Device to use')
    
    args = parser.parse_args()
    
    print("HMAY-TSF Model Evaluation")
    print("="*30)
    
    # Initialize evaluator
    evaluator = ModelEvaluator(args.model, args.data, args.device)
    
    # Run evaluations
    print("Running comprehensive evaluation...")
    
    # Standard metrics
    evaluator.evaluate_standard_metrics()
    
    # FPS evaluation
    if Path(args.test_images).exists():
        evaluator.evaluate_fps(args.test_images)
        evaluator.evaluate_small_objects(args.test_images, args.test_labels)
        evaluator.occlusion_aware_evaluation(args.test_images, args.test_labels)
    
    # Save results
    evaluator.save_results(args.output)

if __name__ == "__main__":
    main() 