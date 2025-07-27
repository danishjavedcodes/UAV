"""
Inference Script for HMAY-TSF Model
Performs inference on images with bounding box visualization
"""

import os
import cv2
import torch
import numpy as np
import argparse
from pathlib import Path
import yaml
from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import time
from collections import defaultdict

class HMAYTSFInference:
    """Advanced inference class for HMAY-TSF model with visualization"""
    
    def __init__(self, model_path='./runs/train/best.pt', device='auto'):
        self.device = device if device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        self.model = None
        self.class_names = None
        self.class_colors = None
        
        # Load model and setup
        self.load_model()
        self.setup_class_info()
        
    def load_model(self):
        """Load the trained HMAY-TSF model"""
        print(f"Loading model from: {self.model_path}")
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        # Load YOLO model
        self.model = YOLO(self.model_path)
        
        # Move to device
        if self.device == 'cuda':
            self.model.to('cuda')
        
        print(f"Model loaded successfully on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
    def setup_class_info(self):
        """Setup class names and colors for visualization"""
        # Class names for VisDrone dataset
        self.class_names = {
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
        
        # Generate distinct colors for each class
        np.random.seed(42)  # For consistent colors
        self.class_colors = {}
        for class_id in self.class_names.keys():
            # Generate a distinct color
            color = tuple(np.random.randint(0, 255, 3).tolist())
            self.class_colors[class_id] = color
            
        print("Class information setup completed")
        
    def predict_image(self, image_path, conf_threshold=0.25, iou_threshold=0.45):
        """Perform inference on a single image"""
        print(f"Processing image: {image_path}")
        
        # Load image
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Perform prediction
        results = self.model.predict(
            source=image_path,
            conf=conf_threshold,
            iou=iou_threshold,
            save=False,
            save_txt=False
        )
        
        return results[0]  # Return first result
    
    def visualize_predictions(self, image_path, results, save_path=None, show_plot=True):
        """Visualize predictions with bounding boxes"""
        # Load image
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create matplotlib figure
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(image_rgb)
        
        # Statistics
        detection_stats = defaultdict(int)
        
        # Draw bounding boxes
        if results.boxes is not None:
            boxes = results.boxes
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = box.conf[0].cpu().numpy()
                class_id = int(box.cls[0].cpu().numpy())
                
                # Get class info
                class_name = self.class_names.get(class_id, f'class_{class_id}')
                color = self.class_colors.get(class_id, (255, 0, 0))
                
                # Normalize color for matplotlib (0-1 range)
                color_normalized = tuple(c/255 for c in color)
                
                # Create rectangle
                width = x2 - x1
                height = y2 - y1
                rect = Rectangle((x1, y1), width, height, 
                               linewidth=2, edgecolor=color_normalized, 
                               facecolor='none')
                ax.add_patch(rect)
                
                # Add label
                label = f'{class_name}: {confidence:.2f}'
                ax.text(x1, y1-5, label, fontsize=10, color='white',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor=color_normalized, alpha=0.8))
                
                # Update statistics
                detection_stats[class_name] += 1
        
        # Add title with statistics
        total_detections = sum(detection_stats.values())
        title = f'Detections: {total_detections} | '
        title += ' | '.join([f'{name}: {count}' for name, count in detection_stats.items()])
        ax.set_title(title, fontsize=12, fontweight='bold')
        
        # Remove axes
        ax.axis('off')
        
        # Save plot if requested
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            print(f"Visualization saved to: {save_path}")
        
        # Show plot if requested
        if show_plot:
            plt.tight_layout()
            plt.show()
        
        return detection_stats
    
    def process_directory(self, input_dir, output_dir=None, conf_threshold=0.25, 
                         iou_threshold=0.45, save_visualizations=True):
        """Process all images in a directory"""
        input_path = Path(input_dir)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        
        # Create output directory if specified
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            print(f"Output directory: {output_path}")
        
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        for ext in image_extensions:
            image_files.extend(input_path.glob(f'*{ext}'))
            image_files.extend(input_path.glob(f'*{ext.upper()}'))
        
        if not image_files:
            print(f"No image files found in {input_dir}")
            return
        
        print(f"Found {len(image_files)} images to process")
        
        # Process each image
        total_stats = defaultdict(int)
        processing_times = []
        
        for i, image_file in enumerate(image_files):
            print(f"\nProcessing {i+1}/{len(image_files)}: {image_file.name}")
            
            start_time = time.time()
            
            # Perform prediction
            results = self.predict_image(str(image_file), conf_threshold, iou_threshold)
            
            # Visualize results
            if save_visualizations and output_dir:
                save_path = output_path / f"{image_file.stem}_detections.jpg"
                stats = self.visualize_predictions(str(image_file), results, str(save_path), show_plot=False)
            else:
                stats = self.visualize_predictions(str(image_file), results, show_plot=True)
            
            # Update statistics
            for class_name, count in stats.items():
                total_stats[class_name] += count
            
            processing_time = time.time() - start_time
            processing_times.append(processing_time)
            
            print(f"  Detections: {sum(stats.values())}")
            print(f"  Processing time: {processing_time:.2f}s")
        
        # Print summary
        print(f"\n{'='*50}")
        print("PROCESSING SUMMARY")
        print(f"{'='*50}")
        print(f"Total images processed: {len(image_files)}")
        print(f"Average processing time: {np.mean(processing_times):.2f}s")
        print(f"Total detections: {sum(total_stats.values())}")
        print("\nDetection breakdown:")
        for class_name, count in sorted(total_stats.items()):
            print(f"  {class_name}: {count}")
        
        if output_dir:
            print(f"\nResults saved to: {output_path}")
    
    def real_time_inference(self, source=0, conf_threshold=0.25, iou_threshold=0.45):
        """Real-time inference from webcam or video"""
        print(f"Starting real-time inference from source: {source}")
        
        # Open video capture
        cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            print(f"Error: Could not open video source {source}")
            return
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Video properties: {width}x{height} @ {fps:.2f} FPS")
        
        frame_count = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Perform prediction
            results = self.model.predict(
                source=frame,
                conf=conf_threshold,
                iou=iou_threshold,
                save=False,
                save_txt=False
            )
            
            # Draw bounding boxes on frame
            annotated_frame = self.draw_boxes_on_frame(frame, results[0])
            
            # Add FPS counter
            elapsed_time = time.time() - start_time
            current_fps = frame_count / elapsed_time
            cv2.putText(annotated_frame, f'FPS: {current_fps:.1f}', 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display frame
            cv2.imshow('HMAY-TSF Real-time Inference', annotated_frame)
            
            # Break on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"Real-time inference completed. Processed {frame_count} frames.")
    
    def draw_boxes_on_frame(self, frame, results):
        """Draw bounding boxes on OpenCV frame"""
        if results.boxes is not None:
            boxes = results.boxes
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = box.conf[0].cpu().numpy()
                class_id = int(box.cls[0].cpu().numpy())
                
                # Get class info
                class_name = self.class_names.get(class_id, f'class_{class_id}')
                color = self.class_colors.get(class_id, (0, 0, 255))
                
                # Convert BGR to RGB for OpenCV
                color_bgr = (color[2], color[1], color[0])
                
                # Draw rectangle
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color_bgr, 2)
                
                # Draw label
                label = f'{class_name}: {confidence:.2f}'
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(frame, (int(x1), int(y1) - label_size[1] - 10), 
                             (int(x1) + label_size[0], int(y1)), color_bgr, -1)
                cv2.putText(frame, label, (int(x1), int(y1) - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame

def main():
    parser = argparse.ArgumentParser(description='HMAY-TSF Model Inference')
    parser.add_argument('--model', type=str, default='./runs/train/best.pt',
                       help='Path to trained model')
    parser.add_argument('--source', type=str, default=None,
                       help='Path to image, directory, or video file (0 for webcam)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory for saved visualizations')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45,
                       help='IoU threshold for NMS')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save visualizations')
    parser.add_argument('--no-show', action='store_true',
                       help='Do not show plots')
    
    args = parser.parse_args()
    
    print("="*60)
    print("HMAY-TSF MODEL INFERENCE")
    print("="*60)
    
    # Initialize inference engine
    try:
        inference = HMAYTSFInference(model_path=args.model, device=args.device)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Process source
    if args.source is None:
        print("No source specified. Please provide --source argument.")
        print("Examples:")
        print("  --source image.jpg          # Single image")
        print("  --source ./dataset/images   # Directory of images")
        print("  --source video.mp4          # Video file")
        print("  --source 0                  # Webcam")
        return
    
    # Check if source is webcam
    if args.source == '0' or args.source == 0:
        inference.real_time_inference(source=0, conf_threshold=args.conf, iou_threshold=args.iou)
        return
    
    # Check if source is a file or directory
    source_path = Path(args.source)
    
    if source_path.is_file():
        # Single image or video file
        if source_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
            # Video file
            inference.real_time_inference(source=str(source_path), conf_threshold=args.conf, iou_threshold=args.iou)
        else:
            # Single image
            results = inference.predict_image(str(source_path), args.conf, args.iou)
            inference.visualize_predictions(str(source_path), results, 
                                          save_path=args.output, 
                                          show_plot=not args.no_show)
    
    elif source_path.is_dir():
        # Directory of images
        inference.process_directory(
            input_dir=str(source_path),
            output_dir=args.output,
            conf_threshold=args.conf,
            iou_threshold=args.iou,
            save_visualizations=not args.no_save
        )
    
    else:
        print(f"Error: Source '{args.source}' not found")
        return
    
    print("\nInference completed successfully!")

if __name__ == "__main__":
    main() 