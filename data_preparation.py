"""
Data Preparation and Augmentation for HMAY-TSF
Includes super-resolution augmentation, copy-paste, and active learning
"""

import os
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from torch.utils.data import Dataset, DataLoader
import yaml
from pathlib import Path
import shutil
from sklearn.model_selection import train_test_split
import random
from PIL import Image
import json
from collections import Counter, defaultdict
import glob
import math

class VisDroneDataset(Dataset):
    """Custom dataset for VisDrone with augmentations"""
    
    def __init__(self, img_dir, label_dir, img_size=640, augment=True, super_res=False):
        self.img_dir = Path(img_dir)
        self.label_dir = Path(label_dir)
        self.img_size = img_size
        self.augment = augment
        self.super_res = super_res
        
        # Get all image files
        self.img_files = list(self.img_dir.glob('*.jpg'))
        self.label_files = [self.label_dir / (img.stem + '.txt') for img in self.img_files]
        
        # Filter out images without labels
        valid_pairs = []
        for img_file, label_file in zip(self.img_files, self.label_files):
            if label_file.exists():
                valid_pairs.append((img_file, label_file))
        
        self.img_files, self.label_files = zip(*valid_pairs) if valid_pairs else ([], [])
        
        print(f"Found {len(self.img_files)} valid image-label pairs in {img_dir}")
        
        # Setup augmentations
        self.setup_augmentations()
    
    def setup_augmentations(self):
        """Setup augmentation pipeline"""
        if self.augment:
            self.transform = A.Compose([
                A.LongestMaxSize(max_size=self.img_size),
                A.PadIfNeeded(min_height=self.img_size, min_width=self.img_size, 
                             border_mode=cv2.BORDER_CONSTANT, value=0),
                A.OneOf([
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.1),
                    A.RandomRotate90(p=0.2),
                ], p=0.5),
                A.OneOf([
                    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
                    A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
                ], p=0.3),
                A.OneOf([
                    A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
                    A.GaussianBlur(blur_limit=(3, 7), p=0.3),
                    A.MotionBlur(blur_limit=7, p=0.3),
                ], p=0.2),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
        else:
            self.transform = A.Compose([
                A.LongestMaxSize(max_size=self.img_size),
                A.PadIfNeeded(min_height=self.img_size, min_width=self.img_size, 
                             border_mode=cv2.BORDER_CONSTANT, value=0),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
    
    def load_labels(self, label_file):
        """Load YOLO format labels"""
        labels = []
        if label_file.exists():
            with open(label_file, 'r') as f:
                for line in f.readlines():
                    if line.strip():
                        parts = line.strip().split()
                        if len(parts) == 5:
                            class_id = int(parts[0])
                            bbox = [float(x) for x in parts[1:]]
                            labels.append([class_id] + bbox)
        return np.array(labels) if labels else np.empty((0, 5))
    
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.img_files[idx]
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load labels
        label_path = self.label_files[idx]
        labels = self.load_labels(label_path)
        
        if len(labels) == 0:
            bboxes = []
            class_labels = []
        else:
            # Filter out invalid bounding boxes
            valid_labels = []
            for label in labels:
                if len(label) >= 5:
                    class_id, x_center, y_center, width, height = label[:5]
                    # Check if coordinates are valid (within [0, 1] range)
                    # Also check that bbox doesn't go outside image bounds
                    x_min = x_center - width/2
                    y_min = y_center - height/2
                    x_max = x_center + width/2
                    y_max = y_center + height/2
                    
                    if (0 <= x_center <= 1 and 0 <= y_center <= 1 and 
                        0 < width <= 1 and 0 < height <= 1 and
                        0 <= x_min <= 1 and 0 <= y_min <= 1 and
                        0 <= x_max <= 1 and 0 <= y_max <= 1):
                        valid_labels.append(label)
            
            if len(valid_labels) == 0:
                bboxes = []
                class_labels = []
            else:
                valid_labels = np.array(valid_labels)
                bboxes = valid_labels[:, 1:].tolist()  # x_center, y_center, width, height
                class_labels = valid_labels[:, 0].astype(int).tolist()
        
        # Apply augmentations with error handling
        try:
            transformed = self.transform(image=image, bboxes=bboxes, class_labels=class_labels)
            
            return {
                'image': transformed['image'],
                'bboxes': transformed['bboxes'],
                'class_labels': transformed['class_labels'],
                'img_path': str(img_path)
            }
        except Exception as e:
            # If augmentation fails, return image without bboxes
            print(f"Warning: Augmentation failed for {img_path}: {e}")
            # Apply only basic transforms
            basic_transform = A.Compose([
                A.LongestMaxSize(max_size=self.img_size),
                A.PadIfNeeded(min_height=self.img_size, min_width=self.img_size, 
                             border_mode=cv2.BORDER_CONSTANT, value=0),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
            
            transformed = basic_transform(image=image)
            
            return {
                'image': transformed['image'],
                'bboxes': [],
                'class_labels': [],
                'img_path': str(img_path)
            }

class SuperResolutionAugmentation:
    """Super-resolution data augmentation using ESRGAN"""
    
    def __init__(self, scale_factor=2, model_path=None):
        self.scale_factor = scale_factor
        self.model_path = model_path
        
    def apply_sr(self, image):
        """Apply super-resolution to image"""
        # Simple bicubic upsampling as fallback
        h, w = image.shape[:2]
        new_size = (w * self.scale_factor, h * self.scale_factor)
        sr_image = cv2.resize(image, new_size, interpolation=cv2.INTER_CUBIC)
        return sr_image
    
    def process_dataset(self, img_dir, output_dir, ratio=0.3):
        """Apply SR to a percentage of dataset images"""
        img_dir = Path(img_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        img_files = list(img_dir.glob('*.jpg'))
        num_to_process = int(len(img_files) * ratio)
        selected_files = random.sample(img_files, num_to_process)
        
        print(f"Applying super-resolution to {num_to_process} images...")
        
        for img_file in selected_files:
            image = cv2.imread(str(img_file))
            sr_image = self.apply_sr(image)
            
            # Save with SR suffix
            output_path = output_dir / f"{img_file.stem}_sr{img_file.suffix}"
            cv2.imwrite(str(output_path), sr_image)
        
        print(f"Super-resolution augmentation completed. Files saved to {output_dir}")

class CopyPasteAugmentation:
    """Copy-paste augmentation for small objects"""
    
    def __init__(self, prob=0.3):
        self.prob = prob
    
    def apply_copy_paste(self, image, labels, background_image, background_labels):
        """Apply copy-paste augmentation"""
        if random.random() > self.prob:
            return image, labels
        
        # Find small objects (area < 0.05 of image)
        small_objects = []
        for i, label in enumerate(labels):
            if len(label) >= 5:
                _, x_center, y_center, width, height = label[:5]
                area = width * height
                if area < 0.05:  # Small object threshold
                    small_objects.append((i, label))
        
        if not small_objects:
            return image, labels
        
        # Select random small object
        obj_idx, obj_label = random.choice(small_objects)
        
        # Extract object from source image
        h, w = image.shape[:2]
        x_center, y_center, width, height = obj_label[1:5]
        
        # Convert normalized coordinates to pixel coordinates
        x1 = int((x_center - width/2) * w)
        y1 = int((y_center - height/2) * h)
        x2 = int((x_center + width/2) * w)
        y2 = int((y_center + height/2) * h)
        
        # Extract object region
        object_patch = image[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
        
        if object_patch.size == 0:
            return image, labels
        
        # Paste to random location in background
        bg_h, bg_w = background_image.shape[:2]
        paste_x = random.randint(0, max(1, bg_w - object_patch.shape[1]))
        paste_y = random.randint(0, max(1, bg_h - object_patch.shape[0]))
        
        # Create augmented image
        augmented_image = background_image.copy()
        patch_h, patch_w = object_patch.shape[:2]
        augmented_image[paste_y:paste_y+patch_h, paste_x:paste_x+patch_w] = object_patch
        
        # Update labels
        new_x_center = (paste_x + patch_w/2) / bg_w
        new_y_center = (paste_y + patch_h/2) / bg_h
        new_width = patch_w / bg_w
        new_height = patch_h / bg_h
        
        new_label = [obj_label[0], new_x_center, new_y_center, new_width, new_height]
        augmented_labels = background_labels + [new_label]
        
        return augmented_image, augmented_labels

class ActiveLearningSelector:
    """Active learning for efficient annotation"""
    
    def __init__(self, uncertainty_threshold=0.7):
        self.uncertainty_threshold = uncertainty_threshold
        self.selected_samples = []
    
    def calculate_uncertainty(self, predictions):
        """Calculate prediction uncertainty"""
        if len(predictions) == 0:
            return 1.0
        
        # Simple uncertainty measure based on confidence scores
        confidences = [pred.get('confidence', 0.5) for pred in predictions]
        if not confidences:
            return 1.0
        
        max_conf = max(confidences)
        uncertainty = 1.0 - max_conf
        return uncertainty
    
    def select_samples(self, image_paths, predictions, num_samples=100):
        """Select samples for annotation based on uncertainty"""
        uncertainties = []
        
        for img_path, preds in zip(image_paths, predictions):
            uncertainty = self.calculate_uncertainty(preds)
            uncertainties.append((img_path, uncertainty))
        
        # Sort by uncertainty (highest first)
        uncertainties.sort(key=lambda x: x[1], reverse=True)
        
        # Select top uncertain samples
        selected = uncertainties[:num_samples]
        self.selected_samples.extend([sample[0] for sample in selected])
        
        return [sample[0] for sample in selected]

class ClassBalancedAugmentation:
    """Advanced class balancing augmentation for severe class imbalance"""
    
    def __init__(self, dataset_path='./dataset', target_samples_per_class=5000):
        self.dataset_path = dataset_path
        self.target_samples_per_class = target_samples_per_class
        self.class_counts = None
        self.class_distribution = None
        self.analyze_class_distribution()
    
    def analyze_class_distribution(self):
        """Analyze current class distribution in the dataset"""
        print("Analyzing class distribution for balancing...")
        
        train_labels_path = os.path.join(self.dataset_path, 'labels', 'train')
        print(f"Looking for train labels in: {train_labels_path}")
        
        if not os.path.exists(train_labels_path):
            print(f"Warning: Train label directory not found: {train_labels_path}")
            # Try alternative paths
            alt_paths = [
                os.path.join(self.dataset_path, 'train/labels'),
                os.path.join(self.dataset_path, 'labels'),
                os.path.join(self.dataset_path, 'train'),
            ]
            
            for alt_path in alt_paths:
                if os.path.exists(alt_path):
                    print(f"Found alternative path: {alt_path}")
                    train_labels_path = alt_path
                    break
            else:
                raise FileNotFoundError(f"No valid label directory found. Checked: {train_labels_path}, {alt_paths}")
        
        class_counts = Counter()
        class_to_files = defaultdict(list)
        
        label_files = glob.glob(os.path.join(train_labels_path, '*.txt'))
        print(f"Found {len(label_files)} training label files")
        
        if len(label_files) == 0:
            print(f"No .txt files found in {train_labels_path}")
            print(f"Directory contents: {os.listdir(train_labels_path) if os.path.exists(train_labels_path) else 'Directory does not exist'}")
            raise ValueError("No label files found")
        
        for label_file in label_files:
            try:
                with open(label_file, 'r') as f:
                    for line in f.readlines():
                        if line.strip():
                            class_id = int(line.split()[0])
                            class_counts[class_id] += 1
                            class_to_files[class_id].append(label_file)
            except Exception as e:
                print(f"Error processing {label_file}: {e}")
                continue
        
        if not class_counts:
            raise ValueError("No classes found in the dataset. Check if label files contain valid annotations.")
        
        self.class_counts = class_counts
        self.class_to_files = class_to_files
        
        print("Current Class Distribution:")
        for class_id in sorted(class_counts.keys()):
            count = class_counts[class_id]
            percentage = (count / sum(class_counts.values())) * 100
            print(f"  Class {class_id}: {count:,} ({percentage:.2f}%)")
        
        # Calculate class weights
        self.calculate_class_weights()
        
        return class_counts
    
    def calculate_class_weights(self):
        """Calculate balanced class weights using inverse frequency"""
        if self.class_counts is None:
            self.analyze_class_distribution()
        
        if not self.class_counts:
            raise ValueError("No class counts available. Cannot calculate weights.")
        
        # Calculate inverse frequency weights
        max_count = max(self.class_counts.values())
        self.class_weights = {}
        
        for class_id, count in self.class_counts.items():
            if count > 0:
                # Inverse frequency weighting with smoothing
                weight = max_count / count
                # Apply square root to reduce extreme weights
                weight = math.sqrt(weight)
                # Cap maximum weight to prevent over-emphasis
                weight = min(weight, 10.0)
                self.class_weights[class_id] = weight
        
        print("Class Weights:")
        for class_id in sorted(self.class_weights.keys()):
            weight = self.class_weights[class_id]
            print(f"  Class {class_id}: {weight:.3f}")
        
        return self.class_weights
    
    def create_balanced_dataset(self, output_path='./dataset_balanced'):
        """Create a balanced dataset with equal representation of all classes"""
        print(f"Creating balanced dataset at {output_path}")
        
        # Create output directories
        os.makedirs(output_path, exist_ok=True)
        os.makedirs(os.path.join(output_path, 'images', 'train'), exist_ok=True)
        os.makedirs(os.path.join(output_path, 'images', 'val'), exist_ok=True)
        os.makedirs(os.path.join(output_path, 'labels', 'train'), exist_ok=True)
        os.makedirs(os.path.join(output_path, 'labels', 'val'), exist_ok=True)
        
        # Determine target samples per class
        max_class_count = max(self.class_counts.values())
        target_samples = min(self.target_samples_per_class, max_class_count)
        
        print(f"Target samples per class: {target_samples}")
        
        # Process each class
        for class_id in sorted(self.class_counts.keys()):
            current_count = self.class_counts[class_id]
            files_for_class = self.class_to_files[class_id]
            
            print(f"Processing class {class_id}: {current_count} -> {target_samples}")
            
            if current_count < target_samples:
                # Need to augment this class
                self.augment_rare_class(class_id, files_for_class, target_samples, output_path)
            else:
                # Need to undersample this class
                self.undersample_common_class(class_id, files_for_class, target_samples, output_path)
    
    def augment_rare_class(self, class_id, files, target_count, output_path):
        """Augment rare classes using various techniques"""
        print(f"Augmenting class {class_id} from {len(files)} to {target_count} samples")
        
        # Get unique files (remove duplicates)
        unique_files = list(set(files))
        random.shuffle(unique_files)
        
        # Calculate how many augmentations needed per file
        augmentations_per_file = max(1, target_count // len(unique_files))
        
        augmented_count = 0
        for file_path in unique_files:
            if augmented_count >= target_count:
                break
            
            # Load image and labels
            img_path = file_path.replace('labels', 'images').replace('.txt', '.jpg')
            if not os.path.exists(img_path):
                continue
            
            # Load original data
            image = cv2.imread(img_path)
            labels = self.load_labels(file_path)
            
            # Filter labels for this class
            class_labels = [label for label in labels if int(label[0]) == class_id]
            
            if len(class_labels) == 0:
                continue
            
            # Create multiple augmented versions
            for i in range(augmentations_per_file):
                if augmented_count >= target_count:
                    break
                
                # Apply different augmentation strategies
                if i == 0:
                    # Original image
                    aug_image, aug_labels = image.copy(), class_labels.copy()
                else:
                    # Augmented versions
                    aug_image, aug_labels = self.apply_class_specific_augmentation(
                        image, class_labels, class_id, i
                    )
                
                # Save augmented sample
                sample_id = f"class_{class_id}_aug_{augmented_count:06d}"
                self.save_augmented_sample(
                    aug_image, aug_labels, sample_id, output_path
                )
                
                augmented_count += 1
        
        print(f"Created {augmented_count} augmented samples for class {class_id}")
    
    def apply_class_specific_augmentation(self, image, labels, class_id, augmentation_id):
        """Apply class-specific augmentation strategies"""
        # Different augmentation strategies based on class type
        if class_id in [4, 5, 6, 9]:  # Vehicles (car, van, truck, bus)
            # Vehicle-specific augmentations
            transform = A.Compose([
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.8),
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.7),
                A.OneOf([
                    A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
                    A.GaussianBlur(blur_limit=(3, 7), p=0.5),
                ], p=0.6),
                A.OneOf([
                    A.RandomRotate90(p=0.3),
                    A.HorizontalFlip(p=0.5),
                ], p=0.7),
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
        else:
            # General augmentations for other classes
            transform = A.Compose([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.8),
                A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=15, p=0.7),
                A.OneOf([
                    A.GaussNoise(var_limit=(5.0, 25.0), p=0.4),
                    A.GaussianBlur(blur_limit=(3, 5), p=0.4),
                ], p=0.5),
                A.HorizontalFlip(p=0.5),
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
        
        # Apply transformation
        transformed = transform(image=image, bboxes=labels, class_labels=[label[0] for label in labels])
        
        return transformed['image'], transformed['bboxes']
    
    def undersample_common_class(self, class_id, files, target_count, output_path):
        """Undersample common classes to balance the dataset"""
        print(f"Undersampling class {class_id} from {len(files)} to {target_count} samples")
        
        # Randomly select target number of files
        selected_files = random.sample(files, min(target_count, len(files)))
        
        for file_path in selected_files:
            # Copy original file
            img_path = file_path.replace('labels', 'images').replace('.txt', '.jpg')
            if not os.path.exists(img_path):
                continue
            
            # Load and filter labels for this class
            labels = self.load_labels(file_path)
            class_labels = [label for label in labels if int(label[0]) == class_id]
            
            if len(class_labels) == 0:
                continue
            
            # Save sample
            sample_id = f"class_{class_id}_orig_{len(selected_files):06d}"
            self.save_augmented_sample(
                cv2.imread(img_path), class_labels, sample_id, output_path
            )
    
    def save_augmented_sample(self, image, labels, sample_id, output_path):
        """Save augmented image and labels"""
        # Save image
        img_path = os.path.join(output_path, 'images', 'train', f"{sample_id}.jpg")
        cv2.imwrite(img_path, image)
        
        # Save labels
        label_path = os.path.join(output_path, 'labels', 'train', f"{sample_id}.txt")
        with open(label_path, 'w') as f:
            for label in labels:
                if isinstance(label, (list, tuple)):
                    f.write(f"{int(label[0])} {label[1]:.6f} {label[2]:.6f} {label[3]:.6f} {label[4]:.6f}\n")
                else:
                    f.write(f"{label}\n")
    
    def load_labels(self, label_file):
        """Load YOLO format labels"""
        labels = []
        if os.path.exists(label_file):
            with open(label_file, 'r') as f:
                for line in f.readlines():
                    if line.strip():
                        parts = line.strip().split()
                        if len(parts) == 5:
                            class_id = int(parts[0])
                            bbox = [float(x) for x in parts[1:]]
                            labels.append([class_id] + bbox)
        return labels

def create_dataset_yaml(dataset_path):
    """Create dataset YAML configuration for YOLOv8"""
    dataset_config = {
        'path': str(Path(dataset_path).absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': 11,  # Number of classes
        'names': {
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
    }
    
    yaml_path = Path(dataset_path) / 'dataset.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(dataset_config, f, default_flow_style=False)
    
    print(f"Dataset YAML created at {yaml_path}")
    return str(yaml_path)

def create_balanced_dataset_yaml(dataset_path):
    """Create balanced dataset YAML configuration"""
    dataset_config = {
        'path': str(Path(dataset_path).absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': 11,  # Number of classes
        'names': {
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
        },
        'balanced': True,
        'target_samples_per_class': 5000
    }
    
    yaml_path = Path(dataset_path) / 'dataset_balanced.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(dataset_config, f, default_flow_style=False)
    
    print(f"Balanced dataset YAML created at {yaml_path}")
    return str(yaml_path)

def prepare_visdrone_dataset(dataset_path='./dataset'):
    """Prepare VisDrone dataset for training"""
    dataset_path = Path(dataset_path)
    
    print("Preparing VisDrone dataset...")
    
    # Check if dataset exists
    if not dataset_path.exists():
        print(f"Dataset path {dataset_path} does not exist!")
        return None
    
    # Create dataset YAML
    yaml_path = create_dataset_yaml(dataset_path)
    
    # Initialize augmentation classes
    sr_aug = SuperResolutionAugmentation(scale_factor=2)
    copy_paste_aug = CopyPasteAugmentation(prob=0.3)
    
    # Apply super-resolution augmentation to training set
    train_img_dir = dataset_path / 'images' / 'train'
    if train_img_dir.exists():
        sr_output_dir = dataset_path / 'images' / 'train_sr'
        # sr_aug.process_dataset(train_img_dir, sr_output_dir, ratio=0.2)
    
    print("Dataset preparation completed!")
    return yaml_path

def get_dataloader(img_dir, label_dir, batch_size=16, img_size=640, augment=True):
    """Create DataLoader for training"""
    dataset = VisDroneDataset(img_dir, label_dir, img_size=img_size, augment=augment)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True,
        collate_fn=lambda x: x  # Custom collate function for variable number of objects
    )
    return dataloader

if __name__ == "__main__":
    # Test data preparation
    dataset_path = "./dataset"
    yaml_path = prepare_visdrone_dataset(dataset_path)
    
    if yaml_path:
        print(f"Dataset prepared successfully! YAML config: {yaml_path}")
        
        # Test dataset loading
        train_img_dir = f"{dataset_path}/images/train"
        train_label_dir = f"{dataset_path}/labels/train"
        
        if os.path.exists(train_img_dir) and os.path.exists(train_label_dir):
            dataloader = get_dataloader(train_img_dir, train_label_dir, batch_size=4)
            print(f"DataLoader created with {len(dataloader)} batches")
            
            # Test loading one batch
            try:
                sample_batch = next(iter(dataloader))
                print(f"Sample batch loaded with {len(sample_batch)} items")
                print(f"Sample image shape: {sample_batch[0]['image'].shape}")
            except Exception as e:
                print(f"Error loading sample batch: {e}")
    else:
        print("Failed to prepare dataset!") 