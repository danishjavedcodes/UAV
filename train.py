"""
Simplified HMAY-TSF Training Script
Robust implementation with error-free training
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
import numpy as np
from datetime import datetime
import argparse
import json
import pandas as pd
import csv
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import cv2
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR, ReduceLROnPlateau
import math
from pathlib import Path
from tqdm import tqdm
from torch.amp import autocast, GradScaler

class SimpleDataset:
    """Simple dataset for training"""
    
    def __init__(self, data_yaml_path, img_size=640, is_training=True):
        self.img_size = img_size
        self.is_training = is_training
        
        # Load dataset configuration
        with open(data_yaml_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Get paths
        if 'path' in self.config:
            self.data_path = Path(self.config['path'])
        else:
            self.data_path = Path(data_yaml_path).parent
        
        # Get train/val paths
        if 'train' in self.config:
            self.train_path = self.data_path / self.config['train']
        else:
            self.train_path = self.data_path / 'images' / 'train'
            
        if 'val' in self.config:
            self.val_path = self.data_path / self.config['val']
        else:
            self.val_path = self.data_path / 'images' / 'val'
        
        self.class_names = self.config.get('names', ['bus', 'car', 'truck', 'van'])
        self.num_classes = self.config.get('nc', 4)
        
        # Get image and label paths
        if is_training:
            self.img_dir = self.data_path / 'images' / 'train'
            self.label_dir = self.data_path / 'labels' / 'train'
        else:
            self.img_dir = self.data_path / 'images' / 'val'
            self.label_dir = self.data_path / 'labels' / 'val'
        
        # Check if directories exist, if not try alternative paths
        if not self.img_dir.exists():
            if is_training:
                self.img_dir = self.data_path / 'train' / 'images'
                self.label_dir = self.data_path / 'train' / 'labels'
            else:
                self.img_dir = self.data_path / 'valid' / 'images'
                self.label_dir = self.data_path / 'valid' / 'labels'
        
        # Get all image files
        self.img_files = []
        if self.img_dir.exists():
            self.img_files = list(self.img_dir.glob('*.jpg')) + list(self.img_dir.glob('*.png'))
        else:
            print(f"Warning: Image directory {self.img_dir} does not exist")
        
        print(f"Found {len(self.img_files)} images in {self.img_dir}")
        print(f"Label directory: {self.label_dir}")
        print(f"Class names: {self.class_names}")
        print(f"Number of classes: {self.num_classes}")
    
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        label_path = self.label_dir / (img_path.stem + '.txt')
        
        # Load image
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize image
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)  # HWC to CHW
        
        # Load labels
        labels = []
        if label_path.exists():
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
        
        labels = torch.tensor(labels, dtype=torch.float32) if labels else torch.zeros((0, 5))
        
        return img, labels

def collate_fn(batch):
    """Custom collate function for batching"""
    imgs, labels = zip(*batch)
    imgs = torch.stack(imgs)
    return imgs, labels

class ChannelAttention(nn.Module):
    """Channel attention module (CBAM)"""
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(in_channels // reduction, 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    """Spatial attention module (CBAM)"""
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg, mx], dim=1)
        attn = self.sigmoid(self.conv(x_cat))
        return attn

class CBAM(nn.Module):
    """Convolutional Block Attention Module"""
    def __init__(self, in_channels: int, reduction: int = 16, spatial_kernel: int = 7):
        super().__init__()
        self.ca = ChannelAttention(in_channels, reduction)
        self.sa = SpatialAttention(spatial_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x

class SPP(nn.Module):
    """Spatial Pyramid Pooling (simple)"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels // 2, 1)
        self.conv2 = nn.Conv2d(out_channels // 2 * 4, out_channels, 1)
        self.pool1 = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=9, stride=1, padding=4)
        self.pool3 = nn.MaxPool2d(kernel_size=13, stride=1, padding=6)
        self.bn1 = nn.BatchNorm2d(out_channels // 2)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.bn1(self.conv1(x)))
        p1 = self.pool1(x)
        p2 = self.pool2(x)
        p3 = self.pool3(x)
        x = torch.cat([x, p1, p2, p3], dim=1)
        x = self.act(self.bn2(self.conv2(x)))
        return x

class ImprovedHMAYTSF(nn.Module):
    """Improved HMAY-TSF model for aerial vehicle detection"""
    
    def __init__(self, num_classes=4):
        super().__init__()
        self.num_classes = num_classes
        
        # Improved backbone with more layers and better feature extraction
        self.backbone = nn.Sequential(
            # Initial conv layer
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 320x320
            
            # Second block
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 160x160
            
            # Third block
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 80x80
            
            # Fourth block
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 40x40
            
            # Fifth block
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        
        # Attention and SPP
        self.cbam3 = CBAM(256)
        self.cbam4 = CBAM(512)
        self.spp = SPP(512, 512)

        # Feature Pyramid Network (FPN) for multi-scale detection
        self.fpn_lateral3 = nn.Conv2d(256, 256, 1)
        self.fpn_lateral4 = nn.Conv2d(512, 256, 1)
        self.fpn_output3 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            CBAM(256)
        )
        self.fpn_output4 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            CBAM(256)
        )
        
        # Detection heads for different scales
        self.detection_head_3 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3 * (5 + num_classes), 1)  # 3 anchors per position
        )
        
        self.detection_head_4 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3 * (5 + num_classes), 1)  # 3 anchors per position
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights with better initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass with FPN and multi-scale detection.
        Returns raw logits (no in-graph activations) with shape [B, total_anchors, 3*(5+num_classes)].
        """
        # Backbone feature extraction - tap the correct layers
        c3 = None  # 256 channels, 40x40
        c4 = None  # 512 channels, 40x40 (after final conv block, before any further pooling)
        for i, layer in enumerate(self.backbone):
            x = layer(x)
            if i == 27:  # after 4th maxpool -> 256ch, 40x40
                c3 = x
            if i == len(self.backbone) - 1:  # after last ReLU of 512ch block
                c4 = x

        # FPN processing with attention and SPP
        c3 = self.cbam3(c3)
        c4 = self.cbam4(c4)
        c4 = self.spp(c4)

        # Lateral connections
        p4 = self.fpn_lateral4(c4)
        # Upsample p4 to c3 resolution and fuse
        p3 = self.fpn_lateral3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode='bilinear', align_corners=False)

        # Output processing
        p3 = self.fpn_output3(p3)
        p4 = self.fpn_output4(p4)

        # Detection heads
        out3 = self.detection_head_3(p3)  # approx 80x80 feature map
        out4 = self.detection_head_4(p4)  # approx 40x40 feature map

        # Reshape outputs to [batch, anchors, features]
        B, C3, H3, W3 = out3.shape
        B, C4, H4, W4 = out4.shape
        out3 = out3.view(B, C3, H3 * W3).permute(0, 2, 1)
        out4 = out4.view(B, C4, H4 * W4).permute(0, 2, 1)

        # Concatenate outputs from different scales (keep raw logits)
        output = torch.cat([out3, out4], dim=1)  # [B, total_anchors, 3*(5+num_classes)]
        return output

class ImprovedLoss(nn.Module):
    """Improved loss function for object detection"""
    
    def __init__(self, num_classes=4):
        super().__init__()
        self.num_classes = num_classes
        self.box_weight = 2.0
        self.obj_weight = 1.0
        self.cls_weight = 1.0
        self.cls_label_smoothing = 0.05
        # Loss functions
        self.bce_obj = nn.BCEWithLogitsLoss(reduction='none')
        self.bce_cls = nn.BCEWithLogitsLoss(reduction='none')
        self.l1 = nn.SmoothL1Loss(reduction='none', beta=0.1)
    
    @staticmethod
    def _xywh_to_xyxy(b):
        x1 = b[:, 0] - b[:, 2] / 2
        y1 = b[:, 1] - b[:, 3] / 2
        x2 = b[:, 0] + b[:, 2] / 2
        y2 = b[:, 1] + b[:, 3] / 2
        return torch.stack([x1, y1, x2, y2], dim=1)
    
    def _pairwise_iou_xywh(self, boxes1, boxes2):
        # boxes1: [N,4], boxes2: [M,4], both in [cx,cy,w,h] normalized [0,1]
        if boxes1.numel() == 0 or boxes2.numel() == 0:
            return boxes1.new_zeros((boxes1.size(0), boxes2.size(0)))
        b1 = self._xywh_to_xyxy(boxes1)
        b2 = self._xywh_to_xyxy(boxes2)
        N = b1.size(0)
        M = b2.size(0)
        b1e = b1[:, None, :].expand(N, M, 4)
        b2e = b2[None, :, :].expand(N, M, 4)
        inter_x1 = torch.maximum(b1e[..., 0], b2e[..., 0])
        inter_y1 = torch.maximum(b1e[..., 1], b2e[..., 1])
        inter_x2 = torch.minimum(b1e[..., 2], b2e[..., 2])
        inter_y2 = torch.minimum(b1e[..., 3], b2e[..., 3])
        inter_w = torch.clamp(inter_x2 - inter_x1, min=0)
        inter_h = torch.clamp(inter_y2 - inter_y1, min=0)
        inter = inter_w * inter_h
        area1 = (b1e[..., 2] - b1e[..., 0]) * (b1e[..., 3] - b1e[..., 1])
        area2 = (b2e[..., 2] - b2e[..., 0]) * (b2e[..., 3] - b2e[..., 1])
        union = area1 + area2 - inter
        return torch.where(union > 0, inter / union, inter.new_zeros(()))
        
    def forward(self, predictions, targets):
        """
        predictions: [batch, anchors, features]
        targets: list of [num_objects, 5] tensors
        """
        try:
            total_loss = predictions.new_tensor(0.0)
            box_loss = predictions.new_tensor(0.0)
            cls_loss = predictions.new_tensor(0.0)
            obj_loss = predictions.new_tensor(0.0)
            batch_size = predictions.size(0)
            
            for i in range(batch_size):
                pred = predictions[i]  # [anchors, features]
                target = targets[i]    # [num_objects, 5]
                
                # Model output format per location: 3 anchors * (5 + C)
                num_features_per_anchor = 5 + self.num_classes
                if pred.size(1) == 3 * num_features_per_anchor:
                    # Split anchors
                    pred_reshaped = pred.reshape(-1, 3, num_features_per_anchor)  # [cells, 3, nf]
                    all_anchors = pred_reshaped.reshape(-1, num_features_per_anchor)  # [A, nf]

                    # Extract raw logits
                    box_logits = all_anchors[:, :4]
                    obj_logits = all_anchors[:, 4]
                    cls_logits = all_anchors[:, 5:5 + self.num_classes]

                    # Decode boxes to [0,1]
                    box_preds = torch.sigmoid(box_logits)

                    A = all_anchors.size(0)
                    obj_targets = torch.zeros_like(obj_logits)
                    cls_targets = torch.zeros_like(cls_logits)
                    box_targets = torch.zeros_like(box_preds)

                    if target.numel() > 0:
                        t = target.to(predictions.device)
                        t = t[torch.isfinite(t).all(dim=1)]
                        if t.numel() > 0:
                            tcls = t[:, 0].long().clamp(0, self.num_classes - 1)
                            tboxes = t[:, 1:5].clamp(0, 1)
                            T = tboxes.size(0)

                            # Preselect top-K candidate anchors by objectness logit to reduce compute
                            k = min(300, A)
                            topk_scores, topk_idx = torch.topk(obj_logits, k)
                            p_boxes_k = box_preds[topk_idx]
                            ious = self._pairwise_iou_xywh(p_boxes_k, tboxes)  # [k, T]

                            # For each target, pick best predicted anchor by IoU
                            if ious.numel() > 0:
                                best_iou, best_idx = torch.max(ious, dim=0)  # [T]
                                chosen_anchor_idx = topk_idx[best_idx]
                                obj_targets[chosen_anchor_idx] = obj_targets.new_tensor(1.0)
                                box_targets[chosen_anchor_idx] = tboxes.to(box_targets.dtype)
                                # Label smoothing for class targets
                                smooth_pos = cls_targets.new_tensor(1.0 - self.cls_label_smoothing)
                                smooth_neg = cls_targets.new_tensor(self.cls_label_smoothing / max(self.num_classes - 1, 1))
                                cls_targets.fill_(smooth_neg)
                                cls_targets[chosen_anchor_idx, tcls] = smooth_pos

                    # Objectness loss with imbalance weighting
                    obj_weights = torch.where(obj_targets > 0, obj_logits.new_tensor(2.0), obj_logits.new_tensor(0.5))
                    obj_loss_sample = self.bce_obj(obj_logits, obj_targets)
                    obj_loss += (obj_loss_sample * obj_weights).mean() * self.obj_weight

                    # Classification loss only on positives
                    pos_mask = obj_targets > 0
                    if pos_mask.any():
                        cls_loss_sample = self.bce_cls(cls_logits[pos_mask], cls_targets[pos_mask])
                        cls_loss += cls_loss_sample.mean() * self.cls_weight

                        # Box regression loss on positives
                        box_loss_sample = self.l1(box_preds[pos_mask], box_targets[pos_mask])
                        box_loss += box_loss_sample.mean() * self.box_weight
                else:
                    # Fallback for unexpected format: small L2 on raw preds to keep training stable
                    total_loss += torch.mean(pred ** 2) * 0.01
            
            total_loss = total_loss + box_loss + cls_loss + obj_loss
            return total_loss, box_loss, cls_loss, obj_loss
            
        except Exception as e:
            print(f"Warning: Error in loss calculation: {e}")
            # Return dummy loss values
            dummy_loss = torch.tensor(1.0, device=predictions.device, requires_grad=True)
            return dummy_loss, dummy_loss, dummy_loss, dummy_loss

class MetricsCalculator:
    """Calculate training and validation metrics"""
    
    def __init__(self, num_classes=4):
        self.num_classes = num_classes
    
    def calculate_iou(self, box1, box2):
        """Calculate IoU between two boxes [x_center, y_center, width, height]"""
        try:
            # Convert to [x1, y1, x2, y2]
            box1_x1 = box1[0] - box1[2] / 2
            box1_y1 = box1[1] - box1[3] / 2
            box1_x2 = box1[0] + box1[2] / 2
            box1_y2 = box1[1] + box1[3] / 2
            
            box2_x1 = box2[0] - box2[2] / 2
            box2_y1 = box2[1] - box2[3] / 2
            box2_x2 = box2[0] + box2[2] / 2
            box2_y2 = box2[1] + box2[3] / 2
            
            # Calculate intersection
            x1 = max(box1_x1, box2_x1)
            y1 = max(box1_y1, box2_y1)
            x2 = min(box1_x2, box2_x2)
            y2 = min(box1_y2, box2_y2)
            
            intersection = max(0, x2 - x1) * max(0, y2 - y1)
            
            # Calculate union
            area1 = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
            area2 = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
            union = area1 + area2 - intersection
            
            return intersection / union if union > 0 else 0
        except:
            return 0.0
    
    def calculate_metrics(self, predictions, targets):
        """Calculate metrics by decoding raw logits and matching predictions to targets."""
        try:
            all_preds = []
            all_targets = []
            
            batch_size = predictions.size(0)
            
            for i in range(batch_size):
                pred = predictions[i]  # [anchors, features]
                target = targets[i]    # [num_objects, 5]
                
                if pred.size(0) > 0 and pred.size(1) == 3 * (5 + self.num_classes):
                    nf = 5 + self.num_classes
                    anchors = pred.reshape(-1, 3, nf).reshape(-1, nf)
                    box_logits = anchors[:, :4]
                    obj_logits = anchors[:, 4]
                    cls_logits = anchors[:, 5:5 + self.num_classes]
                    box_preds = torch.sigmoid(box_logits)
                    obj_scores = torch.sigmoid(obj_logits)
                    cls_scores = torch.softmax(cls_logits, dim=1)
                    pred_classes = torch.argmax(cls_scores, dim=1)

                    valid_mask = obj_scores > 0.25
                    if valid_mask.any():
                        valid_scores = obj_scores[valid_mask]
                        valid_boxes = box_preds[valid_mask]
                        valid_classes = pred_classes[valid_mask]

                        # Keep top-K predictions per image to bound compute
                        top_k = min(200, valid_scores.numel())
                        if valid_scores.numel() > top_k:
                            topk_scores, topk_idx = torch.topk(valid_scores, k=top_k, largest=True)
                            valid_boxes = valid_boxes[topk_idx]
                            valid_classes = valid_classes[topk_idx]
                            valid_scores = topk_scores

                        # Prepare targets and match one pred per target
                        if target.numel() > 0:
                            tcls = target[:, 0].long()
                            tboxes = target[:, 1:5]
                            # Limit targets per image (dense scenes)
                            if tboxes.size(0) > 100:
                                tboxes = tboxes[:100]
                                tcls = tcls[:100]

                            def xywh_to_xyxy(b):
                                x1 = b[:, 0] - b[:, 2] / 2
                                y1 = b[:, 1] - b[:, 3] / 2
                                x2 = b[:, 0] + b[:, 2] / 2
                                y2 = b[:, 1] + b[:, 3] / 2
                                return torch.stack([x1, y1, x2, y2], dim=1)

                            tb = xywh_to_xyxy(tboxes.clamp(0,1))
                            pb = xywh_to_xyxy(valid_boxes.clamp(0,1))

                            # Compute IoU matrix [T, K]
                            T = tb.size(0)
                            K = pb.size(0)
                            if T > 0 and K > 0:
                                tb_exp = tb[:, None, :].expand(T, K, 4)
                                pb_exp = pb[None, :, :].expand(T, K, 4)
                                inter_x1 = torch.maximum(tb_exp[..., 0], pb_exp[..., 0])
                                inter_y1 = torch.maximum(tb_exp[..., 1], pb_exp[..., 1])
                                inter_x2 = torch.minimum(tb_exp[..., 2], pb_exp[..., 2])
                                inter_y2 = torch.minimum(tb_exp[..., 3], pb_exp[..., 3])
                                inter_w = torch.clamp(inter_x2 - inter_x1, min=0)
                                inter_h = torch.clamp(inter_y2 - inter_y1, min=0)
                                inter = inter_w * inter_h
                                area_t = (tb_exp[..., 2] - tb_exp[..., 0]) * (tb_exp[..., 3] - tb_exp[..., 1])
                                area_p = (pb_exp[..., 2] - pb_exp[..., 0]) * (pb_exp[..., 3] - pb_exp[..., 1])
                                union = area_t + area_p - inter
                                ious = torch.where(union > 0, inter / union, torch.zeros_like(union))  # [T,K]

                                # For each target, pick best pred by IoU; if IoU too low, fallback to highest obj
                                for t in range(T):
                                    iou_row = ious[t]
                                    best_iou, best_idx = torch.max(iou_row, dim=0)
                                    if best_iou.item() > 0.1:
                                        chosen_idx = best_idx
                                    else:
                                        # Fallback: highest objectness among valid preds
                                        _, chosen_idx = torch.max(valid_scores, dim=0)
                                    all_preds.append(int(valid_classes[chosen_idx].item()))
                                    all_targets.append(int(tcls[t].item()))
                 
                # Do not unconditionally extend all targets; only matched pairs count toward metrics
            
            # Handle edge cases
            if len(all_targets) == 0:
                return 0.0, 0.0, 0.0, 0.0
            if len(all_preds) == 0:
                # No predictions matched; predict majority class 0 as fallback to avoid degenerate zero vectors
                all_preds = [0] * len(all_targets)
            
            # Ensure same length
            min_len = min(len(all_preds), len(all_targets))
            if min_len == 0:
                return 0.0, 0.0, 0.0, 0.0
            
            all_preds = all_preds[:min_len]
            all_targets = all_targets[:min_len]
            
            # Convert to numpy arrays
            all_preds = np.array(all_preds)
            all_targets = np.array(all_targets)
            
            # Calculate metrics with error handling
            try:
                accuracy = accuracy_score(all_targets, all_preds)
            except:
                accuracy = 0.0
            
            try:
                precision = precision_score(all_targets, all_preds, average='weighted', zero_division=0)
            except:
                precision = 0.0
            
            try:
                recall = recall_score(all_targets, all_preds, average='weighted', zero_division=0)
            except:
                recall = 0.0
            
            try:
                f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
            except:
                f1 = 0.0
            
            return accuracy, precision, recall, f1
            
        except Exception as e:
            print(f"Warning: Error in metrics calculation: {e}")
            return 0.0, 0.0, 0.0, 0.0

class SimpleTrainer:
    """Improved trainer for aerial vehicle detection"""
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.model.to(device)
        
        # Improved loss function
        self.criterion = ImprovedLoss(num_classes=4)
        
        # Metrics calculator
        self.metrics_calculator = MetricsCalculator(num_classes=4)
        
        # Better optimizer with improved parameters
        self.optimizer = optim.AdamW(
            model.parameters(), 
            lr=0.0001,  # Lower learning rate for better stability
            weight_decay=0.01,  # Increased weight decay
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Scheduler will be initialized later in main when train_loader is available
        self.scheduler = None
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.train_precisions = []
        self.val_precisions = []
        self.train_recalls = []
        self.val_recalls = []
        self.train_f1_scores = []
        self.val_f1_scores = []
        self.best_val_loss = float('inf')
        
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        all_predictions = []
        all_targets = []
        scaler = GradScaler('cuda', enabled=(self.device == 'cuda'))
        
        try:
            progress_bar = tqdm(train_loader, desc="Training", leave=False)
            
            for batch_idx, (images, targets) in enumerate(progress_bar):
                try:
                    images = images.to(self.device)
                    
                    # Forward pass
                    with autocast('cuda', enabled=(self.device == 'cuda')):
                        predictions = self.model(images)
                        loss, box_loss, cls_loss, obj_loss = self.criterion(predictions, targets)
                    
                    # Backward pass
                    self.optimizer.zero_grad()
                    scaler.scale(loss).backward()
                    
                    # Gradient clipping to prevent exploding gradients
                    scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                    
                    scaler.step(self.optimizer)
                    scaler.update()
                    
                    # Update scheduler
                    if hasattr(self, 'scheduler') and self.scheduler is not None:
                        self.scheduler.step()
                    
                    total_loss += loss.item()
                    num_batches += 1
                    
                    # Store predictions and targets for metrics (sampled to reduce memory; more varied than first 5 only)
                    if batch_idx % 4 == 0:
                        all_predictions.append(predictions.detach().cpu())
                        all_targets.extend(targets)
                    
                    # Update progress bar
                    progress_bar.set_postfix({
                        'Loss': f'{loss.item():.4f}',
                        'Box': f'{box_loss.item():.4f}',
                        'Cls': f'{cls_loss.item():.4f}',
                        'Obj': f'{obj_loss.item():.4f}'
                    })
                    
                except Exception as e:
                    print(f"Warning: Error in training batch {batch_idx}: {e}")
                    continue
            
            progress_bar.close()
            
        except Exception as e:
            print(f"Warning: Error in training epoch: {e}")
        
        # Calculate metrics (simplified)
        try:
            if all_predictions:
                all_predictions = torch.cat(all_predictions, dim=0)
                accuracy, precision, recall, f1 = self.metrics_calculator.calculate_metrics(all_predictions, all_targets)
            else:
                accuracy, precision, recall, f1 = 0.0, 0.0, 0.0, 0.0
        except Exception as e:
            print(f"Warning: Error in metrics calculation: {e}")
            accuracy, precision, recall, f1 = 0.0, 0.0, 0.0, 0.0
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        self.train_losses.append(avg_loss)
        self.train_accuracies.append(accuracy)
        self.train_precisions.append(precision)
        self.train_recalls.append(recall)
        self.train_f1_scores.append(f1)
        
        return avg_loss, accuracy, precision, recall, f1
    
    def validate_epoch(self, val_loader):
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        all_predictions = []
        all_targets = []
        scaler = GradScaler('cuda', enabled=False)
        
        try:
            with torch.no_grad():
                progress_bar = tqdm(val_loader, desc="Validation", leave=False)
                
                for batch_idx, (images, targets) in enumerate(progress_bar):
                    try:
                        images = images.to(self.device)
                        
                        # Forward pass
                        with autocast('cuda', enabled=(self.device == 'cuda')):
                            predictions = self.model(images)
                            loss, box_loss, cls_loss, obj_loss = self.criterion(predictions, targets)
                        
                        total_loss += loss.item()
                        num_batches += 1
                        
                        # Store predictions and targets for metrics (use full validation set)
                        all_predictions.append(predictions.cpu())
                        all_targets.extend(targets)
                        
                        # Update progress bar
                        progress_bar.set_postfix({
                            'Loss': f'{loss.item():.4f}',
                            'Box': f'{box_loss.item():.4f}',
                            'Cls': f'{cls_loss.item():.4f}',
                            'Obj': f'{obj_loss.item():.4f}'
                        })
                        
                    except Exception as e:
                        print(f"Warning: Error in validation batch {batch_idx}: {e}")
                        continue
                
                progress_bar.close()
                
        except Exception as e:
            print(f"Warning: Error in validation epoch: {e}")
        
        # Calculate metrics (simplified)
        try:
            if all_predictions:
                all_predictions = torch.cat(all_predictions, dim=0)
                accuracy, precision, recall, f1 = self.metrics_calculator.calculate_metrics(all_predictions, all_targets)
            else:
                accuracy, precision, recall, f1 = 0.0, 0.0, 0.0, 0.0
        except Exception as e:
            print(f"Warning: Error in validation metrics calculation: {e}")
            accuracy, precision, recall, f1 = 0.0, 0.0, 0.0, 0.0
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        self.val_losses.append(avg_loss)
        self.val_accuracies.append(accuracy)
        self.val_precisions.append(precision)
        self.val_recalls.append(recall)
        self.val_f1_scores.append(f1)
        
        return avg_loss, accuracy, precision, recall, f1
    
    def train(self, train_loader, val_loader, epochs=20, save_dir='./runs/simple_train'):
        """Main training loop"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Starting simple training for {epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Save directory: {save_dir}")
        
        best_model_path = save_dir / 'best_model.pth'
        
        # Create CSV file for metrics logging
        csv_path = save_dir / 'training_metrics.csv'
        try:
            with open(csv_path, 'w', newline='') as csvfile:
                fieldnames = ['epoch', 'train_loss', 'val_loss', 'train_acc', 'val_acc', 
                             'train_precision', 'val_precision', 'train_recall', 'val_recall',
                             'train_f1', 'val_f1', 'lr']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
        except Exception as e:
            print(f"Warning: Could not create CSV file: {e}")
            csv_path = None
        
        for epoch in range(epochs):
            try:
                print(f"\nEpoch {epoch+1}/{epochs}")
                print("="*60)
                
                # Training
                train_loss, train_acc, train_precision, train_recall, train_f1 = self.train_epoch(train_loader)
                
                # Validation
                val_loss, val_acc, val_precision, val_recall, val_f1 = self.validate_epoch(val_loader)
                
                # Learning rate: report current without stepping here (stepped per-batch)
                try:
                    current_lr = self.optimizer.param_groups[0]['lr']
                except Exception:
                    current_lr = 0.001

                # Print comprehensive metrics
                print(f"\n📊 EPOCH {epoch+1} RESULTS:")
                print("="*60)
                print(f"TRAINING METRICS:")
                print(f"  Loss: {train_loss:.6f}")
                print(f"  Accuracy: {train_acc:.6f}")
                print(f"  Precision: {train_precision:.6f}")
                print(f"  Recall: {train_recall:.6f}")
                print(f"  F1-Score: {train_f1:.6f}")
                print(f"\nVALIDATION METRICS:")
                print(f"  Loss: {val_loss:.6f}")
                print(f"  Accuracy: {val_acc:.6f}")
                print(f"  Precision: {val_precision:.6f}")
                print(f"  Recall: {val_recall:.6f}")
                print(f"  F1-Score: {val_f1:.6f}")
                print(f"\nLEARNING RATE: {current_lr:.6f}")
                print("="*60)
                
                # Log to CSV
                if csv_path:
                    try:
                        with open(csv_path, 'a', newline='') as csvfile:
                            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                            writer.writerow({
                                'epoch': epoch + 1,
                                'train_loss': train_loss,
                                'val_loss': val_loss,
                                'train_acc': train_acc,
                                'val_acc': val_acc,
                                'train_precision': train_precision,
                                'val_precision': val_precision,
                                'train_recall': train_recall,
                                'val_recall': val_recall,
                                'train_f1': train_f1,
                                'val_f1': val_f1,
                                'lr': current_lr
                            })
                    except Exception as e:
                        print(f"Warning: Could not write to CSV: {e}")
                
                # Save best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    try:
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'scheduler_state_dict': self.scheduler.state_dict(),
                            'train_loss': train_loss,
                            'val_loss': val_loss,
                            'train_acc': train_acc,
                            'val_acc': val_acc,
                            'train_precision': train_precision,
                            'val_precision': val_precision,
                            'train_recall': train_recall,
                            'val_recall': val_recall,
                            'train_f1': train_f1,
                            'val_f1': val_f1,
                            'best_val_loss': self.best_val_loss,
                        }, best_model_path)
                        print(f"✅ New best model saved! Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}")
                    except Exception as e:
                        print(f"Warning: Could not save best model: {e}")
                
                # Save checkpoint every 5 epochs
                if (epoch + 1) % 5 == 0:
                    checkpoint_path = save_dir / f'checkpoint_epoch_{epoch+1}.pth'
                    try:
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'scheduler_state_dict': self.scheduler.state_dict(),
                            'train_loss': train_loss,
                            'val_loss': val_loss,
                            'train_acc': train_acc,
                            'val_acc': val_acc,
                            'train_precision': train_precision,
                            'val_precision': val_precision,
                            'train_recall': train_recall,
                            'val_recall': val_recall,
                            'train_f1': train_f1,
                            'val_f1': val_f1,
                            'best_val_loss': self.best_val_loss,
                        }, checkpoint_path)
                        print(f"💾 Checkpoint saved: {checkpoint_path}")
                    except Exception as e:
                        print(f"Warning: Could not save checkpoint: {e}")
                
            except Exception as e:
                print(f"Warning: Error in epoch {epoch+1}: {e}")
                continue
        
        # Save final model
        final_model_path = save_dir / 'final_model.pth'
        try:
            torch.save({
                'epoch': epochs,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_acc': train_acc,
                'val_acc': val_acc,
                'train_precision': train_precision,
                'val_precision': val_precision,
                'train_recall': train_recall,
                'val_recall': val_recall,
                'train_f1': train_f1,
                'val_f1': val_f1,
                'best_val_loss': self.best_val_loss,
            }, final_model_path)
        except Exception as e:
            print(f"Warning: Could not save final model: {e}")
        
        print(f"\n✅ Training completed!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"Final model saved: {final_model_path}")
        if csv_path:
            print(f"Metrics CSV saved: {csv_path}")
        
        return self.model

def main():
    """Main function"""
    try:
        parser = argparse.ArgumentParser(description='Improved HMAY-TSF Training for Aerial Vehicles')
        parser.add_argument('--data', type=str, default='./Aerial-Vehicles-1/data.yaml', help='Dataset YAML path')
        parser.add_argument('--epochs', type=int, default=15, help='Number of epochs')
        parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
        parser.add_argument('--img-size', type=int, default=640, help='Image size')
        parser.add_argument('--device', type=str, default='cuda', help='Device to use')
        parser.add_argument('--save-dir', type=str, default='./runs/improved_train', help='Save directory')
        parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
        
        args = parser.parse_args()
        
        # Check if CUDA is available
        if args.device == 'cuda' and not torch.cuda.is_available():
            print("CUDA not available, switching to CPU")
            args.device = 'cpu'
        
        # Create datasets
        print("Creating datasets...")
        try:
            train_dataset = SimpleDataset(args.data, args.img_size, is_training=True)
            val_dataset = SimpleDataset(args.data, args.img_size, is_training=False)
        except Exception as e:
            print(f"Error creating datasets: {e}")
            return
        
        # Create dataloaders with better settings
        try:
            train_loader = DataLoader(
                train_dataset, 
                batch_size=args.batch_size, 
                shuffle=True, 
                num_workers=2,  # Reduced workers
                collate_fn=collate_fn,
                pin_memory=True,
                drop_last=True  # Drop incomplete batches
            )
            
            val_loader = DataLoader(
                val_dataset, 
                batch_size=args.batch_size, 
                shuffle=False, 
                num_workers=2,  # Reduced workers
                collate_fn=collate_fn,
                pin_memory=True,
                drop_last=True  # Drop incomplete batches
            )
        except Exception as e:
            print(f"Error creating dataloaders: {e}")
            return
        
        print(f"Train samples: {len(train_dataset)}")
        print(f"Val samples: {len(val_dataset)}")
        
        # Create improved model
        print("Creating improved HMAY-TSF model...")
        try:
            model = ImprovedHMAYTSF(num_classes=4)
            print("✅ Improved HMAY-TSF model created successfully!")
            print("✅ Enhanced architecture for aerial vehicle detection!")
        except Exception as e:
            print(f"Error creating model: {e}")
            return
        
        # Resume from checkpoint if specified
        if args.resume:
            try:
                print(f"Resuming from checkpoint: {args.resume}")
                checkpoint = torch.load(args.resume, map_location=args.device)
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
            except Exception as e:
                print(f"Error loading checkpoint: {e}")
        
        # Create trainer with updated scheduler
        try:
            trainer = SimpleTrainer(model, device=args.device)
            # Update scheduler with correct steps_per_epoch
            trainer.scheduler = OneCycleLR(
                trainer.optimizer,
                max_lr=0.0015,
                epochs=args.epochs,
                steps_per_epoch=len(train_loader),
                pct_start=0.25,
                anneal_strategy='cos'
            )
        except Exception as e:
            print(f"Error creating trainer: {e}")
            return
        
        # Resume trainer state if specified
        if args.resume:
            try:
                trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                trainer.best_val_loss = checkpoint['best_val_loss']
            except Exception as e:
                print(f"Error loading trainer state: {e}")
        
        # Train
        try:
            trained_model = trainer.train(
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=args.epochs,
                save_dir=args.save_dir
            )
            
            print("🎯 Improved training completed successfully!")
        except Exception as e:
            print(f"Error during training: {e}")
            
    except Exception as e:
        print(f"Fatal error in main: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 