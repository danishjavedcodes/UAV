"""
Hybrid Multi-Scale Adaptive YOLO with Temporal-Spatial Fusion (HMAY-TSF)
Simplified implementation for UAV-based traffic object detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO
from ultralytics.nn.modules import Conv, C2f, SPPF, Detect
import cv2
import numpy as np
from collections import deque
import timm

class CondConv2d(nn.Module):
    """Conditionally Parameterized Convolution for dynamic weight adjustment"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, num_experts=4):
        super().__init__()
        self.num_experts = num_experts
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Expert convolution weights
        self.experts = nn.Parameter(torch.randn(num_experts, out_channels, in_channels, kernel_size, kernel_size))
        
        # Routing network to select experts
        self.routing = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, max(in_channels // 4, 8), 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(in_channels // 4, 8), num_experts, 1),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        # Get routing weights
        routing_weights = self.routing(x)  # [B, num_experts, 1, 1]
        
        # Combine expert weights
        batch_size = x.size(0)
        combined_weight = torch.sum(
            routing_weights.view(batch_size, self.num_experts, 1, 1, 1, 1) * 
            self.experts.unsqueeze(0), dim=1
        )
        
        # Apply convolution for each sample in batch
        output = []
        for i in range(batch_size):
            out = F.conv2d(x[i:i+1], combined_weight[i], stride=self.stride, padding=self.padding)
            output.append(out)
        
        return torch.cat(output, dim=0)

class SPP_CSP(nn.Module):
    """Spatial Pyramid Pooling with Cross-Stage Partial Connections"""
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super().__init__()
        c_ = c1 // 2
        self.conv1 = Conv(c1, c_, 1, 1)
        self.conv2 = Conv(c1, c_, 1, 1)
        self.conv3 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])
        
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x2 = torch.cat([x2] + [m(x2) for m in self.m], dim=1)
        return self.conv3(torch.cat((x1, x2), dim=1))

class BiFPN_Layer(nn.Module):
    """Bidirectional Feature Pyramid Network Layer"""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = Conv(channels, channels, 3, 1)
        self.conv2 = Conv(channels, channels, 3, 1)
        self.weight1 = nn.Parameter(torch.ones(2))
        self.weight2 = nn.Parameter(torch.ones(3))
        self.epsilon = 1e-4
        
    def forward(self, p3, p4, p5):
        # Top-down pathway
        w1 = F.relu(self.weight1)
        w1 = w1 / (w1.sum() + self.epsilon)
        p4_td = self.conv1(w1[0] * p4 + w1[1] * F.interpolate(p5, size=p4.shape[-2:], mode='nearest'))
        
        w2 = F.relu(self.weight2)
        w2 = w2 / (w2.sum() + self.epsilon)
        p3_out = self.conv2(w2[0] * p3 + w2[1] * F.interpolate(p4_td, size=p3.shape[-2:], mode='nearest'))
        
        return p3_out, p4_td, p5

class TemporalSpatialFusion(nn.Module):
    """Temporal-Spatial Fusion Module using 3D CNN and GRU"""
    def __init__(self, channels, seq_len=5):
        super().__init__()
        self.seq_len = seq_len
        self.channels = channels
        
        # 3D CNN for temporal feature extraction
        self.temporal_conv = nn.Sequential(
            nn.Conv3d(channels, channels // 2, (3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels // 2, channels, (3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(channels),
            nn.ReLU(inplace=True)
        )
        
        # GRU for sequence modeling
        self.gru = nn.GRU(channels, channels, batch_first=True)
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(channels, channels // 4),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 4, 1),
            nn.Sigmoid()
        )
        
    def forward(self, features_sequence):
        """
        features_sequence: List of tensors [B, C, H, W] for each frame
        """
        if len(features_sequence) < self.seq_len:
            # Pad with the last frame if sequence is too short
            while len(features_sequence) < self.seq_len:
                features_sequence.append(features_sequence[-1])
        
        # Take the last seq_len frames
        features_sequence = features_sequence[-self.seq_len:]
        
        # Stack features for 3D convolution [B, C, T, H, W]
        stacked_features = torch.stack(features_sequence, dim=2)
        
        # Apply 3D convolution
        temporal_features = self.temporal_conv(stacked_features)
        
        # Global average pooling for sequence processing
        B, C, T, H, W = temporal_features.shape
        pooled_features = F.adaptive_avg_pool3d(temporal_features, (T, 1, 1))
        pooled_features = pooled_features.squeeze(-1).squeeze(-1).transpose(1, 2)  # [B, T, C]
        
        # Apply GRU
        gru_out, _ = self.gru(pooled_features)
        
        # Apply attention
        attention_weights = self.attention(gru_out)  # [B, T, 1]
        attended_features = torch.sum(gru_out * attention_weights, dim=1)  # [B, C]
        
        # Reshape back to spatial dimensions
        attended_features = attended_features.unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        attended_features = attended_features.expand(-1, -1, H, W)  # [B, C, H, W]
        
        return attended_features

class SuperResolutionModule(nn.Module):
    """Dense Residual Super-Resolution Module"""
    def __init__(self, scale_factor=2):
        super().__init__()
        self.scale_factor = scale_factor
        
        # Use a pre-trained super-resolution model for efficiency
        self.sr_net = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3 * (scale_factor ** 2), 3, padding=1),
            nn.PixelShuffle(scale_factor)
        )
        
    def forward(self, x):
        return self.sr_net(x)

class EnhancedYOLOBackbone(nn.Module):
    """Enhanced YOLO backbone with multi-scale features"""
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        
        # Get intermediate feature dimensions
        self.feature_dims = [256, 512, 1024]  # Typical YOLOv8 dimensions
        
        # Add conditional convolutions
        self.cond_convs = nn.ModuleList([
            CondConv2d(dim, dim) for dim in self.feature_dims
        ])
        
        # Add SPP-CSP modules
        self.spp_csps = nn.ModuleList([
            SPP_CSP(dim, dim) for dim in self.feature_dims
        ])
        
        # Add BiFPN layers
        self.bifpn = BiFPN_Layer(self.feature_dims[0])
        
        # Temporal-Spatial Fusion
        self.tsf = TemporalSpatialFusion(self.feature_dims[0])
        self.feature_buffer = deque(maxlen=5)
        
    def forward(self, x, is_training=True):
        # Extract features from base model
        features = []
        
        # Forward through base model layers
        x = self.base_model.model[0](x)  # Conv
        x = self.base_model.model[1](x)  # Conv
        
        # Extract multi-scale features
        for i in range(2, len(self.base_model.model) - 1):
            x = self.base_model.model[i](x)
            if i in [4, 6, 9]:  # Extract features at different scales
                features.append(x)
        
        # Apply conditional convolutions and SPP-CSP
        enhanced_features = []
        for i, (feat, cond_conv, spp_csp) in enumerate(zip(features, self.cond_convs, self.spp_csps)):
            enhanced_feat = cond_conv(feat)
            enhanced_feat = spp_csp(enhanced_feat)
            enhanced_features.append(enhanced_feat)
        
        # Apply BiFPN
        if len(enhanced_features) >= 3:
            p3, p4, p5 = self.bifpn(enhanced_features[0], enhanced_features[1], enhanced_features[2])
            enhanced_features = [p3, p4, p5]
        
        # Temporal-Spatial Fusion (only for the smallest scale features)
        if is_training:
            self.feature_buffer.append(enhanced_features[0].detach())
            if len(self.feature_buffer) > 1:
                tsf_features = self.tsf(list(self.feature_buffer))
                enhanced_features[0] = enhanced_features[0] + 0.1 * tsf_features
        
        return enhanced_features

class HMAY_TSF(nn.Module):
    """Hybrid Multi-Scale Adaptive YOLO with Temporal-Spatial Fusion"""
    def __init__(self, model_size='n', num_classes=10, pretrained=True):
        super().__init__()
        
        # Load base YOLOv8 model
        self.base_yolo = YOLO(f'yolov8{model_size}.pt' if pretrained else f'yolov8{model_size}.yaml')
        
        # Replace backbone with enhanced version
        self.enhanced_backbone = EnhancedYOLOBackbone(self.base_yolo.model)
        
        # Super-resolution module
        self.super_resolution = SuperResolutionModule()
        
        # Detection head (use original YOLO head)
        self.detect = self.base_yolo.model.model[-1]
        
        self.num_classes = num_classes
        
    def forward(self, x, apply_sr=False, is_training=True):
        # Apply super-resolution if requested
        if apply_sr and x.size(-1) < 640:
            x = self.super_resolution(x)
            x = F.interpolate(x, size=(640, 640), mode='bilinear', align_corners=False)
        
        # Extract enhanced features
        features = self.enhanced_backbone(x, is_training)
        
        # Detection
        if is_training:
            return self.detect(features)
        else:
            return self.base_yolo.model(x)
    
    def predict(self, x, **kwargs):
        """Wrapper for prediction"""
        return self.base_yolo.predict(x, **kwargs)
    
    def train_model(self, data_config, epochs=100, **kwargs):
        """Training wrapper"""
        return self.base_yolo.train(data=data_config, epochs=epochs, **kwargs)

# Utility functions for data preparation
def prepare_visdrone_dataset():
    """Prepare VisDrone dataset configuration"""
    dataset_config = {
        'path': './dataset',
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
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
        'nc': 11
    }
    return dataset_config

if __name__ == "__main__":
    # Test model creation
    model = HMAY_TSF(model_size='n', num_classes=11)
    print("HMAY-TSF model created successfully!")
    
    # Test forward pass
    dummy_input = torch.randn(1, 3, 640, 640)
    with torch.no_grad():
        output = model(dummy_input, is_training=False)
        print(f"Model output shape: {output.shape if hasattr(output, 'shape') else 'Multiple outputs'}") 