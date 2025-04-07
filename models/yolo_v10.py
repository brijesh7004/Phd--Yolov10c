import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class CSPBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_repeats):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, out_channels)
        self.conv2 = ConvBlock(out_channels, out_channels)
        self.residual_blocks = nn.ModuleList([
            nn.Sequential(
                ConvBlock(out_channels, out_channels),
                ConvBlock(out_channels, out_channels)
            ) for _ in range(num_repeats)
        ])

    def forward(self, x):
        x = self.conv1(x)
        residual = x
        for block in self.residual_blocks:
            x = block(x) + residual
            residual = x
        return self.conv2(x)

class SPPF(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, out_channels)
        self.pools = nn.ModuleList([
            nn.MaxPool2d(kernel_size=k, stride=1, padding=k//2)
            for k in [5, 9, 13]
        ])
        self.conv2 = ConvBlock(out_channels * 4, out_channels)

    def forward(self, x):
        x = self.conv1(x)
        pooled = [x] + [pool(x) for pool in self.pools]
        return self.conv2(torch.cat(pooled, dim=1))

class YOLOv10Head(nn.Module):
    def __init__(self, in_channels, num_classes, num_anchors=3):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.conv = ConvBlock(in_channels, in_channels)
        self.detect = nn.Conv2d(in_channels, (num_classes + 5) * num_anchors, 1)

    def forward(self, x):
        x = self.conv(x)
        return self.detect(x)

class YOLOv10(nn.Module):
    def __init__(self, variant='n', num_classes=80):
        super().__init__()
        self.num_classes = num_classes
        self.variant = variant.lower()
        
        # Model configuration based on variant
        configs = {
            'n': {'width': 0.25, 'depth': 0.33, 'num_repeats': 1},
            's': {'width': 0.50, 'depth': 0.33, 'num_repeats': 1},
            'm': {'width': 0.75, 'depth': 0.67, 'num_repeats': 2},
            'l': {'width': 1.00, 'depth': 1.00, 'num_repeats': 3},
            'x': {'width': 1.25, 'depth': 1.33, 'num_repeats': 4}
        }
        
        config = configs.get(self.variant, configs['n'])
        base_channels = int(64 * config['width'])
        
        # Backbone
        self.backbone = nn.ModuleList([
            ConvBlock(3, base_channels),
            CSPBlock(base_channels, base_channels * 2, int(3 * config['depth'])),
            CSPBlock(base_channels * 2, base_channels * 4, int(6 * config['depth'])),
            CSPBlock(base_channels * 4, base_channels * 8, int(9 * config['depth'])),
            SPPF(base_channels * 8, base_channels * 8)
        ])
        
        # Neck
        self.neck = nn.ModuleList([
            CSPBlock(base_channels * 8, base_channels * 4, int(3 * config['depth'])),
            CSPBlock(base_channels * 4, base_channels * 2, int(3 * config['depth'])),
            CSPBlock(base_channels * 2, base_channels, int(3 * config['depth']))
        ])
        
        # Detection heads
        self.heads = nn.ModuleList([
            YOLOv10Head(base_channels * 8, num_classes),
            YOLOv10Head(base_channels * 4, num_classes),
            YOLOv10Head(base_channels * 2, num_classes)
        ])

    def forward(self, x):
        features = []
        for i, layer in enumerate(self.backbone):
            x = layer(x)
            if i in [2, 3]:  # Save features for FPN
                features.append(x)
        
        # FPN
        p3 = self.neck[0](features[-1])
        p4 = self.neck[1](F.interpolate(p3, scale_factor=2, mode='nearest'))
        p5 = self.neck[2](F.interpolate(p4, scale_factor=2, mode='nearest'))
        
        # Detection heads
        return [
            self.heads[0](features[-1]),
            self.heads[1](p4),
            self.heads[2](p5)
        ]

    def load_weights(self, weights_path):
        """Load weights from a .pt file"""
        try:
            state_dict = torch.load(weights_path, map_location='cpu')
            
            # Handle different formats
            if isinstance(state_dict, dict):
                if 'model' in state_dict:
                    state_dict = state_dict['model']
                elif 'model_state_dict' in state_dict:
                    state_dict = state_dict['model_state_dict']
            
            # Handle weights from ultralytics
            if all(k.startswith('model.') for k in state_dict.keys()):
                new_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith('model.'):
                        new_key = k.replace('model.', '')
                        new_state_dict[new_key] = v
                state_dict = new_state_dict
            
            # Load weights
            try:
                self.load_state_dict(state_dict)
                print(f"Successfully loaded weights from {weights_path}")
            except Exception as e:
                print(f"Could not load weights with strict matching: {e}")
                print("Attempting to load with non-strict matching...")
                self.load_state_dict(state_dict, strict=False)
                print(f"Loaded weights with non-strict matching from {weights_path}")
        
        except Exception as e:
            print(f"Error loading weights: {e}")
            print(f"Using randomly initialized weights")

def create_model(variant='n', num_classes=80, weights_path=None):
    """Create a YOLOv10 model with the specified variant and weights"""
    model = YOLOv10(variant=variant, num_classes=num_classes)
    
    if weights_path:
        model.load_weights(weights_path)
    
    return model 