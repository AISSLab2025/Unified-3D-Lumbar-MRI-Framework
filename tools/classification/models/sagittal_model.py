import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np

# =========================================================
# ATTENTION MECHANISMS
# =========================================================
class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        se = self.avg_pool(x).view(b, c)
        se = self.fc(se).view(b, c, 1, 1)
        return x * se


class CBAMBlock(nn.Module):
    """Convolutional Block Attention Module"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        # Channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()
        
        # Spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
    
    def channel_attention(self, x):
        b, c, _, _ = x.size()
        avg = self.avg_pool(x).view(b, c)
        max_pool = self.max_pool(x).view(b, c)
        avg_out = self.fc(avg)
        max_out = self.fc(max_pool)
        out = self.sigmoid(avg_out + max_out)
        return x * out.view(b, c, 1, 1)
    
    def spatial_attention(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return torch.sigmoid(x)
    
    def forward(self, x):
        x = self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x


class SpecialCBAMBlock(nn.Module):
    """Enhanced CBAM with residual connection"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.cbam = CBAMBlock(channels, reduction)
        self.gamma = nn.Parameter(torch.ones(1))
    
    def forward(self, x):
        return x + self.gamma * self.cbam(x)


# =========================================================
# CAM ATTENTION
# =========================================================
class CAMBlock(nn.Module):
    """
    Channel Attention Module only
    Input shape: [B, C, H, W]
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        reduced_channels = max(channels // reduction, 8)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.mlp = nn.Sequential(
            nn.Conv2d(channels, reduced_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, channels, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        attn = self.sigmoid(avg_out + max_out)
        return x * attn


# =========================================================
# MODEL
# =========================================================
class SwinWithCAMUncertainty(nn.Module):
    """
    torchvision Swin-T backbone + CAM attention + optional CBAM/SE
    with uncertainty-guided classification logic
    """
    def __init__(
        self,
        num_classes=3,
        dropout_p=0.45,
        cam_reduction=16,
        attention_mechanism="none",
        attention_stage="before_classification",
        ugr_hidden_dim=256,
        ugr_dropout=0.20
    ):
        super().__init__()

        backbone = models.swin_t(weights=models.Swin_T_Weights.DEFAULT)

        self.features = backbone.features
        self.norm = backbone.norm

        self.cam = CAMBlock(channels=768, reduction=cam_reduction)
        
        self.attention_mechanism = attention_mechanism
        self.attention_stage = attention_stage
        
        if attention_mechanism != "none":
            if attention_mechanism == "cbam":
                self.attention_block = CBAMBlock(channels=768, reduction=16)
            elif attention_mechanism == "se":
                self.attention_block = SEBlock(channels=768, reduction=16)
            elif attention_mechanism == "special_cbam":
                self.attention_block = SpecialCBAMBlock(channels=768, reduction=16)
            else:
                raise ValueError(f"Unsupported attention mechanism: {attention_mechanism}")
        else:
            self.attention_block = None

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=dropout_p)
        
        # Uncertainty-guided classification logic
        self.initial_fc = nn.Linear(768, num_classes)
        self.uncertainty_gate_mlp = nn.Sequential(
            nn.Linear(768 + 1, ugr_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(ugr_dropout),
            nn.Linear(ugr_hidden_dim, 768),
            nn.Sigmoid()
        )
        self.final_fc = nn.Linear(768, num_classes)

    @staticmethod
    def normalized_entropy_from_logits(logits, eps=1e-8):
        probs = torch.softmax(logits, dim=1)
        log_probs = torch.log(probs + eps)
        entropy = -(probs * log_probs).sum(dim=1, keepdim=True)

        num_classes = logits.size(1)
        max_entropy = float(np.log(num_classes))
        entropy = entropy / (max_entropy + eps)
        entropy = torch.clamp(entropy, 0.0, 1.0)
        return entropy

    def forward(self, x):
        x = self.features(x)
        x = self.norm(x)

        x = x.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
        
        # Apply CAM attention
        x = self.cam(x)
        
        # Apply additional attention mechanism if specified
        if self.attention_block is not None and self.attention_stage == "before_classification":
            x = self.attention_block(x)

        x = self.pool(x)
        x = torch.flatten(x, 1)
        features_before = self.dropout(x)
        
        # Uncertainty guidance
        initial_logits = self.initial_fc(features_before)
        uncertainty = self.normalized_entropy_from_logits(initial_logits)

        gate_input = torch.cat([features_before, uncertainty], dim=1)
        gate = self.uncertainty_gate_mlp(gate_input)

        features_after = features_before * (1.0 + uncertainty * gate)
        final_logits = self.final_fc(features_after)

        return {
            "initial_logits": initial_logits,
            "final_logits": final_logits,
            "uncertainty": uncertainty,
            "uncertainty_gate": gate,
            "features_before": features_before,
            "features_after": features_after
        }


def freeze_backbone(model):
    for param in model.features.parameters():
        param.requires_grad = False
    for param in model.norm.parameters():
        param.requires_grad = False


def unfreeze_backbone(model):
    for param in model.features.parameters():
        param.requires_grad = True
    for param in model.norm.parameters():
        param.requires_grad = True
