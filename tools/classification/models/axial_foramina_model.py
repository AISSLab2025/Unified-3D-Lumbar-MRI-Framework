import torch
import torch.nn as nn
from torchvision import models
import numpy as np

# =========================================================
# ATTENTION MECHANISMS
# =========================================================
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        se = self.avg_pool(x).view(b, c)
        se = self.fc(se).view(b, c, 1, 1)
        return x * se


class CBAMBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 8)

        # channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

        # spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def channel_attention(self, x):
        b, c, _, _ = x.size()
        avg = self.avg_pool(x).view(b, c)
        mx = self.max_pool(x).view(b, c)
        avg_out = self.fc(avg)
        max_out = self.fc(mx)
        out = self.sigmoid(avg_out + max_out)
        return x * out.view(b, c, 1, 1)

    def spatial_attention(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attn = torch.cat([avg_out, max_out], dim=1)
        attn = self.conv(attn)
        return torch.sigmoid(attn)

    def forward(self, x):
        x = self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attn = torch.cat([avg_out, max_out], dim=1)
        attn = self.sigmoid(self.conv(attn))
        return x * attn


# =========================================================
# UNCERTAINTY-GUIDED MODEL
# =========================================================
class SwinWithAttentionUncertaintyGuided(nn.Module):
    """
    Pipeline:
        Swin features (with stage-specific attention)
        -> norm
        -> GAP
        -> initial classifier
        -> uncertainty from initial logits
        -> uncertainty-guided feature reweighting
        -> final classifier

    Returns dict:
        {
            "initial_logits": ...,
            "final_logits": ...,
            "uncertainty": ...,
            "uncertainty_gate": ...,
            "features_before": ...,
            "features_after": ...
        }
    """
    def __init__(
        self,
        num_classes=3,
        dropout_p=0.45,
        attention_mechanism="none",
        attention_stage="before_classification",
        reduction=16,
        ugr_hidden_dim=256,
        ugr_dropout=0.20
    ):
        super().__init__()

        backbone = models.swin_t(weights=models.Swin_T_Weights.DEFAULT)
        self.features = backbone.features
        self.norm = backbone.norm

        self.attention_mechanism = attention_mechanism
        self.attention_stage = attention_stage

        # Swin-T stage channel sizes for torchvision backbone
        self.stage_channels = {
            "stage_1_2": 192,
            "stage_2_3": 384,
            "stage_3_4": 768,
            "before_classification": 768,
        }

        if attention_mechanism != "none":
            if attention_stage not in self.stage_channels:
                raise ValueError(f"Unsupported attention_stage: {attention_stage}")

            attn_channels = self.stage_channels[attention_stage]
            self.attention_block = self._make_attention_block(
                attention_mechanism,
                attn_channels,
                reduction
            )
        else:
            self.attention_block = None

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=dropout_p)

        feature_dim = 768

        # initial classifier
        self.initial_fc = nn.Linear(feature_dim, num_classes)

        # uncertainty-guided reweighting
        # input = pooled features + uncertainty scalar
        self.uncertainty_gate_mlp = nn.Sequential(
            nn.Linear(feature_dim + 1, ugr_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(ugr_dropout),
            nn.Linear(ugr_hidden_dim, feature_dim),
            nn.Sigmoid()
        )

        # final classifier
        self.final_fc = nn.Linear(feature_dim, num_classes)

    def _make_attention_block(self, mechanism, channels, reduction):
        if mechanism == "cbam":
            return CBAMBlock(channels=channels, reduction=reduction)
        elif mechanism == "se":
            return SEBlock(channels=channels, reduction=reduction)
        elif mechanism == "spatial":
            return SpatialAttention()
        else:
            raise ValueError(f"Unsupported attention_mechanism: {mechanism}")

    def _apply_attention_if_needed(self, x, stage_name):
        """
        x is expected in [B, H, W, C] from Swin features.
        attention blocks expect [B, C, H, W].
        """
        if self.attention_block is None or self.attention_stage != stage_name:
            return x

        x = x.permute(0, 3, 1, 2)      # [B, H, W, C] -> [B, C, H, W]
        x = self.attention_block(x)
        x = x.permute(0, 2, 3, 1)      # [B, C, H, W] -> [B, H, W, C]
        return x

    @staticmethod
    def normalized_entropy_from_logits(logits, eps=1e-8):
        """
        uncertainty in [0,1]
        0 = confident, 1 = maximally uncertain
        """
        probs = torch.softmax(logits, dim=1)
        log_probs = torch.log(probs + eps)
        entropy = -(probs * log_probs).sum(dim=1, keepdim=True)  # [B,1]

        num_classes = logits.size(1)
        max_entropy = float(np.log(num_classes))
        entropy = entropy / (max_entropy + eps)
        entropy = torch.clamp(entropy, 0.0, 1.0)
        return entropy

    def forward(self, x):
        # Swin feature extraction step by step
        for i, layer in enumerate(self.features):
            x = layer(x)

            # These indices correspond to the stage boundaries in torchvision Swin-T
            if i == 2:
                x = self._apply_attention_if_needed(x, "stage_1_2")
            elif i == 4:
                x = self._apply_attention_if_needed(x, "stage_2_3")
            elif i == 6:
                x = self._apply_attention_if_needed(x, "stage_3_4")

        x = self.norm(x)

        # Final attention after full backbone
        x = self._apply_attention_if_needed(x, "before_classification")

        x = x.permute(0, 3, 1, 2)   # [B, H, W, C] -> [B, C, H, W]

        # Global pooling
        x = self.pool(x)              # [B, C, 1, 1]
        x = torch.flatten(x, 1)       # [B, C]
        features_before = self.dropout(x)

        # Initial classifier
        initial_logits = self.initial_fc(features_before)

        # Uncertainty from initial logits
        uncertainty = self.normalized_entropy_from_logits(initial_logits)   # [B,1]

        # Uncertainty-guided reweighting
        gate_input = torch.cat([features_before, uncertainty], dim=1)       # [B, C+1]
        gate = self.uncertainty_gate_mlp(gate_input)                        # [B, C], in [0,1]

        # Reweighting rule:
        # when uncertainty is high, the model can alter features more strongly
        # features_after = features_before * (1 + uncertainty * gate)
        features_after = features_before * (1.0 + uncertainty * gate)

        # Final classifier
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
