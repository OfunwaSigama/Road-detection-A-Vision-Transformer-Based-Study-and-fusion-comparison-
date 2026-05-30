import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation


class RGBBaseline(nn.Module):
    def __init__(self, num_classes=2, model_name="nvidia/segformer-b2-finetuned-ade-512-512"):
        super().__init__()
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            model_name, num_labels=num_classes, ignore_mismatched_sizes=True
        )

    def forward(self, x):
        return self.model(pixel_values=x).logits


class EarlyFusion(nn.Module):
    def __init__(self, num_classes=2, model_name="nvidia/segformer-b2-finetuned-ade-512-512"):
        super().__init__()
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            model_name, num_labels=num_classes, ignore_mismatched_sizes=True
        )
        
        old_conv = self.model.segformer.encoder.patch_embeddings[0].proj
        new_conv = nn.Conv2d(4, old_conv.out_channels,
                             kernel_size=old_conv.kernel_size,
                             stride=old_conv.stride,
                             padding=old_conv.padding)
        with torch.no_grad():
            new_conv.weight[:, :3] = old_conv.weight
            new_conv.weight[:, 3] = old_conv.weight.mean(dim=1)
            new_conv.bias = old_conv.bias
        self.model.segformer.encoder.patch_embeddings[0].proj = new_conv

    def forward(self, x):
        return self.model(pixel_values=x).logits


class LateFusion(nn.Module):
    def __init__(self, num_classes=2, model_name="nvidia/segformer-b2-finetuned-ade-512-512"):
        super().__init__()
        self.rgb_encoder = SegformerForSemanticSegmentation.from_pretrained(
            model_name, num_labels=num_classes, ignore_mismatched_sizes=True
        )
        self.depth_encoder = SegformerForSemanticSegmentation.from_pretrained(
            model_name, num_labels=num_classes, ignore_mismatched_sizes=True
        )
       
        old_conv = self.depth_encoder.segformer.encoder.patch_embeddings[0].proj
        new_conv = nn.Conv2d(1, old_conv.out_channels,
                             kernel_size=old_conv.kernel_size,
                             stride=old_conv.stride,
                             padding=old_conv.padding)
        with torch.no_grad():
            new_conv.weight.data = old_conv.weight.mean(dim=1, keepdim=True)
            new_conv.bias.data = old_conv.bias
        self.depth_encoder.segformer.encoder.patch_embeddings[0].proj = new_conv

        self.fusion_head = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(16, num_classes, kernel_size=1)
        )

    def forward(self, rgb, depth):
        rgb_logits = self.rgb_encoder(pixel_values=rgb).logits
        depth_logits = self.depth_encoder(pixel_values=depth).logits
        combined = torch.cat([rgb_logits, depth_logits], dim=1)
        return self.fusion_head(combined)


class CrossModalAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.query_conv = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, rgb_feat, depth_feat):
        B, C, H, W = rgb_feat.size()
        proj_query = self.query_conv(rgb_feat).view(B, -1, H * W).permute(0, 2, 1)
        proj_key = self.key_conv(depth_feat).view(B, -1, H * W)
        energy = torch.bmm(proj_query, proj_key)
        attention = F.softmax(energy, dim=-1)
        proj_value = self.value_conv(depth_feat).view(B, -1, H * W)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1)).view(B, C, H, W)
        return self.gamma * out + rgb_feat


class CrossAttentionFusion(nn.Module):
    def __init__(self, num_classes=2, model_name="nvidia/segformer-b2-finetuned-ade-512-512"):
        super().__init__()
        self.rgb_encoder = SegformerForSemanticSegmentation.from_pretrained(
            model_name, num_labels=num_classes, ignore_mismatched_sizes=True
        )
        self.depth_encoder = SegformerForSemanticSegmentation.from_pretrained(
            model_name, num_labels=num_classes, ignore_mismatched_sizes=True
        )
        
        old_conv = self.depth_encoder.segformer.encoder.patch_embeddings[0].proj
        new_conv = nn.Conv2d(1, old_conv.out_channels,
                             kernel_size=old_conv.kernel_size,
                             stride=old_conv.stride,
                             padding=old_conv.padding)
        with torch.no_grad():
            new_conv.weight.data = old_conv.weight.mean(dim=1, keepdim=True)
            new_conv.bias.data = old_conv.bias
        self.depth_encoder.segformer.encoder.patch_embeddings[0].proj = new_conv

        self.cross_attn = CrossModalAttention(channels=512) 
        self.fusion_head = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1), nn.ReLU(),
            nn.Conv2d(512, 256, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )

    def forward(self, rgb, depth):
        rgb_feat = self.rgb_encoder(pixel_values=rgb, output_hidden_states=True).hidden_states[-1]
        depth_feat = self.depth_encoder(pixel_values=depth, output_hidden_states=True).hidden_states[-1]
        fused = self.cross_attn(rgb_feat, depth_feat)
        combined = torch.cat([rgb_feat, fused], dim=1)
        return self.fusion_head(combined)
