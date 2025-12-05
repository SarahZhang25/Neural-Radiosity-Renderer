"""
Transformer-based predictor for radiance prediction.
"""

import torch
import torch.nn as nn
from typing import List, Optional
from einops import rearrange
import math


class RadiancePredictor(nn.Module):
    """
    Transformer predictor that uses cross-attention between rays and scene features.
    """

    def __init__(
        self,
        hidden_dim: int = 512,
        patch_size: int = 16,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
        activation: str = 'gelu',
        norm_type: str = 'layer_norm',
        include_self_attn: bool = True,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.patch_size = patch_size

        # Feature pooling across encoder layers
        self.layer_weights = nn.Parameter(torch.ones(3))

        # PyTorch TransformerDecoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=True  # Pre-norm architecture
        )
        
        self.transformer = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers
        )

        # Output projection: hidden_dim -> 3 RGB channels × patch_size²
        self.out_proj = nn.Linear(hidden_dim, 3 * patch_size * patch_size)
        
        # Initialize with small weights for stable training
        nn.init.xavier_uniform_(self.out_proj.weight, gain=0.01)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        multi_scale_features: List[torch.Tensor],
        query_view_features: torch.Tensor,
        multi_scale_state_features: Optional[List[torch.Tensor]] = None,
        patch_h: Optional[int] = None,
        patch_w: Optional[int] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Predict radiances using cross-attention between rays and scene.

        Args:
            multi_scale_features: List of object features [(B, N_obj, D), ...]
            query_view_features: Ray features from encoder (B, N_PATCHES, D)
            multi_scale_state_features: List of state features [(B, N_state, D), ...]
            patch_h: Number of patches in height
            patch_w: Number of patches in width

        Returns:
            Predicted radiances (B, 3, H, W)
        """
        B, N_patches, D = query_view_features.shape
        
        # Infer patch grid if not provided
        if patch_h is None or patch_w is None:
            patch_grid_size = int(math.sqrt(N_patches))
            patch_h = patch_h or patch_grid_size
            patch_w = patch_w or patch_grid_size

        # Weighted fusion of multi-scale object features
        layer_weights = torch.softmax(self.layer_weights, dim=0)
        obj_features = sum(
            w * feat for w, feat in zip(layer_weights, multi_scale_features[:3])
        )  # (B, N_obj, D)

        # Optionally concatenate state features
        if multi_scale_state_features is not None:
            state_features = sum(
                w * feat for w, feat in zip(layer_weights, multi_scale_state_features[:3])
            )  # (B, N_state, D)
            scene_features = torch.cat([obj_features, state_features], dim=1)
        else:
            scene_features = obj_features

        # Cross-attention: rays query scene
        ray_features = self.transformer(
            tgt=query_view_features,      # (B, N_patches, D)
            memory=scene_features,        # (B, N_obj+N_state, D)
        )  # (B, N_patches, D)

        # Decode to RGB per patch
        patches = self.out_proj(ray_features)  # (B, N_patches, 3*P*P)

        # Reshape to image format
        radiances = rearrange(
            patches,
            'b (h w) (c p1 p2) -> b c (h p1) (w p2)',
            h=patch_h,
            w=patch_w,
            p1=self.patch_size,
            p2=self.patch_size,
            c=3
        )  # (B, 3, H, W)

        return radiances
    
if __name__ == "__main__":
    # Quick sanity check
    B, N_obj, N_state, D = 2, 100, 10, 512
    N_patches = 16 * 16  # 256 patches

    query = torch.randn(B, N_patches, D)
    obj_feats = [torch.randn(B, N_obj, D) for _ in range(3)]
    state_feats = [torch.randn(B, N_state, D) for _ in range(3)]

    predictor = RadiancePredictor(hidden_dim=D, patch_size=16)
    out = predictor(obj_feats, query, state_feats, patch_h=16, patch_w=16)

    assert out.shape == (B, 3, 256, 256)  # 16 patches × 16 pixels = 256px