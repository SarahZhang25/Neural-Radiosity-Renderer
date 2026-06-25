"""
Transformer-based predictor for radiance prediction.
"""

import torch
import torch.nn as nn
from typing import List, Optional
from einops import rearrange
import math
from model.encodings.nerf_encoding import NeRFEncoding
from model.layers.dpt import DPTHead

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
        pe_type: str = 'nerf',
        pe_num_freqs: int = 8,
        use_dpt_decoder: bool = False,
        dpt_features: Optional[int] = None,
        dpt_out_channels: Optional[List[int]] = None,
        include_alpha: bool = False
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.patch_size = patch_size
        self.pe_type = pe_type

        if self.pe_type == 'nerf':
            self.pos_pe = NeRFEncoding(
                in_dim=3,
                num_frequencies=pe_num_freqs,
                include_input=True
            )
            self.pe_token_proj = nn.Linear(
                self.pos_pe.get_out_dim(),
                hidden_dim
            )
            if norm_type == 'layer_norm':
                self.pos_pe_norm = nn.LayerNorm(hidden_dim)
            elif norm_type == 'rms_norm':
                self.pos_pe_norm = nn.RMSNorm(hidden_dim)
            else:
                self.pos_pe_norm = nn.Identity()

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
        if not use_dpt_decoder:
            if norm_type == 'layer_norm':
                self.out_norm = nn.LayerNorm(hidden_dim)
            elif norm_type == 'rms_norm':
                self.out_norm = nn.RMSNorm(hidden_dim)
            else:
                self.out_norm = nn.Identity()
            # Output projection:  -> 3 RGB channels × patch_size × patch_size
            self.out_proj = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 3 * patch_size * patch_size)
            )

            # Initialize last layer with small weights for stable training
            last_layer = self.out_proj[-1]
            nn.init.xavier_uniform_(last_layer.weight, gain=0.01)
            nn.init.zeros_(last_layer.bias)
        else:
            self.out_dpt = DPTHead(
                in_channels=hidden_dim,
                features=dpt_features,
                out_channels=dpt_out_channels,
                out_dim=4 if include_alpha else 3
            )
            self.out_layers = list(range(num_layers - 4, num_layers)) 
        self.out_proj_act = nn.GELU()  # from: nn.ELU(alpha=1e-3) in RenderFormer
        


    def forward(
        self,
        multi_scale_features: List[torch.Tensor],
        query_view_features: torch.Tensor,
        multi_scale_state_features: Optional[List[torch.Tensor]] = None,
        patch_h: Optional[int] = None,
        patch_w: Optional[int] = None,
        w2c: Optional[torch.Tensor] = None,
        obj_positions: Optional[torch.Tensor] = None,
        ray_positions: Optional[torch.Tensor] = None,
        use_dpt_decoder: Optional[bool] = False,
        tf32_mode=False,
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
            use_dpt_decoder: Whether to use DPT decoder
        Returns:
            Predicted radiances (B, 3, H, W)
        """
        B, N_patches, D = query_view_features.shape
        
        positions_cam_space = None
        if self.pe_type in ['nerf', 'rope'] and w2c is not None and obj_positions is not None:
            # Transform object positions to camera space
            # w2c is (B, 4, 4), obj_positions is (B, N_obj, N_v, 3)
            B, N_obj, N_v, _ = obj_positions.shape
            w2c_R = w2c[:, :3, :3]  # (B, 3, 3)
            w2c_t = w2c[:, :3, 3]   # (B, 3)
            
            # Use centroids representing each object token
            centroids = obj_positions.mean(dim=2)  # (B, N_obj, 3)
            
            # Transform centroids to camera space
            positions_cam_space = torch.bmm(centroids, w2c_R.transpose(1, 2)) + w2c_t.unsqueeze(1)
        
        # Infer patch grid if not provided
        if patch_h is None or patch_w is None:
            patch_grid_size = int(math.sqrt(N_patches))
            patch_h = patch_h or patch_grid_size
            patch_w = patch_w or patch_grid_size

        # Weighted fusion of multi-scale object features
        num_feats = min(3, len(multi_scale_features))
        layer_weights_obj = torch.softmax(self.layer_weights[:num_feats], dim=0)
        obj_features = sum(
            w * feat for w, feat in zip(layer_weights_obj, multi_scale_features[:num_feats])
        )  # (B, N_obj, D)

        if self.pe_type == 'nerf' and positions_cam_space is not None:
            # Generate NeRF positional encoding and add to object features
            encoded_pos = self.pos_pe(positions_cam_space)
            pos_emb = self.pos_pe_norm(self.pe_token_proj(encoded_pos)) # (B, N_obj, D)
            obj_features = obj_features + pos_emb

        # Optionally concatenate state features
        if multi_scale_state_features is not None:
            num_state_feats = min(3, len(multi_scale_state_features))
            layer_weights_state = torch.softmax(self.layer_weights[:num_state_feats], dim=0)
            state_features = sum(
                w * feat for w, feat in zip(layer_weights_state, multi_scale_state_features[:num_state_feats])
            )  # (B, N_state, D)
            scene_features = torch.cat([obj_features, state_features], dim=1)
        else:
            scene_features = obj_features

        # Cross-attention: rays query scene
        if use_dpt_decoder:
            with torch.autocast(device_type="cuda", dtype=torch.float32 if tf32_mode else torch.bfloat16):
                out_features = []
                x = query_view_features       # (B, N_patches, D)
                # manually iterate through layers to extract intermediate features for DPT decoding
                for i, layer in enumerate(self.transformer.layers): 
                    x = layer(
                        tgt=x,
                        memory=scene_features, # (B, N_obj+N_state, D)
                    )
                    if i in self.out_layers:
                        out_features.append([x])
            decoded_img = self.out_dpt(out_features, patch_h, patch_w, patch_size=self.patch_size)
            return self.out_proj_act(decoded_img)
        else:
            ray_features = self.transformer(
                tgt=query_view_features,      # (B, N_patches, D)
                memory=scene_features,        # (B, N_obj+N_state, D)
            )  # (B, N_patches, D)
            ray_features = self.out_norm(ray_features)

            # Decode to RGB per patch
            patches = self.out_proj(ray_features)  # (B, N_patches, 3*P*P)
            patches = self.out_proj_act(patches)
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
