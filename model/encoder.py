"""
PointNet-based encoder for extracting features from point clouds.
Compare with point-cloud-dynamics-model/
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

class ObjectPropertyEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 3,
        hidden_dim: int = 128,
        output_dim: int = 512,
        object_type: str = "modifier", # or 'emitter'
        emitter_type_dim: int = 16,
        use_batch_norm: bool = True
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.object_type = object_type
        self.emitter_type_dim = emitter_type_dim

        # if use_physics_params:
        if object_type == 'emitter':
            self.emitter_type_embed = nn.Embedding(
                num_embeddings=4, # Support 4 light types for now? Point, Area, Spot, Directional/Sun
                embedding_dim=emitter_type_dim
            )
            self.property_layer = nn.Linear(3, hidden_dim - emitter_type_dim, bias=not use_batch_norm)
        elif object_type == 'modifier':
            self.property_layer = nn.Linear(3, hidden_dim, bias=not use_batch_norm)

        self.encoder = nn.Sequential(
            nn.BatchNorm1d(hidden_dim) if use_batch_norm else nn.Identity(),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim, bias=not use_batch_norm),
            nn.BatchNorm1d(output_dim) if use_batch_norm else nn.Identity(),
            nn.SiLU()
        )
    
    def forward(
        self,
        properties: torch.Tensor, # (Batch, N, 3) -> e.g. [r, g, b]
        v_type: Optional[torch.LongTensor] = None, # (Batch, N) -> e.g., [0, 1, 0, 3...]
    ) -> torch.Tensor:
        """
        Args:
            properties:
                for modifers, represents diffuse value (R,G,B)
                for emitters, represents radiance intensity (R,G,B)
            v_type: 

        """
        emb = self.property_layer(properties)

        # For emitter, concat type embedding with property layer  
        # before passing through rest of encoder
        if self.object_type == 'emitter':
            type_vec = self.type_embed(v_type)
            emb = torch.cat([emb, type_vec], dim=-1)

        emb = self.encoder(emb) # (B, N, output_dim)
        return emb


class PointNetEncoder(nn.Module):
    """
    PointNet encoder with dual-path architecture:
    - Geometry path: Process point cloud features
    - Physics path: Process physics parameters
    - Fusion: Combine both paths

    Output dimension is always 512D for compatibility with decoder.
    """

    def __init__(
        self,
        input_dim: int = 12,
        hidden_dims: list = [2048, 2048],
        output_dim: int = 512,
        backbone_dim: int = 800,
        # fusion_hidden_dim: int = 800,
        use_batch_norm: bool = True,
        pooling_type: str = 'hierarchical', # or 'max'
        num_hierarchical_levels: int = 5,
        property_encoder_hidden_dim: int = 128,
        emitter_type_dim: int = 16
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.backbone_dim = backbone_dim
        self.pooling_type = pooling_type
        
        self.object_type = object_type
        self.emitter_type_dim = emitter_type_dim
        self.property_encoder_hidden_dim = property_encoder_hidden_dim

        # Geometry path: PointNet backbone
        layers = []
        in_dim = input_dim
        for hdim in hidden_dims:
            layers.append(nn.Conv1d(in_dim, hdim, 1, bias=not use_batch_norm))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hdim))
            layers.append(nn.SiLU())
            in_dim = hdim

        # Final projection to backbone_dim
        layers.append(nn.Conv1d(in_dim, backbone_dim, 1, bias=not use_batch_norm))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(backbone_dim))
        layers.append(nn.SiLU())

        self.point_net = nn.Sequential(*layers)

        # Hierarchical pooling layers (if enabled)
        if pooling_type == 'hierarchical':
            self.hierarchical_layers = nn.ModuleList([
                nn.Conv1d(backbone_dim, backbone_dim, 1)
                for _ in range(num_hierarchical_levels)
            ])
            # Project concatenated multi-scale features back to backbone_dim
            # Each level contributes backbone_dim*2 features (max+mean)
            total_feat_dim = backbone_dim * 2 * (1 + num_hierarchical_levels)
            self.hierarchical_proj = nn.Linear(total_feat_dim, backbone_dim)

        # Separate property encoders for each object type
        self.modifier_property_encoder = ObjectPropertyEncoder(
            input_dim=3,
            hidden_dim=property_encoder_hidden_dim,
            output_dim=output_dim,
            object_type='modifier',
            use_batch_norm=use_batch_norm
        )
        
        self.emitter_property_encoder = ObjectPropertyEncoder(
            input_dim=3,
            hidden_dim=property_encoder_hidden_dim,
            output_dim=output_dim,
            object_type='emitter',
            emitter_type_dim=emitter_type_dim,
            use_batch_norm=use_batch_norm
        )

        # Fusion MLP: Combine geometry and physics
        fusion_input_dim = backbone_dim + output_dim
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_input_dim, output_dim, bias=not use_batch_norm),
            nn.BatchNorm1d(output_dim) if use_batch_norm else nn.Identity(),
            nn.SiLU(),
            nn.Linear(output_dim, output_dim, bias=not use_batch_norm),
            nn.BatchNorm1d(output_dim) if use_batch_norm else nn.Identity(),
            nn.SiLU()
        )
        # else:
        #     # Direct projection from backbone to output
        #     self.geometry_proj = nn.Linear(backbone_dim, output_dim)

    def forward(
        self,
        surface_pos: torch.Tensor,
        normals: Optional[torch.Tensor] = None,
        properties: Optional[torch.Tensor] = None,
        object_type: str = "modifier", # or 'emitter'
        emitter_types: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode point cloud features.

        Args:
            surface_pos: sampled positions from object surface (B, N, 3)
            normals: Optional surface normals (B, N, 3)
            properties: diffuse [r,g,b] (B, 3) if modifier; intensity [r,g,b] [type] (B, 4) if emitter

        Returns:
            Encoded features (B, output_dim)
        """
        B, N, _ = surface_pos.shape

        # Concatenate input features
        features = [surface_pos]
        if normals is not None:
            features.append(normals)

        x = torch.cat(features, dim=-1)  # (B, N, input_dim)
        x = x.transpose(1, 2)  # (B, input_dim, N)

        # PointNet backbone
        per_point_features = self.point_net(x)  # (B, backbone_dim, N)

        # Pooling
        if self.pooling_type == 'max':
            geometry_token = torch.max(per_point_features, dim=2)[0]  # (B, backbone_dim)

        elif self.pooling_type == 'hierarchical':
            # Multi-scale pooling
            global_max = torch.max(per_point_features, dim=2)[0]
            global_mean = torch.mean(per_point_features, dim=2)
            global_feat = torch.cat([global_max, global_mean], dim=1)

            local_features = [global_feat]

            for level, layer in enumerate(self.hierarchical_layers):
                stride = 2 ** (level + 1)
                sampled_feat = per_point_features[:, :, ::stride]

                if sampled_feat.shape[2] < 8:
                    sampled_feat = per_point_features

                local_feat = layer(sampled_feat)
                local_max = torch.max(local_feat, dim=2)[0]
                local_mean = torch.mean(local_feat, dim=2)
                local_combined = torch.cat([local_max, local_mean], dim=1)
                local_features.append(local_combined)

            all_features = torch.cat(local_features, dim=1)
            geometry_token = self.hierarchical_proj(all_features)  # (B, backbone_dim)

        else:
            raise ValueError(f"Unknown pooling type: {self.pooling_type}")

        # Fuse geometry and physical properties      
        # Branch on object type
        if object_type == 'modifier':
            properties_token = self.modifier_property_encoder(properties)
        elif object_type == 'emitter':
            properties_token = self.emitter_property_encoder(properties, emitter_types)
        else:
            raise ValueError(f"Unknown object_type: {object_type}")
        combined = torch.cat([geometry_token, properties_token], dim=-1)  # (B, backbone_dim + output_dim)
        output = self.fusion_mlp(combined)  # (B, output_dim)

        return output  # (B, output_dim)
