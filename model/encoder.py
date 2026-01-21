"""
PointNet-based encoder for extracting features from point clouds.
Compare with point-cloud-dynamics-model/
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from enum import IntEnum

class ObjectClass(IntEnum):
    # Modifiers
    MODIFIER_DIFFUSE = 0
    MODIFIER_SPECULAR = 1  # Easy to extend later
    
    # Emitters (Start indices after modifiers)
    EMITTER_POINT = 10
    EMITTER_AREA = 11
    EMITTER_SPOT = 12
    EMITTER_SUN = 13

class UniversalPropertyEncoder(nn.Module):
    def __init__(
        self,
        prop_dim: int = 3,         # [R, G, B]
        num_classes: int = 20,     # Size of registry
        embed_dim: int = 512,      # Output dimension
        type_embed_dim: int = 64,  # Reserved for class identity
        use_batch_norm: bool = True
    ):
        super().__init__()
        
        # 1. Discrete Knowledge: What IS this thing?
        self.type_embedding = nn.Embedding(num_classes, type_embed_dim)
        
        # 2. Continuous Knowledge: What color/intensity is it?
        # We output (Total - Type_Dim) so they sum to embed_dim
        prop_out_dim = embed_dim - type_embed_dim
        
        self.prop_mlp = nn.Sequential(
            nn.Linear(prop_dim, prop_out_dim, bias=not use_batch_norm),
            nn.BatchNorm1d(prop_out_dim) if use_batch_norm else nn.Identity(),
            nn.SiLU()
        )
        
        # 3. Fusion: Mix them together
        self.final_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim, bias=not use_batch_norm),
            nn.BatchNorm1d(embed_dim) if use_batch_norm else nn.Identity(),
            nn.SiLU()
        )

    def forward(self, properties: torch.Tensor, class_ids: torch.LongTensor) -> torch.Tensor:
        """
        properties: (B, 3) 
        class_ids: (B,) Integers from ObjectClass enum
        """
        # Embed the type (B, type_embed_dim)
        type_feat = self.type_embedding(class_ids)
        
        # Encode physical properties (B, embed_dim - type_embed_dim)
        prop_feat = self.prop_mlp(properties)
        
        # Concatenate: [TypeInfo | PhysicsInfo]
        combined = torch.cat([prop_feat, type_feat], dim=-1)
        
        return self.final_proj(combined)

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

        # Universal property encoders for objects
        self.property_encoder = UniversalPropertyEncoder(
            prop_dim=3,
            num_classes=20,
            embed_dim=output_dim
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
        properties: torch.Tensor,
        object_class_ids: torch.Tensor,
        normals: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode point cloud features.

        Args:
            surface_pos: sampled positions from object surface (B, N, 3)
            normals: Optional surface normals (B, N, 3)
            properties: diffuse [r,g,b] (B, 3) if modifier; intensity [r,g,b] [type] (B, 4) if emitter
            object_types: (B,) tensor with 0=modifier, 1=emitter
            emitter_types: (B,) tensor with light type indices, only used for emitters # NOTE: nah i don't like this design.....

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

        # Process properties based on object type for each item in batch
        properties_token = self.property_encoder(properties, object_class_ids)

        # Fuse geometry and physical properties
        combined = torch.cat([geometry_token, properties_token], dim=-1)  # (B, backbone_dim + output_dim)
        output = self.fusion_mlp(combined)  # (B, output_dim)

        return output  # (B, output_dim)
