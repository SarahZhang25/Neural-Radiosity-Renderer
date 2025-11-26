"""
PointNet-based encoder for extracting features from point clouds.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


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
        fusion_hidden_dim: int = 800,
        use_batch_norm: bool = True,
        pooling_type: str = 'hierarchical',
        num_hierarchical_levels: int = 5,
        use_physics_params: bool = True
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.backbone_dim = backbone_dim
        self.pooling_type = pooling_type
        self.use_physics_params = use_physics_params

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

        # Physics path: Encode physics parameters
        if use_physics_params:
            self.physics_encoder = nn.Sequential(
                nn.Linear(3, 128, bias=not use_batch_norm),
                nn.BatchNorm1d(128) if use_batch_norm else nn.Identity(),
                nn.SiLU(),
                nn.Linear(128, output_dim, bias=not use_batch_norm),
                nn.BatchNorm1d(output_dim) if use_batch_norm else nn.Identity(),
                nn.SiLU()
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
        else:
            # Direct projection from backbone to output
            self.geometry_proj = nn.Linear(backbone_dim, output_dim)

    def forward(
        self,
        dist_vec: torch.Tensor,
        velocity: torch.Tensor,
        relative_pos: torch.Tensor,
        normals: Optional[torch.Tensor] = None,
        physics_params: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode point cloud features.

        Args:
            dist_vec: Distance to nearest point on other objects (B, N, 3)
            velocity: Point velocities (B, N, 3)
            relative_pos: Position relative to reference (B, N, 3)
            normals: Optional surface normals (B, N, 3)
            physics_params: Optional physics parameters [mass, friction, restitution] (B, 3)

        Returns:
            Encoded features (B, output_dim)
        """
        B, N, _ = dist_vec.shape

        # Concatenate input features
        features = [dist_vec, velocity, relative_pos]
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

        # Dual-path fusion
        if self.use_physics_params and physics_params is not None:
            # Encode physics
            physics_token = self.physics_encoder(physics_params)  # (B, output_dim)

            # Fuse geometry and physics
            combined = torch.cat([geometry_token, physics_token], dim=-1)  # (B, backbone_dim + output_dim)
            output = self.fusion_mlp(combined)  # (B, output_dim)
        else:
            # Direct projection
            output = self.geometry_proj(geometry_token) if hasattr(self, 'geometry_proj') else geometry_token

        return output  # (B, output_dim)