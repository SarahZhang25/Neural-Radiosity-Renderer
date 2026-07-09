"""
Provides PointNet-based encoder and LitePT-based encoder classes for extracting features from point clouds.
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
        backbone_dim: int = 768,
        use_batch_norm: bool = True,
        pooling_type: str = 'hierarchical', # 'local_features', 'max' 
        num_hierarchical_levels: int = 5,
        use_local_patches: bool = False,
        num_centroids: int = 16,   # K centroids    
        k_neighbors: int = 128,     # number of neighbors to group
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.backbone_dim = backbone_dim
        self.pooling_type = pooling_type
        self.use_local_patches = use_local_patches
        self.num_centroids = num_centroids
        self.k_neighbors = k_neighbors

        # TODO: If using local_features, assert num_centroids * k_neighbors == # sampled points
        # default: 16 centroids * 128 neighbors = 2048 sampled points, which matches num_points_per_object in dataset

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

        # Set Abstraction projection (used when use_local_patches is True)
        if self.use_local_patches:
            self.sa_proj = nn.Conv1d(backbone_dim, backbone_dim, 1, bias=not use_batch_norm)
            if use_batch_norm:
                self.sa_bn = nn.BatchNorm1d(backbone_dim)
            self.sa_silu = nn.SiLU()

        # Output projection
        self.out_proj = nn.Sequential(
            nn.Linear(backbone_dim, output_dim, bias=not use_batch_norm),
            nn.BatchNorm1d(output_dim) if use_batch_norm else nn.Identity(),
            nn.SiLU()
        )

    def farthest_point_sample(self, xyz: torch.Tensor, npoint: int) -> torch.Tensor:
        """
        Input:
            xyz: pointcloud data, [B, N, 3]
            npoint: number of samples
        Return:
            centroids: sampled pointcloud index, [B, npoint]
        """
        device = xyz.device
        B, N, C = xyz.shape
        centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
        distance = torch.ones(B, N, device=device) * 1e10
        farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
        batch_indices = torch.arange(B, dtype=torch.long, device=device)
        
        for i in range(npoint):
            centroids[:, i] = farthest
            centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
            dist = torch.sum((xyz - centroid) ** 2, -1)
            mask = dist < distance
            distance[mask] = dist[mask]
            farthest = torch.max(distance, -1)[1]
        return centroids

    def index_points(self, points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """
        Input:
            points: input points data, [B, N, C]
            idx: sample index data, [B, S]
        Return:
            new_points: indexed points data, [B, S, C]
        """
        device = points.device
        B = points.shape[0]
        view_shape = list(idx.shape)
        view_shape[1:] = [1] * (len(view_shape) - 1)
        repeat_shape = list(idx.shape)
        repeat_shape[0] = 1
        batch_indices = torch.arange(B, dtype=torch.long, device=device).view(view_shape).repeat(repeat_shape)
        new_points = points[batch_indices, idx, :]
        return new_points

    def forward(
        self,
        surface_pos: torch.Tensor,
        properties: torch.Tensor, 
        normals: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode point cloud features.

        Args:
            surface_pos: sampled positions from object surface (B, N, 3)
            properties: vector of [diffuse_color, specular_color, emissision_color, roughness] (B, 10)
            normals: Optional surface normals (B, N, 3)

        Returns:
            Tuple of:
            - local_features [Batch, num_centroids, output_dim]
            - local_positions [Batch, num_centroids, 3]
        """
        B, N, _ = surface_pos.shape

        # Concatenate input features
        features = [surface_pos]
        if normals is not None:
            features.append(normals)
        features.append(properties)

        x = torch.cat(features, dim=-1)  # (B, N, input_dim)
        x = x.transpose(1, 2)  # (B, input_dim, N)

        # PointNet backbone (extract per-point features)
        per_point_features = self.point_net(x)  # (B, backbone_dim, N)
        
        if self.pooling_type == 'local_features':
            assert self.use_local_patches
            # 1. FPS to get K centroids
            fps_idx = self.farthest_point_sample(surface_pos, self.num_centroids) # [B, num_centroids]
            
            # 2. Extract local_positions
            local_positions = self.index_points(surface_pos, fps_idx) # [B, num_centroids, 3]
            
            # 3. K-NN grouping around centroids
            # Compute distances from surface_pos to local_positions
            dist = torch.cdist(local_positions, surface_pos) # [B, num_centroids, N]
            _, knn_idx = torch.topk(dist, k=self.k_neighbors, dim=-1, largest=False) # [B, num_centroids, k_neighbors]
            
            # Gather features for neighbors
            # per_point_features is [B, backbone_dim, N], transpose to [B, N, backbone_dim]
            per_point_feat_t = per_point_features.transpose(1, 2) # [B, N, backbone_dim]
            
            # To use index_points, flatten idx temporarily to [B, num_centroids*k_neighbors]
            knn_idx_flat = knn_idx.view(B, -1)
            grouped_features_flat = self.index_points(per_point_feat_t, knn_idx_flat) # [B, num_centroids*k_neighbors, backbone_dim]
            grouped_features = grouped_features_flat.view(B, self.num_centroids, self.k_neighbors, self.backbone_dim)
            
            # Max pool over k_neighbors to get local_features per centroid
            local_features_max = torch.max(grouped_features, dim=2)[0] # [B, num_centroids, backbone_dim]
            
            # Additional projection via sa_proj (optional, but standard for Set Abstraction to refine grouped features)
            local_features_max = local_features_max.transpose(1, 2) # [B, backbone_dim, num_centroids]
            local_features_proj = self.sa_proj(local_features_max)
            if hasattr(self, 'sa_bn'):
                local_features_proj = self.sa_bn(local_features_proj)
            local_features_proj = self.sa_silu(local_features_proj)
            geometry_token = local_features_proj.transpose(1, 2) # [B, num_centroids, backbone_dim]

            # backbone handled both geometry and properties, so just project the geometry token
            geometry_token_flat = geometry_token.contiguous().view(B * self.num_centroids, self.backbone_dim)
            output_flat = self.out_proj(geometry_token_flat)
            output = output_flat.view(B, self.num_centroids, -1)

            return output, local_positions
        # single object token branches:
        elif self.pooling_type == 'max':
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
        
        output = self.out_proj(geometry_token)  # (B, output_dim)
        
        local_positions = surface_pos.mean(dim=1)  # (B, 3)
        
        return output, local_positions


class LitePTEncoderAdapter(nn.Module):
    """
    Adapter for the LitePT point cloud encoder to match the PointNetEncoder interface.
    """
    def __init__(
        self,
        in_channels: int = 16,
        out_channels: int = 512,
        pretrained_weights_path: Optional[str] = None,
        drop_path: float = 0.2,
        use_local_patches: bool = False
    ):
        super().__init__()
        self.use_local_patches = use_local_patches
        
        try:
            from LitePT.litept.model import LitePT
        except ImportError:
            raise ImportError("Could not import LitePT. Ensure LitePT/litept exists in the project root.")

        # Instantiate LitePT-S configuration by default
        # TODO: allow for passing in custom arch instead of hard-coded defaults? 
        self.backbone = LitePT(
            in_channels=in_channels,
            enc_depths=(2, 2, 6, 2),
            enc_channels=(96, 192, 384, 512),
            enc_num_head=(3, 6, 12, 16),
            enc_patch_size=(1024, 1024, 1024, 1024),
            enc_conv=(True, True, True, False),
            enc_attn=(False, False, False, True),
            drop_path=drop_path,
            enc_mode=True
        )

        if pretrained_weights_path:
            import os
            if os.path.exists(pretrained_weights_path):
                ckpt = torch.load(pretrained_weights_path, map_location='cpu')
                # Load backbone weights. If it's a full model, we might need to map keys.
                # Assuming standard state_dict loading for now.
                state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
                # Filter out decoder keys if present
                enc_state_dict = {k: v for k, v in state_dict.items() if not k.startswith('dec.')}
                self.backbone.load_state_dict(enc_state_dict, strict=False)
            else:
                print(f"Warning: Pretrained weights {pretrained_weights_path} not found.")

        # Project LitePT output dimension to out_channels
        litept_out_dim = 512
        if litept_out_dim != out_channels:
            self.proj = nn.Linear(litept_out_dim, out_channels)
        else:
            self.proj = nn.Identity()

    def forward(
        self,
        surface_pos: torch.Tensor,
        properties: torch.Tensor,
        normals: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode point cloud features using LitePT.

        Args:
            surface_pos: (B, N, 3)
            properties: (B, 10) # we expand this per point to concatenate
            normals: Optional (B, N, 3)

        Returns:
            Tuple of:
            - global_token [Batch, output_dim]
            - global_pos [Batch, 3]
        """
        if self.use_local_patches:
            raise NotImplementedError("use_local_patches=True is not yet implemented for LitePTEncoderAdapter.")

        B, N, _ = surface_pos.shape

        # Expand properties to per-point (B, N, 10)
        props_expanded = properties.unsqueeze(1).expand(B, N, -1)
        
        # Concatenate features: coords(3), normals(3), props(10) -> 16
        features_list = [surface_pos, props_expanded]
        if normals is not None:
            features_list.append(normals)
        
        # (B, N, in_channels) -> (B*N, in_channels)
        feat = torch.cat(features_list, dim=-1).view(B * N, -1)
        coord = surface_pos.view(B * N, 3)
        
        # Offset array for LitePT (cumulative point counts)
        # Assuming fixed N points per object for now
        device = surface_pos.device
        offset = torch.arange(1, B + 1, device=device, dtype=torch.int32) * N

        data_dict = {
            "feat": feat,
            "coord": coord,
            "grid_size": 0.01, # Typical voxel size for LitePT
            "offset": offset
        }

        # Forward pass
        out_point = self.backbone(data_dict)
        out_feat = out_point.feat # (B*N, litept_out_dim)

        out_feat = self.proj(out_feat) # (B*N, out_channels)
        
        # Reshape back to (B, N, out_channels)
        out_feat = out_feat.view(B, N, -1)

        # Global Pooling
        # TODO: Implement attention pooling as an alternative to global max pooling
        global_token = torch.max(out_feat, dim=1)[0] # (B, out_channels)
        
        # Mean position for centroid
        global_pos = surface_pos.mean(dim=1) # (B, 3)

        # PointNetEncoder outputs (B, output_dim). So we do the same.
        return global_token, global_pos
