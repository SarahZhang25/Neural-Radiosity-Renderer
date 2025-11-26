"""
MLP-based predictor for acceleration prediction.
"""

import torch
import torch.nn as nn
from typing import List, Optional


class AccelerationPredictor(nn.Module):
    """
    Simple MLP predictor that takes multi-scale features and query to predict accelerations.
    """

    def __init__(
        self,
        input_dim: int = 15,  # query(12) + ref_pose(3)
        hidden_dim: int = 512,
        output_dim: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1,
        activation: str = 'swiglu',
        use_ref_pose: bool = True,
        norm_type: str = 'rms_norm',
        qk_norm: bool = True
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.use_ref_pose = use_ref_pose

        # Query projection (excluding ref_pose which is added separately)
        query_dim = input_dim - 3 if use_ref_pose else input_dim
        self.query_proj = nn.Linear(query_dim, hidden_dim)

        # Feature pooling across layers
        self.layer_weights = nn.Parameter(torch.ones(3))  # For 3 decoder layers

        # MLP layers
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),  # query + obj_feat + state_feat
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

        # Initialize output layer with small weights
        nn.init.xavier_uniform_(self.mlp[-1].weight, gain=0.01)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(
        self,
        multi_scale_features: List[torch.Tensor],
        query: torch.Tensor,
        ref_pose: Optional[torch.Tensor] = None,
        multi_scale_state_features: Optional[List[torch.Tensor]] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Predict accelerations from features and query.

        Args:
            multi_scale_features: List of object features from decoder layers [(B, N_obj, D), ...]
            query: Query features at anchor points (B, N_obj, N_anchors, query_dim)
            ref_pose: Reference positions at anchors (B, N_obj, N_anchors, 3)
            multi_scale_state_features: List of state features from decoder layers [(B, N_state, D), ...]

        Returns:
            Predicted accelerations (B, N_obj, N_anchors, 3)
        """
        B, N_obj, N_anchors, _ = query.shape

        # Process query
        if self.use_ref_pose and ref_pose is not None:
            # Concatenate ref_pose to query
            query_with_ref = torch.cat([query, ref_pose], dim=-1)  # (B, N_obj, N_anchors, input_dim)
            query_flat = query_with_ref.view(B * N_obj * N_anchors, -1)
        else:
            query_flat = query.view(B * N_obj * N_anchors, -1)

        # Project query
        query_features = self.query_proj(query_flat[:, :-3])  # Exclude ref_pose from projection
        if self.use_ref_pose and ref_pose is not None:
            ref_pose_flat = query_flat[:, -3:]  # Extract ref_pose
        else:
            ref_pose_flat = None

        # Pool object features across layers
        layer_weights = torch.softmax(self.layer_weights, dim=0)
        obj_features = 0
        for i, feat in enumerate(multi_scale_features[:3]):  # Use first 3 layers
            # feat: (B, N_obj, D)
            feat_expanded = feat.unsqueeze(2).expand(B, N_obj, N_anchors, -1)  # (B, N_obj, N_anchors, D)
            feat_flat = feat_expanded.view(B * N_obj * N_anchors, -1)
            obj_features = obj_features + layer_weights[i] * feat_flat

        # Pool state features across layers (if provided)
        if multi_scale_state_features is not None:
            state_features = 0
            for i, feat in enumerate(multi_scale_state_features[:3]):
                # feat: (B, N_state, D)
                # Average pool across state tokens
                feat_pooled = feat.mean(dim=1, keepdim=True)  # (B, 1, D)
                feat_expanded = feat_pooled.expand(B, N_obj * N_anchors, -1)  # (B, N_obj*N_anchors, D)
                feat_flat = feat_expanded.view(B * N_obj * N_anchors, -1)
                state_features = state_features + layer_weights[i] * feat_flat
        else:
            state_features = torch.zeros_like(obj_features)

        # Concatenate all features
        if ref_pose_flat is not None:
            # Include ref_pose in the concatenation
            combined = torch.cat([
                query_features,
                obj_features,
                state_features,
                ref_pose_flat.expand(-1, self.hidden_dim)[:, :self.hidden_dim]  # Pad ref_pose to match dim
            ], dim=-1)
        else:
            combined = torch.cat([query_features, obj_features, state_features], dim=-1)

        # Predict accelerations
        accelerations = self.mlp(combined)  # (B*N_obj*N_anchors, 3)

        # Reshape to output format
        accelerations = accelerations.view(B, N_obj, N_anchors, 3)

        return accelerations