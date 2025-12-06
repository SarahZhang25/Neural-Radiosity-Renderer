"""
Global Illumination Model
Predict radiance in scene of emitter and modifier objects
"""

import torch
import torch.nn as nn
import yaml
from typing import Dict, Tuple, Optional

from encoder import PointNetEncoder
from decoder import TransformerDecoder
from predictor import RadiancePredictor
from state_manager import StateManager
from ray_encoder import RayEncoder
from utils import ray_generator

class GlobalIlluminationModel(nn.Module):
    """
    Architecture:
        [Object Point Clouds]   -> *PointNetEncoder ->
                                   *State Tokens    -> *Decoder      -> 
                                   [Ray Query]      -> *Ray Encoder  -> *Predictor -> Radiance
    (Key: [] Inputs, * Learnable)

    Input:
        -

    Output:
        - 

    """
    def __init__(self, config: Dict):
        super.__init__()

        # 

        self.pointnet_encoder = PointNetEncoder(
            input_dim=config['encoder']['input_dim'],
            hidden_dims=config['encoder']['hidden_dims'],
            output_dim=config['encoder']['output_dim'],
            backbone_dim=config['encoder']['backbone_dim'],
            fusion_hidden_dim=config['encoder']['fusion_hidden_dim'],
            pooling_type=config['encoder']['pooling_type'],
            num_hierarchical_levels=config['encoder']['num_hierarchical_levels'],
            use_physics_params=config['encoder']['use_physics_params']
        )
        
        self.state_manager = StateManager(
            num_tokens=config['state']['num_tokens'],
            token_dim=config['state']['token_dim'],
            learnable_init=config['state']['learnable_init'],
            init_scale=config['state']['init_scale']
        )

        self.decoder = TransformerDecoder(
            state_dim=config['decoder']['hidden_dim'],
            num_layers=config['decoder']['num_layers'],
            num_heads=config['decoder']['num_heads'],
            feedforward_dim=config['decoder']['feedforward_dim'],
            dropout=config['decoder']['dropout'],
            activation=config['decoder']['activation'],
            return_all_layers=config['decoder']['return_all_layers'],
            use_self_attention=config['decoder']['use_self_attention'],
            norm_type=config['decoder']['norm_type'],
            qk_norm=config['decoder']['qk_norm'],
            num_register_tokens=config['decoder']['num_register_tokens']
        )

        self.predictor = RadiancePredictor(
            hidden_dim=config['predictor']['hidden_dim'],
            patch_size=config['ray_encoder']['patch_size'],
            num_heads=config['predictor']['num_heads'],
            num_layers=config['predictor']['num_layers'],
            dropout=config['predictor']['dropout'],
            activation=config['predictor']['activation'],
            norm_type=config['predictor']['norm_type'],
        )

        self.ray_encoder = RayEncoder(
            pe_type=config['ray_encoder']['pe_type'],
            vertex_pe_num_freqs=config['ray_encoder']['vertex_pe_num_freqs'],
            vdir_pe_type=config['ray_encoder']['vdir_pe_type'],
            vdir_num_freqs=config['ray_encoder']['vdir_num_freqs'],
            patch_size=config['ray_encoder']['patch_size'],
            norm_type=config['ray_encoder']['norm_type'],
            view_transformer_latent_dim=config['ray_encoder']['view_transformer_latent_dim'],
            view_transformer_n_heads=config['ray_encoder']['view_transformer_n_heads']
        )


    def forward(
        self,
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        obj_positions: torch.Tensor,
        obj_properties: Optional[torch.Tensor],
        obj_types: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Forward pass to predict accelerations.

        Args:
            positions: Current positions (B, N_obj, N_vertices, 3)
            properties:  diffuse [r,g,b] (B, 3) if modifier; intensity [r,g,b] [type] (B, 4) if emitter

        Returns:
            accelerations: Predicted radiance
        """
        B, N_obj, N_v, _ = positions.shape
        device = positions.device

        # # 1. Sample anchor points if not provided
        # if anchor_indices is None:
        #     anchor_indices = self._sample_anchors(positions, self.anchor_k)
        # N_q = anchor_indices.shape[2]  # Number of anchors

        # # 2. Compute features
        # # Distance to nearest point on other objects
        # dist_vec = compute_knn_distance(positions)  # (B, N_obj, N_v, 3)

        # # Relative position to reference
        # if ref_positions is None:
        #     ref_positions = positions  # Use current as reference if not provided
        # rel_pos = positions - ref_positions  # (B, N_obj, N_v, 3)

        # 3. Encode per-object features
        object_features = []
        for obj_idx in range(N_obj):
            # Extract per-object data
            pos_obj = obj_positions[:, obj_idx]  # (B, N_v, 3)
            prop_obj = obj_properties[:, obj_idx]
            obj_type_obj = obj_types[:, obj_idx]
            emitter_type_obj = obj_types[:, obj_idx] #NOTE: assume same as obj_type for now
            
            # Encode
            obj_feat = self.pointnet_encoder(
                surface_pos=pos_obj, 
                properties=prop_obj, 
                object_type=obj_type_obj,
                emitter_type=emitter_type_obj
            )
            object_features.append(obj_feat)

        object_features = torch.stack(object_features, dim=1)  # (B, N_obj, D)

        # 4. Get state tokens
        state_tokens = self.state_manager.get_tokens(B)  # (B, N_state, D)

        # 5. Decoder: Cross-attention between state and object tokens
        all_state_layers, all_obj_layers = self.decoder(
            state_tokens=state_tokens,
            obj_tokens=object_features
        )

        # 6. Prepare query for predictor
        # Gather features at anchor points
        anchor_positions = self._gather_at_anchors(positions, anchor_indices)  # (B, N_obj, N_q, 3)
        anchor_velocities = self._gather_at_anchors(velocities, anchor_indices)  # (B, N_obj, N_q, 3)
        anchor_dist_vec = self._gather_at_anchors(dist_vec, anchor_indices)  # (B, N_obj, N_q, 3)
        anchor_rel_pos = self._gather_at_anchors(rel_pos, anchor_indices)  # (B, N_obj, N_q, 3)

        # Concatenate query features
        if physics_params is not None:
            physics_expanded = physics_params.unsqueeze(2).expand(B, N_obj, N_q, 3)
            query = torch.cat([
                anchor_dist_vec, anchor_velocities, anchor_rel_pos, physics_expanded
            ], dim=-1)  # (B, N_obj, N_q, 12)
        else:
            query = torch.cat([
                anchor_dist_vec, anchor_velocities, anchor_rel_pos
            ], dim=-1)  # (B, N_obj, N_q, 9)

        # Reference positions at anchors
        ref_pose_at_anchors = self._gather_at_anchors(ref_positions, anchor_indices)  # (B, N_obj, N_q, 3)

        # 7. Predict accelerations
        accelerations = self.predictor(
            multi_scale_features=all_obj_layers,
            query=query,
            ref_pose=ref_pose_at_anchors,
            multi_scale_state_features=all_state_layers
        )

        return accelerations  # (B, N_obj, N_q, 3)

    def _sample_anchors(self, positions: torch.Tensor, k: int) -> torch.Tensor:
        """Sample k anchor points per object using FPS."""
        B, N_obj, N_v, _ = positions.shape
        anchor_indices = []

        for b in range(B):
            obj_anchors = []
            for obj_idx in range(N_obj):
                pts = positions[b, obj_idx]  # (N_v, 3)
                idx = fps_sampling(pts, k)  # (k,)
                obj_anchors.append(idx)
            anchor_indices.append(torch.stack(obj_anchors))

        return torch.stack(anchor_indices)  # (B, N_obj, k)

    def _gather_at_anchors(self, features: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """Gather features at anchor points."""
        B, N_obj, N_v, D = features.shape
        N_q = indices.shape[2]

        # Expand indices for gathering
        idx_exp = indices.unsqueeze(-1).expand(B, N_obj, N_q, D)

        # Gather
        gathered = torch.gather(features, dim=2, index=idx_exp)

        return gathered  # (B, N_obj, N_q, D)


def load_model(config_path: str, checkpoint_path: Optional[str] = None) -> GlobalIlluminationModel:
    """
    Load model from configuration and optional checkpoint.

    Args:
        config_path: Path to config.yaml
        checkpoint_path: Optional path to model checkpoint

    Returns:
        Initialized model
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    model = GlobalIlluminationModel(config)

    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from {checkpoint_path}")

    return model