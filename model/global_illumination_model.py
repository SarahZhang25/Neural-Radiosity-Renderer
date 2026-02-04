"""
Global Illumination Model
Predict radiance in scene of emitter and modifier objects
"""

import torch
import torch.nn as nn
import yaml
from typing import Dict, Tuple, Optional

from model.encoder import PointNetEncoder
from model.decoder import TransformerDecoder
from model.predictor import RadiancePredictor
from model.state_manager import StateManager
from model.ray_encoder import RayEncoder

class GlobalIlluminationModel(nn.Module):
    """
    Architecture:
        [Object Point Clouds]   -> *PointNetEncoder ->
                                   *State Tokens    -> *Bi-Directional Transformer -> 
                                   [Ray Query]      -> *Ray Encoder  -> *Predictor -> Radiance
    (Key: [] Inputs, * Learnable)
    """
    def __init__(self, config: Dict):
        super().__init__()

        # Scene Representation Stage
        # 1. Object Encoder
        self.pointnet_encoder = PointNetEncoder(
            input_dim=config['encoder']['input_dim'],
            hidden_dims=config['encoder']['hidden_dims'],
            output_dim=config['encoder']['output_dim'],
            backbone_dim=config['encoder']['backbone_dim'],
            # fusion_hidden_dim=config['encoder']['fusion_hidden_dim'], # Removed
            pooling_type=config['encoder']['pooling_type'],
            num_hierarchical_levels=config['encoder']['num_hierarchical_levels'],
            # use_physics_params=config['encoder']['use_physics_params'] # Removed
        )
        
        # 2. State Manager (Learnable 3D Grid)
        self.state_manager = StateManager(
            num_tokens=config['state']['num_tokens'],
            token_dim=config['state']['token_dim'],
            learnable_init=config['state']['learnable_init'],
            init_scale=config['state']['init_scale']
        )

        # 3. View-Independent Bi-Directional Transformer
        self.scene_transformer = TransformerDecoder(
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

        # Rendering Stage
        # 4. Ray Encoder
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

        # 5. Predictor Transformer
        self.predictor = RadiancePredictor(
            hidden_dim=config['predictor']['hidden_dim'],
            patch_size=config['ray_encoder']['patch_size'],
            num_heads=config['predictor']['num_heads'],
            num_layers=config['predictor']['num_layers'],
            dropout=config['predictor']['dropout'],
            activation=config['predictor']['activation'],
            norm_type=config['predictor']['norm_type'],
        )


    def forward(
        self,
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        obj_positions: torch.Tensor,
        obj_properties: torch.Tensor,
        obj_class_ids: torch.Tensor,
        ray_map: Optional[torch.Tensor] = None, # Added ray_map for RayEncoder
        obj_normals: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass to predict radiance.

        Args:
            rays_o: Camera origin (B, 3)
            rays_d: Camera view direction (not explicitly used if ray_map provided, but good for context)
            obj_positions: Object point clouds (B, N_obj, N_vertices, 3)
            obj_properties: Object properties (B, N_obj, 3) (e.g. RGB)
            obj_class_ids: Object class IDs (B, N_obj)
            ray_map: Ray tokens map (B, H, W, 3) - Represents camera rays/directions per pixel
            obj_normals: Object normals (B, N_obj, N_vertices, 3)

        Returns:
            radiance: Predicted RGB image (B, 3, H, W)
        """
        B, N_obj, N_v, _ = obj_positions.shape
        
        # --- Stage 1: Scene Representation ---

        # 1. Encode Objects
        # Flatten batch and objects for encoder: (B*N_obj, N_v, 3)
        flat_positions = obj_positions.view(B * N_obj, N_v, 3)
        
        flat_normals = None
        if obj_normals is not None:
             flat_normals = obj_normals.view(B * N_obj, N_v, 3)

        flat_props = obj_properties.view(B * N_obj, 3)
        flat_ids = obj_class_ids.view(B * N_obj)
        
        obj_features_flat = self.pointnet_encoder(
            surface_pos=flat_positions,
            properties=flat_props,
            object_class_ids=flat_ids,
            normals=flat_normals
        )
        
        # Reshape back: (B, N_obj, D)
        object_tokens = obj_features_flat.view(B, N_obj, -1)

        # 2. Get State Tokens
        state_tokens = self.state_manager.get_tokens(B)  # (B, N_state, D)

        # 3. Spatial Positional Encoding (RoPE preparation)
        # Calculate Object Centroids for RoPE
        # (B, N_obj, 3)
        object_centroids = obj_positions.mean(dim=2)
        
        # 3. Spatial Positional Encoding (RoPE preparation)
        # NOTE: For initial debugging, we are skipping explicit RoPE and state position generation.
        # The PointNetEncoder already encodes absolute surface positions into the object tokens.
        state_positions = None 
        object_centroids = None # Not needed until RoPE is implemented

        # 4. Bi-Directional Interaction
        all_state_layers, all_obj_layers = self.scene_transformer(
            state_tokens=state_tokens,
            obj_tokens=object_tokens,
            state_pos=state_positions,
            obj_pos=object_centroids
        )

        # --- Stage 2: Rendering ---

        # 5. Ray Encoding
        if ray_map is None:
            raise ValueError("ray_map (B, H, W, 3) is required for ray encoding")
             
        # ray_tokens: (B, N_patches, D)
        ray_tokens = self.ray_encoder(rays_o, ray_map) 

        # 6. Predict Radiance
        # Ray tokens query the scene (object + state features)
        radiance = self.predictor(
            multi_scale_features=all_obj_layers,
            query_view_features=ray_tokens,
            multi_scale_state_features=all_state_layers,
            patch_h=ray_map.shape[1] // self.ray_encoder.patch_size,
            patch_w=ray_map.shape[2] // self.ray_encoder.patch_size
        )

        return radiance



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