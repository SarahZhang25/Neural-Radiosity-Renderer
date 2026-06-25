"""
Global Illumination Model
Predict radiance given scene composition and query camera view. 
"""

import torch
import torch.nn as nn
import yaml
from typing import Dict, Tuple, Optional

from model.encoder import PointNetEncoder
from model.layers.attention import TransformerEncoder
from model.decoder import TransformerDecoder
from model.layers.bidirectional_attention import BidirectionalTransformerEncoder
# from model.predictor import RadiancePredictor
from model.predictor_rope import RadiancePredictor
# from model.state_manager import StateManager
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
        self.use_dpt_decoder = config['predictor']['use_dpt_decoder']

        # Calculate rope_dim
        if config['predictor']['pe_type'] == 'nerf':
            self.rope_dim = None
        elif config['predictor']['pe_type'] == 'rope':
            # rope_dim controls how many frequency bands are used for positional encoding.
            # It must not exceed the head's capacity: 3 coords × (rope_dim//2) freqs ≤ head_dim//2.
            # Using max capacity (~42 for head_dim=128) starves content attention; use the config
            # value (typically 8) clamped to the max as a safety bound.
            decoder_head_dim = config['decoder']['hidden_dim'] // config['decoder']['num_heads']
            max_rope_dim = ((decoder_head_dim // 2) // 3) * 2
            self.rope_dim = min(config['ray_encoder']['vertex_pe_num_freqs'], max_rope_dim)
        else:
            raise ValueError(f"Invalid positional encoding type: {config['predictor']['pe_type']}")

        # Scene Representation Stage
        # 1. Object Encoder
        self.pointnet_encoder = PointNetEncoder(
            input_dim=config['encoder']['input_dim'],
            hidden_dims=config['encoder']['hidden_dims'],
            output_dim=config['encoder']['output_dim'],
            backbone_dim=config['encoder']['backbone_dim'],
            property_embed_dim=config['encoder'].get('property_embed_dim', 128),
            pooling_type=config['encoder']['pooling_type'],
            num_hierarchical_levels=config['encoder']['num_hierarchical_levels'],
        )
        
        # # 2. State Manager (Learnable 3D Grid)
        # self.state_manager = StateManager(
        #     num_tokens=config['state']['num_tokens'],
        #     token_dim=config['state']['token_dim'],
        #     learnable_init=config['state']['learnable_init'],
        #     init_scale=config['state']['init_scale']
        # )

        # 3. View-Independent Scene Transformer
        # TODO: possibly redo the config... I don't like "decoder".
        # should be like, representation learning phase or something
        self.scene_transformer = TransformerEncoder(
            num_layers=config['decoder']['num_layers'],
            num_heads=config['decoder']['num_heads'],
            hidden_dim=config['decoder']['hidden_dim'],
            ffn_hidden_dim=config['decoder']['feedforward_dim'],
            dropout=config['decoder']['dropout'],
            activation=config['decoder']['activation'],
            norm_type=config['decoder']['norm_type'],
            rope_dim=self.rope_dim,
            rope_type='object',
            bias=config['decoder']['bias'],# True by default...
            qk_norm=config['decoder']['qk_norm'],
            rope_double_max_freq=config['decoder']['rope_double_max_freq'],
            return_all_layers=config['decoder']['return_all_layers']
        )
        ## OLD VERSION
        # self.scene_transformer = BidirectionalTransformerEncoder(
        # # self.scene_transformer = TransformerDecoder(
        #     state_dim=config['decoder']['hidden_dim'],
        #     num_layers=config['decoder']['num_layers'],
        #     num_heads=config['decoder']['num_heads'],
        #     feedforward_dim=config['decoder']['feedforward_dim'],
        #     dropout=config['decoder']['dropout'],
        #     activation=config['decoder']['activation'],
        #     return_all_layers=config['decoder']['return_all_layers'],
        #     use_self_attention=config['decoder']['use_self_attention'],
        #     norm_type=config['decoder']['norm_type'],
        #     qk_norm=config['decoder']['qk_norm'],
        #     num_register_tokens=config['decoder']['num_register_tokens'],
        #     rope_dim=((config['decoder']['hidden_dim'] // config['decoder']['num_heads']) // 2) // 3 * 2, # Total angles must fit in head_dim // 2
        #     rope_type='object'
        # )

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
        print("Using predictor pe_type: ", config['predictor']['pe_type'])
        self.predictor = RadiancePredictor(
            hidden_dim=config['predictor']['hidden_dim'],
            patch_size=config['ray_encoder']['patch_size'],
            num_heads=config['predictor']['num_heads'],
            num_layers=config['predictor']['num_layers'],
            dropout=config['predictor']['dropout'],
            activation=config['predictor']['activation'],
            norm_type=config['predictor']['norm_type'],
            pe_type=config['predictor']['pe_type'],
            pe_num_freqs=config['predictor']['pe_num_freqs'],
            use_dpt_decoder=config['predictor'].get('use_dpt_decoder', True),
            dpt_features=config['predictor'].get('dpt_features', None),
            dpt_out_channels=config['predictor'].get('dpt_out_channels', None),
            include_alpha=config['predictor'].get('include_alpha', False)            
        )


    def forward(
        self,
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        obj_positions: torch.Tensor,
        obj_properties: torch.Tensor,
        obj_normals: Optional[torch.Tensor] = None,
        obj_mask: Optional[torch.Tensor] = None,
        w2c: Optional[torch.Tensor] = None, # (B, 4, 4) world to camera transform
    ) -> torch.Tensor:
        """
        Forward pass to predict radiance.

        Args:
            rays_o: Camera origin (B, 3)
            rays_d: Camera view directions (B, H, W, 3)
            obj_positions: Object point clouds (B, N_obj, N_vertices, 3)
            obj_properties: Object properties (B, N_obj, 10)
            obj_normals: Object normals (B, N_obj, N_vertices, 3)
            obj_mask: Mask for object padding (B, N_obj)
            w2c: Optional world-to-camera transformation (B, 4, 4) for RoPE in camera space
        Returns:
            radiance: Predicted RGB image (B, 3, H, W)
        """
        B, N_obj, N_v, _ = obj_positions.shape
        
        # --- Stage 1: Scene Representation ---

        # 1. Encode Objects
        # Flatten batch and objects for encoder
        flat_positions = obj_positions.view(B * N_obj, N_v, 3)
        
        flat_normals = None
        if obj_normals is not None:
             flat_normals = obj_normals.view(B * N_obj, N_v, 3)

        flat_props = obj_properties.view(B * N_obj, 10)
        
        obj_features_flat = self.pointnet_encoder(
            surface_pos=flat_positions,
            properties=flat_props,
            normals=flat_normals
        )
        
        # Reshape back: (B, N_obj, D)
        object_tokens = obj_features_flat.view(B, N_obj, -1)

        # 2. Get State Tokens
        # state_tokens = self.state_manager.get_tokens(B)  # (B, N_state, D)

        # 3. Spatial Positional Encoding (RoPE preparation)
        # Calculate Object Centroids for RoPE
        object_centroids = obj_positions.mean(dim=2)  # (B, N_obj, 3)

        # state_positions = None # no longer using self.state_manager.get_positions(B)  # (B, N_state, 3)

        # 4. Bi-Directional Interaction
        # all_state_layers, all_obj_layers = self.scene_transformer(
        #     state_tokens=state_tokens,
        #     obj_tokens=object_tokens,
        #     state_pos=state_positions,
        #     obj_pos=object_centroids
        # )
        # self-attn only transformer
        all_obj_layers =  self.scene_transformer(
            x=object_tokens,
            src_key_padding_mask=obj_mask,
            obj_pos=object_centroids
        )

        # --- Stage 2: Rendering ---

        # 5. Ray Encoding
        if rays_d is None:
            raise ValueError("ray_map (B, H, W, 3) is required for ray encoding")
             
        # ray_tokens: (B, N_patches, D)
        if w2c is not None:
             # rotate rays to camera space, where camera origin is 0
             rays_o_input = torch.zeros_like(rays_o) 
             w2c_R = w2c[:, :3, :3]
             rays_d_input = torch.einsum('bij,bhwj->bhwi', w2c_R, rays_d)
        else:
             rays_o_input = rays_o
             rays_d_input = rays_d
             
        ray_tokens, ray_token_pos = self.ray_encoder(rays_o_input, rays_d_input)

        # 6. Predict Radiance
        # Ray tokens query the scene (object + state features)
        # If no state features, predictor gracefully handles None input and sets scene_features using only object features
        radiance = self.predictor(
            multi_scale_features=all_obj_layers,
            query_view_features=ray_tokens,
            # multi_scale_state_features=all_state_layers,
            patch_h=rays_d.shape[1] // self.ray_encoder.patch_size,
            patch_w=rays_d.shape[2] // self.ray_encoder.patch_size,
            w2c=w2c,
            obj_positions=obj_positions,
            obj_mask=obj_mask,
            ray_positions=ray_token_pos,  # would use this if using predictor with RoPE supported 
            use_dpt_decoder=self.use_dpt_decoder
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