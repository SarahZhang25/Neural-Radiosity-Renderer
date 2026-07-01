"""
Global Illumination Model
Predict radiance given scene composition and query camera view. 
"""

import torch
import yaml
from typing import Dict, Optional

from model.encoder import PointNetEncoder
from model.layers.attention import TransformerEncoder
from model.predictor_rope import RadiancePredictor
# from model.state_manager import StateManager
from model.ray_encoder import RayEncoder


def compute_obb_axes(point_cloud: torch.Tensor) -> torch.Tensor:
    """
    Compute oriented bounding box (OBB) scaled principal axes from a point cloud
    using covariance matrix eigendecomposition.

    Args:
        point_cloud: (..., N_points, 3) point cloud positions

    Returns:
        scaled_axes_flat: (..., 9) three scaled principal axis vectors flattened
            (axis1_x, axis1_y, axis1_z, axis2_x, axis2_y, axis2_z, axis3_x, axis3_y, axis3_z)
            Each axis is the eigenvector scaled by sqrt(eigenvalue), representing
            the principal direction and extent of the point distribution.
    """
    # Center the point cloud
    centroid = point_cloud.mean(dim=-2, keepdim=True)  # (..., 1, 3)
    centered = point_cloud - centroid  # (..., N_points, 3)

    # Covariance matrix: (..., 3, 3)
    N = point_cloud.shape[-2]
    cov = centered.transpose(-1, -2) @ centered / N  # (..., 3, 3)

    # Eigendecomposition (eigh is optimized for symmetric PSD matrices)
    # Convert to float32 as eigh is not implemented for bfloat16 on CUDA
    orig_dtype = cov.dtype
    eigenvalues, eigenvectors = torch.linalg.eigh(cov.float())
    eigenvalues = eigenvalues.to(orig_dtype)
    eigenvectors = eigenvectors.to(orig_dtype)

    # Clamp eigenvalues to avoid sqrt of negative due to numerical precision
    eigenvalues = eigenvalues.clamp(min=0.0)

    # Scale eigenvectors by sqrt(eigenvalue) to get extent-scaled axes
    # eigenvectors: (..., 3, 3), eigenvalues.sqrt(): (..., 3)
    scaled_axes = eigenvectors * eigenvalues.sqrt().unsqueeze(-2)  # (..., 3, 3)

    # Flatten: (..., 3, 3) -> (..., 9) where each group of 3 is one axis vector
    # Transpose so rows are axes: (..., 3, 3) with scaled_axes[..., :, i] -> axis i
    shape = point_cloud.shape[:-2] + (9,)
    scaled_axes_flat = scaled_axes.transpose(-1, -2).reshape(shape)  # (..., 9)

    return scaled_axes_flat

class GlobalIlluminationModel(torch.nn.Module):
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

        # Determine rope variant from pe_type
        pe_type = config['predictor']['pe_type']
        if pe_type == 'rope_obb':
            self.rope_variant = 'obb'
            n_coords = 12
        elif 'rope' in pe_type:  # 'rope' or 'rope_centroid'
            self.rope_variant = 'centroid'
            n_coords = 3
        else:
            self.rope_variant = None
            n_coords = 3

        # Calculate rope_dim
        if pe_type == 'nerf':
            self.rope_dim = None
        elif 'rope' in pe_type:
            # rope_dim controls how many frequency bands are used for positional encoding.
            # It must not exceed the head's capacity: n_coords × (rope_dim//2) freqs ≤ head_dim//2.
            # Using max capacity starves content attention; use the config
            # value (typically 8) clamped to the max as a safety bound.
            decoder_head_dim = config['decoder']['hidden_dim'] // config['decoder']['num_heads']
            max_rope_dim = ((decoder_head_dim // 2) // n_coords) * 2
            self.rope_dim = min(config['ray_encoder']['vertex_pe_num_freqs'], max_rope_dim)
        else:
            raise ValueError(f"Invalid positional encoding type: {pe_type}")

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
            use_local_patches=config['encoder'].get('use_local_patches', False),
            num_centroids=config['encoder'].get('num_centroids', 16),
        )
        
        # 2. Register Tokens
        self.num_register_tokens = config['decoder'].get('num_register_tokens', 0)
        if self.num_register_tokens > 0:
            self.register_tokens = torch.nn.Parameter(torch.randn(1, self.num_register_tokens, config['decoder']['hidden_dim']))

        # 3. View-Independent Scene Transformer
        # TODO: possibly redo the config... I don't like "decoder".
        # should be like, representation learning phase or something
        self.scene_transformer = TransformerEncoder(
            num_layers=config['decoder']['num_layers'],
            num_heads=config['decoder']['num_heads'],
            hidden_dim=config['decoder']['hidden_dim'],
            ffn_hidden_dim=config['decoder']['ffn_hidden_dim'],
            dropout=config['decoder']['dropout'],
            activation=config['decoder']['activation'],
            norm_type=config['decoder']['norm_type'],
            rope_dim=self.rope_dim,
            rope_type='object',
            rope_variant=self.rope_variant or 'centroid',
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
            ffn_hidden_dim=config['predictor']['ffn_hidden_dim'],
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
            obj_properties: Object properties (B, N_obj, N_v, 10) per-point. Legacy: (B, N_obj, 10) uniform texture per object
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

        if obj_properties.dim() == 3:
            # Legacy dataset with uniform texture per object: (B, N_obj, 10) -> (B, N_obj, N_v, 10)
            obj_properties = obj_properties.unsqueeze(2).expand(-1, -1, N_v, -1)
            
        flat_props = obj_properties.reshape(B * N_obj, N_v, -1)
        
        obj_features_flat, obj_positions_local = self.pointnet_encoder(
            surface_pos=flat_positions,
            properties=flat_props,
            normals=flat_normals
        )
        
        if self.pointnet_encoder.use_local_patches:
            n_centroids = self.pointnet_encoder.num_centroids
            
            # obj_features_flat: (B*N_obj, n_centroids, D) -> (B, N_obj*n_centroids, D)
            object_tokens = obj_features_flat.view(B, N_obj * n_centroids, -1)
            # obj_positions_local: (B*N_obj, n_centroids, 3) -> (B, N_obj*n_centroids, 3)
            object_centroids = obj_positions_local.view(B, N_obj * n_centroids, 3)
            
            # Expand mask to match expanded sequence length
            if obj_mask is not None:
                obj_mask = obj_mask.repeat_interleave(n_centroids, dim=1)
        else: # [this should be default branch]
            # Global token per object: (B*N_obj, D) -> (B, N_obj, D)
            object_tokens = obj_features_flat.view(B, N_obj, -1)
            # Centroid per object: (B*N_obj, 3) -> (B, N_obj, 3)
            object_centroids = obj_positions_local.view(B, N_obj, 3)
            # obj_mask: (B, N_obj) -- unchanged

        # Prepend Register Tokens if configured
        if self.num_register_tokens > 0:
            reg_tokens = self.register_tokens.expand(B, -1, -1)
            object_tokens = torch.cat([reg_tokens, object_tokens], dim=1)
            
            if obj_mask is not None:
                mask_weight = (obj_mask.float() / (obj_mask.sum(dim=1, keepdim=True) + 1e-5)).unsqueeze(-1)
                center_pos = (object_centroids * mask_weight).sum(dim=1, keepdim=True)
                
                reg_mask = torch.ones((B, self.num_register_tokens), dtype=obj_mask.dtype, device=obj_mask.device)
                obj_mask = torch.cat([reg_mask, obj_mask], dim=1)
            else:
                center_pos = object_centroids.mean(dim=1, keepdim=True)
                
            center_pos = center_pos.expand(-1, self.num_register_tokens, -1)
            object_centroids = torch.cat([center_pos, object_centroids], dim=1)

        # 4. Bi-Directional Interaction
        # all_state_layers, all_obj_layers = self.scene_transformer(
        #     state_tokens=state_tokens,
        #     obj_tokens=object_tokens,
        #     state_pos=state_positions,
        #     obj_pos=object_centroids
        # )
        # self-attn only transformer
        all_obj_layers = self.scene_transformer(
            x=object_tokens,
            src_key_padding_mask=obj_mask,
            obj_pos=object_centroids
        ) # list of (B, Seq_Len, D) -- passed directly to predictor
        
        # Pre-extract rotation from w2c once, reused for both token positions and ray directions
        if w2c is not None:
            w2c_R = w2c[:, :3, :3]  # (B, 3, 3)
            w2c_t = w2c[:, :3, 3]   # (B, 3)
        
        # 5. Transform per-token world-space centroids to camera space for the predictor
        if w2c is not None:
            obj_token_positions_cam = torch.bmm(object_centroids, w2c_R.transpose(1, 2)) + w2c_t.unsqueeze(1)  # (B, Seq_Len, 3)
        else:
            obj_token_positions_cam = object_centroids  # (B, Seq_Len, 3)

        # 6. Compute OBB if using rope_obb and concatenate centroid + axes -> (B, Seq_Len, 12)
        if self.rope_variant == 'obb':
            # Compute OBB axes for each object from its point cloud
            # obj_positions: (B, N_obj, N_v, 3)
            obb_axes = compute_obb_axes(obj_positions)  # (B, N_obj, 9)

            # Transform axes to camera space (rotation only, axes are direction vectors)
            if w2c is not None:
                # obb_axes: (B, N_obj, 9) -> (B, N_obj, 3, 3) for rotation
                obb_axes_3x3 = obb_axes.view(B, N_obj, 3, 3)  # 3 axes, each 3D
                # Rotate each axis: (B, N_obj, 3, 3) @ (B, 1, 3, 3) -> (B, N_obj, 3, 3)
                obb_axes_cam = torch.einsum('boij,bjk->boik', obb_axes_3x3, w2c_R.transpose(1, 2))  # (B, N_obj, 3, 3)
                obb_axes_cam_flat = obb_axes_cam.reshape(B, N_obj, 9)  # (B, N_obj, 9)
            else:
                obb_axes_cam_flat = obb_axes  # (B, N_obj, 9)

            # Handle register tokens: pad OBB axes with zeros for register tokens
            if self.num_register_tokens > 0:
                reg_axes = torch.zeros(B, self.num_register_tokens, 9, device=obb_axes_cam_flat.device, dtype=obb_axes_cam_flat.dtype)
                obb_axes_cam_flat = torch.cat([reg_axes, obb_axes_cam_flat], dim=1)  # (B, Seq_Len, 9)

            # Concatenate centroid (3D) + axes (9D) -> (B, Seq_Len, 12)
            obj_token_positions = torch.cat([obj_token_positions_cam, obb_axes_cam_flat], dim=-1)
        else:
            obj_token_positions = obj_token_positions_cam

        # --- Stage 2: Rendering ---

        # 6. Ray Encoding
        # ray_tokens: (B, N_patches, D)
        if w2c is not None:
             # rotate rays to camera space, where camera origin is 0
             rays_o_input = torch.zeros_like(rays_o) 
             rays_d_input = torch.einsum('bij,bhwj->bhwi', w2c_R, rays_d)
        else:
             rays_o_input = rays_o
             rays_d_input = rays_d
             
        ray_tokens, ray_token_pos = self.ray_encoder(rays_o_input, rays_d_input)

        # 7. Predict Radiance
        # Ray tokens query the scene (object + state features)
        # If no state features, predictor gracefully handles None input and sets scene_features using only object features
        radiance = self.predictor(
            multi_scale_features=all_obj_layers,
            query_view_features=ray_tokens,
            # multi_scale_state_features=all_state_layers,
            patch_h=rays_d.shape[1] // self.ray_encoder.patch_size,
            patch_w=rays_d.shape[2] // self.ray_encoder.patch_size,
            obj_token_positions=obj_token_positions,  # (B, N_obj (* num_centroids), 3 or 12) in camera space
            obj_mask=obj_mask,
            ray_positions=ray_token_pos,
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