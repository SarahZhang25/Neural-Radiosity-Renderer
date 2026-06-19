import torch
import torch.nn as nn

from model.encodings.nerf_encoding import NeRFEncoding
from einops import rearrange


class RayEncoder(nn.Module):
    def __init__(
        self,
        pe_type: str = 'rope',
        vertex_pe_num_freqs: int = 12,
        vdir_pe_type: str = 'nerf',
        vdir_num_freqs: int = 0,
        patch_size: int = 8,
        norm_type: str = 'rms_norm',
        view_transformer_latent_dim: int = 768, ## should this just be predictor hidden dim???
        view_transformer_n_heads: int = 4 ## should this just be predictor hidden dim???
    ):
        super().__init__()
        self.pe_type = pe_type
        self.vertex_pe_num_freqs = vertex_pe_num_freqs
        self.vdir_pe_type = vdir_pe_type
        self.vdir_num_freqs = vdir_num_freqs
        self.patch_size = patch_size
        self.norm_type = norm_type
        # TODO: possibly refactor these.....
        self.view_transformer_latent_dim = view_transformer_latent_dim
        self.view_transformer_n_heads = view_transformer_n_heads

        if pe_type == 'nerf':
            self.pos_pe = NeRFEncoding(
                in_dim=3,
                num_frequencies=vertex_pe_num_freqs,
                include_input=True
            )
            self.pe_token_proj = nn.Linear(
                self.pos_pe.get_out_dim(),
                view_transformer_latent_dim
            )
            if norm_type == 'layer_norm':
                self.token_pos_pe_norm = nn.LayerNorm(view_transformer_latent_dim)
            elif norm_type == 'rms_norm':
                self.token_pos_pe_norm = nn.RMSNorm(view_transformer_latent_dim)
            else:
                raise ValueError(f"Unsupported normalization type: {norm_type}")
            self.rope_dim = None
        elif pe_type == 'rope':
            self.rope_dim = min(vertex_pe_num_freqs, view_transformer_latent_dim // view_transformer_n_heads // 9 * 2) 
            # us: 9 = 3 (centroid pos xyz) * 2 (sin and cos)
            # renderformer: 18 = 3 (triangles) * 3 (vertex xyz)* 2 (sin and cos) 
        else:
            raise ValueError(f"Unsupported positional encoding type: {pe_type}")

        self.ray_map_patch_token = nn.Parameter(torch.randn(1, 1, view_transformer_latent_dim))
        if vdir_pe_type == 'nerf':
            self.vdir_pe = NeRFEncoding(
                in_dim=3,
                num_frequencies=vdir_num_freqs,
                include_input=True
            )
            self.ray_map_encoder = nn.Linear(
                self.vdir_pe.get_out_dim() * patch_size * patch_size,
                view_transformer_latent_dim
            )
            if norm_type == 'layer_norm':
                self.ray_map_encoder_norm = nn.LayerNorm(view_transformer_latent_dim)
            elif norm_type == 'rms_norm':
                self.ray_map_encoder_norm = nn.RMSNorm(view_transformer_latent_dim)
            else:
                raise ValueError(f"Unsupported normalization type: {norm_type}")
        else:
            raise ValueError(f"Unsupported view direction positional encoding type: {vdir_pe_type}")

    def forward(self, camera_o, ray_map):
        """
        Encode ray map.

        Args:
            camera_o (torch.Tensor): (B, 3) Camera origin
            ray_map (torch.Tensor): (B, H, W, 3) Normalized ray directions
        Returns:
            ray_tokens: (B, N_PATCHES, D)
            ray_token_pos: (B, N_PATCHES, 3)
        """

        # query sequence
        B, H, W, _ = ray_map.shape
        ray_map_enc = self.vdir_pe(ray_map)
        ray_tokens = rearrange(ray_map_enc, 'b (h1 p1) (w1 p2) c -> b (h1 w1) (c p1 p2)', p1=self.patch_size, p2=self.patch_size)
        
        patch_h = H // self.patch_size
        patch_w = W // self.patch_size
        
        ray_tokens = self.ray_map_patch_token + self.ray_map_encoder_norm(self.ray_map_encoder(ray_tokens))  # [B, N_PATCHES, D]
        n_patches = ray_tokens.size(1)

        ray_token_pos = camera_o[:, None].repeat(1, n_patches, 1)  # [B, N_PATCHES, 3]
        if self.pe_type == 'nerf': #NOTE: why do both ray and tri token suse the same pos_pe and pe_token_proj?
            ray_tokens = ray_tokens + self.token_pos_pe_norm(self.pe_token_proj(self.pos_pe(ray_token_pos)))

        return ray_tokens, ray_token_pos
