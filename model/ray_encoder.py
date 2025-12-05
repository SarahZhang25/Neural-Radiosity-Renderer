import torch
import torch.nn as nn

from encodings.nerf_encoding import NeRFEncoding
from einops import rearrange


class RayEncoder(nn.Module):
    def __init__(
        self,
        pe_type: str = 'rope',
        vertex_pe_num_freqs: int = 12,
        vdir_pe_type: str = 'nerf',
        vdir_num_freqs: int = 0,
        patch_size: int = 16,
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
        # refactor these.....
        self.view_transformer_latent_dim = view_transformer_latent_dim
        self.view_transformer_n_heads = view_transformer_n_heads

        if pe_type == 'nerf':
            self.pos_pe = NeRFEncoding(
                in_dim=9,
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
            self.rope_dim = min(vertex_pe_num_freqs, view_transformer_latent_dim // view_transformer_n_heads // 18 * 2)
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
            camera_o (torch.Tensor): (B, 3)
            ray_map (torch.Tensor): (B, H, W, 3)
        Returns:
            ray_tokens: (B, ???)
            # decoded_img: (B, 3, H, W)
        """

        # query sequence
        ray_map = self.vdir_pe(ray_map)
        ray_tokens = rearrange(ray_map, 'b (h1 p1) (w1 p2) c -> b (h1 w1) (c p1 p2)', p1=self.patch_size, p2=self.patch_size)
        patch_h = ray_map.size(1) // self.patch_size
        patch_w = ray_map.size(2) // self.patch_size
        ray_tokens = self.ray_map_patch_token + self.ray_map_encoder_norm(self.ray_map_encoder(ray_tokens))  # [B, N_PATCHES, D]
        n_patches = ray_tokens.size(1)
        ray_token_pos = camera_o[:, None].repeat(1, n_patches, 3)  # [B, N_PATCHES, 3]

        # positional encoding if use 'nerf' pe
        if self.pe_type == 'nerf':
            ray_tokens = ray_tokens + self.token_pos_pe_norm(self.pe_token_proj(self.pos_pe(ray_token_pos)))

        return ray_tokens
    

if __name__ == "__main__":
    print("Running RayEncoder sanity checks...")
    
    # Test parameters
    batch_size = 2
    height = 64
    width = 64
    patch_size = 16
    latent_dim = 768
    n_heads = 4
    
    # Test 1: NeRF PE initialization
    print("\n[Test 1] Testing NeRF PE initialization...")
    try:
        encoder_nerf = RayEncoder(
            pe_type='nerf',
            vertex_pe_num_freqs=12,
            vdir_pe_type='nerf',
            vdir_num_freqs=4,
            patch_size=patch_size,
            norm_type='rms_norm',
            view_transformer_latent_dim=latent_dim,
            view_transformer_n_heads=n_heads
        )
        print("✓ NeRF PE encoder initialized successfully")
    except Exception as e:
        print(f"✗ NeRF PE initialization failed: {e}")
        
    # Test 2: RoPE initialization
    print("\n[Test 2] Testing RoPE initialization...")
    try:
        encoder_rope = RayEncoder(
            pe_type='rope',
            vertex_pe_num_freqs=12,
            vdir_pe_type='nerf',
            vdir_num_freqs=4,
            patch_size=patch_size,
            norm_type='layer_norm',
            view_transformer_latent_dim=latent_dim,
            view_transformer_n_heads=n_heads
        )
        print(f"✓ RoPE encoder initialized successfully (rope_dim={encoder_rope.rope_dim})")
    except Exception as e:
        print(f"✗ RoPE initialization failed: {e}")
    
    # Test 3: Forward pass with NeRF PE
    print("\n[Test 3] Testing forward pass with NeRF PE...")
    try:
        camera_o = torch.randn(batch_size, 3)
        ray_map = torch.randn(batch_size, height, width, 3)
        
        ray_tokens = encoder_nerf(camera_o, ray_map)
        
        expected_n_patches = (height // patch_size) * (width // patch_size)
        assert ray_tokens.shape == (batch_size, expected_n_patches, latent_dim), \
            f"Expected shape ({batch_size}, {expected_n_patches}, {latent_dim}), got {ray_tokens.shape}"
        
        print(f"✓ Forward pass successful")
        print(f"  Input: camera_o={camera_o.shape}, ray_map={ray_map.shape}")
        print(f"  Output: ray_tokens={ray_tokens.shape}")
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
    
    # Test 4: Forward pass with RoPE
    print("\n[Test 4] Testing forward pass with RoPE...")
    try:
        ray_tokens = encoder_rope(camera_o, ray_map)
        
        assert ray_tokens.shape == (batch_size, expected_n_patches, latent_dim), \
            f"Expected shape ({batch_size}, {expected_n_patches}, {latent_dim}), got {ray_tokens.shape}"
        
        print(f"✓ Forward pass with RoPE successful")
        print(f"  Output: ray_tokens={ray_tokens.shape}")
    except Exception as e:
        print(f"✗ Forward pass with RoPE failed: {e}")
    
    # Test 5: Test different image sizes
    print("\n[Test 5] Testing different image sizes...")
    test_sizes = [(32, 32), (48, 64), (128, 128)]
    for h, w in test_sizes:
        try:
            ray_map_test = torch.randn(batch_size, h, w, 3)
            ray_tokens_test = encoder_nerf(camera_o, ray_map_test)
            expected_patches = (h // patch_size) * (w // patch_size)
            assert ray_tokens_test.shape[1] == expected_patches, \
                f"For size ({h}, {w}), expected {expected_patches} patches, got {ray_tokens_test.shape[1]}"
            print(f"  ✓ Size ({h}, {w}): {ray_tokens_test.shape}")
        except Exception as e:
            print(f"  ✗ Size ({h}, {w}) failed: {e}")
    
    # Test 6: Check gradient flow
    print("\n[Test 6] Testing gradient flow...")
    try:
        encoder_nerf.zero_grad()
        camera_o_grad = torch.randn(batch_size, 3, requires_grad=True)
        ray_map_grad = torch.randn(batch_size, height, width, 3, requires_grad=True)
        
        ray_tokens = encoder_nerf(camera_o_grad, ray_map_grad)
        loss = ray_tokens.sum()
        loss.backward()
        
        assert ray_map_grad.grad is not None, "No gradient for ray_map"
        # Note: camera_o gradient may be None for RoPE since it's not used in NeRF PE forward
        
        print(f"✓ Gradients computed successfully")
        print(f"  ray_map grad shape: {ray_map_grad.grad.shape}")
    except Exception as e:
        print(f"✗ Gradient flow test failed: {e}")
    
    # Test 7: Check parameter count
    print("\n[Test 7] Parameter count...")
    total_params = sum(p.numel() for p in encoder_nerf.parameters())
    trainable_params = sum(p.numel() for p in encoder_nerf.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    print("\n" + "="*50)
    print("Sanity checks complete!")