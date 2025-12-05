"""
Minimal sanity tests for RayEncoder.
Run with: pytest tests/test_ray_encoder.py -v
"""

import pytest
import torch
from model.ray_encoder import RayEncoder


class TestRayEncoder:
    """Minimal test suite for RayEncoder."""
    
    @pytest.fixture
    def encoder_nerf(self):
        return RayEncoder(
            pe_type='nerf',
            vertex_pe_num_freqs=12,
            vdir_pe_type='nerf',
            vdir_num_freqs=4,
            patch_size=16,
            norm_type='rms_norm',
            view_transformer_latent_dim=768,
            view_transformer_n_heads=4
        )
    
    @pytest.fixture
    def encoder_rope(self):
        return RayEncoder(
            pe_type='rope',
            vertex_pe_num_freqs=12,
            vdir_pe_type='nerf',
            vdir_num_freqs=4,
            patch_size=16,
            norm_type='layer_norm',
            view_transformer_latent_dim=768,
            view_transformer_n_heads=4
        )
    
    def test_nerf_forward_shape(self, encoder_nerf):
        """Test NeRF PE forward pass produces correct shape."""
        batch_size, height, width = 2, 64, 64
        camera_o = torch.randn(batch_size, 3)
        ray_map = torch.randn(batch_size, height, width, 3)
        
        ray_tokens = encoder_nerf(camera_o, ray_map)
        
        expected_patches = (height // 16) * (width // 16)
        assert ray_tokens.shape == (batch_size, expected_patches, 768)
    
    def test_rope_forward_shape(self, encoder_rope):
        """Test RoPE forward pass produces correct shape."""
        batch_size, height, width = 2, 64, 64
        camera_o = torch.randn(batch_size, 3)
        ray_map = torch.randn(batch_size, height, width, 3)
        
        ray_tokens = encoder_rope(camera_o, ray_map)
        
        expected_patches = (height // 16) * (width // 16)
        assert ray_tokens.shape == (batch_size, expected_patches, 768)
    
    @pytest.mark.parametrize("height,width", [(32, 32), (64, 48), (128, 128)])
    def test_variable_image_sizes(self, encoder_nerf, height, width):
        """Test different image sizes."""
        camera_o = torch.randn(2, 3)
        ray_map = torch.randn(2, height, width, 3)
        
        ray_tokens = encoder_nerf(camera_o, ray_map)
        expected_patches = (height // 16) * (width // 16)
        assert ray_tokens.shape[1] == expected_patches
    
    def test_gradient_flow(self, encoder_nerf):
        """Test gradients backpropagate correctly."""
        camera_o = torch.randn(2, 3, requires_grad=True)
        ray_map = torch.randn(2, 64, 64, 3, requires_grad=True)
        
        ray_tokens = encoder_nerf(camera_o, ray_map)
        loss = ray_tokens.sum()
        loss.backward()
        
        assert ray_map.grad is not None
        assert not torch.isnan(ray_map.grad).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])