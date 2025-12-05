"""
Minimal sanity tests for RadiancePredictor.
Run with: pytest tests/test_predictor.py -v
"""

import pytest
import torch
from model.predictor import RadiancePredictor


class TestRadiancePredictor:
    """Minimal test suite for RadiancePredictor."""
    
    @pytest.fixture
    def predictor(self):
        return RadiancePredictor(
            hidden_dim=512,
            patch_size=16,
            num_heads=8,
            num_layers=4,
            dropout=0.1,
            activation='gelu',
            norm_type='layer_norm',
            include_self_attn=True
        )
    
    @pytest.fixture
    def small_predictor(self):
        """Smaller predictor for faster tests."""
        return RadiancePredictor(
            hidden_dim=256,
            patch_size=8,
            num_heads=4,
            num_layers=2
        )
    
    def test_forward_with_state_features(self, predictor):
        """Test forward pass with both object and state features."""
        B, N_obj, N_state, D = 2, 100, 10, 512
        patch_h, patch_w = 16, 16
        N_patches = patch_h * patch_w
        
        query_view_features = torch.randn(B, N_patches, D)
        multi_scale_features = [torch.randn(B, N_obj, D) for _ in range(3)]
        multi_scale_state_features = [torch.randn(B, N_state, D) for _ in range(3)]
        
        output = predictor(
            multi_scale_features=multi_scale_features,
            query_view_features=query_view_features,
            multi_scale_state_features=multi_scale_state_features,
            patch_h=patch_h,
            patch_w=patch_w
        )
        
        expected_h = patch_h * 16  # patch_size = 16
        expected_w = patch_w * 16
        assert output.shape == (B, 3, expected_h, expected_w)
    
    def test_forward_without_state_features(self, predictor):
        """Test forward pass with only object features."""
        B, N_obj, D = 2, 100, 512
        patch_h, patch_w = 8, 8
        N_patches = patch_h * patch_w
        
        query_view_features = torch.randn(B, N_patches, D)
        multi_scale_features = [torch.randn(B, N_obj, D) for _ in range(3)]
        
        output = predictor(
            multi_scale_features=multi_scale_features,
            query_view_features=query_view_features,
            multi_scale_state_features=None,
            patch_h=patch_h,
            patch_w=patch_w
        )
        
        assert output.shape == (B, 3, 128, 128)  # 8*16 = 128
    
    def test_auto_infer_patch_dimensions(self, predictor):
        """Test automatic inference of patch_h and patch_w."""
        B, N_obj, D = 2, 100, 512
        N_patches = 64  # 8x8 grid
        
        query_view_features = torch.randn(B, N_patches, D)
        multi_scale_features = [torch.randn(B, N_obj, D) for _ in range(3)]
        
        output = predictor(
            multi_scale_features=multi_scale_features,
            query_view_features=query_view_features,
            multi_scale_state_features=None
        )
        
        # Should infer 8x8 patch grid
        assert output.shape == (B, 3, 128, 128)
    
    @pytest.mark.parametrize("patch_h,patch_w", [(4, 4), (8, 16), (16, 16)])
    def test_variable_patch_grids(self, small_predictor, patch_h, patch_w):
        """Test different patch grid configurations."""
        B, N_obj, D = 2, 50, 256
        N_patches = patch_h * patch_w
        
        query_view_features = torch.randn(B, N_patches, D)
        multi_scale_features = [torch.randn(B, N_obj, D) for _ in range(3)]
        
        output = small_predictor(
            multi_scale_features=multi_scale_features,
            query_view_features=query_view_features,
            patch_h=patch_h,
            patch_w=patch_w
        )
        
        expected_h = patch_h * 8  # patch_size = 8 for small_predictor
        expected_w = patch_w * 8
        assert output.shape == (B, 3, expected_h, expected_w)
    
    def test_gradient_flow(self, predictor):
        """Test gradients backpropagate correctly."""
        B, N_obj, D = 2, 100, 512
        N_patches = 16
        
        query_view_features = torch.randn(B, N_patches, D, requires_grad=True)
        multi_scale_features = [
            torch.randn(B, N_obj, D, requires_grad=True) for _ in range(3)
        ]
        
        output = predictor(
            multi_scale_features=multi_scale_features,
            query_view_features=query_view_features,
            patch_h=4,
            patch_w=4
        )
        
        loss = output.sum()
        loss.backward()
        
        assert query_view_features.grad is not None
        assert not torch.isnan(query_view_features.grad).any()
        for feat in multi_scale_features:
            assert feat.grad is not None
            assert not torch.isnan(feat.grad).any()
    
    def test_layer_weights_learnable(self, predictor):
        """Test that layer weights are learnable parameters."""
        assert predictor.layer_weights.requires_grad
        assert predictor.layer_weights.shape == (3,)
    
    def test_output_range_reasonable(self, predictor):
        """Test output values are in reasonable range (not exploding)."""
        B, N_obj, D = 1, 50, 512
        N_patches = 16
        
        query_view_features = torch.randn(B, N_patches, D)
        multi_scale_features = [torch.randn(B, N_obj, D) for _ in range(3)]
        
        predictor.eval()
        with torch.no_grad():
            output = predictor(
                multi_scale_features=multi_scale_features,
                query_view_features=query_view_features,
                patch_h=4,
                patch_w=4
            )
        
        # Check for reasonable output range (should be small due to xavier init)
        assert output.abs().max() < 100.0
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])