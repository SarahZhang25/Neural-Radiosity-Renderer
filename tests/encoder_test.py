"""
Unit tests for PointNetEncoder module.
Run with: 
pytest tests/encoder_test.py -v
"""

import pytest
import torch
import torch.nn as nn
from model.encoder import PointNetEncoder, ObjectPropertyEncoder


class TestObjectPropertyEncoder:
    """Test suite for ObjectPropertyEncoder."""
    
    @pytest.fixture
    def batch_size(self):
        return 4
    
    @pytest.fixture
    def modifier_encoder(self):
        return ObjectPropertyEncoder(
            input_dim=3,
            hidden_dim=128,
            output_dim=512,
            object_type='modifier',
            use_batch_norm=True
        )
    
    @pytest.fixture
    def emitter_encoder(self):
        return ObjectPropertyEncoder(
            input_dim=3,
            hidden_dim=128,
            output_dim=512,
            object_type='emitter',
            emitter_type_dim=16,
            use_batch_norm=True
        )
    
    def test_modifier_encoder_init(self, modifier_encoder):
        """Test modifier encoder initialization."""
        assert modifier_encoder.object_type == 'modifier'
        assert modifier_encoder.input_dim == 3
        assert modifier_encoder.hidden_dim == 128
    
    def test_emitter_encoder_init(self, emitter_encoder):
        """Test emitter encoder initialization."""
        assert emitter_encoder.object_type == 'emitter'
        assert hasattr(emitter_encoder, 'emitter_type_embed')
        assert emitter_encoder.emitter_type_embed.num_embeddings == 4
    
    def test_modifier_forward_shape(self, modifier_encoder, batch_size):
        """Test modifier encoder forward pass shape."""
        properties = torch.rand(batch_size, 3)
        output = modifier_encoder(properties)
        assert output.shape == (batch_size, 512)
    
    def test_emitter_forward_shape(self, emitter_encoder, batch_size):
        """Test emitter encoder forward pass shape."""
        properties = torch.rand(batch_size, 3)
        emitter_types = torch.randint(0, 4, (batch_size,))
        output = emitter_encoder(properties, emitter_types)
        assert output.shape == (batch_size, 512)
    
    def test_emitter_without_types_fails(self, emitter_encoder, batch_size):
        """Test that emitter encoder fails without type information."""
        properties = torch.rand(batch_size, 3)
        with pytest.raises((TypeError, AttributeError)):
            output = emitter_encoder(properties, None)


class TestPointNetEncoder:
    """Test suite for PointNetEncoder."""
    
    @pytest.fixture
    def batch_size(self):
        return 4
    
    @pytest.fixture
    def num_points(self):
        return 1024
    
    @pytest.fixture
    def output_dim(self):
        return 512
    
    @pytest.fixture
    def hierarchical_encoder(self):
        return PointNetEncoder(
            input_dim=6,  # pos + normals
            hidden_dims=[2048, 2048],
            output_dim=512,
            backbone_dim=800,
            use_batch_norm=True,
            pooling_type='hierarchical',
            num_hierarchical_levels=5,
            property_encoder_hidden_dim=128,
            emitter_type_dim=16
        )
    
    @pytest.fixture
    def maxpool_encoder(self):
        return PointNetEncoder(
            input_dim=6,
            hidden_dims=[1024, 512],
            output_dim=512,
            backbone_dim=800,
            use_batch_norm=False,
            pooling_type='max',
            property_encoder_hidden_dim=128,
            emitter_type_dim=16
        )
    
    @pytest.fixture
    def no_normals_encoder(self):
        return PointNetEncoder(
            input_dim=3,  # Only positions
            hidden_dims=[512, 256],
            output_dim=512,
            backbone_dim=400,
            use_batch_norm=True,
            pooling_type='max'
        )
    
    def test_hierarchical_encoder_init(self, hierarchical_encoder):
        """Test hierarchical encoder initialization."""
        assert hierarchical_encoder.pooling_type == 'hierarchical'
        assert hasattr(hierarchical_encoder, 'hierarchical_layers')
        assert len(hierarchical_encoder.hierarchical_layers) == 5
        assert hasattr(hierarchical_encoder, 'hierarchical_proj')
    
    def test_maxpool_encoder_init(self, maxpool_encoder):
        """Test max pooling encoder initialization."""
        assert maxpool_encoder.pooling_type == 'max'
        assert not hasattr(maxpool_encoder, 'hierarchical_layers')
    
    def test_forward_modifier_with_normals(self, hierarchical_encoder, batch_size, num_points, output_dim):
        """Test forward pass with modifier and normals."""
        surface_pos = torch.randn(batch_size, num_points, 3)
        normals = torch.nn.functional.normalize(torch.randn(batch_size, num_points, 3), dim=-1)
        properties = torch.rand(batch_size, 3)
        
        output = hierarchical_encoder(
            surface_pos=surface_pos,
            normals=normals,
            properties=properties,
            object_type='modifier'
        )
        
        assert output.shape == (batch_size, output_dim)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_forward_emitter(self, hierarchical_encoder, batch_size, num_points, output_dim):
        """Test forward pass with emitter."""
        surface_pos = torch.randn(batch_size, num_points, 3)
        normals = torch.nn.functional.normalize(torch.randn(batch_size, num_points, 3), dim=-1)
        properties = torch.rand(batch_size, 3)
        emitter_types = torch.randint(0, 4, (batch_size,))
        
        output = hierarchical_encoder(
            surface_pos=surface_pos,
            normals=normals,
            properties=properties,
            object_type='emitter',
            emitter_types=emitter_types
        )
        
        assert output.shape == (batch_size, output_dim)
    
    def test_forward_without_normals(self, no_normals_encoder, batch_size, num_points, output_dim):
        """Test forward pass without normals."""
        surface_pos = torch.randn(batch_size, num_points, 3)
        properties = torch.rand(batch_size, 3)
        
        output = no_normals_encoder(
            surface_pos=surface_pos,
            normals=None,
            properties=properties,
            object_type='modifier'
        )
        
        assert output.shape == (batch_size, output_dim)
    
    @pytest.mark.parametrize("n_points", [256, 512, 2048, 4096])
    def test_variable_point_cloud_sizes(self, hierarchical_encoder, batch_size, output_dim, n_points):
        """Test encoder with different point cloud sizes."""
        surface_pos = torch.randn(batch_size, n_points, 3)
        normals = torch.nn.functional.normalize(torch.randn(batch_size, n_points, 3), dim=-1)
        properties = torch.rand(batch_size, 3)
        
        output = hierarchical_encoder(surface_pos, normals, properties, 'modifier')
        assert output.shape == (batch_size, output_dim)
    
    def test_maxpool_vs_hierarchical(self, maxpool_encoder, hierarchical_encoder, batch_size, num_points, output_dim):
        """Test that both pooling methods produce correct output shape."""
        surface_pos = torch.randn(batch_size, num_points, 3)
        normals = torch.nn.functional.normalize(torch.randn(batch_size, num_points, 3), dim=-1)
        properties = torch.rand(batch_size, 3)
        
        out_max = maxpool_encoder(surface_pos, normals, properties, 'modifier')
        out_hier = hierarchical_encoder(surface_pos, normals, properties, 'modifier')
        
        assert out_max.shape == (batch_size, output_dim)
        assert out_hier.shape == (batch_size, output_dim)
    
    def test_gradient_flow(self, hierarchical_encoder, batch_size, num_points):
        """Test that gradients flow correctly through the network."""
        hierarchical_encoder.zero_grad()
        
        surface_pos = torch.randn(batch_size, num_points, 3, requires_grad=True)
        normals = torch.nn.functional.normalize(
            torch.randn(batch_size, num_points, 3, requires_grad=True), dim=-1
        )
        properties = torch.rand(batch_size, 3, requires_grad=True)
        
        output = hierarchical_encoder(surface_pos, normals, properties, 'modifier')
        loss = output.sum()
        loss.backward()
        
        assert surface_pos.grad is not None
        assert properties.grad is not None
        assert not torch.isnan(surface_pos.grad).any()
        assert not torch.isnan(properties.grad).any()
    
    def test_parameter_count(self, hierarchical_encoder):
        """Test parameter counting and verify all parameters are trainable."""
        total_params = sum(p.numel() for p in hierarchical_encoder.parameters())
        trainable_params = sum(p.numel() for p in hierarchical_encoder.parameters() if p.requires_grad)
        
        assert total_params > 0
        assert trainable_params == total_params  # All params should be trainable
        
        # Verify all submodules have parameters
        assert sum(p.numel() for p in hierarchical_encoder.point_net.parameters()) > 0
        assert sum(p.numel() for p in hierarchical_encoder.modifier_property_encoder.parameters()) > 0
        assert sum(p.numel() for p in hierarchical_encoder.emitter_property_encoder.parameters()) > 0
        assert sum(p.numel() for p in hierarchical_encoder.fusion_mlp.parameters()) > 0
    
    def test_invalid_object_type(self, hierarchical_encoder, batch_size, num_points):
        """Test that invalid object type raises ValueError."""
        surface_pos = torch.randn(batch_size, num_points, 3)
        normals = torch.nn.functional.normalize(torch.randn(batch_size, num_points, 3), dim=-1)
        properties = torch.rand(batch_size, 3)
        
        with pytest.raises(ValueError, match="Unknown object_type"):
            hierarchical_encoder(surface_pos, normals, properties, 'invalid_type')
    
    def test_invalid_pooling_type(self):
        """Test that invalid pooling type raises ValueError during forward pass."""
        encoder = PointNetEncoder(
            input_dim=6,
            hidden_dims=[512],
            output_dim=512,
            backbone_dim=400,
            pooling_type='hierarchical'  # Valid init
        )
        # Manually change to invalid type
        encoder.pooling_type = 'invalid'
        
        surface_pos = torch.randn(2, 100, 3)
        normals = torch.randn(2, 100, 3)
        properties = torch.rand(2, 3)
        
        with pytest.raises(ValueError, match="Unknown pooling type"):
            encoder(surface_pos, normals, properties, 'modifier')
    
    def test_batch_size_one_eval(self, hierarchical_encoder, num_points, output_dim):
        """Test edge case with batch size of 1."""
        hierarchical_encoder.eval()  # Set to eval mode to use running stats

        surface_pos = torch.randn(1, num_points, 3)
        normals = torch.nn.functional.normalize(torch.randn(1, num_points, 3), dim=-1)
        properties = torch.rand(1, 3)
        
        with torch.no_grad():
            output = hierarchical_encoder(surface_pos, normals, properties, 'modifier')
    
        assert output.shape == (1, output_dim)
    
    def test_deterministic_output(self, hierarchical_encoder, batch_size, num_points):
        """Test that same input produces same output in eval mode."""
        hierarchical_encoder.eval()
        
        surface_pos = torch.randn(batch_size, num_points, 3)
        normals = torch.nn.functional.normalize(torch.randn(batch_size, num_points, 3), dim=-1)
        properties = torch.rand(batch_size, 3)
        
        with torch.no_grad():
            output1 = hierarchical_encoder(surface_pos, normals, properties, 'modifier')
            output2 = hierarchical_encoder(surface_pos, normals, properties, 'modifier')
        
        assert torch.allclose(output1, output2, atol=1e-6)
    
    def test_different_batch_norm_modes(self, batch_size, num_points, output_dim):
        """Test that batch norm and no batch norm both work."""
        encoders = [
            PointNetEncoder(input_dim=6, hidden_dims=[512], output_dim=512, 
                          backbone_dim=400, use_batch_norm=True, pooling_type='max'),
            PointNetEncoder(input_dim=6, hidden_dims=[512], output_dim=512,
                          backbone_dim=400, use_batch_norm=False, pooling_type='max')
        ]
        
        surface_pos = torch.randn(batch_size, num_points, 3)
        normals = torch.nn.functional.normalize(torch.randn(batch_size, num_points, 3), dim=-1)
        properties = torch.rand(batch_size, 3)
        
        for encoder in encoders:
            output = encoder(surface_pos, normals, properties, 'modifier')
            assert output.shape == (batch_size, output_dim)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])