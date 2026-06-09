"""
Unit tests for PointNetEncoder module.
Run with: 
pytest tests/encoder_test.py -v
"""

import pytest
import torch
import torch.nn as nn
from model.encoder import PointNetEncoder, UniversalPropertyEncoder, ObjectClass


class TestUniversalPropertyEncoder:
    """Test suite for UniversalPropertyEncoder."""
    
    @pytest.fixture
    def batch_size(self):
        return 4
    
    @pytest.fixture
    def encoder(self):
        return UniversalPropertyEncoder(
            prop_dim=3,
            num_classes=20,
            embed_dim=512,
            type_embed_dim=64,
            use_batch_norm=True
        )
    
    def test_encoder_init(self, encoder):
        """Test encoder initialization."""
        assert isinstance(encoder.type_embedding, nn.Embedding)
        assert encoder.type_embedding.num_embeddings == 20
        assert encoder.type_embedding.embedding_dim == 64
        
    def test_forward_shape(self, encoder, batch_size):
        """Test encoder forward pass shape."""
        properties = torch.rand(batch_size, 3)
        # Create random class IDs within range
        class_ids = torch.randint(0, 20, (batch_size,))
        
        output = encoder(properties, class_ids)
        assert output.shape == (batch_size, 512)
        
    def test_forward_mixed_types(self, encoder, batch_size):
        """Test that encoder handles mixed object types in one batch."""
        properties = torch.rand(batch_size, 3)
        # Mix modifiers (0) and emitters (10)
        class_ids = torch.tensor([ObjectClass.MODIFIER_DIFFUSE, ObjectClass.EMITTER_POINT, 
                                  ObjectClass.MODIFIER_SPECULAR, ObjectClass.EMITTER_SUN][:batch_size])
        
        # Should run without error
        output = encoder(properties, class_ids)
        assert output.shape == (batch_size, 512)


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
            # These might be unused in new impl but kept for compatibility if needed
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
            pooling_type='max'
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
        assert isinstance(hierarchical_encoder.property_encoder, UniversalPropertyEncoder)
    
    def test_forward_modifier_with_normals(self, hierarchical_encoder, batch_size, num_points, output_dim):
        """Test forward pass with modifier and normals."""
        surface_pos = torch.randn(batch_size, num_points, 3)
        normals = torch.nn.functional.normalize(torch.randn(batch_size, num_points, 3), dim=-1)
        properties = torch.rand(batch_size, 3)
        class_ids = torch.full((batch_size,), int(ObjectClass.MODIFIER_DIFFUSE), dtype=torch.long)
        
        output = hierarchical_encoder(
            surface_pos=surface_pos,
            normals=normals,
            properties=properties,
            object_class_ids=class_ids
        )
        
        assert output.shape == (batch_size, output_dim)
        assert not torch.isnan(output).any()
    
    def test_forward_emitter(self, hierarchical_encoder, batch_size, num_points, output_dim):
        """Test forward pass with emitter."""
        surface_pos = torch.randn(batch_size, num_points, 3)
        normals = torch.nn.functional.normalize(torch.randn(batch_size, num_points, 3), dim=-1)
        properties = torch.rand(batch_size, 3)
        class_ids = torch.full((batch_size,), int(ObjectClass.EMITTER_POINT), dtype=torch.long)
        
        output = hierarchical_encoder(
            surface_pos=surface_pos,
            normals=normals,
            properties=properties,
            object_class_ids=class_ids
        )
        
        assert output.shape == (batch_size, output_dim)
    
    def test_forward_mixed_batch(self, hierarchical_encoder, batch_size, num_points, output_dim):
        """Test forward pass with mix of modifiers and emitters."""
        surface_pos = torch.randn(batch_size, num_points, 3)
        normals = torch.nn.functional.normalize(torch.randn(batch_size, num_points, 3), dim=-1)
        properties = torch.rand(batch_size, 3)
        
        # Mix types: [Modifier, Emitter, Modifier, Emitter]
        class_ids = torch.tensor([
            ObjectClass.MODIFIER_DIFFUSE, 
            ObjectClass.EMITTER_AREA,
            ObjectClass.MODIFIER_SPECULAR,
            ObjectClass.EMITTER_SPOT if hasattr(ObjectClass, 'EMITTER_SPOT') else 12
        ][:batch_size], dtype=torch.long)
        
        output = hierarchical_encoder(
            surface_pos=surface_pos,
            normals=normals,
            properties=properties,
            object_class_ids=class_ids
        )
        
        assert output.shape == (batch_size, output_dim)

    def test_forward_without_normals(self, no_normals_encoder, batch_size, num_points, output_dim):
        """Test forward pass without normals."""
        surface_pos = torch.randn(batch_size, num_points, 3)
        properties = torch.rand(batch_size, 3)
        class_ids = torch.zeros(batch_size, dtype=torch.long)
        
        output = no_normals_encoder(
            surface_pos=surface_pos,
            normals=None,
            properties=properties,
            object_class_ids=class_ids
        )
        
        assert output.shape == (batch_size, output_dim)
    
    def test_variable_point_cloud_sizes(self, hierarchical_encoder, batch_size, output_dim):
        """Test encoder with different point cloud sizes."""
        for n_points in [256, 512, 2048]:
            surface_pos = torch.randn(batch_size, n_points, 3)
            normals = torch.nn.functional.normalize(torch.randn(batch_size, n_points, 3), dim=-1)
            properties = torch.rand(batch_size, 3)
            class_ids = torch.zeros(batch_size, dtype=torch.long)
            
            output = hierarchical_encoder(
                surface_pos=surface_pos, 
                normals=normals, 
                properties=properties, 
                object_class_ids=class_ids
            )
            assert output.shape == (batch_size, output_dim)

    def test_gradient_flow(self, hierarchical_encoder, batch_size, num_points):
        """Test that gradients flow correctly through the network."""
        hierarchical_encoder.zero_grad()
        
        surface_pos = torch.randn(batch_size, num_points, 3, requires_grad=True)
        normals = torch.nn.functional.normalize(
            torch.randn(batch_size, num_points, 3, requires_grad=True), dim=-1
        )
        properties = torch.rand(batch_size, 3, requires_grad=True)
        class_ids = torch.zeros(batch_size, dtype=torch.long)
        
        output = hierarchical_encoder(
            surface_pos=surface_pos, 
            normals=normals, 
            properties=properties, 
            object_class_ids=class_ids
        )
        loss = output.sum()
        loss.backward()
        
        assert surface_pos.grad is not None
        assert properties.grad is not None
        assert not torch.isnan(surface_pos.grad).any()
        assert not torch.isnan(properties.grad).any()
    
    def test_invalid_class_id(self, hierarchical_encoder, batch_size, num_points):
        """Test that invalid class ID raises IndexError (from Embedding layer)."""
        surface_pos = torch.randn(batch_size, num_points, 3)
        normals = torch.randn(batch_size, num_points, 3)
        properties = torch.rand(batch_size, 3)
        # Class 100 is out of bounds (default vocab is 20)
        class_ids = torch.full((batch_size,), 100, dtype=torch.long)
        
        with pytest.raises(IndexError):
            hierarchical_encoder(
                surface_pos=surface_pos, 
                normals=normals, 
                properties=properties, 
                object_class_ids=class_ids
            )
    
    def test_invalid_pooling_type(self):
        """Test that invalid pooling type raises ValueError."""
        encoder = PointNetEncoder(
            input_dim=6,
            hidden_dims=[512],
            output_dim=512,
            backbone_dim=400,
            pooling_type='hierarchical'
        )
        encoder.pooling_type = 'invalid'
        
        surface_pos = torch.randn(2, 100, 3)
        normals = torch.randn(2, 100, 3)
        properties = torch.rand(2, 3)
        class_ids = torch.zeros(2, dtype=torch.long)
        
        with pytest.raises(ValueError, match="Unknown pooling type"):
            encoder(surface_pos, properties, class_ids, normals)

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])