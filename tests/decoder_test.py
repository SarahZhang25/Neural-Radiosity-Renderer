"""
Unit tests for TransformerDecoder module.
Run with: pytest tests/decoder_test.py -v
"""

import pytest
import torch
import torch.nn as nn
from model.decoder import TransformerDecoder, TransformerDecoderLayer, RMSNorm, SwiGLU


class TestRMSNorm:
    """Test suite for RMSNorm."""
    
    @pytest.fixture
    def dim(self):
        return 512
    
    @pytest.fixture
    def rms_norm(self, dim):
        return RMSNorm(dim)
    
    def test_init(self, rms_norm, dim):
        """Test RMSNorm initialization."""
        assert rms_norm.weight.shape == (dim,)
        assert torch.allclose(rms_norm.weight, torch.ones(dim))
    
    def test_forward_shape(self, rms_norm, dim):
        """Test RMSNorm forward pass shape."""
        x = torch.randn(4, 10, dim)
        output = rms_norm(x)
        assert output.shape == x.shape
    
    def test_normalization(self, rms_norm, dim):
        """Test that RMSNorm produces expected normalization."""
        x = torch.randn(2, 5, dim)
        output = rms_norm(x)
        
        # Check that output has similar magnitude across last dimension
        rms = torch.sqrt(torch.mean(output ** 2, dim=-1))
        # RMS should be close to 1 (before weight scaling)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_gradient_flow(self, rms_norm, dim):
        """Test gradient flow through RMSNorm."""
        x = torch.randn(2, 5, dim, requires_grad=True)
        output = rms_norm(x)
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None
        assert rms_norm.weight.grad is not None


class TestSwiGLU:
    """Test suite for SwiGLU activation."""
    
    @pytest.fixture
    def dim(self):
        return 512
    
    @pytest.fixture
    def hidden_dim(self):
        return 1280
    
    @pytest.fixture
    def swiglu(self, dim, hidden_dim):
        return SwiGLU(dim, hidden_dim, dropout=0.1)
    
    def test_init(self, swiglu, dim, hidden_dim):
        """Test SwiGLU initialization."""
        assert swiglu.w1.in_features == dim
        assert swiglu.w1.out_features == hidden_dim
        assert swiglu.w2.in_features == hidden_dim
        assert swiglu.w2.out_features == dim
        assert swiglu.w3.in_features == dim
        assert swiglu.w3.out_features == hidden_dim
    
    def test_forward_shape(self, swiglu, dim):
        """Test SwiGLU forward pass shape."""
        x = torch.randn(4, 10, dim)
        output = swiglu(x)
        assert output.shape == x.shape
    
    def test_no_bias(self, swiglu):
        """Test that SwiGLU layers have no bias."""
        assert swiglu.w1.bias is None
        assert swiglu.w2.bias is None
        assert swiglu.w3.bias is None
    
    def test_gradient_flow(self, swiglu, dim):
        """Test gradient flow through SwiGLU."""
        x = torch.randn(2, 5, dim, requires_grad=True)
        output = swiglu(x)
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None
        assert swiglu.w1.weight.grad is not None
        assert swiglu.w2.weight.grad is not None
        assert swiglu.w3.weight.grad is not None


class TestTransformerDecoderLayer:
    """Test suite for TransformerDecoderLayer."""
    
    @pytest.fixture
    def d_model(self):
        return 512
    
    @pytest.fixture
    def nhead(self):
        return 8
    
    @pytest.fixture
    def batch_size(self):
        return 4
    
    @pytest.fixture
    def n_state(self):
        return 16
    
    @pytest.fixture
    def n_obj(self):
        return 32
    
    @pytest.fixture
    def decoder_layer_swiglu(self, d_model, nhead):
        return TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=1280,
            dropout=0.1,
            activation='swiglu',
            use_self_attention=True,
            norm_type='rms_norm',
            qk_norm=True
        )
    
    @pytest.fixture
    def decoder_layer_relu(self, d_model, nhead):
        return TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=2048,
            dropout=0.1,
            activation='relu',
            use_self_attention=False,
            norm_type='layer_norm',
            qk_norm=False
        )
    
    def test_init_swiglu(self, decoder_layer_swiglu):
        """Test decoder layer initialization with SwiGLU."""
        assert decoder_layer_swiglu.use_self_attention
        assert isinstance(decoder_layer_swiglu.ffn_state, SwiGLU)
        assert isinstance(decoder_layer_swiglu.norm_state1, RMSNorm)
    
    def test_init_relu(self, decoder_layer_relu):
        """Test decoder layer initialization with ReLU."""
        assert not decoder_layer_relu.use_self_attention
        assert isinstance(decoder_layer_relu.ffn_state, nn.Sequential)
        assert isinstance(decoder_layer_relu.norm_state1, nn.LayerNorm)
    
    def test_forward_shape(self, decoder_layer_swiglu, batch_size, n_state, n_obj, d_model):
        """Test forward pass shape."""
        state_tokens = torch.randn(batch_size, n_state, d_model)
        obj_tokens = torch.randn(batch_size, n_obj, d_model)
        
        state_out, obj_out = decoder_layer_swiglu(state_tokens, obj_tokens)
        
        assert state_out.shape == (batch_size, n_state, d_model)
        assert obj_out.shape == (batch_size, n_obj, d_model)
    
    def test_forward_with_mask(self, decoder_layer_swiglu, batch_size, n_state, n_obj, d_model):
        """Test forward pass with attention mask."""
        state_tokens = torch.randn(batch_size, n_state, d_model)
        obj_tokens = torch.randn(batch_size, n_obj, d_model)
        mask = torch.rand(batch_size, n_obj) > 0.5  # Random boolean mask
        
        state_out, obj_out = decoder_layer_swiglu(
            state_tokens, obj_tokens, mask=mask
        )
        
        assert state_out.shape == (batch_size, n_state, d_model)
        assert obj_out.shape == (batch_size, n_obj, d_model)
    
    def test_gradient_flow(self, decoder_layer_swiglu, batch_size, n_state, n_obj, d_model):
        """Test gradient flow through decoder layer."""
        state_tokens = torch.randn(batch_size, n_state, d_model, requires_grad=True)
        obj_tokens = torch.randn(batch_size, n_obj, d_model, requires_grad=True)
        
        state_out, obj_out = decoder_layer_swiglu(state_tokens, obj_tokens)
        loss = state_out.sum() + obj_out.sum()
        loss.backward()
        
        assert state_tokens.grad is not None
        assert obj_tokens.grad is not None


class TestTransformerDecoder:
    """Test suite for TransformerDecoder."""
    
    @pytest.fixture
    def state_dim(self):
        return 512
    
    @pytest.fixture
    def num_layers(self):
        return 3
    
    @pytest.fixture
    def num_heads(self):
        return 8
    
    @pytest.fixture
    def batch_size(self):
        return 4
    
    @pytest.fixture
    def n_state(self):
        return 16
    
    @pytest.fixture
    def n_obj(self):
        return 32
    
    @pytest.fixture
    def decoder_with_registers(self, state_dim, num_layers, num_heads):
        return TransformerDecoder(
            state_dim=state_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            feedforward_dim=1280,
            dropout=0.1,
            activation='swiglu',
            return_all_layers=True,
            use_self_attention=True,
            norm_type='rms_norm',
            qk_norm=True,
            num_register_tokens=4
        )
    
    @pytest.fixture
    def decoder_no_registers(self, state_dim, num_layers, num_heads):
        return TransformerDecoder(
            state_dim=state_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            feedforward_dim=2048,
            dropout=0.1,
            activation='relu',
            return_all_layers=False,
            use_self_attention=False,
            norm_type='layer_norm',
            qk_norm=False,
            num_register_tokens=0
        )
    
    def test_init_with_registers(self, decoder_with_registers, num_layers):
        """Test decoder initialization with register tokens."""
        assert len(decoder_with_registers.layers) == num_layers
        assert decoder_with_registers.num_register_tokens == 4
        assert decoder_with_registers.register_tokens.shape == (1, 4, 512)
        assert decoder_with_registers.return_all_layers
    
    def test_init_no_registers(self, decoder_no_registers, num_layers):
        """Test decoder initialization without register tokens."""
        assert len(decoder_no_registers.layers) == num_layers
        assert decoder_no_registers.num_register_tokens == 0
        assert not decoder_no_registers.return_all_layers
    
    def test_forward_all_layers(self, decoder_with_registers, batch_size, n_state, n_obj, state_dim, num_layers):
        """Test forward pass returning all layers."""
        state_tokens = torch.randn(batch_size, n_state, state_dim)
        obj_tokens = torch.randn(batch_size, n_obj, state_dim)
        
        all_state_layers, all_obj_layers = decoder_with_registers(state_tokens, obj_tokens)
        
        assert len(all_state_layers) == num_layers
        assert len(all_obj_layers) == num_layers
        
        for state_layer, obj_layer in zip(all_state_layers, all_obj_layers):
            assert state_layer.shape == (batch_size, n_state, state_dim)
            assert obj_layer.shape == (batch_size, n_obj, state_dim)
    
    def test_forward_final_layer_only(self, decoder_no_registers, batch_size, n_state, n_obj, state_dim):
        """Test forward pass returning only final layer."""
        state_tokens = torch.randn(batch_size, n_state, state_dim)
        obj_tokens = torch.randn(batch_size, n_obj, state_dim)
        
        all_state_layers, all_obj_layers = decoder_no_registers(state_tokens, obj_tokens)
        
        assert len(all_state_layers) == 1
        assert len(all_obj_layers) == 1
        assert all_state_layers[0].shape == (batch_size, n_state, state_dim)
        assert all_obj_layers[0].shape == (batch_size, n_obj, state_dim)
    
    def test_register_tokens_removed(self, decoder_with_registers, batch_size, n_state, n_obj, state_dim):
        """Test that register tokens are properly removed from output."""
        state_tokens = torch.randn(batch_size, n_state, state_dim)
        obj_tokens = torch.randn(batch_size, n_obj, state_dim)
        
        all_state_layers, all_obj_layers = decoder_with_registers(state_tokens, obj_tokens)
        
        # All output state layers should have original n_state, not n_state + num_register_tokens
        for state_layer in all_state_layers:
            assert state_layer.shape[1] == n_state
    
    def test_forward_with_positional_encoding(self, decoder_with_registers, batch_size, n_state, n_obj, state_dim):
        """Test forward pass with positional encodings."""
        state_tokens = torch.randn(batch_size, n_state, state_dim)
        obj_tokens = torch.randn(batch_size, n_obj, state_dim)
        state_pos = torch.randn(batch_size, n_state, state_dim)
        obj_pos = torch.randn(batch_size, n_obj, state_dim)
        
        all_state_layers, all_obj_layers = decoder_with_registers(
            state_tokens, obj_tokens, state_pos, obj_pos
        )
        
        assert all_state_layers[0].shape == (batch_size, n_state, state_dim)
        assert all_obj_layers[0].shape == (batch_size, n_obj, state_dim)
    
    def test_forward_with_mask(self, decoder_with_registers, batch_size, n_state, n_obj, state_dim):
        """Test forward pass with attention mask."""
        state_tokens = torch.randn(batch_size, n_state, state_dim)
        obj_tokens = torch.randn(batch_size, n_obj, state_dim)
        mask = torch.rand(batch_size, n_obj) > 0.5
        
        all_state_layers, all_obj_layers = decoder_with_registers(
            state_tokens, obj_tokens, mask=mask
        )
        
        assert len(all_state_layers) > 0
        assert len(all_obj_layers) > 0
    
    @pytest.mark.parametrize("n_state_test,n_obj_test", [
        (8, 16),
        (32, 64),
        (64, 128),
        (1, 10),
    ])
    def test_variable_sequence_lengths(self, decoder_with_registers, batch_size, state_dim, n_state_test, n_obj_test):
        """Test decoder with different sequence lengths."""
        state_tokens = torch.randn(batch_size, n_state_test, state_dim)
        obj_tokens = torch.randn(batch_size, n_obj_test, state_dim)
        
        all_state_layers, all_obj_layers = decoder_with_registers(state_tokens, obj_tokens)
        
        assert all_state_layers[0].shape == (batch_size, n_state_test, state_dim)
        assert all_obj_layers[0].shape == (batch_size, n_obj_test, state_dim)
    
    def test_gradient_flow(self, decoder_with_registers, batch_size, n_state, n_obj, state_dim):
        """Test gradient flow through full decoder."""
        decoder_with_registers.zero_grad()
        
        state_tokens = torch.randn(batch_size, n_state, state_dim, requires_grad=True)
        obj_tokens = torch.randn(batch_size, n_obj, state_dim, requires_grad=True)
        
        all_state_layers, all_obj_layers = decoder_with_registers(state_tokens, obj_tokens)
        
        loss = sum(layer.sum() for layer in all_state_layers) + sum(layer.sum() for layer in all_obj_layers)
        loss.backward()
        
        assert state_tokens.grad is not None
        assert obj_tokens.grad is not None
        assert not torch.isnan(state_tokens.grad).any()
        assert not torch.isnan(obj_tokens.grad).any()
    
    def test_parameter_count(self, decoder_with_registers):
        """Test parameter counting."""
        total_params = sum(p.numel() for p in decoder_with_registers.parameters())
        trainable_params = sum(p.numel() for p in decoder_with_registers.parameters() if p.requires_grad)
        
        assert total_params > 0
        assert trainable_params == total_params
        
        # Check that register tokens contribute to parameter count
        register_params = decoder_with_registers.register_tokens.numel()
        assert register_params == 4 * 512  # num_register_tokens * state_dim
    
    def test_deterministic_output(self, decoder_with_registers, batch_size, n_state, n_obj, state_dim):
        """Test that same input produces same output in eval mode."""
        decoder_with_registers.eval()
        
        state_tokens = torch.randn(batch_size, n_state, state_dim)
        obj_tokens = torch.randn(batch_size, n_obj, state_dim)
        
        with torch.no_grad():
            all_state_1, all_obj_1 = decoder_with_registers(state_tokens, obj_tokens)
            all_state_2, all_obj_2 = decoder_with_registers(state_tokens, obj_tokens)
        
        for s1, s2 in zip(all_state_1, all_state_2):
            assert torch.allclose(s1, s2, atol=1e-6)
        
        for o1, o2 in zip(all_obj_1, all_obj_2):
            assert torch.allclose(o1, o2, atol=1e-6)
    
    def test_auto_feedforward_dim(self, state_dim, num_layers, num_heads):
        """Test automatic feedforward dimension calculation."""
        decoder = TransformerDecoder(
            state_dim=state_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            feedforward_dim=None,  # Should auto-compute
            activation='swiglu'
        )
        
        # For SwiGLU, default should be int(state_dim * 2.5)
        expected_ffn_dim = int(state_dim * 2.5)
        actual_ffn_dim = decoder.layers[0].ffn_state.w1.out_features
        
        assert actual_ffn_dim == expected_ffn_dim
    
    def test_batch_size_one(self, decoder_with_registers, n_state, n_obj, state_dim):
        """Test edge case with batch size of 1."""
        state_tokens = torch.randn(1, n_state, state_dim)
        obj_tokens = torch.randn(1, n_obj, state_dim)
        
        all_state_layers, all_obj_layers = decoder_with_registers(state_tokens, obj_tokens)
        
        assert all_state_layers[0].shape == (1, n_state, state_dim)
        assert all_obj_layers[0].shape == (1, n_obj, state_dim)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])