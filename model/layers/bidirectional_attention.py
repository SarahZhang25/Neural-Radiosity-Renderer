import os
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.encodings.rope import (
    ObjectRotaryEmbedding,
    freqs_to_cos_sin,
    apply_rotary_emb_cossin,
    apply_rotary_emb_one_cossin
)
from model.layers.attention import MultiHeadAttention, FeedForwardSwiGLU, FeedForwardGeLU, EPS

class BidirectionalTransformerEncoder(nn.Module):
    def __init__(
        self,
        state_dim: int = 512,
        obj_dim: Optional[int] = None,
        num_layers: int = 3,
        num_heads: int = 4,
        feedforward_dim: Optional[int] = None,
        dropout: float = 0.1,
        activation: str = 'swiglu',
        return_all_layers: bool = True,
        use_self_attention: bool = True,
        norm_type: Literal['layer_norm', 'rms_norm'] = 'rms_norm',
        qk_norm: bool = True,
        num_register_tokens: int = 4,
        rope_dim: Optional[int] = None,
        rope_type: Literal['object', 'object_learned', 'object_mixed'] = 'object',
        rope_double_max_freq: bool = False,
        bias: bool = True,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.obj_dim = obj_dim if obj_dim is not None else state_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.return_all_layers = return_all_layers
        self.use_self_attention = use_self_attention

        if feedforward_dim is None:
            feedforward_dim = int(state_dim * 2.5)

        self.num_register_tokens = num_register_tokens
        if num_register_tokens > 0:
            self.register_tokens = nn.Parameter(
                torch.randn(1, num_register_tokens, state_dim) * 0.02
            )

        self.head_dim = state_dim // num_heads

        self.layers = nn.ModuleList([
            BidirectionalAttentionLayer(
                state_dim=self.state_dim,
                obj_dim=self.obj_dim,
                num_heads=num_heads,
                ffn_hidden_dim=feedforward_dim,
                dropout=dropout,
                bias=bias,
                activation=activation,
                norm_type=norm_type,
                qk_norm=qk_norm,
                use_self_attention_state=use_self_attention
            )
            for _ in range(num_layers)
        ])

        self.rope_dim = rope_dim
        if rope_dim is not None:
            assert rope_dim % 2 == 0, "rope_dim must be even"
            if rope_type != 'object_mixed':
                assert rope_dim * 3 <= self.head_dim, f"rope_dim {rope_dim} is too large for head_dim {self.head_dim}"
            else:
                rope_dim = self.head_dim
            self.rope_emb = ObjectRotaryEmbedding(
                dim=rope_dim,
                double_max_freq=rope_double_max_freq,
            )

    def forward(
        self,
        state_tokens: torch.Tensor,
        obj_tokens: torch.Tensor,
        state_pos: Optional[torch.Tensor] = None,
        obj_pos: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        force_sdpa: bool = False
    ):
        """
        Forward pass through decoder layers, bridging bidirectional attention and RoPE calculation.
        """
        B = state_tokens.shape[0]

        # Add register tokens to state
        if self.num_register_tokens > 0:
            register_tokens_expanded = self.register_tokens.expand(B, -1, -1)
            state_tokens = torch.cat([state_tokens, register_tokens_expanded], dim=1)

        # Compute RoPE tensors once internally
        rope_state_cos, rope_state_sin = None, None # TODO: remove? No need to apply ROPE on state tokens...
        rope_obj_cos, rope_obj_sin = None, None

        if getattr(self, 'rope_emb', None) is not None:
            ## NOTE: No need to apply ROPE on state tokens...
            # if state_pos is not None:
            #     # Register tokens don't map to original state positions. We pad them with zeros to avoid breaking geometry
            #     if self.num_register_tokens > 0:
            #         pad_pos = torch.zeros(B, self.num_register_tokens, state_pos.shape[-1], device=state_pos.device, dtype=state_pos.dtype)
            #         padded_state_pos = torch.cat([state_pos, pad_pos], dim=1)
            #     else:
            #         padded_state_pos = state_pos
            #     state_freqs = self.rope_emb.get_centroid_freqs(padded_state_pos)
            #     rope_state_cos, rope_state_sin = freqs_to_cos_sin(state_freqs, head_dim=self.head_dim)
            if obj_pos is not None:
                obj_freqs = self.rope_emb.get_centroid_freqs(obj_pos)
                rope_obj_cos, rope_obj_sin = freqs_to_cos_sin(obj_freqs, head_dim=self.head_dim)

        all_state_layers = []
        all_obj_layers = []

        # Pass through bidirectional blocks
        for layer in self.layers:
            state_tokens, obj_tokens = layer(
                state_tokens=state_tokens,
                obj_tokens=obj_tokens,
                src_key_padding_mask=mask,
                rope_state_cos=rope_state_cos,
                rope_state_sin=rope_state_sin,
                rope_obj_cos=rope_obj_cos,
                rope_obj_sin=rope_obj_sin,
                force_sdpa=force_sdpa
            )

            if self.return_all_layers:
                # Remove register tokens immediately before output
                if self.num_register_tokens > 0:
                    state_output = state_tokens[:, :-self.num_register_tokens]
                else:
                    state_output = state_tokens

                all_state_layers.append(state_output)
                all_obj_layers.append(obj_tokens)

        if not self.return_all_layers:
            # Only return final layer
            if self.num_register_tokens > 0:
                state_output = state_tokens[:, :-self.num_register_tokens]
            else:
                state_output = state_tokens

            all_state_layers = [state_output]
            all_obj_layers = [obj_tokens]

        return all_state_layers, all_obj_layers


class BidirectionalAttentionLayer(nn.Module):
    def __init__(
        self,
        state_dim: int,
        obj_dim: int,
        num_heads: int,
        ffn_hidden_dim: int,
        dropout: float = 0.1,
        bias: bool = True,
        activation: str = 'swiglu',
        norm_type: Literal['layer_norm', 'rms_norm'] = 'layer_norm',
        qk_norm: bool = False,
        use_self_attention_state: bool = True,
    ):
        """
        Bidirectional layer with cross-attention between state and object tokens.
        Integrates RoPE and optimized attention backends via MultiHeadAttention.
        """
        super().__init__()
        self.use_self_attention_state = use_self_attention_state

        if norm_type == 'layer_norm':
            norm_module = nn.LayerNorm
        elif norm_type == 'rms_norm':
            norm_module = nn.RMSNorm
        else:
            raise ValueError("Unsupported normalization type. Choose from 'layer_norm' and 'rms_norm'.")

        # Self-attention for state tokens
        if self.use_self_attention_state:
            self.self_attn_state = MultiHeadAttention(
                query_dim=state_dim,
                num_heads=num_heads,
                kv_dim=None,
                bias=bias,
                qk_norm=qk_norm,
                norm_type=norm_type
            )
            self.norm_state_self = norm_module(state_dim, eps=EPS)

        # Cross-attention layers
        self.cross_attn_state_to_obj = MultiHeadAttention(
            query_dim=state_dim,
            num_heads=num_heads,
            kv_dim=obj_dim,
            bias=bias,
            qk_norm=qk_norm,
            norm_type=norm_type
        )
        self.cross_attn_obj_to_state = MultiHeadAttention(
            query_dim=obj_dim,
            num_heads=num_heads,
            kv_dim=state_dim,
            bias=bias,
            qk_norm=qk_norm,
            norm_type=norm_type
        )

        # Cross-attn normalizations
        self.norm_state1 = norm_module(state_dim, eps=EPS)
        self.norm_obj1 = norm_module(obj_dim, eps=EPS)

        # Feedforward networks
        if activation == 'swiglu':
            self.ffn_state = FeedForwardSwiGLU(state_dim, hidden_dim=ffn_hidden_dim, dropout=dropout, bias=bias)
            self.ffn_obj = FeedForwardSwiGLU(obj_dim, hidden_dim=ffn_hidden_dim, dropout=dropout, bias=bias)
        elif activation == 'gelu':
            self.ffn_state = FeedForwardGeLU(state_dim, hidden_dim=ffn_hidden_dim, dropout=dropout, bias=bias)
            self.ffn_obj = FeedForwardGeLU(obj_dim, hidden_dim=ffn_hidden_dim, dropout=dropout, bias=bias)
        else:
            raise ValueError("Unsupported activation function. Choose from 'gelu' and 'swiglu'.")
            # Could default to a small MLP as seen in decoder.py 
            
        self.norm_state2 = norm_module(state_dim, eps=EPS)
        self.norm_obj2 = norm_module(obj_dim, eps=EPS)
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(
        self,
        state_tokens,
        obj_tokens,
        src_key_padding_mask=None,
        rope_state_cos=None,
        rope_state_sin=None,
        rope_obj_cos=None,
        rope_obj_sin=None,
        force_sdpa=False
    ):
        """
        Args:
            state_tokens: (B, N_state, state_dim)
            obj_tokens: (B, N_obj, obj_dim)
            src_key_padding_mask: (B, N_obj), mask for obj_tokens (True means attend)
        """
        # 1. State self-attention
        if self.use_self_attention_state:
            state_norm = self.norm_state_self(state_tokens)
            state_attn_out = self.self_attn_state(
                q=state_norm, k=state_norm, v=state_norm,
                rope_cos=rope_state_cos, rope_sin=rope_state_sin,
                force_sdpa=force_sdpa
            )
            state_tokens = state_tokens + self.dropout(state_attn_out)

        # 2. Cross-attention
        state_norm1 = self.norm_state1(state_tokens)
        obj_norm1 = self.norm_obj1(obj_tokens)

        # State queries Object
        state_to_obj_out = self.cross_attn_state_to_obj(
            q=state_norm1, k=obj_norm1, v=obj_norm1,
            src_key_padding_mask=src_key_padding_mask,
            rope_cos=rope_state_cos, rope_sin=rope_state_sin,
            rope_ctx_cos=rope_obj_cos, rope_ctx_sin=rope_obj_sin,
            force_sdpa=force_sdpa
        )
        
        # Object queries State
        obj_to_state_out = self.cross_attn_obj_to_state(
            q=obj_norm1, k=state_norm1, v=state_norm1,
            rope_cos=rope_obj_cos, rope_sin=rope_obj_sin,
            rope_ctx_cos=rope_state_cos, rope_ctx_sin=rope_state_sin,
            force_sdpa=force_sdpa
        )

        state_tokens = state_tokens + self.dropout(state_to_obj_out)
        obj_tokens = obj_tokens + self.dropout(obj_to_state_out)

        # 3. Feedforward
        state_tokens = state_tokens + self.dropout(self.ffn_state(self.norm_state2(state_tokens)))
        obj_tokens = obj_tokens + self.dropout(self.ffn_obj(self.norm_obj2(obj_tokens)))

        return state_tokens, obj_tokens
