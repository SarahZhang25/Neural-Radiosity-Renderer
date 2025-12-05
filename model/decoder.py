"""
Bidirectional Transformer decoder for cross-attention between state and object tokens.
"""

import torch
import torch.nn as nn
from typing import Tuple, List, Optional


class TransformerDecoder(nn.Module):
    """
    Bidirectional decoder with pure cross-attention.
    State tokens query object features and vice versa.
    """

    def __init__(
        self,
        state_dim: int = 512,
        num_layers: int = 3,
        num_heads: int = 4,
        feedforward_dim: Optional[int] = None,
        dropout: float = 0.1,
        activation: str = 'swiglu',
        return_all_layers: bool = True,
        use_self_attention: bool = True,
        norm_type: str = 'rms_norm',
        qk_norm: bool = True,
        num_register_tokens: int = 4
    ):
        super().__init__()

        self.state_dim = state_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.return_all_layers = return_all_layers
        self.use_self_attention = use_self_attention

        # Auto-compute feedforward dimension for SwiGLU
        if feedforward_dim is None:
            feedforward_dim = int(state_dim * 2.5)

        # Register tokens (additional learnable tokens)
        self.num_register_tokens = num_register_tokens
        if num_register_tokens > 0:
            self.register_tokens = nn.Parameter(
                torch.randn(1, num_register_tokens, state_dim) * 0.02
            )

        # Decoder layers
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(
                d_model=state_dim,
                nhead=num_heads,
                dim_feedforward=feedforward_dim,
                dropout=dropout,
                activation=activation,
                use_self_attention=use_self_attention,
                norm_type=norm_type,
                qk_norm=qk_norm
            )
            for _ in range(num_layers)
        ])

    def forward(
        self,
        state_tokens: torch.Tensor,
        obj_tokens: torch.Tensor,
        state_pos: Optional[torch.Tensor] = None,
        obj_pos: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Forward pass through decoder layers.

        Args:
            state_tokens: State tokens (B, N_state, D)
            obj_tokens: Object tokens (B, N_obj, D)
            state_pos: Optional positional encoding for state tokens
            obj_pos: Optional positional encoding for object tokens
            mask: Optional attention mask

        Returns:
            all_state_layers: List of state tokens from each layer
            all_obj_layers: List of object tokens from each layer
        """
        B = state_tokens.shape[0]

        # Add register tokens to state
        if self.num_register_tokens > 0:
            register_tokens_expanded = self.register_tokens.expand(B, -1, -1)
            state_tokens = torch.cat([state_tokens, register_tokens_expanded], dim=1)

        all_state_layers = []
        all_obj_layers = []

        # Pass through decoder layers
        for layer in self.layers:
            state_tokens, obj_tokens = layer(
                state_tokens, obj_tokens,
                state_pos, obj_pos,
                mask
            )

            if self.return_all_layers:
                # Remove register tokens before saving
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


class TransformerDecoderLayer(nn.Module):
    """
    Single layer of bidirectional cross-attention.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        activation: str = 'swiglu',
        use_self_attention: bool = True,
        norm_type: str = 'rms_norm',
        qk_norm: bool = True
    ):
        super().__init__()

        self.use_self_attention = use_self_attention

        # Normalization layers
        if norm_type == 'rms_norm':
            self.norm_state1 = RMSNorm(d_model)
            self.norm_obj1 = RMSNorm(d_model)
            self.norm_state2 = RMSNorm(d_model)
            self.norm_obj2 = RMSNorm(d_model)
            if use_self_attention:
                self.norm_state_self = RMSNorm(d_model)
        else:
            self.norm_state1 = nn.LayerNorm(d_model)
            self.norm_obj1 = nn.LayerNorm(d_model)
            self.norm_state2 = nn.LayerNorm(d_model)
            self.norm_obj2 = nn.LayerNorm(d_model)
            if use_self_attention:
                self.norm_state_self = nn.LayerNorm(d_model)

        # Cross-attention layers
        self.cross_attn_state_to_obj = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.cross_attn_obj_to_state = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )

        # Optional self-attention for state tokens
        if use_self_attention:
            self.self_attn_state = nn.MultiheadAttention(
                d_model, nhead, dropout=dropout, batch_first=True
            )

        # Feedforward networks
        if activation == 'swiglu':
            self.ffn_state = SwiGLU(d_model, dim_feedforward, dropout)
            self.ffn_obj = SwiGLU(d_model, dim_feedforward, dropout)
        else:
            self.ffn_state = nn.Sequential(
                nn.Linear(d_model, dim_feedforward),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(dim_feedforward, d_model),
                nn.Dropout(dropout)
            )
            self.ffn_obj = nn.Sequential(
                nn.Linear(d_model, dim_feedforward),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(dim_feedforward, d_model),
                nn.Dropout(dropout)
            )

    def forward(
        self,
        state_tokens: torch.Tensor,
        obj_tokens: torch.Tensor,
        state_pos: Optional[torch.Tensor] = None,
        obj_pos: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through one decoder layer."""

        # Optional self-attention for state tokens
        if self.use_self_attention:
            state_norm = self.norm_state_self(state_tokens)
            state_attn, _ = self.self_attn_state(state_norm, state_norm, state_norm)
            state_tokens = state_tokens + state_attn

        # Cross-attention: state queries object
        state_norm = self.norm_state1(state_tokens)
        obj_norm = self.norm_obj1(obj_tokens)
        state_attn, _ = self.cross_attn_state_to_obj(
            state_norm, obj_norm, obj_norm,
            key_padding_mask=mask if mask is not None else None
        )
        state_tokens = state_tokens + state_attn

        # Cross-attention: object queries state
        obj_attn, _ = self.cross_attn_obj_to_state(
            obj_norm, state_norm, state_norm
        )
        obj_tokens = obj_tokens + obj_attn

        # Feedforward
        state_tokens = state_tokens + self.ffn_state(self.norm_state2(state_tokens))
        obj_tokens = obj_tokens + self.ffn_obj(self.norm_obj2(obj_tokens))

        return state_tokens, obj_tokens


class RMSNorm(nn.Module):
    """Root Mean Square Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / norm * self.weight


class SwiGLU(nn.Module):
    """SwiGLU activation function as used in LLaMA."""

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(nn.functional.silu(self.w1(x)) * self.w3(x)))