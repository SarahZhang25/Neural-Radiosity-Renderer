"""
State manager for persistent state tokens.
"""

import torch
import torch.nn as nn


class StateManager(nn.Module):
    """
    Manages learnable state tokens that persist across time steps.
    State tokens are arranged in a 3D grid (2x2x2 = 8 tokens).
    """

    def __init__(
        self,
        num_tokens: int = 8,
        token_dim: int = 512,
        learnable_init: bool = True,
        init_scale: float = 0.02
    ):
        super().__init__()

        self.num_tokens = num_tokens
        self.token_dim = token_dim

        if learnable_init:
            # Initialize with small random values
            self.tokens = nn.Parameter(
                torch.randn(1, num_tokens, token_dim) * init_scale
            )
        else:
            # Fixed initialization
            self.register_buffer(
                'tokens',
                torch.randn(1, num_tokens, token_dim) * init_scale
            )

        # 3D grid positions for tokens (used for positional encoding)
        self._init_grid_positions()

    def _init_grid_positions(self):
        """Initialize 3D grid positions for state tokens."""
        # For 8 tokens: 2x2x2 grid
        grid_size = int(self.num_tokens ** (1/3))
        positions = []

        for i in range(grid_size):
            for j in range(grid_size):
                for k in range(grid_size):
                    # Normalize to [-1, 1]
                    x = 2.0 * i / (grid_size - 1) - 1.0 if grid_size > 1 else 0.0
                    y = 2.0 * j / (grid_size - 1) - 1.0 if grid_size > 1 else 0.0
                    z = 2.0 * k / (grid_size - 1) - 1.0 if grid_size > 1 else 0.0
                    positions.append([x, y, z])

        self.register_buffer(
            'grid_positions',
            torch.tensor(positions[:self.num_tokens], dtype=torch.float32)
        )

    def get_tokens(self, batch_size: int) -> torch.Tensor:
        """
        Get state tokens for a batch.

        Args:
            batch_size: Batch size

        Returns:
            State tokens (B, num_tokens, token_dim)
        """
        return self.tokens.expand(batch_size, -1, -1)

    def get_positions(self, batch_size: int) -> torch.Tensor:
        """
        Get 3D grid positions for state tokens.

        Args:
            batch_size: Batch size

        Returns:
            Grid positions (B, num_tokens, 3)
        """
        return self.grid_positions.unsqueeze(0).expand(batch_size, -1, -1)