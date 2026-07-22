import torch
import torch.nn as nn

class GeometryEncoder(nn.Module):
    """
    Base MLP for encoding geometric features into attention query/key biases.
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, extents: torch.Tensor, properties: torch.Tensor) -> torch.Tensor:
        """
        Args:
            extents: (B, N, 9) OBB extents
            properties: (B, N, 10) material properties (diffuse, specular, emissive, roughness)
        Returns:
            bias vector: (B, N, output_dim)
        """
        # Concatenate features
        x = torch.cat([extents, properties], dim=-1)  # (B, N, 19)
        return self.mlp(x)


class ReceiverGeometryEncoder(GeometryEncoder):
    """
    Encodes receiver (query) geometry into $\delta\mathbf{q}$.
    This vector represents the properties of the surface receiving light (the query object),
    including its physical size (extents) and material response (diffuse/specular).

    Inputs:
        extents (9D) + properties (10D) = 19D
    """
    def __init__(self, hidden_dim: int, output_dim: int):
        super().__init__(input_dim=19, hidden_dim=hidden_dim, output_dim=output_dim)


class SenderGeometryEncoder(GeometryEncoder):
    """
    Encodes sender (key) geometry into $\delta\mathbf{k}$.
    This vector represents the properties of the surface emitting or reflecting light (the key object),
    including its physical size (extents), emissivity, and material response.

    Inputs:
        extents (9D) + properties (10D) = 19D
    """
    def __init__(self, hidden_dim: int, output_dim: int):
        super().__init__(input_dim=19, hidden_dim=hidden_dim, output_dim=output_dim)
