# Point Cloud Dynamics Model

A neural network for predicting point cloud accelerations from multi-body rigid dynamics.

## Architecture Overview

```
Input Point Cloud → Encoder → State Tokens → Decoder → Predictor → Acceleration
```

### Components

1. **Encoder**: PointNet-based encoder that extracts geometric and physical features
   - Input: Point cloud features (positions, velocities, etc.)
   - Output: Object tokens (512D)

2. **State Manager**: Persistent state tokens for temporal modeling
   - 8 learnable tokens (2×2×2 grid)
   - 512D each

3. **Decoder**: Bidirectional Transformer
   - 3 layers
   - Cross-attention between state and object tokens
   - No self-attention

4. **Predictor**: MLP-based acceleration predictor
   - Input: Multi-scale features + query
   - Output: Per-point accelerations

## Usage

```python
from dynamics_model import DynamicsModel

# Initialize model
model = DynamicsModel(config)

# Input:
# - positions: (B, N_objects, N_vertices, 3)
# - velocities: (B, N_objects, N_vertices, 3)
# - physics_params: (B, N_objects, 3) [mass, friction, restitution]

# Output:
# - accelerations: (B, N_objects, N_anchors, 3)

accelerations = model(positions, velocities, physics_params)
```

## Configuration

See `config.yaml` for model parameters:
- Encoder: 512D output, 800D backbone
- Decoder: 3 layers, 512D hidden
- State: 8 tokens, 512D each

## Requirements

- PyTorch >= 2.0
- Flash Attention (optional, for acceleration)# Neural_Rendering_Model
