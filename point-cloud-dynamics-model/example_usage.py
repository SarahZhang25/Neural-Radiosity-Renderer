"""
Example usage of the dynamics model for point cloud acceleration prediction.
"""

import torch
import yaml
from dynamics_model import DynamicsModel


def main():
    """Demonstrate how to use the dynamics model."""

    # Load configuration
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Initialize model
    model = DynamicsModel(config)
    model.eval()  # Set to evaluation mode

    # Example input data
    batch_size = 2
    n_objects = 3
    n_vertices = 500  # Variable number of vertices per object
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create dummy input data
    # Positions: current positions of all vertices
    positions = torch.randn(batch_size, n_objects, n_vertices, 3, device=device) * 2.0

    # Velocities: current velocities of all vertices
    velocities = torch.randn(batch_size, n_objects, n_vertices, 3, device=device) * 0.1

    # Physics parameters: [mass, friction, restitution] per object
    physics_params = torch.tensor([
        [1.0, 0.5, 0.3],  # Object 1: mass=1.0, friction=0.5, restitution=0.3
        [2.0, 0.3, 0.7],  # Object 2: mass=2.0, friction=0.3, restitution=0.7
        [0.5, 0.8, 0.2],  # Object 3: mass=0.5, friction=0.8, restitution=0.2
    ], device=device).unsqueeze(0).expand(batch_size, -1, -1)

    # Reference positions (optional, uses current positions if not provided)
    ref_positions = positions.clone()  # Usually the positions at t=0

    # Move model to device
    model = model.to(device)

    # Forward pass
    with torch.no_grad():
        accelerations = model(
            positions=positions,
            velocities=velocities,
            physics_params=physics_params,
            ref_positions=ref_positions,
            anchor_indices=None  # Will be computed automatically using FPS
        )

    print(f"Input shapes:")
    print(f"  Positions: {positions.shape}")
    print(f"  Velocities: {velocities.shape}")
    print(f"  Physics params: {physics_params.shape}")
    print(f"\nOutput shape:")
    print(f"  Accelerations: {accelerations.shape}")
    print(f"\nNote: Output is at {config['anchor_k']} anchor points per object")

    # Example: Integrate to get new positions (simple Verlet integration)
    dt = 0.01  # Time step
    dt_squared = dt * dt

    # For demonstration, we'll use the acceleration at anchors
    # In practice, you'd scatter these to full vertices
    print(f"\nPredicted acceleration statistics:")
    print(f"  Mean: {accelerations.mean().item():.6f}")
    print(f"  Std: {accelerations.std().item():.6f}")
    print(f"  Min: {accelerations.min().item():.6f}")
    print(f"  Max: {accelerations.max().item():.6f}")


def load_checkpoint_example():
    """Example of loading a trained model from checkpoint."""

    import os

    # Paths
    config_path = 'config.yaml'
    checkpoint_path = 'checkpoint_best.pth'  # Your checkpoint file

    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found at {checkpoint_path}")
        return

    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Initialize model
    model = DynamicsModel(config)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Load model weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    print(f"Successfully loaded checkpoint from {checkpoint_path}")

    # Set to evaluation mode
    model.eval()

    return model


def multi_step_rollout_example():
    """Example of multi-step rollout prediction."""

    # Load model
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    model = DynamicsModel(config)
    model.eval()

    # Setup
    batch_size = 1
    n_objects = 2
    n_vertices = 300
    n_steps = 10
    dt = 0.01
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initial conditions
    positions = torch.randn(batch_size, n_objects, n_vertices, 3, device=device)
    velocities = torch.zeros(batch_size, n_objects, n_vertices, 3, device=device)
    physics_params = torch.tensor([[1.0, 0.5, 0.3], [2.0, 0.3, 0.7]], device=device)
    physics_params = physics_params.unsqueeze(0)

    # Reference positions (at t=0)
    ref_positions = positions.clone()

    # Move model to device
    model = model.to(device)

    # Store trajectory
    trajectory = [positions.clone()]

    # Previous positions for Verlet integration
    prev_positions = positions - velocities * dt

    print("Running multi-step rollout...")

    # Rollout loop
    for step in range(n_steps):
        with torch.no_grad():
            # Predict accelerations at anchors
            accelerations_anchors = model(
                positions=positions,
                velocities=velocities,
                physics_params=physics_params,
                ref_positions=ref_positions
            )

            # For demo: apply same acceleration to all vertices
            # In practice, you'd scatter anchor accelerations properly
            accelerations_full = accelerations_anchors.mean(dim=2, keepdim=True)
            accelerations_full = accelerations_full.expand(-1, -1, n_vertices, -1)

            # Verlet integration
            new_positions = 2 * positions - prev_positions + accelerations_full * dt * dt

            # Update for next step
            prev_positions = positions
            positions = new_positions
            velocities = (positions - prev_positions) / dt

            # Store
            trajectory.append(positions.clone())

        print(f"  Step {step + 1}/{n_steps} completed")

    print(f"\nRollout completed!")
    print(f"Trajectory shape: {len(trajectory)} steps × {positions.shape}")

    return trajectory


if __name__ == "__main__":
    print("=" * 60)
    print("Basic Usage Example")
    print("=" * 60)
    main()

    print("\n" + "=" * 60)
    print("Multi-Step Rollout Example")
    print("=" * 60)
    multi_step_rollout_example()