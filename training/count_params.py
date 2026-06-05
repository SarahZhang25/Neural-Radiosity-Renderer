"""
Usage:
python training/count_params.py path/to/checkpoint.pt
"""

import torch
import argparse
import os

def count_parameters(checkpoint_path):
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file not found at {checkpoint_path}")
        return

    try:
        print(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            print("Found 'model_state_dict' in checkpoint dictionary.")
            state_dict = checkpoint['model_state_dict']
        else:
            print("Assuming checkpoint is the state dictionary itself.")
            state_dict = checkpoint
            
        total_params = 0
        trainable_params = 0 # This estimate is less accurate if we don't have the model class instantiated, but usually all in state_dict are parameters.
        
        print("-" * 50)
        print(f"{'Layer':<60} {'Parameters':>15}")
        print("-" * 50)
        
        for key, value in state_dict.items():
            params = value.numel()
            total_params += params
            print(f"{key:<60} {params:>15,}")
            
        print("-" * 50)
        print(f"Total parameters: {total_params:,}")
        print("-" * 50)
        
    except Exception as e:
        print(f"Error loading checkpoint: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Count parameters in a PyTorch checkpoint")
    parser.add_argument("checkpoint_path", type=str, help="Path to the .pt checkpoint file")
    args = parser.parse_args()
    
    count_parameters(args.checkpoint_path)
