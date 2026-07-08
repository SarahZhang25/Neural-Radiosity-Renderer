"""
Script for displaying model parameter counts
Example Usage:
    python check_params.py training/train_config_small.yaml
"""

import argparse
from model.config import NeuralRadiosityConfig
from model.global_illumination_model import GlobalIlluminationModel

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def check_params(config_path):
    config = NeuralRadiosityConfig.from_yaml(config_path)
        
    model = GlobalIlluminationModel(config)
    
    total, trainable = count_parameters(model)
    print(f"Model Configuration: {config_path}")
    print(f"Total Parameters: {total:,}")
    print(f"Trainable Parameters: {trainable:,}")
    
    # Iterate through top-level components
    for name, module in model.named_children():
        comp_total, comp_trainable = count_parameters(module)
        print(f"  {name}: {comp_total:,} params ({comp_total/total:.2%} of total)")
        
    # Print size of DPT head specifically
    if hasattr(model, 'predictor') and hasattr(model.predictor, 'out_dpt'):
        dpt_total, _ = count_parameters(model.predictor.out_dpt)
        print(f"    dpt_head: {dpt_total:,} params ({dpt_total/total:.2%} of total)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check model parameters.")
    parser.add_argument('config', type=str, help='Path to model config yaml file')
    args = parser.parse_args()
    
    check_params(args.config)
