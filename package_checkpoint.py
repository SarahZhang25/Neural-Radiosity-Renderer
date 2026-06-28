import os
import argparse
import yaml
import torch
import torch.package
from model.global_illumination_model import GlobalIlluminationModel

def package_checkpoint(config_path, checkpoint_path):
    print(f"Loading config from {config_path}...")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    device = torch.device(config.get('training', {}).get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"Using device: {device}")
    
    print("Initializing model...")
    model = GlobalIlluminationModel(config).to(device)
    
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle torch.compile '_orig_mod.' prefixes if present
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
    except RuntimeError as e:
        print("Standard load failed, attempting to strip '_orig_mod.' prefix...")
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in checkpoint['model_state_dict'].items()}
        model.load_state_dict(state_dict)
    
    checkpoint_dir = os.path.dirname(os.path.abspath(checkpoint_path))
    base_name = os.path.basename(checkpoint_path)
    
    if "epoch_" in base_name:
        epoch_str = base_name.split("epoch_")[-1].split(".")[0]
        pkg_name = f"model_package_epoch_{epoch_str}.pt"
    else:
        pkg_name = f"model_package_{base_name}"
        
    pkg_path = os.path.join(checkpoint_dir, pkg_name)
    
    print(f"Packaging model to {pkg_path}...")
    with torch.package.PackageExporter(pkg_path) as exp:
        exp.intern("model.**")
        exp.intern("utils.**")
        exp.intern("pos_encodings.**")
        exp.extern("**")
        exp.save_pickle("model", "model.pkl", model)
        
    print(f"Successfully saved packaged model to {pkg_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Package a saved model checkpoint.")
    parser.add_argument('--config', type=str, required=True, help='Path to the config file used for training')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the checkpoint .pt file')
    
    args = parser.parse_args()
    package_checkpoint(args.config, args.checkpoint)
