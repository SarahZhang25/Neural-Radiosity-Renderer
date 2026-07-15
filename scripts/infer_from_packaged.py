"""
Example script for using a packaged version of the model for inference
"""
import torch
import torch.package

def main():
    # Update this path to point to your actual packaged model file
    pkg_path = "training/logs/YOUR_RUN_ID/checkpoints/model_package_epoch_500.pt"
    
    print(f"Loading packaged model from {pkg_path}...")
    
    try:
        # 1. Create a PackageImporter instance for the packaged file
        imp = torch.package.PackageImporter(pkg_path)
        
        # 2. Load the pickled model object from the package
        # The first argument is the sub-directory name ("model") and the second is the file name ("model.pkl")
        model = imp.load_pickle("model", "model.pkl")
        
        # Put the model in evaluation mode
        model.eval()
        print("Model loaded successfully!")
        
        # 3. Move to the appropriate device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        # 4. Prepare dummy inputs matching the GlobalIlluminationModel's forward pass signature
        # (Replace these dummy tensors with your actual dataset inputs)
        B = 1
        H, W = 128, 128 # Replace with your rendering resolution
        N_obj = 2
        N_vertices = 100
        
        print("Creating dummy inputs...")
        rays_o = torch.randn(B, 3).to(device)
        rays_d = torch.randn(B, H, W, 3).to(device) 
        obj_positions = torch.randn(B, N_obj, N_vertices, 3).to(device) 
        obj_properties = torch.randn(B, N_obj, 10).to(device)
        w2c = torch.eye(4).unsqueeze(0).repeat(B, 1, 1).to(device) # World-to-camera transform
        
        print("Running inference...")
        with torch.no_grad():
            radiance = model(
                rays_o=rays_o,
                rays_d=rays_d,
                obj_positions=obj_positions,
                obj_properties=obj_properties,
                w2c=w2c
            )
            
        print(f"Inference successful! Output radiance shape: {radiance.shape}")
        
    except FileNotFoundError:
        print(f"Error: Could not find package at {pkg_path}.")
        print("Make sure you have completed a training run and point this script to the correct .pt file path.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
