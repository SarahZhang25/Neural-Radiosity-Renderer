import sys
import os
import torch

# Add current directory to path so we can import model
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model.encoder import LitePTEncoderAdapter

def main():
    print("Instantiating LitePTEncoderAdapter...")
    try:
        adapter = LitePTEncoderAdapter(
            in_channels=16,
            out_channels=768, # e.g. backbone_dim
            use_local_patches=False,
            # pretrained_weights_path="LitePT-S.pth"
        )
        adapter = adapter.cuda()
        print("Successfully instantiated LitePTEncoderAdapter.")
        
        # Create dummy data
        B = 2
        N = 1000
        print(f"Creating dummy data with Batch={B}, Points={N}")
        surface_pos = torch.randn(B, N, 3).cuda()
        properties = torch.randn(B, 10).cuda()
        normals = torch.randn(B, N, 3).cuda()

        print("Running forward pass...")
        global_token, global_pos = adapter(surface_pos, properties, normals)
        
        print(f"Output global_token shape: {global_token.shape} (Expected: {B}, 768)")
        print(f"Output global_pos shape: {global_pos.shape} (Expected: {B}, 3)")
        print("Forward pass successful!")

    except Exception as e:
        print(f"Failed to test LitePTEncoderAdapter: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
