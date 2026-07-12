"""
Test script to instantiate the full GlobalIlluminationModel with LitePT encoder
and run a forward pass with dummy data.
"""
import sys
import os
import logging

# Suppress harmless FX tracing warning from spconv
logging.getLogger("torch.fx._symbolic_trace").setLevel(logging.ERROR)


import torch
from model.config import NeuralRadiosityConfig
from model.encoder import LitePTEncoderAdapter
from model.global_illumination_model import GlobalIlluminationModel

def test_encoder_adapter_only():
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
        properties = torch.randn(B, N, 10).cuda()
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

def test_full_model_integration():
    config_path = "training/train_config_46M_litePT.yaml"
    print(f"Loading config from {config_path}...")
    config = NeuralRadiosityConfig.from_yaml(config_path)
    print(f"  encoder_type: {config.encoder_type}")
    print(f"  litept_encoder: in_channels={config.litept_encoder.in_channels}, "
          f"out_channels={config.litept_encoder.out_channels}")

    print("\nInstantiating GlobalIlluminationModel with LitePT encoder...")
    model = GlobalIlluminationModel(config)
    model = model.cuda()
    model.eval()

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total Parameters: {total_params:,}")
    print(f"  Trainable Parameters: {trainable_params:,}")

    # Encoder params
    enc_params = sum(p.numel() for p in model.obj_encoder.parameters())
    print(f"  Encoder (LitePT) Parameters: {enc_params:,}")

    # Create dummy data matching GlobalIlluminationModel.forward() signature
    B = 2           # batch size
    N_obj = 3       # number of objects per scene
    N_v = 256       # vertices per object
    H = W = 128     # image resolution

    print(f"\nCreating dummy data: B={B}, N_obj={N_obj}, N_v={N_v}, H=W={H}")
    rays_o = torch.randn(B, 3).cuda()
    rays_d = torch.randn(B, H, W, 3).cuda()
    obj_positions = torch.randn(B, N_obj, N_v, 3).cuda()
    obj_properties = torch.randn(B, N_obj, N_v, 10).cuda()
    obj_normals = torch.randn(B, N_obj, N_v, 3).cuda()
    obj_mask = torch.ones(B, N_obj, dtype=torch.bool).cuda()
    w2c = torch.eye(4).unsqueeze(0).expand(B, -1, -1).cuda()

    print("Running forward pass...")
    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        output = model(
            rays_o=rays_o,
            rays_d=rays_d,
            obj_positions=obj_positions,
            obj_properties=obj_properties,
            obj_normals=obj_normals,
            obj_mask=obj_mask,
            w2c=w2c,
        )

    print(f"\nOutput shape: {output.shape} (Expected: [{B}, 3, {H}, {W}])")
    assert output.shape == (B, 3, H, W), f"Shape mismatch! Got {output.shape}"
    print("Forward pass successful! ✓")


if __name__ == "__main__":
    test_encoder_adapter_only()
    test_full_model_integration()