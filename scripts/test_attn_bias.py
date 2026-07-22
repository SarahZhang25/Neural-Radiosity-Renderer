import torch
from model.attn_bias import ReceiverGeometryEncoder, SenderGeometryEncoder
from model.layers.attention import TransformerEncoder
from model.config import NeuralRadiosityConfig

def test_encoders():
    B = 2
    N = 10
    hidden_dim = 64
    out_dim = 128
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    extents = torch.randn(B, N, 9, device=device, dtype=dtype)
    properties = torch.randn(B, N, 10, device=device, dtype=dtype)
    
    rx_enc = ReceiverGeometryEncoder(hidden_dim, out_dim).to(device, dtype)
    tx_enc = SenderGeometryEncoder(hidden_dim, out_dim).to(device, dtype)
    
    geom_q = rx_enc(extents, properties)
    geom_k = tx_enc(extents, properties)
    
    assert geom_q.shape == (B, N, out_dim), f"Expected {(B, N, out_dim)}, got {geom_q.shape}"
    assert geom_k.shape == (B, N, out_dim), f"Expected {(B, N, out_dim)}, got {geom_k.shape}"
    print("Encoder output shapes are correct.")
    
def test_transformer_integration():
    B = 2
    N = 10
    dim = 64
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    encoder = TransformerEncoder(
        num_layers=2,
        num_heads=2,
        hidden_dim=dim,
        ffn_hidden_dim=dim*4,
        rope_dim=None
    ).to(device, dtype)
    
    x = torch.randn(B, N, dim, device=device, dtype=dtype)
    geom_q = torch.randn(B, N, dim, device=device, dtype=dtype)
    geom_k = torch.randn(B, N, dim, device=device, dtype=dtype)
    
    # If using flash_attn on older GPUs, might need force_sdpa=True if bfloat16 is unsupported,
    # but normally bfloat16 is fine on Ampere+. We'll just run it.
    out = encoder(x, geom_q=geom_q, geom_k=geom_k)
    
    # Since return_all_layers is True by default
    assert len(out) == 2, f"Expected 2 layers returned, got {len(out)}"
    assert out[-1].shape == (B, N, dim), f"Expected {(B, N, dim)}, got {out[-1].shape}"
    print("Transformer integration forward pass works.")

def test_model_integration():
    from model.global_illumination_model import GlobalIlluminationModel
    
    import dataclasses
    
    config = NeuralRadiosityConfig()
    
    # Override config to use the new feature safely using dataclasses.replace
    new_decoder = dataclasses.replace(
        config.decoder,
        use_obj_obj_attention_bias=True,
        obj_obj_bias_hidden_dim=64
    )
    config = dataclasses.replace(config, decoder=new_decoder)
    
    model = GlobalIlluminationModel(config)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = model.to(device, dtype)

    B, N_obj, N_v = 2, 3, 16
    rays_o = torch.randn(B, 3, device=device, dtype=dtype)
    rays_d = torch.randn(B, 16, 16, 3, device=device, dtype=dtype)
    obj_pos = torch.randn(B, N_obj, N_v, 3, device=device, dtype=dtype)
    obj_normals = torch.randn(B, N_obj, N_v, 3, device=device, dtype=dtype)
    obj_props = torch.randn(B, N_obj, N_v, 10, device=device, dtype=dtype)
    
    # Needs to not fail
    out = model(rays_o, rays_d, obj_pos, obj_props, obj_normals=obj_normals)
    assert out.shape == (B, 3, 16, 16)
    print("Global Illumination Model forward pass works with attention bias.")

if __name__ == "__main__":
    test_encoders()
    test_transformer_integration()
    test_model_integration()
