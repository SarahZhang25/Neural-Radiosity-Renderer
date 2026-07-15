import torch
import sys
sys.path.append("/home/sazhang/Neural-Radiosity-Renderer/LitePT")
from litept.model import LitePT

model_s = LitePT(
    in_channels=16,
    stride=(2, 2, 2),
    enc_depths=(2, 2, 6, 2),
    enc_channels=(48, 96, 192, 384),
    enc_num_head=(2, 4, 8, 16),
    enc_patch_size=(1024, 1024, 1024, 1024),
    enc_conv=(True, True, True, False),
    enc_attn=(False, False, False, True),
    enc_rope_freq=(100.0, 100.0, 100.0, 100.0),
    drop_path=0.2,
    enc_mode=True
)

params = sum(p.numel() for p in model_s.parameters())
print(f"LitePT-S hypothesis params: {params/1e6:.2f}M")

model_default = LitePT(
    in_channels=16,
    stride=(2, 2, 2, 2),
    enc_depths=(2, 2, 2, 6, 2),
    enc_channels=(36, 72, 144, 252, 504),
    enc_num_head=(2, 4, 8, 14, 28),
    enc_patch_size=(1024, 1024, 1024, 1024, 1024),
    enc_conv=(True, True, True, False, False),
    enc_attn=(False, False, False, True, True),
    enc_rope_freq=(100.0, 100.0, 100.0, 100.0, 100.0),
    drop_path=0.2,
    enc_mode=True
)
params2 = sum(p.numel() for p in model_default.parameters())
print(f"LitePT default hypothesis params: {params2/1e6:.2f}M")
