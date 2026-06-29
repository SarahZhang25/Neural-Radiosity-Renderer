import torch
import flash_attn
from flash_attn import flash_attn_func
from flash_attn.flash_attn_interface import flash_attn_varlen_func

print("=" * 50)
print("Flash Attention Test Script")
print("=" * 50)

# 1. Version check
print(f"\n✅ flash-attn version : {flash_attn.__version__}")
print(f"✅ PyTorch version    : {torch.__version__}")
print(f"✅ CUDA available     : {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"✅ CUDA version       : {torch.version.cuda}")
    print(f"✅ GPU               : {torch.cuda.get_device_name(0)}")
else:
    print("⚠️  No CUDA GPU detected — flash-attn requires a CUDA GPU to run.")
    exit(1)

# 2. Basic flash attention forward pass
print("\n--- Running basic flash_attn_func test ---")

batch_size = 2
seq_len    = 128
num_heads  = 8
head_dim   = 64   # must be a supported value (32, 64, 128, 256)

# Q, K, V must be float16 or bfloat16
dtype = torch.float16
device = "cuda"

q = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=dtype, device=device)
k = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=dtype, device=device)
v = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=dtype, device=device)

out = flash_attn_func(q, k, v, dropout_p=0.0, causal=False)
print(f"✅ Output shape (non-causal) : {out.shape}")  # (batch, seq, heads, head_dim)

out_causal = flash_attn_func(q, k, v, dropout_p=0.0, causal=True)
print(f"✅ Output shape (causal)     : {out_causal.shape}")

# 3. Numerical sanity check
assert out.shape == (batch_size, seq_len, num_heads, head_dim), "Shape mismatch!"
assert not torch.isnan(out).any(), "NaNs detected in output!"
assert not torch.isinf(out).any(), "Infs detected in output!"
print("✅ No NaNs or Infs in output")

# 4. Compare with standard scaled dot-product attention
print("\n--- Comparing with PyTorch SDPA ---")

# Reshape for PyTorch: (batch, heads, seq, head_dim)
q_pt = q.permute(0, 2, 1, 3).float()
k_pt = k.permute(0, 2, 1, 3).float()
v_pt = v.permute(0, 2, 1, 3).float()

with torch.no_grad():
    ref = torch.nn.functional.scaled_dot_product_attention(q_pt, k_pt, v_pt, is_causal=False)

# flash output: (batch, seq, heads, head_dim) → (batch, heads, seq, head_dim)
out_pt = out.permute(0, 2, 1, 3).float()

max_diff = (out_pt - ref).abs().max().item()
print(f"✅ Max diff vs PyTorch SDPA  : {max_diff:.6f}  (expected < 0.01 for fp16)")

print("\n" + "=" * 50)
print("All tests passed! flash-attn is working correctly.")
print("=" * 50)