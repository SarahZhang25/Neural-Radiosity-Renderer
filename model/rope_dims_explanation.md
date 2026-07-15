### RenderFormer RoPE Dimension (`rope_dim`) Explained

In the RenderFormer paper, each attention head operates on a 128-dimensional query/key representation (since `hidden_dim = 768` and `num_heads = 6`, giving `head_dim = 768 // 6 = 128`). 

The paper states:
> *"Each of the 6 attention heads operates on 128 coefficients of the triangle token embedding (6 heads × 128 = 768). Consequently, we apply the block-rotation only to the first 108 coefficients for each head and leave the remaining 20 coefficients unchanged."*

Here is exactly how RenderFormer sets and implements `rope_dim = 12` to achieve this:

---

### 1. The Head Dimension Constraints and config initialization
In [view_transformer.py](file:///home/sazhang/Neural-Radiosity-Renderer/renderformer/renderformer/models/view_transformer.py#L32-L34), `self.rope_dim` is initialized by taking the minimum of the configured frequency size and the maximum allowable dimensions that fit the head size:

```python
        elif config.pe_type == 'rope':
            self.rope_dim = min(config.vertex_pe_num_freqs, config.view_transformer_latent_dim // config.view_transformer_n_heads // 18 * 2)
```

#### Why `// 18 * 2`?
A triangle token has **3 vertices**, each with **3 spatial coordinates (x, y, z)**. This gives $3 \times 3 = 9$ spatial components.
* For each spatial component, we allocate `rope_dim // 2` frequency bands.
* Each band produces a sine/cosine pair ($2$ coefficients).
* Thus, the total number of rotated coefficients is:
  $$\text{Total Coefficients} = 9 \times \left(\frac{\text{rope\_dim}}{2}\right) \times 2 = 9 \times \text{rope\_dim}$$
* Since the rotated coefficients must fit inside the attention head dimension (`head_dim = latent_dim // n_heads`), we have the constraint:
  $$9 \times \text{rope\_dim} \le \text{head\_dim} \implies \text{rope\_dim} \le \frac{\text{head\_dim}}{9}$$
* Because RoPE operates on coordinate pairs, `rope_dim` must be an even integer:
  $$\text{rope\_dim} \le \left\lfloor\frac{\text{head\_dim}}{18}\right\rfloor \times 2$$

For $\text{head\_dim} = 128$:
$$\text{rope\_dim} \le \left\lfloor\frac{128}{18}\right\rfloor \times 2 = 7 \times 2 = 14$$

In RenderFormer's default configuration, `vertex_pe_num_freqs` is set to **`12`**. So:
$$\text{rope\_dim} = \min(12, 14) = 12$$

---

### 2. Frequency Allocation in `TriangleRotaryEmbedding`
In [rope.py](file:///home/sazhang/Neural-Radiosity-Renderer/renderformer/renderformer/encodings/rope.py#L152-L207), `TriangleRotaryEmbedding` creates $\text{rope\_dim} // 2 = 6$ frequency bands:

```python
class TriangleRotaryEmbedding(Module):
    def __init__(self, dim, hf_format=True, double_max_freq=False):
        super().__init__()
        self.hf_format = hf_format

        # log spaced frequencies (dim = rope_dim)
        max_freq = log(dim // 2 - 1, 2) if not double_max_freq else log(dim - 1, 2)
        freqs = 2 ** torch.linspace(0, max_freq, dim // 2)

        self.freqs = nn.Parameter(freqs, requires_grad=False)
```

---

### 3. Combining Triangle Coordinates and Frequencies
During the forward pass of [rope.py](file:///home/sazhang/Neural-Radiosity-Renderer/renderformer/renderformer/encodings/rope.py#L188-L206), the spatial coordinates of the 3 vertices are multiplied by the frequencies:

```python
    def get_triangle_freqs(self, pos: Tensor):
        # generate all frequencies for all triangles
        freqs = self.forward(pos)
        freqs = rearrange(
            freqs, "batch n_tris n_verts d -> batch 1 n_tris (n_verts d)"
        )  # 1 for head dim

        if self.hf_format:
            freqs = torch.cat([freqs, freqs], dim=-1)
        else:
            freqs = repeat(freqs, "... f -> ... (f r)", r=2)
        return freqs

    def forward(self, t: Tensor, seq_len=None, offset=0):
        freqs = self.freqs
        freqs = einsum("..., f -> ... f", t.type(freqs.dtype), freqs)
        return freqs
```

#### Step-by-Step Shape Walkthrough:
1. `pos` has shape `(B, N_tris, 3_verts, 3_xyz)`.
2. The `forward(t)` method performs `einsum("..., f -> ... f", t, freqs)` where `freqs` has shape `(6,)` (since `rope_dim // 2 = 6`).
3. This outputs `freqs` of shape `(B, N_tris, 3_verts, 3_xyz, 6)`.
4. We flatten the last three dimensions: `3_verts * 3_xyz * 6 = 54` frequency bands.
   - `rearrange` shapes this to `(B, 1, N_tris, 54)`.
5. HuggingFace format (`self.hf_format=True`) concatenates the frequencies along the last dimension:
   - `torch.cat([freqs, freqs], dim=-1)` resulting in shape `(B, 1, N_tris, 108)`.

---

### 4. Applying the Rotation in the Attention Layer
In [attention.py](file:///home/sazhang/Neural-Radiosity-Renderer/renderformer/renderformer/layers/attention.py#L567-L577), the attention class verifies that the required dimension fits:

```python
        self.rope_dim = rope_dim
        if rope_dim is not None:
            assert rope_dim % 2 == 0, "rope_dim must be even"
            if rope_type != 'triangle_mixed':
                assert rope_dim // 2 * 9 <= hidden_dim // num_heads, f"rope_dim {rope_dim} is too large for hidden_dim {hidden_dim} and num_heads {num_heads}"
```

Since the returned `rope_cos` and `rope_sin` from `freqs_to_cos_sin` only cover the first **108 dimensions**, the rotary embedding function only rotates those channels. The remaining **20 dimensions** (`128 - 108 = 20`) in each query/key head vector are untouched.