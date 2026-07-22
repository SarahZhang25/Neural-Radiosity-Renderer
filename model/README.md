# Geometry-Aware Attention Bias (Factored QK Approach)

This directory implements a global illumination scene transformer that uses a novel, physically motivated geometry-aware attention bias.

## Overview

In global illumination, light transport between two objects depends strongly on their relative geometry, distance, orientation, and material properties (the *form factor*). Standard transformers lack this inductive bias. 

We introduce a learned pairwise geometric bias $\phi_{\mathrm{GI}}$ to the scene transformer's attention logits, so that attention is informed by coarse form-factor geometry without breaking the object-level token abstraction or reverting to dense $N \times N$ bias matrices that break FlashAttention compatibility.

## The Factored QK Method

Instead of explicitly computing an $\mathcal{O}(N^2)$ pairwise bias matrix, we encode object-level geometric features into **query-side** and **key-side** feature vectors, and add them directly to the attention queries and keys.

$$ \mathbf{q}_i' = \mathbf{q}_i + \delta\mathbf{q}_i $$
$$ \mathbf{k}_j' = \mathbf{k}_j + \delta\mathbf{k}_j $$

The resulting attention logit implicitly expands to capture both content and pairwise geometric interactions:

$$ \frac{\mathbf{q}_i'^\top \mathbf{k}_j'}{\sqrt{d}} = \underbrace{\frac{\mathbf{q}_i^\top \mathbf{k}_j}{\sqrt{d}}}_{\text{content}} + \underbrace{\frac{\delta\mathbf{q}_i^\top \delta\mathbf{k}_j}{\sqrt{d}}}_{\approx\, \phi_{\text{GI}}} + \underbrace{\frac{\mathbf{q}_i^\top \delta\mathbf{k}_j + \delta\mathbf{q}_i^\top \mathbf{k}_j}{\sqrt{d}}}_{\text{cross-terms}} $$

### Interaction with Rotary Positional Embeddings (RoPE)

Adding $\delta\mathbf{q}_i$ and $\delta\mathbf{k}_j$ to the query and key vectors *before* applying RoPE produces an elegant physical behavior.

Before RoPE, $\delta\mathbf{q}_i$ encapsulates object $i$'s material and scale, while $\delta\mathbf{k}_j$ encapsulates object $j$'s material, emissivity, and scale. **They contain no positional information.**

RoPE rotates these combined vectors based on their 3D centroid positions ($\mathbf{c}_i$ and $\mathbf{c}_j$):
$$ \tilde{\mathbf{q}}_i = R_{\Theta, \mathbf{c}_i} (\mathbf{q}_i + \delta\mathbf{q}_i) = R_{\Theta, \mathbf{c}_i}\mathbf{q}_i + R_{\Theta, \mathbf{c}_i}\delta\mathbf{q}_i $$
$$ \tilde{\mathbf{k}}_j = R_{\Theta, \mathbf{c}_j} (\mathbf{k}_j + \delta\mathbf{k}_j) = R_{\Theta, \mathbf{c}_j}\mathbf{k}_j + R_{\Theta, \mathbf{c}_j}\delta\mathbf{k}_j $$

When FlashAttention computes the inner product, the rotation matrices combine to form a relative rotation $R_{\Theta, \Delta\mathbf{c}_{ij}}$ (where $\Delta\mathbf{c}_{ij} = \mathbf{c}_i - \mathbf{c}_j$):
$$ \tilde{\mathbf{q}}_i^\top \tilde{\mathbf{k}}_j = \underbrace{\mathbf{q}_i^\top R_{\Delta\mathbf{c}} \mathbf{k}_j}_{\text{Pure Content}} + \underbrace{\delta\mathbf{q}_i^\top R_{\Delta\mathbf{c}} \delta\mathbf{k}_j}_{\text{Pure Geometry}} + \underbrace{\mathbf{q}_i^\top R_{\Delta\mathbf{c}} \delta\mathbf{k}_j + \delta\mathbf{q}_i^\top R_{\Delta\mathbf{c}} \mathbf{k}_j}_{\text{Cross-Terms}} $$

**The "Pure Geometry" term:**
$\delta\mathbf{q}_i^\top R_{\Theta, \Delta\mathbf{c}_{ij}} \delta\mathbf{k}_j$ evaluates the compatibility of object $i$'s receiver properties ($\delta\mathbf{q}_i$) with object $j$'s sender properties ($\delta\mathbf{k}_j$), **modulated by the relative distance and angle between them**. 

This means the network doesn't have to learn distance independently—the geometric dot product naturally decays or oscillates based on distance via the RoPE frequency bands. This perfectly approximates a coarse geometric form-factor calculation, while keeping the operations entirely factored for 100% FlashAttention speed.


## Tradeoffs vs traditional bias matrix
Strictly mathematically speaking, the QK factored approach is less expressive than a fully factored bias matrix. 

A full MLP that takes `[geom_i, geom_j]` and outputs an $N \times N$ matrix can learn *any* arbitrary, highly non-linear pairwise function. The factored approach restricts the geometric bias to the family of functions that can be expressed as an inner product: $\delta\mathbf{q}_i^\top \delta\mathbf{k}_j$.

However, there are four major reasons why this restriction usually doesn't hurt performance in practice (and often improves it by preventing overfitting), especially for physics and global illumination:

### 1. Multi-Head Attention acts as an ensemble
The dot product isn't just a single scalar interaction. Because we inject this into the transformer queries and keys, the dot product happens in `head_dim` space, and it happens independently across all `num_heads`. This means the network is actually learning an ensemble of `num_heads` different geometric interactions, giving it plenty of capacity.

### 2. The most important interaction (distance) is NOT factored
If we had to learn distance using only a factored dot product, it would struggle (you'd need high dimensions to approximate a radial basis function). But **because we apply RoPE after the injection**, the relative position $\Delta\mathbf{c}_{ij}$ is modeled using exact trigonometric rotations $R_{\Theta, \Delta\mathbf{c}_{ij}}$. The distance/angle relationship remains fully, exactly pairwise, while only the material and scale ($\rho$, $\mathbf{s}$) interactions are factored. 

### 3. Physics is often naturally factored
In radiosity and light transport, form-factors and BRDFs often decompose beautifully into dot products (e.g., Lambert's cosine law is literally a dot product of normals and light directions). The physical asymmetry between a light emitter (sender) and a surface (receiver) maps perfectly to the asymmetry between keys and queries.

### 4. You get "Cross-Terms" for free
With a full $N \times N$ bias matrix added to the logits, the geometric bias is static—it doesn't care what features the network is currently thinking about. With the factored approach, expanding the dot product gives cross terms: $\mathbf{q}_i^\top \delta\mathbf{k}_j + \delta\mathbf{q}_i^\top \mathbf{k}_j$. This means the *content* of the scene (the actual neural features in $\mathbf{q}$ and $\mathbf{k}$) dynamically interacts with the geometry, making the attention mechanism highly adaptive. 

***

**TL;DR:** it is less expressive than a black-box $N \times N$ MLP, but it retains exactly the expressiveness needed for spatial physics while buying you a massive $O(N)$ speedup via FlashAttention.

A reference work: The most relevant body of work centers around FlashBias (NeurIPS 2025) and related low-rank approximations. In disciplines like protein folding (e.g., AlphaFold 3's Pairformer), spatial relationships are traditionally modeled with explicit $N \times N$ pair bias matrices. However, researchers realized that these matrices destroy memory efficiency on modern GPUs.

The current state-of-the-art solution in the literature is exactly a form of "factoring": approximating the dense $N \times N$ bias matrix as the low-rank outer product of two matrices (or vectors). This is exactly what the geom_q and geom_k vectors in our approach are doing.

Our specific geometric variation—injecting this factored bias before RoPE to let the rotary encoding naturally construct a distance-decaying dot product—is a mathematically elegant intersection of Continuous Relative Positional Encodings (CPB) and these new factored bias approaches.



## Implementation Details

1. **Geometry Encoders (`attn_bias.py`)**
   - `ReceiverGeometryEncoder` ($\delta\mathbf{q}$) and `SenderGeometryEncoder` ($\delta\mathbf{k}$) are lightweight MLPs.
   - They map 9D Object Bounding Box (OBB) extents and 10D material properties (diffuse, specular, emissive, roughness) into vectors matching the transformer's `hidden_dim`.

2. **Attention Integration (`layers/attention.py`)**
   - `MultiHeadAttention` accepts `geom_q` and `geom_k` arguments.
   - The geometric vectors are added to standard queries and keys *after* the initial Q/K projection, but *before* RoPE is applied.

3. **Global Illumination Integration (`global_illumination_model.py`)**
   - Mean-pools the point-level `obj_properties` into object-level materials `(B, N_obj, 10)`.
   - Computes OBB axes before the transformer block to feed the bias encoders.
   - Automatically handles padding for learnable register tokens.

    - Enabled via the opt-in flag `use_obj_obj_attention_bias` inside `SceneTransformerConfig`.
   - Hidden dimension controlled via `geom_bias_hidden_dim`.

## Hybrid RoPE Projection (CamRay vs 3D)

In addition to the factored QK bias, our global illumination model features a novel **Hybrid RoPE Coordinate Projection** to solve a critical limitation when mixing 3D coordinates and 2D unnormalized ray directions.

### The Problem with 3D/2D RoPE Mixing
We use **CamRays** (`[u, v, -1]`) for ray queries to preserve the 2D linearity of the image plane. However, the scene objects exist in physical 3D camera space (`[X, Y, Z]`). Standard RoPE computes the relative distance by subtracting these coordinates: `(X - u, Y - v, Z - (-1))`.

Subtracting a 3D direction vector from an absolute 3D position is mathematically invalid and scrambles the RoPE frequencies. 
Furthermore, one might try to fix this by expanding the coordinate space (e.g. `[X, Y, Z, 0, 0, 0]` vs `[0, 0, 0, u, v, -1]`). However, because RoPE's final attention calculation evaluates as an orthogonal sum via dot products, **the network cannot learn cross-term spatial interactions across independent RoPE slots**. The model would just learn $f(X) + g(u)$ instead of analyzing if the ray $u$ actually aligns with the object position $X$.

### The Solution: Shared Projective Space
To allow RoPE to naturally learn relative spatial alignment, the object position and ray direction MUST be mapped into the exact same coordinate slots. 

We implement a Hybrid Projection `(proj_x, proj_y, depth)` for the first 3 slots of our 12D OBB RoPE encoding:
- **Objects**: Map `(X, Y, Z)` to `(-X/Z, -Y/Z, -Z)` (projected onto the $Z=-1$ plane, keeping $-Z$ for positive absolute depth).
- **CamRays**: Map `(u, v, -1)` to `(u, v, 0)`.

When RoPE subtracts them (`obj - ray = (-X/Z - u, -Y/Z - v, -Z)`):
1. **Slots 1 & 2**: Encodes the **exact 2D image-space alignment** between the ray and the object's physical projection on the screen.
2. **Slot 3**: Encodes the **absolute 3D depth** (since the ray's depth is 0), ensuring occlusion can be reasoned about.
3. **Slots 4-12**: The OBB axes remain untouched in 3D space.

### Connection to Related Works
This conceptual approach of mapping 3D geometry onto a shared 2D projective space for attention alignment is closely related to recent state-of-the-art multimodal research. Frameworks like **MotionWeaver** and **Kinema4D** project camera-space $(X, Y, Z)$ coordinates onto the image plane to ensure geometric tokens and visual queries are mathematically grounded in the exact same spatiotemporal location before computing attention. Similarly, architectures like **Projective Positional Encoding (PRoPE)** bake camera viewing frustums and geometries directly into relative positional encodings, translating 3D spatial relationships directly into image-space offsets to maintain translation invariance and spatial continuity.
