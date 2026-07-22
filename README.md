# Neural-Radiosity-Renderer
Current working repo for object-based neural rendering model.

Model architecture flow:

    [view-independent stage]
    scene data (positions, normals, texture properties per point in point cloud) 
        ➡️ `encoder.PointNetEncoder` 
        ➡️  global object tokens    - register tokens ⤵️       - state manager ⤵️ [REMOVED]
        ➡️ `layers.bidirectional_attention.BidirectionalTransformerEncoder` 
        ➡️ multiscale object and register tokens (context) ⤵️

    [view-dependent stage]
        query ray bundles 
            ➡️ `ray_encoder.RayEncoder` 
            ➡️ ray tokens (query) 
            ➡️ `predictor.RadiancePredictor` 
            ➡️ predicted radiance

Notes:
* Training recipe is designed to mimic RenderFormer based on the paper training description. 


## Repo guide
Directories:
* `data_generation/`: Contains data generation scripts, mainly:
    * `generate_auto_mitsuba.py`: Main data generation script. Example usage at bottom of README
    * `generate_trajectory_dataset.py`: Generate dataset of static scene with camera moving along interpolated trajectory between two endpoints
* `data_generation/`: Can be disregarded. Archive for a bunch of old data generation scripts.

* `model/`: All model components are here:
    * `encodings/`: Contains nerf and rope encodings
    * `layers/`: Contains custom attention layers (supporting custom RoPE embeddings) and DPT head class.
    * `config.py`: Structured model configuration using frozen dataclasses. Defines `NeuralRadiosityConfig` with nested sub-configs (`EncoderConfig`, `SceneTransformerConfig`, `RayEncoderConfig`, `PredictorConfig`, `TrainingConfig`). Default values correspond to the 46M-parameter model. Use `NeuralRadiosityConfig.from_yaml(path)` to load from a YAML file with automatic default merging.
    * `decoder.py`: Bidirectional Transformer decoder module performing cross-attention between state and object tokens using torch.nn transformer layers. *Not currently in use, superceded by layers/bidirectional_attention.py implementation.*
    * `encoder.py`: PointNet-based encoder module for extracting features from point clouds. Includes simple material property encoder for representing diffuse color, specular color, emissive color, and roughness.
    * `global_illumination_model.py`: contains full neural rendering model.
    * `predictor.py`: Transformer module for radiance prediction. Uses cross-attention between rays and scene features. This version uses torch.nn TransformerDecoder and NERF-based positional encoding.
    * `predictor_rope.py`: Alternate version of `predictor.py` with custom implementation of TransformerDecoder that supports RoPE-style positional encoding. *Work-in-progress, being tested*.
    * `ray_encoder.py`: Ray map encoder module that produces ray tokens from a ray map and camera position. Applies NERF-based positional encoding.
    * `state_manager.py`

* `tests/`: Can be disregarded. Contains old test modules, not up to date/in use.
* `training/`: 
    * `dataset.py`: torch dataset for training datasets
    * `ray_generator.py`: Module for generating query rays given camera model (pose, fov) and desired output image resolution
    * `train.py`: Core model training script
    * `train_config.yaml`, `train_config_small.yaml`: training configuration files
    * `count_params.py`: Helper script for tallying params for a given model checkpoint
    * `filter_dataset.ipynb`: Helper interactive notebook for visualizing corrupted renders in a dataset

Home directory files:
* `infer.iypnb`: Notebook for interactively evaluating trained model in different ways and visualizing results
* `neural_radiosity.ipynb`: Can disregard this. Old notebook copied from https://github.com/krafton-ai/neural-radiosity-tutorial-mitsuba3.
* `environment.yml`, `requirements.txt`: for environment setup

## Design notes

### Geometry-Aware Obj-Obj Attention Bias (Factored QK)
To capture coarse radiosity form-factors without relying on $O(N^2)$ bias matrices that break FlashAttention integration, the scene transformer features a physically-motivated **Geometry-Aware Attention Bias**. 
By encoding object materials, extents (OBB axes), and emissivities into query ($\delta\mathbf{q}$) and key ($\delta\mathbf{k}$) vectors and injecting them *before* RoPE, the relative position rotation inherently modulates the dot-product $\delta\mathbf{q}^\top \delta\mathbf{k}$. This allows the attention mechanism to natively decay interactions based on distance and orientation while remaining fully factored and FlashAttention compatible.

For full mathematical details and implementation specifics, see the [model/README.md](file:///home/sazhang/Neural-Radiosity-Renderer/model/README.md).

### Register Tokens
To capture global scene context and reduce high-frequency noise without relying on a dedicated spatial state manager, the architecture supports appending learnable **register tokens** to the scene representation.
* **Initialization:** A configurable number of learnable embeddings (`num_register_tokens`) are prepended to the input sequence of the view-independent scene transformer.
* **Translation Invariance:** To ensure the register tokens are invariant to scene translations, their spatial positional encoding is derived from the mean centroid position of all valid object patches in the scene.
* **Cross-Attention:** Because they are part of the scene representation sequence, the predictor automatically computes cross-attention against both the object geometry tokens and the register tokens when querying rays.

### Ray Tokenization and View Direction Embeddings

The ray encoder maps continuous ray directions into discrete token embeddings (the "queries") for the cross-attention predictor. The model supports two distinct approaches for this view direction (`vdir`) encoding, configured via `vdir_pe_type`:

**1. NeRF-style Positional Encoding (`nerf`) [Legacy approach]**:
- **Token Content**: Normalized global/local ray directions are mapped to a higher-dimensional space using high-frequency sine/cosine waves (NeRF PE, controlled by `vdir_num_freqs`). The image grid is then divided into patches (e.g., 16x16), and the high-frequency values in each patch are linearly projected into a single dense token embedding. 
- **Token Positional Encoding**: Since patches inherently lose fine-grained spatial relationships, a separate geometric positional encoding is required for the attention mechanism. In this legacy approach, the spatial position of all ray tokens for a given camera was tied directly to the camera origin (`[0,0,0]` in local space). This meant the attention mechanism (RoPE) struggled to geographically distinguish the ray queries from each other.

**2. CamRay Encoding (`camray`) [New]**:
Inspired by the PRoPE paper (https://arxiv.org/pdf/2507.10496), this uses "CamRay"—the *unnormalized* camera-frame ray directions ($K^{-1} [u, v, 1]^T$)—which preserve the 2D linearity of the image plane and explicitly encode the focal length in their magnitude.
- **Token Content**: Bypasses high-frequency sine/cosine encoding entirely. The raw, low-frequency CamRay values for each patch are linearly projected directly into the token embedding. This provides a much cleaner, alias-free signal for the Transformer's MLPs to learn absolute viewing directions.
- **Token Positional Encoding**: The 3D centroid of the CamRay vectors for each patch is used as its explicit geometric coordinate (`ray_token_pos`). When passed into RoPE in the predictor, this provides distinct, geometrically meaningful 3D spatial positions for every ray token, allowing the attention mechanism to naturally calculate the relative angular and spatial distances between different image patches and 3D objects.

### Rotary Positional Embedding (RoPE)
We adapt Rotary Positional Embedding (RoPE) for object-level spatial reasoning. The model supports different geometric representations for RoPE (`pe_type`):

**1. Centroid-based RoPE (`pe_type: 'rope_centroid'`)**:
Uses the 3D geometric centroid of each object (or local patch). This provides a spatial proximity bias based on the inverse-square law of light transport.
- Positional dimensionality: 3D (x, y, z)
- Data flow: `PointNetEncoder` computes centroids → transformed to camera space in `GlobalIlluminationModel` → passed to `TransformerEncoder` / `TransformerDecoder`.

**2. OBB-based RoPE (`pe_type: 'rope_obb'`)** [better performance]:
Uses an Oriented Bounding Box (OBB) representation, giving 12D positional information: the 3D centroid plus 3 scaled principal axis vectors (9D).
- Why? Light transport (radiometric form factors) depends on both relative position and mutual orientation. OBB-RoPE encodes both spatial proximity and shape/orientation similarity.
- Computation: We compute the covariance matrix of each object's point cloud and use eigendecomposition (`eigh`) to find principal axes. The axes are scaled by their extents (sqrt of eigenvalues).
- Data flow: `GlobalIlluminationModel` computes OBB from raw point clouds → centroid gets full affine camera transform, axes get rotation only (translation invariant) → 12D positions passed to transformers.
- Note on capacity: Because the 12D representation requires more frequency bands, the maximum `rope_dim` ceiling is lower than for the 3D centroid variant. Ray tokens (which have no OBB) are zero-padded to 12D.



### LitePT Encoder Integration [New]
To capture rich, multi-scale geometric priors from point clouds, the legacy `PointNetEncoder` can be swapped for a state-of-the-art transformer-based backbone via `LitePTEncoderAdapter`. The LitePT integration introduces several critical capabilities designed specifically for scene rendering:

**1. Encoder-Only Execution (`enc_mode=True`)**:
Instead of running a full U-Net style encoder-decoder, the adapter stops at the bottleneck of the LitePT architecture. The deepest encoder stages contain highly aggregated, global semantic features (representing large sections of the object), bypassing the computationally expensive decoder which exists primarily to recover local, per-point spatial resolution.

**2. Hierarchical Pooling**:
Configurable via `pooling_type: 'hierarchical'` and `num_hierarchical_levels`. Rather than taking just the final global token, the model manually intercepts the forward pass at the last $N$ stages of the LitePT encoder. At each of these downsampled scales, it performs both spatial scatter-max and scatter-mean pooling. These multi-scale global descriptors are then concatenated and linearly projected to form an exceptionally rich global object token that encompasses both medium and macro-scale geometric structures.

**3. Pretrained Instance Segmentation Weights Transfer**:
The architecture supports seamless loading of official ScanNet pretrained weights (e.g., `insseg-litept-small-v1m2`). Because the adapter only utilizes the encoder branch, both semantic and instance segmentation weights map perfectly to the model. Furthermore, the adapter includes a custom weight-loading mechanism that overcomes input channel mismatches: if the pretrained model expects 6 channels (XYZ + Normals) but the rendering pipeline provides 16 channels (XYZ + Normals + 10 Material properties), the adapter automatically splices the pretrained geometric weights into the first 6 channels of the `embedding.stem` layer while randomly initializing the new material channels. This provides a massive geometric head start during training.

### Local patches [found to not work well]
There is an option to not just use a single token per object, but instead use N tokens where each token represents a patch of the object.
The motivation for this is that local patch features will preserve more local geometric information than collapsing the object geometry into a single token representation.

The sampling and feature extraction strategy used to create local patches:

*   **Farthest Point Sampling (FPS):** First, we sample a set number of key points (`num_centroids`, default 16) directly from the object's surface point cloud. FPS is used instead of random sampling to guarantee that these centroids are spread as evenly and widely as possible across the entire object's geometry.
*   **K-Nearest Neighbors (K-NN) Grouping:** For every sampled centroid, we calculate the Euclidean distance to all other points on the object. We then select the `k_neighbors` (default 2048 / 16 = 128) closest points to form a "local patch" around that centroid. Because this is done independently for each centroid, patches can overlap, allowing for smooth, continuous geometric context.
*   **Local Feature Aggregation (Max-Pooling per Patch):** We take the rich per-point features (which were already computed by the PointNet backbone) for all the points inside each patch. We then apply a **max-pooling** operation across those 128 neighbors. This compresses the structural information of that specific local neighborhood into a single representative feature vector *for that patch*.
*   **Set Abstraction Refinement:** Finally, those aggregated patch features are passed through a projection layer (`sa_proj` + BatchNorm + SiLU) to refine them. This creates the final sequence of local geometric tokens (`num_centroids` tokens per object, rather than a single global token) that are ready to be fused with the object's material properties.

For positional embeddings, the local patch token's position is given by the mean position of the points sampled to form the patch. 
## Setup notes
`conda create -n neural_radiosity_renderer python=3.11`
Use `environment.yml` and `requirements.txt`

`conda activate neural_radiosity_renderer`
`export PYTHONPATH=.`

Data generation with mitsuba rendering, example usage:

`python generate_auto_mitsuba.py --shapes chair car —shapes_per_class 5 --num_rotations 1 --spp 256 --randomize_camera True --viewpoints_per_case 4
`
<!-- `python data_generation/generate_pure_mitsuba.py --shapes sphere --num_rotations 3 --scale_min 0.3 --scale_max 0.6 --pos_variation 0.15 --light_intensity 10 --spp 1024` -->

Training script:
`python training/train.py`
Training config: `training/train_config.yml` or `training/train_config_small.yml`

Launch tensorboard with `tensorboard --logdir=training/logs`