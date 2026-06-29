# Neural-Radiosity-Renderer
Current working repo for object-based neural rendering model.

Model architecture flow:

    [view-independent stage]
    scene data (point clouds and properties) 
        ➡️ `encoder.PointNetEncoder` 
        ➡️  global object tokens          - state manager ⤵️
        ➡️ `layers.bidirectional_attention.BidirectionalTransformerEncoder` 
        ➡️ multiscale object and state tokens (context) ⤵️

    [view-dependnet stage]
        query ray bundles 
            ➡️ `ray_encoder.RayEncoder` 
            ➡️ ray tokens (query) 
            ➡️ `predictor.RadiancePredictor` 
            ➡️ predicted radiance

Notes:
* Positional embedding: Centroid-based RoPE. We adapt RenderFormer's Relative Spatial Positional Embedding, which uses the 3 vertex positions of each triangle, to the object-level by using the centroid position of each object. 
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
### Local patches 
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