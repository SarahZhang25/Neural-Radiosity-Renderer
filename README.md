# Neural-Radiosity-Renderer
## Repo guide
Directories:
* `data_generation/`: Contains data generation scripts, mainly:
    * `generate_auto_mitsuba.py`: Main data generation script. Example usage at bottom of README
    * `generate_trajectory_dataset.py`: Generate dataset of static scene with camera moving along interpolated trajectory between two endpoints
* `data_generation/`: Can be disregarded. Archive for a bunch of old data generation scripts.

* `model/`: All model components are here:

    Model architecture flow:
    scene data (point clouds and properties) ➡️ `encoder.PointNetEncoder` ➡️ global object tokens ➡️ `layers.bidirectional_attention.BidirectionalTransformerEncoder` ➡️ multiscale object and state tokens (context) ⤵️

    query ray bundles ➡️ `ray_encoder.RayEncoder` ➡️ ray tokens (query) ➡️ `predictor.RadiancePredictor` ➡️ predicted radiance

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