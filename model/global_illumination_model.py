"""
Global Illumination Model
Predict radiance in scene of emitter and modifier objects
"""

import torch
import torch.nn as nn
import yaml
from typing import Dict, Tuple, Optional

from encoder import PointNetEncoder
from decoder import TransformerDecoder
from predictor import RadiancePredictor
from state_manager import StateManager
from ray_encoder import RayEncoder
from utils import ray_generator

class GlobalIlluminationModel(nn.Module):
    """

    Architecture:
        [Object Point Clouds]   -> *PointNetEncoder ->
                                   *State Tokens    -> *Decoder      -> 
                                   [Ray Query]      -> *Ray Encoder  -> *Predictor -> Radiance
    (Key: [] Inputs, * Learnable)

    Input:
        -

    Output:
        - 

    """

    def __init__(self, config: Dict):
        super.__init__()

        # 

        self.encoder = PointNetEncoder(
            input_dim=config['encoder']['input_dim'],
            hidden_dims=config['encoder']['hidden_dims'],
            output_dim=config['encoder']['output_dim'],
            backbone_dim=config['encoder']['backbone_dim'],
            fusion_hidden_dim=config['encoder']['fusion_hidden_dim'],
            pooling_type=config['encoder']['pooling_type'],
            num_hierarchical_levels=config['encoder']['num_hierarchical_levels'],
            use_physics_params=config['encoder']['use_physics_params']
        )