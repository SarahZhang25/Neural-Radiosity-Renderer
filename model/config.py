"""
Structured configuration for the Neural Radiosity Renderer.

Defines frozen dataclasses for each model component, with defaults matching
the 46M-parameter configuration. Follows the same pattern as RenderFormer's
config.py but uses nested sub-dataclasses for logical grouping.

Usage:
    # Load from YAML (missing fields fall back to 46M defaults):
    config = NeuralRadiosityConfig.from_yaml("training/train_config_46M.yaml")

    # Access fields:
    config.encoder.hidden_dims       # [512, 1024, 1536, 2048]
    config.training.batch_size       # 64

    # Instantiate with all defaults (46M config):
    config = NeuralRadiosityConfig()
"""

from dataclasses import dataclass, field, asdict, fields
from typing import List, Literal, Optional, Union
import yaml

class _ConfigMixin:
    """Provides dict-like get() for backward-compatible access."""

    def get(self, key: str, default=None):
        return getattr(self, key, default)


@dataclass(frozen=True)
class PointNetConfig(_ConfigMixin):
    """Configuration for the PointNet-based point cloud encoder."""

    input_dim: int = 16
    """Per-point feature dimension: position (3) + normal (3) + material (10)."""

    hidden_dims: List[int] = field(default_factory=lambda: [512, 1024, 1536, 2048])
    """Hidden layer widths of the PointNet backbone Conv1d stack."""

    output_dim: int = 512
    """Dimensionality of the final object token produced per point cloud."""

    backbone_dim: int = 768
    """Internal feature width of the PointNet backbone before output projection."""

    pooling_type: Literal['hierarchical', 'hierarchical_rich', 'max', 'local_features'] = 'hierarchical'
    """Pooling strategy: 'hierarchical' multi-scale, 'hierarchical_rich' multi-scale with non-linearities, 'max' global, or 'local_features' set abstraction."""

    num_hierarchical_levels: int = 3
    """Number of hierarchical pooling levels (ignored when pooling_type != 'hierarchical')."""

    use_local_patches: bool = False
    """Whether to use FPS + k-NN set abstraction for local patch features."""

    num_centroids: int = 16
    """Number of FPS centroids when use_local_patches is True."""


@dataclass(frozen=True)
class LitePTConfig(_ConfigMixin):
    """Configuration for the LitePT point cloud encoder."""

    in_channels: int = 16
    """Input feature channels (e.g. 3 pos + 3 normal + 10 properties)."""

    out_channels: int = 512
    """Final global output dimension (matches pointnet's output_dim)."""
    
    pooling_type: Literal['hierarchical', 'max'] = 'max'
    """Pooling strategy: 'hierarchical' multi-scale or 'max' global."""

    num_hierarchical_levels: int = 3
    """Number of hierarchical pooling levels (ignored when pooling_type != 'hierarchical')."""
    
    pretrained_weights_path: Optional[str] = "LitePT-S.pth"
    """Path to the pretrained weights for the LitePT backbone."""

    drop_path: float = 0.2
    """Stochastic depth / drop path rate for LitePT."""

    """LitePT-S (Small) architecture"""
    stride: List[int] = field(default_factory=lambda: [2, 2, 2])
    enc_depths: List[int] = field(default_factory=lambda: [2, 2, 6, 2])
    enc_channels: List[int] = field(default_factory=lambda: [48, 96, 192, 384])
    enc_num_head: List[int] = field(default_factory=lambda: [2, 4, 8, 16])
    enc_patch_size: List[int] = field(default_factory=lambda: [1024, 1024, 1024, 1024])
    enc_conv: List[bool] = field(default_factory=lambda: [True, True, True, False])
    enc_attn: List[bool] = field(default_factory=lambda: [False, False, False, True])
    enc_rope_freq: List[float] = field(default_factory=lambda: [100.0, 100.0, 100.0, 100.0])


@dataclass(frozen=True)
class SceneTransformerConfig(_ConfigMixin):
    """Configuration for the view-independent scene transformer.

    Note: the corresponding YAML section is named 'decoder' for historical
    reasons, but the model attribute is ``scene_transformer``.
    """

    hidden_dim: int = 512
    """Token embedding dimension for the transformer."""

    ffn_hidden_dim: int = 2048
    """Hidden dimension of the feed-forward network in each transformer layer."""

    num_layers: int = 4
    """Number of transformer encoder layers."""

    num_heads: int = 4
    """Number of attention heads per layer."""

    dropout: float = 0.0
    """Dropout rate applied in attention and FFN layers."""

    activation: Literal['gelu', 'swiglu'] = 'gelu'
    """Activation function used in the FFN."""

    return_all_layers: bool = True
    """Whether to return intermediate layer outputs for multi-scale fusion."""

    use_self_attention: bool = True
    """Whether to include self-attention (always True for encoder-style transformer)."""

    norm_type: Literal['layer_norm', 'rms_norm'] = 'layer_norm'
    """Type of normalization layer."""

    qk_norm: bool = True
    """Whether to apply L2 normalization to query and key vectors."""

    bias: bool = True
    """Whether linear layers in the transformer include bias terms."""

    rope_double_max_freq: bool = False
    """Whether to double the maximum RoPE frequency."""

    num_register_tokens: int = 2
    """Number of learnable register tokens prepended to the sequence."""

    use_obj_obj_attention_bias: bool = False
    """Whether to use geometry-aware object-to-object attention bias."""

    obj_obj_bias_hidden_dim: int = 64
    """Hidden dimension of the geometry bias encoders."""


@dataclass(frozen=True)
class RayEncoderConfig(_ConfigMixin):
    """Configuration for the ray / view-direction encoder."""

    pe_type: Literal['rope_centroid', 'rope_obb', 'nerf'] = 'rope_obb'
    """Positional encoding type for spatial positions: 'rope_centroid' (3D), 'rope_obb' (12D OBB), or 'nerf'."""

    vertex_pe_num_freqs: int = 8
    """Number of RoPE / NeRF frequency bands for spatial position encoding."""

    vdir_pe_type: Literal['camray', 'nerf'] = 'camray'
    """Positional encoding type for view directions: 'camray' (raw camera-space rays) or 'nerf'."""

    vdir_num_freqs: int = 4
    """Number of NeRF frequency bands for view direction encoding (ignored when vdir_pe_type='camray')."""

    patch_size: int = 16
    """Spatial patch size (in pixels) for tokenizing the ray image."""

    norm_type: Literal['layer_norm', 'rms_norm'] = 'layer_norm'
    """Normalization type used after ray token projection."""

    view_transformer_latent_dim: int = 512
    """Latent dimension for the view transformer (must match predictor hidden_dim)."""

    view_transformer_n_heads: int = 4
    """Number of attention heads in the view transformer."""


@dataclass(frozen=True)
class PredictorConfig(_ConfigMixin):
    """Configuration for the radiance predictor transformer and DPT decoder."""

    hidden_dim: int = 512
    """Token embedding dimension for the predictor transformer."""

    ffn_hidden_dim: int = 1024
    """Hidden dimension of the feed-forward network in each predictor layer."""

    num_layers: int = 4
    """Number of cross-attention transformer layers in the predictor."""

    num_heads: int = 4
    """Number of attention heads per predictor layer."""

    dropout: float = 0.0
    """Dropout rate in the predictor transformer."""

    activation: Literal['gelu', 'swiglu'] = 'gelu'
    """Activation function used in the predictor FFN."""

    norm_type: Literal['layer_norm', 'rms_norm'] = 'layer_norm'
    """Normalization type in the predictor transformer."""

    pe_type: Literal['rope_centroid', 'rope_obb', 'nerf'] = 'rope_obb'
    """Positional encoding for the predictor cross-attention."""

    spherical_cross_attn_rope: bool = True
    """Whether to project the 3D centroids of objects and CamRays into a shared bounded spherical angular coordinate system for RoPE, natively supporting 3D-to-2D spatial alignment without projection singularities."""

    pe_num_freqs: int = 8
    """Number of frequency bands for the predictor positional encoding."""

    use_dpt_decoder: bool = True
    """Whether to use the DPT multi-scale decoder head instead of a simple linear head."""

    dpt_features: int = 128
    """Internal feature dimension of the DPT decoder."""

    dpt_out_channels: List[int] = field(default_factory=lambda: [64, 128, 256, 512])
    """Per-layer output channel counts in the DPT decoder."""

    include_alpha: bool = False
    """Whether to predict an alpha channel in addition to RGB."""


@dataclass(frozen=True)
class TrainingConfig(_ConfigMixin):
    """Configuration for the training loop and optimization."""

    global_batch_size: int = 128
    """Total effective batch size across all GPUs. Overrides batch_size if provided."""

    learning_rate: float = 1.0e-4
    """Peak learning rate for AdamW optimizer."""

    num_steps: Optional[int] = None
    """Total number of training steps (overrides num_epochs)."""

    warmup_steps: Optional[int] = None
    """Number of linear warmup steps before cosine annealing (overrides warmup_epochs)."""

    save_interval_steps: Optional[int] = None
    """Step interval for logging validation visualizations (overrides save_interval)."""
    
    log_interval_steps: int = 100
    """Step interval for logging training scalars to terminal and TensorBoard."""

    checkpoint_interval_steps: Optional[int] = None
    """Step interval for saving model checkpoints (overrides checkpoint_interval)."""

    image_res: int = 128
    """Rendering resolution (square, in pixels)."""

    

    batch_size: Optional[int] = None
    """Legacy: Batch size per GPU (use global_batch_size instead)."""

    num_epochs: int = 5000
    """Total number of training epochs (fallback if num_steps is None)."""

    warmup_epochs: int = 500
    """Number of linear warmup epochs before cosine annealing (fallback if warmup_steps is None)."""

    save_interval: int = 500
    """Epoch interval for logging validation visualizations (fallback)."""

    checkpoint_interval: int = 1000
    """Epoch interval for saving model checkpoints (fallback)."""



    data_dir: Union[str, List[str]] = "data_generation/output_auto/datasets/attempt6_table_chair_540"
    """Path to the training dataset directory."""

    log_dir: str = "training/logs/attempt6_table_chair_540"
    """Root directory for TensorBoard logs and checkpoints."""

    run_name: Optional[str] = "full_ds_params46M_camray"
    """Optional tag appended to the log directory name."""

    max_dataset_size: Optional[int] = None
    """Optional cap on the number of dataset samples (None = use all)."""

    shuffle_dataset: bool = True
    """Whether to shuffle the dataset before splitting."""

    shuffle_data_seed: int = 42
    """Random seed for dataset shuffling."""

    primary_loss: Literal['mse', 'mae'] = 'mae'
    """Primary reconstruction loss function."""

    lpips_backbone: Literal['alex', 'vgg'] = 'alex'
    """Backbone network for the LPIPS perceptual loss."""

    lpips_loss_weighting: float = 0.05
    """Weight of the LPIPS loss term relative to the primary loss."""

    use_amp: bool = True
    """Whether to use automatic mixed precision (bfloat16) during training."""

    use_compile: bool = False
    """Whether to apply torch.compile to the model."""

    package_model: bool = True
    """Whether to save a torch.package archive at the end of training."""

    num_workers: int = 16
    """Number of DataLoader worker processes."""
    
    cache_strategy: Literal['disk', 'ram', 'vram'] = 'disk'
    """How to cache dataset during training.
    "disk": reads directly from h5
    "ram": loads the whole dataset into CPU RAM
    "vram": loads the whole dataset into GPU memory
    """

    device: str = 'cuda'
    """Device to train on ('cuda' or 'cpu')."""

@dataclass(frozen=True)
class NeuralRadiosityConfig(_ConfigMixin):
    """Top-level configuration container for the Neural Radiosity Renderer.

    Groups all component configs. Instantiate with defaults (46M model) or
    load from a YAML file via ``NeuralRadiosityConfig.from_yaml(path)``.
    """

    encoder_type: Literal['pointnet', 'litept'] = 'pointnet'
    """Which point cloud encoder to use."""

    pointnet_encoder: PointNetConfig = field(default_factory=PointNetConfig)
    """PointNet encoder configuration."""

    litept_encoder: LitePTConfig = field(default_factory=LitePTConfig)
    """LitePT encoder configuration."""

    decoder: SceneTransformerConfig = field(default_factory=SceneTransformerConfig)
    """Scene transformer configuration (YAML key: 'decoder')."""

    ray_encoder: RayEncoderConfig = field(default_factory=RayEncoderConfig)
    """Ray / view-direction encoder configuration."""

    predictor: PredictorConfig = field(default_factory=PredictorConfig)
    """Radiance predictor transformer configuration."""

    training: TrainingConfig = field(default_factory=TrainingConfig)
    """Training hyperparameter configuration."""

    # ------------------------------------------------------------------
    # Factory & serialization
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str) -> 'NeuralRadiosityConfig':
        """Load config from a YAML file, merging with 46M defaults.

        Any fields not present in the YAML will use the default values.
        Handles backward-compatible key aliases (e.g. ``feedforward_dim``
        → ``ffn_hidden_dim``).
        """
        with open(path, 'r') as f:
            raw = yaml.safe_load(f) or {}

        def _apply_aliases(section: dict) -> dict:
            """Normalize legacy key names."""
            if 'feedforward_dim' in section and 'ffn_hidden_dim' not in section:
                section['ffn_hidden_dim'] = section.pop('feedforward_dim')
            # 'shuffle_data' -> 'shuffle_dataset' (seen in some configs)
            if 'shuffle_data' in section and 'shuffle_dataset' not in section:
                section['shuffle_dataset'] = section.pop('shuffle_data')
            return section

        def _build(dc_cls, raw_section: dict):
            """Build a dataclass, ignoring unknown keys from the YAML."""
            if raw_section is None:
                return dc_cls()
            section = _apply_aliases(dict(raw_section))
            valid_keys = {f.name for f in fields(dc_cls)}
            filtered = {k: v for k, v in section.items() if k in valid_keys}
            return dc_cls(**filtered)

        return cls(
            encoder_type=raw.get('encoder_type', 'pointnet'),
            pointnet_encoder=_build(PointNetConfig, raw.get('pointnet_encoder') or raw.get('encoder')),
            litept_encoder=_build(LitePTConfig, raw.get('litept_encoder')),
            decoder=_build(SceneTransformerConfig, raw.get('decoder')),
            ray_encoder=_build(RayEncoderConfig, raw.get('ray_encoder')),
            predictor=_build(PredictorConfig, raw.get('predictor')),
            training=_build(TrainingConfig, raw.get('training')),
        )

    def to_dict(self) -> dict:
        """Serialize the full config to a plain dict (suitable for YAML dump)."""
        return asdict(self)