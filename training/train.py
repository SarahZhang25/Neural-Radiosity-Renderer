import os
import time
import argparse
import yaml
from datetime import datetime
import logging
import shutil
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torch.utils.tensorboard import SummaryWriter
import torchvision
from lpips import LPIPS

from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from model.config import NeuralRadiosityConfig
from model.global_illumination_model import GlobalIlluminationModel
from training.dataset import (
    H5SceneDataset as SceneDataset, 
    scene_collate_fn,
    preload_to_ram,
    CpuCachedDataset,
    GpuCachedDataset
)
from training.ray_generator import RayGenerator

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32       = True
torch.backends.cudnn.benchmark        = True

# Suppress harmless FX tracing warning from spconv
logging.getLogger("torch.fx._symbolic_trace").setLevel(logging.ERROR)

def log_transform(x):
    """
    Applies a log transform to the input image tensor for better handling of HDR values in L1 loss.
    Clamp explicitly to prevent log(x < 0) from producing NaN gradients.
    Use log(1 + x) for smooth gradients and true zero-anchoring.
    """
    x = torch.clamp(x, min=0.0)
    return torch.log1p(x)

def linear_to_srgb(x: torch.Tensor) -> torch.Tensor:
    a = 0.055
    x = torch.clamp(x, min=0.0, max=1.0)
    return torch.where(x <= 0.0031308, 12.92 * x, (1 + a) * torch.pow(x, 1/2.4) - a)

def tone_map_reinhard(x, exposure=1.0):
    x = x * exposure
    return x / (x + 1.0)

def to_uint8(x):
    """Convert float image [0, 1] to uint8 [0, 255]."""
    x = torch.clamp(x, 0.0, 1.0)
    return (x * 255).byte()

def hdr_to_ldr(x, method="reinhard", exposure=1.0, to_uint8_output=True):
    """
    Convert HDR image to LDR for visualization using specified tone mapping method.
    """
    x = tone_map_reinhard(x, exposure=exposure)
    x = linear_to_srgb(x)
    if to_uint8_output:
        x = to_uint8(x)
    return x

# RenderFormer LPIPS Tone-mapping as described in RenderFormer paper: "clamp (log I / log 2, 0, 1)"
# Note: log(I) / log(2) is mathematically identical to log2(I).
# def tone_map_lpips(x):
#     """
#     Tone mapping to apply before LPIPS
#     """
#     # We clamp the input to a minimum of 1.0 before log2 so that log2(1) = 0.
#     # This maps the darkest areas to 0, and specular highlights (up to 2.0) to 1.
#     # Then scale to [-1, 1] for LPIPS input.
#     log2_x = torch.log2(x + 1.0)
#     return torch.clamp(log2_x, min=0.0, max=1.0) * 2 - 1

def tone_map_lpips(x):
    """
    Smooth Reinhard-style tone mapping for HDR LPIPS.
    Preserves gradients for extremely bright light sources.
    """
    mapped = x / (x + 1.0) 
    return mapped * 2.0 - 1.0

def calculate_psnr(pred, target):
    """
    Calculate PSNR on tone-mapped LDR images in [0, 1] range to match human perception.
    """
    with torch.no_grad():
        p = linear_to_srgb(tone_map_reinhard(torch.clamp(pred, min=0.0)))
        t = linear_to_srgb(tone_map_reinhard(torch.clamp(target, min=0.0)))
        mse = torch.nn.functional.mse_loss(p, t)
        if mse == 0:
            return torch.tensor(100.0, device=pred.device) # arbitrary high upper bound
        psnr = -10.0 * torch.log10(mse)
        return psnr

def make_vis_grid(linear_rendered, gt_img, max_images=16, diff_amplify=5.0):
    """
    Build a side-by-side (GT | pred | diff) visualization grid.
    Inputs are assumed to be [B, C, H, W] HDR images.
    """
    pred = linear_rendered.detach().cpu()[:max_images]
    gt   = gt_img.detach().cpu()[:max_images]

    # Tone-map to float [0, 1] first so the diff is in a perceptually uniform space
    pred_ldr = hdr_to_ldr(pred.clamp(min=0.0), to_uint8_output=False)  # [N, C, H, W]
    gt_ldr   = hdr_to_ldr(gt.clamp(min=0.0), to_uint8_output=False)

    # Absolute difference, amplified and clamped to [0, 1]
    diff = (pred_ldr - gt_ldr).abs().mul(diff_amplify).clamp(0.0, 1.0)

    vis_pred = to_uint8(pred_ldr)
    vis_gt   = to_uint8(gt_ldr)
    vis_diff = to_uint8(diff)

    vis_img = torch.cat([vis_gt, vis_pred , vis_diff], dim=3)  # [N, C, H, W*3]
    grid = torchvision.utils.make_grid(vis_img.float(), nrow=1, normalize=False)
    return grid.byte()


class Trainer:
    def __init__(self, config_path: str, resume_path: str = None, synthetic: bool = False, profile: bool = False):
        """
        Initialize the trainer with configuration parameters and model setup.
        Sets up datasets, model, optimizer, scheduler, and logging directories.
        """
        import sys, io, warnings
        
        self.is_distributed = "LOCAL_RANK" in os.environ
        if self.is_distributed:
            dist.init_process_group(backend="nccl")
            self.local_rank = int(os.environ["LOCAL_RANK"])
            self.is_main_process = self.local_rank == 0
            self.device = torch.device(f'cuda:{self.local_rank}')
            torch.cuda.set_device(self.device)
        else:
            self.local_rank = 0
            self.is_main_process = True
        
        # Silence all stdout and Python warnings from non-main ranks for the
        # entire init. This prevents duplicate LPIPS/model prints from each rank.
        if not self.is_main_process:
            sys.stdout = io.StringIO()
            warnings.filterwarnings('ignore')
            
        self.config = NeuralRadiosityConfig.from_yaml(config_path)
        tc = self.config.training
        
        if not self.is_distributed:
            self.device = torch.device(tc.device if torch.cuda.is_available() or tc.device != 'cuda' else 'cpu')
            
        if self.is_main_process:
            print(f"CUDA Available: {torch.cuda.is_available()}")
            print(f"Distributed training: {self.is_distributed}")
            print(f"Using device: {self.device}")

        self.use_amp = tc.use_amp
        self.use_compile = tc.use_compile
        self.package_model = tc.package_model
        self.use_synthetic_data = synthetic
        self.profile = profile

        # Datasets - each rank constructs its own dataset index.
        self.train_dataset = SceneDataset(
            data_dir=tc.data_dir,
            image_res=tc.image_res,
            split='train',
            max_dataset_size=tc.max_dataset_size,
            shuffle=tc.shuffle_dataset,
            shuffle_seed=tc.shuffle_data_seed
        )
        self.val_dataset = SceneDataset(
            data_dir=tc.data_dir,
            image_res=tc.image_res,
            split='val',
            max_dataset_size=tc.max_dataset_size,
            shuffle=tc.shuffle_dataset,
            shuffle_seed=tc.shuffle_data_seed
        )
        
        if tc.cache_strategy == 'ram':
            self.train_dataset = CpuCachedDataset(preload_to_ram(self.train_dataset, label="CPU-RAM"))
            self.val_dataset = CpuCachedDataset(preload_to_ram(self.val_dataset, label="CPU-RAM"))
        elif tc.cache_strategy == 'vram':
            self.train_dataset = GpuCachedDataset(preload_to_ram(self.train_dataset, label="GPU-VRAM"), self.device)
            self.val_dataset = GpuCachedDataset(preload_to_ram(self.val_dataset, label="GPU-VRAM"), self.device)
        
        world_size = dist.get_world_size() if self.is_distributed else 1
        
        if hasattr(tc, 'global_batch_size') and tc.global_batch_size is not None:
            self.global_batch_size = tc.global_batch_size
            assert self.global_batch_size % world_size == 0, f"Global batch size {self.global_batch_size} must be divisible by world_size {world_size}"
            per_device_batch_size = self.global_batch_size // world_size
        else:
            per_device_batch_size = tc.batch_size
            self.global_batch_size = per_device_batch_size * world_size

        if self.is_main_process:
            print(f"Global Batch Size: {self.global_batch_size} (Per GPU: {per_device_batch_size})")

        val_batch_size = min(4, per_device_batch_size)
        
        # When caching in RAM/VRAM, we don't need workers or pinned memory logic
        num_workers = tc.num_workers if tc.cache_strategy == 'disk' else 0
        pin_mem = (tc.cache_strategy != 'vram')
        persistent = (tc.cache_strategy == 'disk')
        
        self.train_sampler = DistributedSampler(self.train_dataset, shuffle=True) if self.is_distributed else None
        self.train_loader = DataLoader(
            self.train_dataset, 
            batch_size=per_device_batch_size,  
            shuffle=(self.train_sampler is None), 
            sampler=self.train_sampler,
            num_workers=num_workers,
            pin_memory=pin_mem,
            collate_fn=scene_collate_fn,
            persistent_workers=persistent
        )
        
        self.val_sampler = DistributedSampler(self.val_dataset, shuffle=False) if self.is_distributed else None
        self.val_loader = DataLoader(
            self.val_dataset, 
            batch_size=val_batch_size, 
            shuffle=False, 
            sampler=self.val_sampler,
            num_workers=num_workers,
            pin_memory=pin_mem,
            collate_fn=scene_collate_fn,
            persistent_workers=persistent
        )

        if self.use_synthetic_data:
            if self.is_main_process:
                print("Running in SYNTHETIC data mode. Bypassing DataLoader and using a single cached GPU batch.")
            
            real_batch = next(iter(self.train_loader))
            self.synthetic_batch = {k: v.to(self.device) for k, v in real_batch.items()}
            
            self.train_loader = [self.synthetic_batch] * len(self.train_loader)
            self.val_loader = []

        # Fixed batches for visualization
        self.fixed_train_batch = self._get_seeded_batch(self.train_dataset, min(16, per_device_batch_size), seed=456)
        self.fixed_val_batch = self._get_seeded_batch(self.val_dataset, min(16, per_device_batch_size), seed=123)

        # Model
        self.model = GlobalIlluminationModel(self.config).to(self.device)
        self.ray_generator = RayGenerator().to(self.device)
        self.image_res = tc.image_res
        
        if self.use_compile:
            if self.is_main_process:
                print("Compiling model with torch.compile (mode='reduce-overhead')...")
            self.model = torch.compile(self.model, mode='reduce-overhead')
            
        if self.is_distributed:
            # gradient_as_bucket_view=True resolves the "grad strides do not match
            # bucket view strides" warning by ensuring DDP reuses the gradient memory directly as the bucket view.
            # bucket_cap_mb=100 increases the gradient synchronization chunk size,
            # which optimizes throughput on high-bandwidth NVLink topologies.
            self.model = DDP(self.model, device_ids=[self.local_rank],
                             find_unused_parameters=True,
                             gradient_as_bucket_view=True,
                             bucket_cap_mb=100)

        # Optimizer & Losses
        self.optimizer = AdamW(self.model.parameters(), lr=tc.learning_rate)
        
        self.primary_loss_type = tc.primary_loss
        if self.primary_loss_type == 'mse':
            self.core_loss_fn = nn.MSELoss()
        elif self.primary_loss_type == 'mae':
            self.core_loss_fn = nn.L1Loss()
        else:
            raise ValueError(f"Unsupported primary loss type: {self.primary_loss_type}")
            
        self.lpips_weight = tc.lpips_loss_weighting
        if self.lpips_weight > 0:
            self.lpips_loss_fn = LPIPS(net=tc.lpips_backbone).to(self.device)
            
        # Metrics
        self.ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.lpips_val_metric = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).to(self.device)

        # Compute steps/epochs
        self.steps_per_epoch = len(self.train_loader)
        
        if tc.num_steps is not None:
            self.num_steps = tc.num_steps
            self.num_epochs = (self.num_steps + self.steps_per_epoch - 1) // self.steps_per_epoch
        else:
            self.num_epochs = tc.num_epochs
            self.num_steps = self.num_epochs * self.steps_per_epoch

        if tc.warmup_steps is not None:
            self.warmup_steps = tc.warmup_steps
        else:
            self.warmup_steps = tc.warmup_epochs * self.steps_per_epoch

        if tc.save_interval_steps is not None:
            self.log_viz_interval_steps = tc.save_interval_steps
        else:
            self.log_viz_interval_steps = tc.save_interval * self.steps_per_epoch

        self.log_interval_steps = tc.log_interval_steps

        if tc.checkpoint_interval_steps is not None:
            self.checkpoint_interval_steps = tc.checkpoint_interval_steps
        else:
            self.checkpoint_interval_steps = tc.checkpoint_interval * self.steps_per_epoch

        # Scheduler
        warmup_scheduler = LinearLR(self.optimizer, start_factor=0.01, total_iters=self.warmup_steps)
        cosine_scheduler = CosineAnnealingLR(self.optimizer, T_max=self.num_steps - self.warmup_steps)
        self.scheduler = SequentialLR(self.optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[self.warmup_steps])

        log_dir_root = tc.log_dir
        self.start_epoch = 0
        self.global_step = 0

        if resume_path and os.path.exists(resume_path):
            self.checkpoint_dir = os.path.dirname(resume_path)
            self.log_dir = os.path.dirname(self.checkpoint_dir)
            if self.is_main_process:
                print(f"Resuming run. Appending logs to existing directory: {self.log_dir}")
            self._load_checkpoint(resume_path)
        else:
            run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
            if tc.run_name:
                run_id += f"_{tc.run_name}"
            self.log_dir = os.path.join(log_dir_root, run_id)
            self.checkpoint_dir = os.path.join(self.log_dir, 'checkpoints')
            
            if self.is_main_process:
                os.makedirs(self.log_dir, exist_ok=True)
                os.makedirs(self.checkpoint_dir, exist_ok=True)
                print("Logging to new directory:", self.log_dir)
                shutil.copy(config_path, os.path.join(self.log_dir, 'config.yaml'))


        if self.is_main_process:
            self.writer = SummaryWriter(log_dir=self.log_dir)
            
            # Print configuration summary
            world_size = dist.get_world_size() if self.is_distributed else 1
            summary = (
                f"\n{'='*50}\n"
                f"Batching Info:\n"
                f"\tGlobal Batch Size:     {self.global_batch_size}\n"
                f"\tNumber of GPUs:        {world_size}\n"
                f"\tPer-GPU Batch Size:    {per_device_batch_size}\n"
                f"{'='*50}\n"
            )
            print(summary)
            self.writer.add_text("Batching Info", summary.replace('\n', '  \n'), 0)
            
            # Log the raw YAML configuration to TensorBoard
            with open(config_path, 'r') as f:
                config_yaml_str = f.read()
            self.writer.add_text("Model Config YAML", f"```yaml\n{config_yaml_str}\n```", 0)
        else:
            self.writer = None

    def _get_seeded_batch(self, dataset, batch_size: int, seed: int = 42) -> dict:
        """
        Retrieve a fixed, reproducible batch from a dataset.
        Useful for generating consistent visualizations across epochs.
        """
        g = torch.Generator()
        g.manual_seed(seed)
        loader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=0,
            generator=g,
            collate_fn=scene_collate_fn
        )
        batch = next(iter(loader))
        return {k: v.to(self.device) for k, v in batch.items()}

    def _load_checkpoint(self, path: str):
        """
        Load model, optimizer, scheduler state, and epoch count from a checkpoint file.
        """
        if self.is_main_process:
            print(f"Resuming training from checkpoint: {path}")
        # When mapping location, we want to map to the local rank's device
        checkpoint = torch.load(path, map_location=self.device)
        
        if self.is_distributed:
            # Checkpoint contains un-wrapped DDP state dict if saved properly, or wrapped if saved directly
            # We'll let torch handle it, but if it fails we might need to adjust keys
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1 
        self.global_step = checkpoint.get('global_step', self.start_epoch * self.steps_per_epoch)
        
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        else:
            if self.is_main_process:
                print("No scheduler state found in checkpoint. Fast-forwarding...")
            for _ in range(self.global_step):
                self.scheduler.step()

    def _forward(self, batch: dict) -> torch.Tensor:
        """
        Perform a forward pass through the model using the provided batch of data.
        Handles ray generation and optional mixed-precision context.
        """
        c2w = batch['c2w'].to(self.device)          # (B, 4, 4)
        fov_deg = batch['fov_deg'].to(self.device)  # (B,)
        fov_rad = (fov_deg * (torch.pi / 180.0)).unsqueeze(-1)  # (B, 1)

        rays_o, rays_d = self.ray_generator(c2w, fov_rad, self.image_res)
        # rays_o: (B, 3)  rays_d: (B, H, W, 3)

        w2c = torch.inverse(c2w)  # (B, 4, 4)

        kwargs = dict(
            rays_o=rays_o,
            rays_d=rays_d,
            obj_positions=batch['obj_positions'].to(self.device),
            obj_properties=batch['obj_properties'].to(self.device),
            obj_normals=batch['obj_normals'].to(self.device),
            obj_mask=batch.get('obj_mask').to(self.device) if 'obj_mask' in batch else None,
            w2c=w2c,
        )
        if self.use_amp:
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                return self.model(**kwargs)
        return self.model(**kwargs)

    def _compute_loss(self, pred: torch.Tensor, target: torch.Tensor):
        """
        Calculate the combined loss (e.g., L1 and LPIPS) between the predicted
        and target HDR images using log-transformed values.
        """
        # Base core loss on log-transformed HDR predictions. Ensure float32.
        pred_f32 = pred.float()
        target_f32 = target.float()
        
        loss_l1 = self.core_loss_fn(log_transform(pred_f32), log_transform(target_f32))
        
        if self.lpips_weight > 0:
            pred_mapped = tone_map_lpips(pred_f32)
            target_mapped = tone_map_lpips(target_f32)
            loss_lpips = self.lpips_loss_fn(pred_mapped, target_mapped).mean()
            loss = loss_l1 + self.lpips_weight * loss_lpips
        else:
            loss = loss_l1
            
        return loss

    def _train_epoch(self, epoch: int, pbar=None):
        """
        Run one full training epoch over the GPU-cached inputs.
        Shuffles the cached list each epoch to preserve stochastic ordering.
        Logs metrics and checkpoints based on global_step.
        """
        self.model.train()
        _log_timer = time.perf_counter()
        _log_step_count = 0
        
        for batch in self.train_loader:
            if self.global_step >= self.num_steps:
                break
                
            target = batch['target_image'].to(self.device, non_blocking=True)
            batch = {k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            self.optimizer.zero_grad()
            
            pred_radiance = self._forward(batch)
            loss = self._compute_loss(pred_radiance, target)
            
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            _log_step_count += 1
            
            # Only log at intervals. Use rank-0 local metrics (no all_reduce)
            # to avoid CUDA syncs and collective communication. The local values
            # track the same trends as the global mean; full metrics are computed
            # properly during validation.
            if self.is_main_process and (self.global_step % self.log_interval_steps == 0
                                         or self.global_step == self.num_steps - 1):
                with torch.no_grad():
                    pred_f32 = pred_radiance.float()
                    target_f32 = target.float()

                    step_loss = loss.item()
                    step_psnr = calculate_psnr(pred_f32, target_f32).item()

                    pred_ldr = hdr_to_ldr(pred_f32, to_uint8_output=False)
                    target_ldr = hdr_to_ldr(target_f32, to_uint8_output=False)
                    step_ssim = self.ssim_metric(pred_ldr, target_ldr).item()
                    step_lpips = self.lpips_val_metric(pred_ldr, target_ldr).item()

                self.writer.add_scalar('Train/LR', self.optimizer.param_groups[0]['lr'], self.global_step)
                self.writer.add_scalar('Loss/train', step_loss, self.global_step)
                self.writer.add_scalar('PSNR/train', step_psnr, self.global_step)
                self.writer.add_scalar('SSIM/train', step_ssim, self.global_step)
                self.writer.add_scalar('LPIPS/train', step_lpips, self.global_step)

                now = time.perf_counter()
                avg_ms = (now - _log_timer) / _log_step_count * 1000
                _log_timer = now
                _log_step_count = 0
                self.writer.add_scalar('Train/StepTime_ms', avg_ms, self.global_step)

                if pbar is not None:
                    pbar.set_postfix({'loss': f"{step_loss:.4f}", 'psnr': f"{step_psnr:.2f}", 'ms/step': f"{avg_ms:.0f}"})

            self.global_step += 1
            
            if self.global_step % self.log_viz_interval_steps == 0:
                self._validate(self.global_step)
                
            if self.is_main_process and (self.global_step == self.num_steps or self.global_step % self.checkpoint_interval_steps == 0):
                self._save_checkpoint(epoch)
                
        # Profiler step could go here, but usually it's per batch. Since the user doesn't strictly need it, we'll leave it as is or move it.

    def _validate(self, step: int):
        """
        Evaluate the model on the validation set and compute metrics (Loss, PSNR, SSIM, LPIPS).
        Logs results and visualizations to TensorBoard.
        """
        self.model.eval()
        val_loss = torch.tensor(0.0, device=self.device)
        val_psnr = torch.tensor(0.0, device=self.device)
        val_ssim, val_lpips = 0.0, 0.0
        
        with torch.no_grad():
            # Full validation set metrics
            for batch in self.val_loader:
                target = batch['target_image'].to(self.device, non_blocking=True)
                batch = {k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                pred_radiance = self._forward(batch)
                
                loss = self._compute_loss(pred_radiance, target)
                
                pred_f32 = pred_radiance.float()
                target_f32 = target.float()
                
                val_loss += loss.detach()
                val_psnr += calculate_psnr(pred_f32, target_f32)
                
                pred_ldr = hdr_to_ldr(pred_f32, to_uint8_output=False)
                target_ldr = hdr_to_ldr(target_f32, to_uint8_output=False)
                val_ssim += self.ssim_metric(pred_ldr, target_ldr).item()
                val_lpips += self.lpips_val_metric(pred_ldr, target_ldr).item()
                
            n = len(self.val_loader)
            avg_val_loss = val_loss / n
            avg_val_psnr = val_psnr / n
            avg_val_ssim = val_ssim / n
            avg_val_lpips = val_lpips / n
            
            if self.is_distributed:
                metrics = torch.stack([avg_val_loss, avg_val_psnr, torch.tensor(avg_val_ssim, device=self.device), torch.tensor(avg_val_lpips, device=self.device)])
                dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
                metrics /= dist.get_world_size()
                avg_val_loss, avg_val_psnr, avg_val_ssim, avg_val_lpips = metrics.tolist()
            else:
                avg_val_loss = avg_val_loss.item()
                avg_val_psnr = avg_val_psnr.item()

            if self.is_main_process:
                self.writer.add_scalar('Loss/val', avg_val_loss, step)
                self.writer.add_scalar('PSNR/val', avg_val_psnr, step)
                self.writer.add_scalar('SSIM/val', avg_val_ssim, step)
                self.writer.add_scalar('LPIPS/val', avg_val_lpips, step)
    
                print(f"Step {step}: Val Loss {avg_val_loss:.6f}, Val PSNR: {avg_val_psnr:.2f} dB")
    
                # Visualizations
                pred_fixed = self._forward(self.fixed_val_batch)
                grid_val = make_vis_grid(pred_fixed.float(), self.fixed_val_batch['target_image'].float(), max_images=16)
                self.writer.add_image('Visual/Validation', grid_val, step)
    
                pred_train_fixed = self._forward(self.fixed_train_batch)
                grid_train = make_vis_grid(pred_train_fixed.float(), self.fixed_train_batch['target_image'].float(), max_images=16)
                self.writer.add_image('Visual/Training', grid_train, step)

            # # Debug stats at epoch 500
            # if (epoch + 1) == 500:
            #     target_hdr = self.fixed_val_batch['target_image'].float()
            #     pred_f32 = pred_fixed.float()
            #     pred_log = log_transform(pred_f32 + 1e-6)
            #     target_log = log_transform(target_hdr + 1e-6)

            #     print(f"Pred range: [{pred_f32.min():.3f}, {pred_f32.max():.3f}]")
            #     print(f"Target range: [{target_hdr.min():.3f}, {target_hdr.max():.3f}]")
            #     print(f"Pred log range: [{pred_log.min():.3f}, {pred_log.max():.3f}]")
            #     print(f"Target log range: [{target_log.min():.3f}, {target_log.max():.3f}]")

            #     num_negative = (pred_f32 < 0).sum().item()
            #     print(f"Negative predictions: {num_negative} / {pred_f32.numel()}")
        self.model.train()

    def _save_checkpoint(self, epoch: int):
        print(f"saving model at step {self.global_step}...")
        ckpt_path = os.path.join(self.checkpoint_dir, f"checkpoint_step_{self.global_step}.pt")
        tmp_path = ckpt_path + ".tmp"
        
        # Unpack DDP model state dict for saving
        model_state_dict = self.model.module.state_dict() if self.is_distributed else self.model.state_dict()
        
        try:
            torch.save({
                'epoch': epoch,
                'global_step': self.global_step,
                'model_state_dict': model_state_dict,
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
            }, tmp_path)
            os.replace(tmp_path, ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")
            
            # Cleanup old checkpoints
            for old_file in os.listdir(self.checkpoint_dir):
                old_path = os.path.join(self.checkpoint_dir, old_file)
                if old_path != ckpt_path and old_file.endswith('.pt') and not old_file.startswith('model_package_'):
                    os.remove(old_path)
                    print(f"  Removed old checkpoint: {old_file}")
                    
        except (RuntimeError, OSError) as e:
            print(f"WARNING: Failed to save checkpoint at step {self.global_step}: {e}")
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        # Save torch.package for inference
        if self.package_model and self.global_step == self.num_steps:
            pkg_path = os.path.join(self.checkpoint_dir, f"model_package_step_{self.global_step}.pt")
            try:
                print(f"Packaging model to {pkg_path}...")
                with torch.package.PackageExporter(pkg_path) as exp:
                    exp.intern("model.**")
                    exp.intern("utils.**")
                    exp.intern("pos_encodings.**")
                    exp.extern("**")
                    # Save the un-wrapped model
                    unwrapped_model = self.model.module if self.is_distributed else self.model
                    exp.save_pickle("model", "model.pkl", unwrapped_model)
                print(f"Saved packaged model to {pkg_path}")
            except Exception as e:
                print(f"WARNING: Failed to package model: {e}")

    def run(self):
        """
        Execute the full training loop including recording metrics, validation and checkpointing.
        """
        pbar = tqdm(range(self.start_epoch, self.num_epochs), desc="Epochs", disable=not self.is_main_process)
        for epoch in pbar:
            if self.is_distributed and self.train_sampler is not None:
                self.train_sampler.set_epoch(epoch)

            self._train_epoch(epoch, pbar)
            
            if self.global_step >= self.num_steps:
                break

        if self.is_main_process:
            self.writer.close()
            print("Training Complete.")
            
        if self.is_distributed:
            dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='training/train_config.yaml', help='Path to config file')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--synthetic', action='store_true', help='Use a synthetic dataset (single repeated batch on GPU) for benchmarking')
    parser.add_argument('--profile', action='store_true', help='Enable torch.profiler to dump a trace of the first few epochs')
    args = parser.parse_args()
    
    Trainer(args.config, args.resume, args.synthetic, args.profile).run()
