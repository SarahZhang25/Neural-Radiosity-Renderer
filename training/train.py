import os
import argparse
import shutil
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torchvision
from datetime import datetime
from lpips import LPIPS

from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from model.global_illumination_model import GlobalIlluminationModel
from training.dataset import SceneDataset
from training.ray_generator import RayGenerator

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32       = True
torch.backends.cudnn.benchmark        = True

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
            return 100.0 # arbitrary high upper bound
        psnr = -10.0 * torch.log10(mse)
        return psnr.item()

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
    def __init__(self, config_path: str, resume_path: str = None):
        print(f"CUDA Available: {torch.cuda.is_available()}")
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.tc = self.config['training']
        self.device = torch.device(self.tc.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        print(f"Using device: {self.device}")

        self.use_amp = self.tc.get('use_amp', True)
        self.use_compile = self.tc.get('use_compile', False)

        # Datasets
        self.train_dataset = SceneDataset(
            data_dir=self.tc['data_dir'],
            image_res=self.tc['image_res'],
            split='train',
            max_dataset_size=self.tc.get('max_dataset_size', None),
            shuffle=self.tc.get('shuffle_dataset', True),
            shuffle_seed=self.tc.get('shuffle_seed', 42)
        )
        self.val_dataset = SceneDataset(
            data_dir=self.tc['data_dir'],
            image_res=self.tc['image_res'],
            split='val',
            max_dataset_size=self.tc.get('max_dataset_size', None),
            shuffle=self.tc.get('shuffle_dataset', True),
            shuffle_seed=self.tc.get('shuffle_seed', 42)
        )
        
        batch_size = self.tc['batch_size']
        val_batch_size = min(4, batch_size)
        num_workers = self.tc.get('num_workers', 4)
        
        self.train_loader = DataLoader(
            self.train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=num_workers,
            pin_memory=True
        )
        self.val_loader = DataLoader(
            self.val_dataset, 
            batch_size=val_batch_size, 
            shuffle=False, 
            num_workers=num_workers,
            pin_memory=True
        )

        # Fixed batches for visualization
        self.fixed_train_batch = self._get_seeded_batch(self.train_dataset, min(16, batch_size), seed=456)
        self.fixed_val_batch = self._get_seeded_batch(self.val_dataset, min(16, batch_size), seed=123)

        # Model
        self.model = GlobalIlluminationModel(self.config).to(self.device)
        self.ray_generator = RayGenerator().to(self.device)
        self.image_res = self.tc['image_res']
        
        if self.use_compile:
            print("Compiling model with torch.compile (mode='reduce-overhead')...")
            self.model = torch.compile(self.model, mode='reduce-overhead')

        # Optimizer & Losses
        self.optimizer = AdamW(self.model.parameters(), lr=float(self.tc['learning_rate']))
        
        self.primary_loss_type = self.tc.get('primary_loss', 'mse')
        if self.primary_loss_type == 'mse':
            self.core_loss_fn = nn.MSELoss()
        elif self.primary_loss_type == 'mae':
            self.core_loss_fn = nn.L1Loss()
        else:
            raise ValueError(f"Unsupported primary loss type: {self.primary_loss_type}")
            
        self.lpips_weight = self.tc.get('lpips_loss_weighting', 0)
        if self.lpips_weight > 0:
            self.lpips_loss_fn = LPIPS(net=self.tc.get('lpips_backbone', 'alex')).to(self.device)
            
        # Metrics
        self.ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.lpips_val_metric = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).to(self.device)

        # Training params
        self.num_epochs = self.tc['num_epochs']
        self.warmup_epochs = self.tc.get('warmup_epochs', self.num_epochs // 20)
        self.log_viz_interval = self.tc.get('save_interval', 100)
        self.checkpoint_interval = self.tc.get('checkpoint_interval', 500)

        # Scheduler
        warmup_scheduler = LinearLR(self.optimizer, start_factor=0.01, total_iters=self.warmup_epochs)
        cosine_scheduler = CosineAnnealingLR(self.optimizer, T_max=self.num_epochs - self.warmup_epochs)
        self.scheduler = SequentialLR(self.optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[self.warmup_epochs])

        # Logging setup
        log_dir_root = self.tc['log_dir']
        self.start_epoch = 0

        if resume_path and os.path.exists(resume_path):
            self.checkpoint_dir = os.path.dirname(resume_path)
            self.log_dir = os.path.dirname(self.checkpoint_dir)
            print(f"Resuming run. Appending logs to existing directory: {self.log_dir}")
            self._load_checkpoint(resume_path)
        else:
            run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
            if 'run_name' in self.tc:
                run_id += f"_{self.tc['run_name']}"
            self.log_dir = os.path.join(log_dir_root, run_id)
            self.checkpoint_dir = os.path.join(self.log_dir, 'checkpoints')
            os.makedirs(self.log_dir, exist_ok=True)
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            print("Logging to new directory:", self.log_dir)
            shutil.copy(config_path, os.path.join(self.log_dir, 'config.yaml'))

        self.writer = SummaryWriter(log_dir=self.log_dir)

    def _get_seeded_batch(self, dataset, batch_size: int, seed: int = 42) -> dict:
        g = torch.Generator()
        g.manual_seed(seed)
        loader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=0,
            generator=g
        )
        batch = next(iter(loader))
        return {k: v.to(self.device) for k, v in batch.items()}

    def _load_checkpoint(self, path: str):
        print(f"Resuming training from checkpoint: {path}")
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1 
        
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        else:
            print("No scheduler state found in checkpoint. Fast-forwarding...")
            for _ in range(self.start_epoch):
                self.scheduler.step()

    def _forward(self, batch: dict) -> torch.Tensor:
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
            w2c=w2c,
        )
        if self.use_amp:
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                return self.model(**kwargs)
        return self.model(**kwargs)

    def _compute_loss(self, pred: torch.Tensor, target: torch.Tensor):
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

    def _train_epoch(self) -> tuple[float, float, float, float]:
        self.model.train()
        train_loss, train_psnr, train_ssim, train_lpips = 0.0, 0.0, 0.0, 0.0
        
        for batch in self.train_loader:
            target = batch['target_image'].to(self.device)
            
            self.optimizer.zero_grad()
            
            pred_radiance = self._forward(batch)
            loss = self._compute_loss(pred_radiance, target)
            
            loss.backward()
            self.optimizer.step()
            
            with torch.no_grad():
                pred_f32 = pred_radiance.float()
                target_f32 = target.float()
                
                loss_item = loss.item()
                psnr_item = calculate_psnr(pred_f32, target_f32)

                pred_ldr = hdr_to_ldr(pred_f32, to_uint8_output=False)
                target_ldr = hdr_to_ldr(target_f32, to_uint8_output=False)
                        
                train_loss += loss_item
                train_psnr += psnr_item
                train_ssim += self.ssim_metric(pred_ldr, target_ldr).item()
                train_lpips += self.lpips_val_metric(pred_ldr, target_ldr).item()

        n = len(self.train_loader)
        return train_loss / n, train_psnr / n, train_ssim / n, train_lpips / n

    def _validate(self, epoch: int):
        self.model.eval()
        val_loss, val_psnr, val_ssim, val_lpips = 0.0, 0.0, 0.0, 0.0
        
        with torch.no_grad():
            # Full validation set metrics
            for batch in self.val_loader:
                target = batch['target_image'].to(self.device)
                pred_radiance = self._forward(batch)
                
                loss = self._compute_loss(pred_radiance, target)
                
                pred_f32 = pred_radiance.float()
                target_f32 = target.float()
                
                val_loss += loss.item()
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

            self.writer.add_scalar('Loss/val', avg_val_loss, epoch)
            self.writer.add_scalar('PSNR/val', avg_val_psnr, epoch)
            self.writer.add_scalar('SSIM/val', avg_val_ssim, epoch)
            self.writer.add_scalar('LPIPS/val', avg_val_lpips, epoch)

            print(f"Epoch {epoch+1}: Val Loss {avg_val_loss:.6f}, Val PSNR: {avg_val_psnr:.2f} dB")

            # Visualizations
            pred_fixed = self._forward(self.fixed_val_batch)
            grid_val = make_vis_grid(pred_fixed.float(), self.fixed_val_batch['target_image'].float(), max_images=16)
            self.writer.add_image('Visual/Validation', grid_val, epoch)

            pred_train_fixed = self._forward(self.fixed_train_batch)
            grid_train = make_vis_grid(pred_train_fixed.float(), self.fixed_train_batch['target_image'].float(), max_images=16)
            self.writer.add_image('Visual/Training', grid_train, epoch)

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

    def run(self):
        for epoch in tqdm(range(self.start_epoch, self.num_epochs), desc="Epochs"):
            avg_train_loss, avg_train_psnr, avg_train_ssim, avg_train_lpips = self._train_epoch()

            self.scheduler.step()
            
            self.writer.add_scalar('Train/LR', self.optimizer.param_groups[0]['lr'], epoch)
            self.writer.add_scalar('Loss/train', avg_train_loss, epoch)
            self.writer.add_scalar('PSNR/train', avg_train_psnr, epoch)
            self.writer.add_scalar('SSIM/train', avg_train_ssim, epoch)
            self.writer.add_scalar('LPIPS/train', avg_train_lpips, epoch)
            # TODO: add FLIP metric

            if (epoch + 1) % self.log_viz_interval == 0:
                self._validate(epoch)

            # Save checkpoint
            if (epoch + 1) == self.num_epochs or (epoch + 1) % self.checkpoint_interval == 0:
                print("saving model...")
                ckpt_path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pt")
                tmp_path = ckpt_path + ".tmp"
                try:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scheduler_state_dict': self.scheduler.state_dict(),
                        'loss': avg_train_loss,  # Using train loss since we might not run val every epoch
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
                    print(f"WARNING: Failed to save checkpoint at epoch {epoch+1}: {e}")
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)

                # Save torch.package for inference
                if (epoch + 1) == self.num_epochs:
                    pkg_path = os.path.join(self.checkpoint_dir, f"model_package_epoch_{epoch+1}.pt")
                    try:
                        import torch.package
                        print(f"Packaging model to {pkg_path}...")
                        with torch.package.PackageExporter(pkg_path) as exp:
                            exp.intern("model.**")
                            exp.intern("utils.**")
                            exp.intern("pos_encodings.**")
                            exp.extern("**")
                            exp.save_pickle("model", "model.pkl", self.model)
                        print(f"Saved packaged model to {pkg_path}")
                    except Exception as e:
                        print(f"WARNING: Failed to package model: {e}")

        self.writer.close()
        print("Training Complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='training/train_config.yaml', help='Path to config file')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    args = parser.parse_args()
    
    Trainer(args.config, args.resume).run()
