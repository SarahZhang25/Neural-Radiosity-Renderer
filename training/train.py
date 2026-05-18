import os
import argparse
import shutil
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torchvision
from datetime import datetime
from lpips import LPIPS

from model.global_illumination_model import GlobalIlluminationModel
from training.dataset import SceneDataset 

torch.backends.cuda.matmul.allow_tf32 = True

def log_transform(x):
    """
    Applies a log transform to the input image tensor for better handling of HDR values in L1 loss.
    Clamp explicitly to prevent log(x < 0) from producing NaN gradients.
    Use log(1 + x) for smooth gradients and true zero-anchoring.
    """
    x = torch.clamp(x, min=0.0)
    return torch.log1p(x)

# def log_transform(x):
#     x = torch.clamp(x, min=1e-4)  # Larger eps
#     return torch.log(x)  # Natural log, not log1p

def linear_to_srgb(x: torch.Tensor) -> torch.Tensor:
    a = 0.055
    x = torch.clamp(x, min=0.0, max=1.0)
    return torch.where(x <= 0.0031308, 12.92 * x, (1 + a) * torch.pow(x, 1/2.4) - a)

def tone_map_reinhard(x, exposure=1.0):
    """
    Simple Reinhard tone mapping: x / (x + 1)
    We can also apply an exposure scaling before tone mapping.
    """
    x = x * exposure
    return x / (x + 1.0)

def to_uint8(x):
    """
    Convert float image [0, 1] to uint8 [0, 255].
    """
    x = torch.clamp(x, 0.0, 1.0)
    return (x * 255).byte()

def convert_hdr_for_visualization(x, method="reinhard", exposure=1.0):
    """
    Convert HDR image to LDR for visualization using specified tone mapping method.

    Args:
        x: Input HDR image tensor
        method: Current hard-coded to reinhard # "reinhard", "gamma", or "none". 
        exposure: Exposure value for tone mapping

    Returns:
        Tone-mapped image in [0, 1] range
    """
    x = tone_map_reinhard(x, exposure=exposure)
    x = linear_to_srgb(x)
    x = to_uint8(x)
    return x

# RenderFormer LPIPS Tone-mapping: "clamp (log I / log 2, 0, 1)"
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
    # Map [0, inf) to [0, 1) smoothly
    mapped = x / (x + 1.0) 

    # Gamma Correction (Approximate sRGB)
    # Clamp slightly above 0 to prevent NaN gradients in the power function
    # mapped = torch.clamp(mapped, min=1e-6)
    # gamma_mapped = torch.pow(mapped, 1.0 / 2.2)

    # Scale to [-1, 1] for LPIPS
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

def train(config_path):
    print(f"CUDA Available: {torch.cuda.is_available()}")

    # 1. Load Config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 2. Setup Data
    train_dataset = SceneDataset(
        data_dir=config['training']['data_dir'],
        image_res=config['training']['image_res'],
        split='train'
    )
    val_dataset = SceneDataset(
        data_dir=config['training']['data_dir'],
        image_res=config['training']['image_res'],
        split='val'
    )
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['training']['batch_size'], 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=min(4, config['training']['batch_size']), 
        shuffle=False, 
        num_workers=2
    )

    # 3. Setup Model
    model = GlobalIlluminationModel(config).to(device)

    # 4. Setup Optimizer & Loss
    optimizer = optim.AdamW(model.parameters(), lr=config['training']['learning_rate'])
    lpips_weight = config['training'].get('lpips_loss_weighting', 0)
    primary_loss = config['training'].get('primary_loss', 'mse')
    if primary_loss == 'mse':
        core_loss_fn = nn.MSELoss() # Pixel-wise L2
    elif primary_loss == 'mae':
        core_loss_fn = nn.L1Loss() # Pixel-wise L1
    else:
        raise ValueError(f"Unsupported primary loss type: {primary_loss}")
    
    if lpips_weight > 0:
        lpips_loss_fn = LPIPS(net=config['training']['lpips_backbone']).to(device)
        def criterion(pred, target):
            # Base core loss on log-transformed HDR predictions
            loss_l1 = core_loss_fn(log_transform(pred), log_transform(target))
            # LPIPS loss on tone-mapped LDR predictions
            pred_mapped = tone_map_lpips(pred)
            target_mapped = tone_map_lpips(target)
            loss_lpips = lpips_loss_fn(pred_mapped, target_mapped).mean()
            # Final combined loss
            return loss_l1 + lpips_weight * loss_lpips
    else:
        def criterion(pred, target):
            return core_loss_fn(log_transform(pred), log_transform(target))
    
    # 5. Logging
    log_dir_root = config['training']['log_dir']
    # Create a unique timestamped directory for this run
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(log_dir_root, run_id)
    print("Logging to:", log_dir)
    
    checkpoint_dir = os.path.join(log_dir, 'checkpoints')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    shutil.copy(config_path, os.path.join(log_dir, 'config.yaml'))

    writer = SummaryWriter(log_dir=log_dir)

    # 6. Training Loop
    start_epoch = 0
    num_epochs = config['training']['num_epochs']
    
    # Setup mixed precision training
    scaler = torch.amp.GradScaler('cuda')
    
    # Seeded batch retrieve for consistency
    def get_seeded_batch(dataset, batch_size, seed=42):
        g = torch.Generator()
        g.manual_seed(seed)
        loader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=True, # Shuffle to get variety
            num_workers=4,
            generator=g   # Use seeded generator for consistency
        )
        batch = next(iter(loader))
        return {k: v.to(device) for k, v in batch.items()}

    fixed_val_batch = get_seeded_batch(val_dataset, min(16, config['training']['batch_size']), seed=123)
    fixed_train_batch = get_seeded_batch(train_dataset, min(16, config['training']['batch_size']), seed=456)

    for epoch in range(start_epoch, num_epochs):
        model.train()
        train_loss = 0.0
        train_psnr = 0.0
        
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
            for batch in pbar:
                # Move to device
                rays_o = batch['rays_o'].to(device)
                rays_d = batch['rays_d'].to(device)
                positions = batch['obj_positions'].to(device)
                normals = batch['obj_normals'].to(device)
                properties = batch['obj_properties'].to(device)
                w2c = batch['w2c'].to(device)
                target = batch['target_image'].to(device)
                
                optimizer.zero_grad()
                
                # Forward pass with mixed precision
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    pred_radiance = model(
                        rays_o=rays_o,
                        rays_d=rays_d,
                        obj_positions=positions,
                        obj_properties=properties,
                        obj_normals=normals,
                        w2c=w2c
                    )
                
                # loss = criterion(pred_radiance, target)
                # Force float32 precision for mathematical stability?
                loss = criterion(pred_radiance.float(), target.float())
                
                # Backward pass with scaler, which is needed for fp16 but not bf16
                loss.backward()
                optimizer.step()
                # scaler.scale(loss).backward()
                # scaler.step(optimizer)
                # scaler.update()
                
                loss_item = loss.item()
                psnr_item = calculate_psnr(pred_radiance, target)
                
                train_loss += loss_item
                train_psnr += psnr_item
                pbar.set_postfix({'loss': loss_item, 'psnr': psnr_item})

        avg_train_loss = train_loss / len(train_loader)
        avg_train_psnr = train_psnr / len(train_loader)
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('PSNR/train', avg_train_psnr, epoch)
        
        # Validation & Logging
        if (epoch + 1) % config['training']['save_interval'] == 0:
            val_loss = 0.0
            val_psnr = 0.0
            model.eval()
            with torch.no_grad():
                for batch in val_loader:
                    rays_o = batch['rays_o'].to(device)
                    rays_d = batch['rays_d'].to(device)
                    positions = batch['obj_positions'].to(device)
                    normals = batch['obj_normals'].to(device)
                    properties = batch['obj_properties'].to(device)
                    # class_ids = batch['obj_class_ids'].to(device)
                    w2c = batch['w2c'].to(device)
                    target = batch['target_image'].to(device)

                    pred = model(rays_o, rays_d, positions, properties, obj_normals=normals, w2c=w2c)
                    loss = criterion(pred.float(), target.float())
                    val_loss += loss.item()
                    val_psnr += calculate_psnr(pred, target)
                
                avg_val_loss = val_loss / len(val_loader)
                avg_val_psnr = val_psnr / len(val_loader)
                writer.add_scalar('Loss/val', avg_val_loss, epoch)
                writer.add_scalar('PSNR/val', avg_val_psnr, epoch)
                print(f"Epoch {epoch+1}: Val Loss {avg_val_loss:.6f}, Val PSNR: {avg_val_psnr:.2f} dB")

                # Save model at end of training
                if (epoch + 1 ) == config['training']['num_epochs']:
                    print("saving model...")
                    ckpt_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pt")
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': avg_val_loss,
                    }, ckpt_path)

                # Visual Inspection (Fixed Val Batch)
                pred_fixed = model(
                    fixed_val_batch['rays_o'],
                    fixed_val_batch['rays_d'],
                    fixed_val_batch['obj_positions'],
                    fixed_val_batch['obj_properties'],
                    obj_normals=fixed_val_batch['obj_normals'],
                    w2c=fixed_val_batch['w2c']
                )
                
                # Log images: Target vs Prediction (tone-mapped for visualization)
                # Stack them: [Target, Prediction]
                # Using a simple Reinhard tone mapping followed by sRGB curve for visualization
                vis_target = convert_hdr_for_visualization(fixed_val_batch['target_image'])
                vis_pred = convert_hdr_for_visualization(torch.clamp(pred_fixed, min=0.0))
                vis_img = torch.cat([vis_target, vis_pred], dim=3) # Concatenate width-wise
                grid = torchvision.utils.make_grid(vis_img, nrow=1, normalize=False)
                writer.add_image('Visual/Validation', grid, epoch)

                # Visual Inspection (Fixed Training Batch)=
                pred_train_fixed = model(
                    fixed_train_batch['rays_o'],
                    fixed_train_batch['rays_d'],
                    fixed_train_batch['obj_positions'],
                    fixed_train_batch['obj_properties'],
                    obj_normals=fixed_train_batch['obj_normals'],
                    w2c=fixed_train_batch['w2c']
                )

                # Stack them: [Target, Prediction]
                # Note: We visualize the first 4 samples if batch is large to save space, or all if small
                vis_train_limit = 16 # fixed_train_batch['target_image'].shape[0] # 4 
                vis_train_target = convert_hdr_for_visualization(fixed_train_batch['target_image'][:vis_train_limit])
                vis_train_pred = convert_hdr_for_visualization(torch.clamp(pred_train_fixed[:vis_train_limit], min=0.0))
                vis_train_img = torch.cat([vis_train_target, vis_train_pred], dim=3)
                
                grid_train = torchvision.utils.make_grid(vis_train_img, nrow=1, normalize=False)
                writer.add_image('Visual/Training', grid_train, epoch)

                if (epoch + 1) == 500:
                    # pred = model(test_input)
                    target_hdr = fixed_val_batch['target_image']
                    pred_log = log_transform(pred_fixed + 1e-6)
                    target_log = log_transform(target_hdr + 1e-6)

                    print(f"Pred range: [{pred_fixed.min():.3f}, {pred_fixed.max():.3f}]")
                    print(f"Target range: [{target_hdr.min():.3f}, {target_hdr.max():.3f}]")
                    print(f"Pred log range: [{pred_log.min():.3f}, {pred_log.max():.3f}]")
                    print(f"Target log range: [{target_log.min():.3f}, {target_log.max():.3f}]")

                    # Also check for negative predictions (clamping issue)
                    num_negative = (pred_fixed < 0).sum().item()
                    print(f"Negative predictions: {num_negative} / {pred_fixed.numel()}")

    writer.close()
    print("Training Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='training/train_config.yaml', help='Path to config file')
    args = parser.parse_args()
    
    train(args.config)
