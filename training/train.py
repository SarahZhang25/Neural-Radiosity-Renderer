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
from training.dataset import RadiosityDataset

def log_transform(x):
    """
    Applies a log transform to the input image tensor for better handling of HDR values in L1 loss.
    Add 1e-5 to prevent log(0) from producing -inf / NaN gradients.
    """
    return torch.log(x + 1e-5)

# RenderFormer LPIPS Tone-mapping: "clamp (log I / log 2, 0, 1)"
# Note: log(I) / log(2) is mathematically identical to log2(I).
def tone_map_lpips(x):
    """
    Tone mapping to apply before LPIPS
    """
    # We clamp the input to a minimum of 1.0 before log2 so that log2(1) = 0.
    # This maps the darkest areas to 0, and specular highlights (up to 2.0) to 1.
    # Then scale to [-1, 1] for LPIPS input.
    log2_x = torch.log2(torch.clamp(x, min=1.0))
    return torch.clamp(log2_x, min=0.0, max=1.0) * 2 - 1

def train(config_path):

    print(f"CUDA Available: {torch.cuda.is_available()}")

    
    # 1. Load Config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 2. Setup Data
    train_dataset = RadiosityDataset(
        data_dir=config['training']['data_dir'],
        image_res=config['training']['image_res'],
        split='train'
    )
    val_dataset = RadiosityDataset(
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
            loss_l1 = core_loss_fn(pred, target)
            # LPIPS loss on tone-mapped LDR predictions
            pred_mapped = tone_map_lpips(pred)
            target_mapped = tone_map_lpips(target)
            loss_lpips = lpips_loss_fn(pred_mapped, target_mapped).mean()
            # Final combined loss
            return loss_l1 + lpips_weight * loss_lpips
    else:
        def criterion(pred, target):
            return core_loss_fn(pred, target)
    
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
        
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
            for batch in pbar:
                # Move to device
                rays_o = batch['rays_o'].to(device)
                rays_d = batch['rays_d'].to(device)
                positions = batch['obj_positions'].to(device)
                normals = batch['obj_normals'].to(device)
                properties = batch['obj_properties'].to(device)
                class_ids = batch['obj_class_ids'].to(device)
                target = batch['target_image'].to(device)
                
                optimizer.zero_grad()
                
                # Forward pass with mixed precision
                with torch.amp.autocast('cuda'):
                    pred_radiance = model(
                        rays_o=rays_o,
                        rays_d=rays_d,
                        obj_positions=positions,
                        obj_properties=properties,
                        obj_class_ids=class_ids,
                        ray_map=rays_d, # Using rays_d as ray_map
                        obj_normals=normals
                    )
                    
                    loss = criterion(pred_radiance, target)
                
                # Backward pass with scaler
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                train_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})

        avg_train_loss = train_loss / len(train_loader)
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        
        # Validation & Logging
        if (epoch + 1) % config['training']['save_interval'] == 0:
            val_loss = 0.0
            model.eval()
            with torch.no_grad():
                for batch in val_loader:
                    rays_o = batch['rays_o'].to(device)
                    rays_d = batch['rays_d'].to(device)
                    positions = batch['obj_positions'].to(device)
                    normals = batch['obj_normals'].to(device)
                    properties = batch['obj_properties'].to(device)
                    class_ids = batch['obj_class_ids'].to(device)
                    target = batch['target_image'].to(device)

                    pred = model(rays_o, rays_d, positions, properties, class_ids, rays_d, obj_normals=normals)
                    loss = criterion(pred, target)
                    val_loss += loss.item()
                
                avg_val_loss = val_loss / len(val_loader)
                writer.add_scalar('Loss/val', avg_val_loss, epoch)
                print(f"Epoch {epoch+1}: Val Loss {avg_val_loss:.6f}")

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
                    fixed_val_batch['obj_class_ids'],
                    fixed_val_batch['rays_d'],
                    obj_normals=fixed_val_batch['obj_normals']
                )
                
                # Log images: Target vs Prediction
                # Stack them: [Target, Prediction]
                vis_img = torch.cat([fixed_val_batch['target_image'], pred_fixed], dim=3) # Concatenate width-wise
                grid = torchvision.utils.make_grid(vis_img, nrow=1, normalize=True, value_range=(0,1))
                writer.add_image('Visual/Validation', grid, epoch)

                # Visual Inspection (Fixed Training Batch)=
                pred_train_fixed = model(
                    fixed_train_batch['rays_o'],
                    fixed_train_batch['rays_d'],
                    fixed_train_batch['obj_positions'],
                    fixed_train_batch['obj_properties'],
                    fixed_train_batch['obj_class_ids'],
                    fixed_train_batch['rays_d'],
                    obj_normals=fixed_train_batch['obj_normals']
                )

                # Stack them: [Target, Prediction]
                # Note: We visualize the first 4 samples if batch is large to save space, or all if small
                vis_train_limit = 32 # fixed_train_batch['target_image'].shape[0] # 4 
                vis_train_img = torch.cat([
                    fixed_train_batch['target_image'][:vis_train_limit], 
                    pred_train_fixed[:vis_train_limit]
                ], dim=3)
                
                grid_train = torchvision.utils.make_grid(vis_train_img, nrow=1, normalize=True, value_range=(0,1))
                writer.add_image('Visual/Training', grid_train, epoch)

    writer.close()
    print("Training Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='training/train_config.yaml', help='Path to config file')
    args = parser.parse_args()
    
    train(args.config)
