import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torchvision

from model.global_illumination_model import GlobalIlluminationModel
from training.dataset import RadiosityDataset

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
    criterion = nn.MSELoss() # Pixel-wise L2
    
    # 5. Logging
    log_dir = config['training']['log_dir']
    checkpoint_dir = config['training']['checkpoint_dir']
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    # 6. Training Loop
    start_epoch = 0
    num_epochs = config['training']['num_epochs']
    
    # Setup mixed precision training
    scaler = torch.amp.GradScaler('cuda')
    
    # Save a fixed validation batch for consistent visual progress
    fixed_val_batch = next(iter(val_loader))
    fixed_val_batch = {k: v.to(device) for k, v in fixed_val_batch.items()}

    for epoch in range(start_epoch, num_epochs):
        model.train()
        train_loss = 0.0
        
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
            for batch in pbar:
                # Move to device
                rays_o = batch['rays_o'].to(device)
                rays_d = batch['rays_d'].to(device)
                positions = batch['obj_positions'].to(device)
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
                        ray_map=rays_d # Using rays_d as ray_map
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
                    properties = batch['obj_properties'].to(device)
                    class_ids = batch['obj_class_ids'].to(device)
                    target = batch['target_image'].to(device)

                    pred = model(rays_o, rays_d, positions, properties, class_ids, rays_d)
                    loss = criterion(pred, target)
                    val_loss += loss.item()
                
                avg_val_loss = val_loss / len(val_loader)
                writer.add_scalar('Loss/val', avg_val_loss, epoch)
                print(f"Epoch {epoch+1}: Val Loss {avg_val_loss:.6f}")

                # Save Checkpoint
                ckpt_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pt")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_val_loss,
                }, ckpt_path)

                # Visual Inspection (Fixed Batch)
                pred_fixed = model(
                    fixed_val_batch['rays_o'],
                    fixed_val_batch['rays_d'],
                    fixed_val_batch['obj_positions'],
                    fixed_val_batch['obj_properties'],
                    fixed_val_batch['obj_class_ids'],
                    fixed_val_batch['rays_d']
                )
                
                # Log images: Target vs Prediction
                # Stack them: [Target, Prediction]
                vis_img = torch.cat([fixed_val_batch['target_image'], pred_fixed], dim=3) # Concatenate width-wise
                grid = torchvision.utils.make_grid(vis_img, nrow=1, normalize=True, value_range=(0,1))
                writer.add_image('Visual/Val_Epoch_{}'.format(epoch+1), grid, epoch)

    writer.close()
    print("Training Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='training/train_config.yaml', help='Path to config file')
    args = parser.parse_args()
    
    train(args.config)
