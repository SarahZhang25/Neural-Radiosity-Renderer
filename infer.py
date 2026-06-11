"""pretty much no use for this anymore, was just to figure out ssim/lpips metric..."""
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

import torch
import yaml
import os
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from training.dataset import SceneDataset
from model.global_illumination_model import GlobalIlluminationModel
from training.train import convert_hdr_for_visualization, tone_map_reinhard, linear_to_srgb

# Notebook Settings & Paths
# /home/sazhang/Neural-Radiosity-Renderer/training/logs/attempt6_table_chair_540/20260605-165222
# MODEL_RUN_ID = "attempt6_table_chair_540/20260605-113141" # 128x128
# MODEL_RUN_ID = "attempt6_table_chair_540/20260605-130741" # 256x256
MODEL_RUN_ID = "attempt6_table_chair_540/20260610-111417" # ROPE enabled
EPOCH = 5000
CONFIG_PATH = f'training/logs/{MODEL_RUN_ID}/config.yaml' 
CHECKPOINT_PATH = f'training/logs/{MODEL_RUN_ID}/checkpoints/checkpoint_epoch_{EPOCH}.pt' # Update path

# Helper Functions
def get_seeded_batch(dataset, batch_size, seed=123):
    """Identical seeded retriever from the training script to ensure we get the exact validation batch."""
    g = torch.Generator()
    g.manual_seed(seed)
    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=2,
        generator=g
    )
    batch = next(iter(loader))
    return batch

def tensor_to_image(tensor):
    """Converts a [C, H, W] PyTorch tensor to a [H, W, C] numpy array for Matplotlib."""
    img = tensor.detach().cpu().float().numpy()
    img = np.transpose(img, (1, 2, 0))
    # Ensure it is properly typed as uint8 so matplotlib knows it's a 0-255 image
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
        
    return img


def convert_hdr_for_metrics(x, method="reinhard", exposure=1.0):
    x = tone_map_reinhard(x, exposure=exposure)
    x = linear_to_srgb(x)
    return x




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
lpips_val_metric = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).to(device)

def infer_and_visualize(model, single_item_batch, plot=True, verbose=True):
    # Use CUDA events for accurate GPU timing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    with torch.no_grad():
        # Warmup pass (optional but recommended for accurate timing on GPUs)
        _ = model(
            single_item_batch['rays_o'], single_item_batch['rays_d'], 
            single_item_batch['obj_positions'], single_item_batch['obj_properties'], 
            obj_normals=single_item_batch['obj_normals'], w2c=single_item_batch['w2c']
        )
        torch.cuda.synchronize()

        # Actual timed inference pass using bfloat16 to match training
        start_event.record()
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            pred_radiance = model(
                single_item_batch['rays_o'],
                single_item_batch['rays_d'],
                single_item_batch['obj_positions'],
                single_item_batch['obj_properties'],
                obj_normals=single_item_batch['obj_normals'],
                w2c=single_item_batch['w2c']
            )
        end_event.record()
        torch.cuda.synchronize() # Wait for GPU to finish computing
        
    inference_time_ms = start_event.elapsed_time(end_event)
    if verbose:
        print(f"\n⏱️ Single Image Inference Time: {inference_time_ms:.2f} ms ({1000/inference_time_ms:.2f} FPS)")

    target = single_item_batch['target_image']
    if verbose:
        print(f"Prediction resolution: {pred_radiance.shape[2]}x{pred_radiance.shape[3]}") # [B, C, H, W]

    # Apply tone mapping 
    vis_target = convert_hdr_for_visualization(target)
    vis_pred = convert_hdr_for_visualization(torch.clamp(pred_radiance, min=0.0))
    
    # Calculate absolute difference map on the tone-mapped images
    diff_map = torch.abs(vis_target.float() - vis_pred.float())
    diff_map_gray = diff_map.mean(dim=1, keepdim=True)

    # Convert tensors to numpy arrays for plotting
    # Strip batch dimension using [0]
    img_pred = tensor_to_image(vis_pred[0])
    img_target = tensor_to_image(vis_target[0])
    img_diff = diff_map_gray[0, 0].detach().cpu().float().numpy() # [H, W] for heatmap

    # print(vis_target.shape, vis_pred.shape)
    vis_target_eval = convert_hdr_for_metrics(vis_target)
    vis_pred_eval = convert_hdr_for_metrics(vis_pred)
    # print(type(vis_target_eval), vis_target_eval.shape, vis_target_eval.dtype, vis_target_eval.min(), vis_target_eval.max())
    # print(type(vis_pred_eval), vis_pred_eval.shape, vis_pred_eval.dtype, vis_pred_eval.min(), vis_pred_eval.max())

    ssim = ssim_metric(vis_pred_eval, vis_target_eval)
    lpips = lpips_val_metric(vis_pred_eval, vis_target_eval)
    if verbose:
        print(f"SSIM: {ssim.item():.4f}")
        print(f"LPIPS: {lpips.item():.4f}")

    # if plot:
    #     plt.imsave(f'inference_results_{MODEL_RUN_ID}_epoch_{EPOCH}.png', np.hstack((img_pred, img_target, plt.cm.inferno(img_diff)[:, :, :3]))) 
    # else:
    #     return img_pred, img_target, img_diff
    

if __name__ == "__main__":
    print(f"CUDA Available: {torch.cuda.is_available()}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. Load Config
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
        
    # 2. Setup Data
    val_dataset = SceneDataset(
        data_dir=config['training']['data_dir'],
        image_res=config['training']['image_res'],
        split='val'
    )

    # Retrieve the exact fixed batch from training
    fixed_val_batch = get_seeded_batch(val_dataset, min(16, config['training']['batch_size']), seed=123)

    # Move batch to device
    batch = {k: v.to(device) for k, v in fixed_val_batch.items()}

    # Extract a SINGLE image from the batch for timing and visualization
    # We slice [0:1] to maintain the batch dimension [1, ...]
    single_item_batch = {k: v[0:1] for k, v in batch.items()}

    # 3. Setup Model & Load Weights
    model = GlobalIlluminationModel(config).to(device)

    if os.path.exists(CHECKPOINT_PATH):
        print(f"Loading checkpoint from: {CHECKPOINT_PATH}")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        raise FileNotFoundError(f"Could not find checkpoint at {CHECKPOINT_PATH}")
        
    model.eval()
    target_resolution = single_item_batch['target_image'].shape[2:] # [H, W]
    print(f"Target image resolution: {target_resolution[0]}x{target_resolution[1]}")

    infer_and_visualize(model, single_item_batch)
