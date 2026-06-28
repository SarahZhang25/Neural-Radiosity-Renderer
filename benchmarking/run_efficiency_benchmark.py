"""
Run:
python benchmarking/run_efficiency_benchmark.py \
    --nmr_pkg_path training/logs/rf_ds2_chairs1-3_cbox0-3/20260627-195131_throwaway_run_for_package/checkpoints/model_package_epoch_5.pt \
    --renderformer_checkpoint_path /home/sazhang/Neural-Radiosity-Renderer/renderformer/training/logs/ds2_chairs1-3_cbox0-3/20260623-154203_full_ds_params46M_res128x128/checkpoints/model_epoch_20000.pt \
    --out_csv benchmarking/efficiency_benchmark_results.csv \
    --resolution 128 \
    --faces_per_obj 512

"""

import os
import torch
import argparse
import time
import csv
import sys
import pandas as pd
import matplotlib.pyplot as plt
# Add project root to path so we can import model code if needed
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "renderformer"))

from renderformer.pipelines.rendering_pipeline import RenderFormerRenderingPipeline


def load_pretrained_renderformer(model_id):
    print(f"\n--- Loading Pre-Trained RenderFormer ({model_id}) ---")
    pipeline = RenderFormerRenderingPipeline.from_pretrained(model_id)
    return pipeline


def load_checkpoint_renderformer(ckpt_path):
    import yaml
    from renderformer.models.config import RenderFormerConfig
    from renderformer.models.renderformer import RenderFormer
    
    print(f"\n--- Loading RenderFormer from Checkpoint ({ckpt_path}) ---")
    
    # Assume config.yaml is in the parent directory of the 'checkpoints' folder
    ckpt_dir = os.path.dirname(ckpt_path)
    run_dir = os.path.dirname(ckpt_dir)
    config_path = os.path.join(run_dir, "config.yaml")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Could not find config.yaml at {config_path}")
        
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
        
    model_config_dict = config_dict.get('model', {})
    model_config = RenderFormerConfig(**model_config_dict)
    model = RenderFormer(model_config)
    
    pipeline = RenderFormerRenderingPipeline(model)
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    pipeline.model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded custom RenderFormer weights from {ckpt_path}")
    return pipeline

 
def run_renderformer_benchmark(pipeline, obj_counts, faces_per_object, resolution, warmup=5, repeats=5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device == torch.device('cuda') and os.name == 'posix':
        try:
            from renderformer_liger_kernel import apply_kernels
            apply_kernels(pipeline.model)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except ImportError:
            pass

    pipeline.to(device)
    
    results = {}
    for n_obj in obj_counts:
        num_tris = n_obj * faces_per_object
        print(f"Benchmarking RenderFormer for {n_obj} objects ({num_tris} triangles)...")
        
        # Dummy inputs
        triangles = torch.randn(1, num_tris, 3, 3, device=device, dtype=torch.float32)
        
        # Texture depends on config
        tex_c = pipeline.config.texture_channels
        patch_size = pipeline.config.texture_encode_patch_size
        if patch_size == 1:
            texture = torch.randn(1, num_tris, tex_c, device=device, dtype=torch.float32)
        else:
            texture = torch.randn(1, num_tris, tex_c, patch_size, patch_size, device=device, dtype=torch.float32)
            
        mask = torch.ones(1, num_tris, device=device, dtype=torch.bool)
        vn = torch.randn(1, num_tris, 3, 3, device=device, dtype=torch.float32)
        
        nv = 1
        c2w = torch.eye(4).unsqueeze(0).unsqueeze(0).repeat(1, nv, 1, 1).to(device)
        fov = torch.ones(1, nv, 1, device=device, dtype=torch.float32) * 45.0
        
        try:
            # Warmup
            with torch.no_grad():
                for _ in range(warmup):
                    _ = pipeline(
                        triangles=triangles,
                        texture=texture,
                        mask=mask,
                        vn=vn,
                        c2w=c2w,
                        fov=fov,
                        resolution=resolution,
                        torch_dtype=torch.float16
                    )
            
            # Benchmark
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.empty_cache()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            start_event.record()
            with torch.no_grad():
                for _ in range(repeats):
                    _ = pipeline(
                        triangles=triangles,
                        texture=texture,
                        mask=mask,
                        vn=vn,
                        c2w=c2w,
                        fov=fov,
                        resolution=resolution,
                        torch_dtype=torch.float16
                    )
            end_event.record()
            torch.cuda.synchronize()
            
            time_ms = start_event.elapsed_time(end_event) / repeats
            max_mem_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
            
            results[n_obj] = {'time_ms': time_ms, 'mem_mb': max_mem_mb}
            print(f"  -> Time: {time_ms:.2f} ms | VRAM: {max_mem_mb:.2f} MB")
        except torch.cuda.OutOfMemoryError:
            print(f"  -> Out of Memory!")
            results[n_obj] = {'time_ms': 'OOM', 'mem_mb': 'OOM'}
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  -> Error: {e}")
            break
            
    return results


def run_mymodel_benchmark(pkg_path, obj_counts, faces_per_object, resolution, warmup=5, repeats=5):
    print(f"\n--- Loading My Model from {pkg_path} ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        imp = torch.package.PackageImporter(pkg_path)
        model = imp.load_pickle("model", "model.pkl")
        model.eval()
        model.to(device)
    except Exception as e:
        print(f"Failed to load my model: {e}")
        return {}
        
    results = {}
    n_vertices = faces_per_object * 3 # approximation (assuming mostly unshared vertices for simplicity)
    
    for n_obj in obj_counts:
        print(f"Benchmarking My Model for {n_obj} objects ({n_vertices} vertices/obj)...")
        
        B = 1
        H, W = resolution, resolution
        
        # Dummy inputs
        rays_o = torch.randn(B, 3, device=device)
        rays_d = torch.randn(B, H, W, 3, device=device) 
        obj_positions = torch.randn(B, n_obj, n_vertices, 3, device=device) 
        obj_normals = torch.randn(B, n_obj, n_vertices, 3, device=device) 
        obj_properties = torch.randn(B, n_obj, 10, device=device)
        w2c = torch.eye(4).unsqueeze(0).repeat(B, 1, 1).to(device)
        
        try:
            # Warmup
            with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
                for _ in range(warmup):
                    _ = model(
                        rays_o=rays_o,
                        rays_d=rays_d,
                        obj_positions=obj_positions,
                        obj_properties=obj_properties,
                        obj_normals=obj_normals,
                        w2c=w2c
                    )
                    
            # Benchmark
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.empty_cache()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            start_event.record()
            with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
                for _ in range(repeats):
                    _ = model(
                        rays_o=rays_o,
                        rays_d=rays_d,
                        obj_positions=obj_positions,
                        obj_properties=obj_properties,
                        obj_normals=obj_normals,
                        w2c=w2c
                    )
            end_event.record()
            torch.cuda.synchronize()
            
            time_ms = start_event.elapsed_time(end_event) / repeats
            max_mem_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
            
            results[n_obj] = {'time_ms': time_ms, 'mem_mb': max_mem_mb}
            print(f"  -> Time: {time_ms:.2f} ms | VRAM: {max_mem_mb:.2f} MB")
        except torch.cuda.OutOfMemoryError:
            print(f"  -> Out of Memory!")
            results[n_obj] = {'time_ms': 'OOM', 'mem_mb': 'OOM'}
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  -> Error: {e}")
            break
            
    return results


def plot_results(csv_file, out_dir):
    if not os.path.exists(csv_file):
        print(f"Error: Could not find CSV file at {csv_file}")
        return
        
    df = pd.read_csv(csv_file)
    df.replace('OOM', pd.NA, inplace=True)
    
    cols_to_convert = ['my_model_time_ms', 'my_model_mem_mb', 'renderformer_large_time_ms', 'renderformer_large_mem_mb', 'renderformer_base_time_ms', 'renderformer_base_mem_mb', 'renderformer_custom_time_ms', 'renderformer_custom_mem_mb']
    for col in cols_to_convert:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
    os.makedirs(out_dir, exist_ok=True)
    
    # Plot 1: Inference Time
    plt.figure(figsize=(10, 6))
    if 'my_model_time_ms' in df.columns and not df['my_model_time_ms'].isna().all():
        plt.plot(df['num_objects'], df['my_model_time_ms'], marker='o', label='My Model (46M params)', color='blue')
    if 'renderformer_large_time_ms' in df.columns and not df['renderformer_large_time_ms'].isna().all():
        plt.plot(df['num_objects'], df['renderformer_large_time_ms'], marker='s', label='RenderFormer (Swin Large, 483M params)', color='red')
    if 'renderformer_base_time_ms' in df.columns and not df['renderformer_base_time_ms'].isna().all():
        plt.plot(df['num_objects'], df['renderformer_base_time_ms'], marker='v', label='RenderFormer (Base, 205M params)', color='orange')
    if 'renderformer_custom_time_ms' in df.columns and not df['renderformer_custom_time_ms'].isna().all():
        plt.plot(df['num_objects'], df['renderformer_custom_time_ms'], marker='^', label='RenderFormer (Custom, 46M params)', color='green')
        
    plt.title('Inference Time vs Scene Complexity')
    plt.xlabel('Number of Objects')
    plt.ylabel('Time (ms)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    time_plot_path = os.path.join(out_dir, "inference_time_plot.png")
    plt.savefig(time_plot_path)
    print(f"Saved {time_plot_path}")
    
    # Plot 2: Memory Cost
    plt.figure(figsize=(10, 6))
    if 'my_model_mem_mb' in df.columns and not df['my_model_mem_mb'].isna().all():
        plt.plot(df['num_objects'], df['my_model_mem_mb'], marker='o', label='My Model (46M params)', color='blue')
    if 'renderformer_large_mem_mb' in df.columns and not df['renderformer_large_mem_mb'].isna().all():
        plt.plot(df['num_objects'], df['renderformer_large_mem_mb'], marker='s', label='RenderFormer (Swin Large, 483M params)', color='red')
    if 'renderformer_base_mem_mb' in df.columns and not df['renderformer_base_mem_mb'].isna().all():
        plt.plot(df['num_objects'], df['renderformer_base_mem_mb'], marker='v', label='RenderFormer (Base, 205M params)', color='orange')
    if 'renderformer_custom_mem_mb' in df.columns and not df['renderformer_custom_mem_mb'].isna().all():
        plt.plot(df['num_objects'], df['renderformer_custom_mem_mb'], marker='^', label='RenderFormer (Custom, 46M params)', color='green')
        
    plt.title('Peak VRAM vs Scene Complexity')
    plt.xlabel('Number of Objects')
    plt.ylabel('Peak VRAM (MB)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    mem_plot_path = os.path.join(out_dir, "memory_cost_plot.png")
    plt.savefig(mem_plot_path)
    print(f"Saved {mem_plot_path}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark inference efficiency vs scene complexity")
    parser.add_argument("--nmr_pkg_path", type=str, default="training/logs/YOUR_RUN_ID/checkpoints/model_package_epoch_500.pt", help="Path to your packaged model .pt")
    parser.add_argument("--renderformer_checkpoint_path", type=str, default=None, help="Path to your trained RenderFormer .pt checkpoint (optional)")
    parser.add_argument("--out_csv", type=str, default="benchmarking/efficiency_benchmark_results.csv", help="Output CSV path")
    parser.add_argument("--resolution", type=int, default=128, help="Rendering resolution for benchmarks")
    parser.add_argument("--faces_per_obj", type=int, default=512, help="Number of faces per object to simulate")
    
    args = parser.parse_args()
    
    obj_counts = [1, 5, 10, 25, 50]
    # obj_counts = [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500]
    
    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    
    # Run benchmarks
    my_model =  run_mymodel_benchmark(args.nmr_pkg_path, obj_counts, args.faces_per_obj, args.resolution)
    
    renderformer_custom = {}
    if args.renderformer_checkpoint_path:
        renderformer_custom_pipeline = load_checkpoint_renderformer(args.renderformer_checkpoint_path)
        renderformer_custom = run_renderformer_benchmark(renderformer_custom_pipeline, obj_counts, args.faces_per_obj, args.resolution)
    
    renderformer_large_pipeline = load_pretrained_renderformer("microsoft/renderformer-v1.1-swin-large")
    renderformer_large = run_renderformer_benchmark(renderformer_large_pipeline, obj_counts, args.faces_per_obj, args.resolution)
    
    renderformer_base_pipeline = load_pretrained_renderformer("microsoft/renderformer-v1-base")
    renderformer_base = run_renderformer_benchmark(renderformer_base_pipeline, obj_counts, args.faces_per_obj, args.resolution)
    
    # Save to CSV
    with open(args.out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['num_objects', 'my_model_time_ms', 'my_model_mem_mb', 'renderformer_large_time_ms', 'renderformer_large_mem_mb', 'renderformer_base_time_ms', 'renderformer_base_mem_mb', 'renderformer_custom_time_ms', 'renderformer_custom_mem_mb'])
        
        for n in obj_counts:
            my_time = my_model.get(n, {}).get('time_ms', '')
            my_mem = my_model.get(n, {}).get('mem_mb', '')
            rf_large_time = renderformer_large.get(n, {}).get('time_ms', '')
            rf_large_mem = renderformer_large.get(n, {}).get('mem_mb', '')
            rf_base_time = renderformer_base.get(n, {}).get('time_ms', '')
            rf_base_mem = renderformer_base.get(n, {}).get('mem_mb', '')
            rf_custom_time = renderformer_custom.get(n, {}).get('time_ms', '')
            rf_custom_mem = renderformer_custom.get(n, {}).get('mem_mb', '')
            
            writer.writerow([n, my_time, my_mem, rf_large_time, rf_large_mem, rf_base_time, rf_base_mem, rf_custom_time, rf_custom_mem])
            
    print(f"\nBenchmarking complete. Results saved to {args.out_csv}")
    
    print("\nGenerating plots...")
    plot_results(args.out_csv, os.path.dirname(os.path.abspath(args.out_csv)))

if __name__ == "__main__":
    main()
