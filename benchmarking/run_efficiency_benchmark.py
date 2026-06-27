import os
import torch
import argparse
import time
import csv
import sys

# Add project root to path so we can import model code if needed
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "renderformer"))

def run_renderformer_benchmark(model_id, obj_counts, faces_per_object, resolution, warmup=5, repeats=5):
    try:
        from renderformer.pipelines.rendering_pipeline import RenderFormerRenderingPipeline
    except ImportError as e:
        print(f"Error: {e}")
        print("Warning: RenderFormer not found. Skipping RenderFormer benchmarking.")
        return {}

    print(f"\n--- Loading RenderFormer ({model_id}) ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    pipeline = RenderFormerRenderingPipeline.from_pretrained(model_id)
    
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
        texture = torch.randn(1, num_tris, 13, device=device, dtype=torch.float32)
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
        import torch.package
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
            with torch.no_grad():
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
            with torch.no_grad():
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

def main():
    parser = argparse.ArgumentParser(description="Benchmark inference efficiency vs scene complexity")
    parser.add_argument("--nmr_pkg_path", type=str, default="training/logs/YOUR_RUN_ID/checkpoints/model_package_epoch_500.pt", help="Path to your packaged model .pt")
    parser.add_argument("--renderformer_id", type=str, default="microsoft/renderformer-v1.1-swin-large", help="RenderFormer model ID/path")
    parser.add_argument("--out_csv", type=str, default="benchmarking/efficiency_benchmark_results.csv", help="Output CSV path")
    parser.add_argument("--resolution", type=int, default=128, help="Rendering resolution for benchmarks")
    parser.add_argument("--faces_per_obj", type=int, default=512, help="Number of faces per object to simulate")
    
    args = parser.parse_args()
    
    obj_counts = [1, 5]#, 10, 25, 50]
    # obj_counts = [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500]
    
    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    
    # Run benchmarks
    res_my_model =  {} #run_mymodel_benchmark(args.nmr_pkg_path, obj_counts, args.faces_per_obj, args.resolution)
    res_renderformer = run_renderformer_benchmark(args.renderformer_id, obj_counts, args.faces_per_obj, args.resolution)
    
    # Save to CSV
    with open(args.out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['num_objects', 'my_model_time_ms', 'my_model_mem_mb', 'renderformer_time_ms', 'renderformer_mem_mb'])
        
        for n in obj_counts:
            my_time = res_my_model.get(n, {}).get('time_ms', '')
            my_mem = res_my_model.get(n, {}).get('mem_mb', '')
            rf_time = res_renderformer.get(n, {}).get('time_ms', '')
            rf_mem = res_renderformer.get(n, {}).get('mem_mb', '')
            
            writer.writerow([n, my_time, my_mem, rf_time, rf_mem])
            
    print(f"\nBenchmarking complete. Results saved to {args.out_csv}")

if __name__ == "__main__":
    main()
