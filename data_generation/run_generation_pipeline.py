"""
Streaming Dataset Generation Pipeline Orchestrator

This script overlaps JSON generation, GPU-bound Blender rendering, and CPU-bound 
geometry preprocessing to ensure maximum hardware saturation. Instead of waiting for
thousands of scenes to render before starting H5 processing, it streams completed 
renders directly into the preprocessing pool, and then packs them into chunked H5 
files on the fly, immediately deleting the heavy raw EXR and PNG files to save space.

To eliminate SSD wear-and-tear and bypass disk I/O bottlenecks for the 
intermediate raw renders, use the system's built-in RAM disk:
Simply set your output directory to /dev/shm/streaming_dataset instead.
/dev/shm is already mounted as a tmpfs on Linux and is writable by all users

`--start_idx` argument example usage:
1. Run your initial 5k test run:
   ```bash
   python data_generation/run_generation_pipeline.py --num_scenes 5000 --num_views 4 [other args...]
   ```
2. When you're ready to do the remaining 20k scenes, point the script to the exact same `--output_dir` and simply tell it to resume at index 5000:
   ```bash
   python data_generation/run_generation_pipeline.py --num_scenes 20000 --start_idx 5000 --num_views 4 [other args...]
   ```

**What this does under the hood:**
- The new batch will name its scenes starting at `scene_005000` all the way up to `scene_024999`, so no previous JSONs or H5 groups are overwritten.
- The pipeline's built-in H5 chunker will automatically scan the output directory, detect `dataset_chunk_0004.h5` (or whatever the last one was), and correctly resume chunk numbering from `dataset_chunk_0005.h5`.


Example Usage:
    python data_generation/run_generation_pipeline.py \
        --num_scenes 8 --num_views  2 --transform_scenes \
        --output_dir ./tmp/streaming_dataset \
        --tmp_dir /dev/shm/streaming_dataset \
        --formats nmr --gpus 0,1 \
        --chunk_size 20  \
        --spp 128 \
        --workers_per_gpu 5


    python data_generation/run_generation_pipeline.py \
        --num_scenes 1 --num_views 4 --transform_scenes  --texture_mode per-shading-group \
        --output_dir ./tmp/dataset_test \
        --tmp_dir /dev/shm/dataset_test \
        --formats nmr --gpus 5 \
        --chunk_size 1000  \
        --spp 512 \
        --workers_per_gpu 10 \
        --monitor_gpu --json_only

    
    python data_generation/run_generation_pipeline.py \
        --num_scenes 5000 --num_views 4 --transform_scenes  --texture_mode per-triangle \
        --output_dir ./datasets/dataset_pertriangle \
        --tmp_dir /dev/shm/dataset_pertriangle \
        --formats nmr rf --gpus 2,3 \
        --chunk_size 1000  \
        --spp 512 \
        --workers_per_gpu 5 \

    nohup python data_generation/run_generation_pipeline.py \
        --num_scenes 5000 --num_views 4 --transform_scenes  --texture_mode procedural \
        --output_dir ./datasets/dataset_procedural \
        --tmp_dir /dev/shm/dataset_procedural \
        --formats nmr rf --gpus 2,3 \
        --chunk_size 1000  \
        --spp 512 \
        --workers_per_gpu 5 --json_only > generate_dataset_procedural0-5k_max12_obj.log  2>&1 &


    nohup python data_generation/run_generation_pipeline.py \
        --num_scenes 5000 --num_views 4 --transform_scenes  --texture_mode per-shading-group \
        --output_dir ./datasets/dataset_uniform \
        --tmp_dir /dev/shm/dataset_uniform \
        --formats nmr rf --gpus 2,3 \
        --chunk_size 1000  \
        --spp 512 \
        --workers_per_gpu 50 > generate_dataset_uniform0-5k_max6_obj.log  2>&1 &

    python data_generation/run_generation_pipeline.py \
        --num_scenes 10000 --num_views 4 --start_idx 5000 --transform_scenes  --texture_mode per-shading-group \
        --output_dir ./datasets/dataset_uniform \
        --tmp_dir /dev/shm/dataset_uniform \
        --formats nmr rf --gpus 2,3 \
        --chunk_size 1000  \
        --spp 512 \
        --workers_per_gpu 50 \

    nohup COMMAND > generate_dataset_uniform0-5k.log  2>&1 &


"""

import os
import sys
import glob
import argparse
import subprocess
import threading
import queue
import concurrent.futures
import h5py
import time
import shutil
import tqdm
import random

# Ensure we can import from data_generation/ and renderformer/data_generation/
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

RENDERFORMER_GEN_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'renderformer', 'data_generation')
if RENDERFORMER_GEN_DIR not in sys.path:
    sys.path.append(RENDERFORMER_GEN_DIR)

# Import existing functions to stay DRY
from generate_scenes import generate_scene, find_objaverse_objects
from write_training_datasets_from_jsons import process_json_scene

def get_available_gpus(gpus_arg):
    gpu_list = []
    if gpus_arg:
        try:
            gpu_list = [int(x.strip()) for x in gpus_arg.split(",") if x.strip()]
        except ValueError:
            print("Invalid format for --gpus. Using auto-detect.")
    elif "CUDA_VISIBLE_DEVICES" in os.environ:
        env_gpus = os.environ["CUDA_VISIBLE_DEVICES"]
        if env_gpus:
            try:
                gpu_list = [int(x.strip()) for x in env_gpus.split(",") if x.strip()]
            except ValueError:
                pass
                
    if not gpu_list:
        try:
            import torch
            gpu_list = list(range(torch.cuda.device_count()))
        except ImportError:
            try:
                res = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=True)
                lines = [line for line in res.stdout.split("\n") if line.strip()]
                gpu_list = list(range(len(lines)))
            except Exception:
                gpu_list = []
    return gpu_list

def cleanup_files(scene_file, exr_dir, num_views, remove_json=False, remove_renders=True):
    """Delete intermediate json, exr, and png files after they are safely packed in H5."""
    scene_name = os.path.splitext(os.path.basename(scene_file))[0]
    
    # 1. Delete JSON
    if remove_json:
        try:
            if os.path.exists(scene_file):
                os.remove(scene_file)
        except Exception:
            pass
        
    # 2. Delete EXRs and PNGs
    if remove_renders:
        for i in range(num_views):
            exr_path = os.path.join(exr_dir, f"{scene_name}_{i}.exr")
            png_path = os.path.join(exr_dir, f"{scene_name}_{i}.png")
            try:
                if os.path.exists(exr_path):
                    os.remove(exr_path)
                if os.path.exists(png_path):
                    os.remove(png_path)
            except Exception:
                pass

def render_worker_task(scene_file, output_dir, gpu_id, spp, timeout_seconds, timeout_retries):
    """Submits a rendering task to blender via subprocess on a specific GPU."""
    cmd = [
        "python3", "renderformer/scene_processor/to_blend.py",
        scene_file,
        "--output_dir", output_dir,
        "--spp", str(spp),
        "--no_dump_blend"
    ]
    
    env = os.environ.copy()
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        
    def set_pdeathsig():
        import ctypes
        import signal
        try:
            libc = ctypes.CDLL("libc.so.6")
            libc.prctl(1, signal.SIGTERM)
        except Exception:
            pass

    scene_stem = os.path.splitext(os.path.basename(scene_file))[0]

    def remove_partial_outputs():
        for path in glob.glob(os.path.join(output_dir, f"{scene_stem}_*.exr")) + glob.glob(os.path.join(output_dir, f"{scene_stem}_*.png")) + glob.glob(os.path.join(output_dir, f"{scene_stem}.blend")):
            try:
                if os.path.exists(path):
                    os.remove(path)
            except Exception:
                pass

    for attempt in range(timeout_retries + 1):
        try:
            subprocess.run(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
                preexec_fn=set_pdeathsig,
                timeout=timeout_seconds,
            )
            return scene_file, True
        except subprocess.TimeoutExpired as e:
            print(f"\n[GPU Worker] Render timed out for {scene_file} after {timeout_seconds}s (attempt {attempt + 1}/{timeout_retries + 1}).", flush=True)
            if e.stdout:
                print(f"STDOUT:\n{e.stdout}", flush=True)
            if e.stderr:
                print(f"STDERR:\n{e.stderr}", flush=True)
            remove_partial_outputs()
            if attempt >= timeout_retries:
                return scene_file, False
        except subprocess.CalledProcessError as e:
            print(f"\n[GPU Worker] Render failed for {scene_file}:\nSTDOUT:\n{e.stdout}\nSTDERR:\n{e.stderr}")
            return scene_file, False
        except Exception as e:
            print(f"\n[GPU Worker] Unexpected render error for {scene_file}: {e}")
            return scene_file, False

def process_worker_task(scene_file, exr_dir, points, formats):
    """Runs CPU heavy geometry extraction using the existing unified pipeline function."""
    try:
        result = process_json_scene(scene_file, exr_dir, points, formats)
        return scene_file, result
    except Exception as e:
        import traceback
        print(f"\n[CPU Worker] Preprocessing failed for {scene_file}: {e}")
        traceback.print_exc()
        return scene_file, None

def generate_json_worker_task(loop_idx, start_idx, template_json, objaverse_objects, tmp_dir, num_views, transform_scenes, texture_mode, color_lights):
    """Generates one scene JSON and returns its index for stable output ordering."""
    scene_idx = start_idx + loop_idx
    json_file = os.path.join(tmp_dir, f"scene_{scene_idx:06d}.json")
    if not os.path.exists(json_file):
        generate_scene(
            template_json,
            objaverse_objects,
            scene_idx,
            tmp_dir,
            num_views=num_views,
            transform_scene=transform_scenes,
            random_diffuse_type=texture_mode,
            color_lights=color_lights,
        )
    return loop_idx, json_file

def h5_writer_thread(write_queue, output_dir, tmp_dir, formats, chunk_size, total_scenes, num_views):
    """Dedicated thread to receive completed dictionaries, stream directly to H5, and cleanup."""
    import tqdm
    processed_count = 0
    next_chunk_id = 0
    
    existing_chunks = glob.glob(os.path.join(output_dir, "*_dataset_chunk_*.h5")) + glob.glob(os.path.join(output_dir, "*_dataset_chunk_*.h5.INCOMPLETE"))
    
    scenes_in_current_chunk = 0
    resume_incomplete = False
    
    if existing_chunks:
        chunk_ids = [int(os.path.basename(c).split('_')[3].split('.')[0]) for c in existing_chunks if len(os.path.basename(c).split('_')) >= 4]
        if chunk_ids:
            max_id = max(chunk_ids)
            is_corrupted = False
            is_incomplete = False
            existing_keys_count = 0
            
            for fmt in formats:
                h5_path = os.path.join(output_dir, f"{fmt}_dataset_chunk_{max_id:04d}.h5")
                incomplete_path = h5_path + ".INCOMPLETE"
                
                if os.path.exists(incomplete_path) and not os.path.exists(h5_path):
                    is_incomplete = True
                    path_to_check = incomplete_path
                else:
                    path_to_check = h5_path if os.path.exists(h5_path) else incomplete_path
                
                if os.path.exists(path_to_check):
                    try:
                        with h5py.File(path_to_check, 'r') as f:
                            valid_keys = []
                            for k in f.keys():
                                if 'hdr_target_image' in f[k] and f[k]['hdr_target_image'].shape[0] == num_views:
                                    valid_keys.append(k)
                            if len(valid_keys) > existing_keys_count:
                                existing_keys_count = len(valid_keys)
                    except Exception:
                        is_corrupted = True
                        break
            
            if is_corrupted:
                print(f"[*] Warning: Chunk {max_id} appears corrupted from a previous crash. Overwriting it.", flush=True)
                next_chunk_id = max_id
            elif is_incomplete:
                print(f"[*] Resuming incomplete chunk {max_id} with {existing_keys_count} existing scenes.", flush=True)
                next_chunk_id = max_id
                resume_incomplete = True
                scenes_in_current_chunk = existing_keys_count
            else:
                next_chunk_id = max_id + 1

    current_h5_files = {}

    def open_chunk(chunk_id, append=False):
        base_chunk_name = f"dataset_chunk_{chunk_id:04d}"
        for fmt in formats:
            h5_path = os.path.join(output_dir, f"{fmt}_{base_chunk_name}.h5.INCOMPLETE")
            mode = 'a' if append else 'w'
            current_h5_files[fmt] = h5py.File(h5_path, mode)
            
    def close_chunk(abort=False):
        for fmt, h5f in current_h5_files.items():
            incomplete_path = h5f.filename
            h5f.close()
            if not abort and os.path.exists(incomplete_path):
                final_path = incomplete_path.replace('.h5.INCOMPLETE', '.h5')
                os.rename(incomplete_path, final_path)
        current_h5_files.clear()

    # Open first chunk files
    open_chunk(next_chunk_id, append=resume_incomplete)

    with tqdm.tqdm(total=total_scenes, desc="Pipeline Progress", unit="scene", mininterval=0.5, maxinterval=45.0) as pbar:
        while processed_count < total_scenes:
            try:
                # Block until a processed scene arrives
                scene_file, result = write_queue.get(timeout=60.0) 
                if scene_file == "STOP":
                    pbar.write("[Writer] Received STOP signal. Leaving chunk as INCOMPLETE.")
                    close_chunk(abort=True)
                    write_queue.task_done()
                    break
                processed_count += 1
                pbar.update(1)
                
                if result is not None:
                    # Write immediately to open H5 files (Zero RAM buffering!)
                    for fmt in formats:
                        if fmt in result:
                            h5f = current_h5_files[fmt]
                            scene_name = result['scene_name']
                            if scene_name in h5f:
                                del h5f[scene_name]
                            grp = h5f.create_group(scene_name)
                            grp.create_dataset("hdr_target_image", data=result[fmt]['hdr_target_image'], compression="lzf")
                            grp.create_dataset("c2w", data=result[fmt]['c2w'], compression="lzf")
                            
                            if fmt == 'nmr':
                                grp.create_dataset("entity_vertices", data=result['nmr']['entity_vertices'], compression="lzf")
                                grp.create_dataset("entity_normals", data=result['nmr']['entity_normals'], compression="lzf")
                                grp.create_dataset("entity_materials", data=result['nmr']['entity_materials'], compression="lzf")
                                grp.create_dataset("camera_fov", data=result['nmr']['camera_fov'])
                            elif fmt == 'rf':
                                grp.create_dataset("triangles", data=result['rf']['triangles'], compression="lzf")
                                grp.create_dataset("vn", data=result['rf']['vn'], compression="lzf")
                                grp.create_dataset("texture", data=result['rf']['texture'], compression="lzf")
                                grp.create_dataset("fov", data=result['rf']['fov'])

                    scenes_in_current_chunk += 1
                    
                    # Safe to cleanup intermediate files immediately now that it's in the H5
                    # don't clean up for now so that I can come back and rerun for rf format...
                    cleanup_files(scene_file, tmp_dir, num_views, remove_json=False, remove_renders=False)

                    # If chunk is full, rotate to next
                    if scenes_in_current_chunk >= chunk_size:
                        pbar.write(f"[Writer] Finished writing chunk {next_chunk_id}.")
                        close_chunk()
                        next_chunk_id += 1
                        scenes_in_current_chunk = 0
                        # Open next chunk if we aren't done yet
                        if processed_count < total_scenes:
                            open_chunk(next_chunk_id)
                else:
                    pbar.write(f"[Writer] Skipped {os.path.basename(scene_file)} due to previous errors. Total: {processed_count}/{total_scenes}")
                    
                write_queue.task_done()
            except queue.Empty:
                pbar.write("[Writer] Waiting for scenes...")
                continue
            except Exception as e:
                pbar.write(f"[Writer] Fatal Error: {e}")
                break
                
    # Close any remaining open files at the end
    if current_h5_files:
        close_chunk()

def gpu_monitor_thread(gpu_list, stop_event, peak_mem_dict):
    while not stop_event.is_set():
        try:
            output = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index,memory.used,memory.total", "--format=csv,noheader,nounits"],
                text=True
            )
            for line in output.strip().split('\n'):
                parts = line.split(', ')
                if len(parts) == 3:
                    idx = int(parts[0])
                    used = int(parts[1])
                    total = int(parts[2])
                    if idx in gpu_list:
                        if idx not in peak_mem_dict:
                            peak_mem_dict[idx] = {'peak': used, 'total': total, 'baseline': used}
                        peak_mem_dict[idx]['peak'] = max(peak_mem_dict[idx]['peak'], used)
                        peak_mem_dict[idx]['total'] = total
        except Exception:
            pass
        time.sleep(1)

def main():
    parser = argparse.ArgumentParser(description="Streaming Dataset Pipeline: JSON -> GPU Render -> CPU Preprocess -> H5 Chunk -> Cleanup")
    parser.add_argument("--num_scenes", type=int, default=10, help="Number of scenes to generate")
    parser.add_argument("--num_views", type=int, default=4, help="Number of camera views per scene")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for the final H5 chunk files")
    parser.add_argument("--tmp_dir", type=str, default=None, help="Optional directory for intermediate JSONs, EXRs, and PNGs (e.g., /dev/shm). Defaults to output_dir.")
    parser.add_argument("--chunk_size", type=int, default=1000, help="Scenes per H5 chunk")
    parser.add_argument("--formats", nargs='+', choices=['nmr', 'rf'], default=['nmr', 'rf'], help="Formats to generate")
    parser.add_argument("--points", type=int, default=2048, help="Points to sample per object (nmr)")
    parser.add_argument("--spp", type=int, default=512, help="Blender samples per pixel")
    parser.add_argument("--gpus", type=str, default=None, help="Comma separated list of GPUs to use (e.g. 0,1)")
    parser.add_argument("--workers_per_gpu", type=int, default=1, help="Concurrent Blender processes per GPU")
    parser.add_argument("--render_timeout_seconds", type=int, default=1200, help="Per-scene render timeout in seconds before retrying. Default is 20 minutes. (Note: should increase this for more views per scene.)")
    parser.add_argument("--render_timeout_retries", type=int, default=3, help="Number of retries after a render timeout")
    parser.add_argument("--transform_scenes", action="store_true", help="Apply random global transformation")
    parser.add_argument("--texture_mode", type=str, default=None, help="Texture mode to use (per-shading-group, procedural, or per-triangle)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--monitor_gpu", action="store_true", help="Monitor GPU memory usage")
    parser.add_argument("--start_idx", type=int, default=0, help="Starting index for scene naming (useful for resuming generation)")
    parser.add_argument("--json_only", action="store_true", help="Only generate JSONs, skip rendering and H5 processing")
    parser.add_argument("--resume", action="store_true", help="Resume from incomplete chunks and skip existing scenes")
    parser.add_argument("--color_lights", action="store_true", help="Randomly tint lights to non-white colors to train color constancy")
    args = parser.parse_args()

    # Offset the seed by start_idx so resumed generation runs produce new, unique scenes
    random.seed(args.seed + args.start_idx)    
    os.makedirs(args.output_dir, exist_ok=True)
    
    tmp_dir = args.tmp_dir if args.tmp_dir else args.output_dir
    os.makedirs(tmp_dir, exist_ok=True)

    completed_scenes = set()
    if args.resume:
        import re
        from pathlib import Path
        print(f"[*] Resume flag set. Scanning output_dir for existing scenes for formats {args.formats}...")
        
        scene_formats_count = {}
        for fmt in args.formats:
            chunk_glob = f"{fmt}_dataset_chunk_*.h5"
            parent_dir = Path(args.output_dir)
            chunk_paths = sorted(parent_dir.glob(chunk_glob)) + sorted(parent_dir.glob(chunk_glob + ".INCOMPLETE"))
            
            for chunk_path in chunk_paths:
                try:
                    with h5py.File(chunk_path, "r") as handle:
                        for scene_name in handle.keys():
                            if 'hdr_target_image' in handle[scene_name] and handle[scene_name]['hdr_target_image'].shape[0] == args.num_views:
                                match = re.search(r"scene_(\d+)$", scene_name)
                                if match:
                                    scene_idx = int(match.group(1))
                                    if scene_idx not in scene_formats_count:
                                        scene_formats_count[scene_idx] = set()
                                    scene_formats_count[scene_idx].add(fmt)
                except Exception as e:
                    print(f"[*] Warning: Could not read {chunk_path} during resume scan: {e}")
                    
        scene_indices = [idx for idx, fmts in scene_formats_count.items() if len(fmts) == len(set(args.formats))]
        
        if scene_indices:
            scene_indices = sorted(set(scene_indices))
            min_scene = scene_indices[0]
            max_scene = scene_indices[-1]
            missing = [scene_idx for scene_idx in range(min_scene, max_scene + 1) if scene_idx not in scene_indices]
            
            print(f"[*] Scene index range: {min_scene} to {max_scene}")
            if missing:
                print(f"[*] Missing scenes ({len(missing)}): {missing}")
            else:
                print("[*] Missing scenes: none")
                
            for idx in scene_indices:
                completed_scenes.add(f"scene_{idx:06d}")
        else:
            print("[*] No existing scenes found.")

    json_files = []
    json_loop_indices_to_generate = []
    skipped_count = 0
    for loop_idx in range(args.num_scenes):
        scene_idx = args.start_idx + loop_idx
        scene_name = f"scene_{scene_idx:06d}"
        if args.resume and scene_name in completed_scenes:
            skipped_count += 1
            continue
            
        json_file = os.path.join(tmp_dir, f"{scene_name}.json")
        json_files.append(json_file)
        if not os.path.exists(json_file):
            json_loop_indices_to_generate.append(loop_idx)

    # Start timer
    pipeline_start_time = time.time()

    # 1. Setup GPU list
    gpu_list = get_available_gpus(args.gpus)
    num_gpus = len(gpu_list) if gpu_list else 1
    max_render_workers = num_gpus * args.workers_per_gpu
    print(f"[*] Detected {len(gpu_list)} GPUs: {gpu_list}. Using {max_render_workers} concurrent Blender render workers.")
    
    if args.monitor_gpu:
        # Setup GPU Monitor
        monitor_stop_event = threading.Event()
        peak_mem_dict = {}
        if gpu_list:
            # Pre-populate baseline memory before workers start
            try:
                output = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=index,memory.used,memory.total", "--format=csv,noheader,nounits"],
                    text=True
                )
                for line in output.strip().split('\n'):
                    parts = line.split(', ')
                    if len(parts) == 3:
                        idx = int(parts[0])
                        used = int(parts[1])
                        total = int(parts[2])
                        if idx in gpu_list:
                            peak_mem_dict[idx] = {'peak': used, 'total': total, 'baseline': used}
            except Exception:
                pass
                
            monitor_thread = threading.Thread(target=gpu_monitor_thread, args=(gpu_list, monitor_stop_event, peak_mem_dict))
            monitor_thread.daemon = True
            monitor_thread.start()
        
    # 2. Setup Executors and Queues
    write_queue = queue.Queue()
    
    # ProcessPool for geometry extraction (CPU bound)
    CPU_WORKERS = 200
    process_executor = concurrent.futures.ProcessPoolExecutor(max_workers=max(1, min(CPU_WORKERS, os.cpu_count() - 32)))
    
    # ThreadPool for subprocess launching (GPU bound)
    render_executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_render_workers)
    
    # Thread for writing H5s sequentially (Disk bound)
    writer = threading.Thread(target=h5_writer_thread, args=(write_queue, args.output_dir, tmp_dir, args.formats, args.chunk_size, len(json_files), args.num_views))
    writer.daemon = True
    writer.start()

    # Callback chain logic
    def on_process_done(future):
        # Forward result to writer queue
        try:
            scene_file, result = future.result()
        except Exception as e:
            print(f"[CPU Worker] Unexpected processing callback failure: {e}")
            scene_file, result = ("failed_scene.json", None)
        write_queue.put((scene_file, result))


    # 3. Stage 1: Generate JSONs
    print("[*] Stage 1: Generating JSON templates...")
    template_json = "renderformer/datasets/templates/cbox-3-walls_abspaths.json"
    objaverse_objects = find_objaverse_objects()
    
    if len(objaverse_objects) == 0:
        print("Fatal: No objaverse objects found. Aborting.")
        return
        
    texture_mode = args.texture_mode if args.texture_mode is not None else "per-shading-group"
    json_workers = max(1, min(64, os.cpu_count() or 1))
    print(f"[*] Using {json_workers} concurrent JSON generation workers.")
    print(f"[*] Found {len(json_files) - len(json_loop_indices_to_generate)} existing JSONs, generating {len(json_loop_indices_to_generate)} new ones (skipped {skipped_count} from resume)...")

    target_json_count = len(json_files)
    with tqdm.tqdm(total=target_json_count, desc="Generating scenes", unit="scene", mininterval=0.25) as pbar:
        existing_count = target_json_count - len(json_loop_indices_to_generate)
        if existing_count > 0:
            pbar.update(existing_count)

        if json_loop_indices_to_generate:
            with concurrent.futures.ThreadPoolExecutor(max_workers=json_workers) as json_executor:
                json_futures = [
                    json_executor.submit(
                        generate_json_worker_task,
                        loop_idx,
                        args.start_idx,
                        template_json,
                        objaverse_objects,
                        tmp_dir,
                        args.num_views,
                        args.transform_scenes,
                        texture_mode,
                        args.color_lights,
                    )
                    for loop_idx in json_loop_indices_to_generate
                ]

                for future in concurrent.futures.as_completed(json_futures):
                    loop_idx, json_file = future.result()
                    # json_files is already fully populated with paths, we just needed them to generate
                    pbar.update(1)
        
    print(f"[*] Have {len(json_files)} JSONs ready to go.")
    if args.json_only:
        return
    
    # 4. Stage 2: Submit to Pipeline
    print("[*] Stage 2 & 3: Streaming renders and preprocessing...")
    gpu_counter = 0
    
    try:
        active_futures = set()
        for scene_file in json_files:
            gpu_id = gpu_list[gpu_counter % len(gpu_list)] if gpu_list else None
            gpu_counter += 1
            
            # Submit to rendering threadpool
            rend_future = render_executor.submit(
                render_worker_task,
                scene_file,
                tmp_dir,
                gpu_id,
                args.spp,
                args.render_timeout_seconds,
                args.render_timeout_retries,
            )
            active_futures.add(rend_future)

        # 5. Wait for completion, handling dynamic resubmission for failed scenes
        while active_futures:
            done, active_futures = concurrent.futures.wait(active_futures, return_when=concurrent.futures.FIRST_COMPLETED)
            
            for future in done:
                try:
                    scene_file, success = future.result()
                except Exception as e:
                    print(f"[GPU Worker] Unexpected render callback failure: {e}")
                    scene_file, success = ("failed_scene.json", False)
                    
                if success:
                    # Submit to CPU processing pool
                    proc_future = process_executor.submit(process_worker_task, scene_file, tmp_dir, args.points, args.formats)
                    proc_future.add_done_callback(on_process_done)
                else:
                    if scene_file != "failed_scene.json":
                        print(f"\n[*] Render completely failed for {scene_file}. Force-generating a new scene to retry...", flush=True)
                        import re
                        match = re.search(r'scene_(\d+)\.json', os.path.basename(scene_file))
                        if match:
                            scene_idx = int(match.group(1))
                            try:
                                # Regenerate the JSON with new random objects to avoid the timeout
                                generate_scene(
                                    template_json,
                                    objaverse_objects,
                                    scene_idx,
                                    tmp_dir,
                                    num_views=args.num_views,
                                    transform_scene=args.transform_scenes,
                                    random_diffuse_type=texture_mode,
                                    color_lights=args.color_lights,
                                )
                                # Pick a random GPU from the available pool
                                gpu_id = random.choice(gpu_list) if gpu_list else None
                                rend_future = render_executor.submit(
                                    render_worker_task,
                                    scene_file,
                                    tmp_dir,
                                    gpu_id,
                                    args.spp,
                                    args.render_timeout_seconds,
                                    args.render_timeout_retries,
                                )
                                active_futures.add(rend_future)
                                continue # Successfully requeued
                            except Exception as e:
                                print(f"[*] Failed to force regenerate scene {scene_idx}: {e}", flush=True)
                    
                    # If we reach here, we couldn't regenerate or it's a completely unrecoverable failure
                    write_queue.put((scene_file, None))

        render_executor.shutdown(wait=True)
        process_executor.shutdown(wait=True)
        
        # Wait for the writer thread to drain the queue and flush final chunks
        write_queue.join()
    except KeyboardInterrupt:
        print("\n[*] Caught KeyboardInterrupt! Gracefully aborting...")
        print("[*] Terminating render and process executors (this may take a moment)...")
        # In Python 3.9+, shutdown(wait=False, cancel_futures=True) is supported
        try:
            render_executor.shutdown(wait=False, cancel_futures=True)
            process_executor.shutdown(wait=False, cancel_futures=True)
        except TypeError:
            render_executor.shutdown(wait=False)
            process_executor.shutdown(wait=False)
            
        print("[*] Sending STOP signal to H5 writer thread to preserve INCOMPLETE chunks...")
        write_queue.put(("STOP", None))
        write_queue.join()  # wait for writer to process STOP
    finally:
        writer.join()

    # Stop GPU monitor
    if args.monitor_gpu:
        monitor_stop_event.set()
        if gpu_list:
            monitor_thread.join()

    pipeline_end_time = time.time()
    elapsed_time = pipeline_end_time - pipeline_start_time

    print(f"[*] Pipeline completed successfully in {elapsed_time:.2f} seconds!")
    
    if args.monitor_gpu and gpu_list and peak_mem_dict:
        print("\n--- GPU Memory Profile & Worker Recommendation ---")
        for gpu_idx in gpu_list:
            if gpu_idx in peak_mem_dict:
                stats = peak_mem_dict[gpu_idx]
                baseline = stats['baseline']
                peak = stats['peak']
                total = stats['total']
                used_by_workers = peak - baseline
                
                # Handle cases where memory usage wasn't significant
                if used_by_workers <= 0:
                    used_by_workers = 100 # arbitrary small number to prevent division by zero or nonsensical values
                
                # Estimate per worker
                # Important: we only ever run as many concurrent workers per GPU as there are scenes assigned to it
                import math
                active_workers_on_this_gpu = min(args.workers_per_gpu, math.ceil(args.num_scenes / num_gpus))
                mem_per_worker = used_by_workers / max(1, active_workers_on_this_gpu)
                
                # Max theoretic workers based on free memory at start (leaving 1000MB safety margin)
                free_mem_available = total - baseline
                max_workers = int((free_mem_available - 1000) / max(1, mem_per_worker))
                max_workers = max(0, max_workers) # Ensure non-negative
                
                print(f"GPU {gpu_idx}: Baseline: {baseline}MB | Peak: {peak}MB | Total: {total}MB")
                print(f"  -> Max configured workers: {args.workers_per_gpu} (Actual active: {active_workers_on_this_gpu})")
                print(f"  -> Est. Memory per worker: ~{mem_per_worker:.0f}MB")
                print(f"  -> You can afford to run roughly {max_workers} workers safely on this GPU.")

if __name__ == "__main__":
    main()
